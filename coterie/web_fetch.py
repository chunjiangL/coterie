"""Custom web_fetch for OpenAI backend.

Anthropic has a built-in `web_fetch_20260309` server tool. OpenAI doesn't.
Responses API only ships `web_search`, `code_interpreter`, `file_search`,
etc. So the OpenAI Agent's tool list needs a function tool that does the
same job: pull a URL and return readable text the model can reason over.

Implementation: httpx for fetch (browser-like UA, follow redirects, 30s
timeout) + trafilatura for main-content extraction. trafilatura handles
arxiv abstracts, github READMEs, blog posts well. It's sync, we wrap it
in to_thread.

═══ SSRF defense ═══

The URL comes from user-controlled chat text, so this is a classic SSRF
surface. Before fetching we:

  1. Allowlist scheme to http / https only. Rejects file://, ftp://,
     gopher://, data:, javascript:.
  2. Resolve the hostname to an IP and reject if it falls in:
     - loopback        (127/8, ::1)
     - link-local      (169.254/16 → AWS/GCP/Azure metadata!)
     - private RFC1918 (10/8, 172.16/12, 192.168/16)
     - CGNAT           (100.64/10)
     - IPv6 ULA / link-local
     - any IP marked .is_private / .is_reserved by ipaddress stdlib
  3. Cap redirect count to 5 and re-validate every hop's destination
     (httpx doesn't validate redirect targets against our allowlist).

A user typing `@bot please summarize http://169.254.169.254/...` should
get back `(blocked: private/loopback IP)`, not a dump of cloud creds.
"""

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx

log = logging.getLogger("dc-agent.web_fetch")

MAX_CHARS = 15000
TIMEOUT_SEC = 30.0
MAX_REDIRECTS = 5
ALLOWED_SCHEMES = {"http", "https"}
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
}


def _block_reason(url: str) -> str | None:
    """Return a human-readable block reason, or None if the URL is safe.

    Validation order: scheme → host present → resolve to IP → IP class.
    All checks run before any network IO."""
    try:
        parsed = urlparse(url)
    except Exception:
        return "malformed URL"
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return f"scheme {parsed.scheme!r} not allowed (http/https only)"
    host = parsed.hostname
    if not host:
        return "no host in URL"
    # If the host is already a literal IP, validate it directly.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # Hostname — resolve through DNS.
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return f"DNS resolution failed for {host}"
        ips = []
        for family, _, _, _, sockaddr in infos:
            try:
                ips.append(ipaddress.ip_address(sockaddr[0]))
            except ValueError:
                continue
        if not ips:
            return f"no IPs resolved for {host}"
        # If ANY resolved IP is private/reserved, refuse.
        # (Resolves multi-A DNS rebinding to private addrs.)
        for ip in ips:
            reason = _ip_block_reason(ip)
            if reason:
                return f"{host} resolves to {ip} ({reason})"
        return None
    return _ip_block_reason(ip)


def _ip_block_reason(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        # 169.254.0.0/16 — includes cloud metadata services.
        return "link-local / cloud-metadata range"
    if ip.is_private:
        # 10/8, 172.16/12, 192.168/16, ULA fc00::/7.
        return "private / RFC1918"
    if ip.is_reserved:
        return "reserved"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified (0.0.0.0 / ::)"
    # CGNAT 100.64.0.0/10 — not flagged by ipaddress.is_private. Manual.
    if isinstance(ip, ipaddress.IPv4Address):
        if ip in ipaddress.IPv4Network("100.64.0.0/10"):
            return "CGNAT"
    return None


async def fetch_url(url: str) -> str:
    """Fetch a URL and return readable extracted text.

    Always returns a string, never raises. Failure modes (SSRF block,
    timeout, 4xx/5xx, extraction failure) all produce a short diagnostic
    the model can act on.
    """
    # SSRF gate BEFORE any network IO.
    block = _block_reason(url)
    if block:
        log.warning("web_fetch SSRF blocked url=%s reason=%s", url, block)
        return f"(blocked: {block})"

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,   # we re-validate every hop ourselves
            timeout=TIMEOUT_SEC,
            headers=HEADERS,
        ) as client:
            current = url
            for hop in range(MAX_REDIRECTS + 1):
                resp = await client.get(current)
                if 300 <= resp.status_code < 400 and "location" in resp.headers:
                    if hop == MAX_REDIRECTS:
                        return f"(blocked: too many redirects, >{MAX_REDIRECTS})"
                    # Resolve relative redirects against the current URL.
                    next_url = str(httpx.URL(current).join(resp.headers["location"]))
                    block = _block_reason(next_url)
                    if block:
                        log.warning(
                            "web_fetch SSRF blocked redirect url=%s reason=%s",
                            next_url, block,
                        )
                        return f"(blocked redirect: {block})"
                    current = next_url
                    continue
                break
    except httpx.HTTPError as e:
        log.warning("web_fetch failed url=%s err=%s", url, e)
        return f"(fetch failed: {type(e).__name__}: {e})"

    if resp.status_code >= 400:
        log.warning("web_fetch http=%s url=%s", resp.status_code, current)
        return f"(HTTP {resp.status_code} for {current})"

    html = resp.text
    try:
        extracted = await asyncio.to_thread(_extract, html)
    except Exception:
        log.exception("trafilatura crashed for %s", current)
        extracted = None

    body = extracted or html
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + f"\n\n... (truncated; total {len(body)} chars)"
    return f"[fetched {current} status={resp.status_code}]\n\n{body}"


def _extract(html: str) -> str | None:
    """Run trafilatura in worker thread. Lazy-import so the module doesn't
    crash at import time if trafilatura isn't installed yet."""
    try:
        import trafilatura
    except ImportError:
        log.warning("trafilatura not installed; returning raw HTML")
        return None
    return trafilatura.extract(
        html,
        include_links=True,
        include_tables=True,
        favor_recall=True,
    )
