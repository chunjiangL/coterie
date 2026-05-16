"""Custom web_fetch for OpenAI backend.

Anthropic has a built-in `web_fetch_20260309` server tool — OpenAI doesn't.
Responses API only ships `web_search`, `code_interpreter`, `file_search`, etc.
So the OpenAI Agent's tool list needs a function tool that does the same job:
pull a URL and return readable text the model can reason over.

Implementation: httpx for fetch (browser-like UA, follow redirects, 30s timeout)
+ trafilatura for main-content extraction. trafilatura handles arxiv abstracts,
github READMEs, blog posts well. It's sync — we wrap it in to_thread.

For URLs that need JS rendering (X.com, modern SPA), this falls back to the
raw HTML body. We slice to ~15k chars so a single fetch doesn't dominate the
context window.
"""

import asyncio
import logging

import httpx

log = logging.getLogger("dc-agent.web_fetch")

MAX_CHARS = 15000
TIMEOUT_SEC = 30.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
}


async def fetch_url(url: str) -> str:
    """Fetch a URL and return readable extracted text.

    Always returns a string — never raises. Failure modes (timeout, 4xx/5xx,
    extraction failure) all produce a short diagnostic the model can act on.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=TIMEOUT_SEC, headers=HEADERS
        ) as client:
            resp = await client.get(url)
    except httpx.HTTPError as e:
        log.warning("web_fetch failed url=%s err=%s", url, e)
        return f"(fetch failed: {type(e).__name__}: {e})"

    if resp.status_code >= 400:
        log.warning("web_fetch http=%s url=%s", resp.status_code, url)
        return f"(HTTP {resp.status_code} for {url})"

    html = resp.text
    try:
        extracted = await asyncio.to_thread(_extract, html)
    except Exception:
        log.exception("trafilatura crashed for %s", url)
        extracted = None

    body = extracted or html
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + f"\n\n... (truncated; total {len(body)} chars)"
    return f"[fetched {url} status={resp.status_code}]\n\n{body}"


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
