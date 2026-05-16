"""OpenAI Responses API agent — drop-in equivalent of agent.py.

Tool surface mirrored from the Anthropic version:
- `search_memories` (function tool, client-side, Mem0)
- `web_search` (built-in server tool, Anthropic-hosted equivalent)
- `web_fetch` (function tool we built; OpenAI has no native equivalent)
- `code_interpreter` (built-in server tool, OpenAI's analog of code_execution)

reply() returns (text, generated_files). generated_files comes from code_
interpreter; we walk its output_items, find references to container files,
and download them via /v1/containers/{cid}/files/{fid}/content.

The loop is manual — Responses API doesn't ship a tool_runner abstraction.
On each round-trip we look for `function_call` items in response.output,
execute them locally, and append `function_call_output` items before the
next responses.create call.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from coterie import config
from coterie.memory import BotMemory
from coterie.web_fetch import fetch_url

log = logging.getLogger("dc-agent.agent")

# Default model. Used for both @-mentions and proactive replies unless the
# user explicitly opts into Pro via a trigger phrase in their message.
MODEL_DEFAULT = "gpt-5.5"
# Pro model. Slower + more expensive but deeper reasoning. Selected only when
# the user asks for it (see PRO_TRIGGER_RE).
MODEL_PRO = "gpt-5.5-pro"

# Phrases that switch this @-mention reply to gpt-5.5-pro.
# Intentionally explicit — bare "pro" matches too many false positives
# (sports, programming pros, etc.). We require a verb / qualifier nearby.
PRO_TRIGGER_RE = re.compile(
    r"\buse\s+pro\b|\bswitch\s+to\s+pro\b|"               # use pro / switch to pro
    r"\bgpt-?5\.?5-?pro\b|\b5\.?5\s+pro\b|"               # gpt-5.5-pro / 5.5 pro
    r"\bpro\s+(?:version|model|mode)\b|"                  # pro mode / pro model
    r"\bthink\s+hard(?:er)?\b|\bdeep\s+think(?:ing)?\b|"  # think hard / deep thinking
    r"\bbest\s+model\b|\bstrongest\s+model\b",            # best model / strongest model
    re.IGNORECASE,
)

REASONING_EFFORT = "xhigh"
MAX_TURNS = 12          # safety bound on agent loops


def _pick_model(query: str) -> str:
    """Return MODEL_PRO iff the query explicitly asks for pro; else default."""
    if PRO_TRIGGER_RE.search(query or ""):
        return MODEL_PRO
    return MODEL_DEFAULT


def _system_prompt(model_id: str) -> str:
    """Substitute the model identity into the system prompt template, so the
    bot answers "what model are you" honestly for whichever model is serving
    this particular reply.

    Uses .replace() instead of str.format() so that ANY curly braces appearing
    elsewhere in the prompt (LaTeX examples, JSON examples, etc.) pass through
    unchanged. str.format() interprets every `{...}` as a placeholder, which
    once blew up on `\\frac{}{}` in the Discord-formatting section."""
    display = "GPT-5.5 Pro" if model_id == MODEL_PRO else "GPT-5.5"
    return (
        SYSTEM_PROMPT_TEMPLATE
        .replace("{model_display}", display)
        .replace("{model_id}", model_id)
    )

SYSTEM_PROMPT_TEMPLATE = config.render("""You are a helpful {platform} chat assistant for a community.

You are powered by {model_display} (model ID: `{model_id}`). The default \
backing model for this bot is GPT-5.5; users can request GPT-5.5 Pro by \
including phrases like "use pro", "switch to pro", "think hard", or \
"best model" in their message. If anyone asks what model you're running \
on, answer with the model ID above (the one actually serving this \
reply). Annotator and profile pipelines use GPT-5.5; Mem0's internal \
LLM is GPT-4o-mini.

You have four tools — use the right one for the job. Reason about which is \
needed before calling.

`search_memories(query, author?, since?, until?)`
- Use for anything that depends on what was said in THIS {platform} channel.
- Pass `author` (display name) if the question is about a specific person.
- Pass `since`/`until` (ISO 8601) for time-bounded questions. Resolve \
relative time references ("today", "yesterday", "last week", "March") \
against the "Current time" header in the user message.

Memory records come in three flavors — the tag at the start of each line \
tells you which:
- `[message]` — a real human message in the channel. Ground truth.
- `[annotation]` — an LLM-generated English summary of one or more human \
messages, written to make retrieval easier. Tied back to the original \
messages — also reliable, since it's derived from them.
- `[agent_reply]` — YOUR OWN previous reply in this channel. Treat as your \
prior speculation, not user-confirmed fact. Use it only for conversational \
continuity. NEVER cite your own past reply as evidence for a factual claim — \
find a `[message]` or `[annotation]` to back it up, or be honest that you \
don't have one.

Memories tagged [PINNED] were marked important by channel admins. Treat \
them as authoritative when they're relevant; do NOT surface them when they're \
not. When pinned and non-pinned content conflict, prefer pinned unless the \
non-pinned is clearly more recent and specific.

`web_search` — for discovering current/public information not in the \
channel. Returns a result list with snippets. Do NOT use to recall what \
someone said in this channel.

`web_fetch(url)` — given a SPECIFIC URL, fetch and read its actual content. \
**Use this whenever a user shares a URL** (github repo, arxiv paper, blog \
post). Don't just web_search around the URL — fetch it directly so you see \
the actual README / abstract / post body. web_search snippets are not enough \
for substantive analysis. NOTE: this is a function tool we built around \
httpx + trafilatura. It works for static pages (arxiv, github, blogs); for \
JS-rendered SPAs (X.com tweets) it may return raw HTML or fail — if so, \
say so plainly rather than fabricating content.

`code_interpreter` — for computation, data analysis, plotting (matplotlib), \
reading PDFs programmatically, generating files to share.

PDF attachments: the user may attach a PDF. If you see a file input in the \
user content, read it directly. For deep extraction (tables, chart data, \
page-specific quotes) use code_interpreter with the PDF.

Reply style:
- Be concise. Chat is short-form — most replies are 1-3 sentences. For \
genuinely complex questions: aim ≤ 10 short lines, prose-flow not \
hierarchy. If you cannot fit it that small, ask the user whether they want \
a deep dive first.
- **Just answer the question.** NEVER narrate your retrieval process — do \
not write "I searched and found nothing", "no one in the channel discussed \
this", or any similar meta-talk.
- Attribute only when you actually have a grounding memory to cite. If you \
have none, just answer from general knowledge / web_search without \
mentioning that fact.
- The one exception: if the question is specifically asking about channel \
history AND search returned nothing relevant, a single short acknowledgement \
is fine ("no one's brought it up"). Then either web_search or answer from \
general knowledge.
- Do not fabricate. If you genuinely don't know, say so.

{formatting_rules}

═══ Anti-hallucination on thin web results ═══

If `web_search` returns snippets clearly unrelated to what was asked, DO \
NOT fabricate a plausible-sounding analysis. Instead:
1. Try `web_fetch` on the exact URL the user gave.
2. If web_fetch also fails, say so plainly: "couldn't fetch the github \
page, can only guess from the URL...".
3. NEVER write "the author's prior X project" / similar speculation \
unless you have concrete evidence from a successfully fetched source.

- Match the casual tone of the channel.
- Don't quote unrelated retrievals just because they came back.

Constraints:
- You cannot pin/unpin/delete messages. You read only.
- You cannot reach into other channels.
- If a query is too ambiguous, ask one clarifying question rather than guessing.

═══ Honesty about your own past replies ═══

If a user calls you out for an inconsistency, DO NOT GASLIGHT. Don't claim \
"I've been consistent the whole time" when [agent_reply] shows otherwise. \
Acknowledge:
  ✓ "You're right, my earlier message was wrong — the correct answer is..."
  ✗ "You're remembering wrong"

Trust [agent_reply] records over your memory of what you said. Own the \
mistake, give the right answer, move on.

═══ PROACTIVE MODE ═══

If the user message header includes `[PROACTIVE — jumping in without being \
@-ed]`, you are joining a live conversation on your own initiative:

- Speak ONLY 1-2 sentences. No headers, no bullets, no lectures.
- Add CONCRETE NEW VALUE: a specific datum, a paper/URL, a contrarian \
point, a prior channel discussion. Vague encouragement / summary / \
"interesting question!" is forbidden.
- Do NOT summarize what was just said.
- If you have nothing concrete to add — output exactly `<skip>` as your \
entire response. The bot will then say nothing.
- Don't @ anyone in your text; the wrapper handles mentioning the speaker.
- Don't preface with "Quick thought:" / "Let me chime in:" / similar filler.
- Stay focused on the TRIGGER MESSAGE.
""")


SEARCH_MEMORIES_TOOL = {
    "type": "function",
    "name": "search_memories",
    "description": (
        "Search this channel's message history with optional filters."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic description of what you're looking for.",
            },
            "author": {
                "type": "string",
                "description": "Filter by speaker display name (exact match).",
            },
            "since": {
                "type": "string",
                "description": "ISO 8601 timestamp; messages at/after this time.",
            },
            "until": {
                "type": "string",
                "description": "ISO 8601 timestamp; messages at/before this time.",
            },
        },
        "required": ["query"],
    },
}

WEB_FETCH_TOOL = {
    "type": "function",
    "name": "web_fetch",
    "description": (
        "Fetch a specific URL and return its extracted readable text. Use "
        "this when the user shares a URL and you need to discuss its actual "
        "content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (http or https).",
            }
        },
        "required": ["url"],
    },
}


class Agent:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncOpenAI()

    async def reply(
        self,
        *,
        query: str,
        channel_id: str,
        asker: str,
        asker_profile: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        mode: str = "mention",
        recent_context: list[dict[str, Any]] | None = None,
        prior_relevant: list[dict[str, Any]] | None = None,
        trigger_reason: str | None = None,
    ) -> tuple[str, list[tuple[str, bytes]]]:
        memory = self._memory

        async def call_search_memories(args: dict[str, Any]) -> str:
            results = await memory.search(
                query=args.get("query", ""),
                channel_id=channel_id,
                author=args.get("author"),
                since=args.get("since"),
                until=args.get("until"),
            )
            results = [
                r for r in results
                if (r.get("metadata") or {}).get("record_type") != "profile"
            ]
            return _format_memories(results)

        async def call_web_fetch(args: dict[str, Any]) -> str:
            return await fetch_url(args.get("url", ""))

        tool_handlers = {
            "search_memories": call_search_memories,
            "web_fetch": call_web_fetch,
        }
        tools = [
            SEARCH_MEMORIES_TOOL,
            WEB_FETCH_TOOL,
            {"type": "web_search"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]

        now_iso = datetime.now(timezone.utc).isoformat()
        if mode == "proactive":
            user_text = _build_proactive_user_text(
                now_iso=now_iso,
                asker=asker,
                asker_profile=asker_profile,
                trigger_msg=query,
                recent_context=recent_context or [],
                prior_relevant=prior_relevant or [],
                trigger_reason=trigger_reason,
            )
        else:
            header = f"[Current time: {now_iso}] [Asker: {asker}]"
            if asker_profile:
                header += f"\n[Asker profile]\n{asker_profile}"
            user_text = f"{header}\n\n{query}"

        # Build the initial input list. Responses API uses `input` (list of
        # items) rather than `messages`. PDF attachments are attached as
        # input_file items alongside the text within the same user item.
        content_items: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_text}
        ]
        if attachments:
            content_items.extend(attachments)
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": content_items}
        ]

        # Pick model per-call. Default gpt-5.5; pro only when the user's @
        # message contains an explicit pro-trigger phrase. Proactive mode is
        # forced to the default since the user didn't request anything there.
        model = _pick_model(query) if mode == "mention" else MODEL_DEFAULT
        instructions = _system_prompt(model)
        log.info(
            "agent: query from=%s model=%s query=%s",
            asker, model, _shorten(query, 120),
        )

        last_response = None
        generated_files: list[tuple[str, bytes]] = []
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = await self._client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=input_items,
                    tools=tools,
                    reasoning={"effort": REASONING_EFFORT},
                )
            except Exception:
                log.exception("agent: responses.create failed turn=%d", turn)
                break
            last_response = response
            log_response(response, prefix=f"agent[t{turn}]")

            generated_files.extend(
                await self._extract_files(response)
            )

            # Look for function_call items the model wants to execute.
            pending_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not pending_calls:
                # No more tool calls — the model has produced its final answer.
                break

            # Append the assistant's own output items (so the next call sees
            # the function_call entries that we're about to satisfy), then
            # the corresponding function_call_output items.
            for item in response.output:
                input_items.append(_item_to_dict(item))
            for call in pending_calls:
                handler = tool_handlers.get(call.name)
                if handler is None:
                    out = f"(unknown tool: {call.name})"
                else:
                    try:
                        args = json.loads(call.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    try:
                        out = await handler(args)
                    except Exception as e:
                        log.exception("tool %s crashed", call.name)
                        out = f"(tool error: {type(e).__name__}: {e})"
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": out,
                })

        text = _extract_text(last_response) if last_response else ""
        log.info(
            "agent: done turns=%s files=%d",
            getattr(last_response, "id", "?") if last_response else "no-response",
            len(generated_files),
        )
        return text, generated_files

    async def _extract_files(self, response: Any) -> list[tuple[str, bytes]]:
        """Walk response.output for code_interpreter file references.

        OpenAI's container files come back inside `code_interpreter_call`
        items, with output blocks containing `container_file_citation` or
        similar refs. We download via the containers API.
        """
        out: list[tuple[str, bytes]] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "code_interpreter_call":
                continue
            container_id = getattr(item, "container_id", None) or getattr(
                item, "container", None
            )
            if isinstance(container_id, dict):
                container_id = container_id.get("id")
            outputs = getattr(item, "outputs", None) or []
            for o in outputs:
                files = getattr(o, "files", None) or []
                for f in files:
                    file_id = getattr(f, "file_id", None) or getattr(f, "id", None)
                    name = getattr(f, "filename", None) or "file"
                    if not file_id or not container_id:
                        continue
                    try:
                        data = await self._download_container_file(
                            container_id, file_id
                        )
                        out.append((name, data))
                    except Exception:
                        log.exception("file download failed cid=%s fid=%s",
                                      container_id, file_id)
        return out

    async def _download_container_file(self, container_id: str, file_id: str) -> bytes:
        """Use the raw HTTP client beneath AsyncOpenAI to GET the file content."""
        path = f"/v1/containers/{container_id}/files/{file_id}/content"
        resp = await self._client.with_raw_response.get(path)
        return resp.content


def _item_to_dict(item: Any) -> dict[str, Any]:
    """Convert an SDK response item to a dict suitable for re-posting as input.

    The Responses API accepts the same JSON shape it returns. The SDK's items
    are pydantic models; .model_dump() gives back the dict form.
    """
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return dict(item)  # already a dict


def _extract_text(response: Any) -> str:
    """Concatenate the text content from message items in response.output."""
    if response is None:
        return ""
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct:
        return direct.strip()
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "output_text" or btype == "text":
                text = getattr(block, "text", None) or ""
                if text:
                    parts.append(text)
    return "".join(parts).strip()


def _shorten(s: Any, n: int = 150) -> str:
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.strip().replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "..."


def log_response(response: Any, *, prefix: str) -> None:
    """Log each item type in the Responses API output array."""
    for item in getattr(response, "output", []) or []:
        itype = getattr(item, "type", "unknown")
        if itype == "message":
            text = _extract_text_from_message(item)
            if text:
                log.info("%s text: %s", prefix, _shorten(text, 200))
        elif itype == "reasoning":
            summary = getattr(item, "summary", None) or []
            summary_text = " | ".join(
                getattr(s, "text", "") or "" for s in summary
            )
            if summary_text:
                log.info("%s reasoning: %s", prefix, _shorten(summary_text, 200))
        elif itype == "function_call":
            name = getattr(item, "name", "?")
            args = getattr(item, "arguments", "")
            log.info("%s function_call[%s] args=%s",
                     prefix, name, _shorten(args, 250))
        elif itype == "web_search_call":
            query = ""
            action = getattr(item, "action", None)
            if action:
                query = getattr(action, "query", "") or ""
            log.info("%s web_search_call query=%s", prefix, _shorten(query, 120))
        elif itype == "code_interpreter_call":
            code = getattr(item, "code", "") or ""
            log.info("%s code_interpreter_call code=%s",
                     prefix, _shorten(code, 200))
        else:
            log.info("%s unknown_item type=%s", prefix, itype)


def _extract_text_from_message(item: Any) -> str:
    parts: list[str] = []
    for block in getattr(item, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype in ("output_text", "text"):
            text = getattr(block, "text", None) or ""
            if text:
                parts.append(text)
    return "".join(parts)


def _build_proactive_user_text(
    *,
    now_iso: str,
    asker: str,
    asker_profile: str | None,
    trigger_msg: str,
    recent_context: list[dict[str, Any]],
    prior_relevant: list[dict[str, Any]],
    trigger_reason: str | None,
) -> str:
    parts: list[str] = [
        "[PROACTIVE — jumping in without being @-ed]",
        f"[Current time: {now_iso}] [Speaker: {asker}]",
    ]
    if asker_profile:
        parts.append(f"[Speaker profile]\n{asker_profile}")
    parts.append("")
    parts.append("═════════ TRIGGER MESSAGE ═════════")
    parts.append(f"{asker}: {trigger_msg}")
    parts.append("═══════════════════════════════════")
    if recent_context:
        parts.append("")
        parts.append(
            f"[Recent channel context — last {len(recent_context)} msgs, "
            "for thread continuity only]"
        )
        for m in recent_context:
            author = m.get("author", "?")
            ts = m.get("timestamp", "?")
            content = m.get("content", "") or ""
            tag = "[BOT]" if m.get("is_bot_reply") else "[message]"
            parts.append(f"{tag} [{ts}] {author}: {content}")
    if prior_relevant:
        parts.append("")
        parts.append(
            f"[Relevant prior discussions — top-{len(prior_relevant)} "
            "semantic match from channel memory]"
        )
        for r in prior_relevant:
            meta = r.get("metadata") or {}
            rt = meta.get("record_type", "message")
            ts = meta.get("timestamp", "?")
            author = meta.get("author", "?")
            text = _strip_legacy_author_prefix(
                r.get("memory") or r.get("text") or "", author
            )
            parts.append(f"[{rt}] [{ts}] {author}: {text}")
    parts.append("")
    parts.append("[Instructions]")
    parts.append("- 1-2 sentence reply.")
    parts.append("- Add concrete value. Don't summarize the recent context.")
    parts.append("- If you have nothing concrete to add, output exactly: <skip>")
    if trigger_reason:
        parts.append(f"- Classifier triggered because: {trigger_reason}")
    return "\n".join(parts)


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(no relevant memories found)"
    lines = []
    for m in memories:
        meta = m.get("metadata") or {}
        rt = meta.get("record_type", "message")
        pin = "[PINNED] " if meta.get("is_pinned") else ""
        ts = meta.get("timestamp", "")
        author = meta.get("author", "")
        text = _strip_legacy_author_prefix(m.get("memory") or m.get("text") or "", author)
        who = f"{author}: " if author else ""
        lines.append(f"- {pin}[{rt}] [{ts}] {who}{text}")
    return "\n".join(lines)


def _strip_legacy_author_prefix(text: str, author: str) -> str:
    """Same as agent.py's helper. Duplicated here so the OpenAI backend
    doesn't import from the Anthropic backend (keeps the modules cleanly
    swappable). Remove after the migration window closes."""
    prefix = f"{author}: "
    if author and text.startswith(prefix):
        return text[len(prefix):]
    return text
