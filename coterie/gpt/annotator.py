"""Annotator — OpenAI variant.

Same buffer + flush semantics as annotator.py. Uses gpt-5.5 with an inline
search_memories function tool and a json_schema text.format describing
the annotations array.
"""

import asyncio
import collections
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from coterie import config
from coterie.memory import BotMemory

log = logging.getLogger("dc-agent.annotator")

MODEL = "gpt-5.5"
WINDOW_SIZE = 100
BATCH_THRESHOLD = 10
TIME_THRESHOLD_SEC = 300
MAX_TURNS = 6   # annotator should never need many tool hops

SYSTEM_PROMPT = config.render("""You annotate {platform} chat messages to improve later \
semantic retrieval. You'll see a window of recent channel messages. Some are \
marked [NEW] — your job is to produce one short English annotation for each \
non-trivial [NEW] message. Messages marked [BOT] are the bot's own prior \
replies — use them as context but DO NOT annotate them.

Each annotation should:
1. Resolve pronouns and implicit references.
2. Name the people / products / topics mentioned.
3. Capture intent if non-obvious.

For trivial messages (greetings, reactions, single emoji, one-word replies) \
omit them from your output entirely.

If a [NEW] message references a topic clearly older than the visible window, \
call `search_memories` to look it up before annotating.

You may produce a single annotation spanning multiple consecutive [NEW] \
messages if they form one coherent thought — set message_ids to all involved.

Each annotation should be one English sentence, under 30 words.

Output JSON matching the schema. Annotations only — no preamble, no explanation.""")

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_ids": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"},
                },
                "required": ["message_ids", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["annotations"],
    "additionalProperties": False,
}

SEARCH_TOOL = {
    "type": "function",
    "name": "search_memories",
    "description": (
        "Search older channel history (outside the visible window) for "
        "context. Scoped to original messages, not annotations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "since": {"type": "string"},
            "until": {"type": "string"},
        },
        "required": ["query"],
    },
}


class Annotator:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncOpenAI()
        self._buffers: dict[str, collections.deque[dict[str, Any]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_flush_mono: dict[str, float] = {}

    def buffer_push(
        self,
        *,
        channel_id: str,
        message_id: str,
        author: str,
        content: str,
        timestamp_iso: str,
        is_bot_reply: bool = False,
    ) -> None:
        buf = self._buffer_for(channel_id)
        buf.append({
            "message_id": message_id,
            "author": author,
            "content": content,
            "timestamp": timestamp_iso,
            "annotated": is_bot_reply,
            "is_bot_reply": is_bot_reply,
        })

    def recent_n(self, channel_id: str, n: int = 10) -> list[dict[str, Any]]:
        buf = self._buffers.get(channel_id)
        if not buf:
            return []
        return list(buf)[-n:]

    def _buffer_for(self, channel_id: str) -> collections.deque[dict[str, Any]]:
        if channel_id not in self._buffers:
            self._buffers[channel_id] = collections.deque(maxlen=WINDOW_SIZE)
            self._locks[channel_id] = asyncio.Lock()
            self._last_flush_mono[channel_id] = time.monotonic()
        return self._buffers[channel_id]

    async def maybe_flush_all(self) -> None:
        for channel_id in list(self._buffers.keys()):
            buf = self._buffers[channel_id]
            unann = sum(1 for m in buf if not m["annotated"])
            if unann == 0:
                continue
            elapsed = time.monotonic() - self._last_flush_mono[channel_id]
            if unann >= BATCH_THRESHOLD or elapsed >= TIME_THRESHOLD_SEC:
                try:
                    await self._flush(channel_id)
                except Exception:
                    log.exception("flush failed for channel %s", channel_id)

    async def _flush(self, channel_id: str) -> None:
        async with self._locks[channel_id]:
            buf = self._buffers[channel_id]
            new_msgs = [m for m in buf if not m["annotated"]]
            if not new_msgs:
                return

            new_ids = {m["message_id"] for m in new_msgs}
            window_text = _format_window(buf)
            memory = self._memory

            async def call_search_memories(args: dict[str, Any]) -> str:
                results = await memory.search(
                    query=args.get("query", ""),
                    channel_id=channel_id,
                    since=args.get("since"),
                    until=args.get("until"),
                    record_type="message",
                    limit=8,
                )
                return _format_search_results(results)

            tool_handlers = {"search_memories": call_search_memories}
            input_items: list[dict[str, Any]] = [
                {"role": "user", "content": [
                    {"type": "input_text", "text": window_text}
                ]}
            ]

            log.info(
                "annotator: flushing channel=%s window=%d new=%d",
                channel_id, len(buf), len(new_msgs),
            )

            try:
                last_text = await self._run_loop(
                    input_items=input_items,
                    tools=[SEARCH_TOOL],
                    tool_handlers=tool_handlers,
                    text_format={
                        "format": {
                            "type": "json_schema",
                            "name": "Annotations",
                            "schema": OUTPUT_SCHEMA,
                            "strict": True,
                        },
                    },
                    log_prefix=f"annotator[{channel_id[:8]}]",
                )
            except Exception:
                log.exception("annotator LLM call failed for channel %s", channel_id)
                return

            if not last_text:
                return
            try:
                parsed = json.loads(last_text)
                annotations = parsed.get("annotations", [])
            except Exception:
                log.exception("annotator JSON parse failed: %r", last_text[:200])
                return

            now_iso = datetime.now(timezone.utc).isoformat()
            stored = 0
            for ann in annotations:
                text = (ann.get("text") or "").strip()
                ref_ids = ann.get("message_ids") or []
                if not text or not ref_ids:
                    continue
                ts = next(
                    (m["timestamp"] for m in buf if m["message_id"] in ref_ids),
                    now_iso,
                )
                try:
                    await self._memory.add_annotation(
                        text=text,
                        channel_id=channel_id,
                        references_message_ids=ref_ids,
                        timestamp=ts,
                    )
                    stored += 1
                except Exception:
                    log.exception("failed to store annotation")

            for m in buf:
                if m["message_id"] in new_ids:
                    m["annotated"] = True

            self._last_flush_mono[channel_id] = time.monotonic()
            log.info(
                "annotator: channel=%s stored=%d trivial_or_skipped=%d",
                channel_id, stored, len(new_msgs) - stored,
            )

    async def _run_loop(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        text_format: dict[str, Any],
        log_prefix: str,
    ) -> str:
        """Generic Responses-API tool-call loop returning the final text."""
        last_text = ""
        for turn in range(1, MAX_TURNS + 1):
            response = await self._client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=input_items,
                tools=tools,
                text=text_format,
            )
            _log_response_items(response, prefix=f"{log_prefix} t{turn}")
            pending = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]
            if not pending:
                last_text = (getattr(response, "output_text", None) or "").strip()
                break
            for item in response.output:
                input_items.append(_item_to_dict(item))
            for call in pending:
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
                        out = f"(tool error: {e})"
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": out,
                })
        return last_text


def _format_window(buf: collections.deque[dict[str, Any]]) -> str:
    lines = []
    for m in buf:
        if m.get("is_bot_reply"):
            tag = "[BOT]"
        elif not m["annotated"]:
            tag = "[NEW]"
        else:
            tag = "     "
        lines.append(
            f"{tag} [{m['message_id']}] [{m['timestamp']}] {m['author']}: {m['content']}"
        )
    return "\n".join(lines)


def _format_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "(no older context found)"
    lines = []
    for r in results:
        meta = r.get("metadata") or {}
        ts = meta.get("timestamp", "?")
        author = meta.get("author", "")
        text = r.get("memory") or r.get("text") or ""
        prefix = f"{author}: "
        if author and text.startswith(prefix):
            text = text[len(prefix):]
        who = f"{author}: " if author else ""
        lines.append(f"- [{ts}] {who}{text}")
    return "\n".join(lines)


def _item_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    return dict(item)


def _log_response_items(response: Any, *, prefix: str) -> None:
    for item in getattr(response, "output", []) or []:
        itype = getattr(item, "type", "unknown")
        if itype == "function_call":
            log.info(
                "%s function_call[%s] args=%s",
                prefix,
                getattr(item, "name", "?"),
                _short(getattr(item, "arguments", "")),
            )
        elif itype == "message":
            text = _msg_text(item)
            if text:
                log.info("%s text: %s", prefix, _short(text))


def _msg_text(item: Any) -> str:
    parts = []
    for block in getattr(item, "content", []) or []:
        if getattr(block, "type", None) in ("output_text", "text"):
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _short(s: Any, n: int = 200) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "..."
