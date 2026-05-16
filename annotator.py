"""Sliding-window batch annotator.

For each channel we keep an in-memory deque of the last 100 messages. On each
periodic tick (driven from bot.py), if any channel has ≥10 un-annotated
messages or hasn't been flushed in ≥5 minutes, we ask Sonnet 4.6 to read the
window and produce English context annotations for the un-annotated subset.

Annotations are stored as independent Mem0 records (record_type=annotation),
linked back via references_message_ids in metadata. They get their own
embedding, so retrieval can match either the raw message or its annotation
("那个 deploy" can match "fly.io deployment" through the annotation).

The annotator agent has its own `search_memories` tool — Sonnet can pull
older context out of Mem0 when the 100-msg window doesn't reach far enough
back to resolve a reference. The tool is scoped to record_type=message to
avoid feeding annotations back into the annotation pipeline.
"""

import asyncio
import collections
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic, beta_async_tool

from agent import _strip_legacy_author_prefix, log_message_blocks
from memory import BotMemory

log = logging.getLogger("dc-agent.annotator")

MODEL = "claude-sonnet-4-6"
WINDOW_SIZE = 100
BATCH_THRESHOLD = 10
TIME_THRESHOLD_SEC = 300

SYSTEM_PROMPT = """You annotate Discord chat messages to improve later \
semantic retrieval. You'll see a window of recent channel messages. Some are \
marked [NEW] — your job is to produce one short English annotation for each \
non-trivial [NEW] message. Messages marked [BOT] are the bot's own prior \
replies — use them as context to interpret human messages (e.g. resolve \
"对" / "好的" / "okay" against what the bot just said) but DO NOT annotate \
them.

Each annotation should:

1. Resolve pronouns and implicit references (那个 → the specific thing).
2. Name the people / products / topics mentioned.
3. Capture intent if non-obvious.

For trivial messages — greetings, reactions ("+1", "lol", single emoji, \
"yes" / "好的"), one-word replies — OMIT them from your output entirely.

If a [NEW] message references a topic clearly older than the visible window, \
call `search_memories` to look it up before annotating. Use the timestamps \
in the window to estimate how far back to search.

You may produce a single annotation spanning multiple consecutive [NEW] \
messages if they form one coherent thought — set message_ids to all involved.

Each annotation should be one English sentence, under 30 words.

Output JSON matching the schema. Annotations only — no preamble, no \
explanation."""

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


class Annotator:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncAnthropic()
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
            # Bot's own replies enter the window as context (so user
            # follow-ups like "好的" / "对" can be resolved against them)
            # but are pre-marked annotated so they neither trigger the
            # batch threshold nor get sent for annotation themselves.
            "annotated": is_bot_reply,
            "is_bot_reply": is_bot_reply,
        })

    def recent_n(self, channel_id: str, n: int = 10) -> list[dict[str, Any]]:
        """Return the last N messages from the in-memory buffer.

        Used by proactive dispatch — cheap, no Mem0 round-trip. If the
        buffer hasn't been initialized for this channel (bot hasn't seen
        traffic since restart), returns empty.
        """
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
        """Tick handler — flush any channel that crossed a threshold."""
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

            @beta_async_tool
            async def search_memories(
                query: str,
                since: str | None = None,
                until: str | None = None,
            ) -> str:
                """Search older channel history (outside the visible window)
                for context. Scoped to original messages, not annotations.

                Args:
                    query: Semantic description of the topic to find.
                    since: ISO 8601, only messages at/after this time.
                    until: ISO 8601, only messages at/before this time.
                """
                results = await memory.search(
                    query=query,
                    channel_id=channel_id,
                    since=since,
                    until=until,
                    record_type="message",
                    limit=8,
                )
                return _format_search_results(results)

            log.info(
                "annotator: flushing channel=%s window=%d new=%d",
                channel_id, len(buf), len(new_msgs),
            )

            try:
                runner = self._client.beta.messages.tool_runner(
                    model=MODEL,
                    max_tokens=2048,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=[search_memories],
                    output_config={
                        "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}
                    },
                    messages=[{"role": "user", "content": window_text}],
                )
                last_message = None
                turn = 0
                async for message in runner:
                    turn += 1
                    log_message_blocks(message, prefix=f"annotator[{channel_id[:8]} t{turn}]")
                    last_message = message
                if last_message is None:
                    return
                raw = "".join(
                    b.text for b in last_message.content
                    if getattr(b, "type", None) == "text"
                ).strip()
                parsed = json.loads(raw)
                annotations = parsed.get("annotations", [])
            except Exception:
                log.exception("annotator LLM call failed for channel %s", channel_id)
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

            # Mark all newly-considered msgs annotated (including trivial ones
            # that Sonnet omitted — we already paid the inference cost, no
            # point re-considering them next flush).
            for m in buf:
                if m["message_id"] in new_ids:
                    m["annotated"] = True

            self._last_flush_mono[channel_id] = time.monotonic()
            log.info(
                "annotator: channel=%s stored=%d trivial_or_skipped=%d",
                channel_id, stored, len(new_msgs) - stored,
            )


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
        text = _strip_legacy_author_prefix(r.get("memory") or r.get("text") or "", author)
        who = f"{author}: " if author else ""
        lines.append(f"- [{ts}] {who}{text}")
    return "\n".join(lines)
