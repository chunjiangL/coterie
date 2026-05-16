"""Proactive classifier — OpenAI variant.

Same external surface as proactive.py (ProactiveClassifier with schedule /
cooldown_active / mark_fired / evaluate). Uses GPT-5.5 Pro + Responses-API
`text.format` json_schema instead of Anthropic's output_config.
"""

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

log = logging.getLogger("dc-agent.proactive")

MODEL = "gpt-5.5-pro"
REASONING_EFFORT = "xhigh"

DEBOUNCE_SEC = 8.0
COOLDOWN_SEC = 60
MAX_RECENT_CTX = 10

CLASSIFIER_SYSTEM = """You decide whether a bot should proactively join a \
Discord research-channel conversation. The bot is NOT @-ed; it's listening, \
and you control whether it speaks.

You see: a TRIGGER MESSAGE (the new message that just arrived), the \
speaker's profile, and recent channel context.

Decide fire=true ONLY when ALL hold:
1. The trigger contains substantive research signal. This includes:
   (a) a real research question or non-trivial technical claim about \
model architecture, training, papers, datasets, infra, RL/RLHF, agent \
design, code, hardware, etc.;
   (b) a BARE link to research-relevant content — arxiv abstract/pdf, \
GitHub repo about ML / agents / research code, X.com (Twitter) post about \
a paper or project, huggingface model card, anthropic / openai / deepmind \
/ thinkingmachines blog post, etc. A naked URL with no accompanying text \
counts as fire-worthy if the URL itself signals "this is relevant".
   (c) a screenshot or image clearly showing research content (paper \
title, code, training curve, etc.) — same logic as (b).
2. The bot can plausibly add value: concrete data, references, a \
contrarian point, prior context from this channel that's relevant. NOT \
echo / summary / general encouragement.
3. The trigger was NOT already answered in the recent context.
4. The trigger is NOT about the bot itself.
5. The trigger is NOT pure chitchat / logistics / reactions / emoji-only.
6. The trigger is NOT a non-research link.

═══ Speaker preference override (strong signal) ═══

The speaker's profile may include a `Bot 互动偏好` line. Respect it:
- "希望 bot 主动参与" → on borderline cases, lean fire=true.
- "只在被 @ 时回复" / "讨厌 bot 插话" → fire=false unless the trigger is \
exceptionally substantive.
- No preference → judge purely on the 6 criteria above.

If fire=true, also emit `search_query`: a 3-10 word query targeting the \
SUBSTANTIVE topic (English + technical jargon preferred). E.g. trigger = \
"8 node 不够 不能一个 node 一个 expert" → search_query = "Qwen MoE \
post-training expert parallelism node count".

For a bare link, infer the topic from the URL itself + ANY hint in the \
text. If the URL alone doesn't reveal the topic, pick a query from recent \
channel context.

If fire=false, search_query = "".

Always include `reason`: one short sentence on why you decided. Be factual.

Output ONLY the JSON object — no preamble, no markdown fence."""

CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "fire": {"type": "boolean"},
        "search_query": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["fire", "search_query", "reason"],
    "additionalProperties": False,
}


class ProactiveClassifier:
    def __init__(self) -> None:
        self._client = AsyncOpenAI()
        self._last_fire_unix: dict[str, float] = {}
        self._last_fire_asker: dict[str, str] = {}
        self._pending_debounce: dict[str, asyncio.Task[Any]] = {}
        self._in_flight: set[str] = set()

    def cooldown_active(self, channel_id: str, *, asker: str | None = None) -> bool:
        last = self._last_fire_unix.get(channel_id, 0.0)
        if (time.time() - last) >= COOLDOWN_SEC:
            return False
        if asker and self._last_fire_asker.get(channel_id) == asker:
            return False
        return True

    def mark_fired(self, channel_id: str, *, asker: str) -> None:
        self._last_fire_unix[channel_id] = time.time()
        self._last_fire_asker[channel_id] = asker

    def schedule(self, channel_id: str, coro: Any) -> None:
        pending = self._pending_debounce.pop(channel_id, None)
        if pending and not pending.done():
            pending.cancel()

        if channel_id in self._in_flight:
            log.info(
                "proactive: in-flight in channel=%s, dropping new dispatch",
                channel_id,
            )
            coro.close()
            return

        in_flight = self._in_flight
        pending_debounce = self._pending_debounce

        async def runner() -> None:
            try:
                await asyncio.sleep(DEBOUNCE_SEC)
            except asyncio.CancelledError:
                coro.close()
                return
            pending_debounce.pop(channel_id, None)
            in_flight.add(channel_id)
            try:
                await coro
            except Exception:
                log.exception("proactive dispatch crashed")
            finally:
                in_flight.discard(channel_id)

        self._pending_debounce[channel_id] = asyncio.create_task(runner())

    async def evaluate(
        self,
        *,
        trigger_msg_text: str,
        asker: str,
        asker_profile: str | None,
        recent_msgs: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        recent_block = format_recent(recent_msgs)
        user_msg = (
            f"[Speaker: {asker}]\n"
            f"[Speaker profile]\n"
            f"{asker_profile or '(no profile yet)'}\n"
            f"\n"
            f"═════════ TRIGGER MESSAGE ═════════\n"
            f"{asker}: {trigger_msg_text}\n"
            f"═══════════════════════════════════\n"
            f"\n"
            f"[Recent channel context, last {len(recent_msgs)} msgs]\n"
            f"{recent_block or '(empty)'}\n"
        )
        try:
            response = await self._client.responses.create(
                model=MODEL,
                instructions=CLASSIFIER_SYSTEM,
                input=[{"role": "user", "content": user_msg}],
                reasoning={"effort": REASONING_EFFORT},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ProactiveDecision",
                        "schema": CLASSIFIER_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception:
            log.exception("classifier API call failed")
            return None
        text = (getattr(response, "output_text", None) or "").strip()
        if not text:
            # Fallback: walk output items for the first text block.
            for item in getattr(response, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for block in getattr(item, "content", []) or []:
                        if getattr(block, "type", None) in ("output_text", "text"):
                            text = getattr(block, "text", "") or ""
                            if text:
                                break
                if text:
                    break
        try:
            parsed = json.loads(text)
        except Exception:
            log.exception("classifier JSON parse failed: %r", text[:200])
            return None
        log.info(
            "proactive classify: fire=%s reason=%r query=%r asker=%s",
            parsed.get("fire"), parsed.get("reason"), parsed.get("search_query"),
            asker,
        )
        return parsed


def format_recent(items: list[dict[str, Any]]) -> str:
    lines = []
    for m in items:
        author = m.get("author", "?")
        ts = m.get("timestamp", "?")
        content = m.get("content", "") or ""
        tag = "[BOT]" if m.get("is_bot_reply") else "[message]"
        lines.append(f"{tag} [{ts}] {author}: {content}")
    return "\n".join(lines)


def format_prior_memories(records: list[dict[str, Any]]) -> str:
    lines = []
    for r in records:
        meta = r.get("metadata") or {}
        rt = meta.get("record_type", "message")
        ts = meta.get("timestamp", "?")
        author = meta.get("author", "?")
        body = r.get("memory") or r.get("text") or ""
        prefix = f"{author}: "
        if author and body.startswith(prefix):
            body = body[len(prefix):]
        lines.append(f"[{rt}] [{ts}] {author}: {body}")
    return "\n".join(lines)
