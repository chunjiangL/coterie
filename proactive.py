"""Proactive participation: classify non-@ messages, fire if research-y.

Flow per incoming non-@ message in a PROACTIVE_CHANNELS-listed channel:

    msg arrives
        ↓
    debounce 8s          (if another msg comes in same channel, restart timer)
        ↓
    classify (Opus 4.7 + structured JSON)
        ↓
    fire == true ?
        ↓ yes
    Mem0 search with classifier's search_query → top 5
        ↓
    agent.reply(mode="proactive", recent_context=..., prior_relevant=...)
        ↓
    agent emits "<skip>" ? → drop
        ↓ otherwise
    @user_id prepend → channel.send
        ↓
    mark cooldown (5 min) so this channel quiets for a bit

The classifier looks at: trigger msg + speaker profile + last 10 channel
msgs. NO Mem0 search at classify-time (saves ~0.3s); search happens only
after the classifier says yes, then feeds into agent. Classifier emits
search_query precisely so the post-fire Mem0 search is well-targeted —
trigger msg alone often makes a bad semantic query.
"""

import asyncio
import json
import logging
import time
from typing import Any

from anthropic import AsyncAnthropic

import config
from agent import _strip_legacy_author_prefix
from memory import BotMemory

log = logging.getLogger("dc-agent.proactive")

MODEL = "claude-opus-4-7"

DEBOUNCE_SEC = 8.0           # wait this long after last msg before classifying
COOLDOWN_SEC = 60            # min interval between fires per channel for DIFFERENT askers
MAX_RECENT_CTX = 10

CLASSIFIER_SYSTEM = config.render("""You decide whether a bot should proactively \
join a {platform} {community_domain}-channel conversation. The bot is NOT @-ed; \
it's listening, and you control whether it speaks.

You see: a TRIGGER MESSAGE (the new message that just arrived), the \
speaker's profile, and recent channel context.

Decide fire=true ONLY when ALL hold:
1. The trigger contains substantive {community_domain} signal. This includes:
   (a) a real {community_domain} question or non-trivial technical claim about \
{substantive_topics};
   (b) a BARE link to {community_domain}-relevant content — {relevant_link_domains}. \
A naked URL with no accompanying text counts as fire-worthy if the URL itself \
signals "this is relevant" — the agent can fetch it and add value. Don't \
reject just because the human didn't write a sentence around the link.
   (c) a screenshot or image clearly showing {community_domain} content (paper \
title, code, training curve, etc.) — same logic as (b).
2. The bot can plausibly add value: concrete data, references, a \
contrarian point, prior context from this channel that's relevant. NOT \
echo / summary / general encouragement.
3. The trigger was NOT already answered in the recent context (someone \
just gave the answer 1-2 msgs ago → skip).
4. The trigger is NOT about the bot itself (e.g. "isaac 你怎么不主动", \
"bot 烦死了", meta-complaints about behavior). Those are for the people \
to discuss, not for the bot to defend itself.
5. The trigger is NOT pure chitchat / logistics / reactions ("吃了吗", \
"+1", "好的", "拉进来了", emoji-only).
6. The trigger is NOT a non-{community_domain} link (random news article, meme, \
unrelated tweet about food/politics/etc.). Use judgement on the URL domain \
+ the channel's {community_domain} focus.

═══ Speaker preference override (strong signal) ═══

The speaker's profile may include a `Bot 互动偏好` line. Respect it:
- "希望 bot 主动参与" → on borderline cases, lean fire=true. The user has \
explicitly asked for proactive engagement.
- "只在被 @ 时回复" / "讨厌 bot 插话" → fire=false unless the trigger is \
exceptionally substantive (clear technical question that begs the bot's \
specific channel memory). Default fire=false for this group; they will \
@ when they want the bot.
- No `Bot 互动偏好` line → judge purely on the 5 criteria above.

If fire=true, also emit `search_query`: a 3-10 word query targeting the \
SUBSTANTIVE topic (English + technical jargon preferred). E.g. trigger = \
"8 node 不够 不能一个 node 一个 expert" → search_query = "Qwen MoE \
post-training expert parallelism node count". The query is used to \
retrieve related prior discussion from this channel's memory.

For a bare link, infer the topic from the URL itself + ANY hint in the \
text (often none). E.g. trigger = "https://x.com/rayli234/status/...\
Articraft coding agent articulated 3D" → search_query = "coding agent \
articulated 3D Blender simulation assets". If the URL alone (without \
fetching) doesn't reveal the topic, pick a query from recent channel \
context — the agent will fetch the URL itself for actual content.

If fire=false, search_query = "".

Always include `reason`: one short sentence on why you decided yes/no. \
Keep it factual.

Output the JSON exactly matching the schema. No preamble, no other text.
""")

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
        self._client = AsyncAnthropic()
        self._last_fire_unix: dict[str, float] = {}
        self._last_fire_asker: dict[str, str] = {}
        # Tasks STILL in the debounce sleep — safe to cancel & replace when a
        # new msg arrives in the same channel (no expensive work done yet).
        self._pending_debounce: dict[str, asyncio.Task[Any]] = {}
        # Channels currently past debounce and running classifier→agent.reply.
        # New schedules for these channels DROP silently; cancelling them mid-
        # flight would kill an in-progress reply (the classic bug fixed in K7).
        self._in_flight: set[str] = set()

    def cooldown_active(self, channel_id: str, *, asker: str | None = None) -> bool:
        """Cooldown is per-channel, but if the new message is from the SAME
        speaker as the last fire, bypass it — follow-up questions from the
        same person should never be silently dropped.

        Pure spam-prevention: blocks "bot replies to A, then 30s later
        chimes in on B's unrelated msg." Doesn't block "A asks, A clarifies."
        """
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
        """Debounced schedule.

        - If a previous dispatch is still in the debounce wait for this
          channel, cancel it (we'll replace with this newer one).
        - If a previous dispatch is already PAST debounce and running its
          classifier/agent.reply, leave it alone and DROP this new attempt.
          The in-flight reply will see the new message anyway via the
          annotator's shared recent-buffer.
        """
        # Cancel any pre-debounce task; safe because no LLM work has started.
        pending = self._pending_debounce.pop(channel_id, None)
        if pending and not pending.done():
            pending.cancel()

        if channel_id in self._in_flight:
            log.info(
                "proactive: in-flight in channel=%s, dropping new dispatch",
                channel_id,
            )
            coro.close()  # avoid "coroutine was never awaited" warning
            return

        in_flight = self._in_flight
        pending_debounce = self._pending_debounce

        async def runner() -> None:
            try:
                await asyncio.sleep(DEBOUNCE_SEC)
            except asyncio.CancelledError:
                coro.close()
                return
            # Crossed the debounce boundary — atomically (single-threaded
            # asyncio) move from "pending" to "in-flight" before awaiting
            # the dispatch. This is what protects an in-flight reply from
            # being cancelled when a later msg in the same channel calls
            # pending_debounce.pop().cancel().
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
            resp = await self._client.beta.messages.create(
                model=MODEL,
                max_tokens=512,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": "xhigh",
                    "format": {"type": "json_schema", "schema": CLASSIFIER_SCHEMA},
                },
                system=[
                    {
                        "type": "text",
                        "text": CLASSIFIER_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception:
            log.exception("classifier API call failed")
            return None
        text = "".join(
            b.text for b in resp.content if getattr(b, "type", None) == "text"
        ).strip()
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
    """Format annotator-buffer-shaped dicts as a multiline block for prompts."""
    lines = []
    for m in items:
        author = m.get("author", "?")
        ts = m.get("timestamp", "?")
        content = m.get("content", "") or ""
        tag = "[BOT]" if m.get("is_bot_reply") else "[message]"
        lines.append(f"{tag} [{ts}] {author}: {content}")
    return "\n".join(lines)


def format_prior_memories(records: list[dict[str, Any]]) -> str:
    """Format Mem0 records (after channel.search) for prompts."""
    lines = []
    for r in records:
        meta = r.get("metadata") or {}
        rt = meta.get("record_type", "message")
        ts = meta.get("timestamp", "?")
        author = meta.get("author", "?")
        text = _strip_legacy_author_prefix(
            r.get("memory") or r.get("text") or "", author
        )
        lines.append(f"[{rt}] [{ts}] {author}: {text}")
    return "\n".join(lines)
