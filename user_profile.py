"""Lightweight per-user profile, built from message history.

For each (channel, author) we maintain a short profile blurb covering
identity (names / aliases), personality, speaking style, and visible
tastes/preferences. Loaded on every @ mention to give the agent baseline
context about who's asking — small enough not to dominate the prompt
(< 300 chars typical) but specific enough to shift tone and disambiguate.

Build strategy: lazy first build on first @, then periodic refresh driven
from bot.py's tick. A profile is rebuilt when EITHER the user has produced
≥10 new messages since the last build, OR the profile is ≥24h old AND the
user has produced at least one new message since. A 1h cooldown prevents
thrash. The first build comes from very few messages (often <10) and is
biased — refresh exists specifically to correct that initial signal as
the user accumulates real material.
"""

import logging
import time
from typing import Any

from anthropic import AsyncAnthropic

from agent import _strip_legacy_author_prefix, log_message_blocks
from memory import BotMemory

log = logging.getLogger("dc-agent.profile")

MODEL = "claude-sonnet-4-6"
MIN_MESSAGES = 3
MAX_MESSAGES_FOR_BUILD = 100      # latest N messages fed to LLM
MAX_FETCH_BUFFER = 200            # fetch buffer; sort DESC by ts, then slice top N

# Refresh policy: rebuild when one of these is met (subject to cooldown).
REFRESH_MSG_THRESHOLD = 10        # ≥N new msgs since last build → rebuild
REFRESH_AGE_SEC = 24 * 3600       # OR ≥24h old (catches slow stylistic drift)
MIN_REBUILD_INTERVAL_SEC = 3600   # never rebuild same profile twice within 1h

SYSTEM_PROMPT = """You write a short identity profile for a Discord \
research-channel member.

You may receive a CURRENT PROFILE (the previous version of this user's \
summary). If present, treat it as a prior best guess: keep identity facts \
(names, aliases) unless the new messages explicitly correct them; update \
personality / style / preferences if the recent messages show a different \
or sharper picture. Drop claims the new evidence no longer supports. The \
goal is to converge on a better summary over time, not to preserve an \
early biased one.

If no CURRENT PROFILE is given, just build from the messages alone.

The profile is loaded every time this user @s the bot, so it must be \
LIGHTWEIGHT (under 300 characters total) and only contain things you can \
actually infer from evidence.

Output format (markdown headings + content). Each section is optional — \
skip a section entirely if you don't have evidence. Do NOT pad with guesses.

```
### Identity
<display name + 中文名/英文名/昵称 if visible from messages or signatures>

### 性格
<observed personality — 1 short phrase, e.g. "直接、爱怼人", "温和、爱解释清楚">

### 说话方式
<language patterns — Chinese/English mix? 长句/短句? formal/casual? 常用词?>

### Taste/偏好
<topic preferences, tools they like, opinions they've expressed>

### Bot 互动偏好
<ONLY include this section if there's explicit evidence of how this user \
wants the bot to engage. Examples: "希望 bot 主动参与讨论" (said something \
like '你能不能自己跳出来讨论' / '别每次@你才出来'), "只在被 @ 时回复" \
(said '我没问你你别说话' / '别插话'), "讨厌 bot 插话". If no explicit \
signal, OMIT this entire section — don't even write the heading.>
```

Rules:
- Each section body is one short phrase, NOT a sentence with verbs and clauses.
- Omit a section entirely (heading + body) if you have no real evidence.
- Use Chinese for Chinese-coded fields (性格, 说话方式, 偏好) and English \
for English-coded fields (Identity, Taste).
- No preamble, no "Based on the messages...", no quoted excerpts.
- Blank line between sections (standard markdown).
- If the user has too few or too generic messages AND no current profile, \
output ONLY:
  ```
  ### Identity
  <display name> (insufficient signal for personality/style yet)
  ```
"""


class ProfileBuilder:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncAnthropic()

    async def ensure(
        self, *, channel_id: str, author: str
    ) -> str | None:
        """Return cached profile or build inline if missing.

        Returns None when the user has too few messages to characterize.
        """
        existing = await self._memory.get_profile(
            channel_id=channel_id, author=author
        )
        if existing:
            text = (existing.get("memory") or existing.get("text") or "").strip()
            if text:
                return text
        # No existing profile — build from scratch.
        return await self.rebuild(channel_id=channel_id, author=author)

    async def rebuild(
        self, *, channel_id: str, author: str
    ) -> str | None:
        existing = await self._memory.get_profile(
            channel_id=channel_id, author=author
        )
        existing_text = (
            (existing.get("memory") or existing.get("text") or "").strip()
            if existing else ""
        )

        # Fetch a buffer wider than what we feed, sort by timestamp DESC,
        # take the latest N. Without this, Mem0's get_all order is not
        # guaranteed and authors with > 100 messages would silently miss
        # their most recent ones.
        msgs = await self._memory.list_by_author(
            channel_id=channel_id,
            author=author,
            limit=MAX_FETCH_BUFFER,
        )
        if len(msgs) < MIN_MESSAGES and not existing_text:
            log.info(
                "profile skip: channel=%s author=%s only %d msgs (< %d), no prior profile",
                channel_id, author, len(msgs), MIN_MESSAGES,
            )
            return None
        msgs.sort(
            key=lambda r: (r.get("metadata") or {}).get("timestamp_unix", 0),
            reverse=True,
        )
        msgs = msgs[:MAX_MESSAGES_FOR_BUILD]
        msgs.sort(key=lambda r: (r.get("metadata") or {}).get("timestamp_unix", 0))
        body = "\n".join(
            _strip_legacy_author_prefix(
                r.get("memory") or r.get("text") or "",
                (r.get("metadata") or {}).get("author", ""),
            )
            for r in msgs
        )
        parts = []
        if existing_text:
            parts.append(f"CURRENT PROFILE (previous summary):\n{existing_text}\n")
        parts.append(f"Display name: {author}")
        parts.append(f"Message count (latest {len(msgs)}):")
        parts.append(body)
        user_msg = "\n".join(parts)

        log.info(
            "profile build: channel=%s author=%s n_msgs=%d",
            channel_id, author, len(msgs),
        )
        try:
            runner = self._client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[],
                messages=[{"role": "user", "content": user_msg}],
            )
            last = None
            turn = 0
            async for message in runner:
                turn += 1
                log_message_blocks(message, prefix=f"profile[{author[:12]} t{turn}]")
                last = message
        except Exception:
            log.exception("profile build failed for %s/%s", channel_id, author)
            return None
        if last is None:
            return None
        text = "".join(
            b.text for b in last.content
            if getattr(b, "type", None) == "text"
        ).strip()
        if not text:
            return None
        try:
            await self._memory.upsert_profile(
                channel_id=channel_id,
                author=author,
                text=text,
                built_from_n_msgs=len(msgs),
            )
        except Exception:
            log.exception("profile upsert failed for %s/%s", channel_id, author)
        log.info("profile built: author=%s n_msgs=%d body=%r",
                 author, len(msgs), text[:200])
        return text

    async def maybe_refresh_channel(self, channel_id: str) -> None:
        """Walk every profile in this channel; rebuild any that crossed
        the refresh threshold. Called from bot.py's profile tick."""
        try:
            profiles = await self._memory.list_profiles(channel_id=channel_id)
        except Exception:
            log.exception("list_profiles failed for %s", channel_id)
            return
        now = time.time()
        for p in profiles:
            meta = p.get("metadata") or {}
            author = meta.get("author")
            if not author:
                continue
            last_built = meta.get("last_built_unix") or meta.get("timestamp_unix") or 0
            age = now - float(last_built)
            if age < MIN_REBUILD_INTERVAL_SEC:
                continue
            try:
                new_msgs = await self._memory.list_by_author(
                    channel_id=channel_id,
                    author=author,
                    since=float(last_built),
                    limit=REFRESH_MSG_THRESHOLD + 1,
                )
            except Exception:
                log.exception("count new msgs failed for %s/%s", channel_id, author)
                continue
            n_new = len(new_msgs)
            should_refresh = (
                n_new >= REFRESH_MSG_THRESHOLD
                or (age >= REFRESH_AGE_SEC and n_new >= 1)
            )
            if not should_refresh:
                continue
            log.info(
                "profile refresh: channel=%s author=%s age=%.0fmin new=%d",
                channel_id, author, age / 60, n_new,
            )
            try:
                await self.rebuild(channel_id=channel_id, author=author)
            except Exception:
                log.exception("profile rebuild failed for %s/%s", channel_id, author)


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) < 3:
        print("Usage: python user_profile.py <channel_id> <author_display_name>")
        sys.exit(1)
    channel_id = sys.argv[1]
    author = sys.argv[2]

    async def _main() -> None:
        m = BotMemory()
        pb = ProfileBuilder(memory=m)
        text = await pb.rebuild(channel_id=channel_id, author=author)
        print("=" * 60)
        print(text or "(insufficient messages to build profile)")
        print("=" * 60)

    asyncio.run(_main())
