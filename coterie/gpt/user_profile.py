"""ProfileBuilder — OpenAI variant. Mirrors user_profile.py."""

import logging
import time
from typing import Any

from openai import AsyncOpenAI

from coterie import config
from coterie.memory import BotMemory

log = logging.getLogger("dc-agent.profile")

MODEL = "gpt-5.5"
MIN_MESSAGES = 3
MAX_MESSAGES_FOR_BUILD = 100
MAX_FETCH_BUFFER = 200

REFRESH_MSG_THRESHOLD = 10
REFRESH_AGE_SEC = 24 * 3600
MIN_REBUILD_INTERVAL_SEC = 3600

SYSTEM_PROMPT = config.render("""You write a short identity profile for a {platform} \
{community_domain}-channel member.

You may receive a CURRENT PROFILE (the previous version of this user's \
summary). If present, treat it as a prior best guess: keep identity facts \
(names, aliases) unless the new messages explicitly correct them; update \
personality / style / preferences if the recent messages show a different \
or sharper picture. Drop claims the new evidence no longer supports.

If no CURRENT PROFILE is given, just build from the messages alone.

The profile is loaded every time this user @s the bot, so it must be \
LIGHTWEIGHT (under 300 characters total).

Output format (markdown headings + content). Each section is optional — \
skip a section entirely if you don't have evidence. Do NOT pad with guesses.

```
### Identity
<display name + any other names / aliases if visible from messages>

### Personality
<observed personality — 1 short phrase>

### Speaking style
<language patterns — long sentences vs short? formal vs casual? frequent words?>

### Taste / Preferences
<topic preferences, tools they like, opinions they've expressed>

### Bot interaction preference
<ONLY include this section if there's explicit evidence of how this user \
wants the bot to engage. Examples: "wants bot to participate proactively", \
"only respond when @-ed", "dislikes bot interrupting". If no explicit \
signal, OMIT this entire section.>
```

Rules:
- Each section body is one short phrase, NOT a sentence with clauses.
- Omit a section entirely (heading + body) if you have no real evidence.
- Write the body in the language the user themselves speaks in the channel.
- No preamble, no quoted excerpts.
- Blank line between sections.
- If too few or too generic messages AND no current profile, output ONLY:
  ```
  ### Identity
  <display name> (insufficient signal for personality/style yet)
  ```
""")


class ProfileBuilder:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncOpenAI()

    async def ensure(
        self, *, channel_id: str, author: str
    ) -> str | None:
        existing = await self._memory.get_profile(
            channel_id=channel_id, author=author
        )
        if existing:
            text = (existing.get("memory") or existing.get("text") or "").strip()
            if text:
                return text
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
            response = await self._client.responses.create(
                model=MODEL,
                instructions=SYSTEM_PROMPT,
                input=[{"role": "user", "content": user_msg}],
            )
        except Exception:
            log.exception("profile build failed for %s/%s", channel_id, author)
            return None
        text = (getattr(response, "output_text", None) or "").strip()
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


def _strip_legacy_author_prefix(text: str, author: str) -> str:
    prefix = f"{author}: "
    if author and text.startswith(prefix):
        return text[len(prefix):]
    return text


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) < 3:
        print("Usage: python user_profile_openai.py <channel_id> <author_display_name>")
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
