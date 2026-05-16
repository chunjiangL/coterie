"""Daily / weekly digest — OpenAI variant.

Same external surface as digest.py (Digest.daily / Digest.weekly returning
markdown text). Uses GPT-5.5 Pro + built-in web_search.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

import config
from memory import BotMemory

log = logging.getLogger("dc-agent.digest")

MODEL = "gpt-5.5-pro"
REASONING_EFFORT = "xhigh"
LA = ZoneInfo("America/Los_Angeles")

DAILY_SYSTEM = config.render("""You write a daily digest for a {platform} {community_domain} channel.

You'll receive a chronological list of memory records from a 24-hour window. \
Each line is tagged:
- `[message]` — real human messages. Ground truth — summarize these.
- `[annotation]` — LLM-generated English context for messages. Reliable, use \
to disambiguate references.
- `[agent_reply]` — YOUR OWN past replies / past digests. IGNORE for the \
purpose of summarization.

Your job:
1. Identify 2-5 main topics from the [message] records. A direct {community_domain} \
question someone @-ed the bot counts as its own topic — surface it by name \
with who asked, even if it was a one-off question.
2. For each topic, write 1-2 sentences citing who said what and the takeaway.
3. Use the `web_search` tool to surface 2-3 ADJACENT reading items that \
EXTEND the discussion — material the group has NOT already cited.

   What "adjacent" means:
   - First, scan the [message] records for any papers/URLs/techniques the \
group has already cited. These are OFF-LIMITS.
   - Then search for: (a) follow-ups to cited work, (b) alternative \
approaches, (c) explainers for concepts named without context, (d) recent \
SOTA in the topic area.

   How to search well:
   - Use specific technical terms, but with a TWIST (search "follow-up", \
"vs <cited paper>", "<concept> 2026", an author's later work, etc.).
   - 2-4 search calls is normal. Drop weak hits.
   - **If you genuinely can't find new adjacent material, output FEWER \
items, or skip 📚 entirely. 0 recommendations is better than echoing \
already-cited papers.**

   Each recommendation needs a one-line note on why it extends the \
discussion (NOT just "this is the paper they mentioned").
4. Match the channel's tone (Chinese/English mixed is fine).

Output format (markdown) — follow EXACTLY:
```
📋 <日期> 讨论摘要

主要话题:
• <topic 1> — <who said what, takeaway>
• <topic 2> — ...

📚 延伸阅读（群里未提及）
1. <Title> — <url>
   why: <one-line note on how this extends the discussion>
2. <Title> — <url>
   why: ...
3. ...
```

Format rules (strict):
- Each numbered item has EXACTLY ONE `why:` line.
- Fourth paper? New numbered item. Never pile `why:` lines on existing items.
- The 📚 section is optional. Omit entirely if no adjacent material.
- Don't fabricate URLs. Only cite what web_search actually returned.

Substantive content is about CONTENT, not count. A short question like \
`reward 怎么设计` IS substantive. Only genuinely skippable: greetings, \
reactions ("+1", "lol"), logistics ("ok", "done").

Always produce topic bullets if there's any {community_domain} content at all. Be \
concise. No preamble, no meta-narration about retrieval.""")

WEEKLY_SYSTEM = config.render("""You write a weekly digest for a {platform} {community_domain} channel.

You'll receive a chronological list of memory records from a 7-day window. \
Each line is tagged:
- `[message]` — real human messages. Ground truth.
- `[annotation]` — LLM-generated English context. Reliable.
- `[agent_reply]` — YOUR OWN past replies / past digests. IGNORE.

Your job:
1. Identify 3-7 themes across the week. A direct {community_domain} question someone \
@-ed the bot gets its own theme named by the question.
2. For each theme, write 2-3 sentences: what was discussed, who drove it, \
key conclusions or open questions.
3. Use `web_search` to surface 3-5 ADJACENT reading items that EXTEND the \
week's themes — material the group has NOT already cited.

   First scan [message] records for cited papers/URLs — OFF-LIMITS. Search \
for (a) follow-ups, (b) alternatives, (c) explainers, (d) recent SOTA.
   Evaluate; reformulate if hits echo cited papers. 3-6 search calls normal.
   **If no genuinely new adjacent material, output FEWER items or skip \
📚 entirely.**

One-line relevance note per item — say WHY it extends.
4. Match channel tone.

Output format (markdown) — follow EXACTLY:
```
📅 <本周日期范围> 周报

本周主题:
1. <theme 1>
   <2-3 sentence summary>
2. <theme 2>
   ...

📚 本周延伸阅读（群里未提及）
1. <Title> — <url>
   why: ...
```

Format rules:
- One `why:` per item.
- 📚 optional. Don't fabricate URLs.

Be concise.""")


class Digest:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncOpenAI()

    async def daily(self, channel_id: str) -> str | None:
        now_la = datetime.now(LA)
        until = now_la.replace(hour=0, minute=0, second=0, microsecond=0)
        since = until - timedelta(days=1)
        return await self._compose(
            channel_id=channel_id,
            since=since,
            until=until,
            system_prompt=DAILY_SYSTEM,
            label="daily",
        )

    async def weekly(self, channel_id: str) -> str | None:
        now_la = datetime.now(LA)
        until = now_la.replace(hour=0, minute=0, second=0, microsecond=0)
        since = until - timedelta(days=7)
        return await self._compose(
            channel_id=channel_id,
            since=since,
            until=until,
            system_prompt=WEEKLY_SYSTEM,
            label="weekly",
        )

    async def _compose(
        self,
        *,
        channel_id: str,
        since: datetime,
        until: datetime,
        system_prompt: str,
        label: str,
    ) -> str | None:
        records = await self._memory.list_window(
            channel_id=channel_id,
            since=since.isoformat(),
            until=until.isoformat(),
            limit=500,
        )
        log.info(
            "digest %s: channel=%s window=[%s..%s] records=%d",
            label, channel_id, since.isoformat(), until.isoformat(), len(records),
        )
        if not records:
            return None

        records.sort(
            key=lambda r: (r.get("metadata") or {}).get("timestamp_unix", 0)
        )
        records_text = _format_records(records)
        now_iso = datetime.now(LA).isoformat()
        user_msg = (
            f"[Current time: {now_iso}] [Window: {since.isoformat()} to {until.isoformat()}]\n"
            f"[Channel records: {len(records)}]\n\n"
            f"{records_text}"
        )

        try:
            response = await self._client.responses.create(
                model=MODEL,
                instructions=system_prompt,
                input=[{"role": "user", "content": user_msg}],
                tools=[{"type": "web_search"}],
                reasoning={"effort": REASONING_EFFORT},
            )
        except Exception:
            log.exception("digest %s LLM call failed for channel %s", label, channel_id)
            return None

        text = (getattr(response, "output_text", None) or "").strip()
        return text or None


def _format_records(records: list[dict[str, Any]]) -> str:
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


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python digest_openai.py <channel_id> [daily|weekly]")
        sys.exit(1)
    channel_id = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "daily"

    async def _main() -> None:
        m = BotMemory()
        d = Digest(memory=m)
        fn = d.daily if mode == "daily" else d.weekly
        text = await fn(channel_id)
        print("=" * 60)
        print(text or "(no content)")
        print("=" * 60)

    asyncio.run(_main())
