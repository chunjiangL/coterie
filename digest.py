"""Daily / weekly channel digest.

Pulled on a fixed schedule from bot.py. Computes the time window in
America/Los_Angeles, fetches every record (message + annotation +
agent_reply) in the window via memory.list_window, and sends them to
Sonnet 4.6 + web_search to produce a markdown summary with 2-3 reading
recommendations.

The agent sees record_type tags on each line, so it can ignore prior
[agent_reply] entries (yesterday's own digest) as source material —
no special filter needed; the tagging + system prompt handle it.

CLI for manual verification:
    python digest.py <channel_id> daily
    python digest.py <channel_id> weekly
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from anthropic import AsyncAnthropic

import config
from agent import _strip_legacy_author_prefix, log_message_blocks
from memory import BotMemory

log = logging.getLogger("dc-agent.digest")

MODEL = "claude-opus-4-7"
LA = ZoneInfo("America/Los_Angeles")

DAILY_SYSTEM = config.render("""You write a daily digest for a {platform} {community_domain} channel.

You'll receive a chronological list of memory records from a 24-hour window. \
Each line is tagged:
- `[message]` — real human messages. Ground truth — summarize these.
- `[annotation]` — LLM-generated English context for messages. Reliable, use \
to disambiguate references.
- `[agent_reply]` — YOUR OWN past replies / past digests. IGNORE for the \
purpose of summarization — they are not source material. (They may help you \
understand follow-up messages from humans, but do not summarize them.)

Your job:
1. Identify 2-5 main topics or threads from the [message] records. A \
direct {community_domain} question someone @-ed the bot (e.g. "reward 怎么设计", \
"<url> 这个的技术路线是什么") counts as its own topic — surface it by name \
with who asked, even if it was a one-off question with no further \
discussion. Don't roll those into a general theme.
2. For each topic, write 1-2 sentences citing who said what and the key \
takeaway.
3. Use the `web_search` tool to surface 2-3 ADJACENT reading items that \
EXTEND the discussion — material the group has NOT already cited.

   What "adjacent" means:
   - First, scan the [message] records for any papers, URLs, or named \
techniques the group has already cited. These are OFF-LIMITS for \
recommendation — they already know about them.
   - Then search for: (a) papers building on or following up the cited \
work, (b) alternative approaches in the same {community_domain} direction, (c) \
explainer / tutorial material for concepts mentioned by name without \
context, (d) recent SOTA (past 1-2 weeks where possible) in the topic area.

   How to search well:
   - Use specific terms from the discussion (technical jargon, model \
names, exact concepts), but with a TWIST — search for "follow-up", \
"vs <cited paper>", "<concept> 2026", an author from a cited paper's \
later work, etc.
   - Evaluate results. If hits look like the same papers already in \
chat, search again with different framing.
   - 2-4 search calls is normal. Drop weak hits.
   - **If you genuinely can't find new adjacent material, output FEWER \
items, or skip 📚 entirely. 0 recommendations is far better than echoing \
papers the group already named.**

   Each recommendation needs a one-line note on why it extends what was \
discussed (NOT just "this is the paper they mentioned").
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
- Fourth paper? New numbered item (4. ...). Never pile `why:` lines on \
an existing item.
- The 📚 section is optional. Omit it entirely if you couldn't find new \
adjacent material — DO NOT pad with already-cited papers.
- Don't fabricate URLs or arxiv IDs. Only cite what web_search actually \
returned.

Substantive content is about CONTENT, not count. A 4-character question like \
`reward 怎么设计` IS substantive — it's a real {community_domain} question. Don't \
dismiss messages as trivial just because they're short. The only things \
genuinely skippable: greetings ("hi"), reactions ("+1", "lol"), and pure \
logistics ("ok", "done", "拉进来了").

Always produce topic bullets if there's any {community_domain} content at all. Only \
say "no {community_domain} discussion this window" if literally every message was a \
greeting / reaction / logistics. 📚 is optional (omit if no adjacent material \
found), but topic bullets are not.

Be concise. No preamble, no meta-narration about your retrieval process.
""")

WEEKLY_SYSTEM = config.render("""You write a weekly digest for a {platform} {community_domain} channel.

You'll receive a chronological list of memory records from a 7-day window. \
Each line is tagged:
- `[message]` — real human messages. Ground truth.
- `[annotation]` — LLM-generated English context. Reliable.
- `[agent_reply]` — YOUR OWN past replies / past daily digests. IGNORE for \
summarization (they are not source material).

Your job:
1. Identify 3-7 themes across the week — group related threads even if they \
spanned multiple days. A direct {community_domain} question someone @-ed the bot \
(e.g. "reward 怎么设计", "<url> 这个的技术路线是什么") gets its own theme \
named by the question — don't fold a specific question into a broader \
theme. Surface it by name with who asked.
2. For each theme, write 2-3 sentences: what was discussed, who drove it, \
key conclusions or open questions.
3. Use `web_search` to surface 3-5 ADJACENT reading items that EXTEND \
the week's themes — material the group has NOT already cited.

   First scan [message] records for cited papers/URLs/techniques — those \
are OFF-LIMITS. Then search for: (a) follow-ups to cited work, (b) \
alternative approaches, (c) explainers for concepts named without context, \
(d) recent SOTA in the topic area (past 2-3 weeks).

   Evaluate results; if hits echo already-cited papers, reformulate \
(search "vs <X>", "<concept> 2026", author's later work). 3-6 search \
calls is normal.

   **If no genuinely new adjacent material can be found, output FEWER \
items or skip 📚 entirely. Echoing papers the group already named is \
worse than no recommendations.**

   One-line relevance note per item — say WHY it extends, not "this is \
the paper they mentioned".
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
   why: <one-line note on how this extends the discussion>
2. <Title> — <url>
   why: ...
3. ...
```

Format rules (strict):
- Each numbered item has EXACTLY ONE `why:` line.
- Fourth paper? New numbered item. Never pile `why:` lines on existing items.
- The 📚 section is optional. Omit entirely if no new adjacent material \
found — DO NOT pad with already-cited papers.
- Don't fabricate URLs or arxiv IDs.

Substantive content is about CONTENT, not count. A short question like \
`reward 怎么设计` IS substantive. Don't dismiss messages as trivial just \
because they're short. Only genuinely skippable: greetings, reactions, \
logistics ("ok", "done", "拉进来了").

Always produce theme bullets if there's any {community_domain} content at all. Only \
say "no {community_domain} discussion this week" if literally everything was \
greetings/reactions/logistics. 📚 is optional (omit if no adjacent material). \
Be concise.
""")


class Digest:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncAnthropic()

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
            runner = self._client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                output_config={"effort": "xhigh"},
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=[
                    {"type": "web_search_20260209", "name": "web_search"},
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
            last_message = None
            turn = 0
            async for message in runner:
                turn += 1
                log_message_blocks(
                    message, prefix=f"digest-{label}[{channel_id[:8]} t{turn}]"
                )
                last_message = message
        except Exception:
            log.exception("digest %s LLM call failed for channel %s", label, channel_id)
            return None

        if last_message is None:
            return None
        text = "".join(
            b.text for b in last_message.content
            if getattr(b, "type", None) == "text"
        ).strip()
        return text or None


def _format_records(records: list[dict[str, Any]]) -> str:
    lines = []
    for r in records:
        meta = r.get("metadata") or {}
        rt = meta.get("record_type", "message")
        ts = meta.get("timestamp", "?")
        author = meta.get("author", "?")
        text = _strip_legacy_author_prefix(r.get("memory") or r.get("text") or "", author)
        lines.append(f"[{rt}] [{ts}] {author}: {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage: python digest.py <channel_id> [daily|weekly]")
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
