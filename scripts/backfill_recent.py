"""Backfill missed Discord messages from a channel into Mem0.

Use when the bot was down during a window. Discord's gateway doesn't
replay missed events on reconnect, so REST fetch is the only path.

Optionally calls agent.reply() on the most recent non-bot message and
posts the reply back to the channel via REST. The live bot should be
stopped before running with --reply, both to avoid Mem0 write contention
and to prevent two bot instances racing to respond.

Usage:
    python scripts/backfill_recent.py <channel_id> [--since-minutes N] [--reply]
"""

import argparse
import asyncio
import datetime as _dt
import logging
import os

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("backfill")

API = "https://discord.com/api/v10"


async def _get_bot_user(client: httpx.AsyncClient, token: str) -> dict:
    r = await client.get(
        f"{API}/users/@me", headers={"Authorization": f"Bot {token}"}
    )
    r.raise_for_status()
    return r.json()


async def _fetch_messages_since(
    client: httpx.AsyncClient, token: str, channel_id: str, since_iso: str
) -> list[dict]:
    """Paginate channel history until we cross the since cutoff. Returns
    messages sorted oldest first.

    Discord's GET messages endpoint returns most-recent first and accepts a
    `before=<id>` cursor; we keep walking back until the batch's oldest
    crosses since_iso, then filter client-side."""
    headers = {"Authorization": f"Bot {token}"}
    all_msgs: list[dict] = []
    before: str | None = None
    while True:
        url = f"{API}/channels/{channel_id}/messages?limit=100"
        if before:
            url += f"&before={before}"
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_msgs.extend(batch)
        oldest_ts = batch[-1]["timestamp"]
        if oldest_ts < since_iso:
            break
        before = batch[-1]["id"]
        if len(all_msgs) >= 500:
            log.warning("hit 500-message cap; stopping pagination")
            break
    filtered = [m for m in all_msgs if m["timestamp"] >= since_iso]
    filtered.sort(key=lambda m: m["timestamp"])
    return filtered


async def _post_reply(
    client: httpx.AsyncClient,
    token: str,
    channel_id: str,
    content: str,
    reference_id: str,
) -> None:
    body = {
        "content": content,
        "message_reference": {
            "message_id": reference_id,
            "fail_if_not_exists": False,
        },
        "allowed_mentions": {"replied_user": True},
    }
    r = await client.post(
        f"{API}/channels/{channel_id}/messages",
        json=body,
        headers={"Authorization": f"Bot {token}"},
    )
    r.raise_for_status()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channel_id")
    parser.add_argument("--since-minutes", type=int, default=30)
    parser.add_argument(
        "--reply",
        action="store_true",
        help=(
            "Call agent.reply() on the most recent non-bot message and "
            "post the result to the channel. Stop the live bot first."
        ),
    )
    args = parser.parse_args()

    # Defer heavy imports until after .env is loaded so backends.py sees
    # the right BACKEND value.
    from coterie.backends import Agent, ChannelProfileBuilder, ProfileBuilder
    from coterie.channel_ctx import build_channel_summary_block
    from coterie.memory import BotMemory

    token = os.environ["DISCORD_TOKEN"]
    now_utc = _dt.datetime.now(_dt.timezone.utc)
    since_iso = (now_utc - _dt.timedelta(minutes=args.since_minutes)).isoformat()

    memory = BotMemory()

    async with httpx.AsyncClient(timeout=30) as client:
        me = await _get_bot_user(client, token)
        bot_id = me["id"]
        log.info("bot identity: %s (%s)", me.get("username"), bot_id)

        msgs = await _fetch_messages_since(client, token, args.channel_id, since_iso)
        log.info("fetched %d messages since %s", len(msgs), since_iso)

        added = 0
        last_user_msg: dict | None = None
        for m in msgs:
            content = m.get("content") or ""
            if not content:
                continue
            author = m["author"]
            author_id = author["id"]
            is_bot = author.get("bot", False)
            is_self = author_id == bot_id
            if is_bot and not is_self:
                continue
            display = author.get("global_name") or author["username"]
            await memory.add(
                content=content,
                channel_id=args.channel_id,
                author_name=display,
                author_id=author_id,
                message_id=m["id"],
                timestamp=m["timestamp"],
                record_type="agent_reply" if is_self else "message",
            )
            added += 1
            log.info("stored [%s] %s: %s", m["id"], display, content[:80])
            if not is_self:
                last_user_msg = m

        log.info("backfill done: %d records added", added)

        if not args.reply:
            return
        if last_user_msg is None:
            log.warning("--reply set but no non-bot messages found")
            return

        asker = (
            last_user_msg["author"].get("global_name")
            or last_user_msg["author"]["username"]
        )
        query = last_user_msg["content"]
        log.info("agent.reply for asker=%s query=%r", asker, query[:120])

        profile_builder = ProfileBuilder(memory=memory)
        channel_profile_builder = ChannelProfileBuilder(memory=memory)
        try:
            asker_profile = await profile_builder.ensure(
                channel_id=args.channel_id, author=asker
            )
        except Exception:
            log.exception("profile load failed")
            asker_profile = None
        public_channels_env = os.environ.get("PUBLIC_CHANNELS", "")
        public_channels = {
            c.strip() for c in public_channels_env.split(",") if c.strip()
        }
        channel_summary = await build_channel_summary_block(
            current_channel_id=args.channel_id,
            public_channel_ids=public_channels,
            builder=channel_profile_builder,
        )

        agent = Agent(memory=memory)
        text, _files = await agent.reply(
            query=query,
            channel_id=args.channel_id,
            asker=asker,
            asker_profile=asker_profile,
            channel_summary=channel_summary,
        )
        if not text or text.strip() == "<skip>":
            log.warning("agent returned empty/<skip>; not posting")
            return
        await _post_reply(client, token, args.channel_id, text, last_user_msg["id"])
        log.info("posted reply (%d chars) referencing %s", len(text), last_user_msg["id"])


if __name__ == "__main__":
    asyncio.run(main())
