"""Backfill missed Discord messages from a channel into Mem0.

Use when the bot was down during a window. Discord's gateway doesn't
replay missed events on reconnect, so REST fetch is the only path.

Optionally calls agent.reply() on the most recent non-bot message and
posts the reply back to the channel via REST. The live bot should be
stopped before running with --reply, both to avoid Mem0 write contention
and to prevent two bot instances racing to respond.

Two modes:
- Window mode (default): fetches messages since --since-minutes ago,
  writes new ones to Mem0, optionally replies to the most recent
  non-bot message in the window.
- Targeted mode (--target-message-id): re-runs agent.reply against a
  specific past message, including any image/PDF attachments. Useful
  when the live bot processed a message under a bug and you want it
  to re-answer correctly. Skips Mem0 write (record already exists).

Usage:
    python scripts/backfill_recent.py <channel_id> --since-minutes 30 --reply
    python scripts/backfill_recent.py <channel_id> --reply \\
        --target-message-id <msg_id>
"""

import argparse
import asyncio
import base64
import datetime as _dt
import logging
import os
from typing import Any

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


async def _fetch_single_message(
    client: httpx.AsyncClient, token: str, channel_id: str, message_id: str
) -> dict:
    r = await client.get(
        f"{API}/channels/{channel_id}/messages/{message_id}",
        headers={"Authorization": f"Bot {token}"},
    )
    r.raise_for_status()
    return r.json()


MAX_TEXT_BYTES = 200 * 1024

_TEXT_EXTS = (
    ".txt", ".md", ".json", ".csv", ".log", ".py", ".ts",
    ".js", ".tsx", ".jsx", ".yaml", ".yml", ".toml", ".sh",
)


async def _inline_text_attachments(
    client: httpx.AsyncClient, message: dict
) -> str:
    """Pull text attachments off a Discord message and format them as
    inline-readable blocks. Same shape as the live adapter helper."""
    chunks: list[str] = []
    for att in message.get("attachments") or []:
        ctype = (att.get("content_type") or "").lower()
        filename = att.get("filename") or "file"
        size = int(att.get("size") or 0)
        url = att.get("url")
        if not url:
            continue
        is_text = ctype.startswith("text/") or filename.lower().endswith(_TEXT_EXTS)
        if not is_text:
            continue
        try:
            r = await client.get(url, timeout=60)
            r.raise_for_status()
            data = r.content
        except Exception:
            log.exception("text attachment download failed: %s", filename)
            continue
        truncated = len(data) > MAX_TEXT_BYTES
        text = data[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
        suffix = f"\n[...truncated, original {size} bytes]" if truncated else ""
        chunks.append(
            f"[Attached file: {filename}]\n```\n{text}{suffix}\n```"
        )
        log.info(
            "inlined text attachment %s (%d bytes%s)",
            filename, len(data), " truncated" if truncated else "",
        )
    return "\n\n".join(chunks)


async def _build_attachment_blocks(
    client: httpx.AsyncClient, message: dict
) -> list[dict[str, Any]]:
    """Download a Discord message's attachments and convert to backend-
    aware multimodal blocks. Mirrors discord_adapter._build_attachment_blocks
    but works off REST payloads instead of discord.py models. Text files
    are handled separately via _inline_text_attachments."""
    from coterie.backends import BACKEND  # late import

    blocks: list[dict[str, Any]] = []
    for att in message.get("attachments") or []:
        ctype = (att.get("content_type") or "").lower()
        filename = att.get("filename") or "file"
        url = att.get("url")
        if not url:
            continue
        is_pdf = ctype.startswith("application/pdf") or filename.lower().endswith(".pdf")
        is_image = ctype.startswith("image/")
        if not (is_pdf or is_image):
            continue
        try:
            r = await client.get(url, timeout=60)
            r.raise_for_status()
            data = r.content
        except Exception:
            log.exception("attachment download failed: %s", filename)
            continue
        b64 = base64.standard_b64encode(data).decode("ascii")
        media_type = ctype.split(";")[0].strip() or ("application/pdf" if is_pdf else "image/png")
        if BACKEND == "anthropic":
            if is_pdf:
                blocks.append({
                    "type": "document",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                    "title": filename,
                })
            else:
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
        else:
            if is_pdf:
                blocks.append({
                    "type": "input_file",
                    "filename": filename,
                    "file_data": f"data:{media_type};base64,{b64}",
                })
            else:
                blocks.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{b64}",
                })
        log.info(
            "attached %s (%s, %d bytes) as backend-%s block",
            filename, media_type, len(data), BACKEND,
        )
    return blocks


DISCORD_REPLY_CHARS = 1900  # same chunk size the live bot uses


async def _post_reply(
    client: httpx.AsyncClient,
    token: str,
    channel_id: str,
    content: str,
    reference_id: str,
) -> None:
    """Post a reply, chunking at 1900 chars. Discord caps at 2000."""
    headers = {"Authorization": f"Bot {token}"}
    chunks = [
        content[i : i + DISCORD_REPLY_CHARS]
        for i in range(0, len(content), DISCORD_REPLY_CHARS)
    ] or [""]
    for idx, chunk in enumerate(chunks):
        body: dict[str, Any] = {
            "content": chunk,
            "allowed_mentions": {"replied_user": idx == 0},
        }
        if idx == 0:
            body["message_reference"] = {
                "message_id": reference_id,
                "fail_if_not_exists": False,
            }
        r = await client.post(
            f"{API}/channels/{channel_id}/messages",
            json=body,
            headers=headers,
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
            "Call agent.reply() on the most recent non-bot message (or on "
            "--target-message-id) and post the result to the channel. "
            "Stop the live bot first."
        ),
    )
    parser.add_argument(
        "--target-message-id",
        default=None,
        help=(
            "Re-run agent.reply against this specific past message, "
            "including its image/PDF attachments. Skips Mem0 write "
            "(record already exists). Use when the live bot answered "
            "under a bug and you want it to re-answer."
        ),
    )
    args = parser.parse_args()

    # Defer heavy imports until after .env is loaded so backends.py sees
    # the right BACKEND value.
    from coterie.backends import Agent, ChannelProfileBuilder, ProfileBuilder
    from coterie.channel_ctx import build_channel_summary_block
    from coterie.memory import BotMemory

    token = os.environ["DISCORD_TOKEN"]
    memory = BotMemory()

    async with httpx.AsyncClient(timeout=30) as client:
        me = await _get_bot_user(client, token)
        bot_id = me["id"]
        log.info("bot identity: %s (%s)", me.get("username"), bot_id)

        last_user_msg: dict | None = None

        if args.target_message_id:
            log.info("targeted mode: fetching message %s", args.target_message_id)
            last_user_msg = await _fetch_single_message(
                client, token, args.channel_id, args.target_message_id
            )
            if last_user_msg["author"]["id"] == bot_id:
                log.error("target message is the bot's own; aborting")
                return
        else:
            now_utc = _dt.datetime.now(_dt.timezone.utc)
            since_iso = (
                now_utc - _dt.timedelta(minutes=args.since_minutes)
            ).isoformat()
            msgs = await _fetch_messages_since(
                client, token, args.channel_id, since_iso
            )
            log.info("fetched %d messages since %s", len(msgs), since_iso)
            added = 0
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

        # Strip the bot @-mention from the query, matching live on_message.
        raw_query = last_user_msg["content"] or ""
        query = (
            raw_query.replace(f"<@{bot_id}>", "")
            .replace(f"<@!{bot_id}>", "")
            .strip()
        )
        attachment_blocks = await _build_attachment_blocks(client, last_user_msg)
        text_inlined = await _inline_text_attachments(client, last_user_msg)
        if not query and (attachment_blocks or text_inlined):
            query = "Please take a look at this attachment."
        if text_inlined:
            query = f"{query}\n\n{text_inlined}"
        asker = (
            last_user_msg["author"].get("global_name")
            or last_user_msg["author"]["username"]
        )
        log.info(
            "agent.reply for asker=%s attachments=%d query=%r",
            asker, len(attachment_blocks), query[:120],
        )

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
            attachments=attachment_blocks or None,
        )
        if not text or text.strip() == "<skip>":
            log.warning("agent returned empty/<skip>; not posting")
            return
        await _post_reply(
            client, token, args.channel_id, text, last_user_msg["id"]
        )
        log.info(
            "posted reply (%d chars) referencing %s",
            len(text), last_user_msg["id"],
        )


if __name__ == "__main__":
    asyncio.run(main())
