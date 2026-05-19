"""Slack adapter — sibling of bot.py.

Subscribes to ``app_mention`` (when the bot is @-tagged) and ``message`` (every
message in channels the bot is in) via slack_bolt's async Socket Mode handler.
No public webhook needed: the bot opens an outbound WebSocket to Slack.

Required env (in addition to BACKEND/API keys):
    SLACK_BOT_TOKEN=xoxb-...        # bot-user OAuth token
    SLACK_APP_TOKEN=xapp-...        # app-level token with connections:write

Optional, mirrored from bot.py:
    DIGEST_CHANNELS / DAILY_DIGEST_CHANNELS / WEEKLY_DIGEST_CHANNELS
    PROACTIVE_CHANNELS / PROACTIVE_SERVERS (Slack team IDs here)
    PROFILES_ENABLED / ANNOTATOR_ENABLED

Channel IDs are Slack ``C01ABC...`` strings. Workspace (team) IDs are
``T01ABC...``. Get them by right-clicking a channel → "Open channel details"
→ scrolling to the bottom, or via the Slack admin URL.

Run:
    PLATFORM=slack python slack_bot.py
"""

import asyncio
import base64
import datetime as _dt
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

load_dotenv()

# Force PLATFORM=slack so the shared config/prompts emit Slack-flavored output.
os.environ["PLATFORM"] = "slack"

from slack_bolt.app.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from coterie.backends import (
    BACKEND,
    Agent,
    Annotator,
    ChannelProfileBuilder,
    Digest,
    ProactiveClassifier,
    ProfileBuilder,
)
from coterie.channel_ctx import build_channel_summary_block
from coterie.memory import BotMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dc-agent.slack")

# Same chunk size; Slack's hard limit is 40k chars/message, but keeping it
# small avoids the bot dumping walls of text.
MAX_PDF_BYTES = 20 * 1024 * 1024
SLACK_REPLY_CHARS = 2800


def _csv_set(env_var: str) -> set[str]:
    return {x.strip() for x in os.environ.get(env_var, "").split(",") if x.strip()}


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


DAILY_DIGEST_CHANNELS = _csv_set("DAILY_DIGEST_CHANNELS") or _csv_set("DIGEST_CHANNELS")
WEEKLY_DIGEST_CHANNELS = _csv_set("WEEKLY_DIGEST_CHANNELS") or _csv_set("DIGEST_CHANNELS")
DIGEST_CHANNELS = DAILY_DIGEST_CHANNELS | WEEKLY_DIGEST_CHANNELS
PROACTIVE_CHANNELS = _csv_set("PROACTIVE_CHANNELS")
PROACTIVE_SERVERS = _csv_set("PROACTIVE_SERVERS")  # Slack: team_id
PUBLIC_CHANNELS = _csv_set("PUBLIC_CHANNELS")
PROFILES_ENABLED = _bool_env("PROFILES_ENABLED", default=True)
ANNOTATOR_ENABLED = _bool_env("ANNOTATOR_ENABLED", default=True)
DIGEST_ENABLED = _bool_env("DIGEST_ENABLED", default=True)

LA = ZoneInfo("America/Los_Angeles")

# Slack tokens.
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN:
    raise SystemExit(
        "SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set "
        "(copy .env.example to .env and fill them in)"
    )

app = AsyncApp(token=SLACK_BOT_TOKEN)
memory = BotMemory()
agent = Agent(memory=memory)
annotator = Annotator(memory=memory)
digest = Digest(memory=memory)
profile_builder = ProfileBuilder(memory=memory)
channel_profile_builder = ChannelProfileBuilder(memory=memory)
proactive = ProactiveClassifier()

# Bot user ID, populated once at startup. Used to strip the @-mention prefix
# Slack adds to app_mention events ("<@U01ABC...> hello" → "hello").
_BOT_USER_ID: str | None = None

# Spam guard for @-mention path. Same shape as Discord adapter; see comments
# there for rationale.
MENTION_LIMIT = 3
MENTION_WINDOW_SEC = 30.0
MENTION_LOCKOUT_SEC = 60.0
_mention_history: dict[tuple[str, str], list[float]] = {}
_mention_lockout: dict[tuple[str, str], float] = {}
_mention_warned: dict[tuple[str, str], bool] = {}


def _mention_should_block(channel_id: str, user_id: str) -> tuple[bool, bool]:
    import time as _time
    now = _time.time()
    key = (channel_id, user_id)
    lockout_until = _mention_lockout.get(key, 0.0)
    if now < lockout_until:
        if not _mention_warned.get(key):
            _mention_warned[key] = True
            return True, True
        return True, False
    elif lockout_until > 0:
        _mention_lockout.pop(key, None)
        _mention_warned.pop(key, None)
    history = _mention_history.setdefault(key, [])
    cutoff = now - MENTION_WINDOW_SEC
    history[:] = [t for t in history if t > cutoff]
    history.append(now)
    if len(history) > MENTION_LIMIT:
        _mention_lockout[key] = now + MENTION_LOCKOUT_SEC
        _mention_warned[key] = True
        log.warning(
            "mention spam lockout: channel=%s user=%s (%d hits in %.0fs)",
            channel_id, user_id, len(history), MENTION_WINDOW_SEC,
        )
        return True, True
    return False, False

# Per-channel record of when daily/weekly digest last ran. Same shape as bot.py.
DIGEST_STATE_FILE = Path(
    os.environ.get("DIGEST_STATE_FILE_SLACK")
    or Path.cwd() / "digest_state_slack.json"
)


def _load_digest_state() -> dict[str, dict[str, str]]:
    if not DIGEST_STATE_FILE.exists():
        return {}
    try:
        return json.loads(DIGEST_STATE_FILE.read_text())
    except Exception:
        log.exception("failed to load digest state; starting fresh")
        return {}


def _save_digest_state(state: dict[str, dict[str, str]]) -> None:
    try:
        DIGEST_STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        log.exception("failed to persist digest state")


def _today_9am(now_la: _dt.datetime) -> _dt.datetime:
    return now_la.replace(hour=9, minute=0, second=0, microsecond=0)


def _this_monday_9am(now_la: _dt.datetime) -> _dt.datetime:
    monday = now_la - _dt.timedelta(days=now_la.weekday())
    return monday.replace(hour=9, minute=0, second=0, microsecond=0)


async def _push_to_channel(channel_id: str, text: str) -> None:
    for i in range(0, len(text), SLACK_REPLY_CHARS):
        await app.client.chat_postMessage(
            channel=channel_id, text=text[i : i + SLACK_REPLY_CHARS]
        )


async def _run_daily_if_pending(now_la: _dt.datetime) -> None:
    target = _today_9am(now_la)
    if now_la < target:
        return
    state = _load_digest_state()
    changed = False
    for cid in DAILY_DIGEST_CHANNELS:
        ch = state.setdefault(cid, {})
        last_iso = ch.get("daily_last_run")
        if last_iso:
            try:
                if _dt.datetime.fromisoformat(last_iso) >= target:
                    continue
            except ValueError:
                pass
        log.info("daily digest: running for channel=%s (last_run=%s)", cid, last_iso)
        try:
            text = await digest.daily(cid)
        except Exception:
            log.exception("daily digest failed for channel %s", cid)
            continue
        if text:
            await _push_to_channel(cid, text)
        ch["daily_last_run"] = now_la.isoformat()
        changed = True
    if changed:
        _save_digest_state(state)


async def _run_weekly_if_pending(now_la: _dt.datetime) -> None:
    monday = _this_monday_9am(now_la)
    if now_la < monday:
        return
    state = _load_digest_state()
    changed = False
    for cid in WEEKLY_DIGEST_CHANNELS:
        ch = state.setdefault(cid, {})
        last_iso = ch.get("weekly_last_run")
        if last_iso:
            try:
                if _dt.datetime.fromisoformat(last_iso) >= monday:
                    continue
            except ValueError:
                pass
        log.info("weekly digest: running for channel=%s (last_run=%s)", cid, last_iso)
        try:
            text = await digest.weekly(cid)
        except Exception:
            log.exception("weekly digest failed for channel %s", cid)
            continue
        if text:
            await _push_to_channel(cid, text)
        ch["weekly_last_run"] = now_la.isoformat()
        changed = True
    if changed:
        _save_digest_state(state)


async def _periodic_ticks() -> None:
    """Background loop running annotator + profile refresh + digest catch-up.
    slack_bolt doesn't have discord.py's ``tasks.loop``; roll our own."""
    last_annotator = 0.0
    last_profile = 0.0
    last_digest = 0.0
    while True:
        try:
            now = asyncio.get_event_loop().time()
            if ANNOTATOR_ENABLED and now - last_annotator >= 30:
                await annotator.maybe_flush_all()
                last_annotator = now
            if PROFILES_ENABLED and now - last_profile >= 600:
                await _refresh_profiles_all_channels()
                last_profile = now
            if DIGEST_ENABLED and DIGEST_CHANNELS and now - last_digest >= 600:
                now_la = _dt.datetime.now(LA)
                await _run_daily_if_pending(now_la)
                await _run_weekly_if_pending(now_la)
                last_digest = now
        except Exception:
            log.exception("periodic_ticks loop error")
        await asyncio.sleep(10)


async def _refresh_profiles_all_channels() -> None:
    """Walk every channel the bot has memory for and rebuild stale profiles
    (both user-level and channel-level). Slack doesn't expose a member-
    channels list as easily as Discord; we iterate over channels the bot
    is in via users_conversations."""
    try:
        resp = await app.client.users_conversations(types="public_channel,private_channel", limit=200)
        channels = resp.get("channels", []) or []
    except Exception:
        log.exception("users_conversations failed; skipping profile refresh")
        return
    for ch in channels:
        cid = ch.get("id")
        if not cid:
            continue
        await profile_builder.maybe_refresh_channel(cid)
        if cid in PUBLIC_CHANNELS:
            # Force first-build so the public summary exists for other
            # channels to read, even before anyone @s the bot here.
            await channel_profile_builder.ensure(channel_id=cid)
        await channel_profile_builder.maybe_refresh_channel(cid)


def _strip_mention(text: str) -> str:
    """Remove `<@Uxxx>` mentions of our bot from incoming message text."""
    if not _BOT_USER_ID:
        return text
    return re.sub(rf"<@{re.escape(_BOT_USER_ID)}>", "", text).strip()


# Cache of thread_ts → bool (parent message author == bot). Slack doesn't
# expose this on the message event, so we fetch via conversations.replies
# once per thread and cache for the bot's lifetime. Thread root authorship
# can't change, so cache is safe.
_thread_root_is_bot_cache: dict[str, bool] = {}

# Slack channel-id → `#name`. Populated lazily; channel renames are rare
# so we don't bother invalidating.
_channel_name_cache: dict[str, str] = {}


async def _channel_label(cid: str) -> str:
    cached = _channel_name_cache.get(cid)
    if cached is not None:
        return cached
    try:
        resp = await app.client.conversations_info(channel=cid)
        name = (resp.get("channel") or {}).get("name") or ""
    except Exception:
        log.exception("conversations.info failed for %s", cid)
        name = ""
    label = f"#{name}" if name else cid
    _channel_name_cache[cid] = label
    return label


async def _thread_root_is_bot(channel: str, thread_ts: str) -> bool:
    """True iff the parent message of this thread was authored by us.

    Slack threads have no "reply-to-bot" event; this is how we detect when
    a user clicked Reply on a bot message instead of typing @bot."""
    cached = _thread_root_is_bot_cache.get(thread_ts)
    if cached is not None:
        return cached
    if not _BOT_USER_ID:
        return False
    try:
        resp = await app.client.conversations_replies(
            channel=channel, ts=thread_ts, limit=1, inclusive=True,
        )
        messages = resp.get("messages") or []
        if not messages:
            _thread_root_is_bot_cache[thread_ts] = False
            return False
        parent = messages[0]
        is_bot = parent.get("user") == _BOT_USER_ID or parent.get("bot_id") is not None
        _thread_root_is_bot_cache[thread_ts] = is_bot
        return is_bot
    except Exception:
        log.exception("conversations.replies failed for thread %s", thread_ts)
        return False


def _proactive_channel_enabled(channel_id: str, team_id: str | None) -> bool:
    if channel_id in PROACTIVE_CHANNELS:
        return True
    if team_id and team_id in PROACTIVE_SERVERS:
        return True
    return False


async def _download_files(file_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Download PDF attachments. Slack's file URLs require the bot token in
    an Authorization header — we use httpx directly since slack-bolt doesn't
    expose a download helper."""
    blocks: list[dict[str, Any]] = []
    for f in file_blocks:
        if not (f.get("mimetype") or "").startswith("application/pdf"):
            continue
        size = f.get("size", 0)
        if size > MAX_PDF_BYTES:
            log.warning("skipping %s (%d bytes, too large)", f.get("name"), size)
            continue
        url = f.get("url_private_download") or f.get("url_private")
        if not url:
            continue
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(url, headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"})
        if r.status_code != 200:
            log.warning("file download http=%s for %s", r.status_code, url)
            continue
        blocks.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(r.content).decode("ascii"),
            },
            "title": f.get("name") or "attachment.pdf",
        })
    return blocks


async def _handle_query(
    *,
    channel: str,
    user: str,
    text: str,
    files: list[dict[str, Any]],
    thread_ts: str | None,
    say: Any,
) -> None:
    """Run the @-mention path: load profile, call agent, reply (in thread)."""
    info = None
    try:
        info = await app.client.users_info(user=user)
    except Exception:
        log.exception("users_info failed for %s", user)
    display = (
        ((info or {}).get("user") or {}).get("profile") or {}
    ).get("display_name") or user

    attachments_payload = await _download_files(files)

    asker_profile: str | None = None
    channel_summary: str | None = None
    if PROFILES_ENABLED:
        try:
            asker_profile = await profile_builder.ensure(channel_id=channel, author=display)
        except Exception:
            log.exception("profile lookup failed")
        # Skip channel summary for IM/DM channels — nothing to summarize.
        if not channel.startswith("D"):
            channel_summary = await build_channel_summary_block(
                current_channel_id=channel,
                public_channel_ids=PUBLIC_CHANNELS,
                builder=channel_profile_builder,
                label_for_channel=_channel_label,
            )

    try:
        reply_text, generated = await agent.reply(
            query=text,
            channel_id=channel,
            asker=display,
            asker_profile=asker_profile,
            channel_summary=channel_summary,
            attachments=attachments_payload,
        )
    except Exception:
        log.exception("agent.reply failed")
        await say(text="Something went wrong — check the bot log.", thread_ts=thread_ts)
        return

    # Chunked send into the same thread; first chunk carries any code_interpreter files.
    if generated:
        for fname, data in generated:
            try:
                await app.client.files_upload_v2(
                    channel=channel,
                    file=data,
                    filename=fname,
                    thread_ts=thread_ts,
                )
            except Exception:
                log.exception("files_upload_v2 failed for %s", fname)

    reply_text = (reply_text or "").strip() or "(empty reply)"
    for i in range(0, len(reply_text), SLACK_REPLY_CHARS):
        await say(text=reply_text[i : i + SLACK_REPLY_CHARS], thread_ts=thread_ts)


async def _store_message(
    *,
    channel: str,
    user: str,
    text: str,
    ts: str,
    files: list[dict[str, Any]],
    is_self: bool,
) -> tuple[str, str]:
    """Persist incoming message into Mem0 + annotator buffer. Returns (display_name, content)."""
    info = None
    try:
        info = await app.client.users_info(user=user)
    except Exception:
        pass
    display = (
        ((info or {}).get("user") or {}).get("profile") or {}
    ).get("display_name") or user

    attachment_summary = ""
    if files:
        names = ", ".join(f.get("name") or "?" for f in files)
        attachment_summary = f" [attachments: {names}]"
    content = (text or "") + attachment_summary
    if not content:
        return display, content

    timestamp_iso = _dt.datetime.fromtimestamp(float(ts), tz=_dt.timezone.utc).isoformat()
    await memory.add(
        content=content,
        channel_id=channel,
        author_name=display,
        author_id=user,
        message_id=ts,
        timestamp=timestamp_iso,
        record_type="agent_reply" if is_self else "message",
    )
    annotator.buffer_push(
        channel_id=channel,
        message_id=ts,
        author=display,
        content=content,
        timestamp_iso=timestamp_iso,
        is_bot_reply=is_self,
    )
    return display, content


async def _proactive_dispatch(channel: str, user: str, text: str, thread_ts: str | None) -> None:
    if proactive.cooldown_active(channel, asker=user):
        return
    info = None
    try:
        info = await app.client.users_info(user=user)
    except Exception:
        pass
    display = (
        ((info or {}).get("user") or {}).get("profile") or {}
    ).get("display_name") or user

    recent_msgs = annotator.recent_n(channel, n=10)
    asker_profile: str | None = None
    channel_summary: str | None = None
    if PROFILES_ENABLED:
        try:
            asker_profile = await profile_builder.ensure(channel_id=channel, author=display)
        except Exception:
            log.exception("proactive: profile lookup failed")
        channel_summary = await build_channel_summary_block(
            current_channel_id=channel,
            public_channel_ids=PUBLIC_CHANNELS,
            builder=channel_profile_builder,
            label_for_channel=_channel_label,
        )

    decision = await proactive.evaluate(
        trigger_msg_text=text,
        asker=display,
        asker_profile=asker_profile,
        channel_summary=channel_summary,
        recent_msgs=recent_msgs,
    )
    if not decision or not decision.get("fire"):
        return

    prior_relevant: list[Any] = []
    search_query = (decision.get("search_query") or "").strip()
    if search_query:
        try:
            prior_relevant = await memory.search(
                query=search_query,
                channel_id=channel,
                record_type="message",
                limit=5,
            )
        except Exception:
            log.exception("proactive: Mem0 search failed")

    try:
        reply_text, _ = await agent.reply(
            query=text,
            channel_id=channel,
            asker=display,
            asker_profile=asker_profile,
            channel_summary=channel_summary,
            mode="proactive",
            recent_context=recent_msgs,
            prior_relevant=prior_relevant,
            trigger_reason=decision.get("reason"),
        )
    except Exception:
        log.exception("proactive: agent.reply failed")
        return

    reply_text = (reply_text or "").strip()
    if not reply_text or reply_text.startswith("<skip>"):
        return

    final = f"<@{user}> {reply_text}"
    try:
        for i in range(0, len(final), SLACK_REPLY_CHARS):
            await app.client.chat_postMessage(
                channel=channel,
                text=final[i : i + SLACK_REPLY_CHARS],
                thread_ts=thread_ts,
            )
    except Exception:
        log.exception("proactive: send failed")
        return
    proactive.mark_fired(channel, asker=display)
    log.info("proactive: fired in channel=%s asker=%s", channel, display)


# ─── Event handlers ─────────────────────────────────────────────────


@app.event("app_mention")
async def on_mention(event: dict[str, Any], say: Any) -> None:
    """Bot is @-tagged — produce a reply."""
    channel = event["channel"]
    user = event["user"]
    text = _strip_mention(event.get("text", "") or "")
    thread_ts = event.get("thread_ts") or event.get("ts")
    files = event.get("files") or []
    await _store_message(channel=channel, user=user, text=event.get("text", ""),
                         ts=event["ts"], files=files, is_self=False)
    blocked, warn = _mention_should_block(channel, user)
    if blocked:
        if warn:
            await say(
                text=(
                    "Slow down a bit — too many @s in a row. I'll be back in a "
                    "minute. If you need something urgent, edit your last "
                    "message instead of sending new ones."
                ),
                thread_ts=thread_ts,
            )
        return
    await _handle_query(channel=channel, user=user, text=text, files=files,
                        thread_ts=thread_ts, say=say)


@app.event("message")
async def on_message(event: dict[str, Any]) -> None:
    """Catch-all message handler — stores to memory + maybe proactive."""
    # Ignore message_changed / message_deleted / channel_join etc.
    subtype = event.get("subtype")
    if subtype and subtype not in ("file_share", "thread_broadcast"):
        return
    # Don't re-handle our own messages via this path; app_mention already
    # covered direct invocations. But DO persist them so the agent_reply
    # record_type buffer stays complete.
    user = event.get("user")
    if not user:
        return
    is_self = user == _BOT_USER_ID
    channel = event["channel"]
    text = event.get("text", "") or ""
    files = event.get("files") or []
    ts = event["ts"]
    await _store_message(channel=channel, user=user, text=text, ts=ts,
                         files=files, is_self=is_self)
    if is_self:
        return
    # Explicit @ mentions go to app_mention — skip the duplicate here.
    if _BOT_USER_ID and f"<@{_BOT_USER_ID}>" in text:
        return
    if not text:
        return

    thread_ts = event.get("thread_ts")
    # Discord-style "Reply" → in Slack that's "reply in thread". If the
    # parent of the thread is a bot message, the user is addressing the
    # bot even without re-@ing. Treat as a mention.
    if thread_ts and thread_ts != ts:
        if await _thread_root_is_bot(channel, thread_ts):
            blocked, warn = _mention_should_block(channel, user)
            if blocked:
                if warn:
                    await app.client.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=(
                            "Slow down a bit — too many @s in a row. I'll "
                            "be back in a minute. If you need something "
                            "urgent, edit your last message instead of "
                            "sending new ones."
                        ),
                    )
                return
            files = event.get("files") or []
            async def say(text: str, thread_ts: str | None = None) -> None:
                await app.client.chat_postMessage(
                    channel=channel, text=text, thread_ts=thread_ts,
                )
            await _handle_query(
                channel=channel, user=user, text=text, files=files,
                thread_ts=thread_ts, say=say,
            )
            return

    team_id = event.get("team")
    if _proactive_channel_enabled(channel, team_id):
        proactive.schedule(
            channel,
            _proactive_dispatch(channel, user, text, thread_ts),
        )


@app.event("reaction_added")
async def on_reaction_added(event: dict[str, Any]) -> None:
    """Record reactions on bot messages as implicit feedback.

    Filters: only `item.type == 'message'`; skip self-reactions; require
    `item_user` (the original poster) to be the bot.
    """
    if event.get("item", {}).get("type") != "message":
        return
    reactor = event.get("user")
    item_user = event.get("item_user")
    if not reactor or reactor == _BOT_USER_ID:
        return
    if not _BOT_USER_ID or item_user != _BOT_USER_ID:
        return

    item = event["item"]
    channel = item["channel"]
    ts = item["ts"]

    excerpt = ""
    try:
        resp = await app.client.conversations_history(
            channel=channel, latest=ts, limit=1, inclusive=True,
        )
        messages = resp.get("messages") or []
        if messages:
            excerpt = (messages[0].get("text") or "")[:200]
    except Exception:
        log.exception("conversations.history failed during reaction_added")

    info = None
    try:
        info = await app.client.users_info(user=reactor)
    except Exception:
        pass
    display = (
        ((info or {}).get("user") or {}).get("profile") or {}
    ).get("display_name") or reactor

    now_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    try:
        await memory.add_reaction(
            channel_id=channel,
            reactor_name=display,
            reactor_id=reactor,
            emoji=f":{event.get('reaction', '')}:",
            target_message_id=ts,
            target_excerpt=excerpt,
            timestamp=now_iso,
        )
    except Exception:
        log.exception("memory.add_reaction failed")


@app.event("file_shared")
async def on_file_shared(event: dict[str, Any]) -> None:
    # No-op — file_share comes through message event already.
    pass


# ─── Main ───────────────────────────────────────────────────────────


async def _main() -> None:
    global _BOT_USER_ID
    auth = await app.client.auth_test()
    _BOT_USER_ID = auth.get("user_id")
    log.info("slack_bot: logged in user_id=%s team=%s (backend=%s)",
             _BOT_USER_ID, auth.get("team"), BACKEND)
    if PROACTIVE_CHANNELS or PROACTIVE_SERVERS:
        if BACKEND == "openai":
            from coterie.gpt.proactive import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        else:
            from coterie.claude.proactive import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        log.info(
            "proactive: enabled channels=%s teams=%s (debounce=%ss cooldown=%ss)",
            sorted(PROACTIVE_CHANNELS), sorted(PROACTIVE_SERVERS),
            int(_PDB), int(_PCD),
        )
    if not DIGEST_ENABLED:
        log.info("digest: disabled by DIGEST_ENABLED=false")
    else:
        if DAILY_DIGEST_CHANNELS:
            log.info("daily digest: channels=%s at 9am LA", sorted(DAILY_DIGEST_CHANNELS))
        if WEEKLY_DIGEST_CHANNELS:
            log.info("weekly digest: channels=%s Mon 9am LA", sorted(WEEKLY_DIGEST_CHANNELS))
    log.info("annotator: %s, profiles: %s",
             "enabled" if ANNOTATOR_ENABLED else "disabled",
             "enabled" if PROFILES_ENABLED else "disabled")
    if PUBLIC_CHANNELS:
        log.info(
            "public channels (cross-channel summary source): %s",
            sorted(PUBLIC_CHANNELS),
        )
    else:
        log.info("public channels: none (cross-channel summary disabled)")

    asyncio.create_task(_periodic_ticks())
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


def main() -> None:
    """Entry point for the ``coterie-slack`` console script."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
