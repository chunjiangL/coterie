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

from backends import BACKEND, Agent, Annotator, Digest, ProactiveClassifier, ProfileBuilder
from memory import BotMemory

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
PROFILES_ENABLED = _bool_env("PROFILES_ENABLED", default=True)
ANNOTATOR_ENABLED = _bool_env("ANNOTATOR_ENABLED", default=True)

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
proactive = ProactiveClassifier()

# Bot user ID, populated once at startup. Used to strip the @-mention prefix
# Slack adds to app_mention events ("<@U01ABC...> hello" → "hello").
_BOT_USER_ID: str | None = None

# Per-channel record of when daily/weekly digest last ran. Same shape as bot.py.
DIGEST_STATE_FILE = Path(__file__).parent / "digest_state_slack.json"


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
            if DIGEST_CHANNELS and now - last_digest >= 600:
                now_la = _dt.datetime.now(LA)
                await _run_daily_if_pending(now_la)
                await _run_weekly_if_pending(now_la)
                last_digest = now
        except Exception:
            log.exception("periodic_ticks loop error")
        await asyncio.sleep(10)


async def _refresh_profiles_all_channels() -> None:
    """Walk every channel the bot has memory for and rebuild stale profiles.
    Slack doesn't expose a member-channels list as easily as Discord; we
    iterate over channels we've seen messages from instead."""
    try:
        resp = await app.client.users_conversations(types="public_channel,private_channel", limit=200)
        channels = resp.get("channels", []) or []
    except Exception:
        log.exception("users_conversations failed; skipping profile refresh")
        return
    for ch in channels:
        cid = ch.get("id")
        if cid:
            await profile_builder.maybe_refresh_channel(cid)


def _strip_mention(text: str) -> str:
    """Remove `<@Uxxx>` mentions of our bot from incoming message text."""
    if not _BOT_USER_ID:
        return text
    return re.sub(rf"<@{re.escape(_BOT_USER_ID)}>", "", text).strip()


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
    if PROFILES_ENABLED:
        try:
            asker_profile = await profile_builder.ensure(channel_id=channel, author=display)
        except Exception:
            log.exception("profile lookup failed")

    try:
        reply_text, generated = await agent.reply(
            query=text,
            channel_id=channel,
            asker=display,
            asker_profile=asker_profile,
            attachments=attachments_payload,
        )
    except Exception:
        log.exception("agent.reply failed")
        await say(text="出错了，看看 bot log 吧。", thread_ts=thread_ts)
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

    reply_text = (reply_text or "").strip() or "(空回复)"
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
    if PROFILES_ENABLED:
        try:
            asker_profile = await profile_builder.ensure(channel_id=channel, author=display)
        except Exception:
            log.exception("proactive: profile lookup failed")

    decision = await proactive.evaluate(
        trigger_msg_text=text,
        asker=display,
        asker_profile=asker_profile,
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
    # @ mentions go to app_mention; if the bot user_id appears in the text
    # OR this is a DM (channel starts with 'D'), skip — handled there.
    if _BOT_USER_ID and f"<@{_BOT_USER_ID}>" in text:
        return
    # Proactive only on regular channels in allow-list.
    if not text:
        return
    team_id = event.get("team")
    if _proactive_channel_enabled(channel, team_id):
        proactive.schedule(
            channel,
            _proactive_dispatch(channel, user, text, event.get("thread_ts")),
        )


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
            from proactive_openai import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        else:
            from proactive import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        log.info(
            "proactive: enabled channels=%s teams=%s (debounce=%ss cooldown=%ss)",
            sorted(PROACTIVE_CHANNELS), sorted(PROACTIVE_SERVERS),
            int(_PDB), int(_PCD),
        )
    if DAILY_DIGEST_CHANNELS:
        log.info("daily digest: channels=%s at 9am LA", sorted(DAILY_DIGEST_CHANNELS))
    if WEEKLY_DIGEST_CHANNELS:
        log.info("weekly digest: channels=%s Mon 9am LA", sorted(WEEKLY_DIGEST_CHANNELS))
    log.info("annotator: %s, profiles: %s",
             "enabled" if ANNOTATOR_ENABLED else "disabled",
             "enabled" if PROFILES_ENABLED else "disabled")

    asyncio.create_task(_periodic_ticks())
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(_main())
