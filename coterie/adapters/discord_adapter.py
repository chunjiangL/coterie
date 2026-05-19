"""Discord bot entrypoint."""

import base64
import datetime as _dt
import io
import json
import logging
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import discord
from discord.ext import tasks
from dotenv import load_dotenv

# Load .env BEFORE importing backends — backends.py reads BACKEND at import
# time, so the env var must be present before that line. Same logic for
# memory.py (Mem0's internal LLM picks anthropic vs openai by env).
load_dotenv()

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
log = logging.getLogger("dc-agent")

# Inline-base64 PDFs above this size get skipped — they'd bloat the request
# body. (Future: upload to Files API and pass file_id instead.)
MAX_PDF_BYTES = 20 * 1024 * 1024  # 20 MB
DISCORD_REPLY_CHARS = 1900

def _csv_set(env_var: str) -> set[str]:
    return {x.strip() for x in os.environ.get(env_var, "").split(",") if x.strip()}


def _bool_env(name: str, default: bool = True) -> bool:
    """Parse a boolean env var. true/1/yes → True, anything else → False.
    Missing var → default."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Daily-digest channels. Falls back to DIGEST_CHANNELS for backward compat
# when DAILY_DIGEST_CHANNELS isn't explicitly set.
DAILY_DIGEST_CHANNELS: set[str] = (
    _csv_set("DAILY_DIGEST_CHANNELS") or _csv_set("DIGEST_CHANNELS")
)
# Weekly-digest channels. Falls back to DIGEST_CHANNELS too. Set this to a
# different list than DAILY_DIGEST_CHANNELS to send weekly to a subset (or
# superset) of where daily goes.
WEEKLY_DIGEST_CHANNELS: set[str] = (
    _csv_set("WEEKLY_DIGEST_CHANNELS") or _csv_set("DIGEST_CHANNELS")
)
# Union of both — used to decide whether to start digest ticks at all.
DIGEST_CHANNELS: set[str] = DAILY_DIGEST_CHANNELS | WEEKLY_DIGEST_CHANNELS

# Channels where the bot proactively jumps into research discussions.
# Independent from DIGEST_CHANNELS — a channel can opt into one, both, or neither.
PROACTIVE_CHANNELS: set[str] = _csv_set("PROACTIVE_CHANNELS")
# Servers (Discord guilds) where the bot proactively participates in EVERY
# text channel. PROACTIVE_CHANNELS works as an explicit override.
PROACTIVE_SERVERS: set[str] = _csv_set("PROACTIVE_SERVERS")

# Channels whose channel_profile is treated as "public knowledge" — every
# other channel's reply path loads these summaries alongside its own. Empty
# (default) = no cross-channel context. Never names private channels here.
PUBLIC_CHANNELS: set[str] = _csv_set("PUBLIC_CHANNELS")

# Feature toggles. Default ON (legacy behavior).
# PROFILES: build + auto-refresh per-user profile from message history.
# Off → @ replies see no asker_profile, proactive classifier has no
# `Bot interaction preference` signal, profile_refresh_tick doesn't start.
PROFILES_ENABLED: bool = _bool_env("PROFILES_ENABLED", default=True)
# ANNOTATOR: sliding-window LLM-generated English annotations of messages.
# Off → annotator_tick doesn't start (no LLM calls), but the in-memory
# buffer keeps filling so proactive's recent_n() still works. Trade-off:
# semantic search loses the annotation-layer surface (~half the recall lift).
ANNOTATOR_ENABLED: bool = _bool_env("ANNOTATOR_ENABLED", default=True)
# DIGEST: daily/weekly summaries. Off → no digest ticks start at all,
# regardless of DAILY_DIGEST_CHANNELS / WEEKLY_DIGEST_CHANNELS contents.
# When on, channel-list env vars still control which channels receive
# which digest (and empty lists also disable individually).
DIGEST_ENABLED: bool = _bool_env("DIGEST_ENABLED", default=True)

LA = ZoneInfo("America/Los_Angeles")
DIGEST_TIME = _dt.time(hour=9, minute=0, tzinfo=LA)

# Spam guard for @-mention path. If a single user @-s the bot more than
# MENTION_LIMIT times within MENTION_WINDOW_SEC, lock them out for
# MENTION_LOCKOUT_SEC and reply once with a firm note. Without this,
# anyone can drain the API key by spamming @ + an expensive question.
MENTION_LIMIT = 3
MENTION_WINDOW_SEC = 30.0
MENTION_LOCKOUT_SEC = 60.0

# (channel_id, user_id) → list[recent timestamps]. Trimmed per-call.
_mention_history: dict[tuple[str, str], list[float]] = {}
# (channel_id, user_id) → unix when lockout ends; 0 = no lockout.
_mention_lockout: dict[tuple[str, str], float] = {}
# (channel_id, user_id) → True if we've already sent the warning this
# lockout cycle. Reset when lockout expires.
_mention_warned: dict[tuple[str, str], bool] = {}


def _mention_should_block(channel_id: str, user_id: str) -> tuple[bool, bool]:
    """Returns (block_now, should_send_warning_message).

    block_now=True means: silently drop the @ (or send warning once if
    warn flag is True). block_now=False means: process normally.
    """
    import time as _time
    now = _time.time()
    key = (channel_id, user_id)

    # Active lockout?
    lockout_until = _mention_lockout.get(key, 0.0)
    if now < lockout_until:
        if not _mention_warned.get(key):
            _mention_warned[key] = True
            return True, True   # block, but emit one warning
        return True, False      # block silently
    elif lockout_until > 0:
        # Lockout just expired — clear flags.
        _mention_lockout.pop(key, None)
        _mention_warned.pop(key, None)

    # Slide window for this user.
    history = _mention_history.setdefault(key, [])
    cutoff = now - MENTION_WINDOW_SEC
    history[:] = [t for t in history if t > cutoff]
    history.append(now)

    if len(history) > MENTION_LIMIT:
        # Trip lockout.
        _mention_lockout[key] = now + MENTION_LOCKOUT_SEC
        _mention_warned[key] = True
        log.warning(
            "mention spam lockout: channel=%s user=%s (%d hits in %.0fs)",
            channel_id, user_id, len(history), MENTION_WINDOW_SEC,
        )
        return True, True
    return False, False


async def _channel_label(cid: str) -> str:
    """Return `#name` for a Discord channel ID, falling back to the ID."""
    try:
        ch = client.get_channel(int(cid))
        name = getattr(ch, "name", None)
        return f"#{name}" if name else cid
    except (ValueError, AttributeError):
        return cid


def _proactive_channel_enabled(message: discord.Message) -> bool:
    if str(message.channel.id) in PROACTIVE_CHANNELS:
        return True
    if message.guild and str(message.guild.id) in PROACTIVE_SERVERS:
        return True
    return False

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
memory = BotMemory()
agent = Agent(memory=memory)
annotator = Annotator(memory=memory)
digest = Digest(memory=memory)
profile_builder = ProfileBuilder(memory=memory)
channel_profile_builder = ChannelProfileBuilder(memory=memory)
proactive = ProactiveClassifier()


@tasks.loop(seconds=30)
async def annotator_tick() -> None:
    await annotator.maybe_flush_all()


@tasks.loop(seconds=600)
async def profile_refresh_tick() -> None:
    """Every 10 min, walk every text channel the bot is in and rebuild any
    profile (user or channel) whose threshold was crossed. The per-profile
    cooldown (1h) keeps cost bounded even with many channels.

    PUBLIC_CHANNELS get an extra `ensure()` call so their channel_profile is
    actively built (and stays available for other channels to read) even
    when nobody is @-ing the bot inside the public channel itself."""
    for guild in client.guilds:
        for channel in guild.text_channels:
            cid = str(channel.id)
            await profile_builder.maybe_refresh_channel(cid)
            if cid in PUBLIC_CHANNELS:
                await channel_profile_builder.ensure(channel_id=cid)
            await channel_profile_builder.maybe_refresh_channel(cid)


# Per-channel record of when daily/weekly digest last ran successfully.
# Persisted across bot restarts so wake-from-sleep doesn't double-post,
# and so a bot that was asleep at 9am LA still fires the digest when
# the catch-up tick runs after wake.
# Pin state to the working directory the user launches the bot from, not
# the package install path. After the coterie/ reorg, Path(__file__).parent
# resolved into the package itself (coterie/adapters/), which silently
# orphaned the existing state file at repo root and caused every daily
# digest to re-fire on the next restart. Allow env override for ops who
# want to put runtime state somewhere else.
DIGEST_STATE_FILE = Path(
    os.environ.get("DIGEST_STATE_FILE")
    or Path.cwd() / "digest_state.json"
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


async def _run_daily_if_pending(now_la: _dt.datetime) -> None:
    """Idempotent: only fires if 9am LA has passed today AND we haven't
    yet posted today's daily digest. Called by both the 09:00-LA tick
    and the catch-up tick."""
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
                last = _dt.datetime.fromisoformat(last_iso)
                if last >= target:
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
    """Idempotent weekly fire: only on Monday at/after 9am LA, and only
    once per week. State key = this week's Monday-9am ISO."""
    monday = _this_monday_9am(now_la)
    if now_la < monday:
        return  # Mon 0-9am, or not Monday at all
    state = _load_digest_state()
    changed = False
    for cid in WEEKLY_DIGEST_CHANNELS:
        ch = state.setdefault(cid, {})
        last_iso = ch.get("weekly_last_run")
        if last_iso:
            try:
                last = _dt.datetime.fromisoformat(last_iso)
                if last >= monday:
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


@tasks.loop(time=DIGEST_TIME)
async def daily_digest_tick() -> None:
    await _run_daily_if_pending(_dt.datetime.now(LA))


@tasks.loop(time=DIGEST_TIME)
async def weekly_digest_tick() -> None:
    if _dt.datetime.now(LA).weekday() != 0:
        return
    await _run_weekly_if_pending(_dt.datetime.now(LA))


@tasks.loop(minutes=10)
async def digest_catchup_tick() -> None:
    """Backup for tasks.loop(time=...), which silently misses firings
    when macOS suspends the asyncio loop overnight. Runs every 10 min;
    state file dedupes so this never double-posts."""
    now = _dt.datetime.now(LA)
    await _run_daily_if_pending(now)
    await _run_weekly_if_pending(now)


async def _push_to_channel(channel_id: str, text: str) -> None:
    try:
        channel = client.get_channel(int(channel_id))
    except ValueError:
        log.warning("digest: bad channel id %s", channel_id)
        return
    if channel is None:
        log.warning("digest: channel %s not found in cache", channel_id)
        return
    for i in range(0, len(text), DISCORD_REPLY_CHARS):
        await channel.send(text[i : i + DISCORD_REPLY_CHARS])


async def _proactive_dispatch(message: discord.Message) -> None:
    """Run after the debounce wait inside ProactiveClassifier.schedule.

    Pulls recent ctx, classifies, fires Mem0 search + agent.reply on hit,
    then sends with @-mention. Cooldown is set only AFTER a successful
    send so we don't burn the channel's quiet window on no-ops.
    """
    channel_id = str(message.channel.id)
    asker = message.author.display_name
    if proactive.cooldown_active(channel_id, asker=asker):
        return

    recent_msgs = annotator.recent_n(channel_id, n=10)

    asker_profile: str | None = None
    channel_summary: str | None = None
    if PROFILES_ENABLED:
        try:
            asker_profile = await profile_builder.ensure(
                channel_id=channel_id,
                author=asker,
            )
        except Exception:
            log.exception("proactive: profile lookup failed")
        channel_summary = await build_channel_summary_block(
            current_channel_id=channel_id,
            public_channel_ids=PUBLIC_CHANNELS,
            builder=channel_profile_builder,
            label_for_channel=_channel_label,
        )

    decision = await proactive.evaluate(
        trigger_msg_text=message.content,
        asker=asker,
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
                channel_id=channel_id,
                record_type="message",
                limit=5,
            )
        except Exception:
            log.exception("proactive: Mem0 search failed")

    try:
        reply_text, _ = await agent.reply(
            query=message.content,
            channel_id=channel_id,
            asker=asker,
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
    if not reply_text or reply_text == "<skip>" or reply_text.startswith("<skip>"):
        log.info("proactive: agent emitted <skip>, suppressing post")
        return

    final = f"<@{message.author.id}> {reply_text}"
    try:
        for i in range(0, len(final), DISCORD_REPLY_CHARS):
            await message.channel.send(final[i : i + DISCORD_REPLY_CHARS])
    except Exception:
        log.exception("proactive: send failed")
        return
    proactive.mark_fired(channel_id, asker=asker)
    log.info("proactive: fired in channel=%s asker=%s", channel_id, asker)


@client.event
async def on_ready() -> None:
    log.info("logged in as %s (backend=%s)", client.user, BACKEND)
    for guild in client.guilds:
        log.info("guild: id=%s name=%r channels=%d", guild.id, guild.name, len(guild.text_channels))
    for guild in client.guilds:
        for channel in guild.text_channels:
            try:
                pins = await channel.pins()
            except (discord.Forbidden, discord.HTTPException):
                continue
            for msg in pins:
                await memory.mark_pinned(
                    channel_id=str(channel.id),
                    message_id=str(msg.id),
                )
    if ANNOTATOR_ENABLED:
        if not annotator_tick.is_running():
            annotator_tick.start()
        log.info("annotator: enabled (tick every 30s)")
    else:
        log.info("annotator: disabled by ANNOTATOR_ENABLED=false")

    if PROFILES_ENABLED:
        if not profile_refresh_tick.is_running():
            profile_refresh_tick.start()
        log.info("profiles: enabled (refresh tick every 10min)")
        if PUBLIC_CHANNELS:
            log.info(
                "public channels (cross-channel summary source): %s",
                sorted(PUBLIC_CHANNELS),
            )
        else:
            log.info("public channels: none (cross-channel summary disabled)")
    else:
        log.info("profiles: disabled by PROFILES_ENABLED=false")

    if PROACTIVE_CHANNELS or PROACTIVE_SERVERS:
        # Pull constants from the backend's proactive module without
        # hardcoding either implementation here.
        if BACKEND == "openai":
            from coterie.gpt.proactive import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        else:
            from coterie.claude.proactive import COOLDOWN_SEC as _PCD, DEBOUNCE_SEC as _PDB
        log.info(
            "proactive: enabled channels=%s servers=%s (debounce=%ss cooldown=%ss, same-asker bypass)",
            sorted(PROACTIVE_CHANNELS), sorted(PROACTIVE_SERVERS),
            int(_PDB), int(_PCD),
        )
    else:
        log.info("proactive: no PROACTIVE_CHANNELS/SERVERS env, disabled")

    if not DIGEST_ENABLED:
        log.info("digest: disabled by DIGEST_ENABLED=false")
    else:
        if DAILY_DIGEST_CHANNELS:
            log.info("daily digest: channels=%s at 9am LA", sorted(DAILY_DIGEST_CHANNELS))
            if not daily_digest_tick.is_running():
                daily_digest_tick.start()
        else:
            log.info("daily digest: no DAILY_DIGEST_CHANNELS, disabled")
        if WEEKLY_DIGEST_CHANNELS:
            log.info(
                "weekly digest: channels=%s Mon 9am LA", sorted(WEEKLY_DIGEST_CHANNELS)
            )
            if not weekly_digest_tick.is_running():
                weekly_digest_tick.start()
        else:
            log.info("weekly digest: no WEEKLY_DIGEST_CHANNELS, disabled")
        if DIGEST_CHANNELS:
            if not digest_catchup_tick.is_running():
                digest_catchup_tick.start()


@client.event
async def on_message(message: discord.Message) -> None:
    # Filter OTHER bots only; keep our own replies so the channel record is
    # complete and the annotator window has context for user follow-ups
    # ("ok" / "yeah" against what the bot just said).
    is_self = client.user is not None and message.author.id == client.user.id
    if message.author.bot and not is_self:
        return

    if message.type is discord.MessageType.pins_add and message.reference and not is_self:
        await memory.mark_pinned(
            channel_id=str(message.channel.id),
            message_id=str(message.reference.message_id),
        )
        return

    pdf_attachments = [
        a for a in message.attachments
        if (a.content_type or "").lower().startswith("application/pdf")
        or a.filename.lower().endswith(".pdf")
    ]

    if not message.content and not pdf_attachments:
        return

    attachment_summary = (
        " [attachments: " + ", ".join(a.filename for a in pdf_attachments) + "]"
        if pdf_attachments else ""
    )
    full_content = (message.content or "") + attachment_summary
    timestamp_iso = message.created_at.isoformat()
    await memory.add(
        content=full_content,
        channel_id=str(message.channel.id),
        author_name=message.author.display_name,
        author_id=str(message.author.id),
        message_id=str(message.id),
        timestamp=timestamp_iso,
        record_type="agent_reply" if is_self else "message",
    )
    annotator.buffer_push(
        channel_id=str(message.channel.id),
        message_id=str(message.id),
        author=message.author.display_name,
        content=full_content,
        timestamp_iso=timestamp_iso,
        is_bot_reply=is_self,
    )

    # Bot never @s itself or DMs itself — return early to avoid recursion
    if is_self:
        return

    # Discord "Reply" with the @ toggle OFF doesn't add the bot to
    # message.mentions, so a reply targeted at a bot message would
    # otherwise fall through to the proactive path. Catch that case:
    # if the user is explicitly replying to one of our messages, treat
    # it as a mention.
    is_reply_to_bot = False
    if message.reference is not None and client.user is not None:
        ref = message.reference.resolved
        if ref is not None and getattr(ref, "author", None) is not None:
            is_reply_to_bot = ref.author.id == client.user.id

    if not (
        client.user in message.mentions
        or isinstance(message.channel, discord.DMChannel)
        or is_reply_to_bot
    ):
        # Not @-ed. Maybe still worth jumping in proactively?
        if message.content and _proactive_channel_enabled(message):
            proactive.schedule(
                str(message.channel.id),
                _proactive_dispatch(message),
            )
        return

    # Spam guard before any LLM call. Heavy traffic on the @ path can drain
    # the API budget fast (xhigh effort + 4 tools per reply).
    blocked, warn = _mention_should_block(
        str(message.channel.id), str(message.author.id)
    )
    if blocked:
        if warn:
            await message.reply(
                "Slow down a bit — too many @s in a row. I'll be back in a minute. "
                "If you need something urgent, edit your last message instead of "
                "sending new ones."
            )
        return

    query = message.content
    if client.user is not None:
        query = query.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "")
    query = query.strip()
    if not query and not pdf_attachments:
        return
    if not query and pdf_attachments:
        query = "Please take a look at this attachment."

    attachments_payload = await _build_attachment_blocks(pdf_attachments)

    async with message.channel.typing():
        asker_profile: str | None = None
        channel_summary: str | None = None
        if PROFILES_ENABLED:
            try:
                asker_profile = await profile_builder.ensure(
                    channel_id=str(message.channel.id),
                    author=message.author.display_name,
                )
            except Exception:
                log.exception("profile lookup failed")
            # Skip channel summary for DMs — there is no channel to summarize.
            if not isinstance(message.channel, discord.DMChannel):
                channel_summary = await build_channel_summary_block(
                    current_channel_id=str(message.channel.id),
                    public_channel_ids=PUBLIC_CHANNELS,
                    builder=channel_profile_builder,
                    label_for_channel=_channel_label,
                )
        try:
            reply_text, files = await agent.reply(
                query=query,
                channel_id=str(message.channel.id),
                asker=message.author.display_name,
                asker_profile=asker_profile,
                channel_summary=channel_summary,
                attachments=attachments_payload,
            )
        except Exception:
            log.exception("agent failed")
            await message.reply("Something went wrong — check the bot log.")
            return

    await _send_reply(message, reply_text, files)


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    """Record reactions left on the bot's own messages as implicit feedback.

    Filters: skip self-reactions; skip reactions on non-bot messages. Uses
    the *raw* event so it fires for messages older than the bot's cache,
    not just for cached ones.
    """
    if client.user is None or payload.user_id == client.user.id:
        return
    channel = client.get_channel(payload.channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(payload.channel_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return
    try:
        target = await channel.fetch_message(payload.message_id)
    except (discord.NotFound, discord.Forbidden, discord.HTTPException):
        return
    if target.author.id != client.user.id:
        return  # only bot-message reactions are signal

    reactor = payload.member
    if reactor is None:
        try:
            reactor = await client.fetch_user(payload.user_id)
        except (discord.NotFound, discord.HTTPException):
            return
    reactor_name = getattr(reactor, "display_name", None) or getattr(reactor, "name", str(payload.user_id))

    now_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    try:
        await memory.add_reaction(
            channel_id=str(payload.channel_id),
            reactor_name=reactor_name,
            reactor_id=str(payload.user_id),
            emoji=str(payload.emoji),
            target_message_id=str(payload.message_id),
            target_excerpt=target.content or "",
            timestamp=now_iso,
        )
    except Exception:
        log.exception("memory.add_reaction failed")


async def _build_attachment_blocks(
    pdfs: list[discord.Attachment],
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for att in pdfs:
        if att.size > MAX_PDF_BYTES:
            log.warning("skipping %s (%d bytes, too large)", att.filename, att.size)
            continue
        try:
            data = await att.read()
        except Exception:
            log.exception("failed to download attachment %s", att.filename)
            continue
        blocks.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
            "title": att.filename,
        })
    return blocks


async def _send_reply(
    message: discord.Message,
    text: str,
    files: list[tuple[str, bytes]],
) -> None:
    discord_files = [
        discord.File(io.BytesIO(data), filename=name) for name, data in files
    ]
    if not text and not discord_files:
        await message.reply("(empty reply)")
        return

    # Discord caps replies at 2000 chars; chunk the text. Files attach to the
    # first chunk only.
    chunks = [text[i : i + DISCORD_REPLY_CHARS] for i in range(0, len(text), DISCORD_REPLY_CHARS)] or [""]
    first, rest = chunks[0], chunks[1:]
    await message.reply(first or "(see attached)", files=discord_files or None)
    for chunk in rest:
        await message.reply(chunk)


def main() -> None:
    """Entry point for the ``coterie-discord`` console script."""
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN not set (copy .env.example to .env)")
    client.run(token)


if __name__ == "__main__":
    main()
