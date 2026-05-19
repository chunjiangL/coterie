# Coterie

A bot for small chat communities. Runs on Discord or Slack. Maintains
per-channel memory, builds per-user and per-channel profile blurbs,
records reactions on its own messages as implicit feedback, fires daily
and weekly digests, and optionally chimes in on its own. One process.
Behavior toggles through env vars.

## Quick start

```
git clone https://github.com/chunjiangL/coterie
cd coterie
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# fill in tokens
coterie-discord    # or: coterie-slack
```

## Discord setup

At https://discord.com/developers/applications, create an app. Bot tab,
reset and copy the token. This is `DISCORD_TOKEN`.

On the same page, enable `MESSAGE CONTENT INTENT` under Privileged
Gateway Intents. Without it the bot cannot read message text.

OAuth2, URL Generator: scope `bot`, permissions View Channels, Send
Messages, Read Message History, Embed Links, Attach Files, Add Reactions.
Open the generated URL to invite the bot to a server.

For channel and server IDs, turn on Developer Mode (User Settings,
Advanced), then right-click.

## Slack setup

At https://api.slack.com/apps, create an app from scratch in your
workspace.

Socket Mode: toggle on, generate an app-level token with
`connections:write`. This is `SLACK_APP_TOKEN` (starts with `xapp-`).

OAuth & Permissions, Bot Token Scopes: `app_mentions:read`,
`channels:history`, `channels:read`, `chat:write`, `files:read`,
`files:write`, `groups:history`, `groups:read`, `im:history`, `im:read`,
`im:write`, `reactions:read`, `users:read`.

Event Subscriptions: enable, subscribe to bot events `app_mention`,
`message.channels`, and `reaction_added`. Add `message.groups` and
`message.im` if you want the bot in private channels or DMs.

Install App, Install to Workspace, copy the Bot User OAuth Token (starts
with `xoxb-`). This is `SLACK_BOT_TOKEN`.

`/invite @yourbot` into channels. Channel IDs are at the bottom of View
channel details.

## Configuration

Most variables are optional. Defaults assume an ML research group; change
the community-framing block to retune.

```
# Platform credentials
DISCORD_TOKEN=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

# LLM backend, pick one
BACKEND=anthropic        # default. Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5.
ANTHROPIC_API_KEY=sk-ant-...
# or
BACKEND=openai           # GPT-5.5 Pro / GPT-5.5 / GPT-4o-mini.
OPENAI_API_KEY=sk-...

# Community framing
COMMUNITY_NAME=the channel
COMMUNITY_DOMAIN=research
SUBSTANTIVE_TOPICS=model architecture, training, papers, datasets, ...
RELEVANT_LINK_DOMAINS=arxiv, github, X.com, huggingface, ...

# Feature toggles
PROFILES_ENABLED=true    # per-user + per-channel profile build + auto-refresh
ANNOTATOR_ENABLED=true   # sliding-window LLM annotations for retrieval

# Digest. Fires 9am LA. Comma-separated channel IDs. Empty disables.
DAILY_DIGEST_CHANNELS=
WEEKLY_DIGEST_CHANNELS=
DIGEST_CHANNELS=          # legacy fallback for both

# Proactive (non-@ replies). Empty disables.
PROACTIVE_CHANNELS=
PROACTIVE_SERVERS=        # whole guild / workspace; channels auto-inherit

# Cross-channel context. Channels listed here have their channel_profile
# treated as "public knowledge" — every reply in every channel can read
# them. Private channels never appear here and never leak. Empty (default)
# = no cross-channel sharing.
PUBLIC_CHANNELS=
```

On the OpenAI backend, users can ask for the heavier model in a single
message by including "use pro", "pro mode", "think hard", or "best
model". Proactive replies always run on the default.

## Running

```
coterie-discord
coterie-slack
```

Two independent processes. They share `mem0_data/` (the Chroma store)
but write separate digest state files.

The daily and weekly digests fire on a 9am LA cron. On a laptop that
sleeps overnight the timer misses, and a catch-up tick backfills within
ten minutes of wake. For exact-time delivery, run on a host that stays
awake or wrap with `caffeinate -i coterie-discord`.

## Architecture

```
coterie/
  adapters/
    discord_adapter.py    discord.py event loop
    slack_adapter.py      slack_bolt Socket Mode
  claude/                 Anthropic variants (agent, annotator, channel_profile, digest, proactive, user_profile)
  gpt/                    OpenAI variants (same six)
  backends.py             BACKEND env switch
  config.py               PLATFORM + COMMUNITY_* env + render()
  memory.py               Mem0 wrapper, partitioned per channel
  web_fetch.py            httpx + trafilatura fetcher
scripts/
  init_profiles.py
  dump_memory.py
  verify_filter_search.py
```

Each adapter handles only its platform's SDK quirks. `backends.py` reads
`BACKEND` at startup and re-exports the chosen LLM stack. `config.py`
substitutes `{platform}`, `{community_domain}`, and other placeholders
into prompts at module load.

Mem0 partitions by channel ID. Records carry a `record_type` tag:
`message`, `annotation`, `agent_reply`, `profile`, `channel_profile`,
`reaction`. The agent's `search_memories` drops `profile`,
`channel_profile`, and `reaction` records before returning, so synthesized
or signal-layer rows never leak back into a user-facing reply.

When the bot replies, its user turn is split into one main block and up
to four reference blocks: `<channel_summary>`, `<asker_profile>`,
`<recent_chat>`, `<retrieved_memories>`, and finally
`<untrusted_user_content>` (the message to respond to). The system prompt
tells the model to treat the reference blocks as data and respond only
to the last one. This is both an anti-injection defense and a way to
keep context confusion down when many blocks pile up.

## CLI

```
python scripts/init_profiles.py [channel_id ...]
python -m coterie.claude.channel_profile <channel_id>
python -m coterie.gpt.channel_profile <channel_id>
python -m coterie.claude.digest <channel_id> [daily|weekly]
python -m coterie.gpt.digest <channel_id> [daily|weekly]
python scripts/dump_memory.py <channel_id>
python scripts/verify_filter_search.py
```

## License

MIT
