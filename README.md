# Coterie

A lightweight Discord chat-management bot for small research groups.

- **@-mention replies** with channel-scoped semantic memory (Mem0 + Chroma)
- **Per-user profiles** auto-built from message history (identity, style, taste,
  bot-interaction preference)
- **Daily / weekly digests** with adjacent reading recommendations via
  web_search — posted to selected channels on a 9am-LA cron with macOS-sleep
  catch-up
- **Proactive participation**: non-@ messages classified by a reasoning model;
  the bot chimes in only when it can add concrete value (≤1-2 sentences)
- **Pinned-message awareness** — channel admins' pins get a retrieval boost
- **PDF attachments** read natively; `code_interpreter` for plots & file output
- **Swappable LLM backend** via `BACKEND` env var: Anthropic Claude or OpenAI

Built originally for the "ICLR DDoS" research group; small enough to fork and
re-purpose for any 5-30 person chat that wants light AI-assisted curation.

---

## Quick start

```bash
git clone https://github.com/chunjiangL/coterie
cd coterie
uv venv && source .venv/bin/activate
uv pip install -e .            # reads dependencies from pyproject.toml

cp .env.example .env
# Edit .env — fill DISCORD_TOKEN + at least one API key (see below)

python bot.py                  # foreground
# or: nohup python bot.py > bot.log 2>&1 &
```

---

## Discord bot setup

1. Go to **<https://discord.com/developers/applications>** → **New Application**
   → name it.
2. In the new app, **Bot** tab → **Reset Token** → copy. This is your
   `DISCORD_TOKEN` (paste into `.env`).
3. **Privileged Gateway Intents** → enable **MESSAGE CONTENT INTENT**.
   Required so the bot can actually read message text (Discord blocks
   reading content by default since 2022). Save.
4. **OAuth2** → **URL Generator** → check:
   - **Scopes**: `bot`
   - **Bot Permissions**: `View Channels`, `Send Messages`, `Read Message
     History`, `Embed Links`, `Attach Files`, `Add Reactions`
5. Copy the generated invite URL → open in browser → pick the target server
   → authorize. The bot now shows up offline in your member list; it will
   come online when `python bot.py` is running.

To find **channel** / **server (guild)** IDs for the env config:

- In Discord, **User Settings → Advanced → Developer Mode** (turn on)
- Right-click a channel → **Copy Channel ID**
- Right-click a server icon → **Copy Server ID**

---

## Configuration (`.env`)

### Required

| Var | Example | Notes |
|---|---|---|
| `DISCORD_TOKEN` | `MTUw...` | From step 2 above |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Needed if `BACKEND=anthropic` |
| `OPENAI_API_KEY` | `sk-proj-...` | Needed if `BACKEND=openai` |

### Backend

| Var | Default | Values | What it does |
|---|---|---|---|
| `BACKEND` | `anthropic` | `anthropic` / `openai` | Picks which LLM stack to use. Both share the same Mem0 vector store — swap freely without losing channel memory. |

| Backend | Main agent | Annotator / Profile | Digest | Classifier | Mem0 internal |
|---|---|---|---|---|---|
| `anthropic` | Claude Opus 4.7 | Claude Sonnet 4.6 | Opus 4.7 | Opus 4.7 | Claude Haiku 4.5 |
| `openai` | GPT-5.5 (Pro on user opt-in) | GPT-5.5 | GPT-5.5 Pro | GPT-5.5 Pro | GPT-4o-mini |

For the OpenAI backend, users can request the heavier Pro model in a single
reply by including phrases like `用 pro`, `gpt-5.5-pro`, `深度思考`, `用最好`
in their `@`-mention message. Proactive replies always stay on the default.

### Feature toggles

| Var | Default | What it does |
|---|---|---|
| `PROFILES_ENABLED` | `true` | Build + auto-refresh per-user profiles. Off → no profile in @ reply context, no refresh tick. |
| `ANNOTATOR_ENABLED` | `true` | Sliding-window English annotations of messages. Off → cheaper, but semantic search recall drops noticeably. |

### Digest channels

| Var | What it does |
|---|---|
| `DAILY_DIGEST_CHANNELS` | Comma-separated channel IDs that get a 9am-LA daily summary. |
| `WEEKLY_DIGEST_CHANNELS` | Comma-separated channel IDs that get a Monday-9am-LA weekly summary. |
| `DIGEST_CHANNELS` | Legacy fallback — used for both daily and weekly when the two above are unset. |

Leave all three empty to disable digests.

### Proactive participation

| Var | What it does |
|---|---|
| `PROACTIVE_CHANNELS` | Comma-separated channel IDs where the bot may proactively reply to non-@ messages. |
| `PROACTIVE_SERVERS` | Comma-separated guild (server) IDs where proactive is on for **every** text channel. New channels auto-inherit. |

Empty both → proactive disabled. A non-@ message must pass an LLM classifier
(6 substantive-content criteria) before the bot speaks; rate-limited to 8s
debounce + 60s per-channel cooldown (same-speaker follow-ups bypass cooldown).

---

## Running

```bash
python bot.py
```

Or as a background process so you can close the terminal:

```bash
nohup python bot.py > bot.log 2>&1 &
```

**macOS sleep warning.** `tasks.loop(time=9am)` only fires if the bot is
awake at exactly 9:00 LA. macOS sleep cycles will skip the firing; the
10-min catch-up tick eventually backfills, but exact-time delivery requires
running on a host that never sleeps. Options:

- `caffeinate -i python bot.py` (keeps awake while the bot runs)
- a separate Linux box / VPS / Fly.io machine
- `pmset noidle` while running

---

## CLI utilities

| Script | Purpose |
|---|---|
| `python init_profiles.py [channel_id ...]` | One-shot rebuild of every (channel, author) profile. Useful after a prompt change or to bootstrap profiles for users the bot has never been @-ed by. Uses `DIGEST_CHANNELS` if no IDs passed. |
| `python digest.py <channel_id> [daily\|weekly]` | Generate a digest immediately and print to stdout (no Discord post). Anthropic backend. |
| `python digest_openai.py <channel_id> [daily\|weekly]` | Same, OpenAI backend. |
| `python user_profile.py <channel_id> <author>` | Rebuild a single profile and print. |
| `python dump_memory.py <channel_id>` | Dump all Mem0 records for a channel (debug). |
| `python verify_filter_search.py` | Standalone test that seeds 4 records and exercises author/time filters end-to-end. |

---

## Architecture

```
                     Discord gateway
                           │
                           ▼
                        bot.py
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
   on_message         tasks.loop ticks    on_ready
       │                   │                   │
       │   ┌───────────────┼─────────────┐     │
       │   │               │             │     │
       ▼   ▼               ▼             ▼     ▼
   Agent  Proactive   Annotator   ProfileBuilder  Digest
   reply  classifier  (sliding    (per (channel,  (daily +
   (@/    + dispatch  window      author))         weekly)
   proact) (debounce)  flush)
       │       │            │            │           │
       └───────┴────────────┴────────────┴───────────┘
                           │
                           ▼
                       BotMemory
                      (Mem0 wrapper)
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        Chroma store   HF embedder   internal LLM
        (./mem0_data)                 (Haiku /
                                       GPT-4o-mini)
```

Each module has Anthropic and OpenAI variants (`agent.py` vs `agent_openai.py`,
etc.). `backends.py` reads the `BACKEND` env var at startup and re-exports the
chosen variants, so `bot.py` itself is backend-agnostic.

The vector store is partitioned per channel (Mem0 `user_id = channel_id`).
Records carry a `record_type` tag: `message` / `annotation` / `agent_reply` /
`profile`. The agent's `search_memories` tool excludes `profile` records to
prevent cross-user profile leakage; other types are surfaced with their tag so
the model can weigh them appropriately.

---

## License

MIT (do whatever, attribution appreciated but not required).
