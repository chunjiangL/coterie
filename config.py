"""Centralized non-secret configuration.

Two kinds of knobs:

* ``PLATFORM`` — which chat platform we're talking to: ``discord`` (default)
  or ``slack``. Drives the markdown / formatting rules injected into agent
  prompts; doesn't touch business logic.
* ``COMMUNITY_*`` — human-readable strings describing what the community
  talks about and what counts as "substantive content". Replaces the formerly
  hard-coded "research" / "ML research" vocabulary so the same codebase can
  serve a frontend-dev guild, a DAO governance group, a design-crit channel,
  etc.

All vars are optional with sane defaults that preserve the original
ML-research bot behavior. Set ``COMMUNITY_DOMAIN="frontend engineering"``
(say) and every prompt that used to say "research-channel" will now say
"frontend engineering-channel".

Used by every prompt-bearing module:
``agent[_openai].py``, ``proactive[_openai].py``, ``digest[_openai].py``,
``annotator[_openai].py``, ``user_profile[_openai].py``.

Placeholder substitution is done with ``str.replace`` (NOT ``str.format``)
so prompts can freely contain ``{}`` for LaTeX / JSON examples without
blowing up — a lesson learned the hard way in commit 9a33404.
"""

import os

PLATFORM = os.environ.get("PLATFORM", "discord").lower().strip()
if PLATFORM not in ("discord", "slack"):
    raise SystemExit(
        f"PLATFORM={PLATFORM!r} not supported (valid: discord|slack)"
    )


# ─── Community identity ─────────────────────────────────────────────
# Human-readable label. Currently only injected into prompts; could later
# be used in greeting lines, digest headers, etc.
COMMUNITY_NAME = os.environ.get("COMMUNITY_NAME", "the channel")

# What the community talks about. Replaces "research" wherever that was
# hard-coded. Examples: "research", "frontend engineering", "DAO governance",
# "product design crits".
COMMUNITY_DOMAIN = os.environ.get("COMMUNITY_DOMAIN", "research")

# Short prose listing what counts as substantive content (vs chitchat).
# Used by the proactive classifier and the digest to decide what's worth
# talking about / summarizing. Default keeps the ML-research vocabulary.
SUBSTANTIVE_TOPICS = os.environ.get(
    "SUBSTANTIVE_TOPICS",
    "model architecture, training, papers, datasets, infrastructure, "
    "RL/RLHF, agent design, code, hardware",
)

# Comma-prose listing URL domains that signal real content (not memes /
# random news / unrelated tweets). Used by the proactive classifier.
RELEVANT_LINK_DOMAINS = os.environ.get(
    "RELEVANT_LINK_DOMAINS",
    "arxiv abstract/pdf, GitHub repos about ML / agents / research code, "
    "X.com (Twitter) posts about a paper or project, huggingface model "
    "cards, anthropic / openai / deepmind / thinkingmachines blog posts",
)


def platform_label() -> str:
    """Friendly capitalized name for prompts ("Discord" / "Slack")."""
    return "Slack" if PLATFORM == "slack" else "Discord"


# ─── Platform markdown / formatting rules ───────────────────────────
# Injected into the agent's system prompt so the model emits chat that
# renders correctly on the target platform. Discord and Slack have
# substantially different mrkdwn dialects.

DISCORD_FORMATTING = """═══ Chat formatting (READ CAREFULLY) ═══

This is a Discord chat, NOT a Notion / Markdown editor. Several common \
Markdown elements render badly or not at all. You MUST respect these rules:

DOES render in Discord:
- `**bold**`, `*italic*`, `~~strike~~`, `__underline__`
- `` `inline code` `` and ``` ```code block``` ```
- `> quote` (single line) and `>>> multi-line quote`
- `# H1`, `## H2`, `### H3` — but these are HUGE in Discord, use sparingly
- Numbered (`1.`) and bullet (`-`) lists
- `[text](url)` hyperlinks
- `||spoiler||`

Does NOT render — these show up as literal characters:
- LaTeX / MathJax: `$x^2$`, `\\frac{}{}`, `\\sum`, etc. → use Unicode (² ³ ½ \
× ÷ ≤ ≥ π Σ ∇) or simple ASCII (x^2, sum_i, dF/dx) instead.
- Tables: `| col1 | col2 |` → use bullets with arrows.
- Horizontal rules: `---` → just use a blank line.
- HTML / images (only Discord-attached files render).

Style rules for chat replies:
- DO NOT wrap PROSE in ``` ```text ``` ``` code fences.
- Limit `##` / `###` headings to AT MOST 2 per reply.
- Don't number 10 sub-sections. If your answer needs that, either confirm \
the user wants a deep dive first, or break into ≤ 5 messages of ≤ 8 lines.
- For N-way comparisons, prefer prose with arrows or 1-2 sentence bullets, \
NOT a table.
- Code snippets: < 15 lines.
"""

SLACK_FORMATTING = """═══ Chat formatting (READ CAREFULLY) ═══

This is a Slack channel. Slack uses *mrkdwn* — close to Markdown but NOT \
identical to Discord-style markdown. You MUST respect these rules:

DOES render in Slack:
- `*bold*` (SINGLE asterisks — `**double**` will NOT bold, just shows stars)
- `_italic_` (SINGLE underscores — `*italic*` reads as bold in Slack)
- `~strike~` (SINGLE tilde — `~~double~~` does NOT work)
- `` `inline code` `` and ``` ```code block``` ```
- `> quote`
- Numbered (`1.`) and bullet (`-`) lists
- `<https://example.com|link text>` hyperlinks (NOT `[text](url)` — \
Slack ignores Markdown-style links entirely)
- emoji shortcodes: `:smile:` `:wave:` `:pray:`
- mentions: write plain `@username` (Slack auto-resolves)

Does NOT render — these show up as literal characters:
- LaTeX / MathJax: `$x^2$`, `\\frac{}{}`, etc. → use Unicode (² ³ ½ × ÷ ≤ \
≥ π Σ ∇) or simple ASCII (x^2, sum_i, dF/dx) instead.
- Tables: `| col1 | col2 |` → use bullets with arrows.
- `# H1`, `## H2`, `### H3` — Slack ignores ALL Markdown headings entirely.
- Horizontal rules `---`.
- Discord-style `**bold**`, `[text](url)`, `||spoiler||`, `~~strike~~`.

Style rules for chat replies:
- Most replies 1-3 sentences. Complex: ≤ 10 short lines, prose flow.
- For multi-section answers: use plain text "First, ... Second, ..." or \
*bold* labels. Markdown headings don't render in Slack — don't waste them.
- For N-way comparisons, prefer prose with arrows or bullets, NOT a table.
- Code snippets: < 15 lines.
- Slack threads reply messages by default; the bot wrapper handles threading.
"""


def formatting_rules() -> str:
    return SLACK_FORMATTING if PLATFORM == "slack" else DISCORD_FORMATTING


# ─── Render helper ──────────────────────────────────────────────────
# Substitute all known placeholders in a prompt string. Returns the
# rendered string. Unknown `{placeholder}` patterns pass through unchanged
# (no errors), so prompts can include LaTeX `{}` or JSON examples freely.
#
# Placeholders supported:
#   {platform}              → "Discord" or "Slack"
#   {community_name}        → e.g. "ICLR DDoS"
#   {community_domain}      → e.g. "research"
#   {substantive_topics}    → comma-prose list
#   {relevant_link_domains} → comma-prose list
#   {formatting_rules}      → the full Discord/Slack rules block

def render(template: str) -> str:
    return (
        template
        .replace("{platform}", platform_label())
        .replace("{community_name}", COMMUNITY_NAME)
        .replace("{community_domain}", COMMUNITY_DOMAIN)
        .replace("{substantive_topics}", SUBSTANTIVE_TOPICS)
        .replace("{relevant_link_domains}", RELEVANT_LINK_DOMAINS)
        .replace("{formatting_rules}", formatting_rules())
    )
