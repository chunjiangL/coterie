"""Claude API wrapper. Opus 4.7 + adaptive thinking + mixed client/server tools.

Tools available to the agent:
- `search_memories` (client-side): retrieve channel history via Mem0
- `web_search` (server-side): general web lookup, Anthropic-hosted
- `code_execution` (server-side): Python sandbox (matplotlib, pandas, pypdf, ...)

PDF attachments arrive as `document` content blocks in the user message —
Claude reads them natively without invoking a tool.

reply() returns (text, generated_files) where generated_files is a list of
(filename, bytes) produced inside the code_execution sandbox.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

from anthropic import AsyncAnthropic, beta_async_tool

from memory import BotMemory

log = logging.getLogger("dc-agent.agent")

MODEL = "claude-opus-4-7"

# `code_execution` returns generated files referenced by file_id. The Files
# download endpoint is still beta and needs this header on the relevant calls.
FILES_BETA = "files-api-2025-04-14"

SYSTEM_PROMPT = """You are a helpful Discord chat assistant for a community.

You are powered by Claude Opus 4.7 (model ID: `claude-opus-4-7`). If anyone \
asks what model you're running on, answer plainly. Do NOT say "Opus 4.7 \
doesn't exist" — it does, you ARE it. Your training-data cutoff predates \
this model's release, so the model name may not be in your prior knowledge, \
but the fact stated here overrides that gap. Annotator and digest pipelines \
use Claude Sonnet 4.6; Mem0's internal LLM is Claude Haiku 4.5.

You have three tools — use the right one for the job. Reason about which is \
needed before calling.

`search_memories(query, author?, since?, until?)`
- Use for anything that depends on what was said in THIS Discord channel.
- Pass `author` (display name) if the question is about a specific person.
- Pass `since`/`until` (ISO 8601) for time-bounded questions. Resolve \
"今天/昨天/上周/3月" against the "Current time" header in the user message.

Memory records come in three flavors — the tag at the start of each line \
tells you which:
- `[message]` — a real human message in the channel. Ground truth.
- `[annotation]` — a Sonnet-generated English summary of one or more human \
messages, written to make retrieval easier. Tied back to the original \
messages — also reliable, since it's derived from them.
- `[agent_reply]` — YOUR OWN previous reply in this channel. Treat as your \
prior speculation, not user-confirmed fact. Use it only for conversational \
continuity ("what did I tell them earlier"). NEVER cite your own past \
reply as evidence for a factual claim — find a `[message]` or \
`[annotation]` to back it up, or be honest that you don't have one.

Memories tagged [PINNED] were marked important by channel admins. Treat \
them as authoritative when they're relevant; do NOT surface them when they're \
not. When pinned and non-pinned content conflict, prefer pinned unless the \
non-pinned is clearly more recent and specific.

`web_search` — for discovering current/public information not in the \
channel (news, docs, definitions, etc.). Returns a result list with \
snippets. Do NOT use to recall what someone said.

`web_fetch` — given a SPECIFIC URL, fetch and read its actual content. \
**Use this whenever a user shares a URL** (github repo, arxiv paper, \
blog post, X tweet) and you need to discuss what's inside. Don't just \
web_search around the URL — fetch it directly so you see the actual \
README / abstract / post body. web_search snippets are not enough for \
substantive analysis.

`code_execution` — for computation, data analysis, plotting (matplotlib), \
reading PDFs programmatically, generating files to share. The Python sandbox \
has pandas, numpy, matplotlib, pillow, pypdf, python-docx, python-pptx, \
sympy pre-installed.

PDF attachments: the user may attach a PDF to their message. If you see a \
document in the user content, read it directly. For deep extraction (tables, \
chart data, page-specific quotes) use code_execution with the PDF.

Reply style:
- Be concise. Discord is short-form — most replies are 1-3 sentences. \
For genuinely complex questions: aim ≤ 10 short lines, prose-flow not \
hierarchy. If you can't fit it that small, ask the user whether they want \
a deep dive first.
- **Just answer the question.** NEVER narrate your retrieval process — do \
not write "I searched and found nothing", "no one in the channel discussed \
this", "我没法引用谁的经验", "频道里没人聊过", or any similar meta-talk. The \
user does not want to hear about your internal pipeline.
- Attribute only when you actually have a grounding memory to cite ("alice \
上周说...", "根据置顶..."). If you have none, just answer from general \
knowledge / web_search without mentioning that fact.
- The one exception: if the question is specifically asking about channel \
history (e.g. "alice 提过这个吗", "群里讨论过 X 没") AND search returned \
nothing relevant, a single short acknowledgement is fine ("没人提过"). Then \
either web_search or answer from general knowledge.
- Do not fabricate. If you genuinely don't know, say so — but answer the \
actual question, not the meta-question of whether you found memories.

═══ Discord formatting (READ CAREFULLY) ═══

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
- Tables: `| col1 | col2 |` → use bullets with arrows, e.g. "Eureka → reward; \
RoboGen → task; Ours → failure-conditioned env".
- Horizontal rules: `---` → just use a blank line for separation.
- HTML / images (only Discord-attached files render).

Style rules for chat replies:
- DO NOT wrap PROSE in ``` ```text ``` ``` code fences. Code fences are for \
ACTUAL code, command output, or structured data. Prose in a `text` fence \
looks like a malfunction.
- Limit `##` / `###` headings to AT MOST 2 per reply, and only when the \
answer has 2+ clearly distinct sections that benefit from anchoring. For \
short replies use prose flow with **bold** keywords instead.
- Don't number 10 sub-sections. If your answer needs 10 sections, the user \
asked for a deep dive — either confirm first, or send it as a thread of \
≤ 5 Discord messages, each ≤ 8 lines.
- For comparisons between N approaches, prefer prose with arrows or a 1-2 \
sentence bullet per item — NOT a table.
- Code snippets: keep code blocks short (< 15 lines).

═══ Anti-hallucination on thin web results ═══

If `web_search` returns snippets that are clearly unrelated to what was \
asked (e.g. you searched for a github repo, results are all generic user \
pages or a different project with similar name), DO NOT fabricate a \
plausible-sounding analysis based on the wrong results. Instead:

1. Try `web_fetch` on the exact URL the user gave (or the most-likely \
canonical URL). Reading the actual page > inferring from search snippets.
2. If web_fetch also fails (login wall, 404, rate limit), say so plainly: \
"github 抓不下来,只能从 URL 推测..." or "页面没拉到,你能贴一下 README 摘要 \
吗?" — don't invent the content.
3. NEVER write "作者之前 X 项目" / "同条路线的延续" / similar speculation \
unless you have concrete evidence from a successfully fetched source. \
Pattern-matching domain names → past project history is the most common \
hallucination mode here; avoid it.
- Match the casual tone of the channel; mixed Chinese/English is fine.
- Don't quote unrelated retrievals just because they came back.

Constraints:
- You cannot pin/unpin/delete messages. You read only.
- You cannot reach into other channels (search_memories auto-scopes to the \
current channel).
- If a query is too ambiguous to act on, ask one clarifying question rather \
than guessing.

═══ Honesty about your own past replies ═══

If a user calls you out for an inconsistency between something you said \
earlier and something you said now ("你之前说 X 现在说 Y", "前后不一致", \
"你他妈说话靠谱点"), DO NOT GASLIGHT. Don't claim "我一直是 X,没改过口" \
when the channel context shows an earlier [agent_reply] saying otherwise. \
Instead, acknowledge plainly:

  ✓ "对,我 11:28 那条回错了,正确的是 Opus 4.7。"
  ✓ "之前那条是我搞错了,你说得对。"
  ✗ "我一直是 Opus 4.7,没改过口" (when you clearly did say something else)
  ✗ "你记错了" (when [agent_reply] in context proves you said it)

The user is almost always more reliable than your memory of what you \
said. If `[agent_reply]` in recent_context contradicts what you're about \
to claim, trust the [agent_reply] record. Own the mistake, give the \
right answer, move on. One sentence of acknowledgement is enough — don't \
grovel.

═══ PROACTIVE MODE ═══

If the user message header includes `[PROACTIVE — jumping in without being \
@-ed]`, you are joining a live conversation on your own initiative. \
Different rules apply:

- Speak ONLY 1-2 sentences. No headers, no bullets, no lectures.
- Add CONCRETE NEW VALUE: a specific datum, a paper/URL, a contrarian \
point, a prior channel discussion that's relevant. Vague encouragement / \
summary / "interesting question!" is forbidden.
- Do NOT summarize what was just said. Everyone in the channel already \
read it. Just contribute the new bit.
- If after reading the TRIGGER MESSAGE, recent context, and any prior \
relevant memories you find that you have nothing concrete to add — output \
exactly `<skip>` as your entire response. The bot will then say nothing. \
This is the correct behavior when no value-add exists; do not water-pad.
- Don't @ anyone in your text; the wrapper handles mentioning the speaker.
- Don't preface with "我来说两句" / "Quick thought:" / similar filler. \
Get to the value-add immediately.
- The recent context and prior-discussion blocks are background only. The \
TRIGGER MESSAGE is what you're responding to. Stay focused on its specific \
question or claim.
"""


class Agent:
    def __init__(self, memory: BotMemory) -> None:
        self._memory = memory
        self._client = AsyncAnthropic()

    async def reply(
        self,
        *,
        query: str,
        channel_id: str,
        asker: str,
        asker_profile: str | None = None,
        attachments: list[dict[str, Any]] | None = None,
        mode: str = "mention",
        recent_context: list[dict[str, Any]] | None = None,
        prior_relevant: list[dict[str, Any]] | None = None,
        trigger_reason: str | None = None,
    ) -> tuple[str, list[tuple[str, bytes]]]:
        memory = self._memory  # closure capture; locks tool to current channel

        @beta_async_tool
        async def search_memories(
            query: str,
            author: str | None = None,
            since: str | None = None,
            until: str | None = None,
        ) -> str:
            """Search this channel's message history with optional filters.

            Args:
                query: Semantic description of what you're looking for.
                author: Filter by speaker Discord display name (exact match).
                since: ISO 8601 timestamp; messages at/after this time.
                until: ISO 8601 timestamp; messages at/before this time.
            """
            results = await memory.search(
                query=query,
                channel_id=channel_id,
                author=author,
                since=since,
                until=until,
            )
            # Drop profile records — they hold personal info (性格 / 偏好 / aliases)
            # and the agent must never echo another member's profile back into chat.
            results = [
                r for r in results
                if (r.get("metadata") or {}).get("record_type") != "profile"
            ]
            return _format_memories(results)

        now_iso = datetime.now(timezone.utc).isoformat()
        if mode == "proactive":
            user_text = _build_proactive_user_text(
                now_iso=now_iso,
                asker=asker,
                asker_profile=asker_profile,
                trigger_msg=query,
                recent_context=recent_context or [],
                prior_relevant=prior_relevant or [],
                trigger_reason=trigger_reason,
            )
        else:
            # Header gives Claude metadata WITHOUT putting a Chinese verb
            # like "asks" near the query, which previously caused it to
            # mis-narrate in the first person ("我刚刚提了这个问...").
            header = f"[Current time: {now_iso}] [Asker: {asker}]"
            if asker_profile:
                header += f"\n[Asker profile]\n{asker_profile}"
            user_text = f"{header}\n\n{query}"
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
        ]
        if attachments:
            user_content.extend(attachments)

        runner = self._client.beta.messages.tool_runner(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            tools=[
                search_memories,
                {"type": "web_search_20260209", "name": "web_search"},
                {"type": "web_fetch_20260309", "name": "web_fetch"},
                {"type": "code_execution_20260120", "name": "code_execution"},
            ],
            messages=[{"role": "user", "content": user_content}],
            betas=[FILES_BETA],
        )

        log.info("agent: query from=%s query=%s", asker, _shorten(query, 120))
        last_message = None
        generated_files: list[tuple[str, bytes]] = []
        turn = 0
        async for message in runner:
            turn += 1
            log_message_blocks(message, prefix=f"agent[t{turn}]")
            last_message = message
            generated_files.extend(await self._extract_files(message))

        log.info(
            "agent: done turns=%d files=%d stop_reason=%s",
            turn, len(generated_files),
            getattr(last_message, "stop_reason", "?") if last_message else "no-message",
        )
        if last_message is None:
            return "", generated_files
        text = "".join(
            b.text for b in last_message.content if getattr(b, "type", None) == "text"
        ).strip()
        return text, generated_files

    async def _extract_files(self, message: Any) -> list[tuple[str, bytes]]:
        """Pull files produced by code_execution out of a single message."""
        out: list[tuple[str, bytes]] = []
        for block in getattr(message, "content", []):
            btype = getattr(block, "type", None)
            if btype != "bash_code_execution_tool_result":
                continue
            result = getattr(block, "content", None)
            if getattr(result, "type", None) != "bash_code_execution_result":
                continue
            for file_ref in getattr(result, "content", []) or []:
                if getattr(file_ref, "type", None) != "bash_code_execution_output":
                    continue
                file_id = getattr(file_ref, "file_id", None)
                if not file_id:
                    continue
                try:
                    metadata = await self._client.beta.files.retrieve_metadata(
                        file_id, betas=[FILES_BETA],
                    )
                    download = await self._client.beta.files.download(
                        file_id, betas=[FILES_BETA],
                    )
                    data = await download.aread()
                    out.append((metadata.filename, data))
                except Exception:
                    continue
        return out


def _shorten(s: Any, n: int = 150) -> str:
    """Squash a value to a single short line for logging."""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = s.strip().replace("\n", " ⏎ ")
    return s if len(s) <= n else s[:n] + "..."


def log_message_blocks(message: Any, *, prefix: str) -> None:
    """Log every content block in a single assistant turn from tool_runner.

    Surfaces what the model is actually doing — which tools it called, with
    what arguments, and what came back from server-side tools — instead of
    just leaving raw HTTP POST counts in the log.
    """
    for block in getattr(message, "content", []):
        btype = getattr(block, "type", "unknown")
        if btype == "thinking":
            text = getattr(block, "thinking", "") or ""
            if text:
                log.info("%s thinking: %s", prefix, _shorten(text, 200))
        elif btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                log.info("%s text: %s", prefix, _shorten(text, 200))
        elif btype == "tool_use":
            name = getattr(block, "name", "?")
            inp = getattr(block, "input", {})
            log.info("%s tool_use[%s] args=%s", prefix, name, _shorten(inp, 250))
        elif btype == "server_tool_use":
            name = getattr(block, "name", "?")
            inp = getattr(block, "input", {})
            log.info("%s server_tool_use[%s] args=%s", prefix, name, _shorten(inp, 250))
        elif btype == "web_search_tool_result":
            content = getattr(block, "content", None)
            if isinstance(content, list):
                first_url = ""
                if content:
                    first_url = getattr(content[0], "url", "") or getattr(content[0], "title", "")
                log.info("%s web_search_result hits=%d first=%s",
                         prefix, len(content), _shorten(first_url, 100))
            else:
                err = getattr(content, "error_code", None) or "?"
                log.info("%s web_search_result error=%s", prefix, err)
        elif btype == "web_fetch_tool_result":
            content = getattr(block, "content", None)
            url = getattr(content, "url", "") if content else ""
            log.info("%s web_fetch_result url=%s", prefix, _shorten(url, 120))
        elif btype == "bash_code_execution_tool_result":
            inner = getattr(block, "content", None)
            if inner and getattr(inner, "type", None) == "bash_code_execution_result":
                rc = getattr(inner, "return_code", None)
                stdout = getattr(inner, "stdout", "") or ""
                stderr = getattr(inner, "stderr", "") or ""
                log.info("%s bash_exec rc=%s stdout=%s stderr=%s",
                         prefix, rc, _shorten(stdout, 120), _shorten(stderr, 80))
            else:
                err = getattr(inner, "error_code", None) if inner else "?"
                log.info("%s bash_exec error=%s", prefix, err)
        elif btype == "text_editor_code_execution_tool_result":
            log.info("%s text_editor_result", prefix)
        else:
            log.info("%s unknown_block type=%s", prefix, btype)


def _build_proactive_user_text(
    *,
    now_iso: str,
    asker: str,
    asker_profile: str | None,
    trigger_msg: str,
    recent_context: list[dict[str, Any]],
    prior_relevant: list[dict[str, Any]],
    trigger_reason: str | None,
) -> str:
    """Construct the user-content text for proactive mode.

    The trigger message gets its own visually distinct block so it stays
    salient even with ~1k tokens of supporting context around it.
    """
    parts: list[str] = [
        "[PROACTIVE — jumping in without being @-ed]",
        f"[Current time: {now_iso}] [Speaker: {asker}]",
    ]
    if asker_profile:
        parts.append(f"[Speaker profile]\n{asker_profile}")
    parts.append("")
    parts.append("═════════ TRIGGER MESSAGE ═════════")
    parts.append(f"{asker}: {trigger_msg}")
    parts.append("═══════════════════════════════════")
    if recent_context:
        parts.append("")
        parts.append(
            f"[Recent channel context — last {len(recent_context)} msgs, "
            "for thread continuity only]"
        )
        for m in recent_context:
            author = m.get("author", "?")
            ts = m.get("timestamp", "?")
            content = m.get("content", "") or ""
            tag = "[BOT]" if m.get("is_bot_reply") else "[message]"
            parts.append(f"{tag} [{ts}] {author}: {content}")
    if prior_relevant:
        parts.append("")
        parts.append(
            f"[Relevant prior discussions — top-{len(prior_relevant)} "
            "semantic match from channel memory]"
        )
        for r in prior_relevant:
            meta = r.get("metadata") or {}
            rt = meta.get("record_type", "message")
            ts = meta.get("timestamp", "?")
            author = meta.get("author", "?")
            text = _strip_legacy_author_prefix(
                r.get("memory") or r.get("text") or "", author
            )
            parts.append(f"[{rt}] [{ts}] {author}: {text}")
    parts.append("")
    parts.append("[Instructions]")
    parts.append("- 1-2 sentence reply.")
    parts.append("- Add concrete value. Don't summarize the recent context.")
    parts.append("- If you have nothing concrete to add, output exactly: <skip>")
    if trigger_reason:
        parts.append(f"- Classifier triggered because: {trigger_reason}")
    return "\n".join(parts)


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(no relevant memories found)"
    lines = []
    for m in memories:
        meta = m.get("metadata") or {}
        rt = meta.get("record_type", "message")
        pin = "[PINNED] " if meta.get("is_pinned") else ""
        ts = meta.get("timestamp", "")
        author = meta.get("author", "")
        text = _strip_legacy_author_prefix(m.get("memory") or m.get("text") or "", author)
        who = f"{author}: " if author else ""
        lines.append(f"- {pin}[{rt}] [{ts}] {who}{text}")
    return "\n".join(lines)


def _strip_legacy_author_prefix(text: str, author: str) -> str:
    """Legacy records pre-2026-05-14 stored content as '{author}: {body}'.
    New records store body only. This strip keeps formatter output uniform
    across both during the transition window."""
    prefix = f"{author}: "
    if author and text.startswith(prefix):
        return text[len(prefix):]
    return text
