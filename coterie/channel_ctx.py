"""Cross-channel summary loading.

Each channel has its own channel_profile (built by ChannelProfileBuilder).
For replies in channel X, the agent loads:

    X's own profile  +  every public channel's profile

"Public" is declared explicitly via PUBLIC_CHANNELS env. Default empty,
meaning no cross-channel context at all.

Rule:
    private channel -> public channel    : allowed (private reads public)
    public  channel -> public channel    : allowed (siblings)
    any     channel -> private channel   : never  (we just don't load them)

The flow is one-directional from public into anywhere, which mirrors the
"public is shared knowledge, private is read-only by itself" semantics.

Raw messages NEVER cross channel boundaries. Only the synthesized profile
blurb (already ~600 chars, already abstracted, no specific quotes) is
shared. This is the whole reason this layer is safe to share.

Output is a single string ready to drop into the <channel_summary> XML
tag — adapter doesn't need to know the labeling format.
"""

import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger("dc-agent.channel-ctx")

DEFAULT_BUDGET_CHARS = 2000


async def build_channel_summary_block(
    *,
    current_channel_id: str,
    public_channel_ids: list[str] | set[str],
    builder: Any,  # ChannelProfileBuilder; loose-typed to avoid circular import
    label_for_channel: Callable[[str], Awaitable[str]] | None = None,
    budget_chars: int = DEFAULT_BUDGET_CHARS,
) -> str | None:
    """Assemble the cross-channel summary block for the current reply.

    Returns the inner text for `<channel_summary>...</channel_summary>` or
    None if there is nothing to include (no own summary, no publics).

    `label_for_channel(cid)` returns a display name (e.g. "#engineering");
    fall back to the raw ID if no resolver is provided.
    """
    parts: list[str] = []

    own = await _safe_ensure(builder, current_channel_id)
    if own:
        parts.append(f"[this channel]\n{own}")

    # Iterate publics deterministically so output is stable. Skip the
    # current channel if it's listed in publics (no self-dupe).
    publics = sorted(c for c in public_channel_ids if c and c != current_channel_id)
    for cid in publics:
        summary = await _safe_ensure(builder, cid)
        if not summary:
            continue
        label = await _resolve_label(label_for_channel, cid)
        block = f"[public: {label}]\n{summary}"
        # Stop once we'd exceed the budget. Always keep at least the own
        # block — don't truncate it.
        projected = sum(len(p) for p in parts) + len(block) + 2 * len(parts)
        if projected > budget_chars:
            log.info(
                "channel summary budget reached at %d chars; skipping %d remaining publics",
                projected,
                len(publics) - publics.index(cid),
            )
            break
        parts.append(block)

    if not parts:
        return None
    return "\n\n".join(parts)


async def _safe_ensure(builder: Any, channel_id: str) -> str | None:
    try:
        return await builder.ensure(channel_id=channel_id)
    except Exception:
        log.exception("channel_profile_builder.ensure failed for %s", channel_id)
        return None


async def _resolve_label(
    resolver: Callable[[str], Awaitable[str]] | None, cid: str
) -> str:
    if resolver is None:
        return cid
    try:
        return await resolver(cid)
    except Exception:
        log.exception("label_for_channel failed for %s", cid)
        return cid
