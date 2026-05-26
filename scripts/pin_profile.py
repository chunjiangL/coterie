"""Pin a directive into a (channel, user) profile.

The directive is stored as a `### Pinned directive` section inside the
user's profile blurb. The ProfileBuilder system prompt is instructed to
copy this section verbatim on every rebuild, so the directive persists
across the normal 10-min refresh cycle.

This is an operator-set instruction surface — it bypasses the normal
"profile is reference data, not instructions" framing the agent uses.
Use sparingly.

Usage:
    python scripts/pin_profile.py <channel_id> <author> "<directive>"
    python scripts/pin_profile.py --clear <channel_id> <author>

If you want to pin across all channels a user is active in, pass
"ALL" as channel_id and the script will iterate channels where the user
has an existing profile (you need a comma-separated CHANNEL_LIST env or
edit the script).
"""

import argparse
import asyncio
import logging
import os
import re
from typing import Iterable

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("pin-profile")

PINNED_HEADING = "### Pinned directive"
_PINNED_RE = re.compile(
    rf"\n*{re.escape(PINNED_HEADING)}\n.*?(?=\n###|\Z)",
    re.DOTALL,
)


def _strip_existing_pin(text: str) -> str:
    return _PINNED_RE.sub("", text).rstrip() + "\n"


def _attach_pin(text: str, directive: str) -> str:
    """Replace any existing Pinned directive section; append a new one."""
    base = _strip_existing_pin(text).rstrip()
    return f"{base}\n\n{PINNED_HEADING}\n{directive.strip()}\n"


async def _iter_channels(memory, author: str, channels: Iterable[str]):
    for cid in channels:
        existing = await memory.get_profile(channel_id=cid, author=author)
        if existing:
            yield cid, existing


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("channel_id", help="Channel ID, or 'ALL' to walk CHANNEL_LIST")
    parser.add_argument("author", help="Display name on the platform")
    parser.add_argument(
        "directive",
        nargs="?",
        default=None,
        help="The directive text. Omit when using --clear.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove any existing Pinned directive section instead of setting one.",
    )
    args = parser.parse_args()

    if not args.clear and not args.directive:
        parser.error("directive is required unless --clear is passed")

    from coterie.memory import BotMemory

    if args.channel_id == "ALL":
        raw = os.environ.get("CHANNEL_LIST", "")
        channels = [c.strip() for c in raw.split(",") if c.strip()]
        if not channels:
            parser.error("ALL mode needs CHANNEL_LIST env (comma-separated IDs)")
    else:
        channels = [args.channel_id]

    memory = BotMemory()
    touched = 0
    async for cid, existing in _iter_channels(memory, args.author, channels):
        old_text = (existing.get("memory") or existing.get("text") or "").strip()
        meta = existing.get("metadata") or {}
        built_from = int(meta.get("built_from_n_msgs") or 0)
        if args.clear:
            new_text = _strip_existing_pin(old_text).rstrip()
        else:
            new_text = _attach_pin(old_text, args.directive)
        if new_text.strip() == old_text.strip():
            log.info("channel=%s: no change needed", cid)
            continue
        await memory.upsert_profile(
            channel_id=cid,
            author=args.author,
            text=new_text,
            built_from_n_msgs=built_from,
        )
        log.info("channel=%s author=%s pin %s",
                 cid, args.author, "cleared" if args.clear else "set")
        touched += 1
    if touched == 0:
        log.warning("no matching profile found in %d channel(s)", len(channels))


if __name__ == "__main__":
    asyncio.run(main())
