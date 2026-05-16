"""One-shot: rebuild every (channel, author) profile from full Mem0 history.

Use cases:
- After deploying a new profile schema (added a section, changed prompt) —
  this propagates the change to every existing user without waiting for
  the natural 10-msg / 24h refresh threshold.
- Initializing profiles for users the bot has never been @-ed by — the
  lazy build only triggers on @, so passive listeners stay unprofiled
  until proactive evaluates them. This script bootstraps them all.

Run:
    python init_profiles.py                       # uses DIGEST_CHANNELS env
    python init_profiles.py <channel_id> [...]    # explicit channel list

Safe to run while the bot is up. Chroma serializes writes via SQLite, so
worst case we wait a few hundred ms per write — no corruption.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from memory import BotMemory
from user_profile import MIN_MESSAGES, ProfileBuilder

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("init-profiles")


async def main() -> None:
    channel_ids = sys.argv[1:]
    if not channel_ids:
        channel_ids = [
            c.strip() for c in os.environ.get("DIGEST_CHANNELS", "").split(",")
            if c.strip()
        ]
    if not channel_ids:
        print(
            "Usage: python init_profiles.py <channel_id> [<channel_id> ...]"
            "\n(or set DIGEST_CHANNELS env)"
        )
        sys.exit(1)

    m = BotMemory()
    pb = ProfileBuilder(memory=m)

    far_past = "2020-01-01T00:00:00+00:00"
    far_future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    total_authors = 0
    total_built = 0
    total_skipped = 0

    for cid in channel_ids:
        print(f"\n=== Channel {cid} ===")
        records = await m.list_window(
            channel_id=cid, since=far_past, until=far_future, limit=2000,
        )
        # Group by author; only real human messages (skip agent_reply / annotation).
        authors: dict[str, int] = {}
        for r in records:
            meta = r.get("metadata") or {}
            if meta.get("record_type") != "message":
                continue
            author = meta.get("author")
            if not author or author == "_annotator":
                continue
            authors[author] = authors.get(author, 0) + 1

        print(f"  {len(authors)} distinct authors, {sum(authors.values())} msgs total")
        total_authors += len(authors)

        for author, n in sorted(authors.items(), key=lambda x: -x[1]):
            if n < MIN_MESSAGES:
                print(f"  - {author}: {n} msgs (< {MIN_MESSAGES}, skipped)")
                total_skipped += 1
                continue
            try:
                text = await pb.rebuild(channel_id=cid, author=author)
            except Exception as e:
                print(f"  - {author}: {n} msgs → FAILED: {e}")
                total_skipped += 1
                continue
            if text:
                first_line = text.split("\n")[0] if text else "(empty)"
                print(f"  - {author}: {n} msgs → built")
                print(f"      {first_line}")
                total_built += 1
            else:
                print(f"  - {author}: {n} msgs → no output (skipped)")
                total_skipped += 1

    print(f"\n=== Summary ===")
    print(f"Total authors seen: {total_authors}")
    print(f"Profiles built:     {total_built}")
    print(f"Skipped:            {total_skipped}")


if __name__ == "__main__":
    asyncio.run(main())
