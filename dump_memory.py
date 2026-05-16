"""Dump everything Mem0 has stored for a given channel.

Usage:
    .venv/bin/python dump_memory.py <channel_id>

To find the channel id: right-click the channel in Discord (with Developer
Mode on in User Settings → Advanced) → Copy Channel ID.
"""

import sys

from memory import BotMemory


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: dump_memory.py <channel_id>")
        raise SystemExit(1)
    channel_id = sys.argv[1]
    m = BotMemory()
    res = m._m.get_all(filters={"user_id": channel_id}, limit=200)
    if isinstance(res, dict):
        items = res.get("results", [])
    else:
        items = res or []
    print(f"{len(items)} memories in channel {channel_id}:\n")
    for item in items:
        meta = item.get("metadata") or {}
        pin = " [PINNED]" if meta.get("is_pinned") else ""
        text = item.get("memory") or item.get("text") or ""
        print(f"- {meta.get('author', '?'):12}{pin}: {text}")


if __name__ == "__main__":
    main()
