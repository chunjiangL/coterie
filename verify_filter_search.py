"""Standalone test for Branch A: filter-by-author + filter-by-time search.

Inserts 4 synthetic memories into a TEST channel, then runs queries with
different filter combinations and prints results. No Discord, no Claude — just
exercises the Mem0 filter path directly so you can see whether timestamp
range and author exact-match filters actually work end-to-end.

Run:
    .venv/bin/python verify_filter_search.py
"""

import asyncio
import os
import uuid

from memory import BotMemory

# Fresh partition every run so we don't accumulate duplicate test memories
TEST_CHANNEL = f"verify-{uuid.uuid4().hex[:8]}"

SEED = [
    # (author, timestamp ISO, content)
    ("alice", "2026-05-01T10:00:00+00:00", "macOS 26 已经升级好了，体验不错"),
    ("alice", "2026-05-10T14:30:00+00:00", "python 3.14 比 3.12 快了大概 15%"),
    ("bob",   "2026-05-01T11:00:00+00:00", "fly.io 部署完了，免费档够用"),
    ("bob",   "2026-05-11T09:00:00+00:00", "今天读了一篇关于 RAG 的论文"),
]


async def main() -> None:
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-not-used-since-infer-is-false")
    m = BotMemory()

    print("=" * 60)
    print(f"Seeding {len(SEED)} memories into channel {TEST_CHANNEL}")
    print("=" * 60)
    for i, (author, ts, content) in enumerate(SEED):
        await m.add(
            content=content,
            channel_id=TEST_CHANNEL,
            author_name=author,
            author_id=f"fake_id_{author}",
            message_id=f"msg_{i}_{ts}",
            timestamp=ts,
        )
        print(f"  + [{ts}] {author}: {content}")

    cases: list[tuple[str, dict]] = [
        ("no filter — should return all relevant", {}),
        ("author=alice — only alice's messages", {"author": "alice"}),
        ("author=bob — only bob's messages", {"author": "bob"}),
        ("since=2026-05-08 — only messages after May 8", {"since": "2026-05-08T00:00:00+00:00"}),
        ("until=2026-05-05 — only messages before May 5", {"until": "2026-05-05T00:00:00+00:00"}),
        ("alice + since May 8 — should be 1 result", {
            "author": "alice", "since": "2026-05-08T00:00:00+00:00"
        }),
    ]

    for label, filters in cases:
        print()
        print("-" * 60)
        print(f"QUERY: '操作系统'  filters={filters}")
        print(f"CASE: {label}")
        print("-" * 60)
        results = await m.search(query="操作系统 或 编程语言 或 部署", channel_id=TEST_CHANNEL, limit=10, **filters)
        if not results:
            print("  (no results)")
            continue
        for r in results:
            meta = r.get("metadata") or {}
            score = r.get("score", 0)
            text = r.get("memory") or r.get("text") or ""
            ts = meta.get("timestamp", "?")
            who = meta.get("author", "?")
            print(f"  [{score:.3f}] [{ts}] {who}: {text}")


if __name__ == "__main__":
    asyncio.run(main())
