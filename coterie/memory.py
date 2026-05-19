"""Mem0 wrapper.

Memories are partitioned by channel (Mem0's `user_id` field), so each channel
has its own pool. Pinned messages get `is_pinned=True` in metadata and a score
boost at retrieval — they're routed through the same hybrid search path
(semantic + keyword + entity) as everything else, not pre-loaded wholesale.

LLM: Anthropic Haiku (fact extraction is cheap, doesn't need Opus quality).
Embedder: local sentence-transformers (no third-party key required).
"""

import asyncio
import os
from datetime import datetime
from typing import Any

from mem0 import Memory


def _build() -> Memory:
    """Build Mem0 with the LLM matching BACKEND env.

    Mem0's internal LLM is used for fact extraction etc.; we keep it on
    the same provider family as the user-facing agent so a single API key
    is enough. Embedder + vector store don't change across backends — the
    embedded vectors are reusable regardless of which agent reads them.
    """
    backend = os.environ.get("BACKEND", "anthropic").lower().strip()
    if backend == "openai":
        llm_config = {
            "provider": "openai",
            "config": {
                "model": "gpt-4o-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        }
    else:
        llm_config = {
            "provider": "anthropic",
            "config": {
                "model": "claude-haiku-4-5",
                "api_key": os.environ["ANTHROPIC_API_KEY"],
            },
        }
    return Memory.from_config({
        "llm": llm_config,
        "embedder": {
            "provider": "huggingface",
            "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        },
        "vector_store": {
            "provider": "chroma",
            "config": {"collection_name": "dc_agent", "path": "./mem0_data"},
        },
    })


class BotMemory:
    def __init__(self) -> None:
        self._m = _build()

    async def add(
        self,
        content: str,
        *,
        channel_id: str,
        author_name: str,
        author_id: str,
        message_id: str,
        timestamp: str,
        is_pinned: bool = False,
        record_type: str = "message",
    ) -> None:
        await asyncio.to_thread(
            self._m.add,
            content,
            user_id=channel_id,
            metadata={
                "record_type": record_type,
                "channel_id": channel_id,
                "message_id": message_id,
                "author": author_name,
                "author_id": author_id,
                "timestamp": timestamp,
                # Numeric mirror for filtering. Chroma's $gte/$lte require
                # int/float; ISO strings get rejected.
                "timestamp_unix": _to_unix(timestamp),
                "is_pinned": is_pinned,
            },
            infer=False,
        )

    async def add_annotation(
        self,
        text: str,
        *,
        channel_id: str,
        references_message_ids: list[str],
        timestamp: str,
    ) -> None:
        """Store an LLM-generated annotation as an independent record.

        Linked to original message(s) via references_message_ids in metadata
        (stored as comma-joined string for Chroma compatibility). Annotations
        are embedded separately, so they expand the retrieval surface — the
        agent may match either the raw message or its annotation.
        """
        await asyncio.to_thread(
            self._m.add,
            text,
            user_id=channel_id,
            metadata={
                "record_type": "annotation",
                "channel_id": channel_id,
                "references_message_ids": ",".join(references_message_ids),
                "timestamp": timestamp,
                "timestamp_unix": _to_unix(timestamp),
                "author": "_annotator",
                "is_pinned": False,
            },
            infer=False,
        )

    async def add_reaction(
        self,
        *,
        channel_id: str,
        reactor_name: str,
        reactor_id: str,
        emoji: str,
        target_message_id: str,
        target_excerpt: str,
        timestamp: str,
    ) -> None:
        """Record a human reaction emoji left on one of the bot's messages.

        Stored as record_type='reaction' — an implicit-feedback signal layer
        the profile builder can consult ("Alice 👎'd two of my last replies").
        Excluded from list_window so it never lands in digest output.
        """
        excerpt = (target_excerpt or "").strip().replace("\n", " ")
        if len(excerpt) > 200:
            excerpt = excerpt[:200] + "..."
        content = (
            f"{reactor_name} reacted {emoji} to bot reply: {excerpt}"
            if excerpt
            else f"{reactor_name} reacted {emoji} to bot reply"
        )
        await asyncio.to_thread(
            self._m.add,
            content,
            user_id=channel_id,
            metadata={
                "record_type": "reaction",
                "channel_id": channel_id,
                "message_id": target_message_id,
                "author": reactor_name,
                "author_id": reactor_id,
                "emoji": emoji,
                "timestamp": timestamp,
                "timestamp_unix": _to_unix(timestamp),
                "is_pinned": False,
            },
            infer=False,
        )

    async def list_by_author(
        self,
        *,
        channel_id: str,
        author: str,
        since: str | float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """All real messages from one author in this channel.

        Filtered to record_type=message so we don't feed annotations or
        bot replies into a profile builder. `since` (ISO string or epoch
        float) further restricts to messages at/after that time — used by
        the refresh tick to count new messages since last build.
        """
        filt: dict[str, Any] = {
            "user_id": channel_id,
            "author": author,
            "record_type": "message",
        }
        if since is not None:
            filt["timestamp_unix"] = {"gte": _to_unix(since)}
        hits = await asyncio.to_thread(
            self._m.get_all,
            filters=filt,
            limit=limit,
        )
        return _unwrap(hits)

    async def list_profiles(
        self, *, channel_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        hits = await asyncio.to_thread(
            self._m.get_all,
            filters={"user_id": channel_id, "record_type": "profile"},
            limit=limit,
        )
        return _unwrap(hits)

    async def get_profile(
        self, *, channel_id: str, author: str
    ) -> dict[str, Any] | None:
        hits = await asyncio.to_thread(
            self._m.get_all,
            filters={
                "user_id": channel_id,
                "record_type": "profile",
                "author": author,
            },
            limit=1,
        )
        results = _unwrap(hits)
        return results[0] if results else None

    async def upsert_profile(
        self,
        *,
        channel_id: str,
        author: str,
        text: str,
        built_from_n_msgs: int = 0,
    ) -> None:
        """Replace existing profile or insert new. We delete + add rather than
        use Mem0.update because update can re-run fact extraction; replacing
        is more predictable for this small structured blurb."""
        existing = await self.get_profile(channel_id=channel_id, author=author)
        if existing:
            await asyncio.to_thread(self._m.delete, memory_id=existing["id"])
        now_iso = datetime.now().astimezone().isoformat()
        now_unix = _to_unix(now_iso)
        await asyncio.to_thread(
            self._m.add,
            text,
            user_id=channel_id,
            metadata={
                "record_type": "profile",
                "channel_id": channel_id,
                "author": author,
                "timestamp": now_iso,
                "timestamp_unix": now_unix,
                "last_built_unix": now_unix,
                "built_from_n_msgs": built_from_n_msgs,
                "is_pinned": False,
            },
            infer=False,
        )

    async def mark_pinned(self, *, channel_id: str, message_id: str) -> None:
        hits = await asyncio.to_thread(
            self._m.get_all,
            filters={"user_id": channel_id, "message_id": message_id},
            limit=1,
        )
        results = _unwrap(hits)
        if not results:
            return
        await asyncio.to_thread(
            self._m.update,
            memory_id=results[0]["id"],
            metadata={**results[0].get("metadata", {}), "is_pinned": True},
        )

    async def list_window(
        self,
        *,
        channel_id: str,
        since: str,
        until: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return every NON-PROFILE record in [since, until] for this channel.

        Unlike search(), no semantic query — pure metadata filter via Mem0's
        get_all. Used by digest where we want comprehensive coverage of a
        time window, not the top-k semantically-relevant hits.

        Profile records are excluded — they hold personal user info that
        must not leak into digest output. Reaction records are excluded too
        — they're a signal layer for profile builds, not chat substance.
        """
        base_filter: dict[str, Any] = {
            "user_id": channel_id,
            "timestamp_unix": {
                "gte": _to_unix(since),
                "lte": _to_unix(until),
            },
        }
        hits = await asyncio.to_thread(
            self._m.get_all,
            filters=base_filter,
            limit=limit,
        )
        results = _unwrap(hits)
        return [
            r for r in results
            if (r.get("metadata") or {}).get("record_type") not in ("profile", "reaction")
        ]

    async def search(
        self,
        query: str,
        *,
        channel_id: str,
        author: str | None = None,
        since: str | None = None,
        until: str | None = None,
        record_type: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        base_filter: dict[str, Any] = {"user_id": channel_id}
        if author:
            base_filter["author"] = author
        if record_type:
            base_filter["record_type"] = record_type
        ts_range: dict[str, float] = {}
        if since:
            ts_range["gte"] = _to_unix(since)
        if until:
            ts_range["lte"] = _to_unix(until)
        if ts_range:
            base_filter["timestamp_unix"] = ts_range

        regular = await asyncio.to_thread(
            self._m.search,
            query=query,
            filters=base_filter,
            limit=limit,
        )
        pinned = await asyncio.to_thread(
            self._m.search,
            query=query,
            filters={**base_filter, "is_pinned": True},
            limit=max(2, limit // 2),
        )
        merged: dict[str, dict[str, Any]] = {}
        for r in _unwrap(regular):
            merged[r["id"]] = r
        for r in _unwrap(pinned):
            existing = merged.get(r["id"])
            score = r.get("score", 0) * 1.3
            if existing is None or score > existing.get("score", 0):
                merged[r["id"]] = {**r, "score": score}
        ranked = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        return ranked[:limit]


def _to_unix(iso_or_unix: str | float | int) -> float:
    if isinstance(iso_or_unix, (int, float)):
        return float(iso_or_unix)
    return datetime.fromisoformat(iso_or_unix).timestamp()


def _unwrap(res: Any) -> list[dict[str, Any]]:
    if isinstance(res, dict) and "results" in res:
        return res["results"]
    if isinstance(res, list):
        return res
    return []
