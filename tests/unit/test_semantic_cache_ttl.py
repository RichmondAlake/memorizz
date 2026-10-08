"""Semantic-cache TTL, persistence and hit-count guarantees on a real provider.

Runs the cache against the filesystem provider on a temporary directory with
a stub embedding manager (no network), mirroring
``test_semantic_cache_governance_050``.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.memory_provider.base import MemoryProvider
from memorizz.short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig


class _EmbeddingManager:
    def get_embedding(self, text: str):
        normalized = str(text).lower()
        return [1.0, 0.0] if "inventory" in normalized else [0.0, 1.0]


def _metadata(version: str = "inventory-v1"):
    return {
        "domain": "inventory",
        "tags": ["warehouse-1"],
        "fingerprints": {
            "model": "model-v1",
            "prompt": "prompt-v1",
            "tool_schema": "tools-v1",
            "data_version": version,
        },
        "admission": {"read_only": True, "deterministic": True},
    }


def _provider(tmp_path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "fs-memory",
            embedding_provider=_EmbeddingManager(),
            lazy_vector_indexes=True,
        )
    )


def _cache(provider, ttl_hours: float = 1.0) -> SemanticCache:
    return SemanticCache(
        config=SemanticCacheConfig(
            enable_memory_provider_sync=True,
            ttl_hours=ttl_hours,
        ),
        memory_provider=provider,
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )


def _set_at(monkeypatch, cache, when: float, query: str, response: str, **kwargs):
    """``cache.set`` with the clock pinned, so the row is created in the past."""
    with monkeypatch.context() as patched:
        patched.setattr(time, "time", lambda: when)
        return cache.set(query, response, **kwargs)


def _rows(provider):
    return provider.list_all(MemoryType.SEMANTIC_CACHE)


@pytest.mark.unit
def test_fresh_entry_hits_in_memory_and_from_the_provider(tmp_path):
    provider = _provider(tmp_path)
    cache = _cache(provider)
    assert cache.set(
        "inventory for SKU-7", "12 units", user_id="alice", metadata=_metadata()
    )
    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=_metadata())
        == "12 units"
    )

    reloaded = _cache(provider)
    reloaded.cache.clear()  # no exact in-memory key: must come from the provider
    assert (
        reloaded.get("inventory levels", user_id="alice", lookup_metadata=_metadata())
        == "12 units"
    )
    assert reloaded.statistics()["last_hit"]["similarity"] == pytest.approx(1.0)


@pytest.mark.unit
def test_expired_entry_misses_and_is_swept_from_memory(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    cache = _cache(provider, ttl_hours=1.0)
    assert _set_at(
        monkeypatch,
        cache,
        time.time() - 7200,
        "inventory for SKU-7",
        "stale",
        user_id="alice",
        metadata=_metadata(),
    )
    assert len(cache.cache) == 1

    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=_metadata())
        is None
    )
    assert cache.cache == {}
    assert cache.statistics()["misses"] == 1
    assert cache.statistics()["evictions"] == 1


@pytest.mark.unit
def test_preload_skips_expired_rows_and_purges_them(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    seed = _cache(provider, ttl_hours=1.0)
    assert _set_at(
        monkeypatch,
        seed,
        time.time() - 7200,
        "inventory for SKU-7",
        "stale",
        user_id="alice",
        metadata=_metadata(),
    )
    assert seed.set(
        "inventory for SKU-9", "fresh", user_id="alice", metadata=_metadata()
    )
    assert len(_rows(provider)) == 2

    reloaded = _cache(provider, ttl_hours=1.0)
    assert [entry.response for entry in reloaded.cache.values()] == ["fresh"]
    assert [row["response"] for row in _rows(provider)] == ["fresh"]


@pytest.mark.unit
def test_recaching_an_expired_query_upserts_a_single_row(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    cache = _cache(provider, ttl_hours=1.0)
    assert _set_at(
        monkeypatch,
        cache,
        time.time() - 7200,
        "inventory for SKU-7",
        "stale",
        user_id="alice",
        metadata=_metadata(),
    )
    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=_metadata())
        is None
    )
    assert cache.set(
        "inventory for SKU-7", "fresh", user_id="alice", metadata=_metadata()
    )

    rows = _rows(provider)
    assert len(rows) == 1
    assert rows[0]["response"] == "fresh"
    assert rows[0]["_id"].startswith("sc-")
    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=_metadata())
        == "fresh"
    )

    # The record id derives from the cache key, so another instance re-caching
    # the same scoped query also lands on the same row.
    other = _cache(provider, ttl_hours=1.0)
    assert other.set(
        "inventory for SKU-7", "fresher", user_id="alice", metadata=_metadata()
    )
    rows = _rows(provider)
    assert len(rows) == 1
    assert rows[0]["response"] == "fresher"


@pytest.mark.unit
def test_purge_expired_semantic_cache_removes_only_expired_rows(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    cache = _cache(provider, ttl_hours=1.0)
    assert _set_at(
        monkeypatch,
        cache,
        time.time() - 7200,
        "inventory for SKU-7",
        "stale",
        user_id="alice",
        metadata=_metadata(),
    )
    assert cache.set(
        "inventory for SKU-9", "fresh", user_id="alice", metadata=_metadata()
    )

    assert provider.purge_expired_semantic_cache() == 1
    assert [row["response"] for row in _rows(provider)] == ["fresh"]
    assert provider.purge_expired_semantic_cache() == 0
    assert provider.purge_expired_semantic_cache(now=time.time() + 10 * 3600) == 1
    assert _rows(provider) == []


@pytest.mark.unit
def test_periodic_purge_runs_from_set(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    cache = _cache(provider, ttl_hours=1.0)
    assert _set_at(
        monkeypatch,
        cache,
        time.time() - 7200,
        "inventory for SKU-0",
        "stale",
        user_id="alice",
        metadata=_metadata(),
    )
    for i in range(1, 50):
        assert cache.set(
            f"inventory for SKU-{i}", "fresh", user_id="alice", metadata=_metadata()
        )
    assert cache.statistics()["writes"] == 50
    assert all(row["response"] == "fresh" for row in _rows(provider))
    assert len(_rows(provider)) == 49


@pytest.mark.unit
def test_generic_purge_tolerates_datetime_iso_epoch_and_missing_values():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    past = now - timedelta(hours=1)
    future = now + timedelta(hours=1)

    class Rows:
        def __init__(self):
            self.rows = [
                {"_id": "dt-expired", "expires_at": past},
                # Naive values are UTC (that is how pymongo hands them back).
                {"_id": "naive-expired", "expires_at": past.replace(tzinfo=None)},
                {"_id": "iso-expired", "expires_at": past.isoformat()},
                {
                    "_id": "naive-iso-expired",
                    "expires_at": past.replace(tzinfo=None).isoformat(),
                },
                {"_id": "epoch-expired", "expires_at": past.timestamp()},
                {"_id": "dt-live", "expires_at": future},
                {"_id": "naive-live", "expires_at": future.replace(tzinfo=None)},
                {"_id": "iso-live", "expires_at": future.isoformat()},
                {"_id": "epoch-live", "expires_at": int(future.timestamp())},
                {"_id": "no-expiry"},
                {"_id": "garbage", "expires_at": "not a date"},
                {"id": "id-only-expired", "expires_at": past},
            ]
            self.deleted = []

        def list_all(self, memory_store_type=None):
            return list(self.rows)

        def delete_by_id(self, record_id, memory_store_type=None):
            self.deleted.append(record_id)
            return True

    expected = [
        "dt-expired",
        "naive-expired",
        "iso-expired",
        "naive-iso-expired",
        "epoch-expired",
        "id-only-expired",
    ]
    provider = Rows()
    assert MemoryProvider.purge_expired_semantic_cache(provider, now=now) == 6
    assert provider.deleted == expected

    # ``now`` may also be naive (read as UTC) or epoch seconds.
    provider = Rows()
    assert (
        MemoryProvider.purge_expired_semantic_cache(
            provider, now=now.replace(tzinfo=None)
        )
        == 6
    )
    provider = Rows()
    assert (
        MemoryProvider.purge_expired_semantic_cache(provider, now=now.timestamp()) == 6
    )


@pytest.mark.unit
def test_persisted_timestamps_are_timezone_aware_utc(tmp_path):
    class CapturingProvider:
        def __init__(self):
            self.stored = []

        def retrieve_by_query(self, **_kwargs):
            return []

        def store(self, **kwargs):
            self.stored.append(kwargs)
            return "stored"

    provider = CapturingProvider()
    cache = SemanticCache(
        config=SemanticCacheConfig(enable_memory_provider_sync=True, ttl_hours=2),
        memory_provider=provider,
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
    )
    assert cache.set("inventory for SKU-7", "12 units", user_id="alice")
    data = provider.stored[0]["data"]
    assert data["created_at"].tzinfo == timezone.utc
    assert data["expires_at"].tzinfo == timezone.utc
    assert data["expires_at"] - data["created_at"] == timedelta(hours=2)
    assert data["_id"].startswith("sc-")


@pytest.mark.unit
def test_hit_count_increments_across_hits_and_instances(tmp_path):
    provider = _provider(tmp_path)
    cache = _cache(provider)
    assert cache.set(
        "inventory for SKU-7", "12 units", user_id="alice", metadata=_metadata()
    )
    for _ in range(3):
        assert (
            cache.get(
                "inventory for SKU-7", user_id="alice", lookup_metadata=_metadata()
            )
            == "12 units"
        )
    row = _rows(provider)[0]
    assert row["hit_count"] == 3
    assert row["usage_count"] == 3
    assert cache.last_inspection.hit_count == 3

    reloaded = _cache(provider)  # preload must read hit_count, not usage_count
    assert (
        reloaded.get(
            "inventory for SKU-7", user_id="alice", lookup_metadata=_metadata()
        )
        == "12 units"
    )
    assert _rows(provider)[0]["hit_count"] == 4

    reloaded.cache.clear()  # provider-path hit
    assert (
        reloaded.get("inventory levels", user_id="alice", lookup_metadata=_metadata())
        == "12 units"
    )
    assert _rows(provider)[0]["hit_count"] == 5
    assert _rows(provider)[0]["usage_count"] == 5


def _row(timestamp: float):
    return {
        "_id": "row-1",
        "query": "inventory for SKU-7",
        "response": "12 units",
        "embedding": [1.0, 0.0],
        "timestamp": timestamp,
        "agent_id": "agent-1",
        "memory_id": "memory-1",
        "user_id": "alice",
        "hit_count": 2,
        "metadata": _metadata(),
    }


class _ProjectingProvider:
    """Returns embeddings only when asked, like the MongoDB/Oracle providers."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def retrieve_by_query(
        self, query, memory_store_type=None, limit=1, include_embedding=False, **_
    ):
        self.calls.append(include_embedding)
        if include_embedding:
            return [dict(row) for row in self.rows]
        return [{k: v for k, v in row.items() if k != "embedding"} for row in self.rows]

    def store(self, **_kwargs):
        return "stored"


class _StrictProvider:
    """No ``include_embedding`` keyword at all, and no embeddings projected."""

    def __init__(self, rows):
        self.rows = rows

    def retrieve_by_query(self, query, memory_store_type, limit):
        return [{k: v for k, v in row.items() if k != "embedding"} for row in self.rows]

    def store(self, **_kwargs):
        return "stored"


def _stub_cache(provider) -> SemanticCache:
    return SemanticCache(
        config=SemanticCacheConfig(enable_memory_provider_sync=True),
        memory_provider=provider,
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )


@pytest.mark.unit
def test_preload_requests_embeddings_when_the_provider_can_project_them():
    provider = _ProjectingProvider([_row(time.time())])
    cache = _stub_cache(provider)
    assert provider.calls == [True]
    entries = list(cache.cache.values())
    assert [entry.embedding for entry in entries] == [[1.0, 0.0]]
    assert [entry.usage_count for entry in entries] == [2]


@pytest.mark.unit
def test_preload_still_loads_rows_from_providers_without_embeddings():
    cache = _stub_cache(_StrictProvider([_row(time.time())]))
    entries = list(cache.cache.values())
    assert len(entries) == 1
    assert entries[0].embedding == []
    assert (
        cache.get("inventory for SKU-7", user_id="alice", lookup_metadata=_metadata())
        == "12 units"
    )
