"""Operational semantic-cache guarantees added for MemoRizz 0.5."""

from __future__ import annotations

import time

import pytest

from memorizz.enums import SemanticCacheScope
from memorizz.memagent.managers.cache_manager import CacheManager
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


@pytest.mark.unit
def test_string_scope_is_normalized_before_persistent_filtering():
    config = SemanticCacheConfig(scope="local")
    assert config.scope == SemanticCacheScope.LOCAL


@pytest.mark.unit
def test_legacy_session_scope_preserves_session_isolation():
    config = SemanticCacheConfig(scope="session")
    assert config.scope == SemanticCacheScope.LOCAL
    assert config.enable_session_scoping is True


@pytest.mark.unit
def test_cache_stats_tenant_keys_fingerprints_freshness_and_provenance():
    cache = SemanticCache(
        config=SemanticCacheConfig(
            similarity_threshold=0.9,
            enable_memory_provider_sync=False,
            freshness_by_domain={"inventory": 60.0},
        ),
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )
    metadata = _metadata()
    assert cache.set(
        "inventory for SKU-7",
        "12 units",
        user_id="alice",
        metadata=metadata,
    )
    alice_key = next(iter(cache.cache))
    assert alice_key != cache._generate_cache_key("inventory for SKU-7", user_id="bob")

    assert (
        cache.get(
            "inventory for SKU-7",
            user_id="bob",
            lookup_metadata=metadata,
        )
        is None
    )
    assert (
        cache.get(
            "inventory for SKU-7",
            user_id="alice",
            lookup_metadata=_metadata("inventory-v2"),
        )
        is None
    )
    assert (
        cache.get(
            "inventory for SKU-7",
            user_id="alice",
            lookup_metadata=metadata,
        )
        == "12 units"
    )
    stats = cache.statistics()
    assert stats["writes"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 1
    assert stats["last_hit"]["user_id"] == "alice"
    assert stats["last_hit"]["similarity"] == pytest.approx(1.0)
    assert stats["last_hit"]["age_seconds"] >= 0

    # Similarity is intentionally insufficient once domain freshness expires.
    cache.cache[alice_key].timestamp = time.time() - 61
    assert (
        cache.get(
            "inventory for SKU-7",
            user_id="alice",
            lookup_metadata=metadata,
        )
        is None
    )
    cache.get("inventory", user_id="alice", bypass_reason="side_effecting_tool")
    stats = cache.statistics()
    assert stats["misses"] == 3
    assert stats["bypasses"] == 1
    assert stats["bypass_reasons"]["side_effecting_tool"] == 1


@pytest.mark.unit
def test_cache_admission_and_domain_version_invalidation_are_real_operations():
    manager = CacheManager(enabled=False)
    manager.enabled = True
    manager.cache_instance = SemanticCache(
        config=SemanticCacheConfig(enable_memory_provider_sync=False),
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
    )

    assert (
        manager.cache_response(
            "inventory mutation",
            "changed",
            user_id="alice",
            deterministic=True,
            read_only=False,
            metadata=_metadata(),
        )
        is False
    )
    assert manager.get_statistics()["bypasses"] == 1
    assert manager.get_statistics()["writes"] == 0

    assert (
        manager.cache_response(
            "inventory query",
            "stable response",
            user_id="alice",
            metadata=_metadata("v3"),
        )
        is True
    )
    assert manager.get_statistics()["writes"] == 1
    assert manager.invalidate(data_version="v3") == 1
    assert manager.get_statistics()["size"] == 0
    assert manager.get_statistics()["evictions"] == 1

    assert (
        manager.cache_response(
            "inventory query",
            "new response",
            user_id="alice",
            metadata=_metadata("v4"),
        )
        is True
    )
    assert manager.invalidate(domains=["inventory"]) == 1
    assert manager.get_statistics()["size"] == 0


@pytest.mark.unit
def test_exact_repeat_bypasses_vector_score_rounding_at_strict_threshold():
    cache = SemanticCache(
        config=SemanticCacheConfig(
            similarity_threshold=1.0,
            enable_memory_provider_sync=False,
        ),
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )
    assert cache.set(
        "exact question",
        "stable answer",
        session_id="thread-1",
        user_id="alice",
        metadata=_metadata(),
    )
    cache._cosine_similarity = lambda *_args: 0.999999  # type: ignore[method-assign]

    assert (
        cache.get(
            "exact question",
            session_id="thread-1",
            user_id="alice",
            lookup_metadata=_metadata(),
        )
        == "stable answer"
    )
    assert cache.statistics()["last_hit"]["similarity"] == 1.0


@pytest.mark.unit
def test_persistent_domain_invalidation_is_provider_neutral_and_scoped():
    class PersistentRows:
        def __init__(self):
            self.rows = [
                {
                    "_id": "alice-inventory",
                    "agent_id": "agent-1",
                    "memory_id": "memory-1",
                    "metadata": _metadata("v1"),
                },
                {
                    "_id": "other-agent",
                    "agent_id": "agent-2",
                    "memory_id": "memory-1",
                    "metadata": _metadata("v1"),
                },
                {
                    "_id": "other-domain",
                    "agent_id": "agent-1",
                    "memory_id": "memory-1",
                    "metadata": {
                        "domain": "catalog",
                        "fingerprints": {"data_version": "v2"},
                    },
                },
            ]
            self.deleted = []

        def list_all(self, memory_store_type=None):
            return list(self.rows)

        def delete_by_id(self, record_id, memory_store_type=None):
            self.deleted.append(record_id)
            return True

    provider = PersistentRows()
    removed = MemoryProvider.invalidate_semantic_cache(
        provider,
        agent_id="agent-1",
        memory_id="memory-1",
        domains=["inventory"],
    )

    assert removed == 1
    assert provider.deleted == ["alice-inventory"]


@pytest.mark.unit
def test_persistent_cache_projection_includes_tenant_scope_expiry_and_metadata():
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
        config=SemanticCacheConfig(
            enable_memory_provider_sync=True,
            similarity_threshold=0.93,
            ttl_hours=2,
        ),
        memory_provider=provider,
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )

    assert cache.set(
        "inventory for SKU-7",
        "12 units",
        session_id="session-1",
        user_id="alice",
        metadata=_metadata("inventory-v8"),
    )
    assert len(provider.stored) == 1
    stored = provider.stored[0]
    assert stored["memory_store_type"].value == "semantic_cache"
    data = stored["data"]
    assert data["agent_id"] == "agent-1"
    assert data["memory_id"] == "memory-1"
    assert data["session_id"] == "session-1"
    assert data["user_id"] == "alice"
    assert data["scope"] == "local"
    assert data["similarity_threshold"] == pytest.approx(0.93)
    assert data["hit_count"] == 0
    assert data["expires_at"] > data["created_at"]
    assert data["metadata"]["fingerprints"]["data_version"] == "inventory-v8"
