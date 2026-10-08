# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Regression tests for MongoDB provider defects, run against mongomock.

Each test pins one previously-broken contract:

* ``delete_memagent(cascade=True)`` must delete the agent row *and* every
  memory-unit row for the agent's ``memory_ids`` across all stores.
* ``retrieve_by_id`` / ``update_by_id`` / ``delete_by_id`` must resolve the
  string ``_id`` values the provider itself writes (toolbox rows, trace
  bundles, caller-supplied ids), not only ObjectIds.
* Update helpers report "row found" (``matched_count``) rather than "row
  changed", and ``delete_all`` reports success on an empty store, matching
  the filesystem and Notion providers.
* Semantic cache: native expiry purge, a TTL index on ``expires_at``, and
  ``include_embedding=True`` honoured on the dict-query preload path.
* Shared-memory dict queries honour ``user_id`` like the filesystem provider.
* ``list_summaries`` keeps tenant scoping.
"""

from datetime import datetime, timedelta, timezone

import pytest

mongomock = pytest.importorskip("mongomock")
pytest.importorskip("pymongo")

from bson import ObjectId  # noqa: E402

from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.memory_provider.mongodb.provider import MongoDBProvider  # noqa: E402

# The attribute names ``MongoDBProvider.__init__`` binds for each store, so
# the fixture mirrors a real provider without opening a connection.
_COLLECTION_ATTRIBUTES = {
    MemoryType.PERSONAS: "persona_collection",
    MemoryType.TOOLBOX: "toolbox_collection",
    MemoryType.SKILLBOX: "skillbox_collection",
    MemoryType.SHORT_TERM_MEMORY: "short_term_memory_collection",
    MemoryType.KNOWLEDGE_BASE: "knowledge_base_collection",
    MemoryType.CONVERSATION_MEMORY: "conversation_memory_collection",
    MemoryType.WORKFLOW_MEMORY: "workflow_memory_collection",
    MemoryType.ENTITY_MEMORY: "entity_memory_collection",
    MemoryType.MEMAGENT: "memagent_collection",
    MemoryType.SHARED_MEMORY: "shared_memory_collection",
    MemoryType.SUMMARIES: "summaries_collection",
    MemoryType.SEMANTIC_CACHE: "semantic_cache_collection",
    MemoryType.TOOL_LOG: "tool_log_collection",
}

# Every store the cascade must sweep (the agent row itself is handled by
# ``delete_memagent``). Includes the types the old if/elif chain omitted:
# ENTITY_MEMORY, SUMMARIES, SEMANTIC_CACHE, SHARED_MEMORY and SKILLBOX.
_CASCADE_TYPES = [t for t in MemoryType if t != MemoryType.MEMAGENT]


@pytest.fixture()
def provider():
    instance = MongoDBProvider.__new__(MongoDBProvider)
    instance.db = mongomock.MongoClient()["provider-fixes-test"]
    for memory_type, attribute in _COLLECTION_ATTRIBUTES.items():
        setattr(instance, attribute, instance.db[memory_type.value])
    instance._vector_indexes_unavailable = set()
    instance._vector_index_unavailable_root_causes = set()
    instance._vector_search_status_cache = {}
    return instance


def _store_agent(provider, agent_id, memory_ids):
    from memorizz.memagent.models import MemAgentModel

    provider.store_memagent(
        MemAgentModel(agent_id=agent_id, name=agent_id, instruction="hi")
    )
    assert provider.update_memagent_memory_ids(agent_id, memory_ids)


# ---------------------------------------------------------------------------
# Finding 1: cascade delete
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_delete_memagent_cascade_removes_agent_row_and_every_memory_store(provider):
    _store_agent(provider, "agent-1", ["mem-1"])
    _store_agent(provider, "agent-2", ["mem-other"])

    for memory_type in _CASCADE_TYPES:
        collection = provider.db[memory_type.value]
        collection.insert_one({"memory_id": "mem-1", "name": "doomed"})
        collection.insert_one({"memory_id": "mem-other", "name": "survivor"})
    # Agent-scoped records without a memory_id (persona/toolbox definitions)
    # are not memory units and must survive, as on the filesystem provider.
    provider.db[MemoryType.PERSONAS.value].insert_one({"name": "shared-persona"})
    provider.db[MemoryType.TOOLBOX.value].insert_one(
        {"_id": "agent-1:search", "name": "search", "agent_id": "agent-1"}
    )

    assert provider.delete_memagent("agent-1", cascade=True) is True

    assert provider.retrieve_memagent("agent-1") is None
    assert provider.retrieve_memagent("agent-2") is not None
    for memory_type in _CASCADE_TYPES:
        collection = provider.db[memory_type.value]
        assert collection.count_documents({"memory_id": "mem-1"}) == 0, memory_type
        assert collection.count_documents({"memory_id": "mem-other"}) == 1, memory_type
    assert (
        provider.db[MemoryType.PERSONAS.value].count_documents(
            {"name": "shared-persona"}
        )
        == 1
    )
    assert (
        provider.db[MemoryType.TOOLBOX.value].count_documents({"_id": "agent-1:search"})
        == 1
    )


@pytest.mark.unit
def test_delete_memagent_cascade_exact_reported_scenario(provider):
    """Agent with memory_ids=["mem-1"] and rows in conversation / entity /
    summaries / semantic_cache: previously returned True, left the agent row,
    and only removed the conversation rows."""
    _store_agent(provider, "agent-1", ["mem-1"])
    for memory_type in (
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.ENTITY_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
    ):
        provider.db[memory_type.value].insert_one({"memory_id": "mem-1"})

    assert provider.delete_memagent("agent-1", cascade=True) is True
    assert provider.memagent_collection.count_documents({}) == 0
    for memory_type in (
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.ENTITY_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
    ):
        assert provider.db[memory_type.value].count_documents({}) == 0, memory_type


# ---------------------------------------------------------------------------
# Finding 2: string _id resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_toolbox_row_with_string_id_can_be_retrieved_updated_and_deleted(provider):
    # The provider's own agent->toolbox sync writes "<agent_id>:<tool>" ids.
    provider._sync_agent_tools_to_toolbox(
        "agent-1", [{"name": "search", "description": "Search the web"}]
    )
    assert provider.toolbox_collection.find_one({"_id": "agent-1:search"})

    row = provider.retrieve_by_id("agent-1:search", MemoryType.TOOLBOX)
    assert row is not None and row["name"] == "search"

    assert provider.update_by_id(
        "agent-1:search", {"description": "Search the whole web"}, MemoryType.TOOLBOX
    )
    assert (
        provider.retrieve_by_id("agent-1:search", MemoryType.TOOLBOX)["description"]
        == "Search the whole web"
    )

    assert provider.delete_by_id("agent-1:search", MemoryType.TOOLBOX)
    assert provider.retrieve_by_id("agent-1:search", MemoryType.TOOLBOX) is None
    assert provider.delete_by_id("agent-1:search", MemoryType.TOOLBOX) is False


@pytest.mark.unit
def test_string_id_that_looks_like_object_id_still_resolves(provider):
    hex_id = str(ObjectId())
    provider.store(
        {"_id": hex_id, "name": "string-keyed"},
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )
    assert provider.retrieve_by_id(hex_id, MemoryType.KNOWLEDGE_BASE)["name"] == (
        "string-keyed"
    )
    assert provider.update_by_id(hex_id, {"name": "renamed"}, MemoryType.KNOWLEDGE_BASE)
    assert provider.delete_by_id(hex_id, MemoryType.KNOWLEDGE_BASE)


@pytest.mark.unit
def test_shared_memory_string_id_rows_resolve_and_compare_and_swap(provider):
    provider.store(
        {"_id": "session-1", "content": "before"},
        memory_store_type=MemoryType.SHARED_MEMORY,
    )
    assert provider.retrieve_by_id("session-1", MemoryType.SHARED_MEMORY)
    assert provider.compare_and_swap_shared_memory("session-1", "before", "after")
    assert provider.retrieve_by_id("session-1", MemoryType.SHARED_MEMORY)[
        "content"
    ] == ("after")
    assert not provider.compare_and_swap_shared_memory("session-1", "stale", "x")
    assert not provider.compare_and_swap_shared_memory("missing", "before", "x")


# ---------------------------------------------------------------------------
# Finding 3: return-value semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_idempotent_update_by_id_returns_true_when_row_exists(provider):
    record_id = provider.store(
        {"name": "p", "tone": "calm"}, memory_store_type=MemoryType.PERSONAS
    )
    assert provider.update_by_id(record_id, {"tone": "calm"}, MemoryType.PERSONAS)
    assert provider.update_by_id(record_id, {"tone": "calm"}, MemoryType.PERSONAS)
    assert not provider.update_by_id(
        str(ObjectId()), {"tone": "calm"}, MemoryType.PERSONAS
    )


@pytest.mark.unit
def test_idempotent_semantic_cache_update_by_cache_key_returns_true(provider):
    provider.store(
        {"cache_key": "k-1", "response": "hi"},
        memory_store_type=MemoryType.SEMANTIC_CACHE,
    )
    assert provider.update_by_id("k-1", {"response": "hi"}, MemoryType.SEMANTIC_CACHE)
    assert not provider.update_by_id(
        "missing-key", {"response": "hi"}, MemoryType.SEMANTIC_CACHE
    )


@pytest.mark.unit
def test_idempotent_memagent_updates_return_true(provider):
    _store_agent(provider, "agent-1", ["m1"])
    # Same memory_ids again: matched but not modified.
    assert provider.update_memagent_memory_ids("agent-1", ["m1"])
    assert provider.update_by_id("agent-1", {"name": "agent-1"}, MemoryType.MEMAGENT)
    assert provider.delete_memagent_memory_ids("agent-1")
    # memory_ids already unset: still True because the agent exists.
    assert provider.delete_memagent_memory_ids("agent-1")
    assert not provider.update_memagent_memory_ids("no-such-agent", ["m1"])


@pytest.mark.unit
def test_delete_all_on_empty_collection_returns_true(provider):
    assert provider.list_all(memory_store_type=MemoryType.PERSONAS) == []
    assert provider.delete_all(MemoryType.PERSONAS) is True
    provider.store({"name": "p"}, memory_store_type=MemoryType.PERSONAS)
    assert provider.delete_all(MemoryType.PERSONAS) is True
    assert provider.list_all(memory_store_type=MemoryType.PERSONAS) == []


# ---------------------------------------------------------------------------
# Finding 4: semantic cache expiry + embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_purge_expired_semantic_cache_deletes_only_expired_rows(provider):
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        {"cache_key": "dt-expired", "expires_at": now - timedelta(hours=1)},
        {"cache_key": "dt-live", "expires_at": now + timedelta(hours=1)},
        {"cache_key": "iso-expired", "expires_at": "2026-06-01T10:00:00+00:00"},
        {"cache_key": "iso-z-expired", "expires_at": "2026-06-01T10:00:00Z"},
        {"cache_key": "iso-live", "expires_at": "2026-06-01T14:00:00+00:00"},
        {"cache_key": "iso-garbage", "expires_at": "not-a-date"},
        {"cache_key": "no-expiry"},
    ]
    provider.semantic_cache_collection.insert_many(rows)

    assert provider.purge_expired_semantic_cache(now=now) == 3
    remaining = {
        row["cache_key"] for row in provider.semantic_cache_collection.find({})
    }
    assert remaining == {"dt-live", "iso-live", "iso-garbage", "no-expiry"}
    # Idempotent.
    assert provider.purge_expired_semantic_cache(now=now) == 0


@pytest.mark.unit
def test_purge_expired_semantic_cache_defaults_to_current_time(provider):
    provider.semantic_cache_collection.insert_many(
        [
            {"cache_key": "old", "expires_at": datetime(2000, 1, 1)},
            {"cache_key": "far-future", "expires_at": datetime(2999, 1, 1)},
        ]
    )
    assert provider.purge_expired_semantic_cache() == 1
    assert [r["cache_key"] for r in provider.semantic_cache_collection.find({})] == [
        "far-future"
    ]


@pytest.mark.unit
def test_semantic_cache_has_ttl_index_on_expires_at(provider):
    specs = MongoDBProvider._BTREE_INDEX_SPECS[MemoryType.SEMANTIC_CACHE]
    ttl = [s for s in specs if s[1] == [("expires_at", 1)]]
    assert ttl and ttl[0][2].get("expireAfterSeconds") == 0

    provider._ensure_btree_indexes()
    info = provider.semantic_cache_collection.index_information()
    assert any(
        spec.get("key") == [("expires_at", 1)] and spec.get("expireAfterSeconds") == 0
        for spec in info.values()
    )


@pytest.mark.unit
def test_semantic_cache_dict_query_honours_include_embedding(provider):
    provider.store(
        {
            "cache_key": "k-1",
            "agent_id": "a1",
            "query_text": "hello",
            "response": "hi",
            "embedding": [0.1, 0.2, 0.3],
        },
        memory_store_type=MemoryType.SEMANTIC_CACHE,
    )

    with_vectors = list(
        provider.retrieve_by_query(
            {"agent_id": "a1"},
            memory_store_type=MemoryType.SEMANTIC_CACHE,
            limit=10,
            include_embedding=True,
        )
    )
    assert len(with_vectors) == 1
    assert with_vectors[0]["embedding"] == [0.1, 0.2, 0.3]

    without_vectors = list(
        provider.retrieve_by_query(
            {"agent_id": "a1"}, memory_store_type=MemoryType.SEMANTIC_CACHE, limit=10
        )
    )
    assert len(without_vectors) == 1 and "embedding" not in without_vectors[0]

    listed = provider.list_all(
        memory_store_type=MemoryType.SEMANTIC_CACHE, include_embedding=True
    )
    assert listed[0]["embedding"] == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Finding 5: shared-memory user scoping
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shared_memory_dict_query_scopes_by_user_id(provider):
    provider.shared_memory_collection.insert_many(
        [
            {"memory_id": "s-alice", "user_id": "alice", "state": "open"},
            {"memory_id": "s-bob", "user_id": "bob", "state": "open"},
            {"memory_id": "s-anon", "state": "open"},
        ]
    )

    def ids(**kwargs):
        rows = provider.retrieve_by_query(
            {"state": "open"},
            memory_store_type=MemoryType.SHARED_MEMORY,
            limit=10,
            **kwargs,
        )
        return sorted(row["memory_id"] for row in rows)

    assert ids(user_id="alice") == ["s-alice"]
    assert ids(user_id=None) == ["s-anon"]
    assert ids() == ["s-alice", "s-anon", "s-bob"]


# ---------------------------------------------------------------------------
# Finding 6: tenant scoping on native list helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_summaries_keeps_user_scope(provider):
    provider.summaries_collection.insert_many(
        [
            {"memory_id": "mem-1", "user_id": "alice", "period_end": 2, "summary": "a"},
            {"memory_id": "mem-1", "user_id": "bob", "period_end": 3, "summary": "b"},
            {"memory_id": "mem-1", "period_end": 1, "summary": "anon"},
        ]
    )
    assert [
        r["summary"]
        for r in provider.list_summaries(memory_id="mem-1", user_id="alice")
    ] == ["a"]
    assert [
        r["summary"] for r in provider.list_summaries(memory_id="mem-1", user_id=None)
    ] == ["anon"]
    assert [r["summary"] for r in provider.list_summaries(memory_id="mem-1")] == [
        "b",
        "a",
        "anon",
    ]
