# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""CRUD dispatch tests for the MongoDB provider against mongomock.

The provider's per-type dispatch used to be three collection-mapping dicts
plus five if/elif chains; it is now a single ``_collection()`` resolver.
These tests pin the dispatch behavior for every memory type so the
consolidation (and future memory types) can't silently break a path.
"""

import pytest

mongomock = pytest.importorskip("mongomock")
pytest.importorskip("pymongo")

import memorizz.memory_provider.mongodb.provider as mongodb_module  # noqa: E402
from memorizz.approval import SQLiteApprovalStore  # noqa: E402
from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.long_term.semantic.entity_memory import EntityMemory  # noqa: E402
from memorizz.memory_provider.mongodb.provider import MongoDBProvider  # noqa: E402
from memorizz.metaharness import (  # noqa: E402
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)

# Types with plain name-keyed documents and no special-case handling.
PLAIN_TYPES = [
    MemoryType.PERSONAS,
    MemoryType.TOOLBOX,
    MemoryType.SKILLBOX,
    MemoryType.WORKFLOW_MEMORY,
    MemoryType.SHORT_TERM_MEMORY,
    MemoryType.KNOWLEDGE_BASE,
    MemoryType.CONVERSATION_MEMORY,
    MemoryType.SUMMARIES,
    MemoryType.ENTITY_MEMORY,
    MemoryType.TOOL_LOG,
]


@pytest.fixture()
def provider():
    instance = MongoDBProvider.__new__(MongoDBProvider)
    instance.db = mongomock.MongoClient()["dispatch-test"]
    # The native clear_semantic_cache override reads the bound attribute
    # rather than going through _collection().
    instance.semantic_cache_collection = instance.db[MemoryType.SEMANTIC_CACHE.value]
    instance.knowledge_base_collection = instance.db[MemoryType.KNOWLEDGE_BASE.value]
    instance._vector_indexes_unavailable = set()
    instance._vector_index_unavailable_root_causes = set()
    instance._vector_search_status_cache = {}
    return instance


@pytest.mark.unit
@pytest.mark.parametrize("memory_type", PLAIN_TYPES)
def test_store_retrieve_update_delete_round_trip(provider, memory_type):
    record_id = provider.store(
        {"name": "unit-doc", "payload": 1}, memory_store_type=memory_type
    )
    assert record_id

    fetched = provider.retrieve_by_id(record_id, memory_type)
    assert fetched and fetched["name"] == "unit-doc"

    by_name = provider.retrieve_by_name("unit-doc", memory_type)
    assert by_name and str(by_name["_id"]) == record_id

    assert provider.update_by_id(record_id, {"payload": 2}, memory_type)
    assert provider.retrieve_by_id(record_id, memory_type)["payload"] == 2

    rows = provider.list_all(memory_store_type=memory_type)
    assert len(rows) == 1

    assert provider.delete_by_id(record_id, memory_type)
    assert provider.retrieve_by_id(record_id, memory_type) is None


@pytest.mark.unit
def test_delete_by_name_and_delete_all(provider):
    provider.store({"name": "a"}, memory_store_type=MemoryType.TOOLBOX)
    provider.store({"name": "b"}, memory_store_type=MemoryType.TOOLBOX)
    assert provider.delete_by_name("a", MemoryType.TOOLBOX)
    assert len(provider.list_all(memory_store_type=MemoryType.TOOLBOX)) == 1
    assert provider.delete_all(MemoryType.TOOLBOX)
    assert provider.list_all(memory_store_type=MemoryType.TOOLBOX) == []


@pytest.mark.unit
def test_canonical_entity_identity_uses_one_mongodb_storage_key(provider, monkeypatch):
    provider.entity_memory_collection = provider.db[MemoryType.ENTITY_MEMORY.value]
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda _text: [0.25, 0.75],
    )
    entities = EntityMemory(provider)

    first_id = entities.upsert_entity(
        identity_key="authenticated_user",
        name="user",
        entity_type="person",
        attributes=[{"name": "role", "value": "AI Memory Engineer"}],
        memory_id="primary-user-1",
        user_id="user-1",
    )
    second_id = entities.upsert_entity(
        identity_key="authenticated_user",
        name="Richmond Alake",
        attributes=[{"name": "timezone", "value": "Europe/London"}],
        memory_id="primary-user-1",
        user_id="user-1",
    )

    rows = provider.list_all(MemoryType.ENTITY_MEMORY, user_id="user-1")
    assert second_id == first_id
    assert len(rows) == 1
    assert rows[0]["_id"] == first_id


@pytest.mark.unit
def test_semantic_cache_name_lookup_uses_cache_key(provider):
    provider.store(
        {"cache_key": "k-1", "query_text": "what is up", "response": "hi"},
        memory_store_type=MemoryType.SEMANTIC_CACHE,
    )
    assert provider.retrieve_by_name("k-1", MemoryType.SEMANTIC_CACHE)
    assert provider.retrieve_by_name("what is up", MemoryType.SEMANTIC_CACHE)
    assert provider.delete_by_name("k-1", MemoryType.SEMANTIC_CACHE)


@pytest.mark.unit
def test_metaharness_reuses_one_mongodb_context_snapshot_per_workflow(
    provider, tmp_path, monkeypatch
):
    monkeypatch.setattr(mongodb_module, "get_embedding", lambda _query: [0.2, 0.8])
    provider.store(
        {
            "name": "policy",
            "content": "Only owners may export records.",
            "user_id": "alice",
            "thread_id": "thread-1",
            "embedding": [0.2, 0.8],
        },
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-1",
    )
    service = MetaHarness(
        memory_provider=provider,
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    common = {
        "memory_query": "Review the export policy",
        "shared_context_key": "workflow-mongodb-1",
        "memory_context_strategy": "evidence_pack",
    }
    builds = 0
    original_build = service._build_context_pack

    def counted_build(*args, **kwargs):
        nonlocal builds
        builds += 1
        return original_build(*args, **kwargs)

    service._build_context_pack = counted_build
    first_task = HarnessTask(
        task="Review authorization",
        workspace=str(tmp_path),
        memory_id="memory-1",
        user_id="alice",
        thread_id="thread-1",
        context=common,
    )
    second_task = HarnessTask.from_dict(
        {**first_task.to_dict(), "task": "Review verification coverage"}
    )
    try:
        first = service._context_for_task(first_task)
        second = service._context_for_task(second_task)

        assert builds == 1
        assert first.source_ids == second.source_ids
        assert second.metadata["shared_context_reused"] is True
        assert second.metadata["retrieval_degraded"] is True
        assert "Only owners may export records" in second.rendered
    finally:
        service.close()


@pytest.mark.unit
def test_mongodb_knowledge_base_honors_store_memory_id_argument(provider):
    provider.store(
        {"content": "Scoped policy", "user_id": "alice"},
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
        memory_id="memory-argument",
    )
    row = provider.knowledge_base_collection.find_one({"user_id": "alice"})
    assert row["memory_id"] == "memory-argument"


@pytest.mark.unit
def test_mongodb_store_many_uses_portable_batch_contract(provider):
    identifiers = provider.store_many(
        [
            {"_id": "chunk-1", "content": "First", "user_id": "alice"},
            {"_id": "chunk-2", "content": "Second", "user_id": "alice"},
        ],
        MemoryType.KNOWLEDGE_BASE,
        memory_id="batch-memory",
    )
    rows = provider.list_all(MemoryType.KNOWLEDGE_BASE)
    assert identifiers == ["chunk-1", "chunk-2"]
    assert {row["content"] for row in rows} == {"First", "Second"}
    assert {row["memory_id"] for row in rows} == {"batch-memory"}
    assert provider.memory_capabilities().batch_store is True


@pytest.mark.unit
def test_shared_memory_name_lookup_uses_memory_id(provider):
    # Insert directly: provider.store() strips memory_id for shared memory
    # (pre-existing behavior); the coordination layer writes sessions with
    # its own document shape. This test pins the *dispatch* key only.
    provider.db[MemoryType.SHARED_MEMORY.value].insert_one(
        {"memory_id": "session-1", "state": "open"}
    )
    assert provider.retrieve_by_name("session-1", MemoryType.SHARED_MEMORY)
    assert provider.delete_by_name("session-1", MemoryType.SHARED_MEMORY)


@pytest.mark.unit
def test_workflow_store_preserves_learning_fields(provider):
    """agent_id/workflow_id must survive the store — trajectory aggregation
    groups by agent and the promotion engine stamps rows back by id."""
    provider.store(
        {
            "name": "wf",
            "workflow_id": "wf-123",
            "agent_id": "agent-9",
            "canonical_hash": "abc",
        },
        memory_store_type=MemoryType.WORKFLOW_MEMORY,
    )
    row = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)[0]
    assert row["workflow_id"] == "wf-123"
    assert row["agent_id"] == "agent-9"
    assert row["canonical_hash"] == "abc"


@pytest.mark.unit
def test_skillbox_round_trips_reviewed_instruction_authority(provider):
    """Document providers must preserve the same role contract as Oracle."""
    from memorizz.long_term.procedural.skillbox import (
        Skill,
        Skillbox,
        SkillInjectionRole,
    )

    box = Skillbox(provider, agent_id="agent-9")
    skill = Skill(
        name="reviewed-refund",
        description="Refund an eligible order",
        content="Verify state, refund, then send the receipt.",
        injection_role="developer",
        embedding=[],
    )
    box.add_skill(skill)

    loaded = box.get_skill_by_id(skill.skill_id)
    assert loaded is not None
    assert loaded.injection_role is SkillInjectionRole.DEVELOPER

    loaded.injection_role = SkillInjectionRole.USER
    assert box.update_skill(loaded)
    assert box.get_skill_by_id(skill.skill_id).injection_role is SkillInjectionRole.USER


@pytest.mark.unit
def test_clear_semantic_cache_scoped(provider):
    provider.store(
        {"cache_key": "k1", "agent_id": "a1"},
        memory_store_type=MemoryType.SEMANTIC_CACHE,
    )
    provider.store(
        {"cache_key": "k2", "agent_id": "a2"},
        memory_store_type=MemoryType.SEMANTIC_CACHE,
    )
    assert provider.clear_semantic_cache(agent_id="a1") == 1
    remaining = provider.list_all(memory_store_type=MemoryType.SEMANTIC_CACHE)
    assert len(remaining) == 1 and remaining[0]["agent_id"] == "a2"


@pytest.mark.unit
def test_memagent_custom_string_id_round_trip(provider):
    """Agents saved under custom string agent_ids (the SDK default is a
    uuid4) must be retrievable, updatable, and deletable by that id —
    historically only ObjectId _ids resolved."""
    from memorizz.memagent.models import MemAgentModel

    provider.memagent_collection = provider.db[MemoryType.MEMAGENT.value]
    provider._sync_agent_tools_to_toolbox = lambda *a, **k: None

    model = MemAgentModel(
        agent_id="my-string-agent",
        name="String Id Agent",
        instruction="hello",
        continual_learning=True,
    )
    provider.store_memagent(model)

    loaded = provider.retrieve_memagent("my-string-agent")
    assert loaded is not None
    assert loaded.agent_id == "my-string-agent"
    assert loaded.name == "String Id Agent"
    assert bool(loaded.continual_learning) is True

    # Repeated save upserts instead of duplicating.
    provider.store_memagent(model)
    assert provider.memagent_collection.count_documents({}) == 1

    assert provider.update_memagent_memory_ids("my-string-agent", ["m1"])
    assert provider.retrieve_memagent("my-string-agent").memory_ids == ["m1"]

    assert provider.delete_memagent("my-string-agent")
    assert provider.retrieve_memagent("my-string-agent") is None


@pytest.mark.unit
def test_memagent_objectid_lookup_still_works(provider):
    from memorizz.memagent.models import MemAgentModel

    provider.memagent_collection = provider.db[MemoryType.MEMAGENT.value]
    provider._sync_agent_tools_to_toolbox = lambda *a, **k: None

    stored = provider.store_memagent(
        MemAgentModel(name="ObjectId Agent", instruction="hi")
    )
    oid = str(stored["_id"])
    loaded = provider.retrieve_memagent(oid)
    assert loaded is not None and loaded.name == "ObjectId Agent"
