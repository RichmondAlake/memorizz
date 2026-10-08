"""Regression coverage for filtered pages, bounded reads and delegation authors."""

import json
from unittest.mock import patch

import pytest

from memorizz import MemAgent, MemoryHistory, MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.memory_provider.base import MemoryProvider
from memorizz.observability.store import ObservabilityStore
from tests.mocks.mock_providers import MockLLMProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def provider(tmp_path):
    return FileSystemProvider(FileSystemConfig(tmp_path / "memory", use_faiss=False))


def seed_events(provider):
    history = MemoryHistory(provider, record_changes=True)
    for index in range(18):
        wanted = index % 3 == 0
        with history.recording(
            actor="writer" if wanted else "other",
            agent_id="a",
            memory_id="m",
            run_id="execution",
            user_id=None,
            application_id="app",
        ):
            identifier = provider.store(
                {"content": str(index)},
                MemoryType.KNOWLEDGE_BASE if wanted else MemoryType.SHORT_TERM_MEMORY,
            )
            provider.update_by_id(
                identifier,
                {"content": "updated"},
                MemoryType.KNOWLEDGE_BASE if wanted else MemoryType.SHORT_TERM_MEMORY,
            )
    return history


def assert_filtered_pages(history):
    collected, cursor = [], None
    while True:
        page = history.timeline(
            agent_id="a",
            memory_id="m",
            memory_type=MemoryType.KNOWLEDGE_BASE,
            action="updated",
            actor="writer",
            run_id="execution",
            user_id=None,
            application_id="app",
            limit=2,
            cursor=cursor,
        )
        assert len(page["events"]) == 2 or not page["next_cursor"]
        collected.extend(page["events"])
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert len(collected) == 6
    assert len({event["record_id"] for event in collected}) == 6
    assert all(event["label"] == "knowledge base" for event in collected)
    assert history.timeline(agent_id="a", memory_id="another", limit=2)["events"] == []


def test_filters_apply_before_provider_pagination(provider):
    history = seed_events(provider)
    assert_filtered_pages(history)
    assert len(history.timeline(agent_id="", memory_id="")["events"]) == 36


def test_legacy_provider_without_filter_argument_fills_pages(provider):
    seed_events(provider)

    class Legacy:
        def query_observability_records(self, kind, *, limit, cursor, **scope):
            return MemoryProvider.query_observability_records(
                provider, kind, limit=limit, cursor=cursor, **scope
            )

    assert_filtered_pages(MemoryHistory(Legacy()))


def test_nonadvancing_cursor_fails_instead_of_looping():
    class Broken:
        def query_observability_records(self, kind, **kwargs):
            return {"items": [], "next_cursor": "same"}

    with pytest.raises(ValueError, match="did not advance"):
        MemoryHistory(Broken()).timeline()


def test_bounded_observations_read_only_matching_files_and_exclude_known(provider):
    for index in range(30):
        provider.store(
            {"agent_id": "other", "memory_id": "m", "content": str(index)},
            MemoryType.KNOWLEDGE_BASE,
        )
    known = provider.store(
        {"agent_id": "a", "user_id": None, "application_id": "app"},
        MemoryType.KNOWLEDGE_BASE,
    )
    wanted = [
        provider.store(
            {"memory_id": "m", "user_id": None, "application_id": "app"},
            MemoryType.KNOWLEDGE_BASE,
        )
        for _ in range(3)
    ]
    with patch.object(
        provider, "list_all", side_effect=AssertionError("full scan")
    ), patch.object(provider, "_read_document", wraps=provider._read_document) as read:
        rows = MemoryHistory(provider).observations(
            agent_id="a",
            memory_ids=["m"],
            user_id=None,
            application_id="app",
            limit=2,
            exclude_records=[(MemoryType.KNOWLEDGE_BASE, known)],
        )
        assert [row["target_record_id"] for row in rows] == wanted[:2]
        assert read.call_count == 2
    assert all(
        row["actor"] == "unknown" and row["label"] == "knowledge base" for row in rows
    )


def test_legacy_filesystem_index_upgrades_in_memory_without_writing(provider):
    identifier = provider.store({"owner_agent_id": "a"}, MemoryType.KNOWLEDGE_BASE)
    metadata = provider._indexes[MemoryType.KNOWLEDGE_BASE][identifier]
    for field in ("owner_agent_id", "observation_scope_version"):
        metadata.pop(field)
    with patch.object(
        provider, "_read_document", wraps=provider._read_document
    ) as read:
        assert provider.query_memory_observations(
            MemoryType.KNOWLEDGE_BASE, agent_id="a", limit=1
        )
        assert read.call_count == 1
        assert (
            "observation_scope_version"
            in provider._indexes[MemoryType.KNOWLEDGE_BASE][identifier]
        )
    with patch.object(
        provider, "_read_document", wraps=provider._read_document
    ) as read:
        assert not provider.query_memory_observations(
            MemoryType.KNOWLEDGE_BASE, agent_id="other"
        )
        assert read.call_count == 0


def test_mongodb_filtered_pages_include_old_journals_and_omit_vectors():
    mongomock = pytest.importorskip("mongomock")
    from memorizz.memory_provider.mongodb import MongoDBProvider

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = mongomock.MongoClient()["history"]
    store = ObservabilityStore(provider)
    for index in range(12):
        event = {
            "record_type": "memory_history_change",
            "record_id": str(index),
            "agent_id": "a",
            "memory_id": "m",
            "user_id": None,
            "memory_type": "knowledge_base",
            "action": "updated" if index % 3 == 0 else "created",
            "actor": "writer",
            "timestamp": str(index).zfill(2),
        }
        # Older records contain canonical payload but no indexed change metadata.
        if index % 2:
            provider.db.shared_memory.insert_one(
                {
                    "record_type": event["record_type"],
                    "agent_id": "a",
                    "trace_memory_id": "m",
                    "content": json.dumps(event),
                    "embedding": [1.0],
                }
            )
        else:
            store._put(event)
    history = MemoryHistory(provider)
    first = history.timeline(
        agent_id="a", action="updated", memory_type="knowledge_base", limit=2
    )
    second = history.timeline(
        agent_id="a",
        action="updated",
        memory_type="knowledge_base",
        limit=2,
        cursor=first["next_cursor"],
    )
    assert len(first["events"]) == len(second["events"]) == 2
    assert len({e["record_id"] for e in first["events"] + second["events"]}) == 4
    assert second["next_cursor"] is None
    assert history.timeline(agent_id="a", user_id="other")["events"] == []


def test_mongodb_observations_intersect_owner_and_tenant_before_limit():
    mongomock = pytest.importorskip("mongomock")
    from memorizz.memory_provider.mongodb import MongoDBProvider

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = mongomock.MongoClient()["observations"]
    collection = provider.db.knowledge_base
    collection.insert_many(
        [
            {
                "_id": "wrong-owner",
                "agent_id": "b",
                "memory_id": "m",
                "user_id": None,
                "application_id": "app",
            },
            {
                "_id": "wrong-user",
                "agent_id": "a",
                "user_id": "bob",
                "application_id": "app",
            },
            {"_id": "known", "agent_id": "a", "user_id": None, "application_id": "app"},
            {
                "_id": "owner-alias",
                "owner_agent_id": "a",
                "user_id": None,
                "application_id": "app",
                "embedding": [1.0],
            },
            {
                "_id": "namespace",
                "memory_id": "m",
                "user_id": None,
                "application_id": "app",
            },
        ]
    )
    with patch.object(provider, "list_all", side_effect=AssertionError("full scan")):
        rows = provider.query_memory_observations(
            MemoryType.KNOWLEDGE_BASE,
            agent_id="a",
            memory_ids=["m"],
            user_id=None,
            application_id="app",
            exclude_ids=["known"],
            limit=2,
        )
    assert {row["_id"] for row in rows} == {"owner-alias", "namespace"}
    assert all("embedding" not in row for row in rows)


def test_direct_delegation_keeps_coordinator_writer_and_child_initiator(provider):
    history = MemoryHistory(provider, record_changes=True)
    worker = MemAgent(
        agent_id="worker",
        name="Worker",
        model=MockLLMProvider(["Done"]),
        memory_provider=provider,
        auto_register=False,
        capture_memory_history=False,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
    )
    coordinator = MemAgent(
        agent_id="coordinator",
        name="Coordinator",
        model=MockLLMProvider(),
        memory_provider=provider,
        auto_register=False,
        delegates=[worker],
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        capture_memory_history=True,
        delegation={
            "enabled": True,
            "mode": "deterministic",
            "persist_participants": False,
            "consolidation_strategy": "primary",
            "primary_task_id": "task",
            "max_workers": 2,
        },
    )
    try:
        report = coordinator.delegate(
            "Execute",
            memory_id="m",
            return_report=True,
            plan=[
                {
                    "task_id": "task",
                    "assigned_agent_id": worker.agent_id,
                    "description": "Reply Done",
                }
            ],
        )
        assert report["tasks"][0]["status"] == "completed"
        root = history.timeline(agent_id="coordinator")["events"]
        conversation = [e for e in root if e["memory_type"] == "conversation_memory"]
        assert conversation and all(e["actor"] == "coordinator" for e in conversation)
        assert all(e["actor"] != "unknown" for e in root)
        contributions = [e for e in root if e["actor"] == "worker"]
        assert contributions and all(
            e["initiator"] == "coordinator" for e in contributions
        )
        private = history.timeline(
            agent_id="worker", memory_type="conversation_memory"
        )["events"]
        assert private and all(
            e["actor"] == "worker" and e["initiator"] == "coordinator" for e in private
        )
        identifier = provider.store(
            {"content": "unattributed later"}, MemoryType.KNOWLEDGE_BASE
        )
        later = history.timeline(memory_type="knowledge_base")["events"]
        assert (
            next(e for e in later if e["target_record_id"] == identifier)["actor"]
            == "unknown"
        )
    finally:
        coordinator.close()
        worker.close()


def test_failed_delegation_restores_outer_attribution(provider):
    history = MemoryHistory(provider, record_changes=True)
    coordinator = MemAgent(
        agent_id="coordinator",
        model=MockLLMProvider(),
        memory_provider=provider,
        auto_register=False,
        capture_memory_history=True,
    )
    try:
        with history.recording(actor="host", source="request"):
            with pytest.raises(ValueError, match="no configured delegates"):
                coordinator.delegate("invalid")
            identifier = provider.store(
                {"content": "after failure"}, MemoryType.KNOWLEDGE_BASE
            )
        event = next(
            e
            for e in history.timeline()["events"]
            if e["target_record_id"] == identifier
        )
        assert event["actor"] == "host" and event["source"] == "request"
    finally:
        coordinator.close()


def test_oracle_archive_enabled_shared_memory_loads_logical_and_physical_uuids():
    import sqlite3
    import uuid

    from memorizz.memory_provider.oracle import OracleProvider

    logical, physical = str(uuid.uuid4()), uuid.uuid4()
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE shared_memory(id BLOB, memory_id TEXT, content TEXT)"
    )
    content = '{ "blackboard": [] }'
    connection.execute(
        "INSERT INTO shared_memory VALUES (?, ?, ?)", (physical.bytes, logical, content)
    )
    provider = OracleProvider.__new__(OracleProvider)
    provider._archive_types = {"shared_memory"}
    provider._get_connection = lambda: connection
    provider._get_table_name = lambda kind: kind.value
    provider._read_lob_value = lambda value: value
    try:
        by_session = provider.retrieve_by_id(logical, MemoryType.SHARED_MEMORY)
        by_row = provider.retrieve_by_id(str(physical), MemoryType.SHARED_MEMORY)
        assert by_session == by_row
        assert by_session["memory_id"] == logical
        assert by_session["content"] == content  # CAS needs the exact serialized value.
        assert (
            provider.retrieve_by_id(str(uuid.uuid4()), MemoryType.SHARED_MEMORY) is None
        )
    finally:
        connection.close()
