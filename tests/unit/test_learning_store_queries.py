"""The learning control-plane store must not scan the shared-memory partition on hot paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorizz import FileSystemConfig, FileSystemProvider, LearningControlPlane
from memorizz.enums import MemoryType
from memorizz.learning import LearningEventType
from memorizz.learning.store import (
    LEARNING_ARTIFACT_RECORD,
    LEARNING_TOMBSTONE_RECORD,
    LearningControlPlaneStore,
)


class _Embeddings:
    def get_embedding(self, text):
        return [1.0, 0.0]

    def get_provider_info(self):
        return "unit-test"


@pytest.fixture()
def provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "learning-store",
            embedding_provider=_Embeddings(),
            lazy_vector_indexes=True,
        )
    )


def _count_scans(provider):
    calls = {"list_all": 0}
    original = provider.list_all

    def counted(*args, **kwargs):
        calls["list_all"] += 1
        return original(*args, **kwargs)

    provider.list_all = counted
    return calls


@pytest.mark.unit
def test_writes_and_lookups_never_scan_the_partition(provider):
    store = LearningControlPlaneStore(provider, agent_id="agent-1")
    calls = _count_scans(provider)

    for index in range(3):
        store.put_artifact(
            {
                "record_id": f"artifact-{index}",
                "artifact_kind": "outcome",
                "content": f"fact {index}",
                "memory_id": "memory-1",
            }
        )
    store.put_checkpoint({"record_id": "checkpoint-1", "cursor": 3})
    store.put_tombstone({"target_id": "artifact-0", "approved_by": "operator"})
    # Idempotent re-put of an existing artifact must also be a direct read.
    store.put_artifact(
        {
            "record_id": "artifact-1",
            "artifact_kind": "outcome",
            "content": "fact 1 revised",
            "memory_id": "memory-1",
        }
    )

    assert calls["list_all"] == 0
    assert store.lookup("artifact-2")["content"] == "fact 2"
    assert store.lookup("checkpoint-1")["cursor"] == 3
    assert calls["list_all"] == 0

    # Listing on the filesystem provider still needs one scan (no indexed query).
    rows = store.list_artifacts(memory_id="memory-1")
    assert {row["record_id"] for row in rows} == {
        "artifact-0",
        "artifact-1",
        "artifact-2",
    }
    assert calls["list_all"] == 1
    assert store.tombstoned_ids() == {"artifact-0"}


@pytest.mark.unit
def test_get_falls_back_to_a_scan_only_for_unknown_ids(provider):
    store = LearningControlPlaneStore(provider, agent_id="agent-1")
    store.put_artifact({"record_id": "artifact-1", "artifact_kind": "outcome"})
    calls = _count_scans(provider)

    assert store.get("artifact-1")["record_id"] == "artifact-1"
    assert calls["list_all"] == 0
    assert store.get("does-not-exist") is None
    assert calls["list_all"] == 1


class _IndexedProvider:
    """Duck-typed provider with an indexed observability query and no scans."""

    def __init__(self):
        self.rows = {}
        self.queries = []

    def store(self, data, memory_store_type):
        self.rows[data["_id"]] = dict(data)
        return data["_id"]

    def retrieve_by_id(self, record_id, memory_store_type):
        return self.rows.get(record_id)

    def retrieve_by_name(self, name, memory_store_type):
        return None

    def list_all(self, *args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("indexed provider must not be scanned")

    def query_observability_records(self, memory_store_type, **kwargs):
        self.queries.append(kwargs)
        wanted_type = kwargs.get("record_type")
        agents = set(kwargs.get("agent_ids") or [])
        user_id = kwargs.get("user_id", "__unset__")
        items = []
        for row in self.rows.values():
            if agents and row.get("agent_id") not in agents:
                continue
            if wanted_type and row.get("record_type") != wanted_type:
                continue
            if user_id != "__unset__" and row.get("user_id") != user_id:
                continue
            items.append(row)
        return {"items": items, "next_cursor": None}


@pytest.mark.unit
def test_indexed_provider_query_is_used_instead_of_list_all():
    provider = _IndexedProvider()
    store = LearningControlPlaneStore(provider, agent_id="agent-1")
    store.put_artifact(
        {"record_id": "artifact-a", "artifact_kind": "outcome", "user_id": "alice"}
    )
    store.put_artifact(
        {"record_id": "artifact-b", "artifact_kind": "outcome", "user_id": "bob"}
    )
    store.put_tombstone({"target_id": "artifact-b", "approved_by": "operator"})

    alice_rows = store.list_artifacts(user_id="alice")
    assert [row["record_id"] for row in alice_rows] == ["artifact-a"]
    query = provider.queries[-1]
    assert query["agent_ids"] == ["agent-1"]
    assert query["record_type"] == LEARNING_ARTIFACT_RECORD
    assert query["user_id"] == "alice"

    assert store.tombstoned_ids() == {"artifact-b"}
    assert provider.queries[-1]["record_type"] == LEARNING_TOMBSTONE_RECORD

    stats = store.statistics()
    assert stats["record_count"] == 3
    assert stats["record_types"][LEARNING_ARTIFACT_RECORD] == 2


@pytest.mark.unit
def test_idempotent_emit_does_not_scan(provider):
    plane = LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={"enabled": True, "compile_async": False, "compile_every_n_events": 0},
    )
    calls = _count_scans(provider)
    first = plane.emit(
        LearningEventType.RUN_STARTED,
        {"query": "hello"},
        scope={"memory_id": "memory-1"},
        idempotency_key="turn-1",
        compile_if_due=False,
    )
    second = plane.emit(
        LearningEventType.RUN_STARTED,
        {"query": "hello"},
        scope={"memory_id": "memory-1"},
        idempotency_key="turn-1",
        compile_if_due=False,
    )
    assert first is not None and second is not None
    assert first.event_id == second.event_id
    assert calls["list_all"] == 0
