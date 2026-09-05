"""Rollout, immutable retries, archival expiry and content separation."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityMaintenance,
    ObservabilityRecorder,
    ObservabilityStore,
    TraceContext,
)
from memorizz.observability.index import digest
from memorizz.observability.pipeline import pipeline_health
from memorizz.ui.security import ReadOnlyProviderProxy


@pytest.fixture
def provider(tmp_path):
    p = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield p
    p.close()


def context(turn="turn", user="alice"):
    return TraceContext(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        run_id="run",
        turn_id=turn,
        user_id=user,
    )


def seed(provider, turn="turn", *, content="PRIVATE PREVIEW"):
    return ObservabilityStore(provider).record_trace_bundle(
        trace_context=context(turn).to_carrier(),
        events=[
            {
                "event_id": "event-" + turn,
                "trace_kind": "model_result",
                "content": content,
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    )


def test_backfill_dry_run_parity_rollback_and_readonly(provider, monkeypatch):
    for turn in ("a", "b", "c"):
        seed(provider, turn)
    original = deepcopy(provider.list_all(MemoryType.SHARED_MEMORY))
    m = ObservabilityMaintenance(provider)
    assert m.initialize()["ready"]
    assert m.backfill(since="2020-01-01")["events"] == 3
    assert not m.parity()["passed"]
    page = m.backfill(since="2020-01-01", limit=2, dry_run=False)
    assert page["next_cursor"]
    assert (
        m.backfill(
            since="2020-01-01", limit=2, cursor=page["next_cursor"], dry_run=False
        )["bundles"]
        == 1
    )
    assert m.parity()["passed"]
    assert provider.list_all(MemoryType.SHARED_MEMORY) == original
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "index")
    assert len(provider.query_trace_events()["items"]) == 3
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "bundles")
    assert len(provider.query_trace_events()["items"]) == 3
    readonly = ObservabilityMaintenance(ReadOnlyProviderProxy(provider))
    assert readonly.parity()["passed"]
    for operation in (
        readonly.initialize,
        lambda: readonly.backfill(since="2020-01-01", dry_run=False),
        lambda: readonly.retention(dry_run=False),
    ):
        with pytest.raises(PermissionError):
            operation()


def test_immutable_turn_and_external_event_retries_preserve_first_capture(provider):
    first = seed(provider)
    retry = seed(provider, content="CHANGED")
    assert retry == first
    recorder = ObservabilityRecorder(provider, context(), strict=True)
    first_event = recorder.record_event("work", external_id="job", status="success")
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(
            pool.map(
                lambda _: recorder.record_event(
                    "work", external_id="job", status="error"
                ),
                range(12),
            )
        )
    assert all(value == first_event for value in results)
    assert len(provider.query_trace_events()["items"]) == 2


def test_dual_write_removes_preview_from_canonical_source_and_expiry(
    provider, monkeypatch
):
    m = ObservabilityMaintenance(provider)
    m.initialize()
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_DUAL_WRITE", "true")
    result = seed(provider)
    assert result["preview_storage"] == "separate"
    assert "PRIVATE" not in json.dumps(provider.list_all(MemoryType.SHARED_MEMORY))
    assert (
        provider.get_observability_index().preview("event-turn", user_id="alice")
        == "PRIVATE PREVIEW"
    )
    assert m.parity()["passed"]
    m.retention(now=datetime(2026, 1, 10, tzinfo=timezone.utc), dry_run=False)
    assert provider.get_observability_index().preview("event-turn") is None
    # Idempotent retry must not resurrect a purged preview.
    seed(provider)
    assert provider.get_observability_index().preview("event-turn") is None
    assert len(provider.query_trace_events()["items"]) == 1


def test_index_failure_keeps_canonical_evidence_and_is_visible(provider, monkeypatch):
    m = ObservabilityMaintenance(provider)
    m.initialize()
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_DUAL_WRITE", "true")
    with patch.object(
        m.index, "put", side_effect=RuntimeError("private backend message")
    ):
        result = seed(provider)
    assert result["index_persisted"] is False
    assert len(provider.query_trace_events(read_path="bundles")["items"]) == 1
    assert m.index.query()["coverage"] == "untrusted"
    assert pipeline_health(provider)["counters"]["index_write_failures"] == 1
    assert m.backfill(since="2020-01-01", dry_run=False)["failed"] == 0
    assert m.parity()["passed"]


def test_source_expiry_requires_review_archive_and_atomic_fingerprint(provider):
    payload = seed(provider)
    # Write time is canonical source age, separate from event time.
    with patch(
        "memorizz.observability.store._now", return_value="2020-01-01T00:00:00Z"
    ):
        old = seed(provider, "old")
    provider.store(
        {"_id": "not-telemetry", "content": "KEEP"}, MemoryType.SHARED_MEMORY
    )
    m = ObservabilityMaintenance(provider)
    m.initialize()
    plan = m.plan_source_expiry(before="2025-01-01")
    assert len(plan["entries"]) == 1
    with pytest.raises(ValueError):
        m.apply_source_expiry(plan, confirmation="wrong", archive=lambda _: True)
    with pytest.raises(RuntimeError):
        m.apply_source_expiry(
            plan, confirmation=plan["confirmation"], archive=lambda _: False
        )
    archived = []
    result = m.apply_source_expiry(
        plan,
        confirmation=plan["confirmation"],
        archive=lambda value: (archived.append(value) or True),
    )
    assert result == {"removed": 1, "recoverable_from_archive": True}
    assert archived == [old]
    assert ObservabilityStore(provider)._get(payload["record_id"])
    assert provider.retrieve_by_id("not-telemetry", MemoryType.SHARED_MEMORY)
    assert not provider.delete_observability_bundle(
        payload["record_id"], digest({"wrong": True})
    )


def test_mongo_immutable_retry_and_physical_id_expiry():
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    mongomock = pytest.importorskip("mongomock")
    p = MongoDBProvider.__new__(MongoDBProvider)
    p.db = mongomock.MongoClient()["memorizz-tests"]
    p.shared_memory_collection = p.db[MemoryType.SHARED_MEMORY.value]
    payload = {
        "record_id": "logical",
        "record_type": "observability_trace_bundle",
        "events": [],
    }
    doc = {
        "memory_id": "logical",
        "record_type": payload["record_type"],
        "immutable_trace": True,
        "content": json.dumps(payload),
    }
    first_id = p._write_document(
        p.shared_memory_collection, doc, MemoryType.SHARED_MEMORY
    )
    assert (
        p._write_document(
            p.shared_memory_collection,
            {**doc, "content": "CHANGED"},
            MemoryType.SHARED_MEMORY,
        )
        == first_id
    )
    assert p.shared_memory_collection.count_documents({}) == 1
    assert not p.delete_observability_bundle("logical", "wrong")
    assert p.delete_observability_bundle("logical", digest(payload))
    # Old logical IDs differ from Mongo's physical ObjectId.
    p.shared_memory_collection.insert_one(
        {"memory_id": "logical", "content": json.dumps(payload)}
    )
    assert p.delete_observability_bundle("logical", digest(payload))
