"""The same native index contract runs against SQLite and mongomock."""

from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from memorizz.observability.index import _opaque_fields
from memorizz.observability.normalization import TraceSnapshot, normalize_trace_snapshot
from memorizz.observability.sql_index import SQLiteSpanIndex


def bundle(number=0, *, user="alice", timestamp="2026-09-04T10:00:00Z", size=12):
    return {
        "record_id": f"bundle-{user}-{number}",
        "record_type": "observability_trace_bundle",
        "type": "trace_bundle",
        "version": 2,
        "agent_id": "agent",
        "memory_id": "memory-B",
        "thread_id": "thread",
        "user_id": user,
        "root_trace_id": f"root-{number}",
        "run_id": "run",
        "turn_id": "turn",
        "timestamp": timestamp,
        "events": [
            {
                "event_id": f"event-{number}-{i}",
                "trace_kind": "tool_call" if i < 4 else "model_result",
                "span_id": f"span-{i}",
                "status": "success",
                "timestamp": timestamp,
                "logical_tool_name": "lookup",
                "content": "PRIVATE PREVIEW",
            }
            for i in range(size)
        ],
    }


@pytest.fixture(params=["sqlite", "mongodb"])
def index(request, tmp_path):
    if request.param == "sqlite":
        value = SQLiteSpanIndex(tmp_path)
    else:
        from memorizz.observability.mongo_index import MongoSpanIndex

        value = MongoSpanIndex(
            pytest.importorskip("mongomock").MongoClient()["index-tests"]
        )
    assert not value.ready()
    value.initialize()
    value.initialize()
    return value


def test_index_pages_match_normalizer_and_do_not_duplicate(index):
    expected = []
    for number, size in enumerate((12, 36, 26)):
        payload = bundle(number, size=size)
        expected.extend(
            normalize_trace_snapshot(
                TraceSnapshot(
                    bundle_rows=[
                        {**payload, "_id": payload["record_id"], "content": payload}
                    ]
                )
            ).events
        )
        index.write_bundle(payload)
        index.write_bundle(payload)
    cursor, observed = None, []
    while True:
        page = index.query(
            agent_ids=["agent"],
            memory_ids=["not-registered"],
            user_id="alice",
            limit=7,
            cursor=cursor,
        )
        observed.extend(page["items"])
        if not page["next_cursor"]:
            assert page["window_complete"] is False
            break
        cursor = page["next_cursor"]
    assert len(observed) == len({e["event_id"] for e in observed}) == 74
    assert {_opaque_fields(e)["event_id"] for e in expected} == {
        e["event_id"] for e in observed
    }
    assert "PRIVATE" not in str(observed)
    assert sum(row["event_count"] for row in index.summaries(user_id="alice")) == 74
    assert sum(row["bundle_count"] for row in index.summaries(user_id="alice")) == 3


def test_keyset_cursor_stays_stable_when_new_head_is_inserted(index):
    index.write_bundle(bundle())
    page = index.query(user_id="alice", limit=3)
    index.write_bundle(bundle(2, timestamp="2026-09-05T10:00:00Z"))
    next_page = index.query(user_id="alice", limit=100, cursor=page["next_cursor"])
    assert len(next_page["items"]) == 9
    assert not (
        {e["event_id"] for e in page["items"]}
        & {e["event_id"] for e in next_page["items"]}
    )
    with pytest.raises(ValueError, match="scope"):
        index.query(user_id="bob", cursor=page["next_cursor"])


def test_tenant_filters_are_conjunctive_even_with_agent_memory_or(index):
    index.write_bundle(bundle(user="alice"))
    index.write_bundle(bundle(user="bob"))
    index.write_bundle(bundle(user=None))
    assert (
        len(
            index.query(agent_ids=["agent"], memory_ids=["memory-B"], user_id="alice")[
                "items"
            ]
        )
        == 12
    )
    assert len(index.query(user_id=None)["items"]) == 12
    assert len(index.query()["items"]) == 36
    assert not index.query(thread_id="wrong", user_id="alice")["items"]


def test_filters_and_preview_lookup(index):
    payload = bundle()
    payload["events"][0].update(
        schema_version=3,
        event_kind="application_action",
        operation="persist",
        phase="result",
        input_refs=[{"resource_type": "analysis", "ref": "analysis-1"}],
        attributes={"job_ref": "job-1", "error_code": "example"},
    )
    # v3 children must be canonical, without v2-only trace_kind/logical_tool_name/content.
    child = payload["events"][0]
    for field in ("trace_kind", "logical_tool_name", "content"):
        child.pop(field)
    index.write_bundle(payload)
    assert len(index.query(resource_refs=["analysis-1"])["items"]) == 1
    assert len(index.query(query="job-1")["items"]) == 1
    assert len(index.query(query="example")["items"]) == 1
    assert len(index.query(event_kinds=["tool_call"], success=True)["items"]) == 3
    assert not index.query(start_time="2026-09-05")["items"]
    assert index.preview("event-0-1", user_id="alice") == "PRIVATE PREVIEW"
    assert index.preview("event-0-1", user_id="bob") is None
    with pytest.raises(ValueError):
        index.query(user_id="someone@example.com")


def test_retention_is_dry_run_by_default_and_preserves_verified_outcomes(index):
    old = bundle(timestamp="2026-01-01T00:00:00Z")
    old["events"][0] = {
        "schema_version": 3,
        "event_id": "event-0-0",
        "span_id": "outcome",
        "event_kind": "verified_outcome",
        "operation": "task.outcome",
        "timestamp": old["timestamp"],
        "status": "success",
        "attributes": {"verified": True},
    }
    index.write_bundle(old)
    now = datetime(2026, 9, 4, tzinfo=timezone.utc)
    result = index.retention(now=now)
    assert result["metadata_events"] == 11
    assert len(index.query()["items"]) == 12
    index.retention(now=now, dry_run=False)
    assert len(index.query()["items"]) == 1
    assert index.query()["items"][0]["kind"] == "verified_outcome"
    assert index.preview("event-0-0") is None


def test_failed_index_write_is_visible_and_retry_recovers(index):
    with patch.object(index, "put", side_effect=RuntimeError("backend offline")):
        with pytest.raises(RuntimeError):
            index.write_bundle(bundle())
    assert index.query()["coverage"] == "untrusted"
    index.write_bundle(bundle())
    assert index.query()["coverage"] == "complete"


def test_bundles_not_mutated_by_indexing(index):
    payload = bundle()
    original = deepcopy(payload)
    index.write_bundle(payload)
    assert payload == original


def test_index_rejects_unsafe_tenant_identity_instead_of_silently_dropping_scope(index):
    for field in ("user_id", "application_id"):
        payload = bundle()
        payload[field] = "person@example.com"
        with pytest.raises(ValueError, match="identity"):
            index.write_bundle(payload)
    assert index.query()["items"] == []
    assert index.query()["coverage"] == "untrusted"
