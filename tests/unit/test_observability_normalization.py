"""Regression coverage for the OpenSpeech bundle/agent association incident."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    TraceSnapshot,
    analyze_trace_events,
    normalize_trace_snapshot,
)
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers.traces import (
    _build_agent_thread_rows,
    _build_agent_trace_metrics,
    _load_agent_trace_events,
)


def incident_bundles():
    rows = []
    offset = 0
    for bundle_index, size in enumerate((12, 36, 26)):
        events = []
        for index in range(size):
            number = offset + index
            kind = "tool_call" if number < 14 else "model_result"
            events.append(
                {
                    "trace_kind": kind,
                    "trace_id": f"event-{number}",
                    "span_id": f"span-{number}",
                    "timestamp": f"2026-09-04T10:00:{number // 2:02d}.{number % 2 * 500:03d}Z",
                    "logical_tool_name": "lookup" if kind == "tool_call" else "",
                    "model": "fixture-model",
                    "provider": "fixture-provider",
                    "input_tokens": 12,
                    "status": "success",
                }
            )
        rows.append(
            {
                "_id": f"bundle-{bundle_index}",
                "record_type": "observability_trace_bundle",
                "agent_id": "saved-agent",
                "trace_memory_id": "memory-B",
                "memory_id": "memory-B",
                "thread_id": "affected-thread",
                "root_trace_id": f"root-{bundle_index}",
                "timestamp": "2026-09-04T10:01:00Z",
                "content": json.dumps(
                    {"type": "trace_bundle", "version": 2, "events": events}
                ),
            }
        )
        offset += size
    return rows


def test_incident_bundle_association_counts_and_analysis():
    agent = {"agent_id": "saved-agent", "memory_ids": ["memory-A"]}
    rows = incident_bundles()
    with patch.dict(state._state, {"provider": object()}):
        events = _load_agent_trace_events(
            agent,
            conversation_docs=rows,
            tool_log_docs=[],
            thread_id="affected-thread",
            thread_memory_id="memory-B",
        )
        metrics = _build_agent_trace_metrics([agent], conversation_docs=rows)
        threads = _build_agent_thread_rows([agent], conversation_docs=rows)
    assert len(events) == 74
    assert metrics["saved-agent"]["event_count"] == 74
    assert metrics["saved-agent"]["bundle_count"] == 3
    assert threads["saved-agent"][0]["event_count"] == 74
    assert len({event["timestamp"] for event in events}) == 74
    assert all(event["model"] == "fixture-model" for event in events)
    assert analyze_trace_events(events)["summary"]["tool_calls"] == 14


@pytest.fixture(params=["filesystem", "mongodb", "compatibility"])
def incident_provider(request, tmp_path):
    if request.param == "filesystem":
        provider = FileSystemProvider(
            FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
        )
        for row in incident_bundles():
            provider.store({**row, "memory_id": row["_id"]}, MemoryType.SHARED_MEMORY)
    elif request.param == "mongodb":
        mongomock = pytest.importorskip("mongomock")
        from memorizz.memory_provider.mongodb.provider import MongoDBProvider

        provider = MongoDBProvider.__new__(MongoDBProvider)
        provider.db = mongomock.MongoClient()["incident"]
        provider.db[MemoryType.SHARED_MEMORY.value].insert_many(incident_bundles())
    else:
        from memorizz.memory_provider.base import MemoryProvider

        class CompatibilityProvider:
            query_observability_records = MemoryProvider.query_observability_records
            query_trace_events = MemoryProvider.query_trace_events

            def list_all(self, *args, **kwargs):
                return [
                    {
                        "memory_id": row["_id"],
                        "content": json.dumps(
                            {
                                **{k: v for k, v in row.items() if k != "content"},
                                **json.loads(row["content"]),
                            }
                        ),
                    }
                    for row in incident_bundles()
                ]

        provider = CompatibilityProvider()
    return provider


def test_event_pagination_preserves_every_child_and_scope(incident_provider):
    cursor, ids, pages = None, [], []
    while True:
        page = incident_provider.query_trace_events(
            agent_ids=["saved-agent"],
            memory_ids=["memory-A"],
            thread_id="affected-thread",
            limit=7,
            cursor=cursor,
        )
        assert len(page["items"]) <= 7
        ids.extend(event["event_id"] for event in page["items"])
        pages.append(page)
        cursor = page["next_cursor"]
        if not cursor:
            break
        assert len(pages) < 20
    assert len(ids) == len(set(ids)) == 74
    assert pages[-1]["coverage"] == "complete"
    assert pages[-1]["window_complete"] is False  # final page != full history
    with pytest.raises(ValueError, match="cursor"):
        incident_provider.query_trace_events(
            agent_ids=["another-agent"], cursor=pages[0]["next_cursor"]
        )
    with pytest.raises(ValueError, match="cursor"):
        incident_provider.query_trace_events(cursor="not-valid-json")
    tools = incident_provider.query_trace_events(
        agent_ids=["saved-agent"], event_kinds=["tool_call"], tool_name="lookup"
    )
    assert len(tools["items"]) == 14
    assert tools["window_complete"] is True


def test_time_filters_use_child_time_not_bundle_write_time(incident_provider):
    page = incident_provider.query_trace_events(
        agent_ids=["saved-agent"],
        start_time="2026-09-04T10:00:00Z",
        end_time="2026-09-04T10:00:01Z",
    )
    assert len(page["items"]) == 3
    with pytest.raises(ValueError, match="start_time"):
        incident_provider.query_trace_events(start_time="invalid")
    with pytest.raises(ValueError, match="after"):
        incident_provider.query_trace_events(
            start_time="2026-09-05", end_time="2026-09-04"
        )


def test_incident_html_analysis_and_export_agree(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    agent = {"agent_id": "saved-agent", "memory_ids": ["memory-A"]}
    for row in incident_bundles():
        provider.store({**row, "memory_id": row["_id"]}, MemoryType.SHARED_MEMORY)
    with patch.dict(state._state, {"provider": provider}), patch.object(
        provider, "list_memagents", return_value=[agent]
    ), patch.object(provider, "retrieve_memagent", return_value=agent):
        client = TestClient(create_app())
        suffix = "agent_id=saved-agent&thread_id=affected-thread"
        html = client.get("/traces?" + suffix)
        analysis = client.get("/traces/analysis.json?" + suffix)
        exported = client.get("/traces/events.json?" + suffix)
        raw = client.get("/traces/events.json?view=bundles&" + suffix)
    assert (
        html.status_code
        == analysis.status_code
        == exported.status_code
        == raw.status_code
        == 200
    )
    assert "74 events · 3 bundles" in html.text
    assert "fixture-model" in html.text
    assert (
        analysis.json()["summary"]["events_analyzed"]
        == exported.json()["normalized_events"]
        == 74
    )
    assert analysis.json()["summary"]["tool_calls"] == 14
    assert exported.json()["bundle_count"] == len(raw.json()["items"]) == 3
    assert all(e.get("event_id") for e in exported.json()["items"])


@pytest.mark.parametrize(
    "bad_payload",
    [
        "not-json",
        '{"type":"trace_bundle","version":99,"events":[]}',
        '{"type":"trace_bundle","events":[null]}',
    ],
)
def test_malformed_bundles_are_unknown_never_false_zero(bad_payload):
    row = {**incident_bundles()[0], "content": bad_payload}
    window = normalize_trace_snapshot(TraceSnapshot(bundle_rows=[row]))
    report = analyze_trace_events(window.events)
    assert report["coverage"]["coverage"] == "untrusted"
    assert report["metric_states"]["tool_calls"] == {"value": None, "state": "unknown"}
    assert "trace_coverage_incomplete" in {i["id"] for i in report["insights"]}


def test_partial_bundle_keeps_valid_events_deduplicates_and_reports_errors():
    row = incident_bundles()[0]
    payload = json.loads(row["content"])
    payload["events"] += [payload["events"][0], None]
    row["content"] = json.dumps(payload)
    window = normalize_trace_snapshot(TraceSnapshot(bundle_rows=[row]))
    assert len(window.events) == 12
    assert window.duplicate_count == 1
    assert len(window.normalization_errors) == 1
    assert (
        analyze_trace_events(window.events)["metric_states"]["tool_calls"]["state"]
        == "partial"
    )


def test_limits_queries_and_profiles_surface_incomplete_coverage():
    window = normalize_trace_snapshot(
        TraceSnapshot(bundle_rows=incident_bundles()), limit=10
    )
    assert window.truncated and len(window.events) == 10
    failed = normalize_trace_snapshot(
        TraceSnapshot(query_metadata={"query_errors": ["trace"]})
    )
    assert failed.metadata["coverage"] == "untrusted"
    report = analyze_trace_events(
        [
            {
                "kind": "intent_plan",
                "root_trace_id": "root",
                "coverage_profile": "interactive_response",
            }
        ]
    )
    assert report["coverage"]["coverage"] == "partial"
    assert {s["stage"] for s in report["coverage"]["missing_stages"]} == {
        "model_result",
        "output_contract",
        "ui_delivery",
    }


def test_scope_filters_children_and_durable_tools_use_alternative_identity():
    row = incident_bundles()[0]
    payload = json.loads(row["content"])
    payload["events"][0]["thread_id"] = "another-thread"
    row["content"] = json.dumps(payload)
    tool = {
        "agent_id": "saved-agent",
        "memory_id": "memory-B",
        "thread_id": "affected-thread",
        "tool_name": "lookup",
        "tool_call_id": "call-1",
    }
    window = normalize_trace_snapshot(
        TraceSnapshot(bundle_rows=[row], tool_rows=[tool]),
        agent_ids={"saved-agent"},
        memory_ids={"memory-A"},
        thread_id="affected-thread",
    )
    assert len(window.events) == 12
    assert sum(e["kind"] == "execution_log" for e in window.events) == 1
    assert all(e["thread_id"] == "affected-thread" for e in window.events)


def test_legacy_compaction_preserves_bounded_source_list():
    from memorizz import MemAgent

    agent = MemAgent()
    rows = agent._build_trace_bundle_events(
        [
            {
                "trace_kind": "context_provenance",
                "grounding_source_ids": ["analysis-1", {"secret": "no"}],
                "prompt": "private",
            }
        ]
    )
    assert rows[0]["grounding_source_ids"] == ["analysis-1"]
    assert "prompt" not in rows[0]


def test_metadata_redaction_covers_attributes_and_scopes_pseudonyms(monkeypatch):
    from memorizz.ui.security import redact_trace_events

    monkeypatch.setenv("MEMORIZZ_UI_PSEUDONYM_KEY", "test-key")
    source = {
        "user_id": "u1",
        "content": "raw transcript",
        "arguments": {"password": "secret"},
        "attributes": {"fallback_reason": "email@example.com"},
    }
    first = redact_trace_events([source], mode="metadata")[0]
    assert "raw transcript" not in json.dumps(first)
    assert "email@example.com" not in json.dumps(first)
    assert "arguments" not in first
    assert (
        first["user_id"] == redact_trace_events([source], mode="metadata")[0]["user_id"]
    )
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_SCOPE", "other")
    assert (
        first["user_id"] != redact_trace_events([source], mode="metadata")[0]["user_id"]
    )
    assert source["user_id"] == "u1"
