"""Read-only operator views: pairing, coverage, structural diff and access."""

import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    ObservabilityStore,
    TraceContext,
    build_causal_waterfall,
    build_trace_health,
    compare_trace_windows,
)
from memorizz.observability.normalization import TraceEvents
from memorizz.ui import state
from memorizz.ui.app import create_app


def test_waterfall_pairs_phases_and_orders_parent_before_child():
    rows = [
        dict(
            event_id="1",
            span_id="parent",
            phase="start",
            status="started",
            timestamp="2026-09-04T00:00:00Z",
        ),
        dict(
            event_id="2",
            span_id="child",
            parent_span_id="parent",
            timestamp="2026-09-04T00:00:00.100Z",
        ),
        dict(
            event_id="3",
            span_id="parent",
            phase="result",
            status="success",
            timestamp="2026-09-04T00:00:00.500Z",
        ),
    ]
    parent, child = build_causal_waterfall(rows)[0]["nodes"]
    assert parent["event_ids"] == ["1", "3"]
    assert parent["duration_ms"] == 500
    assert parent["status"] == "success"
    assert child["depth"] == 1
    assert child["duration_ms"] is None
    assert child["offset_ms"] == 100
    assert not child["missing_parent"]


def test_waterfall_terminal_status_wins_over_equal_timestamp_start():
    rows = [
        dict(event_id="result", span_id="span", phase="result", status="success"),
        dict(event_id="start", span_id="span", phase="start", status="started"),
    ]
    assert build_causal_waterfall(rows)[0]["nodes"][0]["status"] == "success"


def test_waterfall_survives_cycles_missing_causes_and_deep_chains():
    rows = [
        dict(event_id=f"e{i}", span_id=str(i), parent_span_id=str(i - 1))
        for i in range(1500)
    ]
    rows += [
        dict(event_id="a", span_id="a", parent_span_id="b"),
        dict(event_id="b", span_id="b", parent_span_id="a"),
        dict(event_id="c", span_id="c", caused_by_event_id="absent"),
    ]
    nodes = build_causal_waterfall(rows)[0]["nodes"]
    assert len(nodes) == len(rows)
    assert sum(n["cycle"] for n in nodes) == 2
    assert max(n["depth"] for n in nodes) == 64
    assert nodes[0]["missing_parent"]
    assert next(n for n in nodes if n["span_id"] == "c")["missing_cause"]


def test_waterfall_does_not_merge_span_ids_across_turns_or_users():
    rows = [
        dict(event_id=str(i), span_id="same", **identity)
        for i, identity in enumerate(
            ({}, {"turn_id": "different"}, {"user_id": "different"})
        )
    ]
    assert len(build_causal_waterfall(rows)) == 3


@pytest.mark.parametrize("duration", ["invalid", float("nan"), float("inf"), -1, True])
def test_legacy_invalid_durations_remain_unknown(duration):
    node = build_causal_waterfall([dict(event_id="id", duration_ms=duration)])[0][
        "nodes"
    ][0]
    assert node["duration_ms"] is None


def test_health_empty_and_partial_are_not_false_zero_guarantees():
    assert build_trace_health([])["coverage"] == "unknown"
    health = build_trace_health(
        TraceEvents(
            [dict(event_id="1", kind="model_result")],
            coverage={"coverage": "partial", "truncated": True},
        )
    )
    assert health["coverage"] == "partial"
    assert health["missing_identity_events"] == 1
    assert health["write_failures"] is health["dropped_events"] is None


def test_compare_reports_structural_counts_without_content_and_is_immutable():
    baseline = TraceEvents(
        [
            dict(
                event_id="b",
                kind="output_contract",
                contract_name="map",
                parser_status="failed",
                content="PRIVATE RESPONSE",
                input_refs=[{"prompt": "PRIVATE REF"}],
            )
        ],
        coverage={"coverage": "partial"},
    )
    candidate = TraceEvents(
        [
            dict(
                event_id="c",
                kind="output_contract",
                contract_name="map",
                parser_status="success",
                content="PRIVATE CANDIDATE",
            )
        ],
        coverage={"coverage": "complete"},
    )
    original = deepcopy(baseline)
    comparison = compare_trace_windows(baseline, candidate)
    assert comparison["baseline"]["coverage"] == "partial"
    assert "contract_parse_failed" in comparison["findings"]["only_in_baseline"]
    assert comparison["changes"]["contracts"]["added"][0]["count"] == 1
    assert "PRIVATE" not in json.dumps(comparison)
    assert baseline == original


@pytest.fixture
def inspection_client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "metadata")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    for root, success in (("baseline", False), ("candidate", True)):
        context = TraceContext(
            agent_id="agent",
            thread_id="thread",
            root_trace_id=root,
            run_id="run",
            turn_id="turn",
        )
        recorder = ObservabilityRecorder(provider, context)
        recorder.record_intent("task", required_contracts=["map"])
        recorder.record_contract_result(
            "map",
            opening_marker_found=True,
            closing_marker_found=success,
            parser_status="success" if success else "failed",
            task_id="task",
        )
        recorder.record_delivery(
            "map", emitted=success, acknowledged=success, task_id="task"
        )
    ObservabilityStore(provider).record_trace_bundle(
        trace_context=dict(
            agent_id="agent", thread_id="thread", root_trace_id="legacy", turn_id="turn"
        ),
        events=[
            dict(
                trace_kind="model_result",
                content="PRIVATE MODEL RESPONSE",
                model="fixture-model",
            )
        ],
    )
    agent = dict(agent_id="agent", memory_ids=["unrelated-registration"])
    with patch.dict(state._state, {"provider": provider}), patch.object(
        provider, "list_memagents", return_value=[agent]
    ), patch.object(provider, "retrieve_memagent", return_value=agent):
        yield TestClient(create_app()), provider


def test_health_and_comparison_html_json_no_writes(inspection_client):
    client, provider = inspection_client
    before = provider.list_all(MemoryType.SHARED_MEMORY)
    with patch.object(provider, "store", side_effect=AssertionError("must not write")):
        health = client.get("/traces/health.json?agent_id=agent")
        assert health.status_code == 200
        assert health.json()["normalized_events"] == 7
        assert health.json()["schema_counts"] == {"2": 1, "3": 6}
        assert health.json()["write_failures"] is None
        assert health.headers["cache-control"].startswith("no-store")
        assert "Observability health" in client.get("/traces/health").text
        params = dict(
            agent_id="agent", baseline_root="baseline", candidate_root="candidate"
        )
        diff = client.get("/traces/compare.json", params=params)
        assert diff.status_code == 200
        assert "contract_parse_failed" in diff.json()["findings"]["only_in_baseline"]
        page = client.get("/traces/compare", params=params)
        assert page.status_code == 200
        assert "PRIVATE" not in page.text + diff.text
        assert "Causal waterfall" in client.get("/traces?agent_id=agent").text
    assert provider.list_all(MemoryType.SHARED_MEMORY) == before


def test_health_query_failures_are_visible_and_do_not_leak_exception_content(
    inspection_client,
):
    client, provider = inspection_client
    with patch.object(
        provider,
        "query_observability_records",
        side_effect=RuntimeError("PRIVATE DATABASE URI"),
    ):
        result = client.get("/traces/health.json")
    assert result.status_code == 200
    assert result.json()["coverage"] == "untrusted"
    assert len(result.json()["failed_queries"]) == 3
    assert "PRIVATE" not in result.text


def test_compare_validates_selection_and_cannot_cross_agent_scope(inspection_client):
    client, _ = inspection_client
    assert client.get("/traces/compare.json").status_code == 400
    assert client.get("/traces/compare").status_code == 200
    assert (
        client.get(
            "/traces/compare.json?agent_id=other&baseline_root=baseline&candidate_root=candidate"
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/traces/compare.json?agent_id=agent&baseline_root=baseline&candidate_root=candidate&candidate_turn=other"
        ).status_code
        == 404
    )


def test_event_export_filters_root_kind_status_and_ref(inspection_client):
    client, provider = inspection_client
    result = client.get(
        "/traces/events.json?agent_id=agent&root_trace_id=candidate&event_kind=ui_delivery&status=success"
    )
    assert result.status_code == 200
    assert len(result.json()["items"]) == 1
    assert result.json()["items"][0]["root_trace_id"] == "candidate"
    assert (
        client.get(
            "/traces/events.json?view=bundles&root_trace_id=candidate"
        ).status_code
        == 400
    )
    assert not client.get("/traces/events.json?resource_ref=absent").json()["items"]


def test_new_views_use_existing_authentication(inspection_client, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "test-token")
    client = TestClient(create_app())
    for path in (
        "/traces/health",
        "/traces/health.json",
        "/traces/compare",
        "/traces/compare.json",
    ):
        assert client.get(path).status_code == 401
    assert (
        client.get(
            "/traces/health.json", headers={"Authorization": "Bearer test-token"}
        ).status_code
        == 200
    )
