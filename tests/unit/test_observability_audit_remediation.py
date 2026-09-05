"""Regressions for the independent 2026-09-05 verification report.

All records are synthetic. No host application, production DB or resolver is used.
"""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import ObservabilityStore
from memorizz.observability.coverage import trace_coverage
from memorizz.observability.inspection import (
    build_causal_waterfall,
    compare_trace_windows,
)
from memorizz.observability.lineage import build_lineage_inspectors
from memorizz.observability.normalization import (
    TraceEvents,
    TraceSnapshot,
    normalize_trace_snapshot,
    select_trace_events,
)
from memorizz.observability.references import source_ids
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers.trace_tools import TraceSelection, _collect
from tests.unit.test_observability_normalization import incident_bundles


@pytest.fixture
def audit_ui(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "false")
    monkeypatch.setenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "metadata")
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "bundles")
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_DUAL_WRITE", "false")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    for row in incident_bundles():
        payload = json.loads(row["content"])
        for i, event in enumerate(payload["events"]):
            event["event_id"] = f"{row['root_trace_id']}-event-{i}"
            event["turn_id"] = "turn-" + row["root_trace_id"]
            event["grounding_source_ids"] = "[]"
        payload["events"][0].update(
            trace_kind="memory_context", memory_supplied_count=11
        )
        row["content"] = json.dumps(payload)
        provider.store({**row, "memory_id": row["_id"]}, MemoryType.SHARED_MEMORY)
    for i in range(16):
        provider.store(
            {
                "memory_id": f"conversation-{i}",
                "trace_memory_id": "memory-A",
                "thread_id": "affected-thread",
                "role": "user",
                "content": "SYNTHETIC PRIVATE",
                "timestamp": "2026-09-04T10:00:00Z",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
    agent = {"agent_id": "saved-agent", "memory_ids": ["memory-A"]}
    authorizer = Mock(return_value=True)
    app = create_app(resource_authorizer=authorizer)
    with patch.dict(
        state._state, provider=provider, provider_type="filesystem", connection_info={}
    ), patch.object(provider, "list_memagents", return_value=[agent]), patch.object(
        provider, "retrieve_memagent", return_value=agent
    ):
        yield TestClient(app), provider, authorizer, app
    provider.close()


def test_legacy_scope_counts_match_insights_health_and_combined_export(audit_ui):
    client, *_ = audit_ui
    params = {"agent_id": "saved-agent", "thread_id": "affected-thread"}
    analysis = client.get("/traces/analysis.json", params=params).json()
    health = client.get("/traces/health.json", params=params).json()
    combined = client.get(
        "/traces/events.json", params={**params, "store": "all"}
    ).json()
    assert (
        analysis["summary"]["events_analyzed"],
        health["normalized_events"],
        combined["normalized_events"],
    ) == (90, 90, 90)
    assert (
        analysis["coverage"]["source_counts"]
        == health["source_counts"]
        == combined["source_counts"]
        == {"conversation": 16, "bundle": 3}
    )
    assert health["missing_identity_events"] >= 16
    trace_only = client.get("/traces/events.json", params=params).json()
    assert trace_only["normalized_events"] == 74 and trace_only["store_selection"] == [
        "trace"
    ]
    assert combined["store_selection"] == ["conversation", "tool_log", "trace"]


def test_root_selection_counts_are_local_and_links_preserve_context(audit_ui):
    client, *_ = audit_ui
    params = {
        "agent_id": "saved-agent",
        "thread_id": "affected-thread",
        "root_trace_id": "root-2",
        "turn_id": "turn-root-2",
        "event_id": "root-2-event-5",
    }
    page = client.get("/traces", params=params)
    assert page.status_code == 200
    assert "26 events · 1 bundles" in page.text
    assert 'data-selected-event="true"' in page.text
    assert 'id="trace-agent-navigator" >' in page.text
    assert (
        "root_trace_id=root-2" in page.text and "event_id=root-2-event-5" in page.text
    )
    assert (
        "baseline_root=root-2" in page.text and "baseline_turn=turn-root-2" in page.text
    )
    health = client.get("/traces/health.json", params=params).json()
    analysis = client.get("/traces/analysis.json", params=params).json()
    assert health["normalized_events"] == analysis["summary"]["events_analyzed"] == 26
    assert health["bundle_count"] == health["source_rows"] == 1
    assert health["duplicates_removed"] is None
    assert health["parent_query_metadata"]["bundle_count"] == 3
    assert (
        client.get(
            "/traces", params={**params, "event_id": "root-0-event-5"}
        ).status_code
        == 404
    )


def test_comparison_does_not_reuse_parent_source_counts(audit_ui):
    client, *_ = audit_ui
    result = client.get(
        "/traces/compare.json",
        params={
            "agent_id": "saved-agent",
            "thread_id": "affected-thread",
            "baseline_root": "root-2",
            "candidate_root": "root-1",
        },
    ).json()
    assert result["baseline"]["normalized_events"] == 26
    assert result["candidate"]["normalized_events"] == 36
    for side in ("baseline", "candidate"):
        assert result[side]["bundle_count"] == result[side]["source_rows"] == 1
        assert result[side]["parent_query_metadata"]["source_rows"] == 19


def test_uninstrumented_coverage_and_empty_panels_are_explicit(audit_ui):
    client, *_ = audit_ui
    params = {
        "agent_id": "saved-agent",
        "thread_id": "affected-thread",
        "root_trace_id": "root-2",
    }
    report = client.get("/traces/analysis.json", params=params).json()
    assert report["coverage"]["read_completeness"] == "complete"
    assert (
        report["coverage"]["instrumentation_coverage"]
        == report["coverage"]["outcome_verification"]
        == "unknown"
    )
    assert report["coverage"]["coverage"] == "unknown"
    for name in ("artifacts", "output_contracts", "ui_deliveries", "verified_outcomes"):
        assert report["metric_states"][name] == {
            "value": None,
            "state": "no_recorded_evidence",
        }
    inspection = client.get("/traces/inspect.json", params=params).json()
    assert inspection["coverage"]["read_completeness"] == "complete"
    assert inspection["coverage"]["instrumentation_coverage"] == "unknown"
    page = client.get("/traces", params=params).text
    assert "All stored events loaded; end-to-end coverage unknown" in page
    assert "Trace coverage: COMPLETE" not in page
    for identity in (
        "trace-lineage-inspectors",
        "trace-artifact-provenance",
        "trace-output-contracts",
    ):
        assert f'id="{identity}"' in page
    assert "No source-to-artifact link can be verified" in page
    assert "does not establish interactive delivery" in page
    assert inspection["memory"][0]["observed_supplied_count"] == 11
    assert (
        inspection["memory"][0]["expected"]
        == inspection["memory"][0]["output_bound"]
        == []
    )


def test_unconfigured_hooks_are_visible_but_disabled(audit_ui):
    client, _, _, app = audit_ui
    app.state.trace_resource_authorizer = None
    capabilities = client.get("/traces/capabilities.json").json()["capabilities"]
    assert not any(value["available"] for value in capabilities.values())
    page = client.get("/traces?agent_id=saved-agent").text
    assert "setup required" in page
    assert 'type="submit" disabled>Resolve account' in page
    assert 'type="button" disabled>Create reviewable draft' in page
    assert "resource_authorizer" in page and "identity_resolver" in page
    assert (
        client.get(
            "/traces/replay-plan.json?agent_id=saved-agent&root_trace_id=root-2"
        ).status_code
        == 501
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ([], []),
        ("[]", []),
        ('["source-1", "source-2"]', ["source-1", "source-2"]),
        (["source-1"], ["source-1"]),
    ],
)
def test_replay_source_arrays_are_canonical_and_empty_means_no_checks(
    audit_ui, value, expected
):
    client, provider, authorizer, _ = audit_ui
    events = TraceEvents(
        [
            {
                "event_id": "event",
                "agent_id": "saved-agent",
                "root_trace_id": "root",
                "grounding_source_ids": value,
            }
        ],
        coverage={"coverage": "complete", "window_complete": True},
    )
    with patch("memorizz.ui.routers.trace_tools._collect", return_value=events):
        result = client.get(
            "/traces/replay-plan.json?agent_id=saved-agent&root_trace_id=root"
        )
    assert result.status_code == 200, result.text
    assert [ref["ref"] for ref in result.json()["resource_refs"]] == expected
    assert authorizer.call_count == len(expected)
    assert source_ids(value) == expected
    draft = ObservabilityStore(provider).create_replay_draft(
        events,
        created_by="operator",
        authorize_resource=lambda ref, scope: True,
        persist=False,
    )
    assert [ref["ref"] for ref in draft["resource_refs"]] == expected


@pytest.mark.parametrize(
    "value",
    ["not-json", '"source"', "{}", "null", [None], [1], [""], {"ref": "source"}],
)
def test_malformed_source_arrays_fail_closed_before_authorization(audit_ui, value):
    client, _, authorizer, _ = audit_ui
    with pytest.raises((ValueError, TypeError)):
        source_ids(value)
    events = TraceEvents(
        [{"event_id": "event", "grounding_source_ids": value}],
        coverage={"coverage": "complete"},
    )
    with patch("memorizz.ui.routers.trace_tools._collect", return_value=events):
        result = client.get(
            "/traces/replay-plan.json?agent_id=saved-agent&root_trace_id=root"
        )
    assert result.status_code == 409
    authorizer.assert_not_called()
    window = normalize_trace_snapshot(
        TraceSnapshot(
            conversation_rows=[{"event_id": "event", "grounding_source_ids": value}]
        )
    )
    assert window.metadata["read_completeness"] == "untrusted"
    assert any(e["code"] == "invalid_source_ids" for e in window.normalization_errors)


def test_legacy_tool_names_distinguish_waterfall_and_equal_count_substitutions():
    first = [{"event_id": "1", "kind": "tool_call", "tool_name": "get_analysis_by_id"}]
    second = [{"event_id": "2", "kind": "tool_call", "tool_name": "unrelated_tool"}]
    assert (
        build_causal_waterfall(first)[0]["nodes"][0]["operation"]
        == "get_analysis_by_id"
    )
    result = compare_trace_windows(first, second)
    assert (
        result["changes"]["operations"]["removed"][0]["value"]["operation"]
        == "get_analysis_by_id"
    )
    assert (
        result["changes"]["operations"]["added"][0]["value"]["operation"]
        == "unrelated_tool"
    )


def test_inspector_preserves_profile_evidence_across_pagination():
    identity = {"agent_id": "agent", "root_trace_id": "root", "turn_id": "turn"}
    pages = [
        {
            "items": [
                {
                    **identity,
                    "event_id": "1",
                    "kind": "model_result",
                    "coverage_profile": "interactive_response",
                }
            ],
            "next_cursor": "next",
            "coverage": "partial",
        },
        {
            "items": [{**identity, "event_id": "2", "kind": "ui_delivery"}],
            "next_cursor": None,
            "coverage": "complete",
        },
    ]
    with patch("memorizz.ui.routers.trace_tools._query", side_effect=pages):
        rows = _collect(TraceSelection(agent_id="agent", root_trace_id="root"))
    assert rows.coverage["read_completeness"] == "complete"
    assert rows.coverage["instrumentation_coverage"] == "partial"
    assert rows.coverage["profiles_declared"] == ["interactive_response"]
    assert rows.coverage["missing_stages"][0]["stage"] == "output_contract"
    assert len(rows.coverage["page_metadata"]) == 2


def test_observed_declared_stages_do_not_imply_verified_success():
    rows = [{"kind": "model_result", "coverage_profile": "chat_response"}]
    coverage = trace_coverage(rows, {"coverage": "complete"})
    assert coverage["instrumentation_coverage"] == "complete"
    assert coverage["outcome_verification"] == "unknown"
    assert "not proof of task success" in coverage["headline"]
    assert (
        trace_coverage(TraceEvents(rows, coverage={"read_completeness": "complete"}))[
            "read_completeness"
        ]
        == "complete"
    )


def test_memory_only_inspection_survives_absent_artifact():
    rows = [
        {
            "kind": "memory_supply",
            "task_id": "task",
            "input_refs": [{"resource_type": "analysis", "ref": "source"}],
        }
    ]
    result = build_lineage_inspectors(rows)
    assert result["memory"][0]["supplied"][0]["ref"] == "source"
    assert result["artifacts"] == []


def test_inspector_does_not_promote_truncated_terminal_page_to_complete():
    page = {
        "items": [{"agent_id": "agent", "root_trace_id": "root", "event_id": "event"}],
        "coverage": "partial",
        "truncated": True,
        "next_cursor": None,
    }
    with patch("memorizz.ui.routers.trace_tools._query", return_value=page):
        rows = _collect(TraceSelection(agent_id="agent", root_trace_id="root"))
    assert rows.coverage["read_completeness"] == "partial"
    assert rows.coverage["window_complete"] is False


def test_instantaneous_events_are_distinct_from_missing_span_completions():
    events = [
        {"event_id": "instant", "phase": "event", "kind": "memory_supply"},
        {"event_id": "start", "phase": "start", "kind": "model_call"},
    ]
    nodes = build_causal_waterfall(events)[0]["nodes"]
    assert [node["timing_state"] for node in nodes] == [
        "instantaneous",
        "missing_completion",
    ]


def test_cross_view_time_filters_use_child_timestamps_and_survive_links(audit_ui):
    client, *_ = audit_ui
    params = {
        "agent_id": "saved-agent",
        "thread_id": "affected-thread",
        "root_trace_id": "root-2",
        "start_time": "2026-09-04T10:00:25Z",
        "end_time": "2026-09-04T10:00:26Z",
    }
    health = client.get("/traces/health.json", params=params).json()
    analysis = client.get("/traces/analysis.json", params=params).json()
    exported = client.get(
        "/traces/events.json", params={**params, "store": "all"}
    ).json()
    assert (
        health["normalized_events"]
        == analysis["summary"]["events_analyzed"]
        == exported["normalized_events"]
        == 3
    )
    page = client.get("/traces", params=params)
    assert (
        page.status_code == 200 and "start_time=2026-09-04T10%3A00%3A25Z" in page.text
    )
    for endpoint in ("/traces", "/traces/health.json", "/traces/analysis.json"):
        assert (
            client.get(
                endpoint, params={**params, "start_time": "malformed"}
            ).status_code
            == 400
        )


def test_developer_example_declares_its_actual_capture_profile(tmp_path, monkeypatch):
    from examples.observability.host_workflow import run_example
    from memorizz.observability import coverage

    monkeypatch.setattr(coverage, "PROFILES", dict(coverage.PROFILES))
    monkeypatch.setattr(coverage, "OPTIONAL_STAGES", dict(coverage.OPTIONAL_STAGES))

    result = run_example(tmp_path / "example")
    assert (
        result["read_completeness"] == result["instrumentation_coverage"] == "complete"
    )
    assert result["events"] == 6 and result["outcome"] == "success"


def _selection_bundle(provider, *, turn="turn", memory="selection-memory", count=4):
    events = [
        {
            "event_id": f"selected-{i}",
            "trace_kind": "model_result",
            "task_id": "slides" if i % 2 == 0 else "document",
            "timestamp": "2026-09-04T10:00:00Z",
            "grounding_source_ids": json.dumps([f"source-{turn}-{i}"]),
            "content": f"SYNTHETIC {turn} {i}",
        }
        for i in range(count)
    ]
    ObservabilityStore(provider).record_trace_bundle(
        trace_context={
            "agent_id": "saved-agent",
            "thread_id": "selection-thread",
            "memory_id": memory,
            "root_trace_id": "selection-root",
            "turn_id": turn,
            "run_id": "selection-run",
        },
        events=events,
    )


def test_combined_export_preserves_task_and_thread_memory(audit_ui):
    client, provider, *_ = audit_ui
    _selection_bundle(provider)
    _selection_bundle(provider, turn="other", memory="other-memory")
    params = {
        "agent_id": "saved-agent",
        "root_trace_id": "selection-root",
        "thread_memory_id": "selection-memory",
        "task_id": "slides",
    }
    reports = [
        client.get("/traces/analysis.json", params=params),
        client.get("/traces/health.json", params=params),
        client.get("/traces/events.json", params={**params, "store": "all"}),
    ]
    assert all(r.status_code == 200 for r in reports), [r.text for r in reports]
    analysis, health, export = [r.json() for r in reports]
    assert (
        analysis["summary"]["events_analyzed"]
        == health["normalized_events"]
        == export["normalized_events"]
        == 2
    )
    assert {e["turn_id"] for e in export["items"]} == {"turn"}
    assert {e["task_id"] for e in export["items"]} == {"slides"}
    # A paginated per-store endpoint must not silently ignore unsupported scope.
    assert client.get("/traces/events.json", params=params).status_code == 400


@pytest.mark.parametrize("persist", [False, True])
def test_replay_preserves_turn_task_run_and_memory(audit_ui, persist):
    client, provider, authorizer, _ = audit_ui
    _selection_bundle(provider)
    _selection_bundle(provider, turn="other", memory="other-memory")
    params = {
        "agent_id": "saved-agent",
        "root_trace_id": "selection-root",
        "thread_id": "selection-thread",
        "thread_memory_id": "selection-memory",
        "turn_id": "turn",
        "task_id": "slides",
        "run_id": "selection-run",
        "start_time": "2026-09-04T09:59:59Z",
        "end_time": "2026-09-04T10:00:01Z",
    }
    response = (
        client.post("/traces/replays", json=params)
        if persist
        else client.get("/traces/replay-plan.json", params=params)
    )
    assert response.status_code == 200, response.text
    draft = response.json()
    assert [e["event_id"] for e in draft["evidence_refs"]] == [
        "selected-0",
        "selected-2",
    ]
    assert {e["turn_id"] for e in draft["evidence_refs"]} == {"turn"}
    assert {ref["ref"] for ref in draft["resource_refs"]} == {
        "source-turn-0",
        "source-turn-2",
    }
    assert authorizer.call_count == 2
    assert draft["replay_policy"]["execution_enabled"] is False


def test_reveal_requires_unambiguous_event_and_preserves_turn(audit_ui, monkeypatch):
    client, provider, *_ = audit_ui
    monkeypatch.setenv("MEMORIZZ_UI_TRACE_CONTENT_MODE", "redacted")
    _selection_bundle(provider)
    _selection_bundle(provider, turn="other")
    params = {
        "agent_id": "saved-agent",
        "root_trace_id": "selection-root",
        "event_id": "selected-0",
    }
    assert client.post("/traces/reveal", json=params).status_code == 409
    response = client.post("/traces/reveal", json={**params, "turn_id": "turn"})
    assert response.status_code == 200, response.text
    assert response.json()["content"] == "SYNTHETIC turn 0"


def test_comparison_caps_each_selected_root_not_the_parent_window(audit_ui):
    client, provider, *_ = audit_ui
    _selection_bundle(provider, count=1100)
    response = client.get(
        "/traces/compare.json",
        params={
            "agent_id": "saved-agent",
            "baseline_root": "selection-root",
            "candidate_root": "root-2",
        },
    )
    assert response.status_code == 200, response.text
    baseline, candidate = response.json()["baseline"], response.json()["candidate"]
    assert baseline["normalized_events"] == 1000
    assert baseline["read_completeness"] == "partial"
    assert candidate["normalized_events"] == 26
    assert candidate["read_completeness"] == "complete"


@pytest.mark.parametrize(
    "endpoint",
    [
        "/traces",
        "/traces/analysis.json",
        "/traces/health.json",
        "/traces/events.json",
        "/traces/compare.json",
    ],
)
def test_agent_association_failure_is_not_a_complete_narrower_read(audit_ui, endpoint):
    client, provider, *_ = audit_ui
    with patch.object(
        provider, "list_memagents", side_effect=RuntimeError("PRIVATE ERROR")
    ):
        response = client.get(
            endpoint,
            params={
                "agent_id": "saved-agent",
                "store": "all",
                "baseline_root": "root-1",
                "candidate_root": "root-2",
            },
        )
    assert response.status_code == 503
    assert "PRIVATE ERROR" not in response.text


@pytest.mark.parametrize(
    "metadata",
    [
        {"coverage": "unknown"},
        {"coverage": "complete", "read_completeness": "unknown"},
    ],
)
def test_unknown_earlier_page_cannot_be_promoted_by_final_complete_page(metadata):
    pages = [
        {
            "items": [{"event_id": "1", "root_trace_id": "root"}],
            "next_cursor": "next",
            **metadata,
        },
        {"items": [{"event_id": "2", "root_trace_id": "root"}], "coverage": "complete"},
    ]
    with patch("memorizz.ui.routers.trace_tools._query", side_effect=pages):
        rows = _collect(TraceSelection(agent_id="agent", root_trace_id="root"))
    assert rows.coverage["read_completeness"] == "partial"
    assert rows.coverage["window_complete"] is False


@pytest.mark.parametrize(
    "failure", ["explicit_untrusted", "repeated_event", "repeated_cursor"]
)
def test_untrusted_or_nonadvancing_pagination_fails_closed(failure):
    from fastapi import HTTPException

    pages = [
        {
            "items": [{"event_id": "1", "root_trace_id": "root"}],
            "coverage": "partial",
            "next_cursor": "next",
        },
        {"items": [{"event_id": "2", "root_trace_id": "root"}], "coverage": "complete"},
    ]
    if failure == "explicit_untrusted":
        pages[0]["read_completeness"] = "untrusted"
    elif failure == "repeated_event":
        pages[1]["items"] = pages[0]["items"]
    else:
        pages[1]["next_cursor"] = "next"
    with patch("memorizz.ui.routers.trace_tools._query", side_effect=pages):
        with pytest.raises(HTTPException) as error:
            _collect(TraceSelection(agent_id="agent", root_trace_id="root"))
    assert error.value.status_code == 409


def test_valid_native_cursor_pages_can_form_a_complete_window():
    pages = [
        {
            "items": [{"event_id": "1", "root_trace_id": "root"}],
            "coverage": "partial",
            "next_cursor": "next",
            "truncated": True,
        },
        {
            "items": [{"event_id": "2", "root_trace_id": "root"}],
            "coverage": "complete",
            "window_complete": False,
        },
    ]
    for page in pages:
        page["coverage_scope"] = "stored_event_page"
    with patch("memorizz.ui.routers.trace_tools._query", side_effect=pages):
        rows = _collect(TraceSelection(agent_id="agent", root_trace_id="root"))
    assert rows.coverage["read_completeness"] == "complete"
    assert rows.coverage["window_complete"] is True


def test_truncation_never_downgrades_untrusted_to_partial():
    events = TraceEvents(
        [{"event_id": "1"}, {"event_id": "2"}],
        coverage={"read_completeness": "untrusted"},
    )
    selected = select_trace_events(events, limit=1)
    assert trace_coverage(selected)["read_completeness"] == "untrusted"


def test_supported_selection_filters_are_documented_in_openapi(audit_ui):
    _, _, _, app = audit_ui
    paths = app.openapi()["paths"]
    for path in [
        "/traces",
        "/traces/health.json",
        "/traces/compare.json",
        "/traces/analysis.json",
        "/traces/inspect.json",
        "/traces/replay-plan.json",
        "/traces/events.json",
    ]:
        params = {p["name"] for p in paths[path]["get"]["parameters"]}
        assert {
            "run_id",
            "start_time",
            "end_time",
            "task_id",
            "thread_memory_id",
        } <= params, path


@pytest.mark.parametrize("value", [[], {}, 1, "", "unsafe@example.com"])
def test_invalid_legacy_task_ids_mark_evidence_untrusted(value):
    window = normalize_trace_snapshot(
        TraceSnapshot(conversation_rows=[{"event_id": "event", "task_id": value}])
    )
    assert window.metadata["read_completeness"] == "untrusted"
    assert "task_id" not in window.events[0]


@pytest.mark.parametrize("value", [None, [], "[]", '["source"]'])
def test_direct_diagnostics_also_decode_legacy_source_arrays(value):
    from memorizz.observability.diagnostics import diagnose_trace

    rows = [
        {"kind": "memory_supply", "grounding_source_ids": value},
        {
            "kind": "artifact_persisted",
            "persistence_verified": True,
            "artifact_status": "created",
            "output_refs": [{"resource_type": "slide_deck", "ref": "deck"}],
            "input_refs": [{"resource_type": "analysis", "ref": "source"}],
        },
    ]
    findings = diagnose_trace(rows)
    assert "artifact_source_mismatch" not in {f["id"] for f in findings}
    mismatch = diagnose_trace(
        [{**rows[0], "grounding_source_ids": '["other-source"]'}, rows[1]]
    )
    assert "artifact_source_mismatch" in {f["id"] for f in mismatch}
    with pytest.raises(ValueError):
        diagnose_trace([{**rows[0], "grounding_source_ids": "malformed"}])


def test_incomplete_replay_is_rejected_before_host_authorization(audit_ui):
    client, _, authorizer, _ = audit_ui
    events = TraceEvents(
        [{"grounding_source_ids": ["source"]}],
        coverage={"read_completeness": "partial"},
    )
    with patch("memorizz.ui.routers.trace_tools._collect", return_value=events):
        response = client.get(
            "/traces/replay-plan.json?agent_id=saved-agent&root_trace_id=root"
        )
    assert response.status_code == 409
    authorizer.assert_not_called()


def test_invalid_replay_time_range_is_rejected_before_query(audit_ui):
    client, *_ = audit_ui
    with patch("memorizz.ui.routers.trace_tools._query") as query:
        response = client.get(
            "/traces/replay-plan.json?agent_id=saved-agent&root_trace_id=root&start_time=invalid"
        )
    assert response.status_code == 400
    query.assert_not_called()
