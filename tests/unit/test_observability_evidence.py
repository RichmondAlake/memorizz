"""Outcome evidence must not leak across tenants, turns, tasks or revisions."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from memorizz.observability import ObservabilityRecorder, TraceContext
from memorizz.observability.coverage import trace_coverage
from memorizz.observability.diagnostics import diagnose_trace
from memorizz.observability.normalization import TraceEvents


@pytest.fixture
def rows():
    identity = dict(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        turn_id="turn",
        run_id="run",
    )
    source = dict(
        resource_type="analysis",
        ref="source",
        version="1",
        role="authoritative_current_source",
    )
    artifact = dict(resource_type="doc", ref="doc", version="1")
    events = [
        dict(
            kind="intent_plan",
            task_id="one",
            input_refs=[source],
            expected_artifact_types=["doc"],
            required_contracts=["map"],
        ),
        dict(
            kind="application_action",
            phase="result",
            span_id="producer",
            status="success",
        ),
        dict(
            kind="artifact_persisted",
            parent_span_id="producer",
            persistence_verified=True,
            artifact_status="created",
            input_refs=[source],
            output_refs=[artifact],
        ),
        dict(
            kind="output_contract",
            contract_name="map",
            parser_status="success",
            opening_marker_found=True,
            closing_marker_found=True,
            required=True,
        ),
        dict(
            kind="ui_delivery",
            contract_name="map",
            emitted=True,
            acknowledged=True,
            input_refs=[artifact],
        ),
    ]
    return TraceEvents(
        [
            {
                **identity,
                "event_id": str(i),
                "timestamp": f"2026-09-04T10:00:0{i}Z",
                **event,
            }
            for i, event in enumerate(events)
        ],
        coverage={"coverage": "complete"},
    )


def outcome(rows, **kwargs):
    context = TraceContext(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        turn_id="turn",
        run_id="run",
    )
    recorder = ObservabilityRecorder(Mock(), context)
    recorder.record_event = Mock(side_effect=lambda *args, **kwargs: kwargs)
    return recorder.reconcile_outcome(rows, **kwargs)["attributes"]["verified"]


def codes(rows):
    return {item["id"] for item in diagnose_trace(rows)}


def test_complete_evidence_succeeds_but_plain_list_requires_query_metadata(rows):
    assert outcome(rows)
    assert not outcome(list(rows))
    assert outcome(list(rows), normalization_metadata={"coverage": "complete"})


@pytest.mark.parametrize(
    "field",
    ["application_id", "agent_id", "user_id", "thread_id", "root_trace_id", "turn_id"],
)
def test_other_identity_cannot_complete_current_turn(rows, field):
    rows[-1][field] = "other"
    assert not outcome(rows)
    assert "contract_not_emitted" in codes(rows)


@pytest.mark.parametrize(
    "metadata",
    [
        {"coverage": "partial"},
        {"coverage": "unknown"},
        {"coverage": "untrusted"},
        {"coverage": "complete", "truncated": True},
        {"coverage": "complete", "normalization_errors": 1},
    ],
)
def test_partial_or_untrusted_evidence_cannot_verify(rows, metadata):
    rows.coverage = metadata
    assert not outcome(rows)


def test_final_continuation_page_cannot_verify_full_outcome(rows):
    rows.coverage = {"coverage": "complete", "window_complete": False}
    assert not outcome(rows)


def test_deleted_and_failed_updated_artifacts_do_not_satisfy_intent(rows):
    update = {
        **rows[2],
        "event_id": "delete",
        "timestamp": "2026-09-04T10:00:06Z",
        "artifact_status": "deleted",
    }
    rows.append(update)
    assert "requested_artifact_missing" in codes(rows)
    assert not outcome(rows)
    update.update(artifact_status="failed", persistence_verified=False)
    assert "artifact_persistence_unverified" in codes(rows)
    assert not outcome(rows)


def test_delivery_of_old_revision_cannot_verify_new_revision(rows):
    rows[2]["output_refs"][0]["version"] = "2"
    # Break the intentionally shared reference before asserting revision mismatch.
    rows[-1]["input_refs"] = [dict(resource_type="doc", ref="doc", version="1")]
    assert not outcome(rows)


def test_old_delivery_cannot_complete_new_parse_or_persistence(rows):
    rows[3]["timestamp"] = "2026-09-04T10:00:09Z"
    assert "contract_not_emitted" in codes(rows)
    assert not outcome(rows)


def test_repair_supersedes_old_failure_but_needs_fresh_delivery(rows):
    old = {
        **rows[3],
        "event_id": "old",
        "timestamp": "2026-09-04T09:00:00Z",
        "parser_status": "failed",
        "closing_marker_found": False,
    }
    rows.insert(0, old)
    assert not codes(rows)
    assert outcome(rows)


def test_ambiguous_unlabelled_evidence_does_not_complete_multiple_tasks(rows):
    rows.append({**rows[0], "event_id": "intent-two", "task_id": "two"})
    assert not outcome(rows)
    assert not outcome(rows, task_id="one")
    for row in rows[1:-1]:
        row["task_id"] = "one"
    assert outcome(rows, task_id="one")
    assert not outcome(rows, task_id="two")
    assert not outcome(rows)


def test_producer_must_finish_and_all_authoritative_sources_must_match(rows):
    rows[1].update(status="started", phase="start")
    assert "artifact_producer_incomplete" in codes(rows)
    assert not outcome(rows)
    rows[1].update(status="success", phase="result")
    rows[0]["input_refs"] = [
        *rows[0]["input_refs"],
        dict(
            resource_type="analysis", ref="second", role="authoritative_current_source"
        ),
    ]
    assert "artifact_source_mismatch" in codes(rows)


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("ownership_verified", False, "artifact_ownership_unverified"),
        ("provenance_status", "missing", "artifact_provenance_invalid"),
    ],
)
def test_explicit_failed_resource_validation_cannot_verify(rows, field, value, code):
    rows[2]["input_refs"][0][field] = value
    assert code in codes(rows)
    assert not outcome(rows)


def test_absence_findings_have_evidence_and_lower_confidence_on_partial_window(rows):
    rows.pop(1)
    rows.coverage = {"coverage": "partial"}
    finding = next(
        f for f in diagnose_trace(rows) if f["id"] == "artifact_without_producer_span"
    )
    assert finding["confidence"] == "low"
    assert finding["evidence_event_ids"] == ["2"]


def test_profiles_are_isolated_by_turn_tenant_and_task(rows):
    rows[0]["coverage_profile"] = "interactive_response"
    for field in ("turn_id", "user_id", "task_id"):
        copy = deepcopy(rows)
        for row in copy[1:]:
            row[field] = "other"
        stages = {s["stage"] for s in trace_coverage(copy)["missing_stages"]}
        assert {"output_contract", "ui_delivery"} <= stages
