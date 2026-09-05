import asyncio
import json
from unittest.mock import patch

import pytest

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    ObservabilityStore,
    ResourceRef,
    TraceContext,
    TraceSnapshot,
    analyze_trace_events,
    normalize_trace_snapshot,
)


@pytest.fixture
def provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )


@pytest.fixture
def context():
    return TraceContext(
        agent_id="agent",
        thread_id="thread",
        memory_id="memory",
        root_trace_id="root",
        run_id="run",
        turn_id="turn",
    )


def events(provider):
    return normalize_trace_snapshot(
        TraceSnapshot(bundle_rows=provider.list_all(MemoryType.SHARED_MEMORY))
    ).events


def findings(provider):
    return {
        i["id"]
        for i in analyze_trace_events(events(provider), max_insights=100)["insights"]
    }


def test_worker_carrier_nested_spans_and_idempotent_writes(provider, context):
    recorder = ObservabilityRecorder(provider, context)
    recorder.record_intent("slides", expected_artifact_types=["slide_deck"])
    with recorder.start_span("enqueue", external_id="job-1") as parent:
        carrier = json.loads(json.dumps(parent.to_carrier()))
        worker = ObservabilityRecorder(provider, TraceContext.from_carrier(carrier))
        for _ in range(2):
            with worker.start_span("slides.safety_net", external_id="job-1") as child:
                worker.record_artifact(
                    ResourceRef(resource_type="slide_deck", ref="deck-1"),
                    input_refs=[
                        ResourceRef(resource_type="analysis", ref="analysis-1")
                    ],
                    producing_span_id=child.span_id,
                    persistence_verified=True,
                    task_id="slides",
                    external_id="deck-1",
                )
    rows = events(provider)
    assert len(rows) == 6  # intent, 2 parent, 2 child, 1 artifact
    assert all(e["root_trace_id"] == "root" for e in rows)
    child_start = next(
        e
        for e in rows
        if e.get("operation") == "slides.safety_net" and e["phase"] == "start"
    )
    assert child_start["parent_span_id"] == parent.span_id
    # Span persistence is correct, but no expected instrumentation profile was
    # declared. A clean write cannot establish end-to-end capture coverage.
    assert findings(provider) == {"trace_coverage_incomplete"}


def test_async_context_is_isolated_and_exception_content_never_persists(
    provider, context
):
    recorder = ObservabilityRecorder(provider, context)

    async def operation(name):
        async with recorder.start_span(name) as span:
            await asyncio.sleep(0)
            recorder.record_event(f"{name}.child")
            return span.span_id

    async def run():
        return await asyncio.gather(operation("one"), operation("two"))

    ids = asyncio.run(run())
    rows = events(provider)
    assert (
        next(e for e in rows if e["operation"] == "one.child")["parent_span_id"]
        == ids[0]
    )
    assert (
        next(e for e in rows if e["operation"] == "two.child")["parent_span_id"]
        == ids[1]
    )
    with pytest.raises(RuntimeError, match="secret"):
        with recorder.start_span("fail"):
            raise RuntimeError("secret prompt with credentials")
    rows = events(provider)
    assert "secret prompt" not in json.dumps(rows)
    assert any(
        e["status"] == "error" and e.get("error_code") == "RuntimeError" for e in rows
    )


def test_invalid_deck_and_truncated_learning_map_are_diagnosed(provider, context):
    recorder = ObservabilityRecorder(provider, context)
    source = ResourceRef(
        resource_type="analysis", ref="current", role="authoritative_current_source"
    )
    recorder.record_intent(
        "slides",
        expected_artifact_types=["slide_deck"],
        required_contracts=["learning_map"],
        input_refs=[source],
    )
    recorder.record_artifact(
        ResourceRef(resource_type="slide_deck", ref="bad-deck"),
        input_refs=[],
        producing_span_id="unobserved-host",
        persistence_verified=True,
        task_id="slides",
    )
    with recorder.contract("learning_map") as contract:
        contract.record(
            opening_marker_found=True,
            closing_marker_found=False,
            parser_status="failed",
            item_counts={"nodes": 0, "edges": 0},
        )
    recorder.record_delivery("learning_map", emitted=False)
    assert {
        "artifact_missing_source_provenance",
        "artifact_without_producer_span",
        "output_contract_truncated",
        "contract_parse_failed",
        "contract_not_emitted",
    } <= findings(provider)
    outcome = recorder.reconcile_outcome(events(provider))
    assert outcome["events"][0]["status"] == "partial"


def test_correct_source_contract_and_delivery_complete_task(provider, context):
    recorder = ObservabilityRecorder(provider, context)
    source = ResourceRef(
        resource_type="analysis",
        ref="current",
        role="authoritative_current_source",
        version="2",
    )
    recorder.record_intent(
        "slides",
        expected_artifact_types=["slide_deck"],
        required_contracts=["learning_map"],
        input_refs=[source],
    )
    with recorder.start_span("slides.create") as span:
        recorder.record_artifact(
            ResourceRef(resource_type="slide_deck", ref="deck"),
            input_refs=[source],
            producing_span_id=span.span_id,
            persistence_verified=True,
            task_id="slides",
        )
    recorder.record_contract_result(
        "learning_map",
        opening_marker_found=True,
        closing_marker_found=True,
        parser_status="success",
        item_counts={"nodes": 3, "edges": 2},
    )
    recorder.record_delivery(
        "learning_map",
        emitted=True,
        acknowledged=True,
        input_refs=[ResourceRef(resource_type="slide_deck", ref="deck")],
    )
    # The recorded intent can reconcile successfully while broader capture
    # coverage stays unknown without a declared instrumentation profile.
    assert findings(provider) == {"trace_coverage_incomplete"}
    outcome = recorder.reconcile_outcome(events(provider))
    assert outcome["events"][0]["status"] == "success"


@pytest.mark.parametrize("source_independent", [False, True])
def test_empty_source_requires_explicit_independence(
    provider, context, source_independent
):
    recorder = ObservabilityRecorder(provider, context)
    with recorder.start_span("create") as span:
        recorder.record_artifact(
            ResourceRef(resource_type="doc", ref="blank"),
            input_refs=[],
            producing_span_id=span.span_id,
            persistence_verified=True,
            source_independent=source_independent,
        )
    assert ("artifact_missing_source_provenance" in findings(provider)) is (
        not source_independent
    )


def test_source_version_mismatch_failed_producer_and_partial_recovery(
    provider, context
):
    recorder = ObservabilityRecorder(provider, context)
    recorder.record_intent(
        "doc",
        input_refs=[
            ResourceRef(
                resource_type="analysis",
                ref="current",
                version="2",
                role="authoritative_current_source",
            )
        ],
    )
    with pytest.raises(ValueError):
        with recorder.start_span("create") as span:
            raise ValueError("private")
    recorder.record_artifact(
        ResourceRef(resource_type="doc", ref="bad"),
        input_refs=[ResourceRef(resource_type="analysis", ref="current", version="1")],
        producing_span_id=span.span_id,
        persistence_verified=True,
    )
    recorder.record_contract_result(
        "map",
        opening_marker_found=True,
        closing_marker_found=False,
        parser_status="partial",
        recovery_used=True,
    )
    recorder.record_delivery("map", emitted=True)
    assert {
        "artifact_source_mismatch",
        "artifact_persisted_after_failed_action",
        "contract_recovered_partial",
        "ui_delivery_unconfirmed",
    } <= findings(provider)


def test_attributes_and_carriers_are_bounded_and_fail_soft_is_visible(
    provider, context
):
    recorder = ObservabilityRecorder(provider, context)
    for attributes in (
        {"prompt": "private"},
        {"fallback_reason": "x" * 241},
        {"selected": {"nested": True}},
        {"relevance_score": float("nan")},
        {"fallback_reason": "user@example.com"},
    ):
        with pytest.raises(ValueError):
            recorder.record_event("invalid", attributes=attributes)
    with pytest.raises(ValueError):
        TraceContext.from_carrier({**context.to_carrier(), "prompt": "private"})
    with pytest.raises(ValueError):
        TraceContext.from_carrier({})
    with patch.object(
        provider, "store", side_effect=RuntimeError("secret database URL")
    ):
        result = recorder.record_event("failed.write")
        assert result["persisted"] is False
        assert recorder.health["write_failures"] == 1
        assert recorder.health["last_error_code"] == "RuntimeError"


def test_same_root_different_turns_do_not_overwrite(provider, context):
    store = ObservabilityStore(provider)
    for turn in ("one", "two"):
        store.record_trace_bundle(
            trace_context={**context.to_carrier(), "turn_id": turn},
            events=[{"trace_kind": "turn_start", "trace_id": turn}],
        )
    assert len(events(provider)) == 2


def test_external_ids_are_isolated_by_turn_and_tenant(provider, context):
    for overrides in ({}, {"turn_id": "other"}, {"user_id": "other"}):
        recorder = ObservabilityRecorder(
            provider, {**context.to_carrier(), **overrides}
        )
        for _ in range(2):
            with recorder.start_span("work", external_id="same-job"):
                pass
    rows = events(provider)
    assert len(rows) == 6
    assert len({row["span_id"] for row in rows}) == 3


def test_task_ids_round_trip_for_contract_and_delivery(provider, context):
    recorder = ObservabilityRecorder(provider, context)
    with recorder.contract("map", task_id="task") as contract:
        contract.record(
            opening_marker_found=True,
            closing_marker_found=True,
            parser_status="success",
        )
    recorder.record_delivery("map", emitted=True, acknowledged=True, task_id="task")
    assert all(row["task_id"] == "task" for row in events(provider))
