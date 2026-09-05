"""Offline example: PYTHONPATH=src python examples/observability/host_workflow.py.

Uses a temporary filesystem provider and synthetic references. No model calls,
production data, database credentials, or real side effects are required.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    ResourceRef,
    TraceContext,
    TraceSnapshot,
    analyze_trace_events,
    normalize_trace_snapshot,
    register_coverage_profile,
)


def run_example(root: Path) -> dict:
    # This offline workflow has no model invocation. Declare exactly the stages
    # it exercises; a complete read alone must not imply capture coverage.
    register_coverage_profile(
        "offline_artifact_example",
        required={
            "intent_plan",
            "application_action",
            "artifact_persisted",
            "output_contract",
            "ui_delivery",
        },
        optional={"verified_outcome"},
        replace=True,
    )
    provider = FileSystemProvider(
        FileSystemConfig(root_path=root, lazy_vector_indexes=True)
    )
    context = TraceContext(
        agent_id="example-agent",
        thread_id="example-thread",
        memory_id="example-memory",
        root_trace_id="example-request",
        run_id="example-run",
        turn_id="example-turn",
    )
    recorder = ObservabilityRecorder(provider, context, strict=True)
    source = ResourceRef(
        resource_type="analysis",
        ref="analysis-1",
        role="authoritative_current_source",
        version="1",
        ownership_verified=True,
        provenance_status="verified",
    )
    recorder.record_intent(
        "slides",
        expected_artifact_types=["slide_deck"],
        required_contracts=["learning_map"],
        input_refs=[source],
        coverage_profile="offline_artifact_example",
    )
    with recorder.start_span(
        "slides.safety_net",
        input_refs=[source],
        attributes={"fallback_reason": "agent_task_outstanding", "side_effect": True},
    ) as span:
        # Real integrations serialize span.to_carrier() into their queue job.
        worker = ObservabilityRecorder(
            provider, TraceContext.from_carrier(span.to_carrier()), strict=True
        )
        worker.record_artifact(
            ResourceRef(
                resource_type="slide_deck", ref="deck-1", provenance_status="verified"
            ),
            input_refs=[source],
            producing_span_id=span.span_id,
            persistence_verified=True,
            task_id="slides",
            external_id="deck-1",
        )
        span.succeed()
    with recorder.contract("learning_map", task_id="slides") as contract:
        # Supply evidence from the application's parser, not from model claims.
        contract.record(
            opening_marker_found=True,
            closing_marker_found=True,
            parser_status="success",
            item_counts={"nodes": 3, "edges": 2},
        )
    recorder.record_delivery(
        "learning_map",
        emitted=True,
        acknowledged=True,
        task_id="slides",
        input_refs=[ResourceRef(resource_type="slide_deck", ref="deck-1")],
    )
    window = normalize_trace_snapshot(
        TraceSnapshot(bundle_rows=provider.list_all(MemoryType.SHARED_MEMORY))
    )
    report = analyze_trace_events(window.events)
    assert not report["insights"], report["insights"]
    outcome = recorder.reconcile_outcome(
        window.events, task_id="slides", external_id="slides-outcome"
    )
    assert outcome["events"][0]["status"] == "success"
    return {
        "events": len(window.events),
        "coverage": report["coverage"]["coverage"],
        "read_completeness": report["coverage"]["read_completeness"],
        "instrumentation_coverage": report["coverage"]["instrumentation_coverage"],
        "outcome": "success",
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="memorizz-observability-example-") as directory:
        print(run_example(Path(directory)))
