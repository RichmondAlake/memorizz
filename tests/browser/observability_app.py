"""Isolated manual/browser-test server. Only synthetic temporary data is used."""

import os
from tempfile import TemporaryDirectory

import uvicorn

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    ObservabilityStore,
    TraceContext,
)
from memorizz.ui import state
from memorizz.ui.app import create_app


def run():
    with TemporaryDirectory(prefix="memorizz-browser-fixture-") as root:
        os.environ.update(
            MEMORIZZ_UI_AUTH_TOKEN="memorizz-browser-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="false",
            MEMORIZZ_UI_TRACE_CONTENT_MODE="redacted",
            MEMORIZZ_UI_AUDIT_LOG=root + "/audit.jsonl",
            MEMORIZZ_OBSERVABILITY_DUAL_WRITE="false",
            MEMORIZZ_OBSERVABILITY_READ_PATH="bundles",
        )
        provider = FileSystemProvider(
            FileSystemConfig(root_path=root + "/memory", lazy_vector_indexes=True)
        )
        context = TraceContext(
            agent_id="browser-agent",
            thread_id="thread",
            root_trace_id="browser-root",
            turn_id="turn",
            run_id="run",
            user_id="synthetic-user",
        )
        r = ObservabilityRecorder(provider, context, strict=True)
        source = {
            "resource_type": "analysis",
            "ref": "source-1",
            "version": "v1",
            "role": "authoritative_current_source",
        }
        r.record_intent(
            "slides",
            input_refs=[source],
            expected_artifact_types=["slide_deck"],
            required_contracts=["map"],
        )
        r.record_selection(
            [
                {
                    "resource": source,
                    "candidate_rank": 1,
                    "selected": True,
                    "selection_reason": "canonical_thread_source",
                }
            ],
            task_id="slides",
        )
        r.record_event(
            "supply",
            kind="memory_supply",
            input_refs=[source],
            attributes={"task_id": "slides"},
        )
        with r.start_span("create.deck", attributes={"task_id": "slides"}) as span:
            r.record_artifact(
                {"resource_type": "slide_deck", "ref": "deck-1", "version": "v1"},
                input_refs=[source],
                producing_span_id=span.span_id,
                persistence_verified=True,
                task_id="slides",
            )
        r.record_contract_result(
            "map",
            parser_status="failed",
            opening_marker_found=True,
            closing_marker_found=False,
            task_id="slides",
        )
        ObservabilityStore(provider).record_trace_bundle(
            trace_context=context.to_carrier(),
            events=[
                {
                    "event_id": "preview",
                    "span_id": "model",
                    "trace_kind": "model_result",
                    "status": "success",
                    "content": "SYNTHETIC PRIVATE PREVIEW user@example.com",
                    "timestamp": "2026-09-04T00:00:00Z",
                }
            ],
        )
        # Reused event IDs in another turn must never widen reveal or draft scope.
        ObservabilityStore(provider).record_trace_bundle(
            trace_context={**context.to_carrier(), "turn_id": "other-turn"},
            events=[
                {
                    "event_id": "preview",
                    "trace_kind": "model_result",
                    "task_id": "slides",
                    "grounding_source_ids": '["other-turn-source"]',
                    "content": "WRONG TURN PRIVATE PREVIEW",
                    "timestamp": "2026-09-06T00:00:00Z",
                }
            ],
        )
        # A busy overview plus uninstrumented legacy evidence reproduces the
        # audit's empty panels and formerly buried selected incident.
        for number in range(40):
            ObservabilityStore(provider).record_trace_bundle(
                trace_context={
                    "agent_id": "browser-agent",
                    "thread_id": f"overview-thread-{number}",
                    "root_trace_id": f"overview-root-{number}",
                    "turn_id": "turn",
                },
                events=[
                    {
                        "event_id": f"overview-{number}",
                        "trace_kind": "model_result",
                        "timestamp": "2026-09-04T00:00:00Z",
                    }
                ],
            )
        ObservabilityStore(provider).record_trace_bundle(
            trace_context={
                "agent_id": "browser-agent",
                "thread_id": "legacy-thread",
                "root_trace_id": "legacy-root",
                "turn_id": "legacy-turn",
            },
            events=[
                {
                    "event_id": "legacy-memory",
                    "trace_kind": "memory_context",
                    "memory_supplied_count": 33,
                    "grounding_source_ids": "[]",
                    "timestamp": "2026-09-04T00:00:00Z",
                }
            ],
        )
        state._state.update(
            provider=provider, provider_type="filesystem", connection_info={}
        )
        app = create_app(
            identity_resolver=lambda email, scope: {"user_id": "synthetic-user"},
            artifact_resolver=lambda ref, scope: {
                "exists": True,
                "ownership_verified": True,
                "version": "v1",
            },
            resource_authorizer=lambda ref, scope: True,
        )
        if os.getenv("MEMORIZZ_BROWSER_TEST_HOOKS") == "off":
            app.state.trace_identity_resolver = None
            app.state.trace_artifact_resolver = None
            app.state.trace_resource_authorizer = None
        try:
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=int(os.getenv("MEMORIZZ_BROWSER_TEST_PORT", "8779")),
                lifespan="off",
            )
        finally:
            provider.close()


if __name__ == "__main__":
    run()
