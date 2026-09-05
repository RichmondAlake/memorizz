"""Selection explanations, profile contracts, health and opt-in MCP inspection."""

import asyncio
import json
from unittest.mock import Mock

import pytest

from memorizz import MemAgent
from memorizz.enums import MemoryType
from memorizz.memagent.utils.context_dedup import dedupe_and_select
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityRecorder,
    SelectionDecision,
    TraceContext,
    build_lineage_inspectors,
    register_coverage_profile,
    trace_coverage,
)
from memorizz.observability.inspection import build_observability_alerts
from memorizz.observability.pipeline import pipeline_health


@pytest.fixture
def provider(tmp_path):
    p = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield p
    p.close()


def context():
    return TraceContext(
        agent_id="agent",
        thread_id="thread",
        root_trace_id="root",
        run_id="run",
        turn_id="turn",
        user_id="alice",
    )


def test_real_selector_records_bounded_private_rejection_reasons():
    candidates = [
        (
            "episodic",
            {
                "_id": "history",
                "content": "Already visible in the conversation",
                "score": 0.9,
            },
        ),
        (
            "episodic",
            {"_id": "best", "content": "A unique retrieval about whales", "score": 0.8},
        ),
        (
            "episodic",
            {
                "_id": "duplicate",
                "content": "A unique retrieval about whales",
                "score": 0.7,
            },
        ),
        (
            "episodic",
            {
                "_id": "budget",
                "content": "Unrelated retrieval about volcanic rocks",
                "score": 0.6,
            },
        ),
    ]
    ledger = []
    selected = dedupe_and_select(
        candidates,
        history_texts=[candidates[0][1]["content"]],
        max_items=1,
        selection_ledger=ledger,
    )
    assert len(selected) == 1
    typed = [SelectionDecision.model_validate(row) for row in ledger]
    assert {row.selection_reason for row in typed} >= {
        "already_in_history",
        "exact_duplicate",
        "budget_exceeded",
        "provider_rank_selected",
    }
    assert sum(row.selected for row in typed) == 1
    assert "whales" not in json.dumps(ledger)
    assert all(row.resource.ref.startswith("memory-ref:") for row in typed)
    ledger = []
    dedupe_and_select(candidates * 100, selection_ledger=ledger)
    assert len(ledger) == 64


def test_selection_ledger_survives_runtime_bundle_and_native_projection(provider):
    from memorizz.observability import ObservabilityMaintenance

    agent = MemAgent(
        memory_provider=provider, memory_types=[MemoryType.CONVERSATION_MEMORY]
    )
    agent._resolve_execution_state("memory", "thread")
    agent._begin_trace_turn("alice")
    agent._last_selection_ledger = [
        {
            "resource": {"resource_type": "memory", "ref": "memory-opaque"},
            "candidate_rank": 1,
            "relevance_score": 0.8,
            "selected": True,
            "selection_reason": "provider_rank_selected",
        }
    ]
    agent._emit_memory_context_trace({})
    events = agent._build_trace_bundle_events(agent._stream_trace_events)
    ledger = next(row for row in events if row.get("trace_kind") == "memory_selection")
    assert ledger["selection_ledger"][0]["selected"]
    from memorizz.observability import ObservabilityStore

    ObservabilityStore(provider).record_trace_bundle(
        trace_context=agent.get_trace_context(), events=events
    )
    m = ObservabilityMaintenance(provider)
    m.initialize()
    assert m.backfill(since="2020-01-01", dry_run=False)["failed"] == 0
    assert m.parity()["passed"]
    page = provider.query_trace_events(
        read_path="index", resource_refs=["memory-opaque"]
    )
    assert any(row.get("selection_ledger") for row in page["items"])


def test_custom_profile_required_optional_and_task_isolation(monkeypatch):
    from memorizz.observability import coverage

    monkeypatch.setattr(coverage, "PROFILES", dict(coverage.PROFILES))
    monkeypatch.setattr(coverage, "OPTIONAL_STAGES", {})
    register_coverage_profile(
        "local_workflow",
        required=["intent_plan", "artifact_persisted"],
        optional=["ui_delivery"],
    )
    rows = [
        {"kind": "intent_plan", "task_id": "one", "coverage_profile": "local_workflow"},
        {"kind": "artifact_persisted", "task_id": "two"},
    ]
    result = trace_coverage(iter(rows))
    assert result["coverage"] == "partial"
    assert result["missing_stages"][0]["task_id"] == "one"
    assert result["optional_stages"] == {"local_workflow": ["ui_delivery"]}
    with pytest.raises(ValueError):
        register_coverage_profile("local_workflow", required=["model_result"])
    with pytest.raises(ValueError):
        register_coverage_profile("bad", required=["one"], optional=["one"])


def test_missing_worker_carrier_is_reported_instead_of_silent_new_root(provider):
    with pytest.raises(ValueError):
        ObservabilityRecorder.for_worker(provider, None)
    worker = ObservabilityRecorder.for_worker(
        provider, None, fallback_context=context(), strict=True
    )
    rows = provider.query_trace_events()["items"]
    assert len(rows) == 1 and rows[0]["context_missing"] is True
    assert rows[0]["root_trace_id"] == "root"
    assert worker.context == context()


def test_callback_failures_and_alert_thresholds_are_visible(provider):
    agent = MemAgent(memory_provider=provider)
    agent.set_stream_event_callback(Mock(side_effect=RuntimeError("PRIVATE ERROR")))
    agent._emit_stream_event("trace", {"trace_kind": "model_result"})
    pipeline = pipeline_health(provider)
    assert pipeline["counters"]["callback_failures"] == 1
    alerts = build_observability_alerts(
        {
            "pipeline": pipeline,
            "queries": {"trace": {"query_duration_ms": 300}},
            "orphan_spans": 2,
        }
    )
    assert {a["code"] for a in alerts} == {
        "callback_failures",
        "slow_trace_query",
        "orphan_spans",
    }
    assert "PRIVATE" not in json.dumps(alerts)


def test_lineage_separates_expected_retrieved_supplied_output_and_contracts(provider):
    r = ObservabilityRecorder(provider, context(), strict=True)
    source = {
        "resource_type": "analysis",
        "ref": "video",
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
    r.record_artifact(
        {"resource_type": "slide_deck", "ref": "deck"},
        input_refs=[],
        producing_span_id="missing-producer",
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
    panels = build_lineage_inspectors(provider.query_trace_events()["items"])
    lineage = panels["memory"][0]
    assert lineage["state"] == "missing_source"
    assert lineage["expected"] and lineage["retrieved"] and lineage["supplied"]
    assert lineage["output_bound"] == []
    assert panels["artifacts"][0]["outside_observed_execution"]
    assert panels["contracts"][0]["parser_status"] == "failed"
    assert panels["contracts"][0]["acknowledged"] is None


def test_lineage_still_displays_supplied_memory_before_artifact_exists():
    panels = build_lineage_inspectors(
        [
            {
                "kind": "memory_supply",
                "timestamp": None,
                "input_refs": [{"resource_type": "analysis", "ref": "source"}],
            }
        ]
    )
    assert panels["memory"][0]["supplied"][0]["ref"] == "source"
    assert panels["memory"][0]["state"] == "not_observable"
    assert panels["artifacts"] == []


def test_old_version_delivery_is_not_attributed_to_updated_artifact():
    panels = build_lineage_inspectors(
        [
            {
                "kind": "artifact_persisted",
                "timestamp": "2026-09-04T10:00:00Z",
                "output_refs": [
                    {"resource_type": "doc", "ref": "doc", "version": "v2"}
                ],
            },
            {
                "kind": "ui_delivery",
                "timestamp": "2026-09-04T10:01:00Z",
                "input_refs": [{"resource_type": "doc", "ref": "doc", "version": "v1"}],
                "emitted": True,
                "acknowledged": True,
            },
        ]
    )
    assert panels["artifacts"][0]["delivered"] is None


def test_optional_mcp_trace_tool_is_disabled_by_default_and_tenant_bound(
    provider, tmp_path
):
    pytest.importorskip("mcp")
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import READ_SCOPE, MemorizzMCPServerConfig
    from memorizz.mcp_server.runtime import MemorizzRuntime, MemorizzServerError
    from memorizz.mcp_server.server import create_memorizz_mcp_server

    r = ObservabilityRecorder(provider, context(), strict=True)
    r.record_event("work")
    ObservabilityRecorder(
        provider, {**context().to_carrier(), "user_id": "bob"}, strict=True
    ).record_event("work")
    identity = RequestIdentity(
        principal="alice", scopes=frozenset({READ_SCOPE}), authenticated=True
    )
    config = MemorizzMCPServerConfig(
        allow_writes=False, allow_agent_execution=False, exposed_agent_ids={"agent"}
    )
    runtime = MemorizzRuntime(
        config,
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    with pytest.raises(MemorizzServerError, match="disabled"):
        runtime.query_traces("agent", identity)
    config.allow_trace_queries = True
    page = runtime.query_traces("agent", identity, explain=True)
    assert len(page["items"]) == 1
    assert "bob" not in json.dumps(page)
    assert page["analysis"]["read_only"]
    server = create_memorizz_mcp_server(config, runtime=runtime)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert "memorizz_query_traces" in names
