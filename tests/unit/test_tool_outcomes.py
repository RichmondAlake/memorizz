"""Structured tool outcome contracts and MemAgent execution integration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from memorizz import ToolOutcome, ToolOutcomeStatus, ToolResult
from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.tooling import ContextPolicy, normalize_tool_result
from tests.mocks.mock_providers import MockMemoryProvider


@pytest.mark.unit
def test_structured_outcome_types_are_exported_from_public_package_api():
    import memorizz

    assert memorizz.ToolOutcome is ToolOutcome
    assert memorizz.ToolOutcomeStatus is ToolOutcomeStatus
    assert memorizz.ToolResult is ToolResult
    assert {"ToolOutcome", "ToolOutcomeStatus", "ToolResult"} <= set(memorizz.__all__)
    feature = memorizz.capabilities()["features"]["structured_tool_outcomes"]
    assert feature["available"] is True
    assert feature["surfaces"] == ["sdk", "cli", "mcp", "ui", "observability"]


def _tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.unit
def test_explicit_tool_result_preserves_payload_and_orthogonal_fallback_signals():
    payload = {"records": [{"id": "r1"}]}
    normalized, outcome = normalize_tool_result(
        ToolResult(
            payload,
            ToolOutcome(
                status=ToolOutcomeStatus.FALLBACK,
                reason_code="vector_index_unavailable",
                primary_provider="oracle_hnsw",
                fallback_provider="oracle_exact",
                result_count=1,
                degraded=True,
            ),
        )
    )

    assert normalized is payload
    assert outcome.ok is True
    assert outcome.status is ToolOutcomeStatus.FALLBACK
    assert outcome.fallback_used is True
    assert outcome.degraded is True
    assert outcome.to_dict()["fallback_provider"] == "oracle_exact"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"results": []}, ToolOutcomeStatus.EMPTY),
        (
            {"results": [], "stdout": ["completed"], "success": True},
            ToolOutcomeStatus.SUCCESS,
        ),
        ([], ToolOutcomeStatus.EMPTY),
        (
            {
                "matches": [{"id": "r1"}],
                "retrieval": {
                    "fallback_used": True,
                    "degraded": True,
                    "degraded_reason": "vector_index_missing",
                },
            },
            ToolOutcomeStatus.FALLBACK,
        ),
        (
            {
                "ok": False,
                "error_code": "provider_error",
                "provider": "notion",
                "error": "unavailable",
            },
            ToolOutcomeStatus.PROVIDER_ERROR,
        ),
        ({"error": "invalid input"}, ToolOutcomeStatus.ERROR),
        ({"ok": True, "error": "non-terminal warning"}, ToolOutcomeStatus.SUCCESS),
    ],
)
def test_legacy_tool_results_are_normalized_conservatively(result, expected):
    _payload, outcome = normalize_tool_result(result)
    assert outcome.status is expected


@pytest.mark.unit
def test_reserved_outcome_metadata_is_removed_before_model_consumption():
    payload, outcome = normalize_tool_result(
        {
            "rows": [1, 2],
            "_memorizz_outcome": {
                "status": "degraded",
                "reason_code": "partial_projection",
                "provider": "warehouse",
            },
        }
    )

    assert payload == {"rows": [1, 2]}
    assert outcome.status is ToolOutcomeStatus.DEGRADED
    assert "_memorizz_outcome" not in payload


@pytest.mark.unit
def test_memagent_records_outcome_without_changing_tool_payload():
    def inventory_lookup(sku: str):
        return ToolResult(
            {"sku": sku, "units": 7},
            ToolOutcome(
                status=ToolOutcomeStatus.FALLBACK,
                reason_code="primary_timeout",
                primary_provider="inventory_api",
                fallback_provider="replica",
                result_count=1,
            ),
        )

    agent = MemAgent(
        tools=[inventory_lookup],
        context_policy=ContextPolicy(progressive_tool_disclosure=False),
    )
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = "thread-1"
    agent._stream_trace_events = []
    events = []
    agent.set_stream_event_callback(events.append)
    agent.semantic_tool_router.begin_turn(user_id="alice")
    agent._build_llm_tools("inventory lookup", user_id="alice")

    messages = []
    result = agent._execute_and_record_tool_call(
        _tool_call("inventory_lookup", {"sku": "A-1"}),
        messages,
        workflow=None,
        user_id="alice",
    )

    assert result == {"sku": "A-1", "units": 7}
    assert json.loads(messages[-1]["content"]) == result
    outcomes = agent.last_tool_outcomes
    assert len(outcomes) == 1
    assert outcomes[0]["duration_ms"] >= 0
    assert {
        key: value for key, value in outcomes[0].items() if key != "duration_ms"
    } == {
        "tool_name": "inventory_lookup",
        "model_tool_name": "inventory_lookup",
        "tool_call_id": "call-1",
        "status": "fallback",
        "ok": True,
        "fallback_used": True,
        "degraded": False,
        "reason_code": "primary_timeout",
        "primary_provider": "inventory_api",
        "fallback_provider": "replica",
        "result_count": 1,
    }
    result_events = [
        event
        for event in events
        if event.get("type") == "trace" and event.get("trace_kind") == "tool_result"
    ]
    assert result_events[-1]["outcome"] == "fallback"
    assert result_events[-1]["status"] == "success"
    assert result_events[-1]["success"] is True
    assert agent._cache_bypass_reason == "tool_outcome_fallback"


@pytest.mark.unit
def test_tool_log_persists_structured_outcome_without_provider_specific_code():
    provider = MockMemoryProvider()
    manager = MemoryManager(provider)

    manager.store_tool_log(
        tool_name="inventory_lookup",
        arguments={"sku": "A-1"},
        result={"units": 7},
        memory_id="memory-1",
        success=True,
        outcome="fallback",
        outcome_details={
            "status": "fallback",
            "ok": True,
            "fallback_used": True,
            "reason_code": "primary_timeout",
        },
    )

    writes = [
        call
        for call in provider.call_history
        if call[0] == "store" and call[3] == MemoryType.TOOL_LOG
    ]
    assert len(writes) == 1
    stored = writes[0][2]
    assert stored["outcome"] == "fallback"
    assert stored["outcome_details"]["reason_code"] == "primary_timeout"
