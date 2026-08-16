"""Regression coverage for MemoRizz 0.5 tool governance boundaries."""

from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional

import pytest

import memorizz.approval as approval_module
from memorizz.approval import (
    ApprovalRequired,
    ApprovalStateError,
    ApprovalStatus,
    SQLiteApprovalStore,
)
from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memagent.managers.tool_manager import ToolManager
from memorizz.tooling import (
    ContextPolicy,
    SemanticToolRouter,
    ToolResultPolicy,
    callable_json_schema,
    governed_tool,
)
from tests.mocks.mock_providers import MockMemoryProvider


def _tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _prepare_direct_tool(agent: MemAgent, query: str, user_id: str | None = None):
    agent.semantic_tool_router.begin_turn(user_id=user_id)
    return agent._build_llm_tools(query, user_id=user_id)


@pytest.mark.unit
def test_callable_schema_resolves_postponed_annotations_losslessly():
    def calendar_lookup(
        calendar_id: str,
        visibility: Literal["private", "public"] = "private",
        options: Optional[Dict[str, Any]] = None,
    ) -> dict:
        return {}

    schema = callable_json_schema(calendar_lookup)

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["calendar_id"]
    assert schema["properties"]["visibility"]["enum"] == ["private", "public"]
    assert schema["properties"]["visibility"]["default"] == "private"
    assert schema["properties"]["options"]["anyOf"] == [
        {"additionalProperties": True, "type": "object"},
        {"type": "null"},
    ]
    assert schema["properties"]["options"]["default"] is None


@pytest.mark.unit
def test_progressive_router_discloses_top_k_and_allowlists_dispatch():
    manager = ToolManager()

    def weather_read(city: str) -> str:
        """Read the current weather for a city."""
        return f"weather:{city}"

    def calendar_create_event(title: str, starts_at: str) -> str:
        """Create a calendar event."""
        return f"created:{title}:{starts_at}"

    def inventory_lookup(sku: str) -> str:
        """Read inventory stock."""
        return f"stock:{sku}"

    for function in (weather_read, calendar_create_event, inventory_lookup):
        assert manager.add_tool(function)

    router = SemanticToolRouter(
        manager,
        top_k=1,
        aliases={"old_weather": "weather_read"},
        deprecated_arguments={"weather_read": {"location": "city"}},
        max_attempts_per_call=2,
    )
    router.begin_turn(user_id="alice")
    schemas = router.schemas_for_turn("weather in London", user_id="alice")
    names = {schema["function"]["name"] for schema in schemas}

    assert names == {"discover_tools", "invoke_tool", "weather_read"}
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False
        for schema in schemas
    )
    hidden = router.invoke_tool(
        "calendar_create_event",
        {"title": "Review", "starts_at": "2026-08-13T10:00:00Z"},
    )
    assert hidden["error_code"] == "invalid_tool_invocation"
    assert "not disclosed" in hidden["error"]

    discovered = router.discover_tools("weather", user_id="alice", limit=1)
    assert discovered["tools"][0]["input_schema"]["required"] == ["city"]
    called = router.invoke_tool("old_weather", {"location": "London"})
    assert called["ok"] is True
    assert called["result"] == "weather:London"
    assert called["warnings"]
    duplicate = router.invoke_tool("weather_read", {"city": "London"})
    assert duplicate["error_code"] == "duplicate_tool_call"


@pytest.mark.unit
def test_disabled_progressive_router_can_invoke_every_disclosed_tool():
    manager = ToolManager()

    def ingest_url(url: str) -> dict:
        """Ingest a URL into the caller's library."""
        return {"queued": url}

    assert manager.add_tool(ingest_url)
    router = SemanticToolRouter(manager, enabled=False)
    router.begin_turn(user_id="alice")

    schemas = router.schemas_for_turn("add this URL", user_id="alice")
    assert {schema["function"]["name"] for schema in schemas} == {"ingest_url"}
    assert router._selected == set()

    called = router.invoke_tool("ingest_url", {"url": "https://example.com"})
    assert called["ok"] is True
    assert called["result"] == {"queued": "https://example.com"}


@pytest.mark.unit
def test_side_effect_candidate_bypasses_cache_before_a_similar_hit_can_short_circuit():
    @governed_tool(side_effects=True, requires_approval=True)
    def update_inventory(sku: str, quantity: int) -> dict:
        """Update inventory quantity for a SKU."""
        return {"sku": sku, "quantity": quantity}

    agent = MemAgent(tools=[update_inventory])
    assert agent.semantic_tool_router._selected == set()

    reason = agent._semantic_cache_preflight_bypass(
        "update inventory quantity", user_id="alice"
    )

    assert reason == "side_effecting_tool_candidate"
    # Preflight cannot mutate the actual per-turn invocation allowlist.
    assert agent.semantic_tool_router._selected == set()


@pytest.mark.unit
def test_hidden_direct_tool_name_is_rejected_even_if_registered():
    calls = []

    def safe_read(topic: str) -> str:
        """Read safe public data."""
        return topic

    def hidden_delete(record_id: str) -> str:
        """Delete a hidden record."""
        calls.append(record_id)
        return "deleted"

    agent = MemAgent(
        tools=[safe_read, hidden_delete],
        context_policy=ContextPolicy(tool_top_k=1),
    )
    _prepare_direct_tool(agent, "read safe public data")
    messages = []
    agent._execute_and_record_tool_call(
        _tool_call("hidden_delete", {"record_id": "secret"}),
        messages,
        workflow=None,
        user_id="alice",
    )
    payload = json.loads(messages[-1]["content"])
    assert payload["error_code"] == "invalid_tool_invocation"
    assert calls == []


@pytest.mark.unit
def test_durable_approval_resumes_exact_checkpoint_once(tmp_path):
    executions = []

    @governed_tool(
        side_effects=True,
        requires_approval=True,
        approval_reason="Delete a customer record",
        domains=("customers",),
    )
    def delete_customer(record_id: str) -> dict:
        executions.append(record_id)
        return {"deleted": record_id}

    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    agent = MemAgent(
        tools=[delete_customer],
        approval_store=store,
        context_policy=ContextPolicy(max_tool_attempts_per_call=1),
    )
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = "thread-1"
    _prepare_direct_tool(agent, "delete customer record", user_id="alice")

    with pytest.raises(ApprovalRequired) as raised:
        agent._execute_and_record_tool_call(
            _tool_call("delete_customer", {"record_id": "customer-7"}),
            [{"role": "user", "content": "delete customer 7"}],
            workflow=None,
            user_id="alice",
            query="delete customer 7",
        )

    proposal = raised.value.proposal
    serialized = proposal.to_dict(include_arguments=True)
    assert serialized["checkpoint_id"] == proposal.proposal_id
    assert serialized["thread_id"] == "thread-1"
    assert serialized["argument_hash"]
    assert executions == []
    assert "approved" not in json.dumps(agent._build_llm_tools("delete customer"))
    assert "confirm" not in json.dumps(agent._build_llm_tools("delete customer"))

    approved = agent.approve(proposal.proposal_id, approver_id="operator@example.com")
    assert approved["approver_id"] == "operator@example.com"
    resumed = agent.resume_approval(proposal.proposal_id, continue_model=False)
    assert resumed.consumed is True
    assert resumed.tool_result == {"deleted": "customer-7"}
    assert resumed.assistant_response is None
    assert executions == ["customer-7"]
    assert store.get(proposal.proposal_id).status == ApprovalStatus.CONSUMED
    replay = agent.resume_approval(proposal.proposal_id)
    assert replay.error_code == "invalid_approval_state"
    assert executions == ["customer-7"]


@pytest.mark.unit
def test_approved_proposal_still_expires_before_resume(tmp_path, monkeypatch):
    store = SQLiteApprovalStore(tmp_path / "expiry.sqlite3")
    proposal = store.propose(
        owner_id="agent-1",
        tool_name="create_event",
        arguments={"title": "Expired"},
        policy_reason="External write",
        ttl_seconds=60,
    )
    store.approve(proposal.proposal_id, approver_id="operator@example.com")
    monkeypatch.setattr(
        approval_module,
        "_utcnow",
        lambda: proposal.expires_at + timedelta(seconds=1),
    )

    assert store.get(proposal.proposal_id).status == ApprovalStatus.EXPIRED
    with pytest.raises(ApprovalStateError, match="expired"):
        store.consume(proposal.proposal_id)


@pytest.mark.unit
def test_tool_results_offload_only_when_large_and_never_reoffload_expansion():
    provider = MockMemoryProvider()

    def small_result() -> str:
        """Return a small result."""
        return "inline"

    def large_result() -> str:
        """Return a large result."""
        return "L" * 900

    def retrieve_tool_log_entry(tool_log_id: str) -> str:
        """Expand a result into deliberately large content."""
        return "E" * 900

    agent = MemAgent(
        tools=[small_result, large_result],
        memory_provider=provider,
        memory_ids=["memory-1"],
        tool_result_policy=ToolResultPolicy(offload_above_chars=256),
    )
    # Replace the built-in expansion callable with a deterministic large value.
    agent.tool_manager.add_tool(retrieve_tool_log_entry)
    agent._current_memory_id = "memory-1"
    agent._current_thread_id = "thread-1"

    messages = []
    _prepare_direct_tool(agent, "small result")
    agent._execute_and_record_tool_call(
        _tool_call("small_result", {}, "small"), messages, None, None
    )
    assert messages[-1]["content"] == "inline"
    tool_log_writes = lambda: [
        call
        for call in provider.call_history
        if call[0] == "store" and call[3] == MemoryType.TOOL_LOG
    ]
    assert tool_log_writes() == []

    _prepare_direct_tool(agent, "large result")
    agent._execute_and_record_tool_call(
        _tool_call("large_result", {}, "large"), messages, None, None
    )
    pointer = json.loads(messages[-1]["content"])
    assert pointer["offloaded"] is True
    assert pointer["result_sha256"]
    assert pointer["result_chars"] == 900
    assert len(tool_log_writes()) == 1

    _prepare_direct_tool(agent, "retrieve tool log entry")
    agent._execute_and_record_tool_call(
        _tool_call("retrieve_tool_log_entry", {"tool_log_id": "any"}, "expand"),
        messages,
        None,
        None,
    )
    assert messages[-1]["content"] == "E" * 900
    assert len(tool_log_writes()) == 1
