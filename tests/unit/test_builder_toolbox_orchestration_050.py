"""Builder, deterministic Toolbox, semantic layer and orchestration parity."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.enums import MemoryType
from memorizz.long_term.procedural.toolbox import Toolbox
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator
from memorizz.sandbox.base import SandboxProvider
from memorizz.sandbox.models import ExecutionResult
from memorizz.semantic_layer import SemanticCatalog, SemanticModel
from memorizz.task_decomposition import TaskDecomposer
from memorizz.tooling import ContextPolicy, ToolResultPolicy
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider


class _Sandbox(SandboxProvider):
    provider_name = "test"

    def execute_code(self, code, language="python", timeout=30, envs=None):
        return ExecutionResult(stdout=[code], metadata={"provider": "test"})

    def write_file(self, path, content):
        return True

    def read_file(self, path):
        return "value"


@pytest.mark.unit
def test_builder_composes_full_surface_without_post_build_mutation(tmp_path):
    provider = MockMemoryProvider()
    approval_store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    sandbox = _Sandbox()
    delegate = MemAgent(
        model=MockLLMProvider(),
        memory_provider=provider,
        agent_id="delegate-1",
    )
    catalog = SemanticCatalog(
        [
            SemanticModel.model_validate(
                {
                    "name": "sales",
                    "version": "1",
                    "entities": [
                        {
                            "name": "orders",
                            "source": "orders",
                            "primary_key": "id",
                        }
                    ],
                    "measures": [
                        {
                            "name": "revenue",
                            "entity": "orders",
                            "expression": "amount",
                            "aggregation": "sum",
                        }
                    ],
                }
            )
        ]
    )

    agent = (
        MemAgentBuilder()
        .with_model(MockLLMProvider())
        .with_memory_provider(provider)
        .with_sandbox(sandbox)
        .with_toolbox(Toolbox(provider, agent_id="agent-1"))
        .with_skills(
            {
                "name": "incident_triage",
                "description": "Triage incidents",
                "content": "Check impact, owner, and mitigation.",
            }
        )
        .with_skill_retrieval(True, top_k=2)
        .with_skills_marketplace("skillsmp", {"base_url": "https://skills.test"})
        .with_tool_result_policy(ToolResultPolicy(offload_above_chars=700))
        .with_context_policy(ContextPolicy(tool_top_k=3))
        .with_approval_store(approval_store)
        .with_delegation([delegate], mode="deterministic", plan=[])
        .with_semantic_layer(catalog)
        .with_continual_learning(False)
        .build(validate=True)
    )

    assert agent.sandbox_manager.provider is sandbox
    assert agent.toolbox is not None
    assert agent.skillbox is not None
    assert agent.skill_retrieval is True
    assert agent.continual_learning is False
    assert agent.tool_result_policy.offload_above_chars == 700
    assert agent.context_policy.tool_top_k == 3
    assert agent.approval_store is approval_store
    assert agent.delegates == [delegate]
    assert agent.delegation_config["mode"] == "deterministic"
    assert agent.semantic_layer is catalog
    assert agent.has_skills_marketplace() is True
    assert {
        "semantic_list_models",
        "semantic_describe_model",
        "semantic_plan_query",
    }.issubset(agent.tool_manager.list_tools())


@pytest.mark.unit
def test_builder_build_and_save_persists_new_runtime_policies(tmp_path):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    agent = (
        MemAgentBuilder()
        .with_name("Persisted 0.5 agent")
        .with_memory_provider(provider)
        .with_tool_result_policy({"offload_above_chars": 600})
        .with_context_policy({"tool_top_k": 4})
        .with_skill_retrieval(True, top_k=3)
        .with_delegation(mode="deterministic", plan=[])
        .build_and_save()
    )
    saved = provider.retrieve_memagent(agent.agent_id)
    assert saved.tool_result_policy["offload_above_chars"] == 600
    assert saved.context_policy["tool_top_k"] == 4
    assert saved.skill_retrieval is True
    assert saved.skill_retrieval_config["top_k"] == 3
    assert saved.delegation_config["mode"] == "deterministic"


@pytest.mark.unit
def test_deterministic_delegates_are_operational_from_agent_run():
    provider = MockMemoryProvider()
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    agent = MemAgent(
        memory_provider=provider,
        agent_id="root",
        delegates=[delegate],
        delegation={"enabled": True, "mode": "deterministic", "plan": []},
    )
    calls = []

    def delegate_call(query, **kwargs):
        calls.append((query, kwargs))
        return "delegated"

    agent.delegate = delegate_call
    result = agent.run(
        "prepare the report",
        user_id="alice",
        context={"quarter": "Q3"},
        tool_context={"request_id": "req-1"},
    )

    assert result == "delegated"
    assert calls[0][0] == "prepare the report"
    assert calls[0][1]["user_id"] == "alice"
    assert calls[0][1]["context"] == {"quarter": "Q3"}
    assert calls[0][1]["tool_context"] == {"request_id": "req-1"}


@pytest.mark.unit
def test_shared_orchestration_lookup_is_scoped_by_workflow_and_user():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(
        root,
        [delegate],
        workflow_id="workflow-7",
    )
    shared = Mock()
    shared.find_active_session_for_agent.return_value = None
    shared.create_shared_session.return_value = "shared-1"
    orchestrator.shared_memory = shared
    orchestrator._request_user_id = "alice"
    orchestrator._trace_id = "trace-1"

    assert orchestrator._find_or_create_shared_session() is None
    shared.find_active_session_for_agent.assert_called_once_with(
        "root",
        workflow_id="workflow-7",
        user_id="alice",
    )
    shared.create_shared_session.assert_called_once_with(
        root_agent_id="root",
        delegate_agent_ids=["delegate-1"],
        workflow_id="workflow-7",
        user_id="alice",
        trace_id="trace-1",
    )


@pytest.mark.unit
def test_toolbox_deterministic_registration_is_llm_and_embedding_free(monkeypatch):
    provider = MockMemoryProvider()

    def lookup_customer(customer_id: str, include_orders: bool = False) -> dict:
        """Look up a customer and optionally include orders."""
        return {"customer_id": customer_id, "include_orders": include_orders}

    monkeypatch.setattr(
        "memorizz.long_term.procedural.toolbox.toolbox.get_embedding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deterministic registration must not embed")
        ),
    )
    toolbox = Toolbox.from_functions(
        [lookup_customer],
        memory_provider=provider,
        agent_id="agent-1",
        augment=False,
        persist=True,
    )
    assert toolbox.llm_provider is None
    assert toolbox._tools_by_name["lookup_customer"] is lookup_customer
    write = next(
        call
        for call in provider.call_history
        if call[0] == "store" and call[3] == MemoryType.TOOLBOX
    )
    metadata = write[2]
    assert "embedding" not in metadata
    assert metadata["input_schema"]["required"] == ["customer_id"]
    assert metadata["input_schema"]["properties"]["include_orders"]["default"] is False
    assert metadata["input_schema"]["additionalProperties"] is False
    assert "pickle" not in json.dumps(metadata).lower()


@pytest.mark.unit
def test_task_decomposer_uses_configured_provider_and_callable_capabilities():
    class PlanningProvider(MockLLMProvider):
        def generate(self, messages, tools=None, tool_choice="auto"):
            self.call_count += 1
            self.last_messages = messages
            return json.dumps(
                [
                    {
                        "task_id": "task-1",
                        "description": "Calculate the total",
                        "assigned_agent_id": "delegate-1",
                        "priority": 1,
                        "dependencies": [],
                    }
                ]
            )

    def calculate_total(values: list[int]) -> int:
        """Calculate a total."""
        return sum(values)

    provider = MockMemoryProvider()
    planning_model = PlanningProvider(provider="custom-provider", model="custom-model")
    root = MemAgent(
        model=planning_model,
        memory_provider=provider,
        agent_id="root",
    )
    delegate = MemAgent(
        model=MockLLMProvider(),
        tools=[calculate_total],
        memory_provider=provider,
        agent_id="delegate-1",
    )
    decomposer = TaskDecomposer(root)
    capabilities = decomposer.analyze_delegate_capabilities([delegate])
    assert "calculate_total" in {
        item["name"] for item in capabilities["delegate-1"]["tools"]
    }
    tasks = decomposer.decompose_task("Total these values", [delegate])
    assert planning_model.call_count == 1
    assert tasks[0].assigned_agent_id == "delegate-1"
    source = (
        Path(__file__).parents[2] / "src" / "memorizz" / "task_decomposition.py"
    ).read_text()
    orchestrator_source = (
        Path(__file__).parents[2] / "src" / "memorizz" / "multi_agent_orchestrator.py"
    ).read_text()
    assert "gpt-4.1" not in source + orchestrator_source
    assert ".client.responses.create" not in source + orchestrator_source


@pytest.mark.unit
def test_task_decomposer_accepts_callable_tool_metadata_without_dict_assumptions():
    def perform_lookup(customer_id: str) -> dict:
        """Look up one customer."""
        return {"customer_id": customer_id}

    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    delegate.tool_manager.get_tool_metadata = lambda: [perform_lookup, object()]

    capabilities = TaskDecomposer(root).analyze_delegate_capabilities([delegate])

    assert capabilities["delegate-1"]["tools"] == [
        {"name": "perform_lookup", "description": "Look up one customer."}
    ]


@pytest.mark.unit
def test_semantic_layer_emits_governed_parameterized_plan_and_round_trips():
    catalog = SemanticCatalog(
        [
            {
                "name": "commerce",
                "version": "2026-08",
                "entities": [
                    {
                        "name": "orders",
                        "source": "analytics.orders",
                        "primary_key": "id",
                    }
                ],
                "measures": [
                    {
                        "name": "revenue",
                        "entity": "orders",
                        "expression": "orders.amount",
                        "aggregation": "sum",
                    }
                ],
                "dimensions": [
                    {
                        "name": "region",
                        "entity": "orders",
                        "expression": "orders.region",
                    }
                ],
                "policies": [
                    {
                        "name": "analyst_access",
                        "entity": "orders",
                        "allowed_roles": ["analyst"],
                        "row_filter": "orders.deleted_at IS NULL",
                    }
                ],
                "synonyms": [
                    {"alias": "sales", "target": "revenue", "kind": "measure"}
                ],
            }
        ]
    )
    plan = catalog.plan(
        "commerce",
        {
            "measures": ["sales"],
            "dimensions": ["region"],
            "filters": [{"field": "region", "operator": "eq", "value": "EMEA"}],
            "roles": ["analyst"],
            "limit": 50,
        },
    )
    assert plan.valid is True
    assert "EMEA" not in plan.sql
    assert plan.parameters["filter_0"] == "EMEA"
    assert "orders.deleted_at IS NULL" in plan.sql
    assert plan.policies == ["analyst_access"]
    assert plan.lineage
    restored = SemanticCatalog.from_dict(catalog.to_dict())
    assert restored.describe("commerce")["version"] == "2026-08"
    with pytest.raises(PermissionError):
        catalog.plan("commerce", {"measures": ["revenue"], "roles": ["guest"]})
