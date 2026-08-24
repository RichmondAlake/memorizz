"""Builder, deterministic Toolbox, semantic layer and orchestration parity."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.completion import CompletionPolicy
from memorizz.enums import MemoryType
from memorizz.long_term.procedural.toolbox import Toolbox
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.metaharness import HarnessContextPack, HarnessResult, HarnessStatus
from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator
from memorizz.sandbox.base import SandboxProvider
from memorizz.sandbox.models import ExecutionResult
from memorizz.semantic_layer import SemanticCatalog, SemanticModel
from memorizz.task_decomposition import SubTask, TaskDecomposer
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
def test_verified_read_only_delegation_can_reuse_semantic_cache():
    provider = MockMemoryProvider()
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    delegate.meta_harness = Mock()
    delegate.meta_harness_mode = "runtime"
    delegate.default_harness = "codex"
    delegate.harness_config = {
        "workspace": "/tmp/project",
        "permissions": {
            "workspace_mode": "read_only",
            "network": "none",
            "mcp_access": "none",
        },
        "verification": {"command": "python -m pytest -q"},
    }
    delegate.run_on_harness = Mock(side_effect=AssertionError("cache should win"))
    plan = [
        {
            "task_id": "review",
            "description": "Review the code",
            "assigned_agent_id": delegate.agent_id,
            "priority": 1,
            "dependencies": [],
        }
    ]
    root = MemAgent(
        memory_provider=provider,
        agent_id="root",
        delegates=[delegate],
        delegation={
            "enabled": True,
            "mode": "deterministic",
            "plan": plan,
            "return_report": True,
        },
    )
    root.cache_manager.enabled = True
    root.cache_manager.get_cached_response = Mock(return_value="Cached review")
    root.cache_manager.cache_response = Mock()

    report = root.delegate(
        "Review the code",
        memory_id="memory-1",
        thread_id="thread-1",
        user_id="alice",
        context={"cache_data_version": "workspace-and-memory-v1"},
        return_report=True,
    )

    assert report["cached"] is True
    assert report["response"] == "Cached review"
    assert report["consolidation"]["strategy"] == "semantic_cache"
    assert (
        root.cache_manager.get_cached_response.call_args.kwargs["bypass_reason"] is None
    )
    delegate.run_on_harness.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("context", "mode", "workspace_mode", "verification", "expected_reason"),
    [
        ({}, "deterministic", "read_only", True, "delegation_data_version_required"),
        (
            {"cache_data_version": "v1"},
            "auto",
            "read_only",
            True,
            "nondeterministic_delegation_plan",
        ),
        (
            {"cache_data_version": "v1"},
            "deterministic",
            "direct",
            True,
            "side_effecting_delegation",
        ),
        (
            {"cache_data_version": "v1"},
            "deterministic",
            "read_only",
            False,
            "unverified_delegation",
        ),
    ],
)
def test_delegation_cache_admission_fails_closed(
    context, mode, workspace_mode, verification, expected_reason
):
    provider = MockMemoryProvider()
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    delegate.meta_harness = Mock()
    delegate.meta_harness_mode = "runtime"
    delegate.harness_config = {
        "permissions": {
            "workspace_mode": workspace_mode,
            "network": "none",
            "mcp_access": "none",
        },
        "verification": ({"command": "python -m pytest -q"} if verification else None),
    }
    plan = [
        {
            "task_id": "review",
            "description": "Review the code",
            "assigned_agent_id": delegate.agent_id,
            "dependencies": [],
        }
    ]
    root = MemAgent(
        memory_provider=provider,
        delegates=[delegate],
        delegation={"enabled": True, "mode": mode, "plan": plan},
    )
    root.cache_manager.enabled = True

    _, bypass_reason = root._delegation_cache_context(context, plan)

    assert bypass_reason == expected_reason


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
def test_model_less_orchestrator_uses_supported_deterministic_consolidation(caplog):
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(root, [delegate])
    results = [
        {
            "description": "Review the implementation",
            "result": "The implementation preserves the invariant.",
            "status": "completed",
        }
    ]

    with caplog.at_level("INFO"):
        response = orchestrator._consolidate_results("Check the code", results)

    assert "Results for: Check the code" in response
    assert "The implementation preserves the invariant." in response
    report = dict(orchestrator._last_consolidation_report or {})
    assert report.pop("latency_ms") >= 0
    assert report == {
        "status": "succeeded",
        "strategy": "deterministic",
        "model_used": False,
        "reason": "root_agent_has_no_llm_provider",
    }
    assert "Root agent has no configured LLMProvider" not in caplog.text


@pytest.mark.unit
def test_orchestrator_reports_model_synthesis_usage():
    provider = MockMemoryProvider()
    model = MockLLMProvider(
        responses=["A unified evidence-based review."],
        provider="test-provider",
        model="test-synthesizer",
    )
    model.last_usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }
    root = MemAgent(
        model=model,
        memory_provider=provider,
        agent_id="root",
        instruction="Preserve evidence identifiers and resolve contradictions.",
    )
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(root, [delegate])

    response = orchestrator._consolidate_results(
        "Check the code",
        [{"description": "Review", "result": "Evidence", "status": "completed"}],
    )

    assert response == "A unified evidence-based review."
    assert "ROOT COORDINATOR INSTRUCTION" in model.last_messages[0]["content"]
    assert "Preserve evidence identifiers" in model.last_messages[0]["content"]
    report = orchestrator._last_consolidation_report
    assert report is not None
    assert report["status"] == "succeeded"
    assert report["strategy"] == "model"
    assert report["model_used"] is True
    assert report["provider"] == "test-provider"
    assert report["model"] == "test-synthesizer"
    assert report["usage"] == model.last_usage


@pytest.mark.unit
def test_orchestrator_model_synthesis_is_grounded_in_coordinator_memory():
    provider = MockMemoryProvider()
    model = MockLLMProvider(responses=["Grounded synthesis"])
    root = MemAgent(model=model, memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(root, [delegate])
    orchestrator._coordinator_memory_context = Mock(
        return_value=(
            "[memory:knowledge_base:source-1] auditors may not export",
            {
                "available": True,
                "strategy": "evidence_pack",
                "source_ids": ["source-1"],
            },
        )
    )

    response = orchestrator._consolidate_results(
        "Review export authorization",
        [
            {
                "task_id": "review",
                "description": "Review",
                "assigned_agent_id": "delegate-1",
                "result": "Auditors can currently export.",
                "status": "completed",
            }
        ],
    )

    assert response == "Grounded synthesis"
    assert "auditors may not export" in model.last_messages[1]["content"]
    assert "must not create new factual claims" in model.last_messages[0]["content"]
    assert orchestrator._last_consolidation_report["memory_context"]["source_ids"] == [
        "source-1"
    ]


@pytest.mark.unit
def test_primary_consolidation_avoids_an_extra_model_call():
    provider = MockMemoryProvider()
    model = MockLLMProvider(responses=["must not be called"])
    root = MemAgent(model=model, memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(
        root,
        [delegate],
        consolidation_strategy="primary",
        primary_task_id="primary-review",
    )

    response = orchestrator._consolidate_results(
        "Review the code",
        [
            {
                "task_id": "primary-review",
                "description": "Primary review",
                "result": "Evidence-preserving primary answer",
                "status": "completed",
            },
            {
                "task_id": "independent-check",
                "description": "Independent check",
                "result": "No blocking contradiction",
                "status": "completed",
            },
        ],
    )

    assert response == "Evidence-preserving primary answer"
    assert model.call_count == 0
    assert orchestrator._last_consolidation_report["strategy"] == "primary"
    assert orchestrator._last_consolidation_report["review_task_ids"] == [
        "independent-check"
    ]


@pytest.mark.unit
def test_structured_consolidation_preserves_coverage_without_model_call():
    provider = MockMemoryProvider()
    model = MockLLMProvider(responses=["must not be called"])
    root = MemAgent(model=model, memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(
        root,
        [delegate],
        consolidation_strategy="structured",
        required_finding_ids=["authorization", "verification"],
        evidence_context=False,
    )
    results = [
        {
            "task_id": "primary",
            "status": "completed",
            "result": json.dumps(
                {
                    "findings": [
                        {
                            "criterion_id": "authorization",
                            "title": "Role bypass",
                            "finding": "Auditors can export.",
                            "severity": "high",
                            "evidence": ["policy.py:4"],
                            "source_ids": ["source-1"],
                        },
                        {
                            "criterion_id": "verification",
                            "title": "Assertions can disappear",
                            "finding": "Bare asserts are removed by -O.",
                            "severity": "medium",
                            "evidence": ["verify.py:1"],
                        },
                    ]
                }
            ),
        },
        {
            "task_id": "critic",
            "status": "completed",
            "result": json.dumps(
                {
                    "findings": [
                        {
                            "criterion_id": "authorization",
                            "title": "Auditor export bypass",
                            "finding": "The auditor role returns True before expiry.",
                            "severity": "high",
                            "evidence": ["policy.py:4-5"],
                        }
                    ]
                }
            ),
        },
    ]

    response = orchestrator._consolidate_results("Review", results)

    assert model.call_count == 0
    assert "[authorization]" in response
    assert "[verification]" in response
    report = orchestrator._last_consolidation_report
    assert report["strategy"] == "structured"
    assert report["model_used"] is False
    assert report["coverage"]["coverage_complete"] is True
    assert report["coverage"]["deduplicated_finding_count"] == 1


@pytest.mark.unit
def test_adaptive_escalation_skips_second_reviewer_when_coverage_is_complete():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegates = [
        MemAgent(memory_provider=provider, agent_id="primary-agent"),
        MemAgent(memory_provider=provider, agent_id="fallback-agent"),
    ]
    orchestrator = MultiAgentOrchestrator(
        root,
        delegates,
        required_finding_ids=["a", "b"],
        adaptive_escalation={
            "enabled": True,
            "escalation_task_ids": ["fallback"],
        },
    )
    primary_result = {
        "task_id": "primary",
        "status": "completed",
        "result": json.dumps(
            {
                "findings": [
                    {"criterion_id": "a", "title": "A", "finding": "A found"},
                    {"criterion_id": "b", "title": "B", "finding": "B found"},
                ]
            }
        ),
    }
    orchestrator._execute_sub_tasks_parallel = Mock(return_value=[primary_result])
    tasks = [
        SubTask("primary", "Primary", "primary-agent"),
        SubTask("fallback", "Fallback", "fallback-agent"),
    ]

    results = orchestrator._execute_sub_tasks(tasks, "memory-1", "thread-1")

    orchestrator._execute_sub_tasks_parallel.assert_called_once()
    assert [result["status"] for result in results] == ["completed", "skipped"]
    assert orchestrator._last_execution_report["escalated"] is False
    assert orchestrator._last_execution_report["missing_ids"] == []


@pytest.mark.unit
def test_adaptive_escalation_targets_only_missing_criteria():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegates = [
        MemAgent(memory_provider=provider, agent_id="primary-agent"),
        MemAgent(memory_provider=provider, agent_id="fallback-agent"),
    ]
    orchestrator = MultiAgentOrchestrator(
        root,
        delegates,
        required_finding_ids=["a", "b"],
        adaptive_escalation={
            "enabled": True,
            "escalation_task_ids": ["fallback"],
            "criterion_descriptions": {"b": "verification integrity"},
        },
    )
    primary_result = {
        "task_id": "primary",
        "status": "completed",
        "result": json.dumps(
            {"findings": [{"criterion_id": "a", "title": "A", "finding": "A found"}]}
        ),
    }
    fallback_result = {
        "task_id": "fallback",
        "status": "completed",
        "result": json.dumps(
            {"findings": [{"criterion_id": "b", "title": "B", "finding": "B found"}]}
        ),
    }
    orchestrator._execute_sub_tasks_parallel = Mock(
        side_effect=[[primary_result], [fallback_result]]
    )
    tasks = [
        SubTask("primary", "Primary", "primary-agent"),
        SubTask("fallback", "Fallback", "fallback-agent"),
    ]

    results = orchestrator._execute_sub_tasks(tasks, "memory-1", "thread-1")

    assert len(results) == 2
    assert orchestrator._execute_sub_tasks_parallel.call_count == 2
    assert "b: verification integrity" in tasks[1].description
    assert orchestrator._last_execution_report["escalated"] is True
    assert orchestrator._last_execution_report["missing_ids"] == []


@pytest.mark.unit
def test_model_consolidation_honors_host_completion_policy():
    provider = MockMemoryProvider()
    model = MockLLMProvider(responses=["draft", "accepted"])
    root = MemAgent(
        model=model,
        memory_provider=provider,
        agent_id="root",
        completion_policy=CompletionPolicy(
            enabled=True,
            max_rejections=1,
            validator=lambda candidate: candidate.response == "accepted",
        ),
    )
    orchestrator = MultiAgentOrchestrator(root, [], evidence_context=False)

    response = orchestrator._consolidate_results(
        "Review",
        [{"task_id": "review", "status": "completed", "result": "Evidence"}],
    )

    assert response == "accepted"
    assert model.call_count == 2
    report = orchestrator._last_consolidation_report
    assert report["generation_attempts"] == 2
    assert [item["accepted"] for item in report["completion_decisions"]] == [
        False,
        True,
    ]


@pytest.mark.unit
def test_builder_ephemeral_mode_disables_registration_and_automations():
    provider = MockMemoryProvider()
    agent = (
        MemAgentBuilder()
        .with_memory_provider(provider)
        .with_memory_ids("memory-1")
        .as_ephemeral()
        .build(validate=False)
    )

    assert agent.auto_register is False
    assert agent.automations_enabled is False
    agent._ensure_agent_registered()
    assert not any(call[0] == "store_memagent" for call in provider.call_history)


@pytest.mark.unit
def test_coordinator_reuses_identical_bounded_delegate_evidence_pack():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    orchestrator = MultiAgentOrchestrator(root, [])
    orchestrator._request_memory_id = "memory-1"
    pack = HarnessContextPack(
        query="review policy",
        rendered="MEMORIZZ EVIDENCE PACK\n[source-1] policy requirement",
        source_ids=["source-1"],
        token_estimate=20,
        metadata={
            "pack_id": "pack-1",
            "candidate_count": 3,
            "selected_count": 1,
            "tokens_saved": 80,
        },
    )
    orchestrator._delegate_context_packs = {"review-a": pack, "review-b": pack}

    rendered, metadata = orchestrator._coordinator_memory_context("review policy")

    assert rendered == pack.rendered
    assert metadata["strategy"] == "shared_harness_evidence_pack"
    assert metadata["context_snapshot_reused"] is True
    assert metadata["delegate_task_ids"] == ["review-a", "review-b"]
    assert metadata["tokens_saved"] == 80


@pytest.mark.unit
def test_dependency_results_reach_later_delegate_as_bounded_model_context():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    delegate.run = Mock(return_value="reviewed")
    orchestrator = MultiAgentOrchestrator(
        root, [delegate], max_dependency_context_chars=1_000
    )
    orchestrator.shared_memory = Mock()
    orchestrator.shared_memory_id = "shared-1"
    orchestrator._request_user_id = "alice"
    orchestrator._request_context = {
        "memory_query": "Original review",
        "shared_context_key": "workflow-1",
    }
    task = SubTask(
        task_id="critic",
        description="Check the primary review",
        assigned_agent_id=delegate.agent_id,
        dependencies=["primary"],
    )

    result = orchestrator._execute_single_task(
        task,
        delegate,
        memory_id="memory-1",
        thread_id="thread-1",
        dependency_results={"primary": "Primary evidence"},
    )

    assert result == "reviewed"
    context = delegate.run.call_args.kwargs["context"]
    assert context["dependency_results"] == {"primary": "Primary evidence"}
    assert context["model_context"]["dependency_results"] == {
        "primary": "Primary evidence"
    }


@pytest.mark.unit
def test_orchestrator_exposes_model_failure_before_deterministic_fallback():
    class BrokenProvider(MockLLMProvider):
        def generate(self, messages, tools=None, tool_choice="auto"):
            raise RuntimeError("provider unavailable")

    provider = MockMemoryProvider()
    root = MemAgent(model=BrokenProvider(), memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    orchestrator = MultiAgentOrchestrator(root, [delegate])

    response = orchestrator._consolidate_results(
        "Check the code",
        [{"description": "Review", "result": "Evidence", "status": "completed"}],
    )

    assert "Result: Evidence" in response
    report = orchestrator._last_consolidation_report
    assert report is not None
    assert report["status"] == "failed"
    assert report["strategy"] == "model"
    assert report["fallback_strategy"] == "deterministic"
    assert report["fallback_used"] is True
    assert report["error_code"] == "RuntimeError"


@pytest.mark.unit
def test_orchestrator_marks_failed_runtime_harness_delegate_as_failed():
    provider = MockMemoryProvider()
    root = MemAgent(memory_provider=provider, agent_id="root")
    delegate = MemAgent(memory_provider=provider, agent_id="delegate-1")
    delegate.meta_harness = Mock()
    delegate.meta_harness_mode = "runtime"
    delegate.run_on_harness = Mock(
        return_value=HarnessResult(
            run_id="run-1",
            harness="claude-code",
            status=HarnessStatus.BUDGET_EXCEEDED,
            error_code="step_budget_exceeded",
            error="Harness step budget was exceeded",
        )
    )
    orchestrator = MultiAgentOrchestrator(root, [delegate])
    orchestrator.shared_memory = Mock()
    orchestrator.shared_memory_id = "shared-1"
    orchestrator._request_user_id = "alice"

    task = SubTask(
        task_id="review",
        description="Review the code",
        assigned_agent_id=delegate.agent_id,
    )
    with pytest.raises(RuntimeError, match="step_budget_exceeded"):
        orchestrator._execute_single_task(
            task, delegate, memory_id="memory-1", thread_id="thread-1"
        )

    delegate.run_on_harness.assert_called_once_with(
        "Review the code",
        memory_id="memory-1",
        thread_id="thread-1",
        user_id="alice",
        context={
            "delegation": {
                "workflow_id": orchestrator.workflow_id,
                "trace_id": None,
                "task_id": "review",
                "dependencies": [],
            }
        },
    )


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
