"""Cost and token limits on MemAgent harness runs, and readable refusals."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    HarnessPermissions,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness import adapters as adapters_module
from memorizz.metaharness.adapters import PersistedMemAgentHarness, _cost_limit_problem
from memorizz.metaharness.router import HarnessReadinessError
from tests.mocks.mock_providers import MockLLMProvider

pytestmark = pytest.mark.unit


def _task(**budget) -> HarnessTask:
    return HarnessTask(
        task="t", workspace="/tmp", harness="memagent", agent_id="a", budget=budget
    )


def test_cost_limits_need_a_priced_or_local_model():
    assert (
        _cost_limit_problem(_task(), {"provider": "openai", "model": "made-up"}) is None
    )
    capped = _task(max_cost_usd=0.5)
    assert (
        _cost_limit_problem(capped, {"provider": "ollama", "model": "qwen2.5:7b"})
        is None
    )
    assert _cost_limit_problem(capped, {"provider": "mlx", "model": "any"}) is None
    assert (
        _cost_limit_problem(capped, {"provider": "openai", "model": "gpt-4.1-mini"})
        is None
    )
    assert (
        _cost_limit_problem(
            capped, {"provider": "anthropic", "model": "claude-haiku-4-5"}
        )
        is None
    )
    refused = _cost_limit_problem(
        capped, {"provider": "openai", "model": "gpt-made-up"}
    )
    assert refused.error_code == "cost_budget_unpriced_model"
    assert (
        "openai/gpt-made-up" in refused.error
        and "Remove the cost limit" in refused.remediation
    )


def test_saved_memagent_harness_reports_usage_and_only_asks_for_setup_when_needed(
    tmp_path: Path,
):
    from memorizz.memagent import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path, lazy_vector_indexes=True)
    )
    try:
        empty = PersistedMemAgentHarness(provider).probe()
        assert empty.remediation and "Create and save a MemAgent" in empty.remediation
        MemAgent(memory_provider=provider).save()
        ready = PersistedMemAgentHarness(provider).probe()
        assert ready.remediation is None and ready.error is None
        assert ready.usage_reporting is True
        assert (
            ready.metadata["cost_reporting"] is True
            and ready.metadata["token_reporting"] is True
        )
    finally:
        provider.close()


def _saved_agent_service(tmp_path: Path, monkeypatch):
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    agent = (
        MemAgentBuilder()
        .with_name("Order Desk")
        .with_model(MockLLMProvider(["order 1042 shipped"]))
        .with_memory_provider(provider)
        .build_and_save()
    )
    agent.close(close_memory_provider=False)
    monkeypatch.setattr(
        "memorizz.memagent.core.create_llm_provider",
        lambda _config: MockLLMProvider(["order 1042 shipped"]),
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[PersistedMemAgentHarness(provider)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return provider, agent.agent_id, service, workspace


def test_a_cost_limit_no_longer_blocks_a_local_memagent(tmp_path: Path, monkeypatch):
    provider, agent_id, service, workspace = _saved_agent_service(tmp_path, monkeypatch)
    try:
        run = lambda: service.run(
            HarnessTask(
                task="Where is order 1042?",
                workspace=str(workspace),
                harness="memagent",
                agent_id=agent_id,
                permissions=HarnessPermissions(mcp_access="none"),
                budget={"max_cost_usd": 0.5, "max_output_tokens": 10_000},
            )
        )
        # A model MemoRizz can't price: refused with a reason and a fix, not
        # "create and save a MemAgent".
        refused = run()
        assert refused.status == HarnessStatus.FAILED
        assert refused.error_code == "cost_budget_unpriced_model"
        # A local model has no API charges, so the limit holds.
        monkeypatch.setattr(
            adapters_module,
            "_agent_model",
            lambda agent: {"provider": "ollama", "model": "qwen2.5:7b"},
        )
        done = run()
        assert done.status == HarnessStatus.SUCCEEDED
        assert done.final_response == "order 1042 shipped"
    finally:
        service.close()
        provider.close()


def test_provider_error_is_a_failed_run_and_not_a_cached_answer(tmp_path, monkeypatch):
    from memorizz.memagent import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.metaharness import NativeMemAgentHarness
    from memorizz.tool_context import get_tool_context

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    model = MockLLMProvider(["unused"])
    error = RuntimeError("Function tools with reasoning_effort are not supported")
    monkeypatch.setattr(
        model, "generate", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )
    agent = MemAgent(model=model, memory_provider=provider, auto_register=False)
    monkeypatch.setattr(agent.cache_manager, "enabled", True)
    monkeypatch.setattr(
        agent.cache_manager, "get_cached_response", lambda *a, **kw: None
    )
    monkeypatch.setattr(
        agent.cache_manager,
        "cache_response",
        lambda *a, **kw: pytest.fail("Do not cache failed answers"),
    )
    service = MetaHarness(
        adapters=[NativeMemAgentHarness(agent)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Check",
                workspace=str(workspace),
                harness="memagent",
                permissions=HarnessPermissions(mcp_access="none"),
            )
        )
        assert result.status == HarnessStatus.FAILED and not result.ok
        assert result.error_code == "memagent_run_failed"
        assert "Function tools with reasoning_effort" in result.error
        assert not result.final_response
        assert not get_tool_context().get("_memorizz_harness_native")
        assert service.get_run(result.run_id)["status"] == "failed"
    finally:
        service.close()
        agent.close(close_memory_provider=False)
        provider.close()


def test_a_task_a_harness_cannot_run_is_explained_in_words(tmp_path: Path, monkeypatch):
    provider, agent_id, service, workspace = _saved_agent_service(tmp_path, monkeypatch)
    try:
        with pytest.raises(HarnessReadinessError) as error:
            service.router.select(
                HarnessTask(
                    task="Return JSON",
                    workspace=str(workspace),
                    harness="memagent",
                    agent_id=agent_id,
                    output_schema={"type": "object"},
                    permissions=HarnessPermissions(mcp_access="none"),
                ),
                service.adapters if hasattr(service, "adapters") else None,
            )
        message = str(error.value)
        assert "can't run this task" in message and "structured output" in message
        assert "Remove the output schema" in error.value.remediation
        assert "Create and save" not in error.value.remediation
    finally:
        service.close()
        provider.close()
