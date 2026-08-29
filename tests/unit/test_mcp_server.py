"""First-party Memorizz MCP server coverage."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.mcp import MCPClientManager
from memorizz.mcp_server import (
    MemorizzMCPServerConfig,
    MemorizzRuntime,
    StaticAPIKeyGrant,
    create_memorizz_mcp_server,
)
from memorizz.mcp_server.auth import RequestIdentity, StaticAPIKeyVerifier
from memorizz.mcp_server.config import (
    ALL_SCOPES,
    EXECUTE_SCOPE,
    READ_SCOPE,
    WRITE_SCOPE,
    parse_api_key_grants,
)
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessEvent,
    HarnessEventType,
    HarnessStatus,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome
from memorizz.personalization import build_personalization_context


def _identity(principal: str, *scopes: str) -> RequestIdentity:
    return RequestIdentity(
        principal=principal,
        scopes=frozenset(scopes or ALL_SCOPES),
        authenticated=True,
    )


def _filesystem_provider(tmp_path: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
        )
    )


def _structured(result):
    return result["result"]["structuredContent"]["result"]


class _MCPFakeHarness(AgentHarness):
    name = "fake"

    def probe(self):
        return HarnessCapabilities(name=self.name, available=True, mcp=True)

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"role": "assistant", "text": "MCP harness complete"},
            )
        )
        if task.writes_workspace:
            (workspace / "mcp-result.txt").write_text("complete\n", encoding="utf-8")
        return AdapterOutcome(final_response="MCP harness complete", exit_code=0)


class _MCPAuthRequiredHarness(AgentHarness):
    name = "mcp-auth-required"

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            command="mcp-auth-required",
            error_code="authentication_required",
            error="The harness API key is not configured.",
            remediation="Set TEST_HARNESS_API_KEY before starting this harness.",
            metadata={"authentication_configured": False},
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        raise AssertionError("An unauthenticated harness must not start")


@pytest.mark.unit
def test_server_config_requires_explicit_remote_auth_and_never_exports_tokens():
    token = "alice-token-that-is-long-enough"
    grants = parse_api_key_grants(
        json.dumps(
            {
                "alice": token,
                "reader": {
                    "token": "reader-token-that-is-long-enough",
                    "scopes": [READ_SCOPE],
                },
            }
        )
    )
    config = MemorizzMCPServerConfig(
        transport="streamable-http",
        public_url="https://memory.example.com",
        api_key_grants=grants,
    )

    assert config.auth_required is True
    assert config.allow_writes is False
    assert config.allow_agent_execution is False
    assert config.public_dict()["agent_management"] == {
        "available": False,
        "operations": ["create", "update", "delete"],
        "transport": "stdio",
        "remote_http": False,
    }
    assert token not in json.dumps(config.public_dict())

    with pytest.raises(ValueError, match="requires MEMORIZZ_MCP_SERVER_API_KEYS"):
        MemorizzMCPServerConfig(transport="streamable-http")
    with pytest.raises(ValueError, match="must use HTTPS"):
        MemorizzMCPServerConfig(
            transport="streamable-http",
            host="0.0.0.0",
            public_url="http://memory.example.com",
            api_key_grants=grants,
        )
    with pytest.raises(ValueError, match="explicit --agent-id"):
        MemorizzMCPServerConfig(
            transport="streamable-http",
            public_url="https://memory.example.com",
            api_key_grants=grants,
            allow_writes=True,
            allow_agent_execution=True,
        )


@pytest.mark.unit
def test_static_api_key_verifier_assigns_stable_principal_and_scopes():
    grant = StaticAPIKeyGrant("alice", "alice-token-that-is-long-enough", (READ_SCOPE,))
    verifier = StaticAPIKeyVerifier([grant], "https://memory.example.com/mcp")

    accepted = asyncio.run(verifier.verify_token(grant.token))
    rejected = asyncio.run(verifier.verify_token("not-the-token"))

    assert accepted is not None
    assert accepted.subject == "alice"
    assert accepted.scopes == [READ_SCOPE]
    assert rejected is None


@pytest.mark.unit
def test_runtime_memory_operations_are_tenant_isolated(tmp_path):
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=True),
        provider=_filesystem_provider(tmp_path),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    alice = _identity("alice")
    bob = _identity("bob")

    stored = runtime.remember(
        "Alice remembers the launch date",
        "knowledge_base",
        alice,
        metadata={"source": "meeting", "token": "must-not-leak"},
    )

    alice_rows = runtime.list_memories("knowledge_base", alice)
    bob_rows = runtime.list_memories("knowledge_base", bob)
    assert alice_rows["count"] == 1
    assert alice_rows["memories"][0]["token"] == "***"
    assert bob_rows["count"] == 0

    with pytest.raises(MemorizzServerError, match="Memory was not found"):
        runtime.get_memory(stored["record_id"], "knowledge_base", bob)
    proposed = runtime.forget(stored["record_id"], "knowledge_base", alice)
    assert proposed["status"] == "approval_required"
    proposal_id = proposed["proposal"]["proposal_id"]
    runtime.approve_proposal(proposal_id, approver_id="operator@example.com")
    assert runtime.resume_proposal(proposal_id)["deleted"] is True

    with pytest.raises(MemorizzServerError, match=WRITE_SCOPE):
        runtime.remember("denied", "knowledge_base", _identity("reader", READ_SCOPE))


class _FakeAgent:
    agent_id = "agent-public"
    name = "Public memory assistant"
    instruction = "Help with memory"
    application_mode = "assistant"
    memory_ids = ["private-conversation-id"]
    knowledge_base_ids = ["private-knowledge-id"]
    model = object()
    tools = ["secret-tool"]
    memory_provider = object()

    def __init__(self):
        self.calls = []
        self.saves = 0
        self.last_tool_outcomes = [
            {
                "tool_name": "calendar_lookup",
                "status": "fallback",
                "ok": True,
                "fallback_used": True,
            }
        ]

    def run(self, message, **kwargs):
        self.calls.append((message, kwargs))
        return f"remembered: {message}"

    def save(self):
        self.saves += 1


class _FakeAgentProvider:
    def __init__(self, agent):
        self.agent = agent

    def list_memagents(self):
        return [self.agent]

    def retrieve_memagent(self, agent_id):
        return self.agent if agent_id == self.agent.agent_id else None


@pytest.mark.unit
def test_agent_execution_is_isolated_and_public_metadata_is_secret_free(tmp_path):
    agent = _FakeAgent()
    provider = _FakeAgentProvider(agent)
    config = MemorizzMCPServerConfig(
        allow_writes=True,
        allow_agent_execution=True,
        exposed_agent_ids=set(),
    )
    runtime = MemorizzRuntime(
        config,
        provider=provider,
        session_builder=lambda **_kwargs: SimpleNamespace(agent=agent),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    alice = _identity("alice")
    bob = _identity("bob")

    first = runtime.execute_agent("hello", alice)
    second = runtime.execute_agent("hello", bob, agent_id=agent.agent_id)

    assert first["memory_id"] != second["memory_id"]
    assert first["thread_id"] != second["thread_id"]
    assert first["tool_outcomes"][0]["status"] == "fallback"
    assert agent.calls[0][1]["user_id"] == "alice"
    assert agent.calls[1][1]["user_id"] == "bob"
    public = runtime.get_agent(agent.agent_id, alice)["agent"]
    assert "memory_ids" not in public
    assert "knowledge_base_ids" not in public
    assert "model" not in public
    assert "tools" not in public

    with pytest.raises(MemorizzServerError, match=WRITE_SCOPE):
        runtime.execute_agent(
            "denied",
            _identity("executor", READ_SCOPE, EXECUTE_SCOPE),
            agent_id=agent.agent_id,
        )

    remote_anonymous = MemorizzRuntime(
        MemorizzMCPServerConfig(transport="streamable-http", allow_anonymous_http=True),
        provider=runtime.provider,
        approval_store=SQLiteApprovalStore(tmp_path / "remote-approvals.sqlite3"),
    )
    with pytest.raises(MemorizzServerError, match="not tenant scoped"):
        remote_anonymous.list_memories(
            "personas",
            RequestIdentity(None, frozenset(ALL_SCOPES), authenticated=False),
        )


@pytest.mark.unit
def test_local_runtime_creates_persisted_agent_and_blocks_remote_creation(
    tmp_path, monkeypatch
):
    provider = _filesystem_provider(tmp_path)
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(
            allow_writes=True,
            harness_workspace_roots={str(tmp_path)},
        ),
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    identity = _identity("local-operator")

    created = runtime.create_agent(
        "MCP Agent",
        identity,
        instruction="Created over local MCP.",
        memory_ids=["mcp-memory", "mcp-memory"],
        semantic_cache=True,
        learning_control_plane=True,
        skill_retrieval=True,
        skill_retrieval_top_k=3,
        tool_access="public",
        automations_enabled=False,
        default_timezone="Europe/London",
        meta_harness_mode="delegate",
        default_harness="codex",
        harness_workspace=str(tmp_path),
    )
    agent_id = created["agent"]["agent_id"]
    assert created["created"] is True
    assert created["agent"]["instruction"] == "Created over local MCP."
    persisted = provider.retrieve_memagent(agent_id)
    assert persisted is not None
    assert persisted.memory_ids == ["mcp-memory"]
    assert persisted.semantic_cache is True
    assert persisted.learning_control_plane is True
    assert persisted.skill_retrieval is True
    assert persisted.skill_retrieval_config["top_k"] == 3
    assert persisted.tool_access == "public"
    assert persisted.automations_enabled is False
    assert persisted.default_timezone == "Europe/London"
    assert persisted.meta_harness is True
    assert persisted.meta_harness_mode == "delegate"
    assert persisted.default_harness == "codex"
    assert persisted.harness_config["workspace"] == str(tmp_path.resolve())

    updated = runtime.update_agent(
        agent_id,
        identity,
        name="Updated MCP Agent",
        application_mode="workflow",
        max_steps=42,
        memory_ids=["mcp-memory", "second-memory"],
        semantic_cache=False,
        continual_learning=True,
        skill_retrieval_top_k=4,
        is_favorite=True,
        meta_harness_mode="runtime",
        default_harness="native",
    )
    assert updated["updated"] is True
    assert updated["agent"]["name"] == "Updated MCP Agent"
    assert updated["agent"]["max_steps"] == 42
    assert updated["agent"]["continual_learning"] is True
    updated_persisted = provider.retrieve_memagent(agent_id)
    assert updated_persisted.memory_ids == ["mcp-memory", "second-memory"]
    assert updated_persisted.is_favorite is True
    assert updated_persisted.meta_harness_mode == "runtime"
    assert updated_persisted.default_harness == "native"
    assert "workflow_memory" in updated_persisted.memory_types
    assert "skillbox" in updated_persisted.memory_types

    # Recalculating mode defaults must not drop an already-enabled learning
    # agent's required workflow/skill stores.
    runtime.update_agent(agent_id, identity, application_mode="assistant")
    mode_updated = provider.retrieve_memagent(agent_id)
    assert "workflow_memory" in mode_updated.memory_types
    assert "skillbox" in mode_updated.memory_types

    inspected = runtime.inspect_agent(
        agent_id,
        identity,
        memory_id="mcp-memory",
    )
    assert inspected["ok"] is True
    assert inspected["capabilities"]["agent_id"] == agent_id
    assert inspected["semantic_cache"]["enabled"] is False
    assert inspected["observability"]["memory_id"] == "mcp-memory"
    compiled = runtime.compile_agent_memory(
        agent_id,
        identity,
        memory_id="mcp-memory",
    )
    assert compiled["ok"] is True
    assert "compile_report" in compiled

    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-only")
    configured = runtime.create_agent(
        "MCP Configured Agent",
        identity,
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    assert configured["agent"]["llm_provider"] == "openai"
    assert configured["agent"]["llm_model"] == "gpt-4o-mini"
    configured_persisted = provider.retrieve_memagent(configured["agent"]["agent_id"])
    assert configured_persisted.llm_config["provider"] == "openai"
    assert "api_key" not in configured_persisted.llm_config

    deletion = runtime.delete_agent(agent_id, identity, cascade=False)
    assert deletion["status"] == "approval_required"
    deletion_id = deletion["proposal"]["proposal_id"]
    runtime.approve_proposal(deletion_id, approver_id="operator@example.com")
    deleted = runtime.resume_proposal(deletion_id)
    assert deleted["deleted"] is True
    assert provider.retrieve_memagent(agent_id) is None

    with pytest.raises(MemorizzServerError, match=WRITE_SCOPE):
        runtime.create_agent("Denied", _identity("reader", READ_SCOPE))

    remote = MemorizzRuntime(
        MemorizzMCPServerConfig(
            transport="streamable-http",
            allow_anonymous_http=True,
            allow_writes=True,
        ),
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "remote-approvals.sqlite3"),
    )
    with pytest.raises(MemorizzServerError, match="local stdio"):
        remote.create_agent("Remote denied", identity)
    with pytest.raises(MemorizzServerError, match="local stdio"):
        remote.update_agent(configured["agent"]["agent_id"], identity, name="Denied")
    with pytest.raises(MemorizzServerError, match="local stdio"):
        remote.delete_agent(configured["agent"]["agent_id"], identity)


@pytest.mark.unit
def test_runtime_operational_parity_calls_are_scoped(tmp_path):
    class OperationalAgent(_FakeAgent):
        model = object()

        def capability_report(self, *, preflight=False):
            return {"agent": {"agent_id": self.agent_id, "preflight": preflight}}

        def semantic_cache_stats(self):
            return {"enabled": True, "hits": 2, "misses": 1}

        def learning_report(self, **scope):
            return {"enabled": True, "scope": scope}

        def observability_summary(self, memory_id, user_id, **kwargs):
            return {"memory_id": memory_id, "user_id": user_id, **kwargs}

        def compile_memory(self, **scope):
            return {"compiled": 2, "scope": scope}

        def generate_summaries(self, **scope):
            self.summary_scope = scope
            return ["summary-1"]

    agent = OperationalAgent()
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(
            allow_writes=True,
            allow_agent_execution=True,
            exposed_agent_ids={agent.agent_id},
        ),
        provider=_FakeAgentProvider(agent),
        approval_store=SQLiteApprovalStore(tmp_path / "operational.sqlite3"),
    )
    runtime._agents[agent.agent_id] = agent
    alice = _identity("alice")

    inspected = runtime.inspect_agent(
        agent.agent_id,
        alice,
        memory_id="memory-a",
        thread_id="thread-a",
    )
    assert inspected["semantic_cache"]["hits"] == 2
    assert inspected["observability"]["user_id"] == "alice"
    assert inspected["learning_control_plane"]["scope"]["user_id"] == "alice"

    compiled = runtime.compile_agent_memory(
        agent.agent_id,
        alice,
        memory_id="memory-a",
        thread_id="thread-a",
    )
    assert compiled["compile_report"]["scope"]["user_id"] == "alice"

    compacted = runtime.compact_conversation(
        agent.agent_id,
        "memory-a",
        alice,
        thread_id="thread-a",
    )
    assert compacted["summary_ids"] == ["summary-1"]
    assert agent.summary_scope["user_id"] == "alice"


@pytest.mark.unit
def test_personalization_preview_is_explicitly_tenant_scoped_and_content_safe(
    tmp_path,
):
    class PersonalizationAgent(_FakeAgent):
        memory_ids = ["memory-a"]

        def build_personalization_context(self, query, **kwargs):
            self.personalization_call = {"query": query, **kwargs}
            return build_personalization_context(
                entity_profiles=[
                    {
                        "entity_id": "private-entity-id",
                        "attributes": {"role": "AI Memory Engineer"},
                    }
                ],
                preferences=kwargs.get("preferences"),
                conversation_memories=[
                    {
                        "id": "private-conversation-id",
                        "text": "Memory-first observability",
                        "score": 0.91,
                    }
                ],
                policy=kwargs.get("policy"),
            )

    agent = PersonalizationAgent()
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(exposed_agent_ids={agent.agent_id}),
        provider=_FakeAgentProvider(agent),
        approval_store=SQLiteApprovalStore(tmp_path / "personalization.sqlite3"),
    )
    runtime._agents[agent.agent_id] = agent

    result = runtime.preview_personalization(
        agent.agent_id,
        "Draft a concise post",
        _identity("alice", READ_SCOPE),
        include_conversation_recall=True,
        exclude_thread_id="current-thread",
        min_relevance_score=0.8,
        preferences={"preferred_tone": "concise"},
        include_content=False,
    )

    assert agent.personalization_call["memory_id"] == "memory-a"
    assert agent.personalization_call["user_id"] == "alice"
    assert agent.personalization_call["exclude_thread_id"] == "current-thread"
    assert agent.personalization_call["policy"]["conversation_recall"] is True
    assert result["context_evidence"]["source_counts"]["entity_attributes"] == 1
    assert "context" not in result
    assert "prompt_block" not in result
    serialized = json.dumps(result)
    assert "AI Memory Engineer" not in serialized
    assert "private-entity-id" not in serialized
    assert "private-conversation-id" not in serialized

    with pytest.raises(MemorizzServerError, match="not attached"):
        runtime.preview_personalization(
            agent.agent_id,
            "Draft a concise post",
            _identity("alice", READ_SCOPE),
            memory_id="memory-from-another-agent",
        )


@pytest.mark.unit
def test_mcp_runtime_harness_surface_is_tenant_scoped_and_approval_bound(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    approvals = SQLiteApprovalStore(tmp_path / "harness-approvals.sqlite3")
    meta = MetaHarness(
        adapters=[_MCPFakeHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "harness-runs.sqlite3"),
        approval_store=approvals,
        allowed_workspace_roots=[str(tmp_path)],
    )
    config = MemorizzMCPServerConfig(
        allow_writes=True,
        allow_agent_execution=True,
        allow_harness_execution=True,
        harness_workspace_roots={str(tmp_path)},
    )
    runtime = MemorizzRuntime(
        config,
        provider=_filesystem_provider(tmp_path),
        approval_store=approvals,
        meta_harness=meta,
    )
    alice = _identity("alice")
    bob = _identity("bob")
    try:
        available = runtime.list_harnesses(alice)
        assert available["harnesses"][0]["name"] == "fake"

        proposed = runtime.start_harness_run(
            "Implement the accepted change",
            str(workspace),
            alice,
            harness="fake",
            write=True,
            mcp_access="none",
            verification_command="test -f mcp-result.txt",
        )
        assert proposed["status"] == "approval_required"
        proposal_id = proposed["proposal"]["proposal_id"]
        run_id = proposed["run"]["run_id"]
        assert proposed["proposal"]["arguments"]["user_id"] == "alice"
        assert (
            runtime.list_approvals(principal="alice")[0]["proposal_id"] == proposal_id
        )

        runtime.approve_proposal(proposal_id, approver_id="operator@example.com")
        result = runtime.resume_proposal(proposal_id)
        assert result["status"] == HarnessStatus.SUCCEEDED.value
        assert result["verified"] is True
        assert runtime.get_harness_run(run_id, alice)["run"]["status"] == "succeeded"
        assert runtime.get_harness_events(run_id, alice)["count"] > 0
        with pytest.raises(MemorizzServerError, match="not found"):
            runtime.get_harness_run(run_id, bob)
    finally:
        meta.close()


@pytest.mark.unit
def test_mcp_harness_surface_returns_structured_authentication_guidance(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _filesystem_provider(tmp_path)
    meta = MetaHarness(
        adapters=[_MCPAuthRequiredHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "harness-runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "harness-approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(
            allow_harness_execution=True,
            harness_workspace_roots={str(tmp_path)},
        ),
        provider=provider,
        meta_harness=meta,
    )
    alice = _identity("alice")
    try:
        available = runtime.list_harnesses(alice)
        assert available["ready_count"] == 0
        assert available["authentication_required"] == ["mcp-auth-required"]
        assert available["harnesses"][0]["error_code"] == "authentication_required"

        started = runtime.start_harness_run(
            "Inspect the workspace",
            str(workspace),
            alice,
            harness="mcp-auth-required",
            mcp_access="none",
        )
        assert started["ok"] is False
        assert started["status"] == "failed"
        assert started["error"]["code"] == "authentication_required"
        assert "TEST_HARNESS_API_KEY" in started["error"]["remediation"]
        assert started["run"]["status"] == "failed"
    finally:
        meta.close()
        provider.close()


@pytest.mark.unit
def test_official_server_registers_complete_surface(tmp_path):
    config = MemorizzMCPServerConfig()
    runtime = MemorizzRuntime(
        config,
        provider=_filesystem_provider(tmp_path),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    server = create_memorizz_mcp_server(config, runtime=runtime)
    assert server is not None


@pytest.mark.unit
def test_cli_serve_builds_secure_remote_configuration(monkeypatch):
    import memorizz.mcp_server as mcp_server

    captured = {}
    monkeypatch.setattr(
        mcp_server,
        "run_memorizz_mcp_server",
        lambda config: captured.setdefault("config", config),
    )
    result = CliRunner().invoke(
        app,
        [
            "mcp",
            "serve",
            "--transport",
            "streamable-http",
            "--public-url",
            "https://memory.example.com",
            "--agent-id",
            "agent-1",
            "--allow-writes",
        ],
        env={
            "MEMORIZZ_MCP_SERVER_API_KEYS": json.dumps(
                {"alice": "alice-token-that-is-long-enough"}
            )
        },
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.auth_required is True
    assert config.exposed_agent_ids == {"agent-1"}
    assert config.allow_writes is True
    assert config.allow_agent_execution is False


@pytest.mark.unit
def test_memorizz_server_works_over_real_stdio_protocol(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    manager = MCPClientManager(
        owner_id="integration",
        servers=[
            {
                "name": "memorizz-server",
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "memorizz.mcp_server"],
                "env": {
                    "MEMORIZZ_HOME": str(tmp_path),
                    "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
                    "OPENAI_API_KEY": "",
                    "ANTHROPIC_API_KEY": "",
                    "AZURE_OPENAI_API_KEY": "",
                    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
                },
                "timeout": 20,
            }
        ],
    )

    tools = manager.list_tools("memorizz-server")
    names = {tool["name"] for tool in tools["tools"]}
    assert tools["ok"] is True
    assert len(names) == 24
    assert all(
        tool["inputSchema"].get("additionalProperties") is False
        for tool in tools["tools"]
    )
    assert {
        "memorizz_create_agent",
        "memorizz_update_agent",
        "memorizz_delete_agent",
        "memorizz_inspect_agent",
        "memorizz_preview_personalization",
        "memorizz_compile_memory",
        "memorizz_compact_conversation",
        "memorizz_execute_agent",
        "memorizz_store_memory",
        "memorizz_forget_memory",
        "memorizz_list_harnesses",
        "memorizz_start_harness_run",
        "memorizz_get_harness_run",
        "memorizz_list_harness_runs",
        "memorizz_get_harness_events",
        "memorizz_cancel_harness_run",
    }.issubset(names)
    strict_rejection = manager.call_tool(
        "memorizz-server", "memorizz_server_info", {"unknown_argument": True}
    )
    assert strict_rejection["ok"] is False
    assert (
        manager.list_resources("memorizz-server")["resources"][0]["uri"]
        == "memorizz://server"
    )
    assert (
        manager.list_prompts("memorizz-server")["prompts"][0]["name"]
        == "memorizz_memory_assistant"
    )

    denied = manager.call_tool(
        "memorizz-server", "memorizz_store_memory", {"content": "stored over MCP"}
    )
    assert denied["ok"] is False
    assert denied["error_code"] == "approval_required"
    proposal_id = denied["proposal"]["proposal_id"]
    manager.approve_tool_call(proposal_id, approver_id="operator@example.com")
    stored = manager.resume_tool_call(proposal_id)
    assert _structured(stored)["ok"] is True
    listed = manager.call_tool("memorizz-server", "memorizz_list_memories", {})
    assert _structured(listed)["memories"][0]["content"] == "stored over MCP"

    create_proposal = manager.call_tool(
        "memorizz-server",
        "memorizz_create_agent",
        {
            "name": "Protocol Agent",
            "instruction": "Created through the real MCP protocol.",
            "semantic_cache": True,
        },
    )
    assert create_proposal["ok"] is False
    assert create_proposal["error_code"] == "approval_required"
    manager.approve_tool_call(
        create_proposal["proposal"]["proposal_id"],
        approver_id="operator@example.com",
    )
    created_agent = manager.resume_tool_call(create_proposal["proposal"]["proposal_id"])
    created_payload = _structured(created_agent)
    assert created_payload["created"] is True
    assert created_payload["agent"]["name"] == "Protocol Agent"

    update_proposal = manager.call_tool(
        "memorizz-server",
        "memorizz_update_agent",
        {
            "agent_id": created_payload["agent"]["agent_id"],
            "name": "Protocol Agent Updated",
            "max_steps": 33,
        },
    )
    assert update_proposal["error_code"] == "approval_required"
    manager.approve_tool_call(
        update_proposal["proposal"]["proposal_id"],
        approver_id="operator@example.com",
    )
    updated_agent = _structured(
        manager.resume_tool_call(update_proposal["proposal"]["proposal_id"])
    )
    assert updated_agent["agent"]["name"] == "Protocol Agent Updated"
    assert updated_agent["agent"]["max_steps"] == 33
    inspected_agent = manager.call_tool(
        "memorizz-server",
        "memorizz_inspect_agent",
        {"agent_id": created_payload["agent"]["agent_id"]},
    )
    assert _structured(inspected_agent)["capabilities"]["agent_id"] == (
        created_payload["agent"]["agent_id"]
    )
    listed_agents = manager.call_tool("memorizz-server", "memorizz_list_agents", {})
    assert _structured(listed_agents)["agents"][0]["agent_id"] == (
        created_payload["agent"]["agent_id"]
    )


@pytest.mark.unit
def test_authenticated_http_protocol_isolates_principals(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("The execution sandbox does not allow loopback sockets")
        port = listener.getsockname()[1]

    alice_token = "alice-http-token-that-is-long-enough"
    bob_token = "bob-http-token-that-is-long-enough"
    environment = dict(os.environ)
    environment.update(
        {
            "MEMORIZZ_HOME": str(tmp_path),
            "MEMORIZZ_MCP_SERVER_API_KEYS": json.dumps(
                {"alice": alice_token, "bob": bob_token}
            ),
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "AZURE_OPENAI_API_KEY": "",
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "memorizz.cli",
            "mcp",
            "serve",
            "--transport",
            "streamable-http",
            "--port",
            str(port),
            "--public-url",
            f"http://127.0.0.1:{port}",
            "--allow-writes",
            "--no-agent-execution",
        ],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("Authenticated Memorizz MCP server exited at startup")
                time.sleep(0.05)
        else:
            pytest.fail("Authenticated Memorizz MCP server did not start")

        def remote_manager(owner: str, token: str) -> MCPClientManager:
            return MCPClientManager(
                owner_id=owner,
                servers=[
                    {
                        "name": "remote",
                        "transport": "streamable_http",
                        "url": f"http://127.0.0.1:{port}/mcp",
                        "allow_private_network": True,
                        "auth": {"type": "bearer", "token": token},
                        "timeout": 10,
                    }
                ],
            )

        alice = remote_manager("alice-client", alice_token)
        bob = remote_manager("bob-client", bob_token)
        proposed = alice.call_tool(
            "remote",
            "memorizz_store_memory",
            {"content": "Alice-only memory"},
        )
        proposal_id = proposed["proposal"]["proposal_id"]
        alice.approve_tool_call(proposal_id, approver_id="operator@example.com")
        stored = alice.resume_tool_call(proposal_id)
        assert _structured(stored)["ok"] is True
        assert (
            _structured(alice.call_tool("remote", "memorizz_list_memories", {}))[
                "count"
            ]
            == 1
        )
        assert (
            _structured(bob.call_tool("remote", "memorizz_list_memories", {}))["count"]
            == 0
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
