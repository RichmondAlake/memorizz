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
    assert len(names) == 11
    assert {
        "memorizz_execute_agent",
        "memorizz_store_memory",
        "memorizz_forget_memory",
    }.issubset(names)
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
