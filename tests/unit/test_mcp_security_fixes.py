"""Regressions for the MCP security review: connected-tool classification,
update_memory fidelity, tenant-scoped agent exports, harness plan and listing
scoping, streamed-execution attribution and turn-capture agent checks."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memorizz import MemoryHistory, MemoryType
from memorizz.approval import SQLiteApprovalStore
from memorizz.mcp import MCPClientManager
from memorizz.mcp.security import tool_is_mutating
from memorizz.mcp_server import (
    MemorizzMCPServerConfig,
    MemorizzRuntime,
    create_memorizz_mcp_server,
)
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.mcp_server.streaming import execute_stream
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from tests.unit.test_mcp_manager import SERVER
from tests.unit.test_mcp_memory_tools import (  # noqa: F401 (fixture)
    _NoEmbedder,
    _WordEmbedder,
    use_embedder,
)

pytestmark = pytest.mark.unit


def _identity(principal="alice", *scopes):
    return RequestIdentity(
        principal=principal,
        scopes=frozenset(scopes or ALL_SCOPES),
        authenticated=True,
    )


def _provider(tmp_path):
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )


def _runtime(tmp_path, *, provider=None, meta_harness=None, **config):
    config.setdefault("allow_writes", True)
    return MemorizzRuntime(
        MemorizzMCPServerConfig(**config),
        provider=provider or _provider(tmp_path),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        meta_harness=meta_harness,
    )


# ------------------------------------------------ 1. connected-tool policy

UNANNOTATED_MUTATIONS = [
    "trash_message",
    "reply",
    "forward",
    "reschedule_booking",
    "confirm_booking",
    "merge_contacts",
    "purchase_domain",
]


@pytest.mark.parametrize("name", UNANNOTATED_MUTATIONS)
def test_unannotated_tools_change_data_unless_clearly_read_only(name):
    assert tool_is_mutating(name) is True
    assert tool_is_mutating(name, {"name": name, "annotations": {}}) is True


@pytest.mark.parametrize(
    "name",
    ["get_message", "list_events", "search-threads", "describe.table", "Lookup_order"],
)
def test_a_read_verb_as_the_first_segment_is_read_only(name):
    assert tool_is_mutating(name) is False


def test_classification_precedence_and_conservative_defaults():
    # Server annotations win over the name.
    assert tool_is_mutating("reply", {"annotations": {"readOnlyHint": True}}) is False
    assert tool_is_mutating("get_message", {"annotations": {"readOnlyHint": False}})
    assert tool_is_mutating("get_message", {"annotations": {"destructiveHint": True}})
    # Host policy wins over everything.
    assert tool_is_mutating("reply", read_only_tools=["reply"]) is False
    assert tool_is_mutating("get_message", mutation_tools=["get_message"]) is True
    # A read verb anywhere but first, a vendor prefix, or export: not read-only.
    # A vendor prefix must not hide the read verb; the host allowlist and
    # annotations remain the authoritative controls.
    assert tool_is_mutating("notion-search") is False
    assert tool_is_mutating("export_contacts") is True
    assert tool_is_mutating("send_get_request") is True
    assert tool_is_mutating("") is True


def test_manager_requires_approval_for_unannotated_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(
        owner_id="agent-1",
        servers=[
            {
                "name": "mail",
                "transport": "stdio",
                "command": "python3",
                "args": [str(SERVER)],
                "read_only_tools": ["forward_preview"],
            }
        ],
    )
    manager._tool_metadata["mail"] = {
        name: {"name": name}
        for name in ("reply", "trash_message", "get_message", "forward_preview")
    }
    calls = []
    monkeypatch.setattr(
        manager,
        "_call_tool_authorized",
        lambda server, tool, arguments: calls.append(tool) or {"ok": True},
    )

    assert manager.call_tool("mail", "reply", {"id": "1"})["error_code"] == (
        "approval_required"
    )
    assert manager.call_tool("mail", "trash_message", {"id": "1"})["error_code"] == (
        "approval_required"
    )
    assert manager.call_tool("mail", "get_message", {"id": "1"})["ok"] is True
    assert manager.call_tool("mail", "forward_preview", {"id": "1"})["ok"] is True
    assert calls == ["get_message", "forward_preview"]


def test_read_path_refuses_unannotated_tools_on_read_only_servers(
    tmp_path, monkeypatch
):
    from memorizz.memagent.models import MemAgentModel

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _provider(tmp_path)
    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-1",
            instruction="x",
            mcp_servers=[
                {
                    "name": "mail",
                    "transport": "stdio",
                    "command": "python3",
                    "args": [str(SERVER)],
                }
            ],
        )
    )
    monkeypatch.setattr(
        MCPClientManager,
        "cached_tools",
        lambda self, server_name: [{"name": "reply"}, {"name": "get_message"}],
    )
    executed = []
    monkeypatch.setattr(
        MCPClientManager,
        "call_tool",
        lambda self, server_name, tool_name, arguments=None: executed.append(tool_name)
        or {"ok": True},
    )
    runtime = _runtime(
        tmp_path, provider=provider, allow_writes=False, exposed_agent_ids={"agent-1"}
    )
    alice = _identity("alice")

    listed = runtime.list_connected_tools("agent-1", alice)
    tools = {tool["name"]: tool for tool in listed["servers"][0]["tools"]}
    assert tools["reply"]["changes_data"] is True
    assert tools["get_message"]["changes_data"] is False
    with pytest.raises(MemorizzServerError, match="changes data"):
        runtime.call_connected_tool(
            "agent-1", "mail", "reply", {"id": "1"}, alice, read_only=True
        )
    with pytest.raises(MemorizzServerError, match="read-only"):
        runtime.call_connected_tool("agent-1", "mail", "reply", {"id": "1"}, alice)
    assert runtime.call_connected_tool(
        "agent-1", "mail", "get_message", {"id": "1"}, alice, read_only=True
    )["ok"]
    assert executed == ["get_message"]


# ------------------------------------------------- 2. update_memory fidelity


def test_update_memory_builds_the_replacement_from_the_raw_record(
    tmp_path, use_embedder
):
    use_embedder(_NoEmbedder())
    runtime = _runtime(tmp_path)
    alice = _identity()
    first = runtime.remember(
        "Deploy key rotates monthly",
        "knowledge_base",
        alice,
        memory_id="proj",
        metadata={"token": "keep", "api_key": "also-keep", "source": "chat"},
    )

    updated = runtime.update_memory(
        first["record_id"], "Deploy key rotates weekly", alice
    )

    raw = runtime.provider.retrieve_by_id(
        updated["record_id"], MemoryType.KNOWLEDGE_BASE
    )
    assert raw["token"] == "keep"
    assert raw["api_key"] == "also-keep"
    assert raw["source"] == "chat"
    assert raw["content"] == "Deploy key rotates weekly"
    # The MCP view of the record is still redacted.
    shown = runtime.get_memory(updated["record_id"], "knowledge_base", alice)
    assert shown["memory"]["token"] == "***"
    # Another tenant still cannot update it.
    with pytest.raises(MemorizzServerError) as error:
        runtime.update_memory(updated["record_id"], "stolen", _identity("bob"))
    assert error.value.code == "memory_not_found"


# ------------------------------------------------ 3. agent-scoped export


def test_agent_scoped_export_works_for_tenant_principals(tmp_path, use_embedder):
    from memorizz.memagent.models import MemAgentModel

    use_embedder(_NoEmbedder())
    provider = _provider(tmp_path)
    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-1",
            name="Scoped",
            memory_ids=["proj"],
            llm_config={"provider": "ollama", "model": "test"},
        )
    )
    runtime = _runtime(tmp_path, provider=provider, exposed_agent_ids={"agent-1"})
    alice, bob = _identity("alice"), _identity("bob")
    runtime.remember("Alice's note", "knowledge_base", alice, memory_id="proj")
    runtime.remember("Bob's note", "knowledge_base", bob, memory_id="elsewhere")

    archive = runtime.export_memories(alice, agent_id="agent-1")["archive"]

    assert archive["manifest"]["scope"]["agent_id"] == "agent-1"
    assert archive["manifest"]["scope"]["user_id"] == "alice"
    assert archive["manifest"]["counts"]["knowledge_base"] == 1
    assert archive["manifest"]["counts"]["agents"] == 1
    assert "Bob's note" not in json.dumps(archive)
    # Bob has nothing in the agent's namespaces, so the agent is out of scope.
    with pytest.raises(MemorizzServerError, match="authorized scope"):
        runtime.export_memories(bob, agent_id="agent-1")


def test_archive_scopes_agents_by_tenant_namespaces_not_a_user_field(tmp_path):
    from memorizz import MemoryArchive, MemoryArchiveError
    from memorizz.memagent.models import MemAgentModel

    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "archive", use_faiss=False)
    )
    store.store_memagent(MemAgentModel(agent_id="a", name="A", memory_ids=["team"]))
    store.store_memagent(MemAgentModel(agent_id="b", name="B", memory_ids=["other"]))
    store.store(
        {"content": "Alice fact", "memory_id": "team", "user_id": "alice"},
        MemoryType.KNOWLEDGE_BASE,
    )
    archive = MemoryArchive(store).export(agent_id="a", user_id="alice")
    assert archive["manifest"]["counts"]["knowledge_base"] == 1
    assert [r["id"] for r in archive["stores"]["agents"]] == ["a"]
    with pytest.raises(MemoryArchiveError, match="authorized scope"):
        MemoryArchive(store).export(agent_id="b", user_id="alice")
    # An unscoped (SDK administrator) export still sees every agent.
    assert len(MemoryArchive(store).export()["stores"]["agents"]) == 2


# ------------------------------------------- 4. plan stages check agents


def test_plan_stage_agents_must_be_exposed(tmp_path):
    from tests.unit.test_harness_parity import _meta, _workspace

    meta = _meta(tmp_path, "alpha")
    runtime = _runtime(
        tmp_path,
        meta_harness=meta,
        allow_agent_execution=True,
        allow_harness_execution=True,
        harness_workspace_roots={str(tmp_path)},
        exposed_agent_ids={"agent-1"},
    )
    alice = _identity("alice")
    try:
        with pytest.raises(MemorizzServerError, match="Agent was not found"):
            runtime.start_harness_plan(
                "Goal",
                [
                    {"name": "review", "harness": "alpha"},
                    {"name": "fix", "harness": "alpha", "agent_id": "agent-2"},
                ],
                str(_workspace(tmp_path)),
                alice,
            )
        assert runtime.list_harness_workflows(alice)["count"] == 0
    finally:
        meta.close()


# ------------------------------------ 5. listings filter before the limit


def _run_row(owner, index, kind="run"):
    key = "run_id" if kind == "run" else "orchestration_id"
    return {key: f"{owner}-{index}", "status": "succeeded", "task": {"user_id": owner}}


class _FakeMeta:
    def __init__(self, runs, workflows):
        self.runs, self.workflows, self.calls = runs, workflows, []

    def list_runs(self, *, limit=100, status=None):
        self.calls.append(("runs", limit))
        rows = [r for r in self.runs if status is None or r["status"] == status]
        return rows[:limit]

    def list_orchestrations(self, *, limit=50, status=None):
        self.calls.append(("workflows", limit))
        rows = [r for r in self.workflows if status is None or r["status"] == status]
        return rows[:limit]


class _TenantAwareMeta(_FakeMeta):
    def list_runs(self, *, limit=100, status=None, user_id=None):
        self.calls.append(("runs", limit, user_id))
        rows = [r for r in self.runs if r["task"]["user_id"] == user_id]
        return rows[:limit]


def test_harness_listings_apply_the_tenant_filter_before_the_limit(tmp_path):
    runs = [_run_row("alice", i) for i in range(60)] + [
        _run_row("bob", i) for i in range(3)
    ]
    workflows = [_run_row("alice", i, "wf") for i in range(30)] + [
        _run_row("bob", i, "wf") for i in range(2)
    ]
    meta = _FakeMeta(runs, workflows)
    runtime = _runtime(tmp_path, meta_harness=meta)
    alice, bob, carol = _identity("alice"), _identity("bob"), _identity("carol")

    listed = runtime.list_harness_runs(bob, limit=50)
    assert [row["run_id"] for row in listed["runs"]] == ["bob-0", "bob-1", "bob-2"]
    assert runtime.list_harness_runs(alice, limit=50)["count"] == 50
    assert runtime.list_harness_runs(carol, limit=50)["runs"] == []
    assert [
        row["orchestration_id"]
        for row in runtime.list_harness_workflows(bob, limit=20)["workflows"]
    ] == ["bob-0", "bob-1"]
    assert runtime.list_harness_workflows(alice, limit=20)["count"] == 20
    # Widening is bounded: a tenant with nothing never scans without limit.
    assert all(limit <= 10_000 for _, limit in meta.calls)
    assert len(meta.calls) < 20


def test_harness_listing_pushes_the_owner_into_a_capable_store(tmp_path):
    meta = _TenantAwareMeta([_run_row("bob", 0)], [])
    runtime = _runtime(tmp_path, meta_harness=meta)
    assert runtime.list_harness_runs(_identity("bob"), limit=5)["count"] == 1
    assert meta.calls == [("runs", 5, "bob")]


# --------------------------------- 6. streamed execution is attributed


class _Stream:
    def __init__(self):
        self.result = {"ok": True, "status": "completed"}
        self.cancellation = SimpleNamespace(cancel=lambda: None)
        self.closed = False

    async def async_events(self):
        return
        yield  # pragma: no cover - makes this an async generator

    def close(self):
        self.closed = True


def _ctx():
    return SimpleNamespace(
        request_context=SimpleNamespace(meta={}, request_id="req-1"),
        report_progress=AsyncMock(),
        session=SimpleNamespace(),
    )


def test_streamed_execution_attributes_memory_writes_to_the_principal(
    tmp_path, monkeypatch
):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    config = MemorizzMCPServerConfig(allow_writes=True, allow_agent_execution=True)
    runtime = MemorizzRuntime(
        config,
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    journaled = runtime.provider  # installs history recording

    def fake_events(message, identity, **kwargs):
        journaled.store(
            {
                "content": message,
                "role": "user",
                "user_id": identity.principal,
                "memory_id": "m1",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
        return _Stream()

    runtime.execute_agent_events = fake_events

    # The streaming helper itself.
    result = asyncio.run(
        execute_stream(
            runtime, "first", _identity("alice"), _ctx(), event_format="progress"
        )
    )
    assert result["ok"] is True
    # The server's default execute path (event_format="progress").
    monkeypatch.setenv("MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL", "alice")
    server = create_memorizz_mcp_server(config, runtime=runtime)
    tool = server._tool_manager._tools["memorizz_execute_agent"]
    assert asyncio.run(tool.fn(message="second", ctx=_ctx()))["ok"] is True

    events = MemoryHistory(provider).timeline(memory_id="m1")["events"]
    assert len(events) == 2
    assert {event["actor"] for event in events} == {"alice"}
    assert all(str(event.get("source") or "").startswith("mcp:") for event in events)


# -------------------------------------- 7. turn capture checks the agent


def test_turn_capture_and_summaries_check_the_agent_allowlist(
    tmp_path, use_embedder, monkeypatch
):
    use_embedder(_WordEmbedder())
    runtime = _runtime(tmp_path, exposed_agent_ids={"agent-1"})
    alice = _identity("alice")

    with pytest.raises(MemorizzServerError, match="Agent was not found"):
        runtime.record_turn("proj", "t1", "x", "y", alice, agent_id="agent-2")
    assert (
        runtime.list_memories("conversation_memory", alice, memory_id="proj")[
            "memories"
        ]
        == []
    )
    monkeypatch.setattr("memorizz.episodic_capture.default_model", lambda: None)
    with pytest.raises(MemorizzServerError, match="Agent was not found"):
        runtime.summarize_session("proj", "t1", alice, agent_id="agent-2")
    # The exposed agent, and no agent at all, still work.
    assert runtime.record_turn("proj", "t1", "x", "y", alice, agent_id="agent-1")["ok"]
    assert runtime.record_turn("proj", "t1", "x", "y", alice)["ok"]
    assert runtime.summarize_session("proj", "t1", alice)["reason"] == "no_model"


@pytest.mark.unit
def test_vendor_prefixes_do_not_hide_read_verbs_and_mutation_verbs_always_win():
    from memorizz.mcp.security import tool_is_mutating

    for name in (
        "notion-search",
        "memorizz_list_memories",
        "gmail.get_message",
        "cal_get_bookings",
    ):
        assert tool_is_mutating(name) is False, name
    for name in (
        "get_and_delete",
        "search_then_send",
        "notion-create-pages",
        "memorizz_forget_memory",
    ):
        assert tool_is_mutating(name) is True, name


@pytest.mark.unit
def test_memorizz_server_tool_names_classify_by_intent():
    import re

    from memorizz.mcp.security import tool_is_mutating

    source = open("src/memorizz/mcp_server/server.py", encoding="utf-8").read()
    names = sorted(set(re.findall(r"async def (memorizz_[a-z_]+)", source)))
    assert len(names) >= 40
    read_prefixes = (
        "memorizz_get_",
        "memorizz_list_",
        "memorizz_search_",
        "memorizz_inspect_",
        "memorizz_preview_",
        "memorizz_lookup_",
        "memorizz_query_",
        "memorizz_read_",
    )
    for name in names:
        if name.startswith(read_prefixes) or name in {
            "memorizz_memory_status",
            "memorizz_server_info",
        }:
            assert tool_is_mutating(name) is False, name
    for name in (
        "memorizz_store_memory",
        "memorizz_forget_memory",
        "memorizz_update_memory",
        "memorizz_delete_agent",
        "memorizz_execute_agent",
        "memorizz_import_memories",
        "memorizz_record_turn",
        "memorizz_summarize_session",
        "memorizz_call_connected_tool",
    ):
        assert tool_is_mutating(name) is True, name
