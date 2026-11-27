"""An MCP server started for one memory stays inside it."""

import asyncio
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES, MemorizzMCPServerConfig
from memorizz.mcp_server.memory_scope import scoped_call
from memorizz.mcp_server.runtime import MemorizzRuntime, MemorizzServerError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

WHO = RequestIdentity(
    principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
)


def _runtime(tmp_path: Path, memory_ids=None, **options) -> MemorizzRuntime:
    return MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=True, memory_ids=memory_ids, **options),
        provider=FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory")),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )


def _call(runtime, function, *args, **kwargs):
    return scoped_call(runtime, function, args, kwargs)


@pytest.fixture
def stores(tmp_path):
    """One store holding a fact in each of two memories."""
    open_runtime = _runtime(tmp_path)
    mine = open_runtime.remember(
        "The release codeword is PELICAN", "knowledge_base", WHO, memory_id="mine"
    )
    theirs = open_runtime.remember(
        "The other team's codeword is HERON", "knowledge_base", WHO, memory_id="theirs"
    )
    return _runtime(tmp_path, {"mine"}), mine["record_id"], theirs["record_id"]


@pytest.mark.unit
def test_an_open_server_is_unchanged(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.remember("A fact", "knowledge_base", WHO, memory_id="any")

    found = _call(runtime, runtime.list_memories, "knowledge_base", WHO)
    assert found["count"] == 1
    assert runtime.config.public_dict()["memory_ids"] == []


@pytest.mark.unit
def test_a_scoped_server_reads_only_its_memory(stores):
    runtime, mine, theirs = stores

    # With no memory named, its own is used.
    listed = _call(runtime, runtime.list_memories, "knowledge_base", WHO)
    assert [row["content"] for row in listed["memories"]] == [
        "The release codeword is PELICAN"
    ]
    found = _call(runtime, runtime.search_memories, "codeword", "knowledge_base", WHO)
    assert "HERON" not in str(found)

    # Naming another memory is refused, by argument and by record.
    with pytest.raises(MemorizzServerError, match="cannot reach 'theirs'"):
        _call(
            runtime, runtime.search_memories, "codeword", "knowledge_base", WHO,
            memory_id="theirs",
        )  # fmt: skip
    assert _call(runtime, runtime.get_memory, mine, "knowledge_base", WHO)["ok"]
    for operation in (runtime.get_memory, runtime.forget):
        with pytest.raises(MemorizzServerError, match="Memory was not found"):
            _call(runtime, operation, theirs, "knowledge_base", WHO)
    with pytest.raises(MemorizzServerError, match="Memory was not found"):
        _call(runtime, runtime.update_memory, theirs, "Changed", WHO)


@pytest.mark.unit
def test_a_scoped_server_writes_only_its_memory(stores):
    runtime, _, _ = stores

    kept = _call(runtime, runtime.remember, "A new fact", "knowledge_base", WHO)
    row = runtime.get_memory(kept["record_id"], "knowledge_base", WHO)["memory"]
    assert row["memory_id"] == "mine"
    with pytest.raises(MemorizzServerError, match="cannot reach"):
        _call(
            runtime, runtime.remember, "Elsewhere", "knowledge_base", WHO,
            memory_id="theirs",
        )  # fmt: skip
    with pytest.raises(MemorizzServerError, match="cannot reach"):
        _call(runtime, runtime.record_turn, "theirs", "t1", "hi", "hello", WHO)


@pytest.mark.unit
def test_what_could_read_across_memories_is_refused(stores):
    runtime, _, _ = stores

    for operation, args in (
        (runtime.export_memories, (WHO,)),
        (runtime.query_traces, ("agent-1", WHO)),
        (runtime.inspect_agent, ("agent-1", WHO)),
        (runtime.list_agents, (WHO,)),
    ):
        with pytest.raises(MemorizzServerError, match="not available here"):
            _call(runtime, operation, *args)
    # What reads no memory still works.
    assert _call(runtime, runtime.server_info, WHO)["ok"] is True
    assert _call(runtime, runtime.list_harnesses, WHO)["ok"] is True


@pytest.mark.unit
def test_a_connected_apps_search_and_fetch_stay_in_the_scoped_memories(stores):
    """``search`` and ``fetch`` are what ChatGPT calls; on a scoped server
    they look in its memories and nowhere else."""
    runtime, mine, theirs = stores

    found = _call(runtime, runtime.connector_search, "codeword", WHO)["results"]
    assert [row["id"] for row in found] == [f"knowledge_base:{mine}"]
    fetched = _call(runtime, runtime.connector_fetch, f"knowledge_base:{mine}", WHO)
    assert "PELICAN" in fetched["text"]
    # Another memory's record is the same answer as one that does not exist.
    with pytest.raises(MemorizzServerError, match="not found"):
        _call(runtime, runtime.connector_fetch, f"knowledge_base:{theirs}", WHO)
    with pytest.raises(MemorizzServerError, match="search returned"):
        _call(runtime, runtime.connector_fetch, "nonsense", WHO)


@pytest.mark.unit
def test_with_several_memories_one_must_be_named(tmp_path):
    runtime = _runtime(tmp_path, {"a", "b"})

    with pytest.raises(MemorizzServerError, match="Say which memory"):
        _call(runtime, runtime.list_memories, "knowledge_base", WHO)
    assert _call(
        runtime, runtime.list_memories, "knowledge_base", WHO, memory_id="b"
    )["count"] == 0  # fmt: skip


@pytest.mark.unit
def test_the_scope_is_read_from_the_environment_and_set_for_a_run(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MEMORIZZ_MCP_SERVER_MEMORY_IDS", "mine, also-mine ,")
    assert MemorizzMCPServerConfig.from_env().memory_ids == {"mine", "also-mine"}
    monkeypatch.delenv("MEMORIZZ_MCP_SERVER_MEMORY_IDS")
    assert MemorizzMCPServerConfig.from_env().memory_ids is None

    from memorizz.metaharness.models import HarnessPermissions

    assert HarnessPermissions().mcp_memory_scope == "run"
    with pytest.raises(ValueError, match="mcp_memory_scope"):
        HarnessPermissions(mcp_memory_scope="some")


@pytest.mark.unit
def test_every_tool_of_a_scoped_server_goes_through_the_scope(stores):
    """The MCP tools themselves, not only the helper: a tool call for another
    memory is refused, and one with no memory lands in the server's own."""
    from mcp.server.mcpserver.exceptions import ToolError

    from memorizz.mcp_server import server as module

    runtime, _, _ = stores

    async def run():
        ok = await module._tool_call(runtime.list_memories, "knowledge_base", WHO)
        with pytest.raises(ToolError, match="memory_out_of_scope"):
            await module._tool_call(
                runtime.list_memories, "knowledge_base", WHO, memory_id="theirs"
            )
        return ok

    token = module.current_identity
    try:
        module.current_identity = lambda: WHO
        assert asyncio.run(run())["count"] == 1
    finally:
        module.current_identity = token


@pytest.mark.unit
def test_a_read_only_memory_can_be_read_and_never_written(tmp_path):
    open_runtime = _runtime(tmp_path)
    open_runtime.remember("Our own note", "knowledge_base", WHO, memory_id="mine")
    shared = open_runtime.remember(
        "The team handbook says ship on Fridays",
        "knowledge_base",
        WHO,
        memory_id="team",
    )["record_id"]
    open_runtime.remember(
        "Private to others", "knowledge_base", WHO, memory_id="theirs"
    )
    runtime = _runtime(tmp_path, {"mine"}, read_memory_ids={"team"})

    read = _call(
        runtime, runtime.search_memories, "handbook", "knowledge_base", WHO,
        memory_id="team",
    )  # fmt: skip
    assert "Fridays" in str(read)
    assert _call(runtime, runtime.get_memory, shared, "knowledge_base", WHO)["ok"]
    # Unnamed, a read still means the server's own memory.
    own = _call(runtime, runtime.list_memories, "knowledge_base", WHO)
    assert [row["content"] for row in own["memories"]] == ["Our own note"]

    with pytest.raises(MemorizzServerError, match="read-only here"):
        _call(
            runtime, runtime.remember, "Sneak", "knowledge_base", WHO, memory_id="team"
        )
    for operation, args in (
        (runtime.forget, (shared, "knowledge_base", WHO)),
        (runtime.update_memory, (shared, "Changed", WHO)),
    ):
        with pytest.raises(MemorizzServerError, match="read-only here"):
            _call(runtime, operation, *args)
    with pytest.raises(MemorizzServerError, match="cannot reach 'theirs'"):
        _call(runtime, runtime.list_memories, "knowledge_base", WHO, memory_id="theirs")
    assert runtime.config.public_dict()["read_only_memory_ids"] == ["team"]
    with pytest.raises(ValueError, match="needs memory_ids"):
        MemorizzMCPServerConfig(read_memory_ids={"team"})


@pytest.mark.unit
def test_a_run_is_given_its_memory_or_a_private_one(tmp_path):
    from memorizz.metaharness import MetaHarness
    from memorizz.metaharness.models import HarnessPermissions, HarnessTask
    from memorizz.metaharness.service import private_memory_id

    meta = MetaHarness(
        memory_provider=FileSystemProvider(FileSystemConfig(root_path=tmp_path / "m"))
    )

    def server_env(**task):
        made = HarnessTask(task="x", workspace=str(tmp_path), harness="codex", **task)
        made.run_id = "run-1"
        folder = tmp_path / f"cfg-{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        meta._configure_mcp(made, folder)
        import json

        config = json.loads((folder / "mcp.json").read_text())
        return config["mcpServers"]["memorizz"]["env"]

    env = server_env(memory_id="billing")
    assert env["MEMORIZZ_MCP_SERVER_MEMORY_IDS"] == "billing"
    assert "MEMORIZZ_MCP_SERVER_READ_MEMORY_IDS" not in env
    # Given no memory, the run gets one of its own, not every memory.
    assert server_env()["MEMORIZZ_MCP_SERVER_MEMORY_IDS"] == private_memory_id("run-1")
    shared = server_env(
        memory_id="billing",
        permissions=HarnessPermissions(mcp_read_memory_ids="handbook, billing ,legal"),
    )
    assert shared["MEMORIZZ_MCP_SERVER_READ_MEMORY_IDS"] == "handbook,legal"
    opened = server_env(permissions=HarnessPermissions(mcp_memory_scope="all"))
    assert "MEMORIZZ_MCP_SERVER_MEMORY_IDS" not in opened
    with pytest.raises(ValueError, match="at most"):
        HarnessPermissions(mcp_read_memory_ids=[f"m{i}" for i in range(9)])


# Words the notes below hold, so a run is handed them with no embedding model
# too: the file store then matches by keyword (as on a CI runner).
DEPLOY_QUESTION = (
    "Which day do deploys happen, and what does the deploy handbook require?"
)


@pytest.mark.unit
def test_read_only_memories_join_a_runs_context(tmp_path):
    from memorizz.metaharness import MetaHarness
    from memorizz.metaharness.models import HarnessPermissions, HarnessTask

    store = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "m"))
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=True),
        provider=store,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    nobody = RequestIdentity(
        principal=None, scopes=frozenset(ALL_SCOPES), authenticated=False
    )
    runtime.remember(
        "Deploys happen on Tuesday", "knowledge_base", nobody, memory_id="mine"
    )
    runtime.remember(
        "The deploy handbook requires two reviewers",
        "knowledge_base",
        nobody,
        memory_id="team",
    )
    runtime.remember(
        "A deploy secret of another team", "knowledge_base", nobody, memory_id="theirs"
    )
    meta = MetaHarness(memory_provider=store)

    def pack(**permissions):
        task = HarnessTask(
            task=DEPLOY_QUESTION, workspace=str(tmp_path), harness="codex",
            memory_id="mine", permissions=HarnessPermissions(**permissions),
        )  # fmt: skip
        return meta._build_context_pack(task, task.task, strategy="retrieval")

    alone = pack()
    assert "Tuesday" in alone.rendered and "reviewers" not in alone.rendered
    both = pack(mcp_read_memory_ids=["team"])
    assert "Tuesday" in both.rendered and "two reviewers" in both.rendered
    assert "From the read-only memory 'team'" in both.rendered
    assert "another team" not in both.rendered
    assert both.metadata["read_memory_ids"] == ["team"]
    assert len(both.records) == len(alone.records) + 1

    # A run given no memory starts with nothing from anyone else's.
    def no_memory(**permissions):
        task = HarnessTask(
            task=DEPLOY_QUESTION, workspace=str(tmp_path), harness="codex",
            permissions=HarnessPermissions(**permissions),
        )  # fmt: skip
        return meta._build_context_pack(task, task.task, strategy="retrieval")

    assert no_memory().records == [] and no_memory().rendered == ""
    assert "two reviewers" in no_memory(mcp_read_memory_ids=["team"]).rendered
    assert no_memory(mcp_memory_scope="all").records  # opened on purpose


@pytest.mark.unit
def test_a_failed_runs_error_is_not_saved_to_shared_memory(tmp_path):
    from memorizz.enums.memory_type import MemoryType
    from memorizz.metaharness import MetaHarness
    from memorizz.metaharness.models import HarnessTask

    store = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "m"))
    meta = MetaHarness(memory_provider=store)
    task = HarnessTask(
        task="x", workspace=str(tmp_path), harness="codex", memory_id="notes"
    )

    def remember(**handoff):
        return meta._remember_handoff(
            task,
            {"harness": "codex", "run_id": "r1", "response": "text", **handoff},
            workflow={},
            thread_id="t",
        )

    assert remember(status="failed") is None
    assert remember(status="canceled") is None
    assert store.list_all(MemoryType.CONVERSATION_MEMORY) == []
    # A run that failed after changing files still says what it did.
    assert remember(status="failed", files_changed=["a.py"])
    assert remember(status="succeeded")
    assert len(store.list_all(MemoryType.CONVERSATION_MEMORY)) == 2
