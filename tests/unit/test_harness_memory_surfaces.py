"""A run's memory reach and the refine cycle are reachable from the SDK's
exports, the CLI and the MCP launch tools, not only from Python."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.enums.memory_type import MemoryType
from memorizz.metaharness.handoff import handoff_memory_record
from tests.cli_text import plain
from tests.unit.test_harness_parity import _cli, _meta, _provider, _wait_run, _workspace

pytestmark = pytest.mark.unit

RUN_A, RUN_B = "aaaaaaaa-1111", "bbbbbbbb-2222"


def _last_json(result) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_refine_classes_are_exported():
    import memorizz.metaharness as package
    from memorizz.metaharness.memory_refine import HarnessMemoryRefine, RefineSettings

    assert package.HarnessMemoryRefine is HarnessMemoryRefine
    assert package.RefineSettings is RefineSettings
    assert {"HarnessMemoryRefine", "RefineSettings"} <= set(package.__all__)


def test_the_cli_sets_what_memory_a_run_can_reach(tmp_path: Path, monkeypatch):
    workspace = _workspace(tmp_path)
    runner = _cli(lambda: _meta(tmp_path, "alpha", "beta"), monkeypatch)

    ran = runner.invoke(
        app,
        ["harness", "run", "Check it", "--harness", "alpha", "--workspace", str(workspace),
         "--memory-id", "billing", "--read-memory", "handbook", "--read-memory", "legal",
         "--mcp-access", "none", "--json"],
    )  # fmt: skip
    assert ran.exit_code == 0, ran.output
    shown = runner.invoke(app, ["harness", "show", _last_json(ran)["run_id"], "--json"])
    assert shown.exit_code == 0, shown.output
    body = _last_json(shown)
    permissions = (body.get("run") or body)["task"]["permissions"]
    assert permissions["mcp_read_memory_ids"] == ["handbook", "legal"]
    assert permissions["mcp_memory_scope"] == "run"

    opened = runner.invoke(
        app,
        ["harness", "compare", "Check it", "--harness", "alpha", "--harness", "beta",
         "--workspace", str(workspace), "--memory-scope", "all", "--mcp-access", "none",
         "--json"],
    )  # fmt: skip
    assert opened.exit_code == 0, opened.output
    assert {
        run["task"]["permissions"]["mcp_memory_scope"]
        for run in _last_json(opened)["runs"]
    } == {"all"}

    wrong = runner.invoke(
        app,
        ["harness", "run", "Check it", "--harness", "alpha", "--workspace", str(workspace),
         "--memory-scope", "some"],
    )  # fmt: skip
    assert wrong.exit_code != 0


def test_a_continued_conversation_keeps_its_memory_reach(tmp_path: Path):
    workspace = _workspace(tmp_path)
    meta = _meta(tmp_path, "alpha")
    try:
        first = meta.start(
            {
                "task": "Start",
                "workspace": str(workspace),
                "harness": "alpha",
                "memory_id": "billing",
                "permissions": {
                    "mcp_access": "none",
                    "mcp_read_memory_ids": ["handbook"],
                },
            }
        )
        _wait_run(meta, first.run_id)
        conversation = meta.conversation_id_for(meta.get_run(first.run_id))
        # The next turn names no memory reach: it keeps the conversation's.
        second = meta.continue_conversation(
            conversation,
            {"task": "Go on", "workspace": str(workspace), "harness": "alpha",
             "memory_id": "billing", "permissions": {"mcp_access": "none"}},
        )  # fmt: skip
        kept = _wait_run(meta, second.run_id)["task"]["permissions"]
        assert kept["mcp_read_memory_ids"] == ["handbook"]
        # A turn that names its own is not overridden.
        third = meta.continue_conversation(
            conversation,
            {"task": "And now", "workspace": str(workspace), "harness": "alpha",
             "memory_id": "billing",
             "permissions": {"mcp_access": "none", "mcp_read_memory_ids": ["legal"]}},
        )  # fmt: skip
        assert _wait_run(meta, third.run_id)["task"]["permissions"][
            "mcp_read_memory_ids"
        ] == ["legal"]
    finally:
        meta.close()


def test_mcp_launch_tools_take_read_only_memories(tmp_path: Path):
    from memorizz.mcp_server import create_memorizz_mcp_server
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import ALL_SCOPES, MemorizzMCPServerConfig
    from memorizz.mcp_server.runtime import MemorizzRuntime

    workspace = _workspace(tmp_path)
    approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    meta = _meta(tmp_path, "alpha")
    meta.approval_store = approvals
    provider = _provider(tmp_path)
    config = MemorizzMCPServerConfig(
        allow_writes=True,
        allow_harness_execution=True,
        harness_workspace_roots={str(workspace)},
    )
    runtime = MemorizzRuntime(
        config, provider=provider, approval_store=approvals, meta_harness=meta
    )
    alice = RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )
    try:
        started = runtime.start_harness_run(
            "Check it",
            str(workspace),
            alice,
            harness="alpha",
            memory_id="billing",
            mcp_access="none",
            mcp_read_memory_ids=["handbook"],
        )
        assert started["ok"], started
        run = _wait_run(meta, started["run"]["run_id"])
        assert run["task"]["permissions"]["mcp_read_memory_ids"] == ["handbook"]

        # Every tool that starts work offers it; none lets a model open a
        # run to every memory.
        server = create_memorizz_mcp_server(config, runtime=runtime)
        tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
        for name in (
            "memorizz_start_harness_run",
            "memorizz_start_harness_plan",
            "memorizz_start_harness_comparison",
        ):
            offered = set(inspect.signature(tools[name].fn).parameters)
            assert "mcp_read_memory_ids" in offered, name
            assert "mcp_memory_scope" not in offered, name
    finally:
        meta.close()
        provider.close()


def _save(provider, harness, run, text, memory="notes"):
    record = handoff_memory_record(
        {"harness": harness, "status": "succeeded", "run_id": run, "response": text},
        workflow={},
        memory_id=memory,
        thread_id=f"harness-run-{run}",
        agent_id="metaharness",
        user_id=None,
    )
    provider.store(
        record, memory_store_type=MemoryType.CONVERSATION_MEMORY, memory_id=memory
    )


class _Distiller:
    def generate_text(self, prompt):
        return json.dumps(
            {
                "findings": [
                    {"action": "add", "text": "The release codeword is PELICAN-4471.",
                     "run": RUN_A, "quote": "release codeword is PELICAN-4471"},
                ]
            }
        )  # fmt: skip


def test_the_cli_refines_a_memory_and_shows_what_it_knows(tmp_path: Path, monkeypatch):
    from memorizz.metaharness.memory_refine import HarnessMemoryRefine, RefineSettings

    provider = _provider(tmp_path)
    _save(provider, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    _save(provider, "codex", RUN_B, "Yes: the release codeword is **PELICAN-4471**.")
    settings = RefineSettings(tmp_path / "refine.json")

    def make():
        meta = _meta(tmp_path, "alpha", provider=provider)
        meta._memory_refine = HarnessMemoryRefine(
            provider, settings=settings, model_factory=lambda _settings: _Distiller()
        )
        return meta

    runner = _cli(make, monkeypatch)
    try:
        shown = runner.invoke(app, ["harness", "refine-settings"])
        assert "No model is set" in plain(shown.output)
        saved = runner.invoke(
            app,
            ["harness", "refine-settings", "--provider", "openai", "--model", "gpt-test",
             "--auto", "--after", "4", "--json"],
        )  # fmt: skip
        assert _last_json(saved)["settings"] == {
            "provider": "openai", "model": "gpt-test", "auto": True, "after_answers": 4,
        }  # fmt: skip
        assert "after 4 new answers" in plain(
            runner.invoke(app, ["harness", "refine-settings"]).output
        )

        raw = plain(
            runner.invoke(app, ["harness", "findings", "notes", "--raw"]).output
        )
        assert "droid" in raw and "(new)" in raw and "2 raw answers." in raw

        preview = runner.invoke(app, ["harness", "refine", "notes", "--dry-run"])
        assert preview.exit_code == 0, preview.output
        text = plain(preview.output)
        assert "Nothing was written." in text
        assert "New: The release codeword is PELICAN-4471." in text
        empty = plain(runner.invoke(app, ["harness", "findings", "notes"]).output)
        assert "0 findings; 2 new answers not distilled yet." in empty

        done = runner.invoke(app, ["harness", "refine", "notes", "--json"])
        assert done.exit_code == 0, done.output
        assert _last_json(done)["report"]["status"] == "done"
        known = plain(runner.invoke(app, ["harness", "findings", "notes"]).output)
        assert "- The release codeword is PELICAN-4471." in known
        assert "2 runs agree: droid, codex" in known
        assert "1 finding; 0 new answers not distilled yet." in known
        listed = _last_json(
            runner.invoke(app, ["harness", "findings", "notes", "--json"])
        )
        assert listed["count"] == 1 and listed["findings"][0]["confirmations"] == 2
        again = plain(runner.invoke(app, ["harness", "refine", "notes"]).output)
        assert "Nothing new to distil." in again
    finally:
        provider.close()


def test_refine_without_a_model_ends_with_a_message_and_a_failing_exit(
    tmp_path: Path, monkeypatch
):
    from memorizz.metaharness.memory_refine import HarnessMemoryRefine, RefineSettings

    provider = _provider(tmp_path)
    _save(provider, "droid", RUN_A, "The release codeword is PELICAN-4471.")

    def make():
        meta = _meta(tmp_path, "alpha", provider=provider)
        meta._memory_refine = HarnessMemoryRefine(
            provider, settings=RefineSettings(tmp_path / "refine.json")
        )
        return meta

    runner = _cli(make, monkeypatch)
    try:
        result = runner.invoke(app, ["harness", "refine", "notes"])
        assert result.exit_code == 1
        assert "No model is set" in plain(result.output)
    finally:
        provider.close()
