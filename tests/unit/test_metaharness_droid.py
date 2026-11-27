"""Factory's Droid as a harness, run against a stand-in executable.

The stand-in prints events recorded from a real ``droid exec -o stream-json``
run (Droid 1.15.0), trimmed to what the adapter reads.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    DroidHarness,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)

FAKE_DROID = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("1.15.0"); sys.exit(0)
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump({{"argv": args, "env": sorted(os.environ), "cwd": os.getcwd()}}, handle)
for event in json.loads(os.environ["FAKE_EVENTS"]):
    print(json.dumps(event), flush=True)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""

USAGE = {
    "input_tokens": 4,
    "output_tokens": 18,
    "cache_read_input_tokens": 13528,
    "cache_creation_input_tokens": 17913,
    "factory_credits": 36947,
    "thinking_tokens": 0,
    "ttft_ms": 2306.5,
}
EVENTS = [
    {"type": "system", "subtype": "init", "cwd": "/w", "session_id": "sess-9",
     "tools": ["Read", "LS", "Execute"], "model": "gpt-6.2-sol"},
    {"type": "message", "role": "user", "id": "m0", "text": "task"},
    {"type": "tool_call", "id": "call_1", "messageId": "m1", "toolId": "Script",
     "toolName": "Script",
     "parameters": {"script": 'const r=await tools.Read({file_path:"calc.py"}); text(r);'}},
    {"type": "tool_result", "id": "call_1", "messageId": "m2", "toolId": "Script",
     "isError": False, "value": "1→def add(a, b):\n2→    return a - b"},
    {"type": "message", "role": "assistant", "id": "m3",
     "text": "add() subtracts; it is wrong."},
    {"type": "completion", "finalText": "add() subtracts; it is wrong.", "numTurns": 4,
     "durationMs": 6856, "session_id": "sess-9", "usage": USAGE},
]  # fmt: skip


def _droid(tmp_path: Path, record: Path, events=EVENTS, **options) -> DroidHarness:
    program = tmp_path / "droid"
    program.write_text(FAKE_DROID.format(python=sys.executable), encoding="utf-8")
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    extra = {"FAKE_RECORD": str(record), "FAKE_EVENTS": json.dumps(events)}
    extra.update(options.pop("extra_env", {}))
    return DroidHarness(command=str(program), extra_env=extra, **options)


def _service(tmp_path: Path, adapter) -> MetaHarness:
    return MetaHarness(
        adapters=[adapter],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )


def _task(tmp_path: Path, **values) -> HarnessTask:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return HarnessTask(
        task="Is add() correct?", workspace=str(workspace), harness="droid", **values
    )


@pytest.fixture
def record(tmp_path, monkeypatch):
    monkeypatch.setenv("FACTORY_API_KEY", "fk-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-for-droid")
    return tmp_path / "record.json"


@pytest.mark.unit
def test_droid_reads_a_workspace_and_reports_its_answer_and_tokens(tmp_path, record):
    service = _service(tmp_path, _droid(tmp_path, record))
    try:
        result = service.run(_task(tmp_path))

        assert result.status == HarnessStatus.SUCCEEDED, result.error
        assert result.final_response == "add() subtracts; it is wrong."
        # New input plus what the cache held; Factory bills credits, not dollars.
        assert result.usage["input_tokens"] == 4 + 13528 + 17913
        assert result.usage["output_tokens"] == 18
        assert result.cost_usd is None
        seen = json.loads(record.read_text())
        argv = seen["argv"]
        # Read-only: no --auto, in the workspace, and only tools that read.
        assert argv[: argv.index("--")] == [
            "exec", "--cwd", argv[2], "-o", "stream-json",
            "--only-tools", "Read,LS,Grep,Glob",
        ]  # fmt: skip
        assert "workspace" in argv[2]
        assert argv[argv.index("--") + 1].startswith("--- Host execution contract")
        assert "FACTORY_API_KEY" in seen["env"] and "OPENAI_API_KEY" not in seen["env"]
    finally:
        service.close()


@pytest.mark.unit
def test_droid_events_become_the_steps_a_run_shows(tmp_path, record):
    adapter = _droid(tmp_path, record)
    kinds, updates = [], {}
    for payload in EVENTS:
        events, changed = adapter.parse_event("run-1", payload)
        kinds += [event.type.value for event in events]
        updates.update(changed)

    assert kinds == ["status", "tool_call", "tool_result", "message", "usage"]
    assert updates["checkpoint"] == {"session_id": "sess-9"}
    assert updates["usage"]["factory_credits"] == 36947

    patch = {"type": "tool_call", "id": "c2", "toolName": "Script",
             "parameters": {"script": "// @input: patch\n*** Begin Patch\nApplyPatch"}}  # fmt: skip
    assert adapter.parse_event("run-1", patch)[0][0].type.value == "file_change"
    stopped = adapter.parse_event(
        "run-1",
        {
            "type": "error",
            "message": "Exec ended early: insufficient permission to proceed.",
        },
    )[1]
    assert stopped["error_code"] == "droid_permission_stop"


@pytest.mark.unit
def test_droid_edits_only_behind_an_isolated_wrapper_and_only_with_file_tools(
    tmp_path, record
):
    plain = _service(tmp_path, _droid(tmp_path, record))
    try:
        refused = plain.run(_task(tmp_path, permissions={"workspace_mode": "direct"}))
        assert refused.status == HarnessStatus.FAILED
        assert "write_requires_external_isolation" in json.dumps(refused.routing)
        # It has no web tools, so it cannot honour a run that asks for the web.
        online = plain.run(_task(tmp_path, permissions={"network": "full"}))
        assert online.status == HarnessStatus.FAILED
        assert "network_policy_unsupported" in json.dumps(online.routing)
    finally:
        plain.close()

    adapter = _droid(
        tmp_path, record, external_isolation=True, default_model="custom:Flint-0"
    )
    workspace = tmp_path / "workspace"
    command = adapter.build_command(
        _task(tmp_path, permissions={"workspace_mode": "direct"}),
        workspace=workspace,
        prompt="Fix add()",
    )
    assert command[1:] == [
        "exec", "--cwd", str(workspace), "-o", "stream-json", "--auto", "low",
        "--only-tools", "Read,LS,Grep,Glob,Edit,ApplyPatch,Create",
        "-m", "custom:Flint-0", "--", "Fix add()",
    ]  # fmt: skip
    # A task may narrow the tools, never add commands or the web.
    narrowed = adapter.build_command(
        _task(tmp_path, permissions={"allowed_tools": ["read", "grep", "execute"]}),
        workspace=workspace,
        prompt="x",
    )
    assert narrowed[narrowed.index("--only-tools") + 1] == "Read,Grep"


@pytest.mark.unit
def test_droid_failures_and_missing_setup_are_said_plainly(
    tmp_path, record, monkeypatch
):
    failed = [
        {"type": "system", "subtype": "init", "session_id": "s"},
        {"type": "error", "source": "cli", "message": "Model refused"},
    ]
    service = _service(tmp_path, _droid(tmp_path, record, events=failed))
    try:
        result = service.run(_task(tmp_path))
        assert result.status == HarnessStatus.FAILED
        assert (result.error_code, result.error) == ("droid_failed", "Model refused")
    finally:
        service.close()

    monkeypatch.delenv("FACTORY_API_KEY")
    capability = _droid(tmp_path, record).probe()
    assert capability.version == "1.15.0"
    assert capability.error_code == "authentication_required"
    assert "FACTORY_API_KEY" in capability.remediation

    missing = DroidHarness(command=str(tmp_path / "no-droid")).probe()
    assert missing.available is False and "docs.factory.com" in missing.remediation
