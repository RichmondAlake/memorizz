"""pi and DeepSeek adapters, run against fake executables in subprocesses."""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memorizz.approval import SQLiteApprovalStore
from memorizz.metaharness import (
    DeepSeekHarness,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    PiHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.adapters import deepseek_cost, deepseek_peak
from memorizz.metaharness.security import redact

# Recorded from a real `pi --mode json` run (pi 0.87.1, Ollama qwen2.5:7b),
# trimmed to the records the adapter reads plus some it must skip.
PI_EVENTS = [
    {"type": "session", "version": 3, "id": "sess-1", "cwd": "/w"},
    {"type": "agent_start"},
    {"type": "message_end", "message": {"role": "user", "content": "task"}},
    {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "call_1",
                    "name": "read",
                    "arguments": {"path": "calc.py"},
                }
            ],
            "provider": "deepseek",
            "model": "deepseek-flash",
            "usage": {
                "input": 1387,
                "output": 28,
                "cacheRead": 0,
                "cacheWrite": 0,
                "reasoning": 0,
                "totalTokens": 1415,
                "cost": {"total": 0.0004},
            },
            "stopReason": "toolUse",
        },
    },
    {
        "type": "tool_execution_start",
        "toolCallId": "call_1",
        "toolName": "read",
        "args": {"path": "calc.py"},
    },
    {
        "type": "tool_execution_end",
        "toolCallId": "call_1",
        "toolName": "read",
        "result": {"content": [{"type": "text", "text": "def add(a, b):\n"}]},
        "isError": False,
    },
    {
        "type": "message_update",
        "assistantMessageEvent": {"type": "text_delta", "delta": "The"},
    },
    {
        "type": "message_end",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "add() subtracts; it is wrong."}],
            "provider": "deepseek",
            "model": "deepseek-flash",
            "usage": {
                "input": 40,
                "output": 31,
                "cacheRead": 1401,
                "cacheWrite": 0,
                "totalTokens": 1472,
                "cost": {"total": 0.0001},
            },
            "stopReason": "stop",
        },
    },
    {"type": "agent_end", "messages": []},
    {"type": "agent_settled"},
]

FAKE_PI = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("0.87.1"); sys.exit(0)
if args[:2] == ["auth", "check"]:
    ready = bool(os.environ.get("DEEPSEEK_API_KEY"))
    print(json.dumps({{"status": "ready" if ready else "not_ready"}}))
    sys.exit(0 if ready else 1)
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump({{"argv": args, "env": sorted(os.environ)}}, handle)
for event in json.loads(os.environ["FAKE_EVENTS"]):
    print(json.dumps(event), flush=True)
"""

FAKE_CLAUDE = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("2.1.284 (Claude Code)"); sys.exit(0)
if args == ["--help"]:
    print("  --restricted  Restricted mode\n  --tools <tools...>\n  --strict-mcp-config")
    sys.exit(0)
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump({{
        "argv": args,
        "env": sorted(os.environ),
        "config_dir": os.environ.get("CLAUDE_CONFIG_DIR"),
        "key": os.environ.get("ANTHROPIC_API_KEY"),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL"),
        "model": os.environ.get("ANTHROPIC_MODEL"),
    }}, handle)
usage = {{"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 4000}}
message = {{"id": "msg_1", "model": "deepseek-flash", "usage": usage,
           "content": [{{"type": "text", "text": "Reviewed."}}]}}
for line in [
    {{"type": "system", "subtype": "init", "session_id": "s1", "model": "deepseek-flash"}},
    {{"type": "assistant", "message": message}},
    {{"type": "assistant", "message": message}},
    {{"type": "result", "result": "Reviewed.", "total_cost_usd": 9.99,
      "usage": usage, "session_id": "s1"}},
]:
    print(json.dumps(line), flush=True)
"""


def _executable(path: Path, source: str) -> Path:
    path.write_text(source.format(python=sys.executable), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _service(tmp_path: Path, adapter) -> MetaHarness:
    return MetaHarness(
        adapters=[adapter],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )


def _task(tmp_path: Path, harness: str, **values) -> HarnessTask:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return HarnessTask(
        task="Is add() correct?", workspace=str(workspace), harness=harness, **values
    )


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "record.json"


def _fixtures(record: Path, events=PI_EVENTS) -> dict:
    """Where the fake executables record argv/env and what they print."""
    return {"FAKE_RECORD": str(record), "FAKE_EVENTS": json.dumps(events)}


@pytest.mark.unit
def test_pi_run_normalizes_events_usage_and_scopes_credentials(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-must-not-reach-pi")
    pi = PiHarness(
        command=str(_executable(tmp_path / "pi", FAKE_PI)),
        provider="deepseek",
        default_model="deepseek-flash",
        extra_env=_fixtures(fake_env),
    )
    service = _service(tmp_path, pi)
    try:
        result = service.run(_task(tmp_path, "pi"))
        assert result.status == HarnessStatus.SUCCEEDED, result.error
        assert result.final_response == "add() subtracts; it is wrong."
        # Prompt tokens include cache reads; cost is pi's per-message sum.
        assert result.usage["input_tokens"] == 1387 + 40 + 1401
        assert result.usage["cached_input_tokens"] == 1401
        assert result.usage["output_tokens"] == 59
        assert result.cost_usd == pytest.approx(0.0005)

        recorded = json.loads(fake_env.read_text())
        argv = recorded["argv"]
        assert argv[: argv.index("--")] == [
            "--mode",
            "json",
            "--offline",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-approve",
            "--no-session",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-flash",
            "--tools",
            "read,grep,find,ls",
        ]
        assert argv[argv.index("--") + 1].startswith("--- Host execution contract")
        assert "DEEPSEEK_API_KEY" in recorded["env"]
        assert "OPENAI_API_KEY" not in recorded["env"]
        assert {"PI_OFFLINE", "PI_TELEMETRY"} <= set(recorded["env"])

        events = service.events(result.run_id)
        kinds = [event["type"] for event in events]
        assert "message_update" not in json.dumps(events)
        call = next(event for event in events if event["type"] == "tool_call")
        assert call["data"] == {
            "call_id": "call_1",
            "name": "read",
            "input": {"path": "calc.py"},
        }
        assert "tool_result" in kinds and kinds.count("message") == 1
        # MemoRizz memory reached pi through the prompt, not MCP.
        assert any(
            event["data"].get("mcp") == "context_pack_fallback" for event in events
        )
    finally:
        service.close()


@pytest.mark.unit
def test_pi_model_errors_fail_the_run_even_with_exit_code_zero(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    failing = [
        {"type": "session", "id": "s"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "429 rate limited",
            },
        },
    ]
    pi = PiHarness(
        command=str(_executable(tmp_path / "pi", FAKE_PI)),
        extra_env=_fixtures(fake_env, failing),
    )
    service = _service(tmp_path, pi)
    try:
        result = service.run(_task(tmp_path, "pi"))
        assert result.status == HarnessStatus.FAILED
        assert (result.error_code, result.error) == ("pi_failed", "429 rate limited")
    finally:
        service.close()


@pytest.mark.unit
def test_pi_probe_reports_missing_provider_credentials(tmp_path: Path, fake_env):
    pi = PiHarness(
        command=str(_executable(tmp_path / "pi", FAKE_PI)), provider="deepseek"
    )
    capability = pi.probe()
    assert capability.version == "0.87.1"
    assert capability.error_code == "authentication_required"
    assert "deepseek" in capability.error
    assert "DEEPSEEK_API_KEY" in capability.remediation


@pytest.mark.unit
def test_pi_policy_routes_reads_and_blocks_unisolated_edits(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    command = str(_executable(tmp_path / "pi", FAKE_PI))
    service = _service(tmp_path, PiHarness(command=command, provider="deepseek"))
    try:
        # Read-only MCP falls back to the context pack; governed writes cannot.
        governed = service.run(
            _task(tmp_path, "pi", permissions={"mcp_access": "governed_write"})
        )
        assert governed.status == HarnessStatus.FAILED
        assert "mcp_unsupported" in json.dumps(governed.routing)
        edit = service.run(
            _task(tmp_path, "pi", permissions={"workspace_mode": "direct"})
        )
        assert edit.status == HarnessStatus.FAILED
        assert "write_requires_external_isolation" in json.dumps(edit.routing)
    finally:
        service.close()

    isolated = PiHarness(command=command, provider="deepseek", external_isolation=True)
    task = _task(
        tmp_path,
        "pi",
        permissions={"workspace_mode": "direct", "denied_tools": ["write"]},
    )
    argv = isolated.build_command(task, workspace=Path(task.workspace), prompt="p")
    assert argv[argv.index("--tools") + 1] == "read,grep,find,ls,edit"
    assert "bash" not in argv[argv.index("--tools") + 1]
    narrowed = _task(tmp_path, "pi", permissions={"allowed_tools": ["read", "bash"]})
    argv = isolated.build_command(
        narrowed, workspace=Path(narrowed.workspace), prompt="p"
    )
    assert argv[argv.index("--tools") + 1] == "read"


@pytest.mark.unit
def test_deepseek_runs_claude_code_on_deepseek_without_anthropic_credentials(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-must-not-leave")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token-must-not-leave")
    harness = DeepSeekHarness(
        command=str(_executable(tmp_path / "claude", FAKE_CLAUDE)),
        extra_env=_fixtures(fake_env),
    )
    service = _service(tmp_path, harness)
    try:
        result = service.run(
            _task(
                tmp_path,
                "deepseek",
                model="deepseek-v4-pro",
                budget={"max_cost_usd": 5},
                permissions={
                    # Even an approved allowlist cannot forward Anthropic keys.
                    "allowed_env": ["ANTHROPIC_AUTH_TOKEN"],
                    "require_approval": False,
                    "mcp_access": "none",
                },
            )
        )
        assert result.status == HarnessStatus.SUCCEEDED, result.error
        recorded = json.loads(fake_env.read_text())
        assert recorded["key"] == "ds-test-key"
        assert recorded["base_url"] == "https://api.deepseek.com/anthropic"
        assert recorded["model"] == "deepseek-v4-pro"
        assert "ANTHROPIC_AUTH_TOKEN" not in recorded["env"]
        assert "DEEPSEEK_API_KEY" not in recorded["env"]
        argv = recorded["argv"]
        assert "--max-budget-usd" not in argv and "--restricted" in argv
        assert argv[argv.index("--model") + 1] == "deepseek-v4-pro"
        # Claude Code's Anthropic-priced 9.99 is replaced by DeepSeek's rates,
        # and the repeated assistant message is counted once.
        expected = deepseek_cost(
            "deepseek-v4-pro",
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 4000,
            },
        )
        assert result.cost_usd == pytest.approx(expected)
        assert result.cost_usd < 0.01
    finally:
        service.close()


@pytest.mark.unit
def test_deepseek_probe_requires_a_deepseek_key(tmp_path: Path, fake_env) -> None:
    harness = DeepSeekHarness(
        command=str(_executable(tmp_path / "claude", FAKE_CLAUDE))
    )
    capability = harness.probe()
    assert capability.available and capability.error_code == "authentication_required"
    assert capability.error == "DEEPSEEK_API_KEY is not set."
    assert "doctor deepseek" in capability.remediation
    assert capability.models == ["deepseek-flash", "deepseek-v4-pro"]
    assert capability.metadata["runtime"] == "claude-code"


@pytest.mark.unit
def test_deepseek_prices_follow_peak_hours() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
    }
    monday_peak = datetime(2026, 9, 28, 7, 0, tzinfo=timezone.utc)
    monday_off = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)
    saturday = datetime(2026, 9, 26, 7, 0, tzinfo=timezone.utc)
    assert deepseek_peak(monday_peak) and not deepseek_peak(monday_off)
    assert not deepseek_peak(saturday)
    assert deepseek_cost("deepseek-flash", usage, at=monday_peak) == pytest.approx(
        0.006 + 0.30 + 1.20
    )
    assert deepseek_cost("deepseek-flash[1m]", usage, at=monday_off) == pytest.approx(
        0.003 + 0.15 + 0.60
    )
    assert deepseek_cost("deepseek-v4-flash", usage, at=monday_off) == pytest.approx(
        0.753
    )
    assert deepseek_cost("claude-sonnet", usage) is None


@pytest.mark.unit
def test_redaction_keeps_camel_case_token_counts_but_not_credentials() -> None:
    value = redact(
        {
            "totalTokens": 12,
            "inputTokens": 3,
            "accessToken": "abc",
            "sessionToken": "1234",
        }
    )
    assert value["totalTokens"] == 12 and value["inputTokens"] == 3
    assert value["accessToken"] == value["sessionToken"] == "[REDACTED]"


@pytest.mark.unit
def test_from_env_registers_deepseek_pi_and_hermes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_PI_PROVIDER", "deepseek")
    monkeypatch.setenv("MEMORIZZ_PI_EXTERNAL_ISOLATION", "true")
    monkeypatch.setenv("MEMORIZZ_HERMES_BASE_URL", "http://127.0.0.1:11434/v1")
    service = MetaHarness.from_env(
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    try:
        assert {"deepseek", "pi", "hermes"} <= set(service.adapters)
        assert service.adapters["hermes"].provider == "custom"
        assert service.adapters["pi"].provider == "deepseek"
        assert service.adapters["pi"].external_isolation is True
        assert service.adapters["deepseek"].default_model == "deepseek-flash"
        assert service.router.preference[-3:] == ["deepseek", "pi", "hermes"]
    finally:
        service.close()


@pytest.mark.unit
def test_existing_environment_still_passes_harness_os_basics() -> None:
    # Guard for the env hook refactor: basics like PATH still reach children.
    from memorizz.metaharness.security import build_child_environment

    assert build_child_environment().get("PATH") == os.environ.get("PATH")


@pytest.mark.unit
def test_claude_code_loads_exactly_the_policy_tools_in_restricted_mode(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    from memorizz.metaharness import ClaudeCodeHarness

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    harness = ClaudeCodeHarness(
        command=str(_executable(tmp_path / "claude", FAKE_CLAUDE)),
        extra_env=_fixtures(fake_env),
    )
    assert harness.probe().metadata["tool_isolation"] == "restricted"
    service = _service(tmp_path, harness)
    try:
        result = service.run(
            _task(
                tmp_path,
                "claude-code",
                # An explicit host decision, as when the operator approves.
                permissions={"network": "full", "require_approval": False},
            )
        )
        assert result.status == HarnessStatus.SUCCEEDED, result.error
        recorded = json.loads(fake_env.read_text())
        argv = recorded["argv"]
        assert "--bare" not in argv
        assert argv[:3] == ["--restricted", "--strict-mcp-config", "--print"]
        # --bare only ever loaded Bash, Edit and Read; these are loaded now.
        assert argv[argv.index("--tools") + 1] == "Read,Glob,Grep,WebFetch,WebSearch"
        assert "mcp__memorizz__*" in argv[argv.index("--allowedTools") + 1]
        assert recorded["config_dir"].endswith("/claude-config")
        assert not Path(recorded["config_dir"]).exists()  # removed with the run
    finally:
        service.close()

    read_only = ClaudeCodeHarness(command=harness.command)
    read_only.probe()
    task = _task(tmp_path, "claude-code", permissions={"mcp_access": "none"})
    argv = read_only.build_command(task, workspace=Path(task.workspace), prompt="p")
    assert argv[argv.index("--tools") + 1] == "Read,Glob,Grep"
    assert "WebSearch" in argv[argv.index("--disallowedTools") + 1]


# Real `hermes chat --format stream-json` records (Hermes Agent 0.21.5).
HERMES_EVENTS = [
    {"type": "system", "subtype": "init", "model": "m", "session_id": "20260929_1"},
    {"type": "tool_use", "name": "read_file", "input": {"path": "hello.txt"}},
    {
        "type": "tool_result",
        "name": "read_file",
        "output": '{"content": "1|hello"}',
        "duration_ms": 28,
        "is_error": False,
    },
    {"type": "tool_use", "name": "write_file", "input": {"path": "out.txt"}},
    {"type": "tool_result", "name": "write_file", "output": "{}", "is_error": True},
    {"type": "text", "text": "\n\nDone. "},
    {"type": "text", "text": "It says hello."},
    {
        "type": "result",
        "session_id": "20260929_1",
        "exit_code": 0,
        "text": "Done. It says hello.",
        "tokens": {
            "input": 2400,
            "output": 68,
            "total": 2468,
            "cache_read": 0,
            "cache_write": 0,
        },
        "duration_ms": 3136,
    },
]

FAKE_HERMES = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
if args == ["--version"]:
    print(os.environ.get("FAKE_VERSION", "Hermes Agent v0.21.5 (2026.9.24)"))
    print("Install method: git"); sys.exit(0)
home = os.environ["HERMES_HOME"]
with open(os.environ["FAKE_RECORD"], "w") as handle:
    json.dump({{
        "argv": args,
        "env": sorted(os.environ),
        "config": json.load(open(os.path.join(home, "config.yaml"))),
        "query": open(args[args.index("--query-file") + 1]).read(),
        "safe_root": os.environ.get("HERMES_WRITE_SAFE_ROOT"),
        "safe_root_files": os.listdir(os.environ["HERMES_WRITE_SAFE_ROOT"]),
    }}, handle)
if os.environ.get("FAKE_PLAIN"):
    print(os.environ["FAKE_PLAIN"]); sys.exit(1)
for event in json.loads(os.environ["FAKE_EVENTS"]):
    print(json.dumps(event), flush=True)
"""


def _hermes(tmp_path: Path, record: Path, **env) -> "HermesHarness":
    from memorizz.metaharness import HermesHarness

    return HermesHarness(
        command=str(_executable(tmp_path / "hermes", FAKE_HERMES)),
        provider="anthropic",
        default_model="claude-sonnet-5",
        extra_env={**_fixtures(record, HERMES_EVENTS), **env},
    )


@pytest.mark.unit
def test_hermes_runs_hermetically_with_only_file_tools(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-reach-hermes")
    service = _service(tmp_path, _hermes(tmp_path, fake_env))
    try:
        result = service.run(_task(tmp_path, "hermes"))
        assert result.status == HarnessStatus.SUCCEEDED, result.error
        assert result.final_response == "Done. It says hello."
        assert result.usage == {
            "input_tokens": 2400,
            "output_tokens": 68,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "total_tokens": 2468,
        }
        recorded = json.loads(fake_env.read_text())
        argv = recorded["argv"]
        assert argv[:2] == ["chat", "--query-file"]
        assert argv[argv.index("-t") + 1] == "file,memorizz"  # no terminal, no web
        assert {"--ignore-rules", "--format", "stream-json"} <= set(argv)
        assert argv[argv.index("--provider") + 1] == "anthropic"
        assert "-z" not in argv
        # The prompt is passed in a private file, not argv.
        assert recorded["query"].startswith("--- Host execution contract")
        assert not any("Host execution contract" in part for part in argv)
        config = recorded["config"]
        assert config["auth"] == {"adopt_external_logins": False}
        assert config["updates"] == {"check": False}
        assert config["approvals"] == {"single_query_mode": "deny"}
        assert config["model"] == {
            "provider": "anthropic",
            "default": "claude-sonnet-5",
        }
        assert config["mcp_servers"]["memorizz"]["args"] == [
            "-m",
            "memorizz.mcp_server",
        ]
        # Read-only: writes may only land in an empty throwaway folder.
        assert recorded["safe_root"].endswith("hermes-no-writes")
        assert recorded["safe_root_files"] == []
        assert "ANTHROPIC_API_KEY" in recorded["env"]
        assert "OPENROUTER_API_KEY" not in recorded["env"]

        events = service.events(result.run_id)
        by_type = {}
        for event in events:
            by_type.setdefault(event["type"], []).append(event["data"])
        assert by_type["tool_call"][0]["name"] == "read_file"
        assert by_type["file_change"][0]["path"] == "out.txt"
        assert by_type["message"] == [
            {"role": "assistant", "text": "Done. It says hello."}
        ]
    finally:
        service.close()


@pytest.mark.unit
def test_hermes_web_access_and_edits_follow_policy(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    service = _service(tmp_path, _hermes(tmp_path, fake_env))
    try:
        pending = service.run(
            _task(
                tmp_path,
                "hermes",
                permissions={
                    "network": "full",
                    "workspace_mode": "direct",
                    "mcp_access": "none",
                },
            )
        )
        # Edits and web access need the host's approval first.
        assert pending.status == HarnessStatus.PENDING_APPROVAL
        proposal = pending.checkpoint["proposal_id"]
        service.approve(proposal, approver_id="operator@example.com")
        result = service.resume_approval(proposal)
        assert result.status == HarnessStatus.SUCCEEDED, result.error
        recorded = json.loads(fake_env.read_text())
        assert recorded["argv"][recorded["argv"].index("-t") + 1] == "file,web"
        assert recorded["safe_root"] == str((tmp_path / "workspace").resolve())
        assert "mcp_servers" not in recorded["config"]
        # Command-running toolsets can never be requested.
        blocked = service.run(
            _task(
                tmp_path,
                "hermes",
                permissions={"allowed_tools": ["terminal"], "mcp_access": "none"},
            )
        )
        assert blocked.status == HarnessStatus.FAILED
        assert "requested_tool_requires_external_isolation" in json.dumps(
            blocked.routing
        )
    finally:
        service.close()


@pytest.mark.unit
def test_hermes_missing_provider_is_an_authentication_error(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    plain = (
        "It looks like Hermes isn't configured yet -- no API keys or providers found."
    )
    service = _service(tmp_path, _hermes(tmp_path, fake_env, FAKE_PLAIN=plain))
    try:
        result = service.run(
            _task(tmp_path, "hermes", permissions={"mcp_access": "none"})
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "authentication_required"
        assert "memorizz harness doctor hermes" in result.remediation
    finally:
        service.close()


@pytest.mark.unit
def test_hermes_probe_checks_identity_version_and_key(
    tmp_path: Path, fake_env, monkeypatch
) -> None:
    ready = _hermes(tmp_path, fake_env)
    capability = ready.probe()
    assert capability.error_code == "authentication_required"  # no key yet
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    fresh = _hermes(tmp_path, fake_env)
    assert fresh.probe().to_dict()["ready"] is True
    assert fresh.probe().version.startswith("Hermes Agent v0.21.5")

    for version, message in [
        ("Hermes JavaScript engine 0.12.0", "is not Nous Research's Hermes Agent"),
        ("Hermes Agent v0.21.3 (2026.9.1)", "0.21.4 or newer"),
    ]:
        capability = _hermes(tmp_path, fake_env, FAKE_VERSION=version).probe()
        assert capability.available is False and message in capability.error

    local = _hermes(tmp_path, fake_env)
    local.provider, local.base_url = "custom", "http://127.0.0.1:11434/v1"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert local.probe().to_dict()["ready"] is True  # local servers need no key
