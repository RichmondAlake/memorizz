"""Node-based CLIs must use their installed runtime through every launch path."""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
from pathlib import Path

import pytest

from memorizz.metaharness import HarnessContextPack, HarnessTask
from memorizz.metaharness.adapters import CodexHarness, PiHarness
from memorizz.metaharness.base import SubprocessHarness
from memorizz.metaharness.security import build_child_environment


def _executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)
    return path


@pytest.fixture
def node_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    system = tmp_path / "old-system-bin"
    _executable(system / "node", "#!/bin/sh\necho 'Error: wrong runtime' >&2\nexit 1\n")
    prefix = tmp_path / "node-install with spaces"
    record = tmp_path / "node-launches.jsonl"
    # A deterministic runtime records the selected script and CLI arguments.
    _executable(
        prefix / "bin/node",
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"with open({str(record)!r}, 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "args = sys.argv[2:]\n"
        "if args == ['--version']: print('test-cli 1.0')\n"
        "elif args == ['login', 'status']: print('Logged in using ChatGPT')\n"
        "elif args[:2] == ['auth', 'check']: print('{\"status\":\"ready\"}')\n"
        "elif args == ['debug', 'models']: print('{\"models\":[{\"slug\":\"test-model\"}]}')\n"
        "elif args == ['--list-models']: print('provider model\\ndeepseek test-model')\n"
        'else: print(\'{"message":"correct runtime"}\')\n',
    )
    for name in ("codex", "pi"):
        script = _executable(
            prefix / f"lib/node_modules/{name}/bin/cli.js",
            "#!/usr/bin/env node\n// Test CLI\n",
        )
        (prefix / "bin" / name).symlink_to(script)
    search_path = os.pathsep.join([str(system), str(prefix / "bin"), os.defpath])
    monkeypatch.setenv("PATH", search_path)
    for name in ("CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_HOME"):
        monkeypatch.delenv(name, raising=False)
    return prefix, record, search_path


@pytest.mark.parametrize("resolved", [False, True])
def test_codex_probe_login_and_catalog_use_installation_runtime(node_install, resolved):
    prefix, record, search_path = node_install
    command = prefix / "bin/codex"
    harness = CodexHarness(command=str(command.resolve() if resolved else command))
    capability = harness.probe()
    assert capability.available and not capability.error
    assert capability.metadata["authentication_status"] == "authenticated"
    assert harness.model_catalog() == ["test-model"]
    assert sorted(
        json.loads(line)[1:] for line in record.read_text().splitlines()
    ) == sorted(
        [
            ["--version"],
            ["login", "status"],
            ["debug", "models"],
        ]
    )
    assert os.environ["PATH"] == search_path


def test_pi_probe_auth_and_catalog_use_installation_runtime(node_install, monkeypatch):
    prefix, record, _ = node_install
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    harness = PiHarness(command=str(prefix / "bin/pi"), provider="deepseek")
    capability = harness.probe()
    assert capability.available and not capability.error
    assert capability.metadata["authentication_status"] == "authenticated"
    assert "deepseek/test-model" in capability.models
    launches = [json.loads(line)[1:] for line in record.read_text().splitlines()]
    assert ["--version"] in launches and ["--list-models"] in launches
    assert ["auth", "check", "--provider", "deepseek", "--json"] in launches


class _NodeHarness(SubprocessHarness):
    node_cli = True

    def build_command(self, task, *, workspace, prompt):
        return [self.command, "run"]

    def parse_event(self, run_id, payload):
        return [], {"final_response": payload.get("message", "")}


def test_shell_wrapper_is_preserved_for_probe_and_workspace_execution(
    tmp_path, node_install, monkeypatch
):
    prefix, record, search_path = node_install
    wrapper_record = tmp_path / "wrapper-used"
    wrapper = _executable(
        tmp_path / "wrappers/codex",
        "#!/bin/sh\n"
        f"echo called >> {shlex.quote(str(wrapper_record))}\n"
        f"exec {shlex.quote(str(prefix / 'bin/codex'))} \"$@\"\n",
    )
    monkeypatch.setenv("PATH", str(wrapper.parent) + os.pathsep + search_path)
    harness = _NodeHarness(command="codex")
    capability = harness.probe()
    assert capability.available and capability.command == str(wrapper.resolve())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _executable(workspace / "codex", "#!/bin/sh\nexit 42\n")
    _executable(workspace / "node", "#!/bin/sh\nexit 43\n")
    task = HarnessTask(task="Inspect", workspace=str(workspace))
    result = harness.run(
        task,
        workspace=workspace,
        context_pack=HarnessContextPack(query="Inspect"),
        emit=lambda event: None,
        cancel_event=threading.Event(),
    )
    assert result.exit_code == 0 and result.final_response == "correct runtime"
    assert len(wrapper_record.read_text().splitlines()) == 2
    assert len(record.read_text().splitlines()) == 2


def test_explicit_runtime_path_override_takes_precedence(node_install):
    prefix, _, _ = node_install
    environment = build_child_environment(
        node_command=str(prefix / "bin/codex"), overrides={"PATH": "/explicit/runtime"}
    )
    assert environment["PATH"] == "/explicit/runtime"


def test_missing_installation_runtime_retains_inherited_path(tmp_path, node_install):
    _, _, search_path = node_install
    script = _executable(tmp_path / "codex", "#!/usr/bin/env node\n// no runtime\n")
    assert build_child_environment(node_command=str(script))["PATH"] == search_path


def test_native_cli_retains_inherited_path(tmp_path, node_install):
    _, _, search_path = node_install
    native = _executable(tmp_path / "codex", f"#!{sys.executable}\nprint('native')\n")
    assert build_child_environment(node_command=str(native))["PATH"] == search_path


def test_failed_probe_shows_redacted_error_instead_of_stack_trace_as_version(tmp_path):
    executable = _executable(
        tmp_path / "broken",
        "#!/bin/sh\necho 'file:///cli.js:107' >&2\n"
        "echo 'Error: failed with sk-test-secret-12345678901234' >&2\nexit 1\n",
    )
    capability = _NodeHarness(command=str(executable)).probe()
    assert not capability.available and capability.version is None
    assert "Error: failed with [REDACTED]" in capability.error
    assert "sk-test-secret" not in capability.error


def test_successful_empty_version_output_does_not_crash(tmp_path):
    executable = _executable(tmp_path / "quiet-cli", "#!/bin/sh\nexit 0\n")
    capability = _NodeHarness(command=str(executable)).probe()
    assert capability.available and capability.version is None
