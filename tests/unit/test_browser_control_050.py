"""MemoRizz 0.5 browser-control provider and governance coverage."""

from __future__ import annotations

import json
import subprocess
from io import StringIO
from types import SimpleNamespace

import pytest
from click import unstyle
from typer.testing import CliRunner

from memorizz.approval import ApprovalRequired, SQLiteApprovalStore
from memorizz.browser_control import BrowserControlProvider, BrowserControlResult
from memorizz.browser_control.providers.browser_use import (
    _RESULT_MARKER,
    BrowserUseProvider,
)
from memorizz.cli import agent_factory, commands
from memorizz.cli.app import app
from memorizz.memagent import MemAgent
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memagent.managers.browser_control_manager import BrowserControlManager


class FakeBrowserProvider(BrowserControlProvider):
    provider_name = "fake-browser"

    def __init__(self):
        super().__init__()
        self.tasks = []

    def get_config(self):
        return {"provider": self.provider_name, "safe": True}

    def run_task(self, task, *, max_steps=None, timeout=None):
        self.tasks.append((task, max_steps, timeout))
        return BrowserControlResult(
            success=True,
            task=task,
            output="done",
            urls=["https://example.com"],
            actions=["navigate"],
            steps=1,
            metadata={"provider": self.provider_name},
        )


def _tool_call(name: str, arguments: dict):
    return SimpleNamespace(
        id="browser-call-1",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.unit
def test_browser_control_tool_is_always_side_effecting_and_approval_required():
    provider = FakeBrowserProvider()
    manager = BrowserControlManager(provider)
    tool = manager.get_tools()[0]
    policy = tool.__memorizz_tool_policy__.to_dict()

    assert tool.__name__ == "browser_control"
    assert policy["deterministic"] is False
    assert policy["side_effects"] is True
    assert policy["requires_approval"] is True
    assert "browser" in policy["domains"]


@pytest.mark.unit
def test_memagent_browser_control_creates_and_resumes_exact_durable_proposal(tmp_path):
    provider = FakeBrowserProvider()
    store = SQLiteApprovalStore(tmp_path / "browser-approvals.sqlite3")
    agent = MemAgent(browser_control=provider, approval_store=store)
    agent._current_memory_id = "browser-memory"
    agent._current_thread_id = "browser-thread"
    agent.semantic_tool_router.begin_turn(user_id="alice")
    agent._build_llm_tools("Use the browser to submit this form", user_id="alice")

    with pytest.raises(ApprovalRequired) as raised:
        agent._execute_and_record_tool_call(
            _tool_call(
                "browser_control",
                {
                    "task": "Open example.com and submit the support form",
                    "max_steps": 8,
                },
            ),
            [],
            workflow=None,
            user_id="alice",
            query="submit the support form",
        )

    proposal = raised.value.proposal
    assert proposal.tool_name == "browser_control"
    assert proposal.arguments["max_steps"] == 8
    assert provider.tasks == []

    agent.approve(proposal.proposal_id, approver_id="operator@example.com")
    resumed = agent.resume_approval(proposal.proposal_id, continue_model=False)

    assert resumed.consumed is True
    assert resumed.tool_result["success"] is True
    assert resumed.tool_result["output"] == "done"
    assert provider.tasks == [("Open example.com and submit the support form", 8, None)]


@pytest.mark.unit
def test_browser_use_provider_runs_fixed_worker_and_normalizes_result(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    captured = {}

    class FakeProcess:
        pid = 99999
        returncode = 0

        def communicate(self, timeout):
            captured["source"] = open(captured["command"][-1], encoding="utf-8").read()
            captured["timeout"] = timeout
            payload = {
                "success": True,
                "output": "The page title is Example Domain",
                "urls": ["https://example.com"],
                "actions": ["go_to_url", "done"],
                "errors": [],
                "steps": 2,
                "duration_seconds": 1.25,
                "browser_use_version": "0.13.7",
                "browser_policy_adapter": "browser_profile",
            }
            return f"provider log\n{_RESULT_MARKER}{json.dumps(payload)}\n", ""

    monkeypatch.setattr(
        "memorizz.browser_control.providers.browser_use.shutil.which",
        lambda _command: "/isolated/bin/browser-use",
    )
    monkeypatch.setattr(
        BrowserUseProvider,
        "_resolved_python_command",
        lambda _self: ["/isolated/bin/python"],
    )

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", popen)
    provider = BrowserUseProvider(
        allowed_domains=["example.com"],
        prohibited_domains=["accounts.example.net"],
        max_steps=12,
        task_timeout=90,
    )

    result = provider.run_task("Read the title at https://example.com", max_steps=7)

    assert result.success is True
    assert result.steps == 2
    assert result.metadata["browser_use_version"] == "0.13.7"
    assert result.metadata["browser_policy_adapter"] == "browser_profile"
    assert captured["command"][0] == "/isolated/bin/python"
    assert len(captured["command"]) == 2
    assert captured["command"][-1].endswith("worker.py")
    assert captured["timeout"] == 90
    assert "CONFIG = json.loads" in captured["source"]
    assert "Agent(" in captured["source"]
    assert "BrowserProfile" in captured["source"]
    assert "required_policy_fields" in captured["source"]
    assert "exec(CONFIG" not in captured["source"]
    assert "shell" not in captured["kwargs"]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert "memorizz-browser-" in captured["kwargs"]["cwd"]
    assert result.metadata["execution_boundary"] == "isolated_browser_use_python"
    assert result.metadata["private_worker_process"] is True
    assert provider.get_config()["requires_durable_approval"] is True


@pytest.mark.unit
def test_browser_control_config_survives_temporarily_unavailable_provider(tmp_path):
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    memory = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory"))
    agent = MemAgent(memory_provider=memory, browser_control=FakeBrowserProvider())
    agent.save()

    # ``fake-browser`` is intentionally absent from the production registry.
    # Loading records the readiness error but must not erase the saved policy.
    loaded = MemAgent.load(agent.agent_id, memory_provider=memory)
    assert loaded.has_browser_control() is False
    assert loaded.browser_control_config == {"provider": "fake-browser", "safe": True}
    loaded.save()

    saved = memory.retrieve_memagent(agent.agent_id)
    assert saved.browser_control == {"provider": "fake-browser", "safe": True}


@pytest.mark.unit
def test_cli_browser_control_is_explicit_and_secret_free(monkeypatch):
    monkeypatch.setenv("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "browseruse")
    monkeypatch.setenv("MEMORIZZ_BROWSER_USE_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv(
        "MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS", "example.com, *.notion.so"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-be-serialized")
    monkeypatch.setenv("MEMORIZZ_BROWSER_USE_PYTHON_COMMAND", "/isolated/bin/python")

    config = agent_factory.make_browser_control_config()

    assert config["provider"] == "browseruse"
    assert config["llm_provider"] == "anthropic"
    assert config["allowed_domains"] == ["example.com", "*.notion.so"]
    assert config["python_command"] == "/isolated/bin/python"
    assert "must-not-be-serialized" not in json.dumps(config)
    assert "/browser" in commands.command_completions()
    assert "/approvals" in commands.command_completions()

    runner = CliRunner()
    # Pin the terminal width so Rich does not split an option name across
    # columns on narrower Linux CI runners.
    chat_help = runner.invoke(app, ["chat", "--help"], terminal_width=160)
    run_help = runner.invoke(app, ["run", "--help"], terminal_width=160)
    assert chat_help.exit_code == 0, chat_help.output
    assert run_help.exit_code == 0, run_help.output
    compact_chat_help = "".join(unstyle(chat_help.output).split())
    compact_run_help = "".join(unstyle(run_help.output).split())
    assert "--browser-control" in compact_chat_help
    assert "--browser-control" in compact_run_help


@pytest.mark.unit
def test_cli_generic_approvals_can_decide_and_resume_exact_browser_call(tmp_path):
    from rich.console import Console

    provider = FakeBrowserProvider()
    agent = MemAgent(
        browser_control=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "cli-approvals.sqlite3"),
    )
    agent._current_memory_id = "cli-browser-memory"
    agent._current_thread_id = "cli-browser-thread"
    agent.semantic_tool_router.begin_turn(user_id="cli-user")
    agent._build_llm_tools(
        "Use browser control to open the website", user_id="cli-user"
    )
    with pytest.raises(ApprovalRequired) as raised:
        agent._execute_and_record_tool_call(
            _tool_call("browser_control", {"task": "Open example.com"}),
            [],
            workflow=None,
            user_id="cli-user",
            query="Use browser control to open the website",
        )

    proposal_id = raised.value.proposal.proposal_id
    session = SimpleNamespace(
        agent=agent,
        console=Console(file=StringIO()),
    )
    commands.cmd_approvals(
        session, f"approve {proposal_id} operator@example.com reviewed"
    )
    commands.cmd_approvals(session, f"resume {proposal_id} --no-model")

    assert provider.tasks == [("Open example.com", 25, None)]
    assert '"tool_result"' in session.console.file.getvalue()
    assert '"consumed": true' in session.console.file.getvalue()


@pytest.mark.unit
def test_browser_use_provider_requires_isolated_cli_and_matching_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "memorizz.browser_control.providers.browser_use.shutil.which",
        lambda _command: None,
    )
    provider = BrowserUseProvider()
    assert "was not found" in provider.validate_configuration()

    monkeypatch.setattr(
        "memorizz.browser_control.providers.browser_use.shutil.which",
        lambda _command: "/isolated/bin/browser-use",
    )
    monkeypatch.setattr(
        BrowserUseProvider,
        "_resolved_python_command",
        lambda _self: ["/isolated/bin/python"],
    )
    assert "OPENAI_API_KEY is required" in provider.validate_configuration()


@pytest.mark.unit
def test_browser_use_provider_discovers_isolated_python_from_entrypoint_shebang(
    tmp_path, monkeypatch
):
    isolated_python = tmp_path / "python"
    isolated_python.write_text("", encoding="utf-8")
    isolated_python.chmod(0o700)
    entrypoint = tmp_path / "browser-use"
    entrypoint.write_text(
        f"#!{isolated_python}\nprint('entrypoint')\n", encoding="utf-8"
    )
    entrypoint.chmod(0o700)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")

    provider = BrowserUseProvider(command=str(entrypoint))

    assert provider._resolved_python_command() == [str(isolated_python)]
    assert provider.validate_configuration() is None


@pytest.mark.unit
def test_builder_clones_browser_control_configuration():
    provider = FakeBrowserProvider()
    builder = MemAgentBuilder().with_browser_control(provider)
    clone = builder.clone()

    agent = clone.build()

    assert agent.has_browser_control() is True
    assert agent.get_browser_control_provider_name() == "fake-browser"
    assert "browser_control" in agent.tool_manager.list_tools()
