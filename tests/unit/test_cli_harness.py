"""CLI contract coverage for the MemoRizz meta-harness."""

from __future__ import annotations

import json

import pytest
import typer
from typer.testing import CliRunner

from memorizz import __version__
from memorizz.approval import SQLiteApprovalStore
from memorizz.cli.app import app
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome


class _CliHarness(AgentHarness):
    name = "cli-fake"

    def __init__(self):
        self.last_agent_id = None

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            usage_reporting=True,
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        self.last_agent_id = task.agent_id
        return AdapterOutcome(
            final_response="CLI harness complete",
            usage={"input_tokens": 10, "output_tokens": 3},
            exit_code=0,
        )


class _Provider:
    def close(self):
        return None


class _AuthRequiredHarness(AgentHarness):
    name = "auth-required"

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            command="auth-required",
            error_code="authentication_required",
            error="The harness API key is not configured.",
            remediation="Set TEST_HARNESS_API_KEY and rerun `memorizz harness doctor auth-required`.",
            metadata={"authentication_configured": False},
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        raise AssertionError("An unauthenticated harness must not start")


@pytest.mark.unit
def test_cli_runs_bounded_harness_and_returns_structured_result(tmp_path, monkeypatch):
    adapter = _CliHarness()
    service = MetaHarness(
        adapters=[adapter],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    monkeypatch.setattr(
        "memorizz.cli.harness_commands._service",
        lambda: (service, _Provider(), []),
    )
    result = CliRunner().invoke(
        app,
        [
            "harness",
            "run",
            "Inspect the workspace",
            "--workspace",
            str(tmp_path),
            "--harness",
            "cli-fake",
            "--agent-id",
            "saved-worker",
            "--mcp-access",
            "none",
            "--max-input-tokens",
            "100",
            "--verify",
            "test -d .",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "succeeded"
    assert payload["verified"] is True
    assert payload["usage"]["input_tokens"] == 10
    assert adapter.last_agent_id == "saved-worker"


@pytest.mark.unit
def test_cli_reports_actionable_missing_harness_authentication(tmp_path, monkeypatch):
    service = MetaHarness(
        adapters=[_AuthRequiredHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    monkeypatch.setattr(
        "memorizz.cli.harness_commands._service",
        lambda: (service, _Provider(), []),
    )

    result = CliRunner().invoke(
        app,
        [
            "harness",
            "run",
            "Inspect the workspace",
            "--workspace",
            str(tmp_path),
            "--harness",
            "auth-required",
            "--mcp-access",
            "none",
            "--json",
        ],
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["error_code"] == "authentication_required"
    assert "API key" in payload["error"]
    assert "TEST_HARNESS_API_KEY" in payload["remediation"]


@pytest.mark.unit
def test_cli_version_uses_source_package_version():
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"memorizz {__version__}"


@pytest.mark.unit
def test_harness_doctor_survives_unavailable_memory_provider(tmp_path, monkeypatch):
    from memorizz.cli import agent_commands

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))

    def unavailable_provider():
        raise RuntimeError("provider failed")

    monkeypatch.setattr(agent_commands, "_provider", unavailable_provider)
    result = CliRunner().invoke(app, ["harness", "doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["harnesses"]
    assert "omits saved MemAgent readiness" in payload["warnings"][0]


@pytest.mark.unit
def test_harness_run_reports_unavailable_memory_provider(tmp_path, monkeypatch):
    from memorizz.cli import agent_commands
    from memorizz.cli.harness_commands import _service

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))

    def unavailable_provider():
        raise RuntimeError("provider failed")

    monkeypatch.setattr(agent_commands, "_provider", unavailable_provider)
    with pytest.raises(typer.BadParameter, match="could not be initialized"):
        _service(require_memory=True)
