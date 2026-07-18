# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tests for the `/automation` REPL command.

These mock the provider-selected store and runner, so no database is required.
"""

import types
from unittest.mock import patch

import pytest

from memorizz.cli import commands

STORE = "memorizz.automation.store.factory.get_automation_store"
RUNNER = "memorizz.automation.runner.run_job_once"


class _Console:
    def __init__(self):
        self.out = []

    def print(self, *a, **k):
        self.out.append(" ".join(str(x) for x in a))

    def text(self):
        return "\n".join(self.out)


def _session(provider="fs-provider", agent_id="a1"):
    agent = types.SimpleNamespace(memory_provider=provider, agent_id=agent_id)
    return types.SimpleNamespace(console=_Console(), agent=agent)


def _job(job_id="j1", agent_id="a1", name="Daily digest"):
    return types.SimpleNamespace(
        job_id=job_id,
        agent_id=agent_id,
        name=name,
        enabled=True,
        schedule_type="interval",
        next_run_at="2026-06-22T00:00:00Z",
        retry_max_attempts=1,
    )


class _Store:
    def __init__(self, jobs):
        self.jobs = jobs
        self.finished = []

    def list_jobs(self, agent_id=None, enabled=None):
        return [j for j in self.jobs if j.agent_id == agent_id]

    def get_job(self, jid):
        return next((j for j in self.jobs if j.job_id == jid), None)

    def start_run(self, job, now, worker_id):
        return types.SimpleNamespace(run_id="run1")

    def finish_run(self, run_id, **kw):
        self.finished.append({"run_id": run_id, **kw})


@pytest.mark.unit
def test_requires_supported_backend():
    s = _session()
    with patch(STORE, return_value=None):
        commands.cmd_automation(s, "")
    output = s.console.text()
    assert "does not support automations" in output
    assert "filesystem, MongoDB, or Oracle" in output


@pytest.mark.unit
def test_lists_only_this_agents_jobs():
    s = _session()
    store = _Store([_job("j1", "a1", "Morning brief"), _job("j2", "other", "Theirs")])
    with patch(STORE, return_value=store):
        commands.cmd_automation(s, "")
    out = s.console.text()
    assert "j1" in out and "Morning brief" in out
    assert "j2" not in out  # other agent's job filtered out


@pytest.mark.unit
def test_run_executes_and_prints_output():
    s = _session()
    store = _Store([_job("j1", "a1", "Daily")])
    payload = {
        "response": "AUTOMATION RESULT",
        "delivery_summary": {"sent": 1, "failed": 0, "total": 1},
    }
    with patch(STORE, return_value=store), patch(RUNNER, return_value=payload) as rjo:
        commands.cmd_automation(s, "run j1")
    out = s.console.text()
    assert "AUTOMATION RESULT" in out
    assert rjo.called
    assert store.finished and store.finished[-1]["status"] == "succeeded"


@pytest.mark.unit
def test_run_unknown_id():
    s = _session()
    with patch(STORE, return_value=_Store([_job("j1", "a1")])):
        commands.cmd_automation(s, "run nope")
    assert "No automation 'nope'" in s.console.text()


@pytest.mark.unit
def test_run_single_job_without_id():
    s = _session()
    store = _Store([_job("only", "a1", "Solo")])
    with patch(STORE, return_value=store), patch(
        RUNNER, return_value={"response": "ran solo"}
    ):
        commands.cmd_automation(s, "run")
    assert "ran solo" in s.console.text()


@pytest.mark.unit
def test_run_failure_marks_run_failed():
    s = _session()
    store = _Store([_job("j1", "a1")])
    with patch(STORE, return_value=store), patch(
        RUNNER, side_effect=RuntimeError("boom")
    ):
        commands.cmd_automation(s, "run j1")
    assert "Automation failed" in s.console.text()
    assert store.finished and store.finished[-1]["status"] == "failed"
