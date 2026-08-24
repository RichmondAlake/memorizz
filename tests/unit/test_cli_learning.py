"""CLI coverage for the learning control-plane operator surface."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memorizz.cli.app import app
from memorizz.learning import LearningControlPlane
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

runner = CliRunner()


@pytest.mark.unit
def test_learning_status_events_and_compile_use_the_filesystem_backend(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory"))
    plane = LearningControlPlane(
        provider,
        agent_id="agent-cli",
        config={"enabled": True, "compile_async": False},
    )
    plane.begin_run(
        "remember this",
        scope={
            "memory_id": "memory-cli",
            "user_id": "user-cli",
            "thread_id": "thread-cli",
            "run_id": "run-cli",
        },
    )
    plane.close()

    common = [
        "--agent-id",
        "agent-cli",
        "--memory-id",
        "memory-cli",
        "--user-id",
        "user-cli",
        "--thread-id",
        "thread-cli",
        "--backend",
        "filesystem",
        "--json",
    ]
    environment = {"MEMORIZZ_HOME": str(tmp_path)}

    status = runner.invoke(app, ["learning", "status", *common], env=environment)
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["records"]["record_count"] == 1

    events = runner.invoke(app, ["learning", "events", *common], env=environment)
    assert events.exit_code == 0, events.output
    assert json.loads(events.output)["events"][0]["event_type"] == "run_started"

    compiled = runner.invoke(app, ["learning", "compile", *common], env=environment)
    assert compiled.exit_code == 0, compiled.output
    assert json.loads(compiled.output)["compiled_events"] == 1
