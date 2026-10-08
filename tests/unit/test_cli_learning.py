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


def test_retention_plan_apply_and_restore_through_the_cli(tmp_path, monkeypatch):
    import time

    from memorizz.enums import MemoryType

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("MEMORIZZ_RETENTION_MIN_SCORE", "0.3")
    provider = FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory"))
    provider.store(
        {
            "_id": "stale-fact",
            "name": "stale-fact",
            "content": "never recalled",
            "agent_id": "agent-cli",
            "memory_id": "memory-cli",
            "timestamp": time.time() - 120 * 86_400,
            "importance": 0.1,
            "embedding": [1.0, 0.0],
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    common = [
        "--agent-id",
        "agent-cli",
        "--memory-id",
        "memory-cli",
        "--backend",
        "filesystem",
        "--json",
    ]
    environment = {
        "MEMORIZZ_HOME": str(tmp_path),
        "MEMORIZZ_RETENTION_MIN_SCORE": "0.3",
    }

    planned = runner.invoke(
        app, ["learning", "retention-plan", *common], env=environment
    )
    assert planned.exit_code == 0, planned.output
    report = json.loads(planned.output)
    assert report["plan_kind"] == "retention" and report["candidate_count"] == 1
    plan_id = report["plan_id"]

    applied = runner.invoke(
        app,
        [
            "learning",
            "retention-apply",
            plan_id,
            "--approved-by",
            "operator-7",
            *common,
        ],
        env=environment,
    )
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["tombstoned"] == 1
    assert (
        provider.retrieve_by_id("stale-fact", MemoryType.KNOWLEDGE_BASE)[
            "retention_state"
        ]
        == "suppressed"
    )

    listed = runner.invoke(
        app,
        [
            "learning",
            "suppressed",
            "--agent-id",
            "agent-cli",
            "--backend",
            "filesystem",
            "--json",
        ],
        env=environment,
    )
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)["count"] == 1

    restored = runner.invoke(
        app,
        [
            "learning",
            "unsuppress",
            "stale-fact",
            "--memory-type",
            "knowledge_base",
            "--agent-id",
            "agent-cli",
            "--approved-by",
            "operator-7",
            "--backend",
            "filesystem",
            "--json",
        ],
        env=environment,
    )
    assert restored.exit_code == 0, restored.output
    assert json.loads(restored.output)["restored"] is True
    assert (
        provider.retrieve_by_id("stale-fact", MemoryType.KNOWLEDGE_BASE)[
            "retention_state"
        ]
        == "active"
    )
