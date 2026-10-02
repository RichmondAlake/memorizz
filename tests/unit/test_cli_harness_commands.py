"""The harness CLI does what the Agent Harnesses page does."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz.cli.app import app
from memorizz.memagent.models import MemAgentModel
from tests.unit.test_harness_parity import _cli, _meta, _provider, _workspace

pytestmark = pytest.mark.unit


def _last_json(result) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_compare_and_plan_take_a_model_per_harness_and_stage(
    tmp_path: Path, monkeypatch
):
    workspace = _workspace(tmp_path)
    runner = _cli(lambda: _meta(tmp_path, "alpha", "beta"), monkeypatch)
    compared = runner.invoke(
        app,
        [
            "harness",
            "compare",
            "Where is the bug?",
            "--harness",
            "alpha",
            "--harness",
            "beta",
            "--harness-model",
            "beta=model-b",
            "--model",
            "shared",
            "--allow-subagents",
            "--mcp-access",
            "none",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert compared.exit_code == 0, compared.output
    body = _last_json(compared)
    models = {run["harness"]: run["task"]["model"] for run in body["runs"]}
    assert models == {"alpha": "shared", "beta": "model-b"}
    assert all(run["task"]["permissions"]["allow_subagents"] for run in body["runs"])

    wrong = runner.invoke(
        app,
        [
            "harness",
            "compare",
            "x",
            "--harness",
            "alpha",
            "--harness",
            "beta",
            "--harness-model",
            "gamma=m",
        ],
    )
    assert wrong.exit_code != 0 and "not a compared --harness" in wrong.output

    planned = runner.invoke(
        app,
        [
            "harness",
            "plan",
            "Ship it",
            "--stage",
            "Plan:alpha",
            "--stage",
            "Review:beta",
            "--stage-model",
            "Review=qwen2.5:3b",
            "--mcp-access",
            "none",
            "--workspace",
            str(workspace),
            "--json",
        ],
    )
    assert planned.exit_code == 0, planned.output
    runs = _last_json(planned)["runs"]
    assert [run["task"]["model"] for run in runs] == [None, "qwen2.5:3b"]


def test_memagent_runs_fill_in_a_saved_agent(tmp_path: Path, monkeypatch):
    provider = _provider(tmp_path)
    saved_id = provider.store_memagent(
        MemAgentModel(name="Research desk", instruction="Lead", llm_config={})
    )
    runner = _cli(lambda: _meta(tmp_path, "memagent", provider=provider), monkeypatch)
    result = runner.invoke(
        app,
        [
            "harness",
            "run",
            "Coordinate",
            "--harness",
            "memagent",
            "--scratch",
            "--mcp-access",
            "none",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "memagent will run Research desk" in result.stderr
    run = json.loads(
        runner.invoke(
            app, ["harness", "show", _last_json(result)["run_id"], "--json"]
        ).stdout
    )["run"]
    assert run["task"]["agent_id"] == saved_id
    provider.close()


def test_runs_workflows_and_conversations_can_be_deleted_rerun_and_continued(
    tmp_path: Path, monkeypatch
):
    workspace = _workspace(tmp_path)
    runner = _cli(lambda: _meta(tmp_path, "alpha", "beta"), monkeypatch)

    first = runner.invoke(
        app,
        [
            "harness",
            "run",
            "Start",
            "--harness",
            "alpha",
            "--mcp-access",
            "none",
            "-C",
            str(workspace),
            "--json",
        ],
    )
    run_id = _last_json(first)["run_id"]

    # A conversation: show it, continue it (setup carried over), delete it.
    shown = runner.invoke(app, ["harness", "conversation", run_id, "--json"])
    assert shown.exit_code == 0, shown.output
    conversation_id = _last_json(shown)["conversation_id"]
    assert conversation_id == f"hxc-{run_id}"
    turned = runner.invoke(
        app, ["harness", "continue", conversation_id, "And then?", "--json"]
    )
    assert turned.exit_code == 0, turned.output
    turn = _last_json(turned)["run"]
    assert turn["status"] == "succeeded" and turn["harness"] == "alpha"
    assert turn["task"]["workspace"] == str(workspace.resolve())
    assert (
        len(
            _last_json(
                runner.invoke(
                    app, ["harness", "conversation", conversation_id, "--json"]
                )
            )["turns"]
        )
        == 2
    )
    gone = runner.invoke(
        app, ["harness", "delete-conversation", conversation_id, "--json"]
    )
    assert gone.exit_code == 0 and len(_last_json(gone)["deleted"]) == 2
    assert (
        runner.invoke(app, ["harness", "conversation", conversation_id]).exit_code == 1
    )

    # A workflow: rerun it, then delete one and see its scratch folder go.
    compared = runner.invoke(
        app,
        [
            "harness",
            "compare",
            "Q",
            "--harness",
            "alpha",
            "--harness",
            "beta",
            "--scratch",
            "--mcp-access",
            "none",
            "--json",
        ],
    )
    workflow = _last_json(compared)["workflow"]
    again = runner.invoke(
        app, ["harness", "rerun-workflow", workflow["orchestration_id"], "--json"]
    )
    assert again.exit_code == 0, again.output
    repeat = _last_json(again)["workflow"]
    assert repeat["orchestration_id"] != workflow["orchestration_id"]
    assert repeat["task"]["metadata"]["rerun_of"] == workflow["orchestration_id"]

    step = workflow["steps"][0]["run_id"]
    kept = runner.invoke(app, ["harness", "delete", step, "--json"])
    assert kept.exit_code == 1 and "a workflow step" in kept.stderr
    deleted = runner.invoke(
        app, ["harness", "delete-workflow", workflow["orchestration_id"], "--json"]
    )
    assert deleted.exit_code == 0, deleted.output
    body = _last_json(deleted)
    assert len(body["deleted"]) == 2 and body["scratch_removed"] == [
        workflow["task"]["workspace"]
    ]
    assert not Path(workflow["task"]["workspace"]).exists()

    single = runner.invoke(
        app,
        [
            "harness",
            "run",
            "x",
            "--harness",
            "beta",
            "--mcp-access",
            "none",
            "-C",
            str(workspace),
            "--json",
        ],
    )
    removed = runner.invoke(
        app, ["harness", "delete", _last_json(single)["run_id"], "--json"]
    )
    assert removed.exit_code == 0 and _last_json(removed)["deleted"]


def test_unknown_ids_end_with_a_message_not_a_traceback(tmp_path: Path, monkeypatch):
    runner = _cli(lambda: _meta(tmp_path, "alpha"), monkeypatch)
    for command in (
        ["harness", "cancel", "nope"],
        ["harness", "retry", "nope"],
        ["harness", "events", "nope"],
        ["harness", "doctor", "nope"],
        ["harness", "approve", "nope", "--approver", "Ada", "--no-resume"],
        ["harness", "reject", "nope", "--approver", "Ada"],
        ["harness", "delete-workflow", "nope"],
        ["harness", "rerun-workflow", "nope"],
        ["harness", "models", "nope"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 1, (command, result.output)
        assert "Traceback" not in result.output and result.stderr.strip(), command


def test_events_follow_and_models(tmp_path: Path, monkeypatch):
    runner = _cli(lambda: _meta(tmp_path, "alpha"), monkeypatch)
    run_id = _last_json(
        runner.invoke(
            app,
            [
                "harness",
                "run",
                "x",
                "--harness",
                "alpha",
                "--scratch",
                "--mcp-access",
                "none",
                "--json",
            ],
        )
    )["run_id"]
    followed = runner.invoke(app, ["harness", "events", run_id, "--follow"])
    assert followed.exit_code == 0
    kinds = [json.loads(line)["type"] for line in followed.stdout.strip().splitlines()]
    assert "message" in kinds and kinds[-1] == "complete"
    limited = runner.invoke(
        app, ["harness", "events", run_id, "--limit", "1", "--json"]
    )
    assert _last_json(limited)["count"] == 1

    with patch(
        "memorizz.llms.model_lists.available_providers", return_value=["ollama"]
    ), patch(
        "memorizz.llms.model_lists.latest_models_for",
        return_value={"ollama": ["qwen2.5:7b"]},
    ):
        listed = runner.invoke(app, ["harness", "models", "--json"])
    assert listed.exit_code == 0, listed.output
    assert "alpha" in _last_json(listed)["models"]


def test_harness_delegates_can_be_created_and_attached(tmp_path: Path, monkeypatch):
    provider = _provider(tmp_path)
    lead_id = provider.store_memagent(
        MemAgentModel(
            name="Lead",
            instruction="Lead",
            llm_config={"provider": "anthropic", "model": "claude-sonnet-5-5"},
        )
    )
    runner = _cli(lambda: _meta(tmp_path, "codex", provider=provider), monkeypatch)
    with patch("memorizz.llms.model_lists.available_providers", return_value=[]), patch(
        "memorizz.llms.model_lists.latest_models_for", return_value={}
    ):
        options = runner.invoke(app, ["harness", "delegate", "options", "--json"])
    assert options.exit_code == 0, options.output
    assert [row["name"] for row in _last_json(options)["harnesses"]] == ["codex"]

    created = runner.invoke(
        app,
        [
            "harness",
            "delegate",
            "create",
            "--harness",
            "codex",
            "--model",
            "gpt-6-luna",
            "--coordinator",
            lead_id,
            "--attach",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    agent = _last_json(created)["agent"]
    assert agent["runs_on"] == {
        "harness": "codex",
        "label": "Codex",
        "model": "gpt-6-luna",
    }
    assert agent["model"] == "claude-sonnet-5-5"  # borrowed from the coordinator
    saved = provider.retrieve_memagent(agent["id"])
    assert saved.meta_harness_mode == "runtime" and saved.default_harness == "codex"
    assert provider.retrieve_memagent(lead_id).delegates == [agent["id"]]

    refused = runner.invoke(app, ["harness", "delegate", "create", "--harness", "word"])
    assert refused.exit_code == 1 and "Choose a harness" in refused.stderr
    lonely = runner.invoke(
        app, ["harness", "delegate", "create", "--harness", "codex", "--attach"]
    )
    assert lonely.exit_code != 0 and "--coordinator" in lonely.output
    provider.close()
