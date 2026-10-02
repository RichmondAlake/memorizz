"""CLI coverage for protocol-aware Evalground commands."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from memorizz.cli.app import app
from tests.cli_text import plain


@pytest.mark.unit
def test_eval_protocol_show_is_machine_readable():
    result = CliRunner().invoke(app, ["eval", "protocol", "show", "beam"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["benchmark_id"] == "beam"
    assert payload["protocol_version"]
    assert payload["synchronization_revision"]


@pytest.mark.unit
def test_eval_dataset_verify_reports_checksums(tmp_path):
    (tmp_path / "locomo10.json").write_text("[]", encoding="utf-8")
    (tmp_path / "locomo_plus.json").write_text("[]", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "dataset",
            "verify",
            "locomo-plus",
            "--data-path",
            str(tmp_path),
            "--variant",
            "cognitive",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert len(payload["artifacts"]) == 2


@pytest.mark.unit
def test_eval_run_forwards_profile_provider_and_oracle_lane(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    data_path.mkdir()
    captured = {}

    def fake_run(benchmark_id, configured_data, **kwargs):
        captured.update(
            {"benchmark_id": benchmark_id, "data_path": configured_data, **kwargs}
        )
        return {
            "benchmark": benchmark_id,
            "protocol": {"profile": {"name": kwargs["profile"]}},
            "comparison_label": "Diagnostic",
            "paper_comparable": False,
            "overall_score": 0.75,
            "retrieval": {"recall_at_k": 0.5},
            "answer_quality": {"gold_evidence_oracle_score": None},
            "usage": {"cost_usd": 0.0},
        }

    monkeypatch.setattr("memorizz.benchmarks.memory_suite.run_memory_suite", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "locomo-plus",
            "--data-path",
            str(data_path),
            "--profile",
            "regression",
            "--limit",
            "12",
            "--memory-provider",
            "filesystem",
            "--no-oracle-reader",
            "--reasoning-effort",
            "medium",
            "--max-output-tokens",
            "768",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["comparison_label"] == "Diagnostic"
    assert captured["profile"] == "regression"
    assert captured["limit"] == 12
    assert captured["memory_backend"] == "filesystem"
    assert captured["oracle_reader"] is False
    assert captured["reasoning_effort"] == "medium"
    assert captured["max_output_tokens"] == 768


@pytest.mark.unit
def test_eval_run_forwards_full_memagent_controls(tmp_path, monkeypatch):
    data_path = tmp_path / "data"
    data_path.mkdir()
    template_path = tmp_path / "agent-template.json"
    template_path.write_text(
        json.dumps({"agent_id": "saved-agent", "name": "Saved agent"}),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(benchmark_id, configured_data, **kwargs):
        captured.update(kwargs)
        return {
            "benchmark": benchmark_id,
            "protocol": {"profile": {"name": kwargs["profile"]}},
            "comparison_label": "Diagnostic",
            "paper_comparable": False,
            "overall_score": 1.0,
            "retrieval": {"recall_at_k": 1.0},
            "answer_quality": {"gold_evidence_oracle_score": 1.0},
            "usage": {"cost_usd": 0.0},
        }

    monkeypatch.setattr("memorizz.benchmarks.memory_suite.run_memory_suite", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "locomo-plus",
            "--data-path",
            str(data_path),
            "--evaluation-mode",
            "memagent",
            "--agent-template",
            str(template_path),
            "--rerank-weight",
            "0.2",
            "--no-query-expansion",
            "--no-reader-repair",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["evaluation_mode"] == "memagent"
    assert captured["agent_template"].agent_id == "saved-agent"
    assert captured["rerank_weight"] == pytest.approx(0.2)
    assert captured["query_expansion"] is False
    assert captured["reader_repair"] is False


@pytest.mark.unit
def test_terminal_bench_forecast_cli_writes_unofficial_report(tmp_path):
    output = tmp_path / "forecast.json"

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "terminal-bench",
            "forecast",
            "--per-trial-spend-guard-usd",
            "1.75",
            "--total-budget-usd",
            "1000",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["result_kind"] == "unofficial_forecast"
    assert payload["accuracy_forecast"]["status"] == "withheld"
    assert payload["cost_forecast"]["nominal_guard_total_usd"] == 778.75
    assert json.loads(output.read_text(encoding="utf-8")) == payload


@pytest.mark.unit
def test_cli_terminal_bench_tasks_status_and_run(tmp_path):
    from typer.testing import CliRunner

    from memorizz.cli.app import app

    cli = CliRunner()
    listed = cli.invoke(app, ["eval", "terminal-bench", "tasks", "--compact"])
    assert listed.exit_code == 0, listed.output
    body = json.loads(listed.stdout)
    assert body["version"] and body["count"] == len(body["tasks"]) > 0
    assert all(not task.get("gpus") for task in body["tasks"])
    one = body["tasks"][0]
    narrowed = cli.invoke(
        app,
        ["eval", "terminal-bench", "tasks", "--category", one["category"], "--compact"],
    )
    assert {task["category"] for task in json.loads(narrowed.stdout)["tasks"]} == {
        one["category"]
    }

    with patch(
        "memorizz.benchmarks.terminal_bench_runner.environment_status",
        return_value={"ready": False, "docker": {"ok": False}},
    ):
        status = cli.invoke(app, ["eval", "terminal-bench", "status", "--compact"])
    assert status.exit_code == 0 and json.loads(status.stdout)["ready"] is False

    seen = []
    with patch(
        "memorizz.benchmarks.terminal_bench_runner.main",
        side_effect=lambda argv: seen.append(argv) or 0,
    ):
        ran = cli.invoke(
            app,
            [
                "eval",
                "terminal-bench",
                "run",
                "--harness",
                "codex",
                "--task",
                f"{one['name']}",
                "--output",
                str(tmp_path / "out.json"),
                "--codex-auth",
                "chatgpt",
                "--dry-run",
            ],
        )
    assert ran.exit_code == 0, ran.output
    argv = seen[0]
    assert argv[argv.index("--tasks") + 1] == one["name"] and "--dry-run" in argv
    assert argv[argv.index("--codex-auth") + 1] == "chatgpt"

    # Bad options from the runner end the command with its exit code.
    refused = cli.invoke(
        app,
        [
            "eval",
            "terminal-bench",
            "run",
            "--harness",
            "codex",
            "--task",
            "not-a-task",
            "--output",
            str(tmp_path / "x.json"),
        ],
    )
    assert refused.exit_code == 2


@pytest.mark.unit
def test_cli_eval_compare_runs_a_comparison_config(tmp_path):
    cli = CliRunner()
    missing = cli.invoke(app, ["eval", "compare", str(tmp_path / "nope.json")])
    assert (
        missing.exit_code == 2
        and "Unable to load the comparison config" in missing.output
    )
    config = tmp_path / "comparison.json"
    config.write_text("{}", encoding="utf-8")
    seen = []
    with patch(
        "memorizz.benchmarks.comparison.ComparisonConfig.model_validate_json",
        return_value="parsed",
    ), patch(
        "memorizz.benchmarks.comparison.run_comparison",
        side_effect=lambda parsed, folder: seen.append((parsed, folder)),
    ):
        ran = cli.invoke(app, ["eval", "compare", str(config)])
    assert ran.exit_code == 0, ran.output
    assert seen == [("parsed", tmp_path.resolve())]


@pytest.mark.unit
def test_eval_run_takes_a_saved_agent_and_an_ollama_host(tmp_path, monkeypatch):
    env = {
        "MEMORIZZ_HOME": str(tmp_path / "home"),
        "MEMORIZZ_BACKEND": "",
        "MEMORIZZ_DEFAULT_LLM_PROVIDER": "",
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "OLLAMA_HOST": "127.0.0.1:1",
    }
    created = CliRunner().invoke(
        app, ["agents", "create", "--name", "Under test", "--no-llm", "--json"], env=env
    )
    assert created.exit_code == 0, created.output
    agent_id = json.loads(created.output)["agent"]["agent_id"]
    data_path = tmp_path / "data"
    data_path.mkdir()
    captured = {}

    def fake_run(benchmark_id, configured_data, **kwargs):
        captured.update(kwargs)
        return {
            "benchmark": benchmark_id,
            "protocol": {"profile": {"name": kwargs["profile"]}},
            "comparison_label": "Diagnostic",
            "paper_comparable": False,
            "overall_score": 1.0,
            "retrieval": {"recall_at_k": 1.0},
            "answer_quality": {"gold_evidence_oracle_score": 1.0},
            "usage": {"cost_usd": 0.0},
        }

    monkeypatch.setattr("memorizz.benchmarks.memory_suite.run_memory_suite", fake_run)
    base = [
        "eval",
        "run",
        "locomo-plus",
        "--data-path",
        str(data_path),
        "--output",
        str(tmp_path / "r.json"),
        "--force",
    ]
    result = CliRunner().invoke(
        app, [*base, "--evaluation-mode", "memagent", "--agent-id", agent_id], env=env
    )
    assert result.exit_code == 0, result.output
    template = captured["agent_template"]
    assert (
        template.name == "Under test"
        and template.tools == []
        and template.meta_harness is False
    )
    assert (
        captured["ollama_host"] == "http://127.0.0.1:1"
    )  # OLLAMA_HOST, given a scheme

    result = CliRunner().invoke(
        app, [*base, "--ollama-host", "http://gpu-box:11434"], env=env
    )
    assert result.exit_code == 0 and captured["ollama_host"] == "http://gpu-box:11434"
    missing = CliRunner().invoke(
        app, [*base, "--evaluation-mode", "memagent", "--agent-id", "nope"], env=env
    )
    assert missing.exit_code != 0 and "Agent not found" in missing.output
    wrong_mode = CliRunner().invoke(app, [*base, "--agent-id", agent_id], env=env)
    assert wrong_mode.exit_code != 0 and "needs --evaluation-mode memagent" in plain(
        wrong_mode.output
    )
