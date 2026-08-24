"""CLI coverage for protocol-aware Evalground commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from memorizz.cli.app import app


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
