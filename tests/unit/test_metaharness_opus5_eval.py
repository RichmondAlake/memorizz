"""Offline contract tests for the Claude Opus 5 MetaHarness regression."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "eval" / "metaharness" / "opus5_small_suite.py"


def namespace():
    return runpy.run_path(str(SCRIPT), run_name="opus5_small_suite_test")


def test_opus5_suite_is_independent_task_native_and_model_explicit() -> None:
    suite = namespace()
    fixtures = suite["load_fixtures"]()
    manifest = suite["protocol_manifest"](
        repo_root=Path(__file__).parents[2],
        fixtures=fixtures,
        seed=20260823,
        codex_model="gpt-5.6-luna",
        claude_model="claude-opus-5",
        maximum_observed_cost_usd=3.0,
    )

    assert len(fixtures) == 4
    assert manifest["models"] == {
        "codex": "gpt-5.6-luna",
        "claude_code": "claude-opus-5",
        "judge": None,
        "coordinator": None,
    }
    assert manifest["controls"]["held_out_task_native_checks_primary"] is True
    assert manifest["controls"]["llm_judge_calls"] == 0
    assert manifest["controls"]["model_effort"] == "medium"
    assert manifest["controls"]["fresh_workspace_per_task_and_strategy"] is True
    assert manifest["paper_comparable"] is False


def test_opus5_suite_routing_is_predeclared_from_risk() -> None:
    suite = namespace()
    fixtures = {item["id"]: item for item in suite["load_fixtures"]()}

    assert suite["routed_harness"](fixtures["numeric_contract"]) == "codex"
    assert suite["routed_harness"](fixtures["retry_schedule"]) == "codex"
    assert suite["routed_harness"](fixtures["tenant_cache"]) == "claude-code"
    assert suite["routed_harness"](fixtures["durable_approval"]) == "claude-code"


def test_opus5_hidden_checks_score_fresh_buggy_fixtures(tmp_path: Path) -> None:
    suite = namespace()
    for fixture in suite["load_fixtures"]():
        workspace = tmp_path / fixture["id"]
        shutil.copytree(fixture["initial"], workspace)
        score = suite["hidden_score"](fixture, workspace)

        assert score["runner_error"] is None
        assert score["total"] == fixture["hidden_check_count"] == 8
        assert 0 < score["passed"] < score["total"]


def test_opus5_frontier_decision_is_fail_closed() -> None:
    suite = namespace()
    base = {
        "wall_latency_ms_total": 100,
        "cost_usd_total": 1.0,
    }
    summaries = [
        {**base, "strategy": "memagent_codex", "accuracy_pct": 75.0},
        {
            **base,
            "strategy": "memagent_opus",
            "accuracy_pct": 100.0,
            "cost_usd_total": 2.0,
        },
        {**base, "strategy": "panel_always", "accuracy_pct": 100.0},
        {
            **base,
            "strategy": "panel_routed",
            "accuracy_pct": 100.0,
            "cost_usd_total": 1.5,
        },
    ]

    decision = suite["evaluate_decision"](summaries)
    assert decision["panel_cost_quality_frontier_supported"] is True

    summaries[-1]["accuracy_pct"] = 99.0
    assert (
        suite["evaluate_decision"](summaries)["panel_cost_quality_frontier_supported"]
        is False
    )
