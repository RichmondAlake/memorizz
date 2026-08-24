import pytest

from memorizz.benchmarks.terminal_bench_forecast import (
    forecast_terminal_bench,
    wilson_interval,
)


def test_reference_forecast_withholds_accuracy_and_stays_below_budget():
    report = forecast_terminal_bench(
        per_trial_spend_guard_usd=1.75,
        total_budget_usd=1_000,
    )

    assert report["submitted"] is False
    assert report["paper_comparable"] is False
    assert report["official_run_requirements"]["minimum_total_trials"] == 445
    assert report["cost_forecast"]["nominal_guard_total_usd"] == 778.75
    assert report["cost_forecast"]["point_usd"] == 421.15
    assert report["accuracy_forecast"]["status"] == "withheld"


def test_pilot_forecast_reports_uncertainty_and_inferred_rank():
    trials = [
        {
            "task_id": f"task-{index}",
            "reward": 1 if index < 8 else 0,
            "cost_usd": 0.5 + index / 100,
            "duration_seconds": 60,
        }
        for index in range(10)
    ]

    report = forecast_terminal_bench(pilot_trials=trials)

    accuracy = report["accuracy_forecast"]
    assert accuracy["status"] == "pilot_forecast"
    assert accuracy["point_percent"] == 80.0
    assert accuracy["wilson_95_percent"][0] < 80 < accuracy["wilson_95_percent"][1]
    assert accuracy["rank_context"]["inferred_point_rank"] == 4
    assert report["cost_forecast"]["basis"] == "pilot_mean"


def test_wilson_interval_requires_observations():
    with pytest.raises(ValueError, match="positive"):
        wilson_interval(0, 0)
