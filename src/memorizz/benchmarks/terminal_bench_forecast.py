"""Budget and rank forecasting for Terminal-Bench 2.1 calibration runs.

Forecasts are intentionally separate from Harbor execution.  They cannot be
mistaken for submitted results and never turn a small pilot into a leaderboard
claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

TERMINAL_BENCH_21_DATASET_SHA256 = (
    "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
)
TERMINAL_BENCH_21_TASKS = 89
TERMINAL_BENCH_21_MIN_TRIALS_PER_TASK = 5

# A dated comparison snapshot, not a claim that MemoRizz achieved these scores.
# Users can replace it with a newer snapshot when forecasting a later pilot.
LEADERBOARD_SNAPSHOT_2026_08_23: tuple[dict[str, Any], ...] = (
    {
        "rank": 1,
        "agent": "Claude Code",
        "model": "Fable 5",
        "accuracy_percent": 83.8,
        "cost_usd": 552.67,
    },
    {
        "rank": 2,
        "agent": "Codex",
        "model": "GPT-5.5",
        "accuracy_percent": 83.1,
        "cost_usd": 2059.19,
    },
    {
        "rank": 3,
        "agent": "Terminus2",
        "model": "Fable 5",
        "accuracy_percent": 80.4,
        "cost_usd": 438.64,
    },
    {
        "rank": 4,
        "agent": "Cursor CLI",
        "model": "Grok 4.5",
        "accuracy_percent": 79.3,
        "cost_usd": 134.09,
    },
    {
        "rank": 5,
        "agent": "Claude Code",
        "model": "Opus 4.8",
        "accuracy_percent": 78.9,
        "cost_usd": 286.94,
    },
    {
        "rank": 6,
        "agent": "Codex",
        "model": "GPT-5.6 Terra",
        "accuracy_percent": 78.4,
        "cost_usd": 421.15,
    },
    {
        "rank": 10,
        "agent": "Claude Code",
        "model": "Sonnet 5",
        "accuracy_percent": 74.6,
        "cost_usd": None,
    },
)


def wilson_interval(
    successes: int, total: int, *, z: float = 1.959964
) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a Bernoulli pilot."""

    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _trial_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        rows = value
    elif isinstance(value, Mapping):
        rows = value.get("trials") or value.get("results") or []
    else:
        rows = []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def load_pilot_trials(path: str | Path) -> list[dict[str, Any]]:
    """Load the small, normalized pilot format documented by the CLI."""

    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    rows = _trial_rows(value)
    if not rows:
        raise ValueError("Pilot JSON must contain a non-empty trials/results array")
    return rows


def _number(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _rank_for_accuracy(accuracy_percent: float) -> dict[str, Any]:
    better = sum(
        1
        for row in LEADERBOARD_SNAPSHOT_2026_08_23
        if float(row["accuracy_percent"]) > accuracy_percent
    )
    return {
        "inferred_point_rank": better + 1,
        "basis": "dated_public_snapshot_anchors",
        "caveat": (
            "This is an interpolation against selected live rows, not an official "
            "rank or a leaderboard submission. Ties and omitted rows can change it."
        ),
    }


def forecast_terminal_bench(
    *,
    pilot_trials: Iterable[Mapping[str, Any]] = (),
    model: str = "openai/gpt-5.6-terra",
    per_trial_spend_guard_usd: float = 1.75,
    total_budget_usd: float = 1_000.0,
    reference_full_run_cost_usd: float = 421.15,
) -> dict[str, Any]:
    """Build a fail-closed full-run forecast from a small pilot or reference."""

    rows = [dict(row) for row in pilot_trials]
    required_trials = TERMINAL_BENCH_21_TASKS * TERMINAL_BENCH_21_MIN_TRIALS_PER_TASK
    guard = max(0.01, float(per_trial_spend_guard_usd))
    budget = max(0.01, float(total_budget_usd))
    nominal_guard_total = guard * required_trials
    costs = [
        value
        for row in rows
        if (value := _number(row, "cost_usd", "estimated_cost_usd")) is not None
    ]
    durations = [
        value
        for row in rows
        if (value := _number(row, "duration_seconds", "elapsed_seconds")) is not None
    ]
    task_names = {
        str(row.get("task_name") or row.get("task_id") or "").strip()
        for row in rows
        if str(row.get("task_name") or row.get("task_id") or "").strip()
    }
    rewards = [
        value for row in rows if (value := _number(row, "reward", "score")) is not None
    ]

    if costs:
        point_cost = mean(costs) * required_trials
        cost_basis = "pilot_mean"
        scenario_low = min(costs) * required_trials
        scenario_high = max(costs) * required_trials
    else:
        point_cost = max(0.0, float(reference_full_run_cost_usd))
        cost_basis = "external_codex_terra_reference"
        scenario_low = point_cost * 0.75
        scenario_high = point_cost * 1.5

    accuracy: dict[str, Any]
    if len(rewards) >= 10 and len(task_names) >= 10:
        successes = sum(value >= 1.0 for value in rewards)
        low, high = wilson_interval(successes, len(rewards))
        point = successes / len(rewards)
        accuracy = {
            "status": "pilot_forecast",
            "successes": successes,
            "trials": len(rewards),
            "unique_tasks": len(task_names),
            "point_percent": round(point * 100, 3),
            "wilson_95_percent": [round(low * 100, 3), round(high * 100, 3)],
            "rank_context": _rank_for_accuracy(point * 100),
        }
    else:
        accuracy = {
            "status": "withheld",
            "trials": len(rewards),
            "unique_tasks": len(task_names),
            "reason": (
                "At least 10 distinct tasks are required before MemoRizz emits an "
                "accuracy or rank forecast. A cost reference is not performance evidence."
            ),
        }

    return {
        "schema_version": "1.0",
        "benchmark": "terminal-bench@2.1",
        "result_kind": "unofficial_forecast",
        "submitted": False,
        "paper_comparable": False,
        "model": model,
        "official_run_requirements": {
            "dataset_sha256": TERMINAL_BENCH_21_DATASET_SHA256,
            "tasks": TERMINAL_BENCH_21_TASKS,
            "minimum_trials_per_task": TERMINAL_BENCH_21_MIN_TRIALS_PER_TASK,
            "minimum_total_trials": required_trials,
            "timeout_or_resource_overrides_allowed": False,
            "failed_and_error_trials_count_as_zero": True,
            "atif_trajectory_required_for_successful_trials": True,
        },
        "pilot": {
            "trial_count": len(rows),
            "unique_task_count": len(task_names),
            "measured_cost_count": len(costs),
            "measured_duration_count": len(durations),
        },
        "cost_forecast": {
            "basis": cost_basis,
            "point_usd": round(point_cost, 2),
            "scenario_usd": [round(scenario_low, 2), round(scenario_high, 2)],
            "per_trial_spend_guard_usd": guard,
            "nominal_guard_total_usd": round(nominal_guard_total, 2),
            "configured_total_budget_usd": budget,
            "within_budget_before_in_flight_overshoot": nominal_guard_total < budget,
            "headroom_usd": round(budget - nominal_guard_total, 2),
            "caveat": (
                "The guard is checked between model calls. An in-flight final call can "
                "overshoot a trial's guard, so the nominal total is not a hard invoice cap."
            ),
        },
        "latency_forecast": {
            "serial_hours": (
                round(mean(durations) * required_trials / 3600.0, 2)
                if durations
                else None
            ),
            "basis": "pilot_mean" if durations else "not_measured",
        },
        "accuracy_forecast": accuracy,
        "leaderboard_context": {
            "snapshot_date": "2026-08-23",
            "rows": [dict(row) for row in LEADERBOARD_SNAPSHOT_2026_08_23],
            "submission_status": (
                "Community submissions are closed; current additions are "
                "maintainer-run only."
            ),
        },
        "recommended_next_step": (
            "Run 10 stratified tasks once with the same model, reasoning effort, "
            "resources, and guard; then regenerate this forecast before any 445-trial run."
        ),
    }


__all__ = [
    "LEADERBOARD_SNAPSHOT_2026_08_23",
    "TERMINAL_BENCH_21_DATASET_SHA256",
    "forecast_terminal_bench",
    "load_pilot_trials",
    "wilson_interval",
]
