"""Task-native comparison of MemAgent harness strategies with Claude Opus 5.

The suite is a bounded engineering regression, not a public leaderboard. It
uses four independent synthetic repair tasks, a fresh workspace for every arm,
source-linked Filesystem memory, durable write approval, public verification,
and held-out deterministic checks. No LLM judge is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence

import memorizz
from memorizz import MemAgentBuilder
from memorizz._env_io import load_layered_env
from memorizz.enums.memory_type import MemoryType
from memorizz.metaharness import (
    HarnessBudget,
    HarnessPermissions,
    HarnessTask,
    VerificationSpec,
)
from memorizz.metaharness.models import canonical_hash
from memorizz.metaharness.security import redact, redact_paths

try:
    from . import provider_comparison as legacy
except ImportError:
    try:
        from eval.metaharness import provider_comparison as legacy
    except ImportError:
        import provider_comparison as legacy


SCHEMA_VERSION = "memorizz.metaharness.opus5-small-suite.v1"
PROTOCOL_NAME = "opus5_task_native_small_suite_v1"
USER_ID = "eval-user"
APPROVER_ID = "eval-host"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
DEFAULT_CLAUDE_MODEL = "claude-opus-5"

FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "opus5_small_suite"

STRATEGIES: Dict[str, Dict[str, str]] = {
    "memagent_codex": {
        "display_name": "MemAgent + Codex",
        "kind": "single_codex",
    },
    "memagent_opus": {
        "display_name": "MemAgent + Claude Code",
        "kind": "single_opus",
    },
    "panel_always": {
        "display_name": "MemoRizz Always-on Panel",
        "kind": "sequential_panel",
    },
    "panel_routed": {
        "display_name": "MemoRizz Risk-routed Panel",
        "kind": "risk_router",
    },
}

SHARED_BUDGET = {
    "max_wall_time_seconds": 180,
    "max_steps": 12,
    "max_output_tokens": 12_000,
    "max_retries": 0,
}
CLAUDE_MAX_COST_USD = 0.35
MODEL_EFFORT = "medium"

PRICE_SNAPSHOT = {
    "as_of": "2026-08-23",
    "currency": "USD",
    "note": (
        "Codex cost is estimated from API-equivalent token pricing. "
        "Claude Code cost is reported by its CLI."
    ),
    "per_million_tokens": {
        "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
        "claude-opus-5": {"input": 5.00, "output": 25.00},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision(root: Path) -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def publishable(
    value: Mapping[str, Any], *, repo_root: Path, run_root: Optional[Path] = None
) -> Dict[str, Any]:
    replacements = {
        str(repo_root.resolve()): "[REPOSITORY]",
        str(Path.home().resolve()): "[HOME]",
    }
    if run_root is not None:
        replacements[str(run_root.resolve())] = "[RUN_WORKSPACE]"
    return dict(redact_paths(redact(value), replacements))


def load_fixtures() -> List[Dict[str, Any]]:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    fixtures = []
    for item in manifest["fixtures"]:
        root = FIXTURE_ROOT / item["id"]
        fixture = {
            **item,
            "root": root,
            "requirements": (root / "requirements.md").read_text(encoding="utf-8"),
            "task": (root / "task.md").read_text(encoding="utf-8"),
            "initial": root / "initial",
            "hidden_verify": root / "hidden_verify.py",
        }
        fixtures.append(fixture)
    return fixtures


def routed_harness(fixture: Mapping[str, Any]) -> str:
    """Predeclared rule applied before any model sees the task."""
    if fixture["risk"] in {"security", "tenant_isolation"}:
        return "claude-code"
    return "codex"


def stage_plan(
    strategy: str,
    fixture: Mapping[str, Any],
    codex_model: str,
    claude_model: str,
) -> List[Dict[str, str]]:
    if strategy == "memagent_codex":
        return [{"harness": "codex", "model": codex_model}]
    if strategy == "memagent_opus":
        return [{"harness": "claude-code", "model": claude_model}]
    if strategy == "panel_always":
        return [
            {"harness": "codex", "model": codex_model},
            {"harness": "claude-code", "model": claude_model},
        ]
    harness = routed_harness(fixture)
    return [
        {
            "harness": harness,
            "model": codex_model if harness == "codex" else claude_model,
        }
    ]


def model_configuration(strategy: str, codex_model: str, claude_model: str) -> str:
    if strategy == "memagent_codex":
        return f"Codex: {codex_model} ({MODEL_EFFORT} effort)"
    if strategy == "memagent_opus":
        return f"Claude Code: {claude_model} ({MODEL_EFFORT} effort)"
    if strategy == "panel_always":
        return (
            f"Codex: {codex_model} then Claude Code: {claude_model} "
            f"({MODEL_EFFORT} effort each)"
        )
    return (
        f"Codex: {codex_model} or Claude Code: {claude_model} "
        f"({MODEL_EFFORT} effort)"
    )


def protocol_manifest(
    *,
    repo_root: Path,
    fixtures: Sequence[Mapping[str, Any]],
    seed: int,
    codex_model: str,
    claude_model: str,
    maximum_observed_cost_usd: float,
) -> Dict[str, Any]:
    arms = list(STRATEGIES)
    fixture_rows = []
    task_orders = {}
    for index, fixture in enumerate(fixtures):
        task_orders[fixture["id"]] = random.Random(seed + index).sample(
            arms, k=len(arms)
        )
        fixture_rows.append(
            {
                "id": fixture["id"],
                "title": fixture["title"],
                "risk": fixture["risk"],
                "routed_harness": routed_harness(fixture),
                "requirements_sha256": sha256_file(fixture["root"] / "requirements.md"),
                "initial_tree_hash": canonical_hash(
                    {
                        path.name: sha256_file(path)
                        for path in sorted(fixture["initial"].iterdir())
                    }
                ),
                "hidden_verify_sha256": sha256_file(fixture["hidden_verify"]),
                "hidden_check_count": fixture["hidden_check_count"],
            }
        )
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "profile": "small_regression",
        "paper_comparable": False,
        "models": {
            "codex": codex_model,
            "claude_code": claude_model,
            "judge": None,
            "coordinator": None,
        },
        "memory_provider": "filesystem",
        "strategies": {
            name: {
                **spec,
                "model_configuration": model_configuration(
                    name, codex_model, claude_model
                ),
            }
            for name, spec in STRATEGIES.items()
        },
        "fixtures": fixture_rows,
        "task_orders": task_orders,
        "seed": seed,
        "repeats": 1,
        "controls": {
            "independent_task_count": len(fixtures),
            "fresh_workspace_per_task_and_strategy": True,
            "same_fixture_and_requirements_per_task": True,
            "same_per_harness_resource_budget": True,
            "per_harness_budget": dict(SHARED_BUDGET),
            "claude_per_call_cost_cap_usd": CLAUDE_MAX_COST_USD,
            "model_effort": MODEL_EFFORT,
            "source_linked_memory_context_required": True,
            "durable_host_approval_for_writes": True,
            "public_verification_required": True,
            "held_out_task_native_checks_primary": True,
            "llm_judge_calls": 0,
            "semantic_cache": "disabled_for_cold_comparison",
            "summarization_compaction": "not_triggered_in_single_turn_tasks",
            "maximum_observed_execution_cost_usd": maximum_observed_cost_usd,
        },
        "decision_rule": {
            "quality_metric": "micro_hidden_check_pass_rate",
            "panel_frontier_supported_when": (
                "The risk-routed panel equals or exceeds the better singleton, "
                "strictly exceeds Codex-only, and costs less than Opus-only."
            ),
            "no_post_hoc_weighting": True,
        },
        "claim_boundaries": [
            "This is a four-task synthetic regression, not a public benchmark.",
            "Held-out checks replace a style-sensitive scalar LLM judge.",
            "The always-on panel receives two model calls and more compute.",
            "The risk-routed panel receives one call per task, matching singletons.",
            "Codex cost is API-equivalent; Claude Code cost is CLI-reported.",
            "Filesystem is held constant; provider effects are outside this run.",
            "A positive result supports a bounded frontier claim, not universality.",
        ],
        "implementation": {"runner_sha256": sha256_file(Path(__file__))},
        "reproducibility": {
            "generated_at": utc_now(),
            "memorizz_version": memorizz.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": git_revision(repo_root),
        },
    }
    stable = {key: value for key, value in manifest.items() if key != "reproducibility"}
    manifest["protocol_fingerprint"] = canonical_hash(stable)
    return manifest


def write_fixture(workspace: Path, fixture: Mapping[str, Any]) -> None:
    shutil.copytree(fixture["initial"], workspace)


def seed_scope(
    runtime: Mapping[str, Any],
    *,
    run_scope: str,
    strategy: str,
    fixture: Mapping[str, Any],
) -> Dict[str, Any]:
    provider = runtime["providers"]["filesystem"]
    memory_id = f"opus5-eval-{run_scope}-{fixture['id']}-{strategy}"
    started = time.perf_counter()
    source_id = provider.store(
        {
            "title": f"{fixture['title']} requirements",
            "content": "Applicable implementation requirements:\n"
            + str(fixture["requirements"]),
            "user_id": USER_ID,
            "thread_id": str(fixture["id"]),
        },
        MemoryType.KNOWLEDGE_BASE,
        memory_id=memory_id,
    )
    runtime["scope_ids"]["filesystem"].add(memory_id)
    return {
        "memory_id": memory_id,
        "source_id": str(source_id),
        "seed_latency_ms": int((time.perf_counter() - started) * 1_000),
    }


def actual_model(service: Any, run_id: str, requested_model: str) -> Dict[str, str]:
    for event in service.events(run_id, limit=10_000):
        data = dict(event.get("data") or {})
        if data.get("model"):
            return {
                "requested": requested_model,
                "actual": str(data["model"]),
                "verification": "runtime_reported",
            }
    return {
        "requested": requested_model,
        "actual": requested_model,
        "verification": "cli_argument_pinned",
    }


def permissions_for(harness: str, workspace: Path) -> HarnessPermissions:
    return HarnessPermissions(
        workspace_mode="direct",
        allowed_roots=[str(workspace)],
        network="none",
        mcp_access="none",
        # Claude Code supports a task-level tool allowlist. Codex enforces the
        # same boundary through host-owned sandbox flags and rejects
        # adapter-specific tool lists.
        allowed_tools=(
            ["Read", "Glob", "Grep", "Edit", "Write"]
            if harness == "claude-code"
            else []
        ),
        allowed_env=[],
    )


def task_envelope_preflight(
    runtime: Mapping[str, Any],
    *,
    workspaces: Path,
    codex_model: str,
    claude_model: str,
) -> Dict[str, Any]:
    """Prove write tasks reach durable approval before any model call."""
    service = runtime["services"]["filesystem"]
    rows = {}
    for harness, model in (
        ("codex", codex_model),
        ("claude-code", claude_model),
    ):
        workspace = workspaces / f"preflight-{harness}"
        workspace.mkdir()
        (workspace / "verify.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
        task = HarnessTask(
            task="Validate the host execution envelope without executing a model.",
            workspace=str(workspace),
            harness=harness,
            model=model,
            agent_id=f"preflight-{harness}",
            permissions=permissions_for(harness, workspace),
            budget=HarnessBudget(
                **SHARED_BUDGET,
                max_cost_usd=(
                    CLAUDE_MAX_COST_USD if harness == "claude-code" else None
                ),
            ),
            verification=VerificationSpec(
                command=f'"{sys.executable}" -B verify.py',
                timeout_seconds=30,
                required=True,
            ),
        )
        result = service.run(task)
        proposal_id = str((result.checkpoint or {}).get("proposal_id") or "")
        if result.status.value != "pending_approval" or not proposal_id:
            raise RuntimeError(
                f"{harness} write-envelope preflight did not reach durable approval: "
                f"{result.error_code}: {result.error}"
            )
        service.reject(
            proposal_id,
            approver_id=APPROVER_ID,
            reason="Preflight only; no external model execution was authorized.",
        )
        rows[harness] = {
            "ok": True,
            "status": "pending_approval",
            "model": model,
            "external_calls": 0,
        }
    return {"ok": True, "harnesses": rows}


def run_stage(
    runtime: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    strategy: str,
    stage_number: int,
    harness: str,
    model: str,
    workspace: Path,
    scope: Mapping[str, Any],
) -> Dict[str, Any]:
    provider = runtime["providers"]["filesystem"]
    service = runtime["services"]["filesystem"]
    permissions = permissions_for(harness, workspace)
    budget = HarnessBudget(
        **SHARED_BUDGET,
        max_cost_usd=CLAUDE_MAX_COST_USD if harness == "claude-code" else None,
    )
    verification = VerificationSpec(
        command=f'"{sys.executable}" -B verify.py',
        timeout_seconds=30,
        required=True,
    )
    agent = (
        MemAgentBuilder()
        .with_name(f"{strategy} {fixture['id']} stage {stage_number}")
        .with_memory_provider(provider)
        .with_memory_ids(scope["memory_id"])
        .with_execution_harness(
            harness,
            meta_harness=service,
            config={
                "workspace": str(workspace),
                "model": model,
                "permissions": permissions.to_dict(),
                "budget": budget.to_dict(),
                "verification": verification.to_dict(),
                "metadata": (
                    {"claude_effort": MODEL_EFFORT}
                    if harness == "claude-code"
                    else {"codex_config": [f'model_reasoning_effort="{MODEL_EFFORT}"']}
                ),
            },
        )
        .with_learning_control_plane(
            True,
            {
                "evidence_token_budget": 700,
                "evidence_max_items": 3,
                "compile_async": False,
                "compile_every_n_events": 0,
            },
        )
        .with_semantic_cache(enabled=False)
        .as_ephemeral()
        .build(validate=False)
    )
    runtime["agent_ids"]["filesystem"].add(agent.agent_id)
    task = str(fixture["task"])
    if stage_number > 1:
        task += (
            "\nA previous reviewer attempted the repair. Independently inspect the "
            "current workspace, correct remaining contract violations, and preserve "
            "valid changes."
        )
    started = time.perf_counter()
    try:
        initial = agent.run_on_harness(
            task,
            workspace=str(workspace),
            harness=harness,
            memory_id=scope["memory_id"],
            thread_id=str(fixture["id"]),
            user_id=USER_ID,
            context={
                "evaluation_strategy": strategy,
                "fixture_id": fixture["id"],
                "risk": fixture["risk"],
                "memory_query": fixture["requirements"],
                "memory_context_strategy": "evidence_pack",
            },
            write=True,
            verification_command=verification.command,
        )
        proposal_id = None
        approved = False
        if initial.status.value == "pending_approval":
            proposal_id = str((initial.checkpoint or {}).get("proposal_id") or "")
            if not proposal_id:
                raise RuntimeError("Durable approval omitted proposal_id")
            service.approve(
                proposal_id,
                approver_id=APPROVER_ID,
                reason="User-authorized isolated synthetic evaluation.",
            )
            result = agent.resume_harness_approval(proposal_id)
            approved = True
        else:
            result = initial
        metrics = legacy.normalized_result_metrics(
            service,
            result,
            model=model,
            expected_source_id=scope["source_id"],
        )
        metrics.update(
            {
                "model_identity": actual_model(service, result.run_id, model),
                "wall_latency_ms": int((time.perf_counter() - started) * 1_000),
                "approval": {
                    "required": True,
                    "approved": approved,
                    "proposal_id": proposal_id,
                    "approver_id": APPROVER_ID if approved else None,
                },
                "response_digest": hashlib.sha256(
                    str(result.final_response or "").encode("utf-8")
                ).hexdigest(),
            }
        )
        return metrics
    finally:
        agent.close(close_memory_provider=False)


def hidden_score(fixture: Mapping[str, Any], workspace: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(fixture["hidden_verify"]),
            str(workspace),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"},
    )
    try:
        payload = json.loads((completed.stdout or "").strip().splitlines()[-1])
        checks = list(payload["checks"])
    except Exception as exc:
        return {
            "ok": False,
            "passed": 0,
            "total": int(fixture["hidden_check_count"]),
            "pass_rate": 0.0,
            "checks": [],
            "runner_error": f"{type(exc).__name__}: {exc}",
            "return_code": completed.returncode,
            "stderr": (completed.stderr or "")[-2_000:],
        }
    passed = sum(bool(check.get("passed")) for check in checks)
    return {
        "ok": completed.returncode == 0 and passed == len(checks),
        "passed": passed,
        "total": len(checks),
        "pass_rate": round(100 * passed / len(checks), 3) if checks else 0.0,
        "checks": checks,
        "runner_error": None,
        "return_code": completed.returncode,
        "stderr": (completed.stderr or "")[-2_000:],
    }


def sum_usage(runs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    keys = (
        "input_tokens",
        "cached_tokens",
        "cache_write_input_tokens",
        "output_tokens",
    )
    return {
        key: sum(int((run.get("usage_totals") or {}).get(key, 0) or 0) for run in runs)
        for key in keys
    }


def execute_record(
    runtime: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    strategy: str,
    workspace: Path,
    scope: Mapping[str, Any],
    codex_model: str,
    claude_model: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    plan = stage_plan(strategy, fixture, codex_model, claude_model)
    runs = [
        run_stage(
            runtime,
            fixture=fixture,
            strategy=strategy,
            stage_number=index,
            harness=stage["harness"],
            model=stage["model"],
            workspace=workspace,
            scope=scope,
        )
        for index, stage in enumerate(plan, start=1)
    ]
    score = hidden_score(fixture, workspace)
    costs = [run.get("cost_usd") for run in runs]
    usage = sum_usage(runs)
    record = {
        "fixture_id": fixture["id"],
        "fixture_title": fixture["title"],
        "risk": fixture["risk"],
        "strategy": strategy,
        "display_name": STRATEGIES[strategy]["display_name"],
        "model_configuration": model_configuration(strategy, codex_model, claude_model),
        "memory_provider": "filesystem",
        "source_id": scope["source_id"],
        "seed_latency_ms": scope["seed_latency_ms"],
        "wall_latency_ms": int((time.perf_counter() - started) * 1_000),
        "runs": runs,
        "harness_calls": len(runs),
        "actual_models": [run["model_identity"]["actual"] for run in runs],
        "public_verified": all(bool(run.get("verified")) for run in runs),
        "grounded": all(bool(run.get("grounded")) for run in runs),
        "approvals_consumed": all(
            bool((run.get("approval") or {}).get("approved")) for run in runs
        ),
        "hidden_score": score,
        "cost_usd": (
            round(sum(float(value) for value in costs), 8)
            if all(value is not None for value in costs)
            else None
        ),
        "cost_complete": all(value is not None for value in costs),
        "cost_basis": [run.get("cost_basis") for run in runs],
        "usage_totals": usage,
        "final_workspace_hash": canonical_hash(
            {
                path.name: path.read_text(encoding="utf-8")
                for path in sorted(workspace.glob("*.py"))
            }
        ),
    }
    record["valid"] = bool(
        record["grounded"]
        and record["approvals_consumed"]
        and record["cost_complete"]
        and all(
            run.get("status") in {"succeeded", "verification_failed"} for run in runs
        )
        and score.get("runner_error") is None
    )
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for strategy, spec in STRATEGIES.items():
        rows = [row for row in records if row["strategy"] == strategy]
        passed = sum(int(row["hidden_score"]["passed"]) for row in rows)
        total = sum(int(row["hidden_score"]["total"]) for row in rows)
        costs = [row.get("cost_usd") for row in rows]
        usage = sum_usage([{"usage_totals": row["usage_totals"]} for row in rows])
        summaries.append(
            {
                "strategy": strategy,
                "display_name": spec["display_name"],
                "model_configuration": rows[0]["model_configuration"],
                "memory_provider": "filesystem",
                "task_count": len(rows),
                "hidden_checks_passed": passed,
                "hidden_checks_total": total,
                "accuracy_pct": round(100 * passed / total, 3) if total else 0.0,
                "fully_correct_tasks": sum(
                    bool(row["hidden_score"]["ok"]) for row in rows
                ),
                "public_verified_rate": round(
                    mean(bool(row["public_verified"]) for row in rows), 3
                ),
                "grounded_rate": round(mean(bool(row["grounded"]) for row in rows), 3),
                "wall_latency_ms_total": sum(
                    int(row["wall_latency_ms"]) for row in rows
                ),
                "wall_latency_ms_mean": round(
                    mean(int(row["wall_latency_ms"]) for row in rows)
                ),
                "cost_usd_total": (
                    round(sum(float(value) for value in costs), 8)
                    if all(value is not None for value in costs)
                    else None
                ),
                "cost_coverage": (
                    f"{sum(value is not None for value in costs)}/{len(costs)}"
                ),
                "harness_calls": sum(int(row["harness_calls"]) for row in rows),
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                "actual_models": sorted(
                    {model for row in rows for model in row.get("actual_models", [])}
                ),
            }
        )
    return summaries


def evaluate_decision(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = {row["strategy"]: row for row in summaries}
    codex = rows["memagent_codex"]
    opus = rows["memagent_opus"]
    routed = rows["panel_routed"]
    complete_cost = all(
        row.get("cost_usd_total") is not None for row in (codex, opus, routed)
    )
    quality_floor = max(float(codex["accuracy_pct"]), float(opus["accuracy_pct"]))
    supported = bool(
        complete_cost
        and float(routed["accuracy_pct"]) >= quality_floor
        and float(routed["accuracy_pct"]) > float(codex["accuracy_pct"])
        and float(routed["cost_usd_total"]) < float(opus["cost_usd_total"])
    )
    return {
        "panel_cost_quality_frontier_supported": supported,
        "quality_floor_pct": quality_floor,
        "routed_minus_codex_accuracy_points": round(
            float(routed["accuracy_pct"]) - float(codex["accuracy_pct"]), 3
        ),
        "routed_minus_opus_accuracy_points": round(
            float(routed["accuracy_pct"]) - float(opus["accuracy_pct"]), 3
        ),
        "routed_minus_opus_cost_usd": (
            round(
                float(routed["cost_usd_total"]) - float(opus["cost_usd_total"]),
                8,
            )
            if complete_cost
            else None
        ),
        "routed_minus_opus_latency_ms": (
            int(routed["wall_latency_ms_total"]) - int(opus["wall_latency_ms_total"])
        ),
        "interpretation": (
            "The preregistered MemoRizz frontier rule passed."
            if supported
            else "The preregistered MemoRizz frontier rule did not pass."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        default=truthy(os.getenv("MEMORIZZ_RUN_HARNESS_EVALUATION")),
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-observed-cost-usd", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.gettempdir()) / f"memorizz-opus5-suite-{timestamp}.json"


def write_artifact(
    output: Path,
    artifact: Mapping[str, Any],
    *,
    repo_root: Path,
    run_root: Optional[Path] = None,
) -> None:
    output.write_text(
        json.dumps(
            publishable(artifact, repo_root=repo_root, run_root=run_root),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.max_observed_cost_usd <= 0:
        raise ValueError("--max-observed-cost-usd must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    output = args.output or default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not truthy(os.getenv("MEMORIZZ_EVAL_DISABLE_DOTENV")):
        load_layered_env(extra_paths=[repo_root / ".env"])
    codex_model = os.getenv("MEMORIZZ_EVAL_CODEX_MODEL", DEFAULT_CODEX_MODEL).strip()
    claude_model = os.getenv("MEMORIZZ_EVAL_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
    fixtures = load_fixtures()
    manifest = protocol_manifest(
        repo_root=repo_root,
        fixtures=fixtures,
        seed=args.seed,
        codex_model=codex_model,
        claude_model=claude_model,
        maximum_observed_cost_usd=args.max_observed_cost_usd,
    )
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "dry_run" if not args.execute else "initializing",
        "valid": False,
        "paper_comparable": False,
        "protocol_manifest": manifest,
        "records": [],
        "summaries": [],
        "decision": {},
        "validation_failures": [],
    }
    if not args.execute:
        write_artifact(output, artifact, repo_root=repo_root)
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "external_calls": 0,
                    "task_count": len(fixtures),
                    "arm_count": len(STRATEGIES),
                    "maximum_harness_calls": len(fixtures) * 5,
                    "models": manifest["models"],
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0

    missing = [
        name
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
        if not str(os.getenv(name) or "").strip()
    ]
    if missing:
        artifact.update(
            {
                "status": "error",
                "error": {
                    "code": "authentication_required",
                    "message": "Paid execution requires provider credentials.",
                    "missing_environment": missing,
                    "remediation": (
                        "Inject credentials through the process environment or a "
                        "gitignored local .env file."
                    ),
                },
            }
        )
        write_artifact(output, artifact, repo_root=repo_root)
        return 2

    run_root = Path(tempfile.mkdtemp(prefix="memorizz-opus5-suite-"))
    workspaces = run_root / "workspaces"
    workspaces.mkdir()
    runtime: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    observed_cost = 0.0
    run_scope = uuid.uuid4().hex[:12]
    exit_code = 0
    try:
        runtime = legacy.create_runtime(
            run_root,
            workspaces,
            provider_names=["filesystem"],
        )
        not_ready = {
            name: {
                "error_code": probe.get("error_code"),
                "error": probe.get("error"),
                "remediation": probe.get("remediation"),
            }
            for name, probe in runtime["probes"].items()
            if name in {"codex", "claude-code"} and not probe.get("ready")
        }
        if not_ready:
            raise RuntimeError(
                "Harness preflight failed before paid calls: "
                + json.dumps(not_ready, ensure_ascii=False)
            )
        envelope_preflight = task_envelope_preflight(
            runtime,
            workspaces=workspaces,
            codex_model=codex_model,
            claude_model=claude_model,
        )
        artifact.update(
            {
                "status": "running",
                "provider_preflight": runtime["preflight"],
                "harness_probes": runtime["probes"],
                "task_envelope_preflight": envelope_preflight,
                "pricing": PRICE_SNAPSHOT,
            }
        )
        fixture_by_id = {fixture["id"]: fixture for fixture in fixtures}
        for fixture_id, order in manifest["task_orders"].items():
            fixture = fixture_by_id[fixture_id]
            for strategy in order:
                if observed_cost >= args.max_observed_cost_usd:
                    raise RuntimeError(
                        "Observed cost reached the configured stop before "
                        f"{fixture_id}/{strategy}: USD {observed_cost:.6f}"
                    )
                workspace = workspaces / f"{fixture_id}-{strategy}"
                write_fixture(workspace, fixture)
                scope = seed_scope(
                    runtime,
                    run_scope=run_scope,
                    strategy=strategy,
                    fixture=fixture,
                )
                record = execute_record(
                    runtime,
                    fixture=fixture,
                    strategy=strategy,
                    workspace=workspace,
                    scope=scope,
                    codex_model=codex_model,
                    claude_model=claude_model,
                )
                records.append(record)
                if record.get("cost_usd") is not None:
                    observed_cost += float(record["cost_usd"])
                issues = []
                if not record.get("valid"):
                    issues.append("record_invalid")
                if not record.get("grounded"):
                    issues.append("memory_grounding_missing")
                if not record.get("approvals_consumed"):
                    issues.append("approval_not_consumed")
                if record.get("cost_usd") is None:
                    issues.append("cost_telemetry_incomplete")
                if record["hidden_score"].get("runner_error"):
                    issues.append("hidden_runner_failed")
                if issues:
                    failures.append(
                        {
                            "fixture_id": fixture_id,
                            "strategy": strategy,
                            "issues": issues,
                        }
                    )
                    raise RuntimeError(
                        f"Rejected {fixture_id}/{strategy}: " + ", ".join(issues)
                    )
                print(
                    json.dumps(
                        {
                            "fixture": fixture_id,
                            "strategy": strategy,
                            "models": record["actual_models"],
                            "score": (
                                f"{record['hidden_score']['passed']}/"
                                f"{record['hidden_score']['total']}"
                            ),
                            "calls": record["harness_calls"],
                            "latency_ms": record["wall_latency_ms"],
                            "cost_usd": record["cost_usd"],
                            "observed_cost_usd": round(observed_cost, 6),
                        }
                    ),
                    flush=True,
                )
        expected = len(fixtures) * len(STRATEGIES)
        if len(records) != expected:
            raise RuntimeError(f"Expected {expected} records, found {len(records)}")
        summaries = summarize(records)
        artifact.update(
            {
                "status": "valid",
                "valid": True,
                "records": records,
                "summaries": summaries,
                "decision": evaluate_decision(summaries),
                "validation_failures": [],
                "accounting": {
                    "harness_calls": sum(row["harness_calls"] for row in records),
                    "judge_model_calls": 0,
                    "known_execution_cost_usd": round(observed_cost, 8),
                    "cost_complete": True,
                },
            }
        )
    except Exception as exc:
        exit_code = 1
        artifact.update(
            {
                "status": "error",
                "valid": False,
                "records": records,
                "validation_failures": failures,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(redact(str(exc)))[:2_000],
                },
                "accounting": {
                    "harness_calls": sum(
                        int(row.get("harness_calls", 0)) for row in records
                    ),
                    "judge_model_calls": 0,
                    "known_execution_cost_usd": round(observed_cost, 8),
                    "cost_complete": all(
                        row.get("cost_usd") is not None for row in records
                    ),
                },
            }
        )
    finally:
        if runtime is not None:
            legacy.cleanup_runtime(runtime)
        write_artifact(
            output,
            artifact,
            repo_root=repo_root,
            run_root=run_root,
        )
        shutil.rmtree(run_root, ignore_errors=True)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "valid": artifact["valid"],
                "records": len(records),
                "known_execution_cost_usd": round(observed_cost, 8),
                "output": str(output),
            },
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
