"""Factorial MetaHarness evaluation with explicit causal comparison pairs.

This protocol answers three separate systems questions on one controlled fixture:

1. wrapper overhead: direct harness vs the same harness through MemAgent;
2. coordination value: both harnesses always vs an adaptive fallback panel; and
3. provider sensitivity: every strategy on Filesystem and Oracle.

The protocol is intentionally diagnostic rather than paper-comparable. It never
turns a one-fixture engineering experiment into a general harness leaderboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import memorizz
from memorizz import MemAgentBuilder
from memorizz._env_io import load_layered_env
from memorizz.metaharness import HarnessPermissions, VerificationSpec
from memorizz.metaharness.models import canonical_hash
from memorizz.metaharness.security import redact, redact_paths

try:
    from . import provider_comparison as legacy
except ImportError:  # direct script execution
    try:
        from eval.metaharness import provider_comparison as legacy
    except ImportError:
        import provider_comparison as legacy


SCHEMA_VERSION = "memorizz.metaharness.factorial-comparison.v1"
PROTOCOL_NAME = "paired_factorial_v1"
USER_ID = "eval-user"

STRATEGIES: Dict[str, Dict[str, Any]] = {
    "direct_codex": {
        "display_name": "Direct MetaHarness → Codex",
        "kind": "direct",
        "harness": "codex",
        "call_policy": "exactly_one_codex",
    },
    "memagent_codex": {
        "display_name": "MemAgent → Codex",
        "kind": "wrapper",
        "harness": "codex",
        "call_policy": "exactly_one_codex",
    },
    "direct_claude": {
        "display_name": "Direct MetaHarness → Claude Code",
        "kind": "direct",
        "harness": "claude-code",
        "call_policy": "exactly_one_claude",
    },
    "memagent_claude": {
        "display_name": "MemAgent → Claude Code",
        "kind": "wrapper",
        "harness": "claude-code",
        "call_policy": "exactly_one_claude",
    },
    "panel_full": {
        "display_name": "MemAgent Full Panel",
        "kind": "panel",
        "adaptive": False,
        "call_policy": "exactly_one_codex_and_one_claude",
    },
    "panel_adaptive": {
        "display_name": "MemAgent Adaptive Panel",
        "kind": "panel",
        "adaptive": True,
        "call_policy": "one_codex_then_claude_only_for_missing_coverage",
    },
}

PROFILE_DEFAULTS = {
    "smoke": {"providers": ["filesystem"], "repeats": 1},
    "factorial": {"providers": ["filesystem", "oracle"], "repeats": 2},
}
JUDGE_SYSTEM_PROMPT = (
    "You are a strict identity-blind code-review evaluator. Output valid JSON only."
)
JUDGE_INSTRUCTIONS = (
    "Score this candidate independently. Do not infer or reward the model, provider, "
    "execution strategy, verbosity, redundancy, repetition, polish, formatting, or "
    "brevity. Do not deduct for failing to restate that no files were modified, for "
    "failing to explicitly say there are no unsupported claims, or for paraphrasing "
    "rather than quoting code when evidence locations are specific. Score technical "
    "content only and penalize actual unsupported claims. Return "
    "one JSON object with key score containing candidate_id, scale='points', "
    "correctness_0_35, "
    "gold_coverage_0_30, evidence_0_15, actionability_0_10, grounding_0_10, "
    "covered_gold_ids, unsupported_claims, and a concise rationale. Dimension values "
    "must be point values: full correctness is 35, never 0.35."
)
JUDGE_DIMENSIONS = {
    "correctness_0_35": 35,
    "gold_coverage_0_30": 30,
    "evidence_0_15": 15,
    "actionability_0_10": 10,
    "grounding_0_10": 10,
}
JUDGE_MAX_ATTEMPTS = 2
_DISALLOWED_JUDGE_RATIONALE = re.compile(
    r"(?<!no )(?<!not )(?<!without )"
    r"\b(?:deductions?|deducted|penali[sz]ed|penalt(?:y|ies))\b.{0,220}"
    r"(?:\b(?:verbosity|verbose|redundan(?:cy|t)|repetition|repetitive|polish|"
    r"style|formatting|brevity)\b|"
    r"\bnot explicitly stat(?:e|ed|ing)\b.{0,80}\b(?:no files|unsupported claims)\b|"
    r"\bnot quot(?:e|ed|ing)\b.{0,40}\bverbatim\b)",
    re.I | re.S,
)


class JudgeValidationError(RuntimeError):
    """Fail-closed judge error that retains paid-call accounting."""

    def __init__(self, message: str, calls: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.calls = [dict(call) for call in calls]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _publishable_artifact(
    artifact: Mapping[str, Any], *, repo_root: Path, run_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Redact credentials and environment-specific path prefixes."""
    replacements = {
        str(repo_root.resolve()): "[REPOSITORY]",
        str(Path.home().resolve()): "[HOME]",
    }
    if run_root is not None:
        replacements[str(run_root.resolve())] = "[RUN_WORKSPACE]"
    return dict(redact_paths(redact(artifact), replacements))


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> Optional[str]:
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


def arm_specs(providers: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Return the full strategy × provider matrix in stable order."""
    return {
        f"{strategy}__{provider}": {
            **spec,
            "strategy": strategy,
            "provider": provider,
            "arm": f"{strategy}__{provider}",
            "display_name": f"{spec['display_name']} ({provider.title()})",
        }
        for provider in providers
        for strategy, spec in STRATEGIES.items()
    }


def counterbalanced_orders(
    arms: Sequence[str], repeats: int, seed: int
) -> List[List[str]]:
    """Pair each seeded order with its reverse to balance execution position."""
    ordered_arms = list(arms)
    orders: List[List[str]] = []
    for pair_index in range((repeats + 1) // 2):
        order = list(ordered_arms)
        random.Random(seed + pair_index).shuffle(order)
        orders.append(order)
        if len(orders) < repeats:
            orders.append(list(reversed(order)))
    return orders


def protocol_manifest(
    *,
    repo_root: Path,
    profile: str,
    providers: Sequence[str],
    repeats: int,
    seed: int,
    codex_model: str,
    claude_model: str,
    judge_model: str,
    maximum_observed_cost_usd: float,
) -> Dict[str, Any]:
    specs = arm_specs(providers)
    orders = counterbalanced_orders(list(specs), repeats, seed)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "profile": profile,
        "paper_comparable": False,
        "fixture": {
            "name": "synthetic_export_policy_review_v1",
            "requirements_hash": canonical_hash(legacy.REQUIREMENTS),
            "workspace_hash": canonical_hash(legacy.FIXTURE_FILES),
            "file_sha256": {
                name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                for name, content in legacy.FIXTURE_FILES.items()
            },
            "gold_hash": canonical_hash(legacy.GOLD_FINDINGS),
            "required_criterion_ids": list(legacy.REQUIRED_FINDING_IDS),
        },
        "models": {
            "codex": codex_model,
            "claude_code": claude_model,
            "judge": judge_model,
        },
        "providers": list(providers),
        "strategies": STRATEGIES,
        "arms": specs,
        "repeats": repeats,
        "seed": seed,
        "execution_orders": orders,
        "mean_execution_position": {
            arm: round(mean(order.index(arm) + 1 for order in orders), 3)
            for arm in specs
        },
        "controls": {
            "cold_unique_memory_scope_per_arm": True,
            "same_fixture_base_prompt_schema_and_verification": True,
            "same_per_executed_harness_budget": True,
            "equal_total_compute": False,
            "identity_blinded_judge": True,
            "independent_candidate_judging": True,
            "judge_score_scale": "explicit_points_with_legacy_fractional_normalization",
            "primary_quality_metric": "required_criterion_recall",
            "llm_scalar_judge_role": "secondary_diagnostic",
            "deterministic_panel_consolidation": True,
            "coordinator_model_calls": 0,
            "provider_content_fingerprint_required": True,
            "fail_closed_before_judging": True,
            "maximum_observed_execution_cost_usd": maximum_observed_cost_usd,
        },
        "claim_boundaries": [
            "Direct means MetaHarness.run without a MemAgent wrapper; it is not an ungoverned raw CLI call.",
            "Wrapper pairs estimate MemoRizz interface overhead for the same harness.",
            "The full panel tests mandatory two-harness coordination.",
            "The adaptive panel tests coverage-triggered early stopping.",
            "Provider effects are estimated only from the same strategy across providers.",
            "One synthetic fixture cannot establish general harness accuracy.",
            "Two repeats are descriptive and do not support significance claims.",
            "The fixture is the unit of inference; repeated executions are not independent task samples.",
            "API-equivalent Codex cost estimates and Claude-reported cost have different bases.",
        ],
        "implementation": {
            "runner_sha256": _sha256_file(Path(__file__)),
            "legacy_core_sha256": _sha256_file(Path(legacy.__file__)),
        },
        "reproducibility": {
            "generated_at": _utc_now(),
            "memorizz_version": memorizz.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_revision": _git_revision(repo_root),
            "prompt_hash": canonical_hash(legacy.COMMON_TASK),
            "output_schema_hash": canonical_hash(legacy.FINDING_OUTPUT_SCHEMA),
            "judge_rubric_hash": canonical_hash(
                {
                    "requirements": legacy.REQUIREMENTS,
                    "gold": legacy.GOLD_FINDINGS,
                    "system": JUDGE_SYSTEM_PROMPT,
                    "instructions": JUDGE_INSTRUCTIONS,
                    "dimensions": JUDGE_DIMENSIONS,
                }
            ),
        },
        "inference": {
            "unit": "fixture_task",
            "independent_task_count": 1,
            "execution_repeats": repeats,
            "inferential_statistics_allowed": False,
        },
    }
    protocol_material = {
        key: value for key, value in manifest.items() if key != "reproducibility"
    }
    environment_material = {
        key: value
        for key, value in manifest["reproducibility"].items()
        if key != "generated_at"
    }
    manifest["protocol_fingerprint"] = canonical_hash(protocol_material)
    manifest["environment_fingerprint"] = canonical_hash(environment_material)
    return manifest


def _shared_context(repeat: int, arm: str) -> Dict[str, Any]:
    return {
        "evaluation_arm": arm,
        "repeat": repeat,
        "grounding_required": True,
        "memory_query": legacy.REQUIREMENTS,
        "memory_context_strategy": "evidence_pack",
    }


def run_wrapper(
    runtime: Mapping[str, Any],
    *,
    arm: str,
    display_name: str,
    provider_name: str,
    harness: str,
    model: str,
    scope: Mapping[str, Any],
    repeat: int,
    workspace: Path,
    permissions: HarnessPermissions,
    verification: VerificationSpec,
) -> Dict[str, Any]:
    """Run exactly one vendor harness through the MemAgent runtime interface."""
    provider = runtime["providers"][provider_name]
    service = runtime["services"][provider_name]
    budget = legacy.CODEX_BUDGET if harness == "codex" else legacy.CLAUDE_BUDGET
    agent = (
        MemAgentBuilder()
        .with_name(f"Factorial wrapper {arm} {repeat}")
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
                "output_schema": legacy.FINDING_OUTPUT_SCHEMA,
            },
        )
        .with_learning_control_plane(
            True,
            {
                "evidence_token_budget": 900,
                "evidence_max_items": 4,
                "compile_async": False,
                "compile_every_n_events": 0,
            },
        )
        .with_semantic_cache(enabled=False)
        .as_ephemeral()
        .build(validate=False)
    )
    runtime["agent_ids"][provider_name].add(agent.agent_id)
    before = {row["run_id"] for row in service.list_runs(limit=10_000)}
    started = time.perf_counter()
    try:
        response = agent.run(
            legacy.COMMON_TASK,
            memory_id=scope["memory_id"],
            user_id=USER_ID,
            thread_id=f"repeat-{repeat}",
            context=_shared_context(repeat, arm),
        )
        wall_latency_ms = int((time.perf_counter() - started) * 1_000)
        rows = [
            row
            for row in service.list_runs(limit=10_000)
            if row["run_id"] not in before
        ]
        if len(rows) != 1:
            raise RuntimeError(
                f"MemAgent wrapper must execute exactly one harness run; found {len(rows)}"
            )
        run = legacy.normalized_result_metrics(
            service,
            rows[0]["result"],
            model=model,
            expected_source_id=scope["source_id"],
        )
        try:
            rendered, coverage = legacy.structured_response(
                response, contributor=f"{arm}-wrapper"
            )
        except Exception as exc:
            rendered = str(response or "")
            coverage = {
                "coverage_complete": False,
                "required_ids": legacy.REQUIRED_FINDING_IDS,
                "covered_ids": [],
                "missing_ids": legacy.REQUIRED_FINDING_IDS,
                "parse_errors": {f"{arm}-wrapper": f"{type(exc).__name__}: {exc}"},
            }
        return {
            "arm": arm,
            "display_name": display_name,
            "strategy": arm.split("__", 1)[0],
            "interface": "memagent_runtime",
            "memory_provider": provider_name,
            "repeat": repeat,
            "source_id": scope["source_id"],
            "response": rendered,
            "structured_coverage": coverage,
            "ok": bool(run["ok"]),
            "wall_latency_ms": wall_latency_ms,
            "wrapper_overhead_ms": max(
                0, wall_latency_ms - int(run.get("latency_ms") or 0)
            ),
            "seed_latency_ms": scope["seed_latency_ms"],
            "runs": [run],
            "cost_usd": run["cost_usd"],
            "cost_complete": run["cost_usd"] is not None,
            "cost_basis": [run["cost_basis"]],
            "agent_ids": [agent.agent_id],
            "memory_first": {
                "semantic_cache": agent.semantic_cache_stats(),
                "learning": agent.learning_report(
                    memory_id=scope["memory_id"],
                    user_id=USER_ID,
                    thread_id=f"repeat-{repeat}",
                ),
                "policy": {
                    "cold_run": True,
                    "semantic_cache_enabled": False,
                    "summarize_in_turn": False,
                    "compact_in_turn": False,
                    "forget_in_turn": False,
                },
            },
        }
    finally:
        agent.close(close_memory_provider=False)


def execute_arm(
    runtime: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    scope: Mapping[str, Any],
    repeat: int,
    run_scope: str,
    workspace: Path,
    permissions: HarnessPermissions,
    verification: VerificationSpec,
    codex_model: str,
    claude_model: str,
) -> Dict[str, Any]:
    arm = str(spec["arm"])
    provider_name = str(spec["provider"])
    kind = str(spec["kind"])
    if kind == "direct":
        harness = str(spec["harness"])
        model = codex_model if harness == "codex" else claude_model
        record = legacy.run_single(
            runtime,
            arm=arm,
            harness=harness,
            model=model,
            scope=scope,
            repeat=repeat,
            workspace=workspace,
            permissions=permissions,
            verification=verification,
            provider_name=provider_name,
            display_name=str(spec["display_name"]),
        )
        record.update(
            {
                "strategy": spec["strategy"],
                "interface": "direct_metaharness",
                "source_id": scope["source_id"],
            }
        )
        return record
    if kind == "wrapper":
        harness = str(spec["harness"])
        model = codex_model if harness == "codex" else claude_model
        return run_wrapper(
            runtime,
            arm=arm,
            display_name=str(spec["display_name"]),
            provider_name=provider_name,
            harness=harness,
            model=model,
            scope=scope,
            repeat=repeat,
            workspace=workspace,
            permissions=permissions,
            verification=verification,
        )
    record = legacy.run_panel(
        runtime,
        run_scope=f"{run_scope}-{arm}-{repeat}",
        arm=arm,
        provider_name=provider_name,
        scope=scope,
        repeat=repeat,
        workspace=workspace,
        permissions=permissions,
        verification=verification,
        codex_model=codex_model,
        claude_model=claude_model,
        adaptive=bool(spec.get("adaptive")),
        display_name=str(spec["display_name"]),
    )
    record.update(
        {
            "strategy": spec["strategy"],
            "interface": "memagent_multi_agent",
            "source_id": scope["source_id"],
        }
    )
    return record


def validate_record(record: Mapping[str, Any], spec: Mapping[str, Any]) -> List[str]:
    """Fail closed on execution, grounding, schema, and call-policy violations."""
    failures: List[str] = []
    if not record.get("ok"):
        failures.append("arm_not_ok")
    coverage = dict(record.get("structured_coverage") or {})
    if not coverage.get("coverage_complete"):
        failures.append(
            "structured_coverage_incomplete="
            + ",".join(coverage.get("missing_ids") or [])
        )
    if coverage.get("parse_errors"):
        failures.append("structured_parse_errors")
    runs = list(record.get("runs") or [])
    kind = str(spec["kind"])
    harnesses = [str(run.get("harness")) for run in runs]
    if kind in {"direct", "wrapper"}:
        if len(runs) != 1 or harnesses != [spec["harness"]]:
            failures.append(
                f"call_policy_violation:expected_one_{spec['harness']}:got={harnesses}"
            )
    elif not spec.get("adaptive"):
        if len(runs) != 2 or set(harnesses) != {"codex", "claude-code"}:
            failures.append(
                "call_policy_violation:full_panel_requires_codex_and_claude:"
                f"got={harnesses}"
            )
        if (record.get("execution") or {}).get("strategy") != "parallel":
            failures.append("full_panel_parallel_execution_missing")
    else:
        if not 1 <= len(runs) <= 2 or "codex" not in harnesses:
            failures.append(f"call_policy_violation:adaptive_panel_runs={harnesses}")
        if (record.get("execution") or {}).get("strategy") != "adaptive_coverage":
            failures.append("adaptive_execution_report_missing")
    for run in runs:
        if run.get("status") != "succeeded":
            failures.append(
                f"{run.get('harness')}:status={run.get('status')}:"
                f"{run.get('error_code')}"
            )
        if not run.get("verified"):
            failures.append(f"{run.get('harness')}:verification_missing")
        if not run.get("grounded"):
            failures.append(f"{run.get('harness')}:grounding_missing")
        if not run.get("context_content_fingerprint"):
            failures.append(f"{run.get('harness')}:content_fingerprint_missing")
        usage = dict(run.get("usage_totals") or {})
        if int(usage.get("input_tokens", 0) or 0) <= 0:
            failures.append(f"{run.get('harness')}:input_usage_missing")
        if int(usage.get("output_tokens", 0) or 0) <= 0:
            failures.append(f"{run.get('harness')}:output_usage_missing")
    if kind == "panel":
        consolidation = dict(record.get("consolidation") or {})
        if consolidation.get("strategy") != "structured":
            failures.append("panel_structured_consolidation_missing")
        if consolidation.get("model_used"):
            failures.append("panel_unexpected_coordinator_model")
        root_sources = set(
            (consolidation.get("memory_context") or {}).get("source_ids") or []
        )
        if record.get("source_id") not in root_sources:
            failures.append("panel_root_grounding_missing")
        execution = dict(record.get("execution") or {})
        if spec.get("adaptive"):
            if execution.get("primary_task_ids") != ["policy-review"]:
                failures.append("adaptive_primary_task_policy_missing")
            if execution.get("escalation_task_ids") != ["verification-review"]:
                failures.append("adaptive_escalation_task_policy_missing")
    if record.get("workflow_failures"):
        failures.append(f"workflow_failures={len(record['workflow_failures'])}")
    return failures


def context_matrix_preflight(
    runtime: Mapping[str, Any],
    *,
    specs: Mapping[str, Mapping[str, Any]],
    scopes: Mapping[tuple[int, str], Mapping[str, Any]],
    repeats: int,
    workspace: Path,
) -> Dict[str, Any]:
    """Prove every arm receives the same source content before model spend."""
    rows: List[Dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for arm, spec in specs.items():
            scope = scopes[(repeat, arm)]
            task = legacy.HarnessTask(
                task=legacy.COMMON_TASK,
                workspace=str(workspace),
                harness="codex",
                memory_id=scope["memory_id"],
                user_id=USER_ID,
                thread_id=f"repeat-{repeat}",
                agent_id="factorial-context-preflight",
                permissions={"mcp_access": "none"},
                context=_shared_context(repeat, arm),
            )
            pack = runtime["services"][spec["provider"]]._context_for_task(task)
            rows.append(
                {
                    "repeat": repeat,
                    "arm": arm,
                    "provider": spec["provider"],
                    "grounded": scope["source_id"] in pack.source_ids,
                    "content_fingerprint": pack.content_fingerprint,
                    "token_estimate": pack.token_estimate,
                }
            )
    fingerprints = {row["content_fingerprint"] for row in rows if row["grounded"]}
    report = {
        "ok": all(row["grounded"] for row in rows) and len(fingerprints) == 1,
        "arm_count": len(rows),
        "unique_content_fingerprints": sorted(fingerprints),
        "token_estimates": sorted({row["token_estimate"] for row in rows}),
        "rows": rows,
    }
    if not report["ok"]:
        raise RuntimeError(f"Context matrix preflight failed: {report}")
    return report


def _sample_stdev(values: Iterable[float]) -> float:
    items = list(values)
    return float(stdev(items)) if len(items) > 1 else 0.0


def _required_criterion_recall(record: Mapping[str, Any]) -> float:
    coverage = dict(record.get("structured_coverage") or {})
    required = {str(item) for item in coverage.get("required_ids") or []}
    covered = {str(item) for item in coverage.get("covered_ids") or []}
    if not required:
        return 0.0
    return 100.0 * len(required & covered) / len(required)


def summarize(
    records: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for arm, spec in specs.items():
        values = [record for record in records if record["arm"] == arm]
        scores = [float(record["judge"]["score_0_100"]) for record in values]
        latencies = [float(record["wall_latency_ms"]) for record in values]
        costs = [record.get("cost_usd") for record in values]
        known_costs = [float(value) for value in costs if value is not None]
        input_tokens = [
            sum(run["usage_totals"]["input_tokens"] for run in record["runs"])
            for record in values
        ]
        output_tokens = [
            sum(run["usage_totals"]["output_tokens"] for run in record["runs"])
            for record in values
        ]
        total_tokens = [
            input_value + output_value
            for input_value, output_value in zip(input_tokens, output_tokens)
        ]
        call_counts = [len(record["runs"]) for record in values]
        criterion_recall = [_required_criterion_recall(record) for record in values]
        summary = {
            "arm": arm,
            "strategy": spec["strategy"],
            "display_name": spec["display_name"],
            "provider": spec["provider"],
            "kind": spec["kind"],
            "repeats": len(values),
            "judge_score_mean": round(mean(scores), 3),
            "judge_score_sample_stdev": round(_sample_stdev(scores), 3),
            "gold_findings_mean": round(
                mean(
                    len(record["judge"].get("covered_gold_ids") or [])
                    for record in values
                ),
                3,
            ),
            "required_criterion_recall_mean": round(mean(criterion_recall), 3),
            "unsupported_claims_mean": round(
                mean(
                    len(record["judge"].get("unsupported_claims") or [])
                    for record in values
                ),
                3,
            ),
            "verified_rate": round(
                mean(
                    all(run["verified"] for run in record["runs"]) for record in values
                ),
                3,
            ),
            "grounded_rate": round(
                mean(
                    all(run["grounded"] for run in record["runs"]) for record in values
                ),
                3,
            ),
            "wall_latency_ms_mean": round(mean(latencies)),
            "wall_latency_ms_sample_stdev": round(_sample_stdev(latencies), 3),
            "cost_usd_mean": (
                round(mean(known_costs), 8) if len(known_costs) == len(values) else None
            ),
            "cost_coverage": f"{len(known_costs)}/{len(values)}",
            "input_tokens_mean": round(mean(input_tokens)),
            "output_tokens_mean": round(mean(output_tokens)),
            "total_tokens_mean": round(mean(total_tokens)),
            "harness_calls_mean": round(mean(call_counts), 3),
            "raw_judge_scores": scores,
            "raw_wall_latency_ms": latencies,
            "raw_cost_usd": known_costs,
        }
        summaries.append(summary)
    return summaries


def _paired_values(
    records: Sequence[Mapping[str, Any]],
    *,
    left: str,
    right: str,
    metric: str,
) -> List[float]:
    values: List[float] = []
    repeats = sorted({int(record["repeat"]) for record in records})
    for repeat in repeats:
        by_arm = {
            record["arm"]: record
            for record in records
            if int(record["repeat"]) == repeat
        }
        if left not in by_arm or right not in by_arm:
            continue

        def record_value(record: Mapping[str, Any]) -> Any:
            if metric == "judge_score":
                return record["judge"]["score_0_100"]
            if metric == "required_criterion_recall":
                return _required_criterion_recall(record)
            if metric in {"input_tokens", "output_tokens", "total_tokens"}:
                input_value = sum(
                    run["usage_totals"]["input_tokens"] for run in record["runs"]
                )
                output_value = sum(
                    run["usage_totals"]["output_tokens"] for run in record["runs"]
                )
                if metric == "input_tokens":
                    return input_value
                if metric == "output_tokens":
                    return output_value
                return input_value + output_value
            return record.get(metric)

        left_value = record_value(by_arm[left])
        right_value = record_value(by_arm[right])
        if left_value is not None and right_value is not None:
            values.append(float(left_value) - float(right_value))
    return values


def paired_effects(
    records: Sequence[Mapping[str, Any]], providers: Sequence[str]
) -> Dict[str, Any]:
    """Return descriptive paired deltas; positive means left minus right."""

    def comparison(left: str, right: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"left": left, "right": right}
        for metric in (
            "judge_score",
            "required_criterion_recall",
            "wall_latency_ms",
            "cost_usd",
            "input_tokens",
            "output_tokens",
            "total_tokens",
        ):
            values = _paired_values(records, left=left, right=right, metric=metric)
            payload[f"{metric}_paired_deltas"] = values
            payload[f"{metric}_mean_delta"] = round(mean(values), 8) if values else None
        return payload

    wrapper: Dict[str, Any] = {}
    coordination: Dict[str, Any] = {}
    for provider in providers:
        wrapper[f"codex__{provider}"] = comparison(
            f"memagent_codex__{provider}", f"direct_codex__{provider}"
        )
        wrapper[f"claude__{provider}"] = comparison(
            f"memagent_claude__{provider}", f"direct_claude__{provider}"
        )
        coordination[f"full_vs_direct_codex__{provider}"] = comparison(
            f"panel_full__{provider}", f"direct_codex__{provider}"
        )
        coordination[f"full_vs_direct_claude__{provider}"] = comparison(
            f"panel_full__{provider}", f"direct_claude__{provider}"
        )
        coordination[f"full_vs_memagent_codex__{provider}"] = comparison(
            f"panel_full__{provider}", f"memagent_codex__{provider}"
        )
        coordination[f"full_vs_memagent_claude__{provider}"] = comparison(
            f"panel_full__{provider}", f"memagent_claude__{provider}"
        )
        coordination[f"adaptive_vs_memagent_codex__{provider}"] = comparison(
            f"panel_adaptive__{provider}", f"memagent_codex__{provider}"
        )
        coordination[f"adaptive_vs_full__{provider}"] = comparison(
            f"panel_adaptive__{provider}", f"panel_full__{provider}"
        )
    provider_effects: Dict[str, Any] = {}
    if {"filesystem", "oracle"}.issubset(providers):
        for strategy in STRATEGIES:
            provider_effects[strategy] = comparison(
                f"{strategy}__oracle", f"{strategy}__filesystem"
            )
    return {
        "direction": "left_minus_right",
        "inferential_status": "descriptive_only_single_fixture",
        "wrapper_overhead": wrapper,
        "coordination_value": coordination,
        "provider_sensitivity": provider_effects,
    }


def _validated_judge_score(
    value: Mapping[str, Any], candidate_id: str
) -> Dict[str, Any]:
    """Validate one blinded judge response and derive its bounded total."""
    if value.get("candidate_id") != candidate_id:
        raise RuntimeError(
            f"Judge returned candidate_id={value.get('candidate_id')!r}; "
            f"expected {candidate_id!r}"
        )
    normalized = dict(value)
    raw_scores: Dict[str, float] = {}
    for field, upper in JUDGE_DIMENSIONS.items():
        score = float(normalized.get(field, -1))
        if not 0 <= score <= upper:
            raise RuntimeError(f"Judge field {field}={score} is outside 0..{upper}")
        raw_scores[field] = score
    covered = list(normalized.get("covered_gold_ids") or [])
    valid_gold_ids = {item["id"] for item in legacy.GOLD_FINDINGS}
    if len(covered) != len(set(covered)) or not set(covered).issubset(valid_gold_ids):
        raise RuntimeError(f"Judge returned invalid covered_gold_ids: {covered}")
    unsupported = normalized.get("unsupported_claims")
    if not isinstance(unsupported, list):
        raise RuntimeError("Judge unsupported_claims must be a list")
    if not str(normalized.get("rationale") or "").strip():
        raise RuntimeError("Judge rationale is required")
    if _DISALLOWED_JUDGE_RATIONALE.search(str(normalized["rationale"])):
        raise RuntimeError(
            "Judge rationale used a prohibited style/verbosity scoring factor"
        )

    # Hosted judges occasionally return the rubric's point values divided by
    # 100 (35 -> 0.35, 30 -> 0.30, and so on). That is a transport-scale error,
    # not a 1/100-quality answer. Detect only this bounded fractional form,
    # normalize it deterministically, and retain provenance in the artifact.
    declared_scale = str(normalized.get("scale") or "").strip().lower()
    fractional_scale = declared_scale in {"fraction", "normalized", "0_to_1"}
    legacy_fractional_scale = (
        not declared_scale
        and any(score > 0 for score in raw_scores.values())
        and all(
            raw_scores[field] <= (float(upper) / 100.0)
            for field, upper in JUDGE_DIMENSIONS.items()
        )
    )
    if fractional_scale or legacy_fractional_scale:
        raw_scores = {field: score * 100.0 for field, score in raw_scores.items()}
        normalized["scale_normalization"] = "fractional_x100"
    else:
        normalized["scale_normalization"] = "none"
    for field, upper in JUDGE_DIMENSIONS.items():
        score = raw_scores[field]
        if not 0 <= score <= upper:
            raise RuntimeError(
                f"Normalized judge field {field}={score} is outside 0..{upper}"
            )
        normalized[field] = round(score, 4)
    normalized["scale"] = "points"
    normalized["score_0_100"] = round(
        sum(normalized[field] for field in JUDGE_DIMENSIONS), 2
    )
    return normalized


def judge_records_independently(
    records: List[Dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    judge_model: str,
) -> List[Dict[str, Any]]:
    """Judge candidates one at a time to avoid cross-candidate anchoring."""
    judge = legacy.create_llm_provider(
        {
            "provider": "openai",
            "model": judge_model,
            "temperature": 0,
            "max_tokens": 2_000,
            "response_format": {"type": "json_object"},
        }
    )
    reports: List[Dict[str, Any]] = []
    try:
        for repeat in range(1, repeats + 1):
            candidates = [record for record in records if record["repeat"] == repeat]
            random.Random(seed * 10 + repeat).shuffle(candidates)
            labels = [
                f"candidate_{index:04d}" for index in range(101, 101 + len(candidates))
            ]
            random.Random(seed * 100 + repeat).shuffle(labels)
            calls: List[Dict[str, Any]] = []
            label_to_arm: Dict[str, str] = {}
            for label, record in zip(labels, candidates):
                label_to_arm[label] = str(record["arm"])
                rubric = {
                    "requirements": legacy.REQUIREMENTS,
                    "gold_findings": legacy.GOLD_FINDINGS,
                    "candidate": {
                        "candidate_id": label,
                        "response": legacy.normalize_candidate(record["response"]),
                    },
                    "instructions": JUDGE_INSTRUCTIONS,
                }
                score: Optional[Dict[str, Any]] = None
                last_error: Optional[Exception] = None
                for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
                    raw = judge.generate(
                        [
                            {
                                "role": "system",
                                "content": JUDGE_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": json.dumps(rubric, ensure_ascii=False),
                            },
                        ],
                        tools=None,
                    )
                    usage = judge.get_last_usage() or {}
                    call = {
                        "candidate_id": label,
                        "attempt": attempt,
                        "usage": usage,
                        "cost_usd": legacy.estimate_openai_cost(usage, judge_model),
                    }
                    try:
                        judged = legacy.parse_json_object(raw)
                        score = _validated_judge_score(
                            dict(judged.get("score") or {}), label
                        )
                    except (TypeError, ValueError, RuntimeError) as exc:
                        last_error = exc
                        call.update(
                            {
                                "status": "rejected",
                                "error": str(redact(str(exc)))[:500],
                            }
                        )
                        calls.append(call)
                        continue
                    call["status"] = "accepted"
                    calls.append(call)
                    break
                if score is None:
                    prior_calls = [
                        call
                        for completed_report in reports
                        for call in completed_report["calls"]
                    ]
                    raise JudgeValidationError(
                        f"Judge failed validation for {label} after "
                        f"{JUDGE_MAX_ATTEMPTS} attempts: {last_error}",
                        [*prior_calls, *calls],
                    )
                record["judge"] = score
            ranking = [
                label
                for label, _ in sorted(
                    (
                        (label, record["judge"]["score_0_100"])
                        for label, record in zip(labels, candidates)
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            reports.append(
                {
                    "repeat": repeat,
                    "mode": "independent_candidate_scoring",
                    "judge_calls": len(calls),
                    "blinded_mapping": label_to_arm,
                    "ranking": ranking,
                    "calls": calls,
                    "cost_usd": (
                        round(sum(float(call["cost_usd"]) for call in calls), 8)
                        if all(call["cost_usd"] is not None for call in calls)
                        else None
                    ),
                }
            )
    finally:
        close_judge = getattr(judge, "close", None)
        if callable(close_judge):
            close_judge()
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        default=_truthy(os.getenv("MEMORIZZ_RUN_HARNESS_EVALUATION")),
        help="Make paid harness and judge calls.",
    )
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_DEFAULTS), default="factorial"
    )
    parser.add_argument("--providers", nargs="+", choices=("filesystem", "oracle"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-observed-cost-usd",
        type=float,
        default=float(os.getenv("MEMORIZZ_EVAL_MAX_OBSERVED_COST_USD", "5.0")),
    )
    parser.add_argument("--allow-incomplete-cost", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(tempfile.gettempdir()) / f"memorizz-factorial-{timestamp}.json"


def main() -> int:
    args = parse_args()
    defaults = PROFILE_DEFAULTS[args.profile]
    providers = list(dict.fromkeys(args.providers or defaults["providers"]))
    repeats = args.repeats if args.repeats is not None else defaults["repeats"]
    if not 1 <= repeats <= 10:
        raise ValueError("--repeats must be between 1 and 10")
    if args.max_observed_cost_usd <= 0:
        raise ValueError("--max-observed-cost-usd must be positive")
    output = args.output or _default_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    if not _truthy(os.getenv("MEMORIZZ_EVAL_DISABLE_DOTENV")):
        load_layered_env(extra_paths=[repo_root / ".env"])
    codex_model = os.getenv("MEMORIZZ_EVAL_CODEX_MODEL", "gpt-5.6-luna").strip()
    claude_model = os.getenv("MEMORIZZ_EVAL_CLAUDE_MODEL", "sonnet").strip()
    judge_model = os.getenv("MEMORIZZ_EVAL_JUDGE_MODEL", "gpt-4.1").strip()
    manifest = protocol_manifest(
        repo_root=repo_root,
        profile=args.profile,
        providers=providers,
        repeats=repeats,
        seed=args.seed,
        codex_model=codex_model,
        claude_model=claude_model,
        judge_model=judge_model,
        maximum_observed_cost_usd=args.max_observed_cost_usd,
    )
    artifact: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "status": "dry_run" if not args.execute else "initializing",
        "valid": False,
        "paper_comparable": False,
        "protocol_manifest": manifest,
        "records": [],
        "validation_failures": [],
        "summaries": [],
        "paired_effects": {},
    }
    if not args.execute:
        output.write_text(
            json.dumps(
                _publishable_artifact(artifact, repo_root=repo_root),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "protocol": PROTOCOL_NAME,
                    "arms_per_repeat": len(manifest["arms"]),
                    "repeats": repeats,
                    "external_calls": 0,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        artifact.update(
            {
                "status": "error",
                "error": {
                    "type": "AuthenticationRequired",
                    "code": "openai_authentication_required",
                    "message": "OpenAI authentication is required for paid execution.",
                    "remediation": (
                        "Set OPENAI_API_KEY in the process environment, then rerun the "
                        "factorial evaluator. Do not place credentials in notebooks or "
                        "artifacts."
                    ),
                },
                "cleanup": [],
            }
        )
        output.write_text(
            json.dumps(
                _publishable_artifact(artifact, repo_root=repo_root),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"status": "error", "valid": False, "output": str(output)},
                indent=2,
            )
        )
        return 2

    root = Path(tempfile.mkdtemp(prefix="memorizz-factorial-comparison-"))
    workspace = root / "workspace"
    workspace.mkdir()
    legacy._write_fixture(workspace)
    run_scope = uuid.uuid4().hex[:12]
    permissions = HarnessPermissions(
        workspace_mode="read_only",
        allowed_roots=[str(workspace)],
        network="none",
        mcp_access="none",
        allowed_env=[],
    )
    verification = VerificationSpec(
        command=f'"{sys.executable}" -B verify.py', timeout_seconds=30
    )
    specs = arm_specs(providers)
    orders = counterbalanced_orders(list(specs), repeats, args.seed)
    runtime: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    cleanup_reports: List[Dict[str, Any]] = []
    observed_cost = 0.0
    unknown_cost_records = 0
    exit_code = 0
    try:
        runtime = legacy.create_runtime(root, workspace, provider_names=providers)
        not_ready = {
            name: {
                "error_code": probe.get("error_code"),
                "error": probe.get("error"),
                "remediation": probe.get("remediation"),
            }
            for name, probe in runtime["probes"].items()
            if not probe.get("ready")
        }
        if not_ready:
            raise RuntimeError(
                "All harnesses must be ready before paid calls: "
                + json.dumps(not_ready, ensure_ascii=False)
            )
        scopes = {
            (repeat, arm): legacy.seed_scope(
                runtime,
                run_scope=run_scope,
                arm=arm,
                provider_name=str(spec["provider"]),
                repeat=repeat,
            )
            for repeat in range(1, repeats + 1)
            for arm, spec in specs.items()
        }
        context_preflight = context_matrix_preflight(
            runtime,
            specs=specs,
            scopes=scopes,
            repeats=repeats,
            workspace=workspace,
        )
        artifact.update(
            {
                "status": "running",
                "provider_preflight": runtime["preflight"],
                "harness_probes": runtime["probes"],
                "context_matrix_preflight": context_preflight,
                "pricing": legacy.PRICE_SNAPSHOT,
            }
        )
        for repeat, order in enumerate(orders, start=1):
            for arm in order:
                if observed_cost > args.max_observed_cost_usd:
                    raise RuntimeError(
                        "Observed execution cost exceeded the configured stop budget "
                        f"before {arm}: ${observed_cost:.6f} > "
                        f"${args.max_observed_cost_usd:.6f}"
                    )
                spec = specs[arm]
                record = execute_arm(
                    runtime,
                    spec=spec,
                    scope=scopes[(repeat, arm)],
                    repeat=repeat,
                    run_scope=run_scope,
                    workspace=workspace,
                    permissions=permissions,
                    verification=verification,
                    codex_model=codex_model,
                    claude_model=claude_model,
                )
                records.append(record)
                issues = validate_record(record, spec)
                if record.get("cost_usd") is None:
                    unknown_cost_records += 1
                    if not args.allow_incomplete_cost:
                        issues.append("cost_telemetry_incomplete")
                else:
                    observed_cost += float(record["cost_usd"])
                if issues:
                    failures.append({"arm": arm, "repeat": repeat, "issues": issues})
                    raise RuntimeError(
                        f"Fail-fast validation rejected {arm} repeat {repeat}: "
                        + "; ".join(issues)
                    )
                print(
                    {
                        "arm": arm,
                        "repeat": repeat,
                        "calls": len(record["runs"]),
                        "latency_ms": record["wall_latency_ms"],
                        "cost_usd": record.get("cost_usd"),
                        "observed_cost_usd": round(observed_cost, 6),
                    },
                    flush=True,
                )
        expected = repeats * len(specs)
        if len(records) != expected:
            raise RuntimeError(f"Expected {expected} records, found {len(records)}")
        judge_reports = judge_records_independently(
            records,
            repeats=repeats,
            seed=args.seed,
            judge_model=judge_model,
        )
        judge_cost_values = [item.get("cost_usd") for item in judge_reports]
        judge_cost_complete = all(value is not None for value in judge_cost_values)
        if not judge_cost_complete and not args.allow_incomplete_cost:
            raise RuntimeError("Judge cost telemetry is incomplete")
        judge_cost = sum(
            float(value) for value in judge_cost_values if value is not None
        )
        summaries = summarize(records, specs)
        artifact.update(
            {
                "status": "valid",
                "valid": True,
                "records": records,
                "validation_failures": [],
                "judge_reports": judge_reports,
                "summaries": summaries,
                "paired_effects": paired_effects(records, providers),
                "cost_accounting": {
                    "execution_observed_usd": round(observed_cost, 8),
                    "judge_estimated_usd": round(judge_cost, 8),
                    "total_measured_usd": round(observed_cost + judge_cost, 8),
                    "unknown_execution_cost_records": unknown_cost_records,
                    "judge_cost_complete": judge_cost_complete,
                    "execution_harness_calls": sum(
                        len(record["runs"]) for record in records
                    ),
                    "judge_model_calls": sum(
                        int(report["judge_calls"]) for report in judge_reports
                    ),
                    "stop_budget_usd": args.max_observed_cost_usd,
                    "stop_budget_is_post_call_observed_not_a_preflight_quote": True,
                },
                "interpretation_policy": {
                    "winner_allowed": False,
                    "reason": (
                        "Single synthetic fixture and insufficient independent task "
                        "families for a general accuracy claim."
                    ),
                    "permitted_claims": [
                        "paired wrapper overhead on this fixture",
                        "full versus adaptive panel behavior on this fixture",
                        "provider sensitivity for matched strategies on this fixture",
                        "required-criterion recall and deterministic validity gates",
                    ],
                },
            }
        )
    except Exception as exc:
        exit_code = 2
        failed_judge_calls = list(getattr(exc, "calls", []))
        known_judge_costs = [
            float(call["cost_usd"])
            for call in failed_judge_calls
            if call.get("cost_usd") is not None
        ]
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
                "cost_accounting": {
                    "execution_observed_usd": round(observed_cost, 8),
                    "judge_estimated_usd": round(sum(known_judge_costs), 8),
                    "total_measured_usd": round(
                        observed_cost + sum(known_judge_costs), 8
                    ),
                    "unknown_execution_cost_records": unknown_cost_records,
                    "judge_cost_complete": len(known_judge_costs)
                    == len(failed_judge_calls),
                    "execution_harness_calls": sum(
                        len(record.get("runs") or []) for record in records
                    ),
                    "judge_model_calls": len(failed_judge_calls),
                    "failed_judge_calls": failed_judge_calls,
                },
            }
        )
    finally:
        cleanup_reports = legacy.cleanup_runtime(runtime)
        artifact["cleanup"] = cleanup_reports
        shutil.rmtree(root, ignore_errors=True)
        output.write_text(
            json.dumps(
                _publishable_artifact(artifact, repo_root=repo_root, run_root=root),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "valid": artifact["valid"],
                "records": len(records),
                "output": str(output),
            },
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
