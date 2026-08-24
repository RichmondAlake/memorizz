"""Paired MetaHarness evaluation across filesystem and Oracle memory providers.

The paid path compares four blinded arms over the same read-only fixture:
Codex alone, Claude Code alone, a MemoRizz panel backed by the filesystem,
and the identical panel backed by Oracle AI Database.  Run order is paired
with its reverse so every arm has the same mean execution position over two
repeats.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional

from memorizz import MemAgentBuilder
from memorizz.approval import SQLiteApprovalStore
from memorizz.coordination import consolidate_structured_findings
from memorizz.enums.memory_type import MemoryType
from memorizz.llms.llm_factory import create_llm_provider
from memorizz.memory_provider.filesystem.provider import (
    FileSystemConfig,
    FileSystemProvider,
)
from memorizz.metaharness import (
    HarnessBudget,
    HarnessPermissions,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
    VerificationSpec,
    count_harness_steps,
)
from memorizz.metaharness.security import redact

REQUIREMENTS = (
    "Export policy requirements: (1) an expired grant must deny export before "
    "every role or ownership check, including administrators; (2) only "
    "administrators may bypass record ownership—auditors may inspect but never "
    "export; (3) owner IDs are case-sensitive opaque identifiers; and (4) "
    "expiry timestamps are timezone-aware UTC. Review evidence must identify "
    "missing tests and must not modify files."
)
GOLD_FINDINGS = [
    {
        "id": "G1",
        "finding": "Expiry is checked after role/owner grants, so expired admins and owners are incorrectly allowed.",
    },
    {
        "id": "G2",
        "finding": "The auditor role is incorrectly granted export permission.",
    },
    {
        "id": "G3",
        "finding": "casefold() makes opaque owner identifiers case-insensitive.",
    },
    {
        "id": "G4",
        "finding": "datetime.now() is naive and comparison with required timezone-aware UTC timestamps raises TypeError.",
    },
    {
        "id": "G5",
        "finding": "verify.py uses bare asserts that disappear under python -O.",
    },
    {
        "id": "G6",
        "finding": "Verification omits expired admin/owner, auditor, case-sensitive IDs, and aware-datetime cases.",
    },
]

FINDING_CRITERIA = {
    "expiry_precedence": (
        "implementation behavior against the requirement that expiry denies first"
    ),
    "role_bypass": "role authorization behavior for admin, auditor, and member",
    "owner_identifier": "owner-ID comparison behavior",
    "timezone_utc": "timezone-awareness and UTC expiry behavior",
    "verifier_integrity": (
        "whether verify.py reliably executes its checks under normal and optimized "
        "interpreter modes"
    ),
    "missing_tests": "the smallest requirement-derived missing test matrix",
}
FIXTURE_FILES = {
    "access_policy.py": (
        "from datetime import datetime\n"
        "\n"
        "def can_export(owner_id: str, actor_id: str, role: str, "
        "expires_at: datetime | None) -> bool:\n"
        '    """Return whether an actor may export one protected record."""\n'
        '    if role in {"admin", "auditor"}:\n'
        "        return True\n"
        "    if owner_id.casefold() == actor_id.casefold():\n"
        "        return True\n"
        "    if expires_at is not None and expires_at <= datetime.now():\n"
        "        return False\n"
        "    return False\n"
    ),
    "verify.py": (
        "from access_policy import can_export\n"
        'assert can_export("acct-1", "acct-1", "member", None)\n'
        'assert can_export("acct-1", "acct-2", "admin", None)\n'
        'assert not can_export("acct-1", "acct-2", "member", None)\n'
    ),
}
REQUIRED_FINDING_IDS = list(FINDING_CRITERIA)
FINDING_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "minItems": 1,
            "maxItems": len(REQUIRED_FINDING_IDS),
            "items": {
                "type": "object",
                "properties": {
                    "criterion_id": {
                        "type": "string",
                        "enum": REQUIRED_FINDING_IDS,
                    },
                    "title": {"type": "string", "maxLength": 140},
                    "finding": {"type": "string", "maxLength": 800},
                    "severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 400},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "missing_tests": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 300},
                        "maxItems": 3,
                    },
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 160},
                        "minItems": 1,
                        "maxItems": 2,
                    },
                },
                "required": [
                    "criterion_id",
                    "title",
                    "finding",
                    "severity",
                    "evidence",
                    "missing_tests",
                    "source_ids",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}
STRUCTURED_OUTPUT_CONTRACT = (
    "Return JSON only with this shape: "
    '{"findings":[{"criterion_id":"one required ID","title":"short title",'
    '"finding":"specific conclusion","severity":"high|medium|low",'
    '"evidence":["file:line and behavior"],"missing_tests":["case"],'
    '"source_ids":["MemoRizz source ID"]}]}. '
    "Return exactly one finding for every required criterion ID: "
    + ", ".join(REQUIRED_FINDING_IDS)
    + ". Criterion definitions: "
    + "; ".join(f"{key}={value}" for key, value in FINDING_CRITERIA.items())
    + ". If a criterion has no defect, state that conclusion with evidence instead "
    "of omitting it. Keep each finding concise and do not add findings outside "
    "these criteria."
)

PRICE_SNAPSHOT = {
    "as_of": "2026-08-22",
    "currency": "USD",
    "per_million_tokens": {
        "gpt-5.6-luna": {
            "input": 0.20,
            "cached_input": 0.02,
            "cache_write": 0.25,
            "output": 1.20,
        },
        "gpt-4.1": {
            "input": 2.00,
            "cached_input": 0.50,
            "cache_write": 2.00,
            "output": 8.00,
        },
    },
}

CODEX_BUDGET = HarnessBudget(
    max_wall_time_seconds=240, max_steps=20, max_output_tokens=18_000
)
CLAUDE_BUDGET = HarnessBudget(
    max_wall_time_seconds=240,
    max_steps=20,
    max_cost_usd=0.50,
    max_output_tokens=18_000,
)
# Resource parity is per executed harness. The adaptive panel may invoke a
# second harness, but no individual Codex/Claude run receives a larger budget
# than its standalone counterpart.
PANEL_CODEX_BUDGET = CODEX_BUDGET
PANEL_CLAUDE_BUDGET = CLAUDE_BUDGET

ARM_SPECS: Dict[str, Dict[str, str]] = {
    "codex_only": {
        "display_name": "Codex only",
        "kind": "single",
        "provider": "filesystem",
        "harness": "codex",
    },
    "claude_only": {
        "display_name": "Claude Code only",
        "kind": "single",
        "provider": "filesystem",
        "harness": "claude-code",
    },
    "memorizz_panel_filesystem": {
        "display_name": "MemoRizz Panel (Filesystem)",
        "kind": "panel",
        "provider": "filesystem",
    },
    "memorizz_panel_oracle": {
        "display_name": "MemoRizz Panel (Oracle AI Database)",
        "kind": "panel",
        "provider": "oracle",
    },
}

COMMON_TASK = (
    "Review access_policy.py and verify.py for security and correctness against "
    "the retrieved export-policy requirements. Do not modify files. Identify all "
    "material defects, cite the applicable MemoRizz memory source ID, and give "
    "severity, concrete evidence, and missing tests. " + STRUCTURED_OUTPUT_CONTRACT
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def usage_totals(usage: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    """Normalize OpenAI and Claude Code token telemetry."""
    value = dict(usage or {})
    input_tokens = int(value.get("input_tokens", value.get("prompt_tokens", 0)) or 0)
    output_tokens = int(
        value.get("output_tokens", value.get("completion_tokens", 0)) or 0
    )
    cached_tokens = int(
        value.get("cached_input_tokens", value.get("cached_tokens", 0)) or 0
    )
    if not cached_tokens:
        cached_tokens = int(
            (value.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        )
    cache_write_input_tokens = int(
        value.get(
            "cache_write_input_tokens", value.get("cache_creation_input_tokens", 0)
        )
        or 0
    )
    if "cache_read_input_tokens" in value or "cache_creation_input_tokens" in value:
        input_tokens += int(value.get("cache_read_input_tokens", 0) or 0)
        input_tokens += cache_write_input_tokens
        cached_tokens = int(value.get("cache_read_input_tokens", 0) or 0)
    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_input_tokens": cache_write_input_tokens,
        "output_tokens": output_tokens,
    }


def estimate_openai_cost(
    usage: Optional[Mapping[str, Any]], model: str
) -> Optional[float]:
    """Return an API-equivalent estimate, never an invented price."""
    prices = PRICE_SNAPSHOT["per_million_tokens"].get(model)
    if not prices:
        return None
    totals = usage_totals(usage)
    uncached = max(
        totals["input_tokens"]
        - totals["cached_tokens"]
        - totals["cache_write_input_tokens"],
        0,
    )
    input_multiplier = 1.0
    output_multiplier = 1.0
    if model == "gpt-5.6-luna" and totals["input_tokens"] > 272_000:
        input_multiplier, output_multiplier = 2.0, 1.5
    cost = (
        uncached * prices["input"] * input_multiplier
        + totals["cached_tokens"] * prices["cached_input"] * input_multiplier
        + totals["cache_write_input_tokens"] * prices["cache_write"] * input_multiplier
        + totals["output_tokens"] * prices["output"] * output_multiplier
    ) / 1_000_000
    return round(cost, 8)


def counterbalanced_orders(repeats: int, seed: int) -> List[List[str]]:
    """Pair each random arm order with its reverse."""
    arms = list(ARM_SPECS)
    orders: List[List[str]] = []
    for pair_index in range((repeats + 1) // 2):
        order = list(arms)
        random.Random(seed + pair_index).shuffle(order)
        orders.append(order)
        if len(orders) < repeats:
            orders.append(list(reversed(order)))
    return orders


def _write_fixture(workspace: Path) -> None:
    for name, content in FIXTURE_FILES.items():
        (workspace / name).write_text(content, encoding="utf-8")


def _oracle_preflight(provider: Any) -> Dict[str, Any]:
    report = provider.preflight()
    keys = (
        "ok",
        "database_product",
        "version_full",
        "pdb_open_state",
        "index_policy",
        "vector_dimensions",
        "exact_search_fallback",
        "embedding_dimension_compatible",
        "diagnostics",
    )
    return {key: report.get(key) for key in keys}


def create_runtime(
    root: Path,
    workspace: Path,
    *,
    provider_names: Iterable[str] = ("filesystem", "oracle"),
) -> Dict[str, Any]:
    """Create requested providers and fail before paid calls if one is unhealthy."""
    requested = tuple(
        dict.fromkeys(str(item).strip().lower() for item in provider_names)
    )
    unsupported = sorted(set(requested) - {"filesystem", "oracle"})
    if unsupported or not requested:
        raise ValueError(
            "provider_names must contain filesystem and/or oracle"
            + (f"; unsupported: {unsupported}" if unsupported else "")
        )
    providers: Dict[str, Any] = {}
    services: Dict[str, MetaHarness] = {}
    try:
        if "filesystem" in requested:
            providers["filesystem"] = FileSystemProvider(
                FileSystemConfig(
                    root_path=root / "memory-filesystem",
                    embedding_provider=None,
                    lazy_vector_indexes=True,
                    use_faiss=False,
                )
            )
        if "oracle" in requested:
            from memorizz.memory_provider.oracle import OracleProvider

            providers["oracle"] = OracleProvider.from_env(
                provision_if_missing=False,
                index_policy="lazy",
                in_database_embedding=True,
                embedding_config={"install_if_missing": False},
            )
        preflight = {
            name: (
                {"ok": True, "provider": "filesystem"}
                if name == "filesystem"
                else _oracle_preflight(provider)
            )
            for name, provider in providers.items()
        }
        if "oracle" in preflight and not preflight["oracle"].get("ok"):
            raise RuntimeError(
                "Oracle preflight failed before paid calls: "
                f"{preflight['oracle'].get('diagnostics') or ['not ready']}"
            )
        for provider_name, provider in providers.items():
            services[provider_name] = MetaHarness.from_env(
                memory_provider=provider,
                run_store=SQLiteHarnessRunStore(root / f"runs-{provider_name}.sqlite3"),
                approval_store=SQLiteApprovalStore(
                    root / f"approvals-{provider_name}.sqlite3"
                ),
                allowed_workspace_roots=[str(workspace)],
            )
        probe_service = services[next(iter(services))]
        probes = {name: probe_service.probe(name) for name in ("codex", "claude-code")}
        return {
            "providers": providers,
            "services": services,
            "preflight": preflight,
            "probes": probes,
            "scope_ids": {name: set() for name in providers},
            "agent_ids": {name: set() for name in providers},
        }
    except Exception:
        for service in services.values():
            service.close()
        for provider in providers.values():
            provider.close()
        raise


def seed_scope(
    runtime: Mapping[str, Any],
    *,
    run_scope: str,
    arm: str,
    provider_name: str,
    repeat: int,
) -> Dict[str, Any]:
    provider = runtime["providers"][provider_name]
    memory_id = f"harness-eval-{run_scope}-{repeat}-{arm}"
    started = time.perf_counter()
    source_id = provider.store(
        {
            "title": "Export policy requirements",
            "content": f"Applicable review requirements:\n{REQUIREMENTS}",
            "user_id": "eval-user",
            "thread_id": f"repeat-{repeat}",
        },
        MemoryType.KNOWLEDGE_BASE,
        memory_id=memory_id,
    )
    runtime["scope_ids"][provider_name].add(memory_id)
    return {
        "memory_id": memory_id,
        "source_id": str(source_id),
        "seed_latency_ms": int((time.perf_counter() - started) * 1_000),
    }


def context_provider_preflight(
    runtime: Mapping[str, Any],
    *,
    run_scope: str,
    workspace: Path,
) -> Dict[str, Any]:
    """Prove provider-equivalent grounded evidence before any paid call."""
    rows: Dict[str, Dict[str, Any]] = {}
    for provider_name in ("filesystem", "oracle"):
        scope = seed_scope(
            runtime,
            run_scope=run_scope,
            arm="context_preflight",
            provider_name=provider_name,
            repeat=0,
        )
        task = HarnessTask(
            task=COMMON_TASK,
            workspace=str(workspace),
            harness="codex",
            memory_id=scope["memory_id"],
            user_id="eval-user",
            thread_id="repeat-0",
            agent_id="context-parity-preflight",
            permissions={"mcp_access": "none"},
            context={
                "memory_query": REQUIREMENTS,
                "memory_context_strategy": "evidence_pack",
            },
        )
        pack = runtime["services"][provider_name]._context_for_task(task)
        rows[provider_name] = {
            "memory_id": scope["memory_id"],
            "source_id": scope["source_id"],
            "grounded": scope["source_id"] in pack.source_ids,
            "source_ids": list(pack.source_ids),
            "token_estimate": pack.token_estimate,
            "content_fingerprint": pack.content_fingerprint,
            "metadata": dict(pack.metadata or {}),
            "seed_latency_ms": scope["seed_latency_ms"],
        }
    fingerprints = {
        value["content_fingerprint"] for value in rows.values() if value["grounded"]
    }
    valid = all(value["grounded"] for value in rows.values()) and len(fingerprints) == 1
    report = {"ok": valid, "providers": rows}
    if not valid:
        raise RuntimeError(
            "Filesystem/Oracle context preflight did not produce equivalent "
            f"grounded evidence: {report}"
        )
    return report


def normalized_result_metrics(
    service: MetaHarness,
    result: Any,
    *,
    model: str,
    expected_source_id: str,
) -> Dict[str, Any]:
    value = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    direct_cost = value.get("cost_usd")
    cost = (
        float(direct_cost)
        if direct_cost is not None
        else estimate_openai_cost(value.get("usage"), model)
    )
    context_pack = dict(value.get("context_pack") or {})
    events = service.events(value["run_id"], limit=10_000)
    return {
        "run_id": value["run_id"],
        "harness": value["harness"],
        "model": model,
        "status": value["status"],
        "ok": bool(value.get("ok")),
        "verified": bool(value.get("verified")),
        "grounded": expected_source_id in context_pack.get("source_ids", []),
        "context_tokens": int(context_pack.get("token_estimate", 0) or 0),
        "context_fingerprint": context_pack.get("fingerprint"),
        "context_content_fingerprint": context_pack.get("content_fingerprint"),
        "context_metadata": dict(context_pack.get("metadata") or {}),
        "latency_ms": value.get("latency_ms"),
        "phase_timings_ms": dict(value.get("phase_timings_ms") or {}),
        "usage": dict(value.get("usage") or {}),
        # Avoid the word ``token`` in this field name. The persisted artifact is
        # passed through MemoRizz's credential redactor, where token-shaped keys
        # are intentionally treated as sensitive. The nested raw usage remains
        # available for audit, while this numeric roll-up stays readable.
        "usage_totals": usage_totals(value.get("usage")),
        "actions": count_harness_steps(events),
        "cost_usd": cost,
        "cost_basis": (
            "reported"
            if direct_cost is not None
            else "estimated_api_equivalent"
            if cost is not None
            else "unknown"
        ),
        "error_code": value.get("error_code"),
        "error": value.get("error"),
        "remediation": value.get("remediation"),
    }


def structured_response(text: Any, *, contributor: str) -> tuple[str, Dict[str, Any]]:
    """Normalize every arm through the same deterministic result renderer."""
    return consolidate_structured_findings(
        [
            {
                "task_id": contributor,
                "status": "completed",
                "result": str(text or ""),
            }
        ],
        required_ids=REQUIRED_FINDING_IDS,
    )


def run_single(
    runtime: Mapping[str, Any],
    *,
    arm: str,
    harness: str,
    model: str,
    scope: Mapping[str, Any],
    repeat: int,
    workspace: Path,
    permissions: HarnessPermissions,
    verification: VerificationSpec,
    provider_name: str = "filesystem",
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    service = runtime["services"][provider_name]
    budget = CODEX_BUDGET if harness == "codex" else CLAUDE_BUDGET
    started = time.perf_counter()
    result = service.run(
        HarnessTask(
            task=COMMON_TASK,
            workspace=str(workspace),
            harness=harness,
            model=model,
            memory_id=scope["memory_id"],
            user_id="eval-user",
            thread_id=f"repeat-{repeat}",
            agent_id=f"{arm}-baseline-{repeat}",
            permissions=permissions,
            budget=budget,
            verification=verification,
            output_schema=FINDING_OUTPUT_SCHEMA,
            context={
                "evaluation_arm": arm,
                "repeat": repeat,
                "grounding_required": True,
                "memory_query": REQUIREMENTS,
                "memory_context_strategy": "evidence_pack",
            },
        )
    )
    run = normalized_result_metrics(
        service,
        result,
        model=model,
        expected_source_id=scope["source_id"],
    )
    try:
        response, coverage = structured_response(
            result.final_response,
            contributor=f"{harness}-single",
        )
    except Exception as exc:
        response = str(result.final_response or "")
        coverage = {
            "coverage_complete": False,
            "required_ids": REQUIRED_FINDING_IDS,
            "covered_ids": [],
            "missing_ids": REQUIRED_FINDING_IDS,
            "parse_errors": {f"{harness}-single": f"{type(exc).__name__}: {exc}"},
        }
    return {
        "arm": arm,
        "display_name": display_name or ARM_SPECS[arm]["display_name"],
        "memory_provider": provider_name,
        "repeat": repeat,
        "response": response,
        "structured_coverage": coverage,
        "ok": result.ok,
        "wall_latency_ms": int((time.perf_counter() - started) * 1_000),
        "seed_latency_ms": scope["seed_latency_ms"],
        "runs": [run],
        "cost_usd": run["cost_usd"],
        "cost_complete": run["cost_usd"] is not None,
        "cost_basis": [run["cost_basis"]],
        "agent_ids": [f"{arm}-baseline-{repeat}"],
    }


def run_panel(
    runtime: Mapping[str, Any],
    *,
    run_scope: str,
    arm: str,
    provider_name: str,
    scope: Mapping[str, Any],
    repeat: int,
    workspace: Path,
    permissions: HarnessPermissions,
    verification: VerificationSpec,
    codex_model: str,
    claude_model: str,
    adaptive: bool = True,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    provider = runtime["providers"][provider_name]
    service = runtime["services"][provider_name]
    shared_config = {
        "workspace": str(workspace),
        "permissions": permissions.to_dict(),
        "verification": verification.to_dict(),
        "output_schema": FINDING_OUTPUT_SCHEMA,
    }
    name_suffix = run_scope
    codex_agent = (
        MemAgentBuilder()
        .with_name(f"Correctness reviewer {name_suffix}")
        .with_memory_provider(provider)
        .with_memory_ids(scope["memory_id"])
        .with_execution_harness(
            "codex",
            meta_harness=service,
            config={
                **shared_config,
                "model": codex_model,
                "budget": PANEL_CODEX_BUDGET.to_dict(),
            },
        )
        .with_semantic_cache(enabled=False)
        .as_ephemeral()
        .build(validate=False)
    )
    claude_agent = (
        MemAgentBuilder()
        .with_name(f"Adversarial reviewer {name_suffix}")
        .with_memory_provider(provider)
        .with_memory_ids(scope["memory_id"])
        .with_execution_harness(
            "claude-code",
            meta_harness=service,
            config={
                **shared_config,
                "model": claude_model,
                "budget": PANEL_CLAUDE_BUDGET.to_dict(),
            },
        )
        .with_semantic_cache(enabled=False)
        .as_ephemeral()
        .build(validate=False)
    )
    plan = [
        {
            "task_id": "policy-review",
            "description": COMMON_TASK,
            "assigned_agent_id": codex_agent.agent_id,
            "priority": 1,
            "dependencies": [],
        },
        {
            "task_id": "verification-review",
            "description": (
                COMMON_TASK
                + (
                    " Act as the fallback reviewer. Concentrate on criterion gaps named "
                    "by the host's adaptive escalation instruction, verify them against "
                    "the files, and avoid unrelated observations."
                    if adaptive
                    else ""
                )
            ),
            "assigned_agent_id": claude_agent.agent_id,
            "priority": 1,
            "dependencies": [],
        },
    ]
    coordinator = (
        MemAgentBuilder()
        .with_name(f"Evaluation coordinator {name_suffix}")
        .with_instruction(
            "Coordinate bounded evidence retrieval and deterministic structured "
            "finding union. Escalate only missing coverage."
        )
        .with_memory_provider(provider)
        .with_memory_ids(scope["memory_id"])
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
        .with_delegation(
            [codex_agent, claude_agent],
            enabled=True,
            mode="deterministic",
            plan=plan,
            return_report=True,
            persist_participants=False,
            evidence_context=True,
            consolidation_strategy="structured",
            required_finding_ids=REQUIRED_FINDING_IDS,
            adaptive_escalation={
                "enabled": adaptive,
                "escalation_task_ids": ["verification-review"],
                "criterion_descriptions": FINDING_CRITERIA,
            },
            max_consolidation_result_chars=8_000,
        )
        .as_ephemeral()
        .build(validate=False)
    )
    agent_ids = [coordinator.agent_id, codex_agent.agent_id, claude_agent.agent_id]
    runtime["agent_ids"][provider_name].update(agent_ids)
    before = {row["run_id"] for row in service.list_runs(limit=1_000)}
    try:
        started = time.perf_counter()
        report = coordinator.run(
            COMMON_TASK,
            memory_id=scope["memory_id"],
            user_id="eval-user",
            thread_id=f"repeat-{repeat}",
            context={
                "evaluation_arm": arm,
                "repeat": repeat,
                "grounding_required": True,
                "memory_query": REQUIREMENTS,
                "memory_context_strategy": "evidence_pack",
            },
        )
        wall_latency_ms = int((time.perf_counter() - started) * 1_000)
        if not isinstance(report, dict):
            raise RuntimeError(
                f"The multi-harness workflow returned an invalid report: {report}"
            )
        new_rows = [
            row for row in service.list_runs(limit=1_000) if row["run_id"] not in before
        ]
        if not 1 <= len(new_rows) <= 2:
            raise RuntimeError(
                "Expected one primary run and at most one adaptive escalation, "
                f"found {len(new_rows)} harness runs"
            )
        model_by_harness = {"codex": codex_model, "claude-code": claude_model}
        runs = [
            normalized_result_metrics(
                service,
                row["result"],
                model=model_by_harness[row["harness"]],
                expected_source_id=scope["source_id"],
            )
            for row in new_rows
        ]
        runs.sort(key=lambda value: value["harness"])
        synthesis = dict(report.get("consolidation") or {})
        execution = dict(report.get("execution") or {})
        synthesis_cost = 0.0
        component_costs = [run["cost_usd"] for run in runs]
        shared_memory_id = report.get("shared_memory_id")
        if shared_memory_id:
            runtime["scope_ids"][provider_name].add(str(shared_memory_id))
        return {
            "arm": arm,
            "display_name": display_name or ARM_SPECS[arm]["display_name"],
            "memory_provider": provider_name,
            "repeat": repeat,
            "source_id": scope["source_id"],
            "response": report.get("response", ""),
            "ok": bool(report.get("ok")),
            "wall_latency_ms": wall_latency_ms,
            "seed_latency_ms": scope["seed_latency_ms"],
            "runs": runs,
            "consolidation": synthesis,
            "execution": execution,
            "structured_coverage": dict(synthesis.get("coverage") or {}),
            "memory_first": {
                "root_evidence": dict(synthesis.get("memory_context") or {}),
                "semantic_cache": coordinator.semantic_cache_stats(),
                "learning": coordinator.learning_report(
                    memory_id=scope["memory_id"],
                    user_id="eval-user",
                    thread_id=f"repeat-{repeat}",
                ),
                "policy": {
                    "cold_run": True,
                    "shared_evidence_snapshot": True,
                    "summarize_in_turn": False,
                    "compact_in_turn": False,
                    "forget_in_turn": False,
                },
            },
            "synthesis_cost_usd": synthesis_cost,
            "workflow_failures": list(report.get("failures") or []),
            "shared_memory_id": shared_memory_id,
            "agent_ids": agent_ids,
            "cost_complete": all(value is not None for value in component_costs),
            "cost_usd": (
                round(sum(component_costs), 8)
                if all(value is not None for value in component_costs)
                else None
            ),
            "cost_basis": [run["cost_basis"] for run in runs]
            + ["deterministic_no_model"],
        }
    finally:
        coordinator.close(close_memory_provider=False)
        codex_agent.close(close_memory_provider=False)
        claude_agent.close(close_memory_provider=False)


def validation_failures(record: Mapping[str, Any]) -> List[str]:
    failures: List[str] = []
    if not record.get("ok"):
        failures.append("arm_not_ok")
    if str(record.get("arm", "")).startswith("memorizz_panel"):
        consolidation = dict(record.get("consolidation") or {})
        coverage = dict(consolidation.get("coverage") or {})
        execution = dict(record.get("execution") or {})
        if consolidation.get("strategy") != "structured":
            failures.append("panel:structured_consolidation_missing")
        if consolidation.get("model_used"):
            failures.append("panel:unexpected_synthesis_model")
        if not coverage.get("coverage_complete"):
            failures.append(
                "panel:coverage_incomplete="
                + ",".join(coverage.get("missing_ids") or [])
            )
        if coverage.get("parse_errors"):
            failures.append("panel:structured_parse_errors")
        if execution.get("strategy") != "adaptive_coverage":
            failures.append("panel:adaptive_execution_missing")
        expected_run_count = 2 if execution.get("escalated") else 1
        if len(record.get("runs") or []) != expected_run_count:
            failures.append(
                f"panel:run_count={len(record.get('runs') or [])}:"
                f"expected={expected_run_count}"
            )
        root_sources = set(
            (consolidation.get("memory_context") or {}).get("source_ids", [])
        )
        if record.get("source_id") not in root_sources:
            failures.append("panel:root_grounding_missing")
    if record.get("workflow_failures"):
        failures.append(f"panel:workflow_failures={len(record['workflow_failures'])}")
    structured = dict(record.get("structured_coverage") or {})
    if structured.get("parse_errors"):
        failures.append("structured_parse_errors")
    for run in record.get("runs", []):
        if run.get("status") != "succeeded":
            failures.append(
                f"{run['harness']}:status={run.get('status')}:"
                f"{run.get('error_code')}:{run.get('error')}"
            )
        if not run.get("verified"):
            failures.append(f"{run['harness']}:verification_missing")
        if not run.get("grounded"):
            failures.append(f"{run['harness']}:grounding_missing")
        if not run.get("context_content_fingerprint"):
            failures.append(f"{run['harness']}:content_fingerprint_missing")
    return failures


def provider_context_failures(
    records: Iterable[Mapping[str, Any]], repeats: int
) -> List[Dict[str, Any]]:
    """Assert equivalent memory content across both MemoRizz provider arms."""
    failures: List[Dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        fingerprints: Dict[str, List[str]] = {}
        for arm in ("memorizz_panel_filesystem", "memorizz_panel_oracle"):
            record = next(
                (
                    item
                    for item in records
                    if item.get("repeat") == repeat and item.get("arm") == arm
                ),
                None,
            )
            fingerprints[arm] = sorted(
                {
                    str(run.get("context_content_fingerprint"))
                    for run in (record or {}).get("runs", [])
                    if run.get("context_content_fingerprint")
                }
            )
        filesystem = fingerprints["memorizz_panel_filesystem"]
        oracle = fingerprints["memorizz_panel_oracle"]
        if len(filesystem) != 1 or len(oracle) != 1 or filesystem != oracle:
            failures.append(
                {
                    "repeat": repeat,
                    "issues": ["panel:provider_context_content_mismatch"],
                    "fingerprints": fingerprints,
                }
            )
    return failures


def normalize_candidate(text: Any) -> str:
    value = re.sub(
        r"(?i)(?:memory:)?(?:knowledge_base:)?[0-9a-f]{8}-[0-9a-f-]{27,}",
        "memory:[SOURCE]",
        str(text or ""),
    )
    return re.sub(r"/private/[^\s)]+", "[WORKSPACE_PATH]", value)


def parse_json_object(value: Any) -> Dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    return json.loads(text)


def judge_records(
    records: List[Dict[str, Any]],
    *,
    repeats: int,
    seed: int,
    judge_model: str,
) -> List[Dict[str, Any]]:
    candidates_per_repeat = max(
        (
            sum(record.get("repeat") == repeat for record in records)
            for repeat in range(1, repeats + 1)
        ),
        default=0,
    )
    judge = create_llm_provider(
        {
            "provider": "openai",
            "model": judge_model,
            "temperature": 0,
            "max_tokens": min(16_000, max(4_000, candidates_per_repeat * 850)),
            "response_format": {"type": "json_object"},
        }
    )
    reports: List[Dict[str, Any]] = []
    try:
        for repeat in range(1, repeats + 1):
            candidates = [value for value in records if value["repeat"] == repeat]
            random.Random(seed * 10 + repeat).shuffle(candidates)
            label_pool = [f"candidate_{index:02d}" for index in range(17, 81, 4)]
            if not 2 <= len(candidates) <= len(label_pool):
                raise RuntimeError(
                    f"Expected two to {len(label_pool)} comparison arms, "
                    f"found {len(candidates)}"
                )
            labels = label_pool[: len(candidates)]
            random.Random(seed * 100 + repeat).shuffle(labels)
            label_to_arm = {
                label: record["arm"] for label, record in zip(labels, candidates)
            }
            payload = [
                {
                    "candidate_id": label,
                    "response": normalize_candidate(record["response"]),
                }
                for label, record in zip(labels, candidates)
            ]
            rubric = {
                "requirements": REQUIREMENTS,
                "gold_findings": GOLD_FINDINGS,
                "candidates": payload,
                "instructions": (
                    "Score each candidate independently. Do not reward verbosity, "
                    "polish, consensus language, or guessed topology/provider. Penalize "
                    "unsupported claims. Return one JSON object with key scores. Each "
                    "score item must contain candidate_id, correctness_0_35, "
                    "gold_coverage_0_30, evidence_0_15, actionability_0_10, "
                    "grounding_0_10, covered_gold_ids, unsupported_claims, and rationale. "
                    "Also return ranking (best first) and confidence_0_1."
                ),
            }
            raw = judge.generate(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict, identity-blind code-review evaluator. "
                            "Output valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(rubric, ensure_ascii=False),
                    },
                ],
                tools=None,
            )
            judged = parse_json_object(raw)
            scores = list(judged.get("scores") or [])
            if {item.get("candidate_id") for item in scores} != set(labels):
                raise RuntimeError(
                    f"Judge did not score every blinded candidate: {judged}"
                )
            repeat_records = [value for value in records if value["repeat"] == repeat]
            for item in scores:
                total = sum(
                    float(item.get(key, 0) or 0)
                    for key in (
                        "correctness_0_35",
                        "gold_coverage_0_30",
                        "evidence_0_15",
                        "actionability_0_10",
                        "grounding_0_10",
                    )
                )
                record = next(
                    value
                    for value in repeat_records
                    if value["arm"] == label_to_arm[item["candidate_id"]]
                )
                record["judge"] = {**item, "score_0_100": round(total, 2)}
            usage = judge.get_last_usage() or {}
            reports.append(
                {
                    "repeat": repeat,
                    "blinded_mapping": label_to_arm,
                    "ranking": judged.get("ranking"),
                    "confidence": judged.get("confidence_0_1"),
                    "usage": usage,
                    "cost_usd": estimate_openai_cost(usage, judge_model),
                }
            )
    finally:
        close_judge = getattr(judge, "close", None)
        if callable(close_judge):
            close_judge()
    return reports


def _sample_stdev(values: Iterable[float]) -> float:
    items = list(values)
    return float(stdev(items)) if len(items) > 1 else 0.0


def summarize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for arm, spec in ARM_SPECS.items():
        values = [record for record in records if record["arm"] == arm]
        known_costs = [
            float(record["cost_usd"])
            for record in values
            if record.get("cost_usd") is not None
        ]
        scores = [float(record["judge"]["score_0_100"]) for record in values]
        latencies = [float(record["wall_latency_ms"]) for record in values]
        total_input = sum(
            sum(run["usage_totals"]["input_tokens"] for run in record["runs"])
            + int(
                (record.get("consolidation") or {})
                .get("usage", {})
                .get("prompt_tokens", 0)
                or 0
            )
            for record in values
        )
        total_output = sum(
            sum(run["usage_totals"]["output_tokens"] for run in record["runs"])
            + int(
                (record.get("consolidation") or {})
                .get("usage", {})
                .get("completion_tokens", 0)
                or 0
            )
            for record in values
        )
        total_context = sum(
            sum(run["context_tokens"] for run in record["runs"])
            + (
                int(
                    (
                        (record.get("consolidation") or {}).get("memory_context") or {}
                    ).get("tokens_used", 0)
                    or 0
                )
                if (
                    (record.get("consolidation") or {}).get("memory_context") or {}
                ).get("used_for_synthesis")
                else 0
            )
            for record in values
        )
        context_reuses = sum(
            sum(
                bool(run["context_metadata"].get("shared_context_reused"))
                for run in record["runs"]
            )
            for record in values
        )
        phase_names = sorted(
            {
                phase
                for record in values
                for run in record["runs"]
                for phase in run.get("phase_timings_ms", {})
            }
        )
        mean_phase_timings_ms = {
            phase: round(
                mean(
                    float(run["phase_timings_ms"].get(phase, 0))
                    for record in values
                    for run in record["runs"]
                )
            )
            for phase in phase_names
        }
        summary = {
            "arm": arm,
            "display_name": spec["display_name"],
            "memory_provider": spec["provider"],
            "repeats": len(values),
            "judge_score_0_100": round(mean(scores), 2),
            "judge_score_stdev": round(_sample_stdev(scores), 3),
            "gold_findings_covered": round(
                mean(
                    len(record["judge"].get("covered_gold_ids") or [])
                    for record in values
                ),
                2,
            ),
            "unsupported_claims": round(
                mean(
                    len(record["judge"].get("unsupported_claims") or [])
                    for record in values
                ),
                2,
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
            "success_rate": round(mean(bool(record["ok"]) for record in values), 3),
            "mean_wall_latency_ms": round(mean(latencies)),
            "latency_stdev_ms": round(_sample_stdev(latencies), 3),
            "mean_seed_latency_ms": round(
                mean(float(record.get("seed_latency_ms", 0)) for record in values)
            ),
            "mean_cost_usd": (
                round(mean(known_costs), 6) if len(known_costs) == len(values) else None
            ),
            "cost_stdev_usd": (
                round(_sample_stdev(known_costs), 6)
                if len(known_costs) == len(values)
                else None
            ),
            "cost_coverage": f"{len(known_costs)}/{len(values)}",
            "mean_input_tokens": round(total_input / len(values)),
            "mean_output_tokens": round(total_output / len(values)),
            "mean_memory_context_tokens": round(total_context / len(values)),
            "mean_context_snapshot_reuses": round(context_reuses / len(values), 2),
            "mean_harness_runs": round(
                mean(len(record["runs"]) for record in values), 2
            ),
            "adaptive_escalation_rate": round(
                mean(
                    bool((record.get("execution") or {}).get("escalated"))
                    for record in values
                ),
                3,
            ),
            "model_synthesis_rate": round(
                mean(
                    bool((record.get("consolidation") or {}).get("model_used"))
                    for record in values
                ),
                3,
            ),
            "mean_phase_timings_ms": mean_phase_timings_ms,
            "mean_actions": round(
                mean(
                    sum(run["actions"] for run in record["runs"]) for record in values
                ),
                2,
            ),
            "mean_output_chars": round(
                mean(len(record["response"]) for record in values)
            ),
            "raw_judge_scores": scores,
            "raw_wall_latency_ms": latencies,
            "raw_cost_usd": known_costs,
        }
        summary["cost_per_judge_point_usd"] = (
            round(summary["mean_cost_usd"] / summary["judge_score_0_100"], 8)
            if summary["mean_cost_usd"] is not None and summary["judge_score_0_100"]
            else None
        )
        summaries.append(summary)
    return summaries


def provider_delta(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm = {value["arm"]: value for value in summaries}
    filesystem = by_arm["memorizz_panel_filesystem"]
    oracle = by_arm["memorizz_panel_oracle"]

    def percent_delta(oracle_value: Optional[float], filesystem_value: Optional[float]):
        if oracle_value is None or filesystem_value in (None, 0):
            return None
        return round((oracle_value / filesystem_value - 1) * 100, 3)

    return {
        "oracle_minus_filesystem_judge_points": round(
            oracle["judge_score_0_100"] - filesystem["judge_score_0_100"], 3
        ),
        "oracle_vs_filesystem_latency_pct": percent_delta(
            oracle["mean_wall_latency_ms"], filesystem["mean_wall_latency_ms"]
        ),
        "oracle_vs_filesystem_seed_latency_pct": percent_delta(
            oracle["mean_seed_latency_ms"], filesystem["mean_seed_latency_ms"]
        ),
        "oracle_vs_filesystem_cost_pct": percent_delta(
            oracle["mean_cost_usd"], filesystem["mean_cost_usd"]
        ),
        "oracle_minus_filesystem_input_tokens": (
            oracle["mean_input_tokens"] - filesystem["mean_input_tokens"]
        ),
        "oracle_minus_filesystem_output_tokens": (
            oracle["mean_output_tokens"] - filesystem["mean_output_tokens"]
        ),
    }


def decision_surface(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_score = max(value["judge_score_0_100"] for value in summaries)
    fastest = min(value["mean_wall_latency_ms"] for value in summaries)
    priced = [value for value in summaries if value["mean_cost_usd"] is not None]
    cheapest = min(value["mean_cost_usd"] for value in priced)
    return {
        "highest_judged_quality": [
            value["arm"]
            for value in summaries
            if value["judge_score_0_100"] == best_score
        ],
        "fastest": [
            value["arm"]
            for value in summaries
            if value["mean_wall_latency_ms"] == fastest
        ],
        "lowest_known_cost": [
            value["arm"] for value in priced if value["mean_cost_usd"] == cheapest
        ],
        "note": (
            "Two-repeat paired engineering evaluation; useful for smoke-level provider "
            "comparison, not a population-level benchmark."
        ),
    }


def cleanup_runtime(runtime: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    if not runtime:
        return reports
    oracle = runtime.get("providers", {}).get("oracle")
    if oracle is not None:
        for memory_id in sorted(runtime.get("scope_ids", {}).get("oracle", set())):
            try:
                reports.append(
                    {
                        "memory_id": memory_id,
                        "report": oracle.delete_scope(
                            memory_id=memory_id, user_id="eval-user"
                        ),
                    }
                )
            except Exception as exc:  # cleanup evidence must survive partial failures
                reports.append(
                    {
                        "memory_id": memory_id,
                        "error": str(redact(str(exc)))[:1_000],
                    }
                )
        agent_ids = sorted(runtime.get("agent_ids", {}).get("oracle", set()))
        if agent_ids:
            try:
                reports.append(
                    {
                        "agent_ids": agent_ids,
                        "report": oracle.delete_scope(agent_ids=agent_ids),
                    }
                )
            except Exception as exc:
                reports.append(
                    {
                        "agent_ids": agent_ids,
                        "error": str(redact(str(exc)))[:1_000],
                    }
                )
    for service in runtime.get("services", {}).values():
        service.close()
    for provider in runtime.get("providers", {}).values():
        provider.close()
    return reports


def _default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(tempfile.gettempdir()) / f"memorizz-provider-comparison-{timestamp}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        default=_truthy(os.getenv("MEMORIZZ_RUN_HARNESS_EVALUATION")),
        help="Make paid Codex, Claude, and judge calls.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", type=Path, default=_default_output())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.repeats <= 4:
        raise ValueError("--repeats must be between 1 and 4")
    codex_model = os.getenv("MEMORIZZ_EVAL_CODEX_MODEL", "gpt-5.6-luna").strip()
    claude_model = os.getenv("MEMORIZZ_EVAL_CLAUDE_MODEL", "sonnet").strip()
    judge_model = os.getenv("MEMORIZZ_EVAL_JUDGE_MODEL", "gpt-4.1").strip()
    run_scope = uuid.uuid4().hex[:12]
    root = Path(tempfile.mkdtemp(prefix="memorizz-provider-comparison-"))
    workspace = root / "workspace"
    workspace.mkdir()
    _write_fixture(workspace)
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
    runtime: Optional[Dict[str, Any]] = None
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    judge_reports: List[Dict[str, Any]] = []
    artifact: Dict[str, Any] = {
        "schema_version": 3,
        "created_at": _utc_now(),
        "status": "initializing",
        "valid": False,
        "run_scope": run_scope,
    }
    exit_code = 0
    try:
        runtime = create_runtime(root, workspace)
        context_preflight = context_provider_preflight(
            runtime,
            run_scope=run_scope,
            workspace=workspace,
        )
        orders = counterbalanced_orders(args.repeats, args.seed)
        artifact.update(
            {
                "status": "dry_run" if not args.execute else "running",
                "fixture": {
                    "requirements": REQUIREMENTS,
                    "gold_findings": GOLD_FINDINGS,
                    "required_finding_ids": REQUIRED_FINDING_IDS,
                    "finding_criteria": FINDING_CRITERIA,
                    "structured_output_contract": STRUCTURED_OUTPUT_CONTRACT,
                    "output_schema": FINDING_OUTPUT_SCHEMA,
                },
                "configuration": {
                    "protocol": "structured_adaptive_v1",
                    "repeats": args.repeats,
                    "seed": args.seed,
                    "run_orders": orders,
                    "mean_execution_position": {
                        arm: round(mean(order.index(arm) + 1 for order in orders), 3)
                        for arm in ARM_SPECS
                    },
                    "cold_scopes": True,
                    "per_harness_budget_parity": True,
                    "provider_context_content_equivalence_required": True,
                    "codex_model": codex_model,
                    "claude_model": claude_model,
                    "coordinator": "deterministic_structured_union",
                    "judge_model": judge_model,
                    "budgets": {
                        "codex_only": CODEX_BUDGET.to_dict(),
                        "claude_only": CLAUDE_BUDGET.to_dict(),
                        "panel_codex": PANEL_CODEX_BUDGET.to_dict(),
                        "panel_claude": PANEL_CLAUDE_BUDGET.to_dict(),
                    },
                },
                "provider_preflight": runtime["preflight"],
                "context_provider_preflight": context_preflight,
                "harness_probes": runtime["probes"],
                "pricing": PRICE_SNAPSHOT,
            }
        )
        if not args.execute:
            print(
                json.dumps(
                    {"status": "dry_run", **artifact["provider_preflight"]}, indent=2
                )
            )
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is required for Codex and judge stages"
                )
            not_ready = {
                name: probe.get("error")
                for name, probe in runtime["probes"].items()
                if not probe.get("ready")
            }
            if not_ready:
                raise RuntimeError(
                    f"All harnesses must be ready before paid calls: {not_ready}"
                )
            for repeat, order in enumerate(orders, start=1):
                scopes = {
                    arm: seed_scope(
                        runtime,
                        run_scope=run_scope,
                        arm=arm,
                        provider_name=spec["provider"],
                        repeat=repeat,
                    )
                    for arm, spec in ARM_SPECS.items()
                }
                print(f"Repeat {repeat} arm order: {order}", flush=True)
                for arm in order:
                    spec = ARM_SPECS[arm]
                    if spec["kind"] == "single":
                        model = (
                            codex_model if spec["harness"] == "codex" else claude_model
                        )
                        record = run_single(
                            runtime,
                            arm=arm,
                            harness=spec["harness"],
                            model=model,
                            scope=scopes[arm],
                            repeat=repeat,
                            workspace=workspace,
                            permissions=permissions,
                            verification=verification,
                        )
                    else:
                        record = run_panel(
                            runtime,
                            run_scope=run_scope,
                            arm=arm,
                            provider_name=spec["provider"],
                            scope=scopes[arm],
                            repeat=repeat,
                            workspace=workspace,
                            permissions=permissions,
                            verification=verification,
                            codex_model=codex_model,
                            claude_model=claude_model,
                        )
                    record.update(scopes[arm])
                    records.append(record)
                    issues = validation_failures(record)
                    if issues:
                        failures.append(
                            {"arm": arm, "repeat": repeat, "issues": issues}
                        )
                    print(
                        {
                            "arm": arm,
                            "provider": spec["provider"],
                            "ok": record["ok"],
                            "latency_ms": record["wall_latency_ms"],
                            "cost_usd": record["cost_usd"],
                            "valid": not issues,
                        },
                        flush=True,
                    )
                    if issues:
                        raise RuntimeError(
                            f"Fail-fast validation rejected {arm} repeat {repeat}: "
                            + "; ".join(issues)
                        )
            failures.extend(provider_context_failures(records, args.repeats))
            expected_records = args.repeats * len(ARM_SPECS)
            evaluation_valid = not failures and len(records) == expected_records
            if evaluation_valid:
                judge_reports = judge_records(
                    records,
                    repeats=args.repeats,
                    seed=args.seed,
                    judge_model=judge_model,
                )
                summaries = summarize(records)
                arm_cost = round(
                    sum(record.get("cost_usd") or 0 for record in records), 8
                )
                judge_cost = round(
                    sum(value.get("cost_usd") or 0 for value in judge_reports), 8
                )
                artifact.update(
                    {
                        "status": "valid",
                        "valid": True,
                        "records": records,
                        "judge_reports": judge_reports,
                        "validation_failures": [],
                        "summaries": summaries,
                        "provider_delta": provider_delta(summaries),
                        "decision": decision_surface(summaries),
                        "arm_execution_cost_usd": arm_cost,
                        "evaluation_overhead_cost_usd": judge_cost,
                        "total_measured_cost_usd": round(arm_cost + judge_cost, 8),
                    }
                )
            else:
                artifact.update(
                    {
                        "status": "invalid",
                        "records": records,
                        "judge_reports": [],
                        "validation_failures": failures,
                        "summaries": [],
                        "decision": None,
                        "evaluation_overhead_cost_usd": 0.0,
                    }
                )
                exit_code = 2
    except Exception as exc:
        artifact.update(
            {
                "status": "error",
                "valid": False,
                "records": records,
                "judge_reports": judge_reports,
                "validation_failures": failures,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(redact(str(exc)))[:2_000],
                },
            }
        )
        exit_code = 1
    finally:
        artifact["cleanup"] = cleanup_runtime(runtime)
        shutil.rmtree(root, ignore_errors=True)
        artifact["finished_at"] = _utc_now()
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(redact(artifact), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Evidence artifact: {args.output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
