# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Trajectory identity for workflow memory.

A canonical signature reduces a concrete run to the ordered sequence of
(tool, argument-shape, error-class) units, collapsing retries, so that runs
that are "the same procedure" hash identically. Argument *values*, result
payloads, and timestamps are all dropped — two runs of ``lookup_order`` +
``issue_refund`` with different order IDs produce the same
``canonical_hash``.

This identity is the foundation of the continual-learning loop: promotion
is counting, and counting requires identity. Without canonicalization every
run is unique and nothing ever crosses a promotion threshold.

# FUTURE: benign-reorder equivalence and sub-workflow mining. Full
# trajectories repeat less often than their subroutines, so mining repeated
# sub-sequences is the biggest quality lever after v1 — but it explodes the
# identity design space. ``canonical_signature`` is the single seam: any
# richer equivalence lands here without touching consumers.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ....enums.memory_type import MemoryType

# Step keys are written by MemAgent as ``"Step {n}: {tool_name}"``. The tool
# NAME is the stable trajectory handle — toolbox ``_id`` values change on
# re-registration, names do not.
_STEP_NAME_RE = re.compile(r"^Step\s+\d+\s*:\s*(.+)$")


def _tool_name_from_step_key(step_key: str) -> str:
    match = _STEP_NAME_RE.match(str(step_key).strip())
    if match:
        return match.group(1).strip()
    return str(step_key).strip()


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def canonical_signature(steps: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ordered canonical units from a ``Workflow.steps`` dict.

    Rules (each unit-tested in ``tests/unit/test_canonicalization.py``):

    R1  Order preserved. Steps are sorted by their embedded ``timestamp``
        when *every* step carries a parseable one; otherwise dict insertion
        order is used (step-name numbering is never trusted).
    R2  Identity per unit = (tool_name, sorted argument key names,
        errored flag). Argument VALUES are dropped. Tool identity uses the
        tool NAME parsed from the step key (``"Step N: {tool_name}"``), not
        the toolbox ``_id``.
    R3  Retry collapse: an attempt whose immediately preceding unit has the
        same (tool_name, arg_keys) and ``errored=True`` folds into that
        unit — ``retry_count`` increments and ``errored`` takes the newest
        attempt's value. fail→fail→success on one tool therefore becomes a
        single unit with ``retry_count=3, errored=False``.
    R4  ``errored`` = the step's ``error`` field is neither ``None`` nor
        the empty string.

    ``retry_count`` is carried on each unit for debuggability but is
    EXCLUDED from :func:`canonical_hash` — a run that needed one retry and a
    run that needed none are the same procedure.
    """
    if not steps:
        return []

    entries = []
    for index, (step_key, step_data) in enumerate(steps.items()):
        data = step_data if isinstance(step_data, dict) else {}
        arguments = data.get("arguments")
        arg_keys = (
            sorted(str(key) for key in arguments.keys())
            if isinstance(arguments, dict)
            else []
        )
        error = data.get("error")
        errored = error is not None and str(error) != ""
        entries.append(
            {
                "index": index,
                "timestamp": _parse_timestamp(data.get("timestamp")),
                "tool": _tool_name_from_step_key(step_key),
                "arg_keys": arg_keys,
                "errored": errored,
            }
        )

    # R1: only trust timestamps when every step has one — a partial sort
    # would interleave stamped and unstamped steps arbitrarily.
    if all(entry["timestamp"] is not None for entry in entries):
        entries.sort(key=lambda entry: (entry["timestamp"], entry["index"]))

    units: List[Dict[str, Any]] = []
    for entry in entries:
        if units:
            previous = units[-1]
            if (
                previous["tool"] == entry["tool"]
                and previous["arg_keys"] == entry["arg_keys"]
                and previous["errored"]
            ):
                previous["retry_count"] += 1
                previous["errored"] = entry["errored"]
                continue
        units.append(
            {
                "tool": entry["tool"],
                "arg_keys": entry["arg_keys"],
                "errored": entry["errored"],
                "retry_count": 1,
            }
        )
    return units


def canonical_hash(signature: List[Dict[str, Any]]) -> Optional[str]:
    """Deterministic sha256 over the signature, ignoring ``retry_count``.

    Returns ``None`` for an empty signature — a run with no tool calls has
    no trajectory identity.
    """
    if not signature:
        return None
    normalized = [
        {
            "tool": unit.get("tool"),
            "arg_keys": unit.get("arg_keys", []),
            "errored": bool(unit.get("errored", False)),
        }
        for unit in signature
    ]
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def toolset_hash(signature: List[Dict[str, Any]]) -> Optional[str]:
    """Order-insensitive hash of the distinct tool names.

    Analytics only (near-match clustering across trajectory classes) — NOT
    used for promotion identity in v1.
    """
    if not signature:
        return None
    tools = sorted({str(unit.get("tool")) for unit in signature})
    payload = json.dumps(tools, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_query(query: Optional[str]) -> str:
    """Lowercase + whitespace-collapse a user query for distinct counting."""
    if not query:
        return ""
    return re.sub(r"\s+", " ", str(query).strip().lower())


@dataclass
class RunRef:
    """Lightweight reference to one stored workflow run within a class.

    ``record_id`` is the provider-level document id (Mongo ``_id`` /
    filesystem ``_id``) — the only id ``update_by_id`` accepts on every
    provider. ``workflow_id`` is the application-level uuid.
    """

    workflow_id: Optional[str]
    outcome: str
    created_at: Optional[datetime]
    user_query: Optional[str]
    promoted_skill_id: Optional[str] = None
    record_id: Optional[str] = None


@dataclass
class TrajectoryStats:
    """Aggregated view of one trajectory class (one ``canonical_hash``)."""

    canonical_hash: str
    executions: int = 0
    successes: int = 0
    success_rate: float = 0.0
    distinct_query_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    sample_workflow_ids: List[str] = field(default_factory=list)
    failure_workflow_ids: List[str] = field(default_factory=list)
    already_promoted: bool = False
    promoted_skill_id: Optional[str] = None
    runs: List[RunRef] = field(default_factory=list)

    def baseline_snapshot(self) -> Dict[str, Any]:
        """Frozen promotion-time baseline stored on the skill document."""
        return {
            "executions": self.executions,
            "success_rate": self.success_rate,
            "window_start": self.first_seen.isoformat() if self.first_seen else None,
            "window_end": self.last_seen.isoformat() if self.last_seen else None,
        }


def aggregate_trajectory_stats(
    provider,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[TrajectoryStats]:
    """Group workflow documents by ``canonical_hash``.

    Pure-Python aggregation over ``provider.list_all(WORKFLOW_MEMORY)`` —
    provider-agnostic by construction. Add a MongoDB ``$group`` fast path
    only if production volume demands it.

    Documents written before canonicalization shipped (no stored
    ``canonical_hash``) are hashed on the fly from their ``steps``, so the
    aggregation is correct even before the backfill script has run.

    Works on raw dicts deliberately: ``Workflow.from_dict`` would construct
    full objects (and historically re-embed each row) for fields we never
    read here.
    """
    documents = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY) or []

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        if agent_id is not None and doc.get("agent_id") != agent_id:
            continue
        if user_id is not None and doc.get("user_id") not in (None, user_id):
            continue
        doc_hash = doc.get("canonical_hash")
        if not doc_hash:
            doc_hash = canonical_hash(canonical_signature(doc.get("steps") or {}))
        if not doc_hash:
            continue
        grouped.setdefault(doc_hash, []).append(doc)

    stats: List[TrajectoryStats] = []
    for doc_hash, docs in grouped.items():
        runs: List[RunRef] = []
        queries = set()
        for doc in docs:
            record_id = doc.get("_id") or doc.get("workflow_id")
            runs.append(
                RunRef(
                    workflow_id=doc.get("workflow_id"),
                    outcome=str(doc.get("outcome") or "success"),
                    created_at=_parse_timestamp(doc.get("created_at")),
                    user_query=doc.get("user_query"),
                    promoted_skill_id=doc.get("promoted_skill_id"),
                    record_id=str(record_id) if record_id is not None else None,
                )
            )
            normalized = normalize_query(doc.get("user_query"))
            if normalized:
                queries.add(normalized)

        # Most-recent-first; runs without timestamps sort last.
        runs.sort(
            key=lambda run: run.created_at or datetime.min,
            reverse=True,
        )
        successes = [run for run in runs if run.outcome == "success"]
        failures = [run for run in runs if run.outcome != "success"]
        timestamps = [run.created_at for run in runs if run.created_at]
        promoted = next(
            (run.promoted_skill_id for run in runs if run.promoted_skill_id), None
        )

        stats.append(
            TrajectoryStats(
                canonical_hash=doc_hash,
                executions=len(runs),
                successes=len(successes),
                success_rate=(len(successes) / len(runs)) if runs else 0.0,
                distinct_query_count=len(queries),
                first_seen=min(timestamps) if timestamps else None,
                last_seen=max(timestamps) if timestamps else None,
                sample_workflow_ids=[
                    run.workflow_id for run in successes if run.workflow_id
                ],
                failure_workflow_ids=[
                    run.workflow_id for run in failures if run.workflow_id
                ],
                already_promoted=promoted is not None,
                promoted_skill_id=promoted,
                runs=runs,
            )
        )

    stats.sort(key=lambda item: item.executions, reverse=True)
    return stats
