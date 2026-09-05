"""Bounded memory, artifact and contract inspectors built from recorded facts."""

from .evidence import (
    evidence_groups,
    is_delivered,
    latest_artifacts,
    latest_named,
    resource_key,
    task_key,
)
from .inspection import _references
from .normalization import timestamp
from .references import source_ids


def _unique(refs):
    result = {}
    for ref in _references(refs):
        if ref.get("ref"):
            result[resource_key(ref, version=True)] = ref
    return list(result.values())[:64]


def build_lineage_inspectors(events):
    events = list(events)
    memory, artifacts, contracts = [], [], []
    for scope, rows in evidence_groups(events).items():
        identity = {
            "agent_id": scope[1],
            "thread_id": scope[3],
            "root_trace_id": scope[4],
            "turn_id": scope[5],
        }
        actions = {}
        for row in rows:
            if row.get("kind") in {
                "application_action",
                "tool_call",
                "tool_result",
                "model_call",
                "model_result",
            }:
                previous = actions.get(row.get("span_id"))
                if (
                    not previous
                    or row.get("phase") != "start"
                    or previous.get("phase") == "start"
                ):
                    actions[row.get("span_id")] = row
        deliveries = list(latest_named(rows, "ui_delivery").values())
        for artifact in latest_artifacts(rows):
            task = task_key(artifact, rows)
            task_rows = [row for row in rows if task_key(row, rows) == task]
            expected = _unique(
                [
                    ref
                    for row in task_rows
                    for ref in row.get("input_refs", [])
                    if ref.get("role") == "authoritative_current_source"
                ]
            )
            retrieved = _unique(
                [
                    entry.get("resource", {})
                    for row in task_rows
                    for entry in row.get("selection_ledger", [])
                ]
                + [
                    ref
                    for row in task_rows
                    if row.get("kind") == "memory_retrieval"
                    for ref in row.get("output_refs", [])
                ]
            )
            supplied = _unique(
                [
                    ref
                    for row in task_rows
                    if row.get("kind") in {"memory_context", "memory_supply"}
                    for ref in row.get("input_refs", [])
                ]
            )
            bound = _unique(artifact.get("input_refs", []))
            matches = sum(
                any(
                    resource_key(e) == resource_key(b)
                    and (not e.get("version") or e.get("version") == b.get("version"))
                    for b in bound
                )
                for e in expected
            )
            state = (
                "not_observable"
                if not expected
                else "missing_source"
                if not bound
                else "exact_match"
                if matches == len(expected)
                else "partial_match"
                if matches
                else "conflicting_source"
            )
            memory.append(
                {
                    **identity,
                    "task_id": task,
                    "artifact": _unique(artifact.get("output_refs", [])),
                    "expected": expected,
                    "retrieved": retrieved,
                    "supplied": supplied,
                    "output_bound": bound,
                    "state": state,
                }
            )
            producer = actions.get(artifact.get("parent_span_id"))
            at = timestamp(artifact.get("timestamp"))
            times = [timestamp(row.get("timestamp")) for row in rows]
            before = [value for value in times if value and at and value < at]
            after = [value for value in times if value and at and value > at]
            delivered = [
                d
                for d in deliveries
                if task_key(d, rows) == task
                and timestamp(d.get("timestamp")) >= at
                and any(
                    resource_key(ref) == resource_key(out)
                    and (not out.get("version") or ref.get("version") == out["version"])
                    for ref in d.get("input_refs", [])
                    for out in artifact.get("output_refs", [])
                )
            ]
            artifacts.append(
                {
                    **identity,
                    "task_id": task,
                    "artifact_refs": _unique(artifact.get("output_refs", [])),
                    "source_refs": bound,
                    "created_at": at,
                    "artifact_status": artifact.get("artifact_status"),
                    "persistence_verified": artifact.get("persistence_verified"),
                    "producer_span_id": artifact.get("parent_span_id"),
                    "producer_component": producer.get("component")
                    if producer
                    else None,
                    "producer_status": producer.get("status")
                    if producer
                    else "unobserved",
                    "outside_observed_execution": producer is None,
                    "untraced_interval": {
                        "before": max(before, default=None),
                        "after": min(after, default=None),
                    }
                    if producer is None
                    else None,
                    "delivered": any(is_delivered(d) for d in delivered)
                    if delivered
                    else None,
                    "credit_decisions": [
                        row["credit_decision"]
                        for row in task_rows
                        if row.get("credit_decision")
                    ],
                    "verified_outcome": any(
                        row.get("kind") == "verified_outcome"
                        and row.get("verified") is True
                        for row in task_rows
                    ),
                    "lineage_state": state,
                }
            )
        represented = {task_key(artifact, rows) for artifact in latest_artifacts(rows)}
        for task in {task_key(row, rows) for row in rows} - represented:
            task_rows = [row for row in rows if task_key(row, rows) == task]
            expected = _unique(
                [
                    ref
                    for row in task_rows
                    for ref in row.get("input_refs", [])
                    if ref.get("role") == "authoritative_current_source"
                ]
            )
            retrieved = _unique(
                [
                    entry["resource"]
                    for row in task_rows
                    for entry in row.get("selection_ledger", [])
                ]
                + [
                    ref
                    for row in task_rows
                    if row.get("kind") == "memory_retrieval"
                    for ref in row.get("output_refs", [])
                ]
            )
            supplied = _unique(
                [
                    ref
                    for row in task_rows
                    if row.get("kind") in {"memory_context", "memory_supply"}
                    for ref in row.get("input_refs", [])
                ]
            )
            memory_rows = [
                row
                for row in task_rows
                if row.get("kind")
                in {
                    "memory_context",
                    "memory_supply",
                    "memory_retrieval",
                    "memory_selection",
                    "context_provenance",
                    "context_binding",
                }
            ]
            if expected or retrieved or supplied or memory_rows:
                memory.append(
                    {
                        **identity,
                        "task_id": task,
                        "artifact": [],
                        "expected": expected,
                        "retrieved": retrieved,
                        "supplied": supplied,
                        "output_bound": [],
                        "state": "not_observable",
                        "capture_note": "No artifact binding recorded. Aggregate memory counts do not establish which source produced an artifact.",
                        "observed_supplied_count": sum(
                            row.get("memory_supplied_count", 0)
                            for row in memory_rows
                            if type(row.get("memory_supplied_count", 0)) in (int, float)
                        ),
                        "legacy_source_ids": sorted(
                            {
                                ref
                                for row in memory_rows
                                for ref in source_ids(row.get("grounding_source_ids"))
                            }
                        ),
                    }
                )
        for key, contract in latest_named(rows, "output_contract").items():
            preceding = [
                row
                for row in rows
                if row.get("kind") == "model_result"
                and timestamp(row.get("timestamp"))
                and timestamp(contract.get("timestamp"))
                and timestamp(row.get("timestamp"))
                <= timestamp(contract.get("timestamp"))
            ]
            model = preceding[-1] if preceding else {}
            delivery = latest_named(rows, "ui_delivery").get(key, {})
            contracts.append(
                {
                    **identity,
                    "task_id": key[0],
                    "contract_name": key[1],
                    **{
                        field: contract.get(field)
                        for field in (
                            "required",
                            "opening_marker_found",
                            "closing_marker_found",
                            "parser_status",
                            "recovery_used",
                            "recovery_status",
                            "nodes",
                            "edges",
                            "output_limit",
                        )
                    },
                    "model_evidence": {
                        field: model.get(field)
                        for field in (
                            "provider",
                            "model",
                            "output_tokens",
                            "max_output_tokens",
                            "finish_reason",
                            "response_chars",
                            "response_bytes",
                        )
                    },
                    "model_relation": "preceding_in_turn_not_proven_cause"
                    if model
                    else "not_observed",
                    "emitted": delivery.get("emitted"),
                    "acknowledged": delivery.get("acknowledged"),
                }
            )
    ledger = [
        {"event_id": row.get("event_id"), "turn_id": row.get("turn_id"), **entry}
        for rows in evidence_groups(events).values()
        for row in rows
        for entry in row.get("selection_ledger", [])
    ][:256]
    return {
        "memory": memory[:128],
        "artifacts": artifacts[:128],
        "contracts": contracts[:128],
        "selection_ledger": ledger,
    }
