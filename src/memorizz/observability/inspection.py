"""Read-only structural inspection. No prompts, replay, or provider writes."""

import hashlib
import json
import math
from collections import Counter
from datetime import datetime

from .coverage import trace_coverage
from .diagnostics import diagnose_trace
from .evidence import evidence_groups, evidence_rows, latest_artifacts, latest_named
from .models import ResourceRef
from .normalization import timestamp


def _references(refs):
    result = []
    for ref in refs or []:
        try:
            result.append(
                ResourceRef.model_validate(ref).model_dump(
                    mode="json", exclude_none=True
                )
            )
        except (ValueError, TypeError):
            result.append({"invalid_reference": True})
    return result


def event_anchor(event_id):
    return "trace-event-" + hashlib.sha256(str(event_id).encode()).hexdigest()[:24]


def _milliseconds(value):
    value = timestamp(value)
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000
        if value
        else None
    )


def build_causal_waterfall(events):
    """Pair span phases and expose recorded parentage, never inferred lineage.

    Bad parentage remains visible. Iterative traversal terminates on cycles or
    arbitrarily deep graphs. Unknown timings stay unknown, not zero-duration.
    """
    groups = []
    for scope, rows in evidence_groups(events).items():
        nodes = {}
        event_to_span = {}
        for index, event in enumerate(rows):
            span_id = event.get("span_id") or event.get("event_id") or f"legacy-{index}"
            event_to_span[event.get("event_id")] = span_id
            node = nodes.setdefault(span_id, {"span_id": span_id, "events": []})
            node["events"].append(event)
        for node in nodes.values():
            captured = node.pop("events")
            first = next(
                (row for row in captured if row.get("phase") == "start"), captured[0]
            )
            terminals = [
                row
                for row in captured
                if row.get("phase") == "result"
                or row.get("kind") in {"model_result", "tool_result"}
            ]
            last = terminals[-1] if terminals else captured[-1]
            parent = first.get("parent_span_id")
            cause = first.get("caused_by_event_id")
            if not parent and cause:
                parent = event_to_span.get(cause)
            start = _milliseconds(first.get("timestamp"))
            end = _milliseconds(last.get("timestamp"))
            duration = last.get("duration_ms")
            if (
                type(duration) not in (float, int)
                or not math.isfinite(duration)
                or duration < 0
            ):
                duration = None
            if (
                duration is None
                and len(captured) > 1
                and start is not None
                and end is not None
            ):
                duration = max(0, end - start)
            node.update(
                {
                    "parent_span_id": parent,
                    "caused_by_event_id": cause,
                    "missing_cause": bool(cause and cause not in event_to_span),
                    "missing_parent": bool(
                        parent and parent not in nodes and parent != scope[4]
                    ),
                    "operation": first.get("operation")
                    or first.get("logical_tool_name")
                    or first.get("tool_name")
                    or first.get("kind")
                    or "unknown",
                    "kind": first.get("kind"),
                    "status": last.get("status") or "unknown",
                    "event_ids": [
                        e.get("event_id") for e in captured if e.get("event_id")
                    ],
                    "anchor": event_anchor(first.get("event_id")),
                    "timestamp": first.get("timestamp"),
                    "start_ms": start,
                    "duration_ms": round(duration, 3)
                    if isinstance(duration, (float, int))
                    else None,
                    "timing_state": "measured"
                    if duration is not None
                    else "missing_completion"
                    if first.get("phase") == "start" and not terminals
                    else "instantaneous"
                    if first.get("phase") == "event"
                    else "unknown_legacy",
                    "task_id": first.get("task_id"),
                }
            )
        starts = [
            node["start_ms"] for node in nodes.values() if node["start_ms"] is not None
        ]
        origin = min(starts) if starts else None
        for node in nodes.values():
            path, current = set(), node["span_id"]
            while current in nodes and current not in path:
                path.add(current)
                current = nodes[current]["parent_span_id"]
            node["cycle"] = current in path
            node["depth"] = 0 if node["cycle"] else min(64, max(0, len(path) - 1))
            node["offset_ms"] = (
                round(node["start_ms"] - origin, 3)
                if node["start_ms"] is not None
                else None
            )
            node.pop("start_ms")
        ordered, seen = [], set()
        children = {}
        for node in nodes.values():
            children.setdefault(node["parent_span_id"], []).append(node["span_id"])
        roots = [
            key for key, node in nodes.items() if node["parent_span_id"] not in nodes
        ]
        # A second pass includes disconnected cyclic components exactly once.
        for root in [*roots, *nodes]:
            stack = [root]
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                ordered.append(nodes[current])
                stack.extend(reversed(children.get(current, [])))
        ends = [(n["offset_ms"] or 0) + (n["duration_ms"] or 0) for n in ordered]
        extent = max(ends, default=0) or 1
        for node in ordered:
            node["offset_percent"] = round(100 * (node["offset_ms"] or 0) / extent, 3)
            node["duration_percent"] = round(
                100 * (node["duration_ms"] or 0) / extent, 3
            )
        groups.append(
            {
                "agent_id": scope[1],
                "thread_id": scope[3],
                "root_trace_id": scope[4],
                "turn_id": scope[5],
                "nodes": ordered,
            }
        )
    return groups


def build_trace_health(events, *, metadata=None):
    metadata = dict(
        metadata if metadata is not None else getattr(events, "coverage", {})
    )
    rows = evidence_rows(events)
    coverage = trace_coverage(rows, metadata)
    waterfall = build_causal_waterfall(rows)
    nodes = [node for group in waterfall for node in group["nodes"]]
    return {
        "scope": "bounded_loaded_window",
        "coverage": coverage["coverage"],
        "read_completeness": coverage["read_completeness"],
        "instrumentation_coverage": coverage["instrumentation_coverage"],
        "outcome_verification": coverage["outcome_verification"],
        "headline": coverage["headline"],
        "normalized_events": len(rows),
        "source_rows": metadata.get("source_rows"),
        "source_counts": metadata.get("source_counts", {}),
        "count_scope": metadata.get("count_scope", "unknown"),
        "parent_query_metadata": metadata.get("parent_query_metadata"),
        "store_selection": metadata.get("store_selection", "unknown"),
        "bundle_count": metadata.get("bundle_count"),
        "schema_counts": dict(
            Counter(str(row.get("schema_version", "unknown")) for row in rows)
        ),
        "kind_counts": dict(Counter(row.get("kind") or "unknown" for row in rows)),
        "normalization_errors": metadata.get("normalization_errors", 0),
        "error_codes": dict(
            Counter(e.get("code", "unknown") for e in metadata.get("errors", []))
        ),
        "duplicates_removed": metadata.get("duplicates_removed", 0),
        "truncated": bool(metadata.get("truncated")),
        "missing_stages": coverage["missing_stages"],
        "missing_identity_events": sum(
            not all(
                e.get(k) for k in ("agent_id", "thread_id", "root_trace_id", "turn_id")
            )
            for e in rows
        ),
        "orphan_spans": sum(n["missing_parent"] for n in nodes),
        "missing_causal_events": sum(n["missing_cause"] for n in nodes),
        "cyclic_spans": sum(n["cycle"] for n in nodes),
        "unknown_duration_spans": sum(n["duration_ms"] is None for n in nodes),
        "timing_states": dict(Counter(n["timing_state"] for n in nodes)),
        "latest_event_at": max(
            (timestamp(e.get("timestamp")) for e in rows), default=""
        )
        or None,
        # Historical bundles cannot prove how many events were never written.
        "write_failures": None,
        "dropped_events": None,
    }


def build_observability_alerts(health, *, query_latency_ms=250):
    """Deterministic alert facts suitable for a host's monitoring pipeline."""
    alerts = []
    counters = health.get("pipeline", {}).get("counters", {})
    for code in (
        "write_failures",
        "index_write_failures",
        "callback_failures",
        "dropped_metadata_fields",
    ):
        if counters.get(code, 0):
            alerts.append(
                {
                    "code": code,
                    "severity": "error",
                    "count": counters[code],
                    "scope": "process",
                }
            )
    for code in (
        "normalization_errors",
        "orphan_spans",
        "missing_causal_events",
        "missing_identity_events",
    ):
        if health.get(code, 0):
            alerts.append(
                {
                    "code": code,
                    "severity": "warning",
                    "count": health[code],
                    "scope": "loaded_window",
                }
            )
    for store, query in health.get("queries", {}).items():
        if (query.get("query_duration_ms") or 0) > query_latency_ms:
            alerts.append(
                {
                    "code": "slow_trace_query",
                    "severity": "warning",
                    "store": store,
                    "duration_ms": query["query_duration_ms"],
                }
            )
    if health.get("failed_queries"):
        alerts.append(
            {
                "code": "trace_query_failed",
                "severity": "error",
                "count": len(health["failed_queries"]),
            }
        )
    return alerts


def compare_trace_windows(baseline, candidate):
    """Content-free structural comparison, not a re-execution or quality score."""

    def summarize(events):
        rows = evidence_rows(events)
        for row in rows:
            row["input_refs"] = _references(row.get("input_refs"))
            row["output_refs"] = _references(row.get("output_refs"))
        structures = {
            "operations": [],
            "artifacts": [],
            "contracts": [],
            "deliveries": [],
            "sources": [],
        }
        for group in evidence_groups(rows).values():
            structures["artifacts"].extend(
                {
                    k: row.get(k)
                    for k in (
                        "task_id",
                        "artifact_status",
                        "persistence_verified",
                        "input_refs",
                        "output_refs",
                    )
                }
                for row in latest_artifacts(group)
            )
            structures["contracts"].extend(
                {
                    k: row.get(k)
                    for k in (
                        "task_id",
                        "contract_name",
                        "parser_status",
                        "closing_marker_found",
                        "recovery_used",
                    )
                }
                for row in latest_named(group, "output_contract").values()
            )
            structures["deliveries"].extend(
                {
                    k: row.get(k)
                    for k in (
                        "task_id",
                        "contract_name",
                        "emitted",
                        "acknowledged",
                        "input_refs",
                    )
                }
                for row in latest_named(group, "ui_delivery").values()
            )
        structures["operations"] = [
            {
                k: (
                    row.get("operation")
                    or row.get("logical_tool_name")
                    or row.get("tool_name")
                )
                if k == "operation"
                else row.get(k)
                for k in (
                    "kind",
                    "operation",
                    "phase",
                    "status",
                    "model",
                    "provider",
                    "finish_reason",
                )
            }
            for row in rows
        ]
        structures["sources"] = [
            ref for row in rows for ref in row.get("input_refs", [])
        ]
        return structures

    left, right = summarize(baseline), summarize(candidate)
    changes = {}
    for category in left:
        before = Counter(json.dumps(v, sort_keys=True) for v in left[category])
        after = Counter(json.dumps(v, sort_keys=True) for v in right[category])
        changes[category] = {
            "removed": [
                {"value": json.loads(v), "count": count}
                for v, count in sorted((before - after).items())
            ],
            "added": [
                {"value": json.loads(v), "count": count}
                for v, count in sorted((after - before).items())
            ],
        }
    before_findings, after_findings = diagnose_trace(baseline), diagnose_trace(
        candidate
    )
    before_ids, after_ids = {f["id"] for f in before_findings}, {
        f["id"] for f in after_findings
    }
    return {
        "mode": "read_only_structural_comparison",
        "baseline": build_trace_health(baseline),
        "candidate": build_trace_health(candidate),
        "changes": changes,
        "findings": {
            "only_in_baseline": sorted(before_ids - after_ids),
            "only_in_candidate": sorted(after_ids - before_ids),
            "in_both": sorted(before_ids & after_ids),
        },
        "warning": "Absence in a partial window does not establish resolution. No model, tool, or application action was executed.",
    }
