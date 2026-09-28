"""Shape learning control-plane records for the control-plane monitor page.

The page reads every learning record in one scope once (events, compiled
artifacts, compiler checkpoints and tombstones) and this module turns them
into what the monitor shows: a summary tape, one grid row per immutable event
with its compile status, per-stream compile state, artifacts and the latest
forgetting plan. Pure functions over plain dicts, so the page stays testable.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from ..observability.normalization import timestamp

EVENT_LIMIT = 200
ARTIFACT_LIMIT = 100
RECORD_CHARS = 20000
SUMMARY_CHARS = 200

EVENT_RECORD = "learning_event"
ARTIFACT_RECORD = "learning_artifact"
CHECKPOINT_RECORD = "learning_checkpoint"
TOMBSTONE_RECORD = "learning_tombstone"

# Event type -> (label, family). Families drive the grid's filter chips.
EVENT_TYPES: Dict[str, Tuple[str, str]] = {
    "run_started": ("Run started", "run"),
    "run_completed": ("Run completed", "run"),
    "cache_hit": ("Cache hit", "cache"),
    "cache_bypassed": ("Cache bypassed", "cache"),
    "memory_retrieved": ("Memory retrieved", "evidence"),
    "evidence_pack_built": ("Evidence pack built", "evidence"),
    "tool_executed": ("Tool executed", "tool"),
    "outcome_recorded": ("Outcome recorded", "outcome"),
    "workflow_recorded": ("Workflow recorded", "workflow"),
    "memory_compiled": ("Memory compiled", "compile"),
    "skill_candidate_created": ("Skill candidate created", "skill"),
    "skill_promoted": ("Skill promoted", "skill"),
    "skill_demoted": ("Skill demoted", "skill"),
    "skill_deprecated": ("Skill deprecated", "skill"),
    "forgetting_planned": ("Forgetting planned", "forgetting"),
    "memory_forgotten": ("Memory forgotten", "forgetting"),
}

FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("run", "Runs"),
    ("tool", "Tools"),
    ("evidence", "Evidence"),
    ("outcome", "Outcomes"),
    ("workflow", "Workflows"),
    ("skill", "Skills"),
    ("cache", "Cache"),
    ("forgetting", "Forgetting"),
    ("compile", "Compilation"),
    ("other", "Other"),
)

ARTIFACT_KINDS = {
    "run_digest": "Run digest",
    "outcome": "Outcome",
    "procedure_evidence": "Procedure evidence",
    "memory_fact": "Memory fact",
    "compiler_checkpoint": "Compiler checkpoint",
}

_GOOD = {"success", "succeeded", "ok", "completed", "approved"}
_BAD = {"failure", "failed", "error", "provider_error", "timeout", "cancelled"}
_DEGRADED = {"fallback", "degraded", "empty", "partial"}


def _parse(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    normalized = timestamp(value)
    return datetime.fromisoformat(normalized) if normalized else None


def _shown(value: Any) -> str:
    """ISO timestamps as 'YYYY-MM-DD HH:MM:SS' for dense, aligned columns."""
    if value in (None, ""):
        return ""
    parsed = _parse(value)
    if parsed is None:
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _count(value: Any) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:,.0f}"


def _ms(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "—"
    return f"{number:,.0f} ms" if number < 1000 else f"{number / 1000:.1f} s"


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _json(value: Any, limit: int = RECORD_CHARS) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


def event_label(event_type: str) -> str:
    known = EVENT_TYPES.get(event_type)
    if known:
        return known[0]
    return str(event_type or "Event").replace("_", " ").capitalize()


def event_family(event_type: str) -> str:
    known = EVENT_TYPES.get(event_type)
    return known[1] if known else "other"


def _status(payload: Mapping[str, Any], key: str = "status") -> str:
    return str(payload.get(key) or "").strip().lower()


def event_signal(event_type: str, payload: Mapping[str, Any]) -> str:
    """'good', 'warn', 'bad' or '' for one event, from its own payload."""
    if event_type in {"run_completed", "outcome_recorded"}:
        status = _status(payload)
        if status in _GOOD:
            return "good"
        if status in _BAD:
            return "bad"
        return "warn" if status in _DEGRADED else ""
    if event_type == "workflow_recorded":
        outcome = _status(payload, "outcome")
        return "good" if outcome in _GOOD else ("bad" if outcome in _BAD else "")
    if event_type == "tool_executed":
        if not payload.get("success"):
            return "bad"
        outcome = payload.get("outcome")
        status = _status(outcome) if isinstance(outcome, Mapping) else ""
        if status in _BAD:
            return "bad"
        return "warn" if status in _DEGRADED else "good"
    if event_type == "evidence_pack_built":
        used, budget = _number(payload.get("tokens_used")), _number(
            payload.get("token_budget")
        )
        if payload.get("warnings") or (used and budget and used >= 0.9 * budget):
            return "warn"
        return ""
    if event_type == "cache_hit" or event_type == "skill_promoted":
        return "good"
    if event_type in {"skill_demoted", "skill_deprecated"}:
        return "warn"
    if event_type == "forgetting_planned":
        return "warn" if payload.get("candidate_count") else ""
    if event_type == "memory_forgotten":
        return "bad" if payload.get("errors") else ""
    return ""


def summarize_event(
    event_type: str, payload: Mapping[str, Any], workflow_id: Any = None
) -> str:
    """One readable line for the grid."""
    if event_type == "run_started":
        query = _text(payload.get("query"), SUMMARY_CHARS)
        return query or "Run started"
    if event_type == "run_completed":
        parts = [f"Status {payload.get('status') or 'unknown'}"]
        calls = _number(payload.get("tool_call_count"))
        if calls is not None:
            parts.append(_plural(int(calls), "tool call"))
        chars = _number(payload.get("response_chars"))
        if chars is not None:
            parts.append(f"{chars:,.0f} response chars")
        return " · ".join(parts)
    if event_type == "tool_executed":
        name = payload.get("tool_name") or "unknown tool"
        outcome = payload.get("outcome")
        status = _status(outcome) if isinstance(outcome, Mapping) else ""
        if not payload.get("success"):
            verb = "failed"
        elif status == "fallback":
            verb = "completed via fallback"
        elif status == "degraded":
            verb = "completed with degraded capability"
        elif status == "empty":
            verb = "completed with no results"
        else:
            verb = "succeeded"
        took = _number(payload.get("duration_ms"))
        return f"{name} {verb}" + (f" in {_ms(took)}" if took is not None else "")
    if event_type == "evidence_pack_built":
        selected = _number(payload.get("selected_count"))
        if selected is None and isinstance(payload.get("items"), list):
            selected = len(payload["items"])
        parts = [
            f"{_count(selected)} of {_count(payload.get('candidate_count'))} "
            "candidates selected",
            f"{_count(payload.get('tokens_used'))} of "
            f"{_count(payload.get('token_budget'))} tokens",
        ]
        warnings = payload.get("warnings") or []
        if warnings:
            parts.append(_plural(len(warnings), "warning"))
        return " · ".join(parts)
    if event_type in {"cache_hit", "cache_bypassed"}:
        reason = payload.get("reason")
        return event_label(event_type) + (f": {reason}" if reason else "")
    if event_type == "outcome_recorded":
        authority = "Verified" if payload.get("verified") else "Unverified"
        text = (
            f"{authority} {payload.get('status') or 'unknown'} "
            f"from {payload.get('source') or 'unknown source'}"
        )
        score = _number(payload.get("score"))
        return text + (f" · score {score:g}" if score is not None else "")
    if event_type == "workflow_recorded":
        steps = _number(payload.get("step_count"))
        return (
            f"Workflow {payload.get('workflow_id') or workflow_id or 'unknown'}: "
            f"{payload.get('outcome') or 'unknown'}"
            + (f" in {_plural(int(steps), 'step')}" if steps is not None else "")
        )
    if event_family(event_type) == "skill":
        skill = payload.get("skill_id") or payload.get("name") or "unknown"
        reason = payload.get("reason")
        return f"Skill {skill}" + (
            f": {_text(reason, SUMMARY_CHARS)}" if reason else ""
        )
    if event_type == "forgetting_planned":
        candidates = int(_number(payload.get("candidate_count")) or 0)
        retained = int(_number(payload.get("retained")) or 0)
        return f"Dry run: {_plural(candidates, 'candidate')}, {retained} retained"
    if event_type == "memory_forgotten":
        tombstoned = int(_number(payload.get("tombstoned")) or 0)
        errors = payload.get("errors") or []
        return f"Tombstoned {_plural(tombstoned, 'artifact')}" + (
            f" · {_plural(len(errors), 'error')}" if errors else ""
        )
    return event_label(event_type)


def event_metrics(event_type: str, payload: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Up to six (label, value) figures for the detail pane."""
    if event_type == "evidence_pack_built":
        selected = payload.get("selected_count")
        if selected is None and isinstance(payload.get("items"), list):
            selected = len(payload["items"])
        used, candidates = _number(payload.get("tokens_used")), _number(
            payload.get("candidate_tokens")
        )
        saved = (
            max(0.0, candidates - used)
            if used is not None and candidates is not None
            else None
        )
        return [
            (
                "Selected",
                f"{_count(selected)} / {_count(payload.get('candidate_count'))}",
            ),
            ("Tokens used", _count(used)),
            ("Budget", _count(payload.get("token_budget"))),
            ("Tokens saved", _count(saved)),
            ("Rejected", _count(payload.get("rejected_count"))),
            ("Latency", _ms(payload.get("latency_ms"))),
        ]
    if event_type == "tool_executed":
        outcome = payload.get("outcome")
        return [
            ("Result", "Succeeded" if payload.get("success") else "Failed"),
            ("Duration", _ms(payload.get("duration_ms"))),
            (
                "Outcome",
                str(outcome.get("status") or "—")
                if isinstance(outcome, Mapping)
                else "—",
            ),
        ]
    if event_type == "run_completed":
        return [
            ("Status", str(payload.get("status") or "—")),
            ("Tool calls", _count(payload.get("tool_call_count"))),
            ("Response chars", _count(payload.get("response_chars"))),
        ]
    if event_type == "outcome_recorded":
        score = _number(payload.get("score"))
        return [
            ("Status", str(payload.get("status") or "—")),
            ("Verified", "Yes" if payload.get("verified") else "No"),
            ("Score", f"{score:g}" if score is not None else "—"),
        ]
    if event_type == "workflow_recorded":
        return [
            ("Outcome", str(payload.get("outcome") or "—")),
            ("Steps", _count(payload.get("step_count"))),
            ("Skills", _count(len(payload.get("skills_activated") or []))),
        ]
    if event_type == "forgetting_planned":
        return [
            ("Candidates", _count(payload.get("candidate_count"))),
            ("Retained", _count(payload.get("retained"))),
        ]
    if event_type == "memory_forgotten":
        return [
            ("Tombstoned", _count(payload.get("tombstoned"))),
            ("Errors", _count(len(payload.get("errors") or []))),
        ]
    return []


def _processed_hashes(checkpoints: Iterable[Mapping[str, Any]]) -> Dict[str, Set[str]]:
    processed: Dict[str, Set[str]] = {}
    for checkpoint in checkpoints:
        stream = str(checkpoint.get("stream_id") or "")
        if not stream:
            continue
        processed.setdefault(stream, set()).update(
            str(item)
            for item in (checkpoint.get("processed_event_hashes") or [])
            if item
        )
    return processed


def shape_event(
    record: Mapping[str, Any], processed: Mapping[str, Set[str]]
) -> Optional[Dict[str, Any]]:
    """One grid row plus everything the detail pane shows, or None if invalid."""
    event_id = str(record.get("event_id") or "").strip()
    event_type = str(record.get("event_type") or "").strip()
    if not event_id or not event_type:
        return None
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    when = _parse(record.get("timestamp"))
    stream_id = str(record.get("stream_id") or "")
    event_hash = str(record.get("event_hash") or "")
    compiled = bool(event_hash) and event_hash in processed.get(stream_id, set())
    scope = {
        key: str(record.get(key) or "") for key in ("memory_id", "user_id", "thread_id")
    }
    refs = [
        (label, str(record.get(key)))
        for key, label in (
            ("run_id", "Run"),
            ("turn_id", "Turn"),
            ("trace_id", "Trace"),
            ("workflow_id", "Workflow"),
            ("parent_event_id", "Parent event"),
            ("stream_id", "Stream"),
        )
        if record.get(key)
    ]
    family = event_family(event_type)
    signal = event_signal(event_type, payload)
    summary = summarize_event(event_type, payload, record.get("workflow_id"))
    tags = [family, "compiled" if compiled else "pending"]
    if signal == "bad":
        tags.append("failed")
    return {
        "key": event_id,
        "event_id": event_id,
        "event_type": event_type,
        "label": event_label(event_type),
        "family": family,
        "signal": signal,
        "summary": summary,
        "timestamp": str(record.get("timestamp") or ""),
        "recorded": _shown(record.get("timestamp")),
        "recorded_at": when,
        "sort": when.timestamp() if when else 0.0,
        "scope": scope,
        "scope_text": " · ".join(value for value in scope.values() if value),
        "run_ref": str(record.get("run_id") or record.get("workflow_id") or ""),
        "trace_id": str(record.get("trace_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "refs": refs,
        "stream_id": stream_id,
        "event_hash": event_hash,
        "compiled": compiled,
        "metrics": event_metrics(event_type, payload),
        "tags": " ".join(tags),
        "search": " ".join(
            [
                event_type,
                event_label(event_type),
                summary,
                event_id,
                *scope.values(),
                str(record.get("run_id") or ""),
                str(record.get("workflow_id") or ""),
                str(payload.get("tool_name") or ""),
            ]
        ).lower(),
        "record": _json(dict(record)),
    }


def shape_artifact(record: Mapping[str, Any], tombstoned: Set[str]) -> Dict[str, Any]:
    artifact_id = str(record.get("artifact_id") or record.get("record_id") or "")
    kind = str(record.get("artifact_kind") or "")
    updated = record.get("updated_at") or record.get("timestamp")
    utility = _number(record.get("utility"))
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "kind_label": ARTIFACT_KINDS.get(
            kind, kind.replace("_", " ").capitalize() or "—"
        ),
        "verified": bool(record.get("verified")),
        "utility": utility,
        "updated_at": updated,
        "updated": _shown(updated),
        "sort": (_parse(updated) or datetime.min.replace(tzinfo=timezone.utc)),
        "content": _text(record.get("content"), 400),
        "tools": [str(item) for item in (record.get("tools") or []) if item],
        "sources": len(record.get("source_event_ids") or []),
        "run_ref": str(record.get("run_id") or record.get("workflow_id") or ""),
        "tombstoned": artifact_id in tombstoned,
    }


def _plan_candidates(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for item in payload.get("candidates") or []:
        if not isinstance(item, Mapping):
            continue
        target_type = str(item.get("target_type") or "")
        rows.append(
            {
                "target_id": str(item.get("target_id") or ""),
                "kind_label": ARTIFACT_KINDS.get(
                    target_type, target_type.replace("_", " ").capitalize() or "—"
                ),
                "reason": str(item.get("reason") or ""),
                "utility": _number(item.get("utility")),
                "age_days": _number(item.get("age_days")),
                "superseded_by": str(item.get("superseded_by") or ""),
                "action": str(item.get("action") or "tombstone"),
            }
        )
    return rows


def latest_forgetting_plan(
    events: Iterable[Mapping[str, Any]],
    tombstones: Iterable[Mapping[str, Any]] = (),
) -> Optional[Dict[str, Any]]:
    """The newest dry-run plan in scope and whether it was applied."""
    planned: Optional[Mapping[str, Any]] = None
    applied: Dict[str, Mapping[str, Any]] = {}
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or not payload.get("plan_id"):
            continue
        if event.get("event_type") == "forgetting_planned":
            if planned is None or str(event.get("timestamp") or "") >= str(
                planned.get("timestamp") or ""
            ):
                planned = event
        elif event.get("event_type") == "memory_forgotten":
            applied[str(payload["plan_id"])] = event
    if planned is None:
        return None
    payload = planned["payload"]
    plan_id = str(payload["plan_id"])
    application = applied.get(plan_id)
    approvers = sorted(
        {
            str(item.get("approved_by"))
            for item in tombstones
            if item.get("plan_id") == plan_id and item.get("approved_by")
        }
    )
    candidates = _plan_candidates(payload)
    if application is not None:
        state, label, signal = "applied", "Applied", "good"
    elif candidates:
        state, label, signal = "awaiting", "Awaiting approval", "warn"
    else:
        state, label, signal = "clear", "Nothing to forget", "good"
    applied_payload = application.get("payload") if application else {}
    return {
        "plan_id": plan_id,
        "planned": _shown(planned.get("timestamp")),
        "planned_at": _parse(planned.get("timestamp")),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "retained": int(_number(payload.get("retained")) or 0),
        "state": state,
        "state_label": label,
        "signal": signal,
        "applied": _shown(application.get("timestamp")) if application else "",
        "tombstoned": int(_number((applied_payload or {}).get("tombstoned")) or 0),
        "approved_by": ", ".join(approvers),
    }


def _series(events: List[Dict[str, Any]], days: int = 14) -> List[Dict[str, int]]:
    dated = [row for row in events if row["recorded_at"]]
    if not dated:
        return []
    newest = max(row["recorded_at"] for row in dated)
    start = (newest - timedelta(days=days - 1)).date()
    counts: Counter = Counter()
    failed: Counter = Counter()
    for row in dated:
        day = row["recorded_at"].date()
        if day >= start:
            counts[day] += 1
            failed[day] += row["signal"] == "bad"
    return [
        {
            "day": (start + timedelta(days=i)).isoformat(),
            "count": counts.get(start + timedelta(days=i), 0),
            "failed": failed.get(start + timedelta(days=i), 0),
        }
        for i in range(days)
    ]


def _streams(
    events: List[Dict[str, Any]], checkpoints: List[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    by_checkpoint = {
        str(item.get("stream_id")): item
        for item in checkpoints
        if item.get("stream_id")
    }
    streams: Dict[str, Dict[str, Any]] = {}
    for row in events:
        stream = streams.setdefault(
            row["stream_id"],
            {
                "stream_id": row["stream_id"],
                "scope": row["scope"],
                "events": 0,
                "pending": 0,
                "latest": "",
                "latest_sort": 0.0,
            },
        )
        stream["events"] += 1
        stream["pending"] += not row["compiled"]
        if row["sort"] >= stream["latest_sort"]:
            stream["latest_sort"], stream["latest"] = row["sort"], row["recorded"]
    for stream_id, stream in streams.items():
        checkpoint = by_checkpoint.get(stream_id) or {}
        stream["last_compile"] = _shown(checkpoint.get("timestamp"))
        stream["compiled_total"] = int(
            _number(checkpoint.get("compiled_event_count")) or 0
        )
    return sorted(
        streams.values(), key=lambda item: (-item["pending"], -item["latest_sort"])
    )


def build_control_plane_view(
    records: Iterable[Mapping[str, Any]],
    *,
    scope_stream_id: Optional[str] = None,
    evidence_budget: Optional[int] = None,
    event_limit: int = EVENT_LIMIT,
    artifact_limit: int = ARTIFACT_LIMIT,
) -> Dict[str, Any]:
    """Everything the control-plane monitor shows for one agent scope.

    ``records`` are learning-record payloads (events, artifacts, checkpoints,
    tombstones) already filtered to the scope. ``scope_stream_id`` is the
    stream the page's compile action works on, so the tape can say how many
    uncompiled events that action would pick up.
    """
    by_type: Dict[str, List[Mapping[str, Any]]] = {}
    total = 0
    for record in records:
        if not isinstance(record, Mapping):
            continue
        total += 1
        by_type.setdefault(str(record.get("record_type") or "unknown"), []).append(
            record
        )
    raw_events = by_type.get(EVENT_RECORD, [])
    checkpoints = by_type.get(CHECKPOINT_RECORD, [])
    tombstones = by_type.get(TOMBSTONE_RECORD, [])
    processed = _processed_hashes(checkpoints)
    tombstoned = {
        str(item.get("target_id")) for item in tombstones if item.get("target_id")
    }

    events = [
        row for row in (shape_event(item, processed) for item in raw_events) if row
    ]
    events.sort(key=lambda row: row["sort"], reverse=True)
    artifacts = [
        shape_artifact(item, tombstoned) for item in by_type.get(ARTIFACT_RECORD, [])
    ]
    artifacts.sort(key=lambda row: row["sort"], reverse=True)

    pending = [row for row in events if not row["compiled"]]
    families = Counter(row["family"] for row in events)
    runs = [row for row in events if row["event_type"] == "run_completed"]
    tools = [row for row in events if row["event_type"] == "tool_executed"]
    packs = [
        item.get("payload") or {}
        for item in raw_events
        if item.get("event_type") == "evidence_pack_built"
    ]
    used = [
        value
        for value in (_number(pack.get("tokens_used")) for pack in packs)
        if value is not None
    ]
    budget = evidence_budget or next(
        (
            int(_number(pack.get("token_budget")) or 0)
            for pack in packs
            if pack.get("token_budget")
        ),
        None,
    )
    mean_used = round(sum(used) / len(used)) if used else None
    last_checkpoint = max(
        checkpoints,
        key=lambda item: str(item.get("timestamp") or item.get("updated_at") or ""),
        default=None,
    )
    last_compile_raw = (
        (last_checkpoint.get("timestamp") or last_checkpoint.get("updated_at"))
        if last_checkpoint
        else None
    )
    plan = latest_forgetting_plan(raw_events, tombstones)
    run_failed = sum(row["signal"] == "bad" for row in runs)
    tool_failed = sum(row["signal"] == "bad" for row in tools)
    return {
        "record_count": total,
        "event_count": len(raw_events),
        "artifact_count": len(by_type.get(ARTIFACT_RECORD, [])),
        "checkpoint_count": len(checkpoints),
        "tombstone_count": len(tombstones),
        "events": events[:event_limit],
        "has_more_events": len(events) > event_limit,
        "artifacts": artifacts[:artifact_limit],
        "has_more_artifacts": len(artifacts) > artifact_limit,
        "pending_count": len(pending),
        "pending_in_scope": sum(
            1
            for row in pending
            if scope_stream_id and row["stream_id"] == scope_stream_id
        ),
        "compiled_count": len(events) - len(pending),
        "families": [
            (key, label, families[key]) for key, label in FAMILIES if families.get(key)
        ],
        "failed_count": sum(row["signal"] == "bad" for row in events),
        "runs": len(runs),
        "run_failed": run_failed,
        "run_success_rate": (
            round(100.0 * (len(runs) - run_failed) / len(runs), 1) if runs else None
        ),
        "tools": len(tools),
        "tool_failed": tool_failed,
        "evidence_packs": len(packs),
        "evidence_mean_tokens": mean_used,
        "evidence_budget": budget,
        "evidence_use_percent": (
            round(100.0 * mean_used / budget)
            if mean_used is not None and budget
            else None
        ),
        "last_compile": _shown(last_compile_raw),
        "last_compile_at": _parse(last_compile_raw),
        "latest_event": events[0]["recorded"] if events else "",
        "latest_event_at": events[0]["recorded_at"] if events else None,
        "memory_ids": sorted(
            {row["scope"]["memory_id"] for row in events if row["scope"]["memory_id"]}
        ),
        "streams": _streams(events, checkpoints),
        "series": _series(events),
        "plan": plan,
    }


__all__ = [
    "build_control_plane_view",
    "event_label",
    "event_metrics",
    "event_signal",
    "latest_forgetting_plan",
    "shape_artifact",
    "shape_event",
    "summarize_event",
]
