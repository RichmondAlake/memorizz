"""Operational overview of recent agent runs from normalized trace events.

Groups events into runs (one agent turn each) and summarizes what an operator
checks first: volume, failures, run time, tokens, spend, prompt-cache reuse and
failing tools. Pure and content-free: it reads identifiers, statuses, counters
and timestamps only. Scope events to the authorized tenant *before* calling.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Optional

from .normalization import timestamp

# Window key -> (span, bucket size) for the activity series.
WINDOWS = {
    "24h": (timedelta(hours=24), timedelta(hours=1)),
    "7d": (timedelta(days=7), timedelta(hours=6)),
    "30d": (timedelta(days=30), timedelta(days=1)),
    "90d": (timedelta(days=90), timedelta(days=1)),
}
DEFAULT_WINDOW = "7d"
# A custom range takes the smallest of these buckets that keeps its chart at
# 60 bars or fewer, and covers a year at most.
CUSTOM_BUCKETS = tuple(
    timedelta(minutes=minutes)
    for minutes in (5, 15, 60, 180, 360, 720, 1440, 2880, 10080)
)
MAX_CUSTOM_SPAN = timedelta(days=366)
# A run without a result this long after its last event is reported incomplete.
RUNNING_GRACE = timedelta(minutes=10)
_OK_STATUSES = {"success", "completed", "ok"}

# Plain-language causes for safe, content-free codes, so the overview explains
# failures even when trace content is hidden (metadata-only mode).
TOOL_REASONS = {
    "tool_not_disclosed": "called before discover_tools disclosed it this turn",
    "unknown_tool": "no tool has that name",
    "invalid_arguments": "arguments did not match the tool's signature",
    "tool_not_callable": "no callable is registered for it in this process",
    "invalid_tool_invocation": "rejected before running (not disclosed, bad "
    "arguments or no callable; newer versions report which)",
    "tool_log_not_found": "the referenced tool log does not exist",
    "approval_required": "waiting for approval",
    "duplicate_tool_call": "the model repeated a call that had already "
    "succeeded this turn, so the repeat was skipped",
    "tool_retry_limit": "the same call kept failing until its retry limit",
    "tool_budget_exhausted": "the turn had used all of its tool calls "
    "(ContextPolicy.max_tool_invocations_per_turn), so the rest were not run",
    "unreported": "the tool failed without a reason code",
}
PROVIDER_ERRORS = {
    "RemoteProtocolError": "the provider closed the connection mid-response",
    "StreamClosed": "the stream closed before any output",
    "ProviderStreamError": "the provider sent a stream error event",
    "ReadTimeout": "the provider stopped responding",
    "APIConnectionError": "the provider could not be reached",
    "APITimeoutError": "the provider request timed out",
}


def explain_tool_reason(code: str) -> str:
    text = TOOL_REASONS.get(code)
    return f"{text} ({code})" if text else f"reason code {code}"


def explain_provider_error(code: str) -> str:
    text = PROVIDER_ERRORS.get(code)
    return f"{code}: {text}" if text else code


def _when(value: Any) -> Optional[datetime]:
    text = timestamp(value)
    return datetime.fromisoformat(text) if text else None


def resolve_window(
    window: str,
    *,
    start: Any = None,
    end: Any = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """The span a window covers: a key of ``WINDOWS``, or ``custom`` from
    ``start`` to ``end`` (now when empty; naive times are UTC). An unusable
    custom range falls back to the default window and says why in ``error``."""
    now = now or datetime.now(timezone.utc)
    error = ""
    if window == "custom":
        begin, finish = _when(start), (_when(end) if end else now)
        if begin is None or finish is None:
            error = "Choose a start and an end for a custom range."
        elif begin >= min(finish, now):
            error = "A custom range has to start before it ends, and before now."
        elif min(finish, now) - begin > MAX_CUSTOM_SPAN:
            error = "A custom range can cover a year at most."
        else:
            finish = min(finish, now)
            bucket = next(
                (size for size in CUSTOM_BUCKETS if (finish - begin) / size <= 60),
                CUSTOM_BUCKETS[-1],
            )
            return {
                "key": "custom",
                "start": begin,
                "end": finish,
                "bucket": bucket,
                "error": "",
            }
        window = DEFAULT_WINDOW
    if window not in WINDOWS:
        window = DEFAULT_WINDOW
    span, bucket = WINDOWS[window]
    return {
        "key": window,
        "start": now - span,
        "end": now,
        "bucket": bucket,
        "error": error,
    }


def _decimal(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _run_key(event: Dict[str, Any]) -> str:
    return str(event.get("turn_id") or event.get("root_trace_id") or "")


def _recorded_cost(event: Dict[str, Any]) -> Optional[Decimal]:
    return _decimal(event.get("cost_usd"))


def _summarize_run(
    key: str,
    events: List[Dict[str, Any]],
    now: datetime,
    price: Callable[[Dict[str, Any]], Optional[Decimal]],
) -> Dict:
    times = [t for t in (_when(e.get("timestamp")) for e in events) if t]
    first = events[0]
    run: Dict[str, Any] = {
        "key": key,
        "agent_id": next((e["agent_id"] for e in events if e.get("agent_id")), ""),
        "thread_id": next((e["thread_id"] for e in events if e.get("thread_id")), ""),
        "root_trace_id": first.get("root_trace_id") or "",
        "turn_id": first.get("turn_id") or "",
        "started_at": min(times) if times else None,
        "ended_at": None,
        "duration_ms": None,
        "status": "unknown",
        "error_code": "",
        "model_calls": 0,
        "model_errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "cost_usd": Decimal(0),
        "unpriced_calls": 0,
        "unpriced": [],
        "provider_errors": [],
        "tool_calls": 0,
        "tool_failures": 0,
        "tools": [],
        "models": [],
    }
    result = None
    for event in events:
        kind = event.get("trace_kind") or event.get("kind")
        if kind == "turn_result":
            result = event
        elif kind == "model_result":
            run["model_calls"] += 1
            label = " / ".join(
                str(part)
                for part in (event.get("provider"), event.get("model"))
                if part
            )
            if str(event.get("status") or "").lower() not in _OK_STATUSES:
                run["model_errors"] += 1
                run["provider_errors"].append(
                    {
                        "code": str(event.get("error_code") or "unreported")[:80],
                        "model": label,
                        # Whether any text had streamed before the failure.
                        "mid_response": _int(event.get("response_chars")) > 0,
                    }
                )
                continue
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_write_tokens",
            ):
                run[field] += _int(event.get(field))
            priced = price(event)
            cost, reason = priced if isinstance(priced, tuple) else (priced, "")
            if cost is None:
                run["unpriced_calls"] += 1
                # The provider, model and tier a rate card would be keyed by.
                run["unpriced"].append(
                    (
                        label,
                        reason or "no price recorded",
                        str(event.get("provider") or ""),
                        str(event.get("model") or ""),
                        str(event.get("service_tier") or "default"),
                    )
                )
            else:
                run["cost_usd"] += cost
            if label and label not in run["models"]:
                run["models"].append(label)
        elif kind == "tool_result":
            name = str(
                event.get("logical_tool_name") or event.get("tool_name") or "tool"
            )
            failed = str(event.get("status") or "").lower() not in _OK_STATUSES
            run["tool_calls"] += 1
            run["tool_failures"] += int(failed)
            run["tools"].append(
                {
                    "name": name,
                    "failed": failed,
                    "reason_code": str(
                        event.get("outcome_reason_code")
                        or event.get("error_code")
                        or ""
                    )[:80]
                    if failed
                    else "",
                }
            )
    last = max(times) if times else None
    if result is not None:
        status = str(result.get("status") or "").lower()
        run["status"] = "success" if status in _OK_STATUSES else "failed"
        run["error_code"] = str(result.get("error_code") or "")[:80]
        run["ended_at"] = _when(result.get("timestamp")) or last
    elif last is not None:
        run["status"] = "running" if now - last < RUNNING_GRACE else "incomplete"
    if run["started_at"] and run["ended_at"]:
        run["duration_ms"] = max(
            0.0, (run["ended_at"] - run["started_at"]).total_seconds() * 1000
        )
    return run


def _bucket_series(runs, start: datetime, end: datetime, size: timedelta) -> List:
    # Align to whole buckets (hours, quarter days, UTC days) so bars read cleanly.
    seconds = size.total_seconds()
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    start = epoch + timedelta(
        seconds=math.floor((start - epoch).total_seconds() / seconds) * seconds
    )
    count = max(1, math.ceil((end - start) / size))
    buckets = [
        {
            "start": start + size * index,
            "end": start + size * (index + 1),
            "runs": 0,
            "succeeded": 0,
            "failed": 0,
            "tokens": 0,
            "cost_usd": Decimal(0),
            # A bucket holding one run links straight to it.
            "run": None,
        }
        for index in range(count)
    ]
    for run in runs:
        index = int((run["started_at"] - start) / size)
        if 0 <= index < count:
            bucket = buckets[index]
            bucket["runs"] += 1
            bucket["succeeded"] += int(run["status"] == "success")
            bucket["failed"] += int(run["status"] == "failed")
            bucket["tokens"] += run["input_tokens"] + run["output_tokens"]
            bucket["cost_usd"] += run["cost_usd"]
            bucket["run"] = run if bucket["runs"] == 1 else None
    return buckets


def agent_health(row: Optional[Dict[str, Any]]) -> str:
    """idle, failing, degraded or healthy, from an agent's runs in the window."""
    if not row or not row.get("runs"):
        return "idle"
    rate = row.get("success_rate")
    if row.get("last_status") == "failed" or (rate is not None and rate < 90):
        return "failing"
    if row.get("failed") or row.get("tool_failures"):
        return "degraded"
    return "healthy"


def _group(runs, key) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return groups


_VERDICT_WORDS = {
    "success": "pass",
    "failure": "fail",
    "partial": "partial",
    "unknown": "unknown",
}


def _verdict_time(row: Dict[str, Any]) -> Optional[datetime]:
    return _when(row.get("updated_at") or row.get("created_at") or row.get("timestamp"))


def _grade_runs(
    runs: List[Dict[str, Any]],
    verdicts: Iterable[Dict[str, Any]],
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    """Join recorded outcomes and grader verdicts to the window's runs.

    A turn's grade is the newest verdict of each grader (or source): it fails
    when any of them is a failure. A verdict whose run is not in the loaded
    events still counts when it was recorded inside the window; its record
    names the turn to link. Returns one entry per graded turn, each with its
    run (``verdicts`` and ``grade`` set on it), the grade and when it ran.
    """
    by_turn: Dict[str, List[Dict[str, Any]]] = {}
    for row in verdicts or ():
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "").lower() not in _VERDICT_WORDS:
            continue
        key = str(row.get("turn_id") or row.get("root_trace_id") or "")
        if key:
            by_turn.setdefault(key, []).append(row)
    runs_by_key = {run["key"]: run for run in runs}
    runs_by_root = {run["root_trace_id"]: run for run in runs if run["root_trace_id"]}
    graded = []
    for key, rows in by_turn.items():
        rows.sort(
            key=lambda row: _verdict_time(row)
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        latest: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            latest[str(row.get("grader") or row.get("source") or "")] = row
        final = list(latest.values())
        recorded = _verdict_time(final[-1])
        run = runs_by_key.get(key) or runs_by_root.get(
            str(final[-1].get("root_trace_id") or "")
        )
        if run is None:
            if recorded is None or not start <= recorded <= end:
                continue
            run = {
                field: str(final[-1].get(field) or "")
                for field in ("agent_id", "thread_id", "root_trace_id", "turn_id")
            }
            run["started_at"] = None
        statuses = {str(row.get("status") or "").lower() for row in final}
        grade = next(
            (s for s in ("failure", "partial", "success") if s in statuses),
            "unknown",
        )
        run["grade"] = _VERDICT_WORDS[grade]
        run["verdicts"] = [
            {
                "verdict": _VERDICT_WORDS[str(row.get("status")).lower()],
                "grader": str(row.get("grader") or row.get("source") or ""),
                "reason_code": str(row.get("reason_code") or ""),
                "score": row.get("score"),
                "verified": row.get("verified") is True,
            }
            for row in final
        ]
        graded.append(
            {
                "run": run,
                "grade": grade,
                "at": run.get("started_at") or recorded,
                "failing": [
                    row for row in final if str(row.get("status")).lower() == "failure"
                ],
            }
        )
    return graded


def build_operations_overview(
    events: Iterable[Dict[str, Any]],
    *,
    window: str = DEFAULT_WINDOW,
    now: Optional[datetime] = None,
    recent_limit: int = 12,
    price: Optional[Callable[[Dict[str, Any]], Optional[Decimal]]] = None,
    span: Optional[Dict[str, Any]] = None,
    verdicts: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Summarize runs that started inside ``window`` (a key of ``WINDOWS``),
    or inside ``span`` from ``resolve_window`` when given.

    ``price`` returns a model result's cost or None when it cannot be priced;
    by default only a recorded ``cost_usd`` counts. ``verdicts`` are recorded
    task outcomes (``record_task_outcome``); turns that failed grading become
    a ``graded_failures`` attention item linked to the newest one.
    """
    price = price or _recorded_cost
    now = now or datetime.now(timezone.utc)
    span = span or resolve_window(window, now=now)
    window, start, end = span["key"], span["start"], span["end"]
    bucket_size = span["bucket"]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for event in events:
        if isinstance(event, dict) and _run_key(event):
            grouped.setdefault(_run_key(event), []).append(event)
    runs = [
        run
        for run in (
            _summarize_run(key, rows, now, price) for key, rows in grouped.items()
        )
        if run["started_at"] and start <= run["started_at"] <= end
    ]
    runs.sort(key=lambda run: run["started_at"], reverse=True)

    finished = [run for run in runs if run["status"] in {"success", "failed"}]
    failed = [run for run in runs if run["status"] == "failed"]
    graded = _grade_runs(runs, verdicts or (), start, end)
    graded_failures = [item for item in graded if item["grade"] == "failure"]
    durations = [
        run["duration_ms"] for run in finished if run["duration_ms"] is not None
    ]
    input_tokens = sum(run["input_tokens"] for run in runs)
    cached_tokens = sum(run["cached_tokens"] for run in runs)
    cost = sum((run["cost_usd"] for run in runs), Decimal(0))
    totals = {
        "runs": len(runs),
        "succeeded": len(finished) - len(failed),
        "failed": len(failed),
        "in_progress": sum(run["status"] == "running" for run in runs),
        "incomplete": sum(run["status"] in {"incomplete", "unknown"} for run in runs),
        "success_rate": (
            100.0 * (len(finished) - len(failed)) / len(finished) if finished else None
        ),
        "p50_ms": _percentile(durations, 0.5),
        "p95_ms": _percentile(durations, 0.95),
        "model_calls": sum(run["model_calls"] for run in runs),
        "model_errors": sum(run["model_errors"] for run in runs),
        "input_tokens": input_tokens,
        "output_tokens": sum(run["output_tokens"] for run in runs),
        "cached_tokens": cached_tokens,
        "cache_read_percent": (
            100.0 * cached_tokens / input_tokens if input_tokens else None
        ),
        "cost_usd": cost,
        "cost_per_run": cost / len(runs) if runs else None,
        "unpriced_calls": sum(run["unpriced_calls"] for run in runs),
        "tool_calls": sum(run["tool_calls"] for run in runs),
        "tool_failures": sum(run["tool_failures"] for run in runs),
        "runs_with_tool_failures": sum(bool(run["tool_failures"]) for run in runs),
        "graded_runs": len(graded),
        "graded_failures": len(graded_failures),
    }

    agents = []
    for agent_id, rows in _group(runs, lambda run: run["agent_id"]).items():
        spans = [r["duration_ms"] for r in rows if r["duration_ms"] is not None]
        done = [r for r in rows if r["status"] in {"success", "failed"}]
        failed_runs = sum(r["status"] == "failed" for r in rows)
        agent_input = sum(r["input_tokens"] for r in rows)
        models: List[str] = []
        for run in rows:
            models.extend(label for label in run["models"] if label not in models)
        row = {
            "agent_id": agent_id,
            "runs": len(rows),
            "failed": failed_runs,
            "tool_failures": sum(r["tool_failures"] for r in rows),
            "success_rate": (
                100.0 * (len(done) - failed_runs) / len(done) if done else None
            ),
            "p95_ms": _percentile(spans, 0.95),
            "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
            "cache_read_percent": (
                100.0 * sum(r["cached_tokens"] for r in rows) / agent_input
                if agent_input
                else None
            ),
            "cost_usd": sum((r["cost_usd"] for r in rows), Decimal(0)),
            "last_run_at": rows[0]["started_at"],
            "last_status": rows[0]["status"],
            "models": models,
            "series": [
                {"runs": b["runs"], "failed": b["failed"]}
                for b in _bucket_series(rows, start, end, bucket_size)
            ],
            "recent": rows[:5],
        }
        row["health"] = agent_health(row)
        agents.append(row)
    agents.sort(key=lambda row: (-row["runs"], row["agent_id"]))

    tool_rows: Dict[str, Dict[str, Any]] = {}
    # Per failing tool, its newest failed run and the agents whose calls failed.
    tool_failures: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        for call in run["tools"]:
            row = tool_rows.setdefault(
                call["name"],
                {"name": call["name"], "calls": 0, "failures": 0, "reasons": Counter()},
            )
            row["calls"] += 1
            if call["failed"]:
                row["failures"] += 1
                row["reasons"][call["reason_code"] or "unreported"] += 1
                seen = tool_failures.setdefault(
                    call["name"], {"run": run, "agents": Counter()}
                )
                seen["agents"][run["agent_id"]] += 1
    tools = []
    for row in tool_rows.values():
        reasons = row.pop("reasons")
        row["failure_rate"] = 100.0 * row["failures"] / row["calls"]
        row["top_reason"] = reasons.most_common(1)[0][0] if reasons else ""
        tools.append(row)
    tools.sort(key=lambda row: (-row["failures"], -row["calls"], row["name"]))

    attention = []
    provider_errors = [error for run in runs for error in run["provider_errors"]]
    if failed:
        codes = Counter(run["error_code"] or "unreported" for run in failed)
        # Failed runs per cause: a run that retried logs the same error twice.
        causes = Counter(
            code
            for run in failed
            for code in {error["code"] for error in run["provider_errors"]}
        )
        detail = f"Run error code: {codes.most_common(1)[0][0]}"
        if causes:
            cause, count = causes.most_common(1)[0]
            detail = (
                f"Most common cause: {explain_provider_error(cause)} "
                f"(in {count} of {len(failed)} failed run"
                f"{'s' if len(failed) != 1 else ''}). {detail}"
            )
        # Each item names what a fix starts from: the newest run it happened
        # in, the agent, or the model a rate card would price.
        attention.append(
            {
                "severity": "error",
                "kind": "failed_runs",
                "title": f"{len(failed)} failed run{'s' if len(failed) != 1 else ''}",
                "detail": detail,
                "run": failed[0],
                "count": len(failed),
            }
        )
    if graded_failures:
        newest = max(
            graded_failures,
            key=lambda item: item["at"] or datetime.min.replace(tzinfo=timezone.utc),
        )
        count = len(graded_failures)
        reasons = Counter(
            str(row.get("reason_code") or "unreported")
            for item in graded_failures
            for row in item["failing"]
        )
        graders = sorted(
            {
                str(row.get("grader") or row.get("source") or "unnamed grader")
                for item in graded_failures
                for row in item["failing"]
            }
        )
        reason, times = reasons.most_common(1)[0]
        attention.append(
            {
                "severity": "error",
                "kind": "graded_failures",
                "title": f"{count} turn{'s' if count != 1 else ''} failed grading",
                "detail": f"Most common reason: {reason} ({times} of "
                f"{sum(reasons.values())} failing verdicts). Graded by "
                f"{', '.join(graders)}.",
                "run": newest["run"],
                "count": count,
                "agent_id": newest["run"].get("agent_id") or "",
            }
        )
    for row in tools:
        if not row["failures"]:
            continue
        seen = tool_failures[row["name"]]
        attention.append(
            {
                "severity": "warning",
                "kind": "tool",
                "title": f"{row['name']} failed {row['failures']} of {row['calls']} calls",
                "detail": "Most often " + explain_tool_reason(row["top_reason"]),
                "tool": row["name"],
                "reason": row["top_reason"],
                "run": seen["run"],
                "agent_id": seen["agents"].most_common(1)[0][0],
            }
        )
    if provider_errors:
        codes = Counter(error["code"] for error in provider_errors)
        models = Counter(error["model"] or "unknown model" for error in provider_errors)
        mid = sum(error["mid_response"] for error in provider_errors)
        newest = next(run for run in runs if run["provider_errors"])
        attention.append(
            {
                "severity": "error",
                "kind": "provider_errors",
                "title": f"{len(provider_errors)} provider call error"
                + ("s" if len(provider_errors) != 1 else ""),
                "detail": ", ".join(f"{code} ×{n}" for code, n in codes.most_common(3))
                + f"; {len(provider_errors) - mid} before any output, {mid} mid-response;"
                + f" mostly {models.most_common(1)[0][0]}",
                "run": newest,
                "agent_id": newest["agent_id"],
            }
        )
    unpriced = Counter(item for run in runs for item in run["unpriced"])
    for (label, reason, provider, model, tier), count in unpriced.most_common(2):
        attention.append(
            {
                "severity": "info",
                "kind": "unpriced",
                "title": f"{count} {label or 'model'} call{'s' if count != 1 else ''}"
                " without a price",
                "detail": f"{reason}. Spend excludes these calls.",
                "provider": provider,
                "model": model,
                "service_tier": tier,
            }
        )

    return {
        "window": window,
        "window_start": start,
        "window_end": end,
        "bucket_seconds": int(bucket_size.total_seconds()),
        "totals": totals,
        "series": _bucket_series(runs, start, end, bucket_size),
        "recent": runs[: max(0, recent_limit)],
        "agents": agents,
        "tools": tools,
        "attention": attention,
    }
