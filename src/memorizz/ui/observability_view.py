"""Monitor summaries for the Observability pages (traces, health, compare, usage).

Pure data shaping, like ``memory_view``: routes pass the rows they already
loaded, and templates format the numbers with the ``_fmt.html`` macros.

A tape item is a dict with ``label``, ``value``, ``kind`` (``count``, ``usd``,
``percent``, ``ms`` or ``text``), ``tone`` (``""``, ``is-good``, ``is-warn`` or
``is-bad``), an optional ``title`` (exact value or context shown on hover) and
``meta`` (a secondary, smaller cell at the end of the strip).
"""

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional

GOOD, WARN, BAD = "is-good", "is-warn", "is-bad"


def item(
    label: str,
    value: Any,
    kind: str = "count",
    tone: str = "",
    title: str = "",
    meta: bool = False,
) -> Dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "kind": kind,
        "tone": tone,
        "title": title,
        "meta": meta,
    }


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _int(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def format_age(seconds: float) -> str:
    """Seconds elapsed as ``3h ago``, ``just now``, or ``in 2d`` when negative."""
    future = seconds < 0
    seconds = abs(seconds)
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            amount = f"{int(seconds // size)}{unit}"
            return f"in {amount}" if future else f"{amount} ago"
    return "soon" if future else "just now"


def relative_age(timestamp: Any, now: float) -> str:
    """Seconds-since-epoch to a short age such as ``3h ago``; ``—`` when unknown."""
    stamp = _number(timestamp)
    if not stamp or stamp <= 0:
        return "—"
    return format_age(now - stamp)


def iso_timestamp(value: Any) -> Optional[float]:
    """Parse an ISO-8601 string (``Z`` or offset) to epoch seconds."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def exact_usd(value: Any) -> str:
    """The unrounded recorded charge, for hover titles (``$1.10741417``)."""
    if value is None:
        return ""
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return ""
    if not amount.is_finite():
        return ""
    return f"${amount.normalize():f}" if amount else "$0"


# ---------- Traces: agent and thread navigator ----------


def navigator_rows(
    agent_rows: Iterable[Mapping[str, Any]], *, now: float
) -> List[Dict]:
    """Add ages, thread counts, filter text and chip tags to navigator rows."""
    rows = []
    for source in agent_rows:
        threads = []
        for thread in source.get("thread_rows") or []:
            threads.append(
                {
                    **thread,
                    "age": relative_age(thread.get("last_ts"), now),
                }
            )
        events = _int(source.get("event_count"))
        tags = ["active" if events else "idle"]
        if threads:
            tags.append("threads")
        coverage = str(source.get("coverage") or "unknown")
        if coverage != "complete":
            tags.append("partial")
        search = " ".join(
            [
                str(source.get("name") or ""),
                str(source.get("agent_id") or ""),
                str(source.get("mode") or ""),
                *(str(thread.get("thread_id") or "") for thread in threads),
                *(str(thread.get("memory_id") or "") for thread in threads),
            ]
        ).lower()
        rows.append(
            {
                **source,
                "thread_rows": threads,
                "thread_count": len(threads),
                "age": relative_age(source.get("last_ts"), now),
                "coverage": coverage,
                "coverage_tone": GOOD if coverage == "complete" else WARN,
                "tags": tags,
                "search": search,
            }
        )
    return rows


def navigator_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {"all": 0, "active": 0, "idle": 0, "threads": 0, "partial": 0}
    for row in rows:
        counts["all"] += 1
        for tag in row.get("tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def navigator_tape(
    rows: List[Mapping[str, Any]],
    *,
    now: float,
    read_only: bool,
    content_mode: str,
    query_metadata: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Summary strip for the loaded trace window across all listed agents."""
    query_metadata = query_metadata or {}
    last_ts = max((_number(row.get("last_ts")) or 0 for row in rows), default=0)
    coverages = {str(row.get("coverage") or "unknown") for row in rows}
    if not rows:
        coverage, coverage_tone = "—", ""
    elif coverages == {"complete"}:
        coverage, coverage_tone = "Complete", GOOD
    elif "partial" in coverages or "complete" in coverages:
        coverage, coverage_tone = "Partial", WARN
    else:
        coverage, coverage_tone = "Unknown", WARN
    native = bool(query_metadata.get("provider_native"))
    truncated = bool(query_metadata.get("truncated"))
    errors = query_metadata.get("query_errors") or []
    query = "Native" if native else "Fallback"
    if truncated:
        query += " · truncated"
    query_title = (
        "Provider-native index query" if native else "Compatibility fallback scan"
    )
    if errors:
        query_title += f"; {len(errors)} store queries failed"
    elif truncated:
        query_title += "; window truncated"
    last_title = (
        datetime.fromtimestamp(last_ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        if last_ts
        else ""
    )
    return [
        item("Agents", len(rows)),
        item(
            "With events",
            sum(1 for row in rows if _int(row.get("event_count"))),
        ),
        item("Threads", sum(_int(row.get("thread_count")) for row in rows)),
        item("Trace events", sum(_int(row.get("event_count")) for row in rows)),
        item("Bundles", sum(_int(row.get("bundle_count")) for row in rows)),
        item("Last activity", relative_age(last_ts, now), "text", title=last_title),
        item("Coverage", coverage, "text", coverage_tone),
        item(
            "Query",
            query,
            "text",
            BAD if errors else (WARN if truncated else ""),
            title=query_title,
        ),
        item(
            f"Content mode {content_mode or 'full'}",
            "Read-only provider" if read_only else "Writable provider",
            "text",
            meta=True,
        ),
    ]


# ---------- Traces: timeline ----------

# Filter order, with the label each group's events are shown under.
TIMELINE_GROUPS = (
    ("message", "Messages", "Message"),
    ("model", "Model", "Model"),
    ("tool", "Tools", "Tool"),
    ("memory", "Memory", "Memory"),
    ("output", "Output", "Output"),
    ("run", "Run", "Run"),
)
# Statuses that only restate a normal event, so the timeline doesn't chip them.
_ROUTINE_STATUSES = {
    "",
    "success",
    "ok",
    "completed",
    "recorded",
    "started",
    "measured",
}
_HANDLED_OUTCOMES = {"success", "empty", "fallback", "degraded"}


def timeline_group(event: Mapping[str, Any]) -> str:
    """The kind decides; a runtime span's role is "tool" whatever it records."""
    kind = str(event.get("kind") or "")
    role = event.get("role")
    if kind == "conversation" or (not kind and role in ("user", "assistant")):
        return "message"
    if kind.startswith(("model", "cache", "prompt")):
        return "model"
    if kind.startswith("tool") or (not kind and role == "tool"):
        return "tool"
    if kind.startswith(("memory", "context")):
        return "memory"
    if kind.startswith("artifact") or kind in ("ui_delivery", "output_contract"):
        return "output"
    return "run"


def timeline_error(event: Mapping[str, Any]) -> bool:
    outcome = event.get("outcome")
    return bool(
        str(event.get("status") or "") in ("error", "partial")
        or event.get("error_code")
        or (outcome and outcome not in _HANDLED_OUTCOMES)
    )


# Why a run was cancelled, by the reason code its cancellation gave.
STOP_REASONS = {
    "reader_closed": "the reply stopped being read (the Stop button, a closed "
    "tab or a dropped connection)",
    "deadline_exceeded": "it ran past its time limit",
    "cancel_requested": "a cancel was requested",
}


def _plural(count: int, noun: str) -> str:
    return f"{count:,} {noun}{'' if count == 1 else 's'}"


def _names(names: List[str]) -> str:
    unique = list(dict.fromkeys(names))
    if len(unique) <= 2:
        return " and ".join(unique)
    return ", ".join(unique[:-1]) + " and " + unique[-1]


def _tool_name(event: Mapping[str, Any]) -> str:
    name = event.get("logical_tool_name") or event.get("tool_name")
    if not name:
        title = str(event.get("title") or "")
        name = title.split(":", 1)[1].strip() if ":" in title else ""
    return str(name or "a tool")


def explain_error(event: Mapping[str, Any]) -> str:
    """One plain sentence on why an event failed, from its codes only."""
    from ..observability.overview import explain_provider_error, explain_tool_reason

    status = str(event.get("status") or "")
    code = str(event.get("error_code") or "")
    if status == "cancelled" or code == "cancelled":
        reason = STOP_REASONS.get(str(event.get("stop_reason") or ""))
        return "The run was stopped before it finished" + (
            f" because {reason}." if reason else "."
        )
    if code == "approval_required" or status == "approval_required":
        return "The run is waiting for an approval."
    kind = str(event.get("kind") or "")
    if kind.startswith("tool"):
        reason = event.get("outcome_reason_code") or code
        if reason:
            return "The tool call failed: " + explain_tool_reason(str(reason)) + "."
        outcome = str(event.get("outcome") or "")
        return (
            f"The tool call failed ({outcome.replace('_', ' ') or 'no reason given'})."
        )
    if code:
        return "Failed: " + explain_provider_error(code) + "."
    return f"Ended with status {status.replace('_', ' ')}." if status else ""


def _turn_story(rows: List[Dict[str, Any]], result: Mapping[str, Any]) -> str:
    """What a turn that didn't finish had done before it stopped."""
    turn = result.get("turn_id")
    events = [row["event"] for row in rows if row["event"].get("turn_id") == turn]
    replies = [e for e in events if e.get("kind") == "model_result"]
    written = sum(int(e.get("response_chars") or 0) for e in replies)
    tools = [_tool_name(e) for e in events if e.get("kind") == "tool_call"]
    parts = []
    if replies:
        parts.append(
            _plural(len(replies), "model call")
            + (
                f" wrote {written:,} characters of reply"
                if written
                else " wrote no reply text"
            )
        )
    if tools:
        parts.append(f"it called {_names(tools)}")
    partial = int(result.get("partial_reply_chars") or 0)
    story = ("Before it stopped, " + "; ".join(parts) + ".") if parts else ""
    if partial:
        story += f" {partial:,} characters of reply had been sent."
    return story.strip()


def _model_note(event: Mapping[str, Any], earlier: List[Mapping[str, Any]]) -> str:
    """What a model call produced: reply text, a tool request, or nothing.

    The tools a call asked for run before its result is recorded, so they are
    the tool calls since the model call it closes."""
    chars = event.get("response_chars")
    if chars is None:
        return ""
    tools = []
    for before in reversed(earlier):
        if before.get("kind") in ("model_call", "model_result", "turn_start"):
            break
        if before.get("kind") == "tool_call":
            tools.insert(0, _tool_name(before))
    asked = f"asked for {_names(tools)}" if tools else ""
    if int(chars):
        return f"Wrote {int(chars):,} characters of reply" + (
            f" and {asked}." if asked else "."
        )
    return f"Wrote no text; {asked}." if asked else "Wrote no reply text."


def turn_verdicts(outcomes: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict]]:
    """Recorded outcomes and grader verdicts per turn (turn id, else root
    trace id), oldest first, ready to show as chips."""
    from ..observability.store import verdict_for

    by_turn: Dict[str, List[Dict[str, Any]]] = {}
    for row in outcomes or ():
        if not isinstance(row, Mapping):
            continue
        key = str(row.get("turn_id") or row.get("root_trace_id") or "")
        if not key:
            continue
        status = str(row.get("status") or "").lower()
        by_turn.setdefault(key, []).append(
            {
                "verdict": verdict_for(status),
                "status": status,
                "grader": str(row.get("grader") or row.get("source") or ""),
                "reason_code": str(row.get("reason_code") or ""),
                "score": _number(row.get("score")),
                "verified": row.get("verified") is True,
                "recorded_at": str(
                    row.get("updated_at") or row.get("created_at") or ""
                ),
            }
        )
    for rows in by_turn.values():
        rows.sort(key=lambda row: row["recorded_at"])
    return by_turn


def _attach_verdicts(rows: List[Dict[str, Any]], outcomes) -> None:
    """Put each turn's verdicts on its result row (else its start, else its
    last event)."""
    verdicts = turn_verdicts(outcomes)
    if not verdicts:
        return
    anchors: Dict[str, tuple] = {}
    for index, row in enumerate(rows):
        event = row["event"]
        kind = event.get("kind")
        rank = 2 if kind == "turn_result" else 1 if kind == "turn_start" else 0
        for value in (event.get("turn_id"), event.get("root_trace_id")):
            key = str(value or "")
            if key in verdicts and rank >= anchors.get(key, (0, -1))[1]:
                anchors[key] = (index, rank)
    for key, (index, _rank) in anchors.items():
        rows[index]["verdicts"].extend(verdicts[key])


def trace_timeline(
    events: Iterable[Mapping[str, Any]],
    *,
    agent_label: str = "Agent",
    content_event_ids: Iterable[Any] = (),
    previews_indexed: bool = False,
    outcomes: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Rows for the Trace Timeline: who acted, a readable title, the group
    the filter uses, and the time split into date and clock.

    A message is "Message" from the user and "Reply" from the agent. An event
    offers a reveal only when it has content to show: its own, or a preview
    in the provider's span index. A turn's start shows its run tags, and its
    result shows the outcomes and grader verdicts recorded for it.
    """
    singular = {key: label for key, _plural, label in TIMELINE_GROUPS}
    with_content = set(content_event_ids)
    rows: List[Dict[str, Any]] = []
    counts = {key: 0 for key, _plural, _label in TIMELINE_GROUPS}
    errors = 0
    for event in events:
        group = timeline_group(event)
        if group == "message":
            actor = "user" if event.get("role") == "user" else "agent"
            actor_label = "User" if actor == "user" else agent_label
            title = "Message" if actor == "user" else "Reply"
        else:
            actor, actor_label = group, singular[group]
            kind = str(event.get("kind") or "").replace("_", " ").title()
            title = str(event.get("title") or kind or "Event")
        stamp = str(event.get("timestamp") or "")
        status = str(event.get("status") or "")
        if status == event.get("cache_decision"):
            status = ""  # The cache chip already says it.
        error = timeline_error(event)
        counts[group] += 1
        errors += error
        rows.append(
            {
                "event": event,
                "group": group,
                "actor": actor,
                "actor_label": actor_label,
                "title": title,
                "timestamp": stamp,
                "date": stamp[:10],
                "clock": stamp[11:19],
                "status": "" if status in _ROUTINE_STATUSES else status,
                "error": error,
                "has_content": previews_indexed
                or event.get("event_id") in with_content,
                "why": explain_error(event) if error else "",
                "note": "",
                "tags": sorted((event.get("trace_tags") or {}).items())
                if event.get("kind") == "turn_start"
                and isinstance(event.get("trace_tags"), Mapping)
                else [],
                "verdicts": [],
            }
        )
    _attach_verdicts(rows, outcomes)
    for index, row in enumerate(rows):
        event = row["event"]
        if event.get("kind") == "model_result" and not row["error"]:
            row["note"] = _model_note(event, [r["event"] for r in rows[:index]])
        elif event.get("kind") == "turn_result" and row["error"]:
            row["note"] = _turn_story(rows, event)
            if row["has_content"]:
                row["note"] += " Reveal content shows the message and any reply."
    filters = [
        {"key": key, "label": plural, "count": counts[key]}
        for key, plural, _label in TIMELINE_GROUPS
        if counts[key]
    ]
    if errors:
        filters.append({"key": "errors", "label": "Errors", "count": errors})
    threads = {str(row["event"].get("thread_id") or "") for row in rows} - {""}
    memories = {str(row["event"].get("memory_id") or "") for row in rows} - {""}
    return {
        "rows": rows,
        # A filter earns its place when there is more than one thing to pick.
        "filters": filters if len(filters) > 1 else [],
        "thread_id": next(iter(threads)) if len(threads) == 1 else "",
        "memory_id": next(iter(memories)) if len(memories) == 1 else "",
        "first": rows[0]["timestamp"] if rows else "",
        "last": rows[-1]["timestamp"] if rows else "",
        "revealable": sum(1 for row in rows if row["has_content"]),
    }


# ---------- Usage and cost ----------


def usage_tape(usage: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Calls, tokens and charges first: the order the usage suite reads them."""
    totals = usage.get("totals") or {}
    cache = usage.get("prompt_cache") or {}
    coverage = usage.get("coverage") or {}
    unpriced = _int(totals.get("unpriced_calls"))
    missing = _int(totals.get("unknown_usage_calls"))
    cost = totals.get("cost_usd")
    cost_title = exact_usd(cost)
    if cost_title:
        cost_title = f"Exact {cost_title}"
        if unpriced:
            cost_title += f"; {unpriced} unpriced calls are excluded, not free"
    read_complete = coverage.get("read_complete")
    return [
        item("Calls", _int(totals.get("calls"))),
        item(
            "Tokens",
            _int(totals.get("total_tokens")),
            title="{:,} input · {:,} output".format(
                _int(totals.get("input_tokens")), _int(totals.get("output_tokens"))
            ),
        ),
        item("Known charge", cost, "usd", title=cost_title),
        item("Unpriced calls", unpriced, tone=WARN if unpriced else ""),
        item("Missing usage", missing, tone=WARN if missing else ""),
        item(
            "Cache read",
            cache.get("read_percent"),
            "percent",
            WARN if cache.get("attention") else "",
            title="Share of measured input tokens read from the provider cache",
        ),
        item("Mean latency", totals.get("mean_latency_ms"), "ms"),
        item("Latency p95", totals.get("p95_latency_ms"), "ms"),
        item(
            "Events read",
            _int(coverage.get("events_scanned")),
            tone=GOOD if read_complete else WARN,
            title="Read complete"
            if read_complete
            else "Read partial or not established",
        ),
    ]


# ---------- Health ----------


def health_tape(health: Mapping[str, Any], *, now: float) -> List[Dict[str, Any]]:
    coverage = str(health.get("coverage") or "unknown")
    alerts = list(health.get("alerts") or [])
    severe = any(
        str(alert.get("severity")) in {"critical", "error"} for alert in alerts
    )
    failed = len(health.get("failed_queries") or [])
    errors = _int(health.get("normalization_errors"))
    missing = _int(health.get("missing_identity_events"))
    orphans = _int(health.get("orphan_spans"))
    latest = health.get("latest_event_at")
    index = health.get("index") or {}
    truncated = bool(health.get("truncated"))
    return [
        item(
            "Coverage",
            coverage.capitalize(),
            "text",
            GOOD if coverage == "complete" else WARN,
            title=str(health.get("headline") or ""),
        ),
        item("Alerts", len(alerts), tone=BAD if severe else (WARN if alerts else GOOD)),
        item("Failed queries", failed, tone=BAD if failed else GOOD),
        item(
            "Errors",
            errors,
            tone=BAD if errors else GOOD,
            title="Events that could not be normalized",
        ),
        item("Events", _int(health.get("normalized_events"))),
        item("Missing identity", missing, tone=WARN if missing else ""),
        item("Orphan spans", orphans, tone=WARN if orphans else ""),
        item(
            "Span index",
            "Ready" if index.get("ready") else "Not ready",
            "text",
            GOOD if index.get("ready") else "",
            title=f"Schema {index.get('schema_version') or 'unknown'}",
        ),
        item(
            "Latest event",
            relative_age(iso_timestamp(latest), now),
            "text",
            title=str(latest or ""),
        ),
        item(
            "Truncated", "Yes" if truncated else "No", "text", WARN if truncated else ""
        ),
    ]


# ---------- Compare ----------


def compare_tape(
    comparison: Optional[Mapping[str, Any]],
    *,
    agent_id: Optional[str],
    baseline_root: Optional[str],
    candidate_root: Optional[str],
) -> List[Dict[str, Any]]:
    items = [
        item("Agent", agent_id or "—", "text", title=agent_id or ""),
        item("Baseline root", baseline_root or "—", "text", title=baseline_root or ""),
        item(
            "Candidate root", candidate_root or "—", "text", title=candidate_root or ""
        ),
    ]
    if not comparison:
        items.append(item("Status", "Choose two roots", "text", meta=True))
        return items
    findings = comparison.get("findings") or {}
    changes = comparison.get("changes") or {}
    only_baseline = len(findings.get("only_in_baseline") or [])
    only_candidate = len(findings.get("only_in_candidate") or [])
    changed = sum(
        1
        for change in changes.values()
        if (change or {}).get("added") or (change or {}).get("removed")
    )
    items += [
        item(
            "Baseline events",
            _int((comparison.get("baseline") or {}).get("normalized_events")),
        ),
        item(
            "Candidate events",
            _int((comparison.get("candidate") or {}).get("normalized_events")),
        ),
        item("Only in baseline", only_baseline, tone=WARN if only_baseline else ""),
        item("Only in candidate", only_candidate, tone=WARN if only_candidate else ""),
        item("In both", len(findings.get("in_both") or [])),
        item(
            "Changed categories",
            changed,
            tone=WARN if changed else GOOD,
            title=f"{changed} of {len(changes)} categories differ",
        ),
    ]
    return items
