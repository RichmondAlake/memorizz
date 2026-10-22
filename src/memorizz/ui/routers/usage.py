"""Authorized, content-free usage pages. Sync routes run reads off the event loop."""

import csv
import io
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ...observability.analytics import aggregate_usage
from ...observability.overview import WINDOWS, _when, resolve_window
from ...observability.pricing import DEFAULT_PRICING
from ...observability.usage_query import query_usage
from .. import usage_view
from ..analytics import memory_label, usage_charts
from ..security import audit_trace_view
from ..state import _state, templates
from ..trace_access import require_trace_permission, scoped_trace_filters

router = APIRouter(tags=["usage"])

# The page opens on the last 30 days; a trace selection, the JSON and the CSV
# cover all recorded usage unless given a window.
PAGE_WINDOW = "30d"
# CSV exports by what each row is: group parameter -> usage table.
CSV_GROUPS = {
    "day": "daily",
    "agent": "agents",
    "model": "models",
    "run": "interactions",
}
CSV_COLUMNS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "total_tokens",
    "cost_usd",
    "priced_calls",
    "unpriced_calls",
    "unknown_usage_calls",
    "mean_latency_ms",
    "p95_latency_ms",
)
# Zones offered for day boundaries; any IANA name still works in the URL.
TIMEZONES = (
    "UTC",
    "America/Los_Angeles",
    "America/New_York",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Paris",
    "Africa/Lagos",
    "Asia/Dubai",
    "Asia/Kolkata",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Australia/Sydney",
)


def preserved_filters_of(selection, scope):
    """The trace selection the usage page keeps across its own filters."""
    return {
        key: value
        for key, value in {**selection, **scope}.items()
        if key not in {"agent_id", "start_time", "end_time"}
    }


def usage_span(window: str, start: str, end: str, *, default: str, now: datetime):
    """The span a usage view covers. ``all`` has no time bounds; its start
    only fills the custom range form."""
    if (window or default) == "all":
        return {
            "key": "all",
            "start": now - WINDOWS["30d"][0],
            "end": now,
            "bucket": timedelta(days=1),
            "error": "",
        }
    return resolve_window(window or default, start=start, end=end, now=now)


def _read_usage(
    request,
    *,
    selection,
    agent_id,
    model,
    timezone_name,
    max_events,
    start=None,
    end=None,
    interval=None,
):
    """Usage between ``start`` and ``end`` (both optional) for the selection."""
    pricing = request.app.state.usage_pricing or DEFAULT_PRICING
    interval_seconds = int(interval.total_seconds()) if interval else None
    if selection.get("task_id") or selection.get("thread_memory_id"):
        # Preserve the same legacy association rules as the selected trace.
        from .traces import _selection_window

        _, events = _selection_window(
            request=None,
            agent_id=agent_id or None,
            thread_id=selection.get("thread_id"),
            root_trace_id=selection.get("root_trace_id"),
            turn_id=selection.get("turn_id"),
            task_id=selection.get("task_id"),
            thread_memory_id=selection.get("thread_memory_id"),
            limit=1000,
            event_limit=max_events,
        )
        if start or end:
            low = start.isoformat() if start else ""
            high = end.isoformat() if end else "~"
            events = [
                event
                for event in events
                if low <= str(event.get("timestamp") or "") <= high
            ]
        return aggregate_usage(
            events,
            pricing=pricing,
            timezone_name=timezone_name,
            max_events=max_events,
            model=model or None,
            interval_seconds=interval_seconds,
        )
    filters = {
        key: value
        for key, value in selection.items()
        if key not in {"agent_id", "task_id", "thread_memory_id"}
    }
    if agent_id:
        filters["agent_ids"] = [agent_id]
    if start:
        filters["start_time"] = start.isoformat()
    if end:
        filters["end_time"] = end.isoformat()
    return query_usage(
        _state["provider"],
        filters=scoped_trace_filters(filters),
        pricing=pricing,
        timezone_name=timezone_name,
        max_events=max_events,
        model=model or None,
        interval_seconds=interval_seconds,
    )


def _csv_cell(value):
    # A cell a spreadsheet would run as a formula starts with a quote instead.
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _usage_csv(usage, group, names):
    rows = usage.get(CSV_GROUPS.get(group, "interactions")) or []
    out = io.StringIO()
    writer = csv.writer(out)
    lead = {
        "day": ["date"],
        "agent": ["agent_id", "agent_name"],
        "model": ["provider", "model"],
        "run": ["started_at", "agent_id", "agent_name", "thread_id", "root_trace_id"],
    }.get(group, ["started_at", "agent_id", "agent_name", "thread_id", "root_trace_id"])
    writer.writerow([*lead, *CSV_COLUMNS])
    for row in rows:
        if group == "day":
            first = [row["label"]]
        elif group == "agent":
            first = [row["label"], names.get(row["label"], "")]
        elif group == "model":
            first = list(row["label"].partition(" / ")[::2])
        else:
            agent = row.get("agent_id") or ""
            first = [
                row.get("started_at") or "",
                agent,
                names.get(agent, ""),
                row.get("thread_id") or "",
                row.get("root_trace_id") or "",
            ]
        writer.writerow(
            [_csv_cell(v) for v in (*first, *(row.get(c) for c in CSV_COLUMNS))]
        )
    group_name = group if group in CSV_GROUPS else "run"
    return Response(
        out.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="memorizz-usage-by-{group_name}.csv"'
        },
    )


@router.get("/traces/usage", response_class=HTMLResponse)
@router.get("/traces/usage.json", response_class=JSONResponse)
@router.get("/traces/usage.csv")
def usage_page(
    request: Request,
    agent_id: str = "",
    thread_id: str = "",
    root_trace_id: str = "",
    turn_id: str = "",
    run_id: str = "",
    thread_memory_id: str = "",
    task_id: str = "",
    start_time: str = "",
    end_time: str = "",
    window: str = "",
    start: str = "",
    end: str = "",
    metric: str = "tokens",
    group: str = "run",
    model: str = "",
    timezone_name: str = "UTC",
    max_events: int = Query(5000, ge=1, le=10000),
):
    require_trace_permission("trace.read")
    provider = _state.get("provider")
    if provider is None:
        raise HTTPException(status_code=503, detail="No memory provider connected")
    try:
        ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError):
        raise HTTPException(status_code=400, detail="Unknown IANA timezone") from None
    selection = {
        key: value
        for key, value in {
            "agent_id": agent_id,
            "thread_id": thread_id,
            "root_trace_id": root_trace_id,
            "turn_id": turn_id,
            "run_id": run_id,
            "thread_memory_id": thread_memory_id,
            "task_id": task_id,
        }.items()
        if value
    }
    texts = [*selection.values(), model, window, start, end, start_time, end_time]
    if any(len(value) > 240 for value in texts):
        raise HTTPException(status_code=400, detail="Usage filters are too long")
    metric = metric if metric in usage_view.METRICS else "tokens"
    page = request.url.path.endswith("/usage")
    # start_time and end_time, the API's bounds before the range picker, keep
    # their meaning: either may be empty, and an unreadable one is refused.
    bounds = None
    if (start_time or end_time) and not window:
        bounds = (_when(start_time), _when(end_time))
        if (start_time and bounds[0] is None) or (end_time and bounds[1] is None):
            raise HTTPException(status_code=400, detail="Invalid usage selection")
        window = "all"
        if page and bounds[0]:
            window, start, end, bounds = "custom", start_time, end_time, None
    scope = scoped_trace_filters()
    trace_selection = {k: v for k, v in selection.items() if k != "agent_id"}
    now = datetime.now(timezone.utc)
    span = usage_span(
        window,
        start,
        end,
        default=PAGE_WINDOW if page and not trace_selection else "all",
        now=now,
    )
    if span["error"] and not page:
        raise HTTPException(status_code=400, detail=span["error"])
    bounded = span["key"] != "all"
    read = dict(
        selection=selection,
        agent_id=agent_id,
        model=model,
        timezone_name=timezone_name,
        max_events=max_events,
    )
    try:
        # Read in this request's context: the tenant scope is a context
        # variable, which a worker thread would not see.
        usage = _read_usage(
            request,
            **read,
            start=span["start"] if bounded else (bounds or (None, None))[0],
            end=span["end"] if bounded else (bounds or (None, None))[1],
            interval=span["bucket"],
        )
        # The same length of time just before, for each figure's change.
        previous = (
            _read_usage(
                request,
                **read,
                start=span["start"] - (span["end"] - span["start"]),
                end=span["start"],
            )
            if bounded and page
            else None
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid usage selection") from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503, detail="Usage query failed; coverage is unavailable"
        ) from None
    audit_trace_view(
        request,
        "trace_usage",
        result_count=usage["totals"]["calls"],
        content_mode="metadata",
    )
    if request.url.path.endswith(".json"):
        return JSONResponse(usage)
    from ..dashboard import _agent_names, template_helpers

    names = _agent_names()
    if request.url.path.endswith(".csv"):
        return _usage_csv(usage, group, names)
    for row in usage["interactions"]:
        if row.get("root_trace_id") or row.get("thread_id"):
            keys = ("agent_id", "thread_id", "root_trace_id", "turn_id")
            row["trace_url"] = (
                "/traces?"
                + urlencode({**scope, **{k: row[k] for k in keys if row.get(k)}})
                + "#trace-timeline"
            )
    # The page's own filters, so each link keeps all but the one it changes.
    query = {
        key: value
        for key, value in {
            **trace_selection,
            **scope,
            "agent_id": agent_id,
            "model": model,
            "timezone_name": "" if timezone_name == "UTC" else timezone_name,
            "max_events": "" if max_events == 5000 else str(max_events),
        }.items()
        if value
    }
    ranged = {"window": span["key"]}
    if span["key"] == "custom":
        ranged.update(
            start=span["start"].strftime("%Y-%m-%dT%H:%M:%S"),
            end=span["end"].strftime("%Y-%m-%dT%H:%M:%S"),
        )
    # Codex and Claude Code sessions are counted from their own session logs,
    # apart from traced agent runs; a trace selection or agent filter
    # narrows to traces, so they are shown only without one.
    coding_agents = []
    if not preserved_filters_of(selection, scope) and not agent_id:
        from ..dashboard import coding_agent_sessions

        coding_agents = [
            row
            for row in coding_agent_sessions(
                span["start"] if bounded else None,
                span["end"] if bounded else None,
                by_model=True,
            )
            if not model or model.lower() in row["model"].lower()
        ]
    from ..usage_budget import budget_messages, budget_status, can_edit

    budget = {
        "status": budget_status(request, now=now),
        "can_edit": can_edit(),
        **budget_messages(request.query_params),
    }
    view = usage_view.page(
        usage,
        previous,
        span=span,
        interval=span["bucket"],
        timezone_name=timezone_name,
        metric=metric,
        names=names,
        query=query,
        now=now,
    )
    return templates.TemplateResponse(
        "usage.html",
        {
            "coding_agents": coding_agents,
            "request": request,
            "active_page": "traces",
            "agents_nav": [],
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "usage": usage,
            "usage_charts": usage_charts(usage),
            "view": view,
            "span": span,
            "window": span["key"],
            "windows": [*WINDOWS, "all"],
            "metric": metric,
            "query": query,
            "ranged": ranged,
            # Saved agents, and agents seen in the window that aren't saved
            # here (a plugin's, or one from another store).
            "agent_choices": sorted(
                {
                    **names,
                    **{
                        row["agent_id"]: row["name"]
                        for row in view["agents"]
                        if row["agent_id"]
                    },
                    **({agent_id: names.get(agent_id, agent_id)} if agent_id else {}),
                }.items(),
                key=lambda item: item[1].lower(),
            ),
            "model_choices": sorted(
                {row["label"].partition(" / ")[2] for row in usage["models"]}
                | ({model} if model else set())
            ),
            "timezones": TIMEZONES
            if timezone_name in TIMEZONES
            else (*TIMEZONES, timezone_name),
            "timezone_name": timezone_name,
            "max_events": max_events,
            "preserved_filters": preserved_filters_of(selection, scope),
            "budget": budget,
            "export_query": urlencode({**query, **ranged}),
            "trace_url": "/traces?" + urlencode({**selection, **scope}),
            "generated_at": now,
            "memory_label": memory_label,
            **template_helpers(),
        },
    )
