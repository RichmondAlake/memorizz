"""Authorized, content-free usage pages. Sync routes run reads off the event loop."""

from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ...observability.analytics import aggregate_usage
from ...observability.pricing import DEFAULT_PRICING
from ...observability.usage_query import query_usage
from ..analytics import usage_charts
from ..security import audit_trace_view
from ..state import _state, templates
from ..trace_access import require_trace_permission, scoped_trace_filters

router = APIRouter(tags=["usage"])


@router.get("/traces/usage", response_class=HTMLResponse)
@router.get("/traces/usage.json", response_class=JSONResponse)
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
            "start_time": start_time,
            "end_time": end_time,
        }.items()
        if value
    }
    if any(len(value) > 240 for value in selection.values()) or len(model) > 240:
        raise HTTPException(status_code=400, detail="Usage filters are too long")
    filters = {
        key: value
        for key, value in selection.items()
        if key not in {"agent_id", "task_id", "thread_memory_id"}
    }
    if agent_id:
        filters["agent_ids"] = [agent_id]
    filters = scoped_trace_filters(filters)
    pricing = request.app.state.usage_pricing or DEFAULT_PRICING
    try:
        if task_id or thread_memory_id:
            # Preserve the same legacy association rules as the selected trace.
            from .traces import _selection_window

            _, events = _selection_window(
                request=request,
                agent_id=agent_id or None,
                thread_id=thread_id or None,
                root_trace_id=root_trace_id or None,
                turn_id=turn_id or None,
                task_id=task_id or None,
                thread_memory_id=thread_memory_id or None,
                limit=1000,
                event_limit=max_events,
            )
            usage = aggregate_usage(
                events,
                pricing=pricing,
                timezone_name=timezone_name,
                max_events=max_events,
                model=model or None,
            )
        else:
            usage = query_usage(
                provider,
                filters=filters,
                pricing=pricing,
                timezone_name=timezone_name,
                max_events=max_events,
                model=model or None,
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
    scope = scoped_trace_filters()
    for row in usage["interactions"]:
        if row.get("root_trace_id") or row.get("thread_id"):
            row["trace_url"] = "/traces?" + urlencode(
                {
                    **scope,
                    **{
                        key: row[key]
                        for key in ("agent_id", "thread_id", "root_trace_id", "turn_id")
                        if row.get(key)
                    },
                }
            )
    params = {
        **selection,
        **scope,
        "model": model,
        "timezone_name": timezone_name,
        "max_events": max_events,
    }
    return templates.TemplateResponse(
        "usage.html",
        {
            "request": request,
            "active_page": "traces",
            "agents_nav": [],
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "usage": usage,
            "usage_charts": usage_charts(usage),
            "filters": params,
            "preserved_filters": {
                key: value
                for key, value in {**selection, **scope}.items()
                if key not in {"agent_id", "start_time", "end_time"}
            },
            "max_events": max_events,
            "export_url": "/traces/usage.json?" + urlencode(params),
            "trace_url": "/traces?" + urlencode({**selection, **scope}),
        },
    )
