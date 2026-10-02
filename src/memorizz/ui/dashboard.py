"""Data for the console home page: live run health plus platform inventory.

Every section degrades on its own: a store that cannot be read leaves its
panel empty with a reason instead of failing the page.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import HTTPException, Request

from ..observability.analytics import _quote, aggregate_usage
from ..observability.normalization import timestamp
from ..observability.overview import WINDOWS, build_operations_overview
from ..observability.pricing import DEFAULT_PRICING
from .helpers import (
    _duplicate_agent_notes,
    _extract_agent_identifier,
    _extract_agent_persona_name,
    _list_agents,
    _unique_agents,
)
from .observability_view import format_age
from .security import audit_trace_view
from .state import _state
from .trace_access import require_trace_permission, scoped_trace_filters

logger = logging.getLogger(__name__)

# Providers without cheap queries (a remote API per page of records) only read
# traces when the viewer asks, never on every home-page load.
DEFERRED_PROVIDERS = {"notion"}

# Bounded read: the newest trace records, as the Observability pages load them.
TRACE_RECORD_LIMIT = 1000
TRACE_EVENT_LIMIT = 20000


def _agent_names() -> Dict[str, str]:
    try:
        agents = _list_agents() or []
    except Exception:
        return {}
    names = {}
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if agent_id:
            names[agent_id] = _extract_agent_persona_name(agent) or agent_id
    return names


def _automation_summary(store, now: datetime) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "available": store is not None,
        "enabled": 0,
        "paused": 0,
        "upcoming": [],
        "failing": [],
    }
    if store is None:
        return summary
    try:
        jobs = store.list_jobs(agent_id=None, enabled=None)
    except Exception:
        summary["available"] = False
        return summary
    upcoming = []
    for job in jobs:
        if not job.enabled:
            summary["paused"] += 1
            continue
        summary["enabled"] += 1
        next_run = job.next_run_at
        if next_run is not None and next_run.tzinfo is None:
            next_run = next_run.replace(tzinfo=timezone.utc)
        latest = None
        try:
            runs = store.list_runs(job.job_id, limit=1)
            latest = runs[0] if runs else None
        except Exception:
            latest = None
        status = str(getattr(getattr(latest, "status", None), "value", "") or "")
        row = {
            "job_id": job.job_id,
            "name": job.name,
            "agent_id": job.agent_id,
            "next_run_at": next_run,
            "last_status": status,
        }
        upcoming.append(row)
        if status.lower() in {"failed", "error", "timed_out", "timeout"}:
            summary["failing"].append(row)
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    upcoming.sort(key=lambda row: row["next_run_at"] or far_future)
    summary["upcoming"] = upcoming[:5]
    return summary


def _trace_url(run: Dict[str, Any], scope: Dict[str, Any]) -> str:
    return "/traces?" + urlencode(
        {
            **scope,
            **{
                key: run[key]
                for key in ("agent_id", "thread_id", "root_trace_id", "turn_id")
                if run.get(key)
            },
        }
    )


def load_run_health(
    request: Request, window: str, *, now: datetime, action: str, load: bool = False
) -> Dict[str, Any]:
    """Read recent traces once and summarize them; shared by overview pages.

    Returns ``overview`` (or None), ``events`` inside the window, ``error``,
    ``deferred`` and ``coverage``. Never raises for missing access or stores.
    """
    result: Dict[str, Any] = {
        "overview": None,
        "events": [],
        "error": None,
        "deferred": _state.get("provider_type") in DEFERRED_PROVIDERS and not load,
        "coverage": {},
    }
    if result["deferred"]:
        return result
    try:
        require_trace_permission("trace.read")
    except HTTPException:
        result["error"] = "Your account cannot read execution traces."
        return result

    from .routers.traces import _selection_window

    try:
        _, events = _selection_window(
            request=None, limit=TRACE_RECORD_LIMIT, event_limit=TRACE_EVENT_LIMIT
        )
    except HTTPException as exc:
        result["error"] = str(exc.detail)
        return result
    except Exception:
        logger.warning("Run health trace read failed", exc_info=True)
        result["error"] = "Execution traces could not be read."
        return result

    pricing = getattr(request.app.state, "usage_pricing", None) or DEFAULT_PRICING

    def price(event):
        # Same rule as the usage page: recorded quotes first, then the catalog.
        quote = _quote(event, pricing)
        cost = quote.get("cost_usd")
        if cost in (None, ""):
            return None, str(quote.get("cost_reason") or "No price recorded")
        return Decimal(str(cost)), ""

    overview = build_operations_overview(events, window=window, now=now, price=price)
    window_start = overview["window_start"]
    result["events"] = [
        event
        for event in events
        if timestamp(event.get("timestamp"))
        and datetime.fromisoformat(timestamp(event.get("timestamp"))) >= window_start
    ]
    scope = scoped_trace_filters()
    for run in overview["recent"]:
        run["trace_url"] = _trace_url(run, scope)
    for agent in overview["agents"]:
        for run in agent["recent"]:
            run["trace_url"] = _trace_url(run, scope)
    coverage = getattr(events, "coverage", {}) or {}
    result["coverage"] = {
        "records": coverage.get("source_rows"),
        "truncated": bool(
            coverage.get("truncated") or coverage.get("window_truncated")
        ),
    }
    result["overview"] = overview
    audit_trace_view(
        request,
        action,
        result_count=overview["totals"]["runs"],
        content_mode="metadata",
    )
    return result


def build_dashboard_context(
    request: Request, window: str, *, automation_store=None, load: bool = False
) -> Dict[str, Any]:
    """Collect the overview, per-model usage, automations and agent names."""
    window = window if window in WINDOWS else "7d"
    now = datetime.now(timezone.utc)
    health = load_run_health(
        request, window, now=now, action="dashboard_overview", load=load
    )
    context: Dict[str, Any] = {
        "window": window,
        "windows": list(WINDOWS),
        "generated_at": now,
        "overview": health["overview"],
        "usage": None,
        "trace_error": health["error"],
        "deferred": health["deferred"],
        "loaded": load,
        "coverage": health["coverage"],
        "agent_names": {},
        "automations": {"available": False, "deferred": True},
    }
    if health["deferred"]:
        return context
    context["agent_names"] = _agent_names()
    context["automations"] = _automation_summary(automation_store, now)
    overview = health["overview"]
    if overview is None:
        return context
    pricing = getattr(request.app.state, "usage_pricing", None) or DEFAULT_PRICING
    try:
        context["usage"] = aggregate_usage(
            health["events"], pricing=pricing, max_events=TRACE_EVENT_LIMIT
        )
    except Exception:
        logger.warning("Dashboard usage aggregation failed", exc_info=True)
    cache = (context["usage"] or {}).get("prompt_cache") or {}
    for warning in cache.get("warnings") or []:
        overview["attention"].append(
            {
                "severity": "warning",
                "title": f"Prompt cache: {warning.get('model') or 'model'}",
                "detail": str(warning.get("message") or "")
                + (f" ({warning['count']}×)" if warning.get("count", 1) > 1 else ""),
                "href": "/traces/usage#provider-prompt-cache",
            }
        )
    try:
        _, copies = _unique_agents(_state["provider"].list_memagents() or [])
    except Exception:
        copies = {}
    overview["attention"][:0] = _duplicate_agent_notes(copies)
    for job in context["automations"]["failing"]:
        overview["attention"].append(
            {
                "severity": "error",
                "title": f"Automation {job['name']} failed its last run",
                "detail": "Open the automation to see the run error.",
                "href": f"/automations/{job['job_id']}",
            }
        )
    return context


def template_helpers() -> Dict[str, Any]:
    """Formatting helpers shared by the overview templates."""
    return {
        "fmt_duration": format_duration,
        "fmt_compact": format_compact,
        "fmt_usd": format_usd,
        "relative_time": relative_time,
        "sparkline": sparkline_points,
    }


def format_duration(ms: Optional[float]) -> str:
    if ms is None:
        return "—"
    if ms < 1000:
        return f"{ms:.0f} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def format_compact(value: Any) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "—"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(number) >= size:
            return f"{number / size:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{number:,.0f}"


def format_usd(value: Any) -> str:
    try:
        number = float(value or 0)
    except (TypeError, ValueError):
        return "—"
    if number and abs(number) < 0.01:
        return f"${number:.4f}"
    return f"${number:,.2f}"


def relative_time(value: Optional[datetime], now: datetime) -> str:
    if value is None:
        return "—"
    return format_age((now - value).total_seconds())


def sparkline_points(values: List[float], width: int = 120, height: int = 28) -> str:
    """SVG polyline points for a small trend line, padded so strokes aren't clipped."""
    if not values:
        return ""
    peak = max(values) or 1
    step = width / max(1, len(values) - 1)
    return " ".join(
        f"{index * step:.1f},{height - 2 - (value / peak) * (height - 4):.1f}"
        for index, value in enumerate(values)
    )
