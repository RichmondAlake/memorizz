"""Data for the console home page: live run health plus platform inventory.

Every section degrades on its own: a store that cannot be read leaves its
panel empty with a reason instead of failing the page.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote, urlencode

from fastapi import HTTPException, Request

from ..observability.analytics import _quote, aggregate_usage
from ..observability.normalization import timestamp
from ..observability.overview import WINDOWS, build_operations_overview, resolve_window
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
from .trace_access import (
    current_principal,
    require_trace_permission,
    scoped_trace_filters,
)

logger = logging.getLogger(__name__)

# Providers without cheap queries (a remote API per page of records) only read
# traces when the viewer asks, never on every home-page load.
DEFERRED_PROVIDERS = {"notion"}

# Bounded read: the newest trace records, as the Observability pages load them.
TRACE_RECORD_LIMIT = 1000
TRACE_EVENT_LIMIT = 20000
# The newest recorded outcomes and grader verdicts the overview joins to runs.
VERDICT_RECORD_LIMIT = 1000


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
    """The run's trace, scrolled to its timeline."""
    return (
        "/traces?"
        + urlencode(
            {
                **scope,
                **{
                    key: run[key]
                    for key in ("agent_id", "thread_id", "root_trace_id", "turn_id")
                    if run.get(key)
                },
            }
        )
        + "#trace-timeline"
    )


# Tool failures fixed in the agent's tool settings: the tool is missing, not
# callable, not disclosed, or called with arguments its signature refuses.
_TOOL_SETUP_REASONS = {
    "unknown_tool",
    "tool_not_callable",
    "tool_not_disclosed",
    "invalid_tool_invocation",
    "invalid_arguments",
}
# Built-in tools an agent gets from its memory or persona settings, so a
# missing one is fixed there, not in its tool list: section, link label.
_SETTINGS_TOOLS = {
    "remember": ("memory", "Fix the agent's memory settings"),
    "read_persona": ("persona", "Fix the agent's persona"),
}


def _tool_settings(name: str):
    if name.startswith("entity_memory_"):
        return _SETTINGS_TOOLS["remember"]
    return _SETTINGS_TOOLS.get(name, ("tools", "Fix the agent's tools"))


def link_attention(
    items: List[Dict[str, Any]], editable_agent_ids: Iterable[str]
) -> List[Dict[str, Any]]:
    """Give each needs-attention item ``actions``: links to where it can be
    fixed, the first the main one. Agent settings are linked only for agents
    saved in this store; a plugin's or a deleted agent's runs have none."""
    editable = set(editable_agent_ids)
    scope = scoped_trace_filters()

    def settings(agent_id: str, section: str, label: str):
        if agent_id not in editable:
            return None
        return {
            "label": label,
            "href": f"/agents/{quote(agent_id, safe='')}/edit#section-{section}",
        }

    for item in items:
        run, kind = item.get("run"), item.get("kind")
        trace = (
            {"label": "Open the run", "href": _trace_url(run, scope)} if run else None
        )
        agent_id = item.get("agent_id") or (run or {}).get("agent_id") or ""
        if kind == "failed_runs":
            if trace and item.get("count", 1) > 1:
                trace["label"] = "Open the newest failed run"
            actions = [trace]
        elif kind == "graded_failures":
            if trace:
                trace["label"] = (
                    "Open the newest failing turn"
                    if item.get("count", 1) > 1
                    else "Open the failing turn"
                )
            actions = [trace]
        elif kind == "tool" and item.get("reason") in _TOOL_SETUP_REASONS:
            actions = [settings(agent_id, *_tool_settings(item["tool"])), trace]
        elif kind == "tool":
            actions = [trace, settings(agent_id, "tools", "Agent tools")]
        elif kind == "provider_errors":
            actions = [trace, settings(agent_id, "model", "Model settings")]
        elif kind == "unpriced":
            actions = [
                {
                    "label": "Set a price",
                    "href": "/traces/usage/rates?"
                    + urlencode(
                        {
                            "provider": item.get("provider") or "",
                            "model": item.get("model") or "",
                            "service_tier": item.get("service_tier") or "default",
                        }
                    ),
                }
            ]
        elif kind == "duplicate_agent":
            actions = [settings(item["agent_id"], "identity", "Open the agent")]
        else:
            actions = [
                {"label": item.get("action") or "Open", "href": item["href"]}
                if item.get("href")
                else None
            ]
        item["actions"] = [action for action in actions if action]
    return items


def load_verdicts(limit: int = VERDICT_RECORD_LIMIT) -> List[Dict[str, Any]]:
    """The newest recorded outcomes and grader verdicts, as stored payloads.

    A bounded read of ``record_task_outcome`` records. Scoped (restricted)
    viewers get none, as on the Traces page. Never raises.
    """
    from ..enums.memory_type import MemoryType
    from ..observability.normalization import read_payload

    provider = _state.get("provider")
    query = getattr(provider, "query_observability_records", None)
    if not callable(query) or current_principal.get().restricted:
        return []
    try:
        page = query(
            MemoryType.SHARED_MEMORY, record_type="observability_outcome", limit=limit
        )
    except Exception:
        logger.warning("Recorded outcomes could not be read", exc_info=True)
        return []
    rows = []
    for row in page.get("items") or []:
        payload = read_payload(row) if isinstance(row, dict) else None
        if payload and payload.get("record_type") == "observability_outcome":
            rows.append(payload)
    return rows


def load_run_health(
    request: Request,
    window: str,
    *,
    now: datetime,
    action: str,
    load: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Read recent traces once and summarize them; shared by overview pages.

    ``window`` is a key of ``WINDOWS`` or ``custom`` from ``start`` to ``end``.
    Returns ``overview`` (or None), ``events`` inside the window, ``span``,
    ``error``, ``deferred`` and ``coverage``. Never raises for missing access
    or stores.
    """
    span = resolve_window(window, start=start, end=end, now=now)
    result: Dict[str, Any] = {
        "overview": None,
        "events": [],
        "span": span,
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

    # A custom range lists every run it holds, up to a page of them.
    overview = build_operations_overview(
        events,
        now=now,
        price=price,
        span=span,
        recent_limit=50 if span["key"] == "custom" else 12,
        verdicts=load_verdicts(),
    )
    window_start, window_end = overview["window_start"], overview["window_end"]
    result["events"] = [
        event
        for event in events
        if timestamp(event.get("timestamp"))
        and window_start
        <= datetime.fromisoformat(timestamp(event.get("timestamp")))
        <= window_end
    ]
    scope = scoped_trace_filters()
    for run in overview["recent"]:
        run["trace_url"] = _trace_url(run, scope)
    for agent in overview["agents"]:
        for run in agent["recent"]:
            run["trace_url"] = _trace_url(run, scope)
    for bucket in overview["series"]:
        if bucket["run"]:
            bucket["trace_url"] = _trace_url(bucket["run"], scope)
    coverage = getattr(events, "coverage", {}) or {}
    times = [
        stamp for stamp in (timestamp(e.get("timestamp")) for e in events) if stamp
    ]
    result["coverage"] = {
        "records": coverage.get("source_rows"),
        "truncated": bool(
            coverage.get("truncated") or coverage.get("window_truncated")
        ),
        # The oldest event read, so a range reaching further back can say
        # that older runs were not loaded.
        "oldest": datetime.fromisoformat(min(times)) if times else None,
    }
    result["overview"] = overview
    audit_trace_view(
        request,
        action,
        result_count=overview["totals"]["runs"],
        content_mode="metadata",
    )
    return result


def coding_agent_sessions(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    *,
    by_model: bool = False,
) -> List[Dict[str, Any]]:
    """Codex and Claude Code sessions the MemoRizz plugin recorded as runs,
    active between ``start`` and ``end``, per agent (and model). Their usage
    comes from the agents' own session logs, so it is reported apart from
    MemoRizz's own agent runs."""
    from .state import harness_run_store

    try:
        with harness_run_store() as store:
            runs = store.list(limit=2000)
    except Exception:
        logger.warning("Coding-agent sessions could not be read", exc_info=True)
        return []
    labels = {"codex": "Codex", "claude-code": "Claude Code"}
    rows: Dict[tuple, Dict[str, Any]] = {}
    for run in runs:
        task = run.task or {}
        metadata = task.get("metadata") or {}
        moment = timestamp(run.updated_at)
        if metadata.get("source") != "plugin" or not moment:
            continue
        updated = datetime.fromisoformat(moment)
        if (start and updated < start) or (end and updated > end):
            continue
        result = run.result or {}
        usage = result.get("usage") or {}
        model = str(usage.get("model") or task.get("model") or "")
        key = (str(run.harness), model if by_model else "")
        row = rows.setdefault(
            key,
            {
                "agent": key[0],
                "label": labels.get(key[0], key[0]),
                "model": key[1],
                "sessions": 0,
                "turns": 0,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "cost_usd": Decimal(0),
                "priced": 0,
                "last_at": None,
            },
        )
        cached = int(usage.get("cached_input_tokens") or 0) + int(
            usage.get("cache_read_input_tokens") or 0
        )
        # Codex counts cached tokens inside input_tokens; Claude Code apart.
        uncounted = int(usage.get("cache_read_input_tokens") or 0) + int(
            usage.get("cache_creation_input_tokens") or 0
        )
        row["sessions"] += 1
        row["turns"] += int(metadata.get("turns") or 0)
        row["input_tokens"] += int(usage.get("input_tokens") or 0) + uncounted
        row["cached_tokens"] += cached
        row["output_tokens"] += int(usage.get("output_tokens") or 0)
        if result.get("cost_usd") is not None:
            row["cost_usd"] += Decimal(str(result["cost_usd"]))
            row["priced"] += 1
        if row["last_at"] is None or updated > row["last_at"]:
            row["last_at"] = updated
    return sorted(rows.values(), key=lambda row: (row["label"], row["model"]))


def build_dashboard_context(
    request: Request,
    window: str,
    *,
    automation_store=None,
    load: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Collect the overview, per-model usage, automations and agent names."""
    now = datetime.now(timezone.utc)
    health = load_run_health(
        request,
        window,
        now=now,
        action="dashboard_overview",
        load=load,
        start=start,
        end=end,
    )
    span = health["span"]
    context: Dict[str, Any] = {
        "window": span["key"],
        "windows": list(WINDOWS),
        "span": span,
        "generated_at": now,
        "overview": health["overview"],
        "usage": None,
        "trace_error": health["error"],
        "deferred": health["deferred"],
        "loaded": load,
        "coverage": health["coverage"],
        "agent_names": {},
        "automations": {"available": False, "deferred": True},
        "coding_agents": coding_agent_sessions(span["start"], span["end"]),
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
                "action": "Open the prompt-cache report",
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
                "action": "Open the automation",
            }
        )
    from .usage_budget import budget_attention

    budget = budget_attention(request, now=now)
    if budget:
        overview["attention"].insert(0, budget)
    link_attention(overview["attention"], context["agent_names"])
    return context


def template_helpers() -> Dict[str, Any]:
    """Formatting helpers shared by the overview templates."""
    return {
        "fmt_duration": format_duration,
        "fmt_compact": format_compact,
        "fmt_usd": format_usd,
        "relative_time": relative_time,
        "sparkline": sparkline_points,
        "fmt_range": format_range,
        "range_query": range_query,
    }


def format_range(span: Dict[str, Any]) -> str:
    """A custom range as one short label, such as "Oct 15 06:00 – 18:00 UTC"."""
    start, end = span["start"], span["end"]
    if start.time() == end.time() == datetime.min.time():
        return f"{start:%b %d} – {end:%b %d, %Y}"
    if start.date() == end.date():
        return f"{start:%b %d %H:%M} – {end:%H:%M} UTC"
    return f"{start:%b %d %H:%M} – {end:%b %d %H:%M} UTC"


def range_query(span: Dict[str, Any]) -> str:
    """The query parameters that select ``span`` again, for links on its page."""
    if span["key"] != "custom":
        return urlencode({"window": span["key"]})
    return urlencode(
        {
            "window": "custom",
            "start": span["start"].strftime("%Y-%m-%dT%H:%M:%S"),
            "end": span["end"].strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )


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
