"""What failed across agents, for a time range, and how to get a copy of it."""

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ...observability.normalization import timestamp
from ...observability.overview import DEFAULT_WINDOW, WINDOWS, resolve_window
from ..dashboard import (
    DEFERRED_PROVIDERS,
    TRACE_EVENT_LIMIT,
    TRACE_RECORD_LIMIT,
    _agent_names,
    template_helpers,
)
from ..failure_alerts import ALERT_WINDOWS, alert_messages, statuses
from ..observability_view import fleet_failures, window_phrase
from ..security import audit_trace_view
from ..state import _state, templates
from ..trace_access import current_principal, require_trace_permission
from ..usage_budget import can_edit
from .traces import _inspection_context, _json_safe, _selection_window
from .usage import _csv_cell

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])

FAILURES_WINDOW = "24h"


def _stamp_of(event) -> Optional[datetime]:
    stamp = timestamp(event.get("timestamp"))
    return datetime.fromisoformat(stamp) if stamp else None


def _in_span(events, span) -> List[Dict[str, Any]]:
    start, end = span["start"], span["end"]
    return [
        event
        for event in events
        if (stamp := _stamp_of(event)) and start <= stamp <= end
    ]


def read_fleet_failures(span: Dict[str, Any]) -> Dict[str, Any]:
    """Failures in a span, through the one authorized read every trace view
    uses, so an account sees only what its scope allows."""
    _, events = _selection_window(
        request=None, limit=TRACE_RECORD_LIMIT, event_limit=TRACE_EVENT_LIMIT
    )
    names = {} if current_principal.get().restricted else _agent_names()
    coverage = getattr(events, "coverage", {}) or {}
    stamps = [stamp for stamp in map(_stamp_of, events) if stamp]
    return {
        "fleet": fleet_failures(_in_span(events, span), names=names),
        # The read stopped at its limit before reaching the start of the range,
        # so the counts are a floor. Hitting the limit alone proves nothing.
        "truncated": bool(coverage.get("truncated") or coverage.get("window_truncated"))
        and (not stamps or min(stamps) > span["start"]),
    }


def _csv(fleet: Dict[str, Any]) -> Response:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "failure",
            "code",
            "why",
            "count",
            "turns",
            "stopped_a_turn",
            "agents",
            "last_seen",
        ]
    )
    for group in fleet["groups"]:
        writer.writerow(
            [
                _csv_cell(group["name"]),
                _csv_cell(group["code"]),
                _csv_cell(group["why"]),
                group["count"],
                group["turns"],
                "yes" if group["stopped"] else "no",
                _csv_cell(" ".join(group["agents"])),
                f"{group['last_date']} {group['last_clock']}".strip(),
            ]
        )
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="failures.csv"'},
    )


@router.get("/traces/failures", response_class=HTMLResponse)
@router.get("/traces/failures.json", response_class=JSONResponse)
@router.get("/traces/failures.csv")
def failures_page(
    request: Request,
    window: str = FAILURES_WINDOW,
    start: str = "",
    end: str = "",
    load: int = 0,
):
    require_trace_permission("trace.read")
    if not _state.get("provider"):
        raise HTTPException(status_code=503, detail="No memory provider connected")
    now = datetime.now(timezone.utc)
    span = resolve_window(window, start=start, end=end, now=now)
    page = request.url.path == "/traces/failures"
    deferred = page and _state.get("provider_type") in DEFERRED_PROVIDERS and not load
    read = (
        {"fleet": fleet_failures([]), "truncated": False}
        if deferred
        else read_fleet_failures(span)
    )
    fleet = read["fleet"]
    audit_trace_view(
        request,
        "trace_failures",
        result_count=fleet["total"],
        content_mode="metadata",
    )
    if request.url.path.endswith(".json"):
        return JSONResponse(
            _json_safe({**fleet, "truncated": read["truncated"], "window": span["key"]})
        )
    if request.url.path.endswith(".csv"):
        return _csv(fleet)
    # Rules cover every tenant, so a scoped account isn't shown them.
    rules = None if current_principal.get().restricted else statuses()
    return templates.TemplateResponse(
        "observability_failures.html",
        _inspection_context(
            request,
            **template_helpers(),
            fleet=fleet,
            truncated=read["truncated"],
            deferred=deferred,
            span=span,
            window=span["key"],
            windows=list(WINDOWS),
            phrase=window_phrase(span["key"]),
            rules=rules,
            agent_choices=sorted(
                _agent_names().items(), key=lambda item: item[1].lower()
            )
            if rules is not None
            else [],
            can_edit_rules=can_edit() and rules is not None,
            alert_windows=[(key, window_phrase(key)) for key in ALERT_WINDOWS],
            **{
                f"alert_{key}": value
                for key, value in alert_messages(request.query_params).items()
            },
        ),
    )


__all__ = ["DEFAULT_WINDOW", "failures_page", "read_fleet_failures", "router"]
