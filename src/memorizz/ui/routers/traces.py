# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Traces / Observability pages.

Extracted verbatim from ``ui/app.py``: GET /observability (legacy redirect)
and GET /traces, plus the trace helpers used only by these routes. Route
paths, response classes, and behavior are unchanged.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...observability import ObservabilityStore, analyze_trace_events
from ...observability.coverage import trace_coverage
from ...observability.inspection import (
    build_causal_waterfall,
    build_trace_health,
    compare_trace_windows,
    event_anchor,
)
from ...observability.lineage import build_lineage_inspectors
from ...observability.normalization import (
    TraceEvents,
    TraceSnapshot,
    is_bundle,
    normalize_trace_snapshot,
    query_trace_events,
    select_trace_events,
)
from ..helpers import (
    _build_agent_nav_items,
    _build_agent_tool_count_map,
    _coerce_timestamp,
    _extract_agent_identifier,
    _extract_agent_memory_ids,
    _extract_agent_persona_name,
    _load_agent_last_run_map,
    _sort_agents_by_last_run_desc,
    _to_text,
)
from ..security import (
    audit_trace_view,
    redact_trace_events,
    trace_content_mode,
    ui_read_only,
)
from ..state import _state, templates
from ..trace_access import current_principal, scoped_trace_filters
from .trace_tools import trace_capabilities

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])

_RUNTIME_TRACE_AGENT_ID = "__memorizz_runtime_traces__"
_TRACE_BUNDLE_RECORD_TYPE = "observability_trace_bundle"


def _selection_window(
    *,
    request=None,
    agent_id=None,
    selected_agent=None,
    thread_id=None,
    thread_memory_id=None,
    root_trace_id=None,
    turn_id=None,
    task_id=None,
    limit=500,
    event_limit=1000,
):
    """One authorized identity/snapshot contract for every combined trace view."""
    if selected_agent is None and agent_id and not current_principal.get().restricted:
        try:
            selected_agent = next(
                (
                    a
                    for a in (_state["provider"].list_memagents() or [])
                    if _extract_agent_identifier(a) == agent_id
                ),
                None,
            )
        except Exception:
            raise HTTPException(
                status_code=503, detail="Trace agent associations are unavailable"
            ) from None
    ids = (
        sorted(_trace_agent_ids(selected_agent))
        if selected_agent
        else [agent_id]
        if agent_id
        else None
    )
    memories = _extract_agent_memory_ids(selected_agent) if selected_agent else []
    snapshot = _load_trace_snapshot(
        agent_ids=ids,
        memory_ids=memories,
        thread_id=thread_id,
        limit=max(1, min(limit, 1000)),
    )
    events = normalize_trace_snapshot(
        snapshot,
        agent_ids=set(ids or []),
        memory_ids=set(memories),
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
        **scoped_trace_filters(),
    ).events
    events.coverage["store_selection"] = ["conversation", "tool_log", "trace"]
    extra = {
        k: request.query_params[k]
        for k in ("start_time", "end_time", "run_id")
        if request is not None and k in request.query_params
    }
    try:
        events = select_trace_events(
            events,
            root_trace_id=root_trace_id,
            turn_id=turn_id,
            task_id=task_id,
            limit=event_limit,
            **extra,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid trace selection") from None
    return snapshot, events


def _selection_urls(request, **selection):
    selection.update(
        {
            k: request.query_params[k]
            for k in ("start_time", "end_time", "run_id")
            if k in request.query_params
        }
    )
    params = {k: v for k, v in selection.items() if v is not None and v != ""}
    params.update(scoped_trace_filters())
    return {
        "params": params,
        "traces": "/traces?" + urlencode(params),
        "health": "/traces/health?" + urlencode(params),
        "compare": "/traces/compare?"
        + urlencode(
            {
                **params,
                **(
                    {"baseline_root": params["root_trace_id"]}
                    if params.get("root_trace_id")
                    else {}
                ),
                **(
                    {"baseline_turn": params["turn_id"]}
                    if params.get("turn_id")
                    else {}
                ),
            }
        ),
    }


def _targeted_runtime_agent(agent_id, thread_id=None):
    snapshot = _load_trace_snapshot(agent_ids=[agent_id], thread_id=thread_id, limit=1)
    window = normalize_trace_snapshot(
        snapshot, agent_ids=[agent_id], **scoped_trace_filters()
    )
    if window.events:
        return {
            "agent_id": agent_id,
            "name": agent_id,
            "memory_ids": [],
            "_is_virtual_trace_source": True,
            "_runtime_agent_ids": [agent_id],
        }
    return None


@router.get("/observability")
async def observability_redirect(request: Request):
    """Backwards-compatible redirect to traces page."""
    query = request.url.query
    suffix = f"?{query}" if query else ""
    return RedirectResponse(url=f"/traces{suffix}", status_code=302)


@router.get("/traces", response_class=HTMLResponse)
async def traces_page(
    request: Request,
    agent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
    q: Optional[str] = None,
    root_trace_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    task_id: Optional[str] = None,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Show traces dashboard with searchable agent table and per-agent timeline."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agents: List[Any] = []
    try:
        agents = (
            []
            if current_principal.get().restricted
            else _state["provider"].list_memagents()
        )
    except Exception as exc:
        logger.warning("Failed to list trace agents (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503, detail="Trace agent associations are unavailable"
        ) from None

    conversation_docs, tool_log_docs, trace_query_metadata = _load_trace_snapshot()
    agents = _discover_runtime_trace_agents(
        agents,
        conversation_docs=conversation_docs,
        tool_log_docs=tool_log_docs,
    )
    indexed_summaries = None
    from ...observability.index import read_path

    if read_path() == "index":
        try:
            native = _state["provider"].get_observability_index()
            indexed_summaries = native.summaries(limit=1000, **scoped_trace_filters())
            known = {_extract_agent_identifier(agent) for agent in agents}
            for row in indexed_summaries:
                key = row.get("agent_id")
                if key and key not in known:
                    agents.append(
                        {
                            "agent_id": key,
                            "name": key,
                            "memory_ids": [],
                            "_is_virtual_trace_source": True,
                            "_runtime_agent_ids": [key],
                        }
                    )
                    known.add(key)
            trace_query_metadata[
                "overview_scope"
            ] = "indexed_agent_thread_groups_first_1000"
        except Exception:
            trace_query_metadata.setdefault("query_errors", []).append("summary")

    agent_query = _to_text(q).strip()
    trace_metrics = _build_agent_trace_metrics(
        agents, conversation_docs=conversation_docs, tool_log_docs=tool_log_docs
    )
    if indexed_summaries is not None:
        for agent in agents:
            key = _extract_agent_identifier(agent)
            groups = [
                row
                for row in indexed_summaries
                if row.get("agent_id") in _trace_agent_ids(agent)
            ]
            trace_metrics[key].update(
                event_count=sum(row["event_count"] for row in groups),
                bundle_count=sum(row["bundle_count"] for row in groups),
                last_ts=max(
                    (_coerce_timestamp(row["latest_timestamp"]) or 0 for row in groups),
                    default=0,
                ),
                coverage="partial" if len(indexed_summaries) == 1000 else "complete",
            )
    last_run_by_agent = {
        row_agent_id: float(metrics.get("last_ts") or 0.0)
        for row_agent_id, metrics in trace_metrics.items()
        if float(metrics.get("last_ts") or 0.0) > 0
    }
    if len(last_run_by_agent) < len(agents) and not current_principal.get().restricted:
        fallback_last_run = _load_agent_last_run_map(agents)
        for row_agent_id, timestamp in fallback_last_run.items():
            last_run_by_agent.setdefault(row_agent_id, timestamp)
    agents = _sort_agents_by_last_run_desc(agents, last_run_by_agent=last_run_by_agent)

    tool_counts = (
        {}
        if current_principal.get().restricted
        else _build_agent_tool_count_map(agents)
    )
    for agent in agents:
        if not isinstance(agent, dict) or not agent.get("_is_virtual_trace_source"):
            continue
        runtime_tool_names = agent.get("_runtime_tool_names") or []
        tool_counts[_extract_agent_identifier(agent)] = len(runtime_tool_names)
    thread_rows_by_agent = _build_agent_thread_rows(
        agents, conversation_docs=conversation_docs, tool_log_docs=tool_log_docs
    )
    if indexed_summaries is not None:
        for agent in agents:
            groups = [
                row
                for row in indexed_summaries
                if row.get("agent_id") in _trace_agent_ids(agent)
            ]
            thread_rows_by_agent[_extract_agent_identifier(agent)] = [
                {
                    "thread_id": row["thread_id"] or "",
                    "memory_id": "",
                    "event_count": row["event_count"],
                    "last_ts": _coerce_timestamp(row["latest_timestamp"]) or 0,
                    "last_activity": _format_trace_timestamp(row["latest_timestamp"]),
                }
                for row in groups
            ]
    agent_rows = _build_trace_agent_rows(
        agents=agents,
        tool_counts=tool_counts,
        trace_metrics=trace_metrics,
        thread_rows_by_agent=thread_rows_by_agent,
        search_query=agent_query,
    )

    selected_agent = None
    trace_events: List[Dict[str, Any]] = []
    trace_summary: Optional[Dict[str, Any]] = None
    trace_analysis: Optional[Dict[str, Any]] = None
    trace_analysis_export_url: Optional[str] = None
    trace_signals: Dict[str, List[Dict[str, Any]]] = {
        "feedback": [],
        "outcomes": [],
    }
    persisted_recommendations: Dict[str, Dict[str, Any]] = {}
    selected_thread_row: Optional[Dict[str, Any]] = None
    error = None

    selected_agent_id = _to_text(agent_id).strip() or None
    selected_thread_id = _to_text(thread_id).strip() or None
    selected_thread_memory_id = _to_text(thread_memory_id).strip() or None
    if selected_agent_id:
        try:
            selected_agent = next(
                (
                    candidate
                    for candidate in agents
                    if _extract_agent_identifier(candidate) == selected_agent_id
                ),
                None,
            )
            if selected_agent is None:
                selected_agent = (
                    None
                    if current_principal.get().restricted
                    else _state["provider"].retrieve_memagent(selected_agent_id)
                )
            if selected_agent is None:
                selected_agent = _targeted_runtime_agent(
                    selected_agent_id, selected_thread_id
                )
            if not selected_agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            selected_snapshot, trace_events = _selection_window(
                request=request,
                agent_id=selected_agent_id,
                selected_agent=selected_agent,
                thread_id=selected_thread_id,
                thread_memory_id=selected_thread_memory_id,
                root_trace_id=root_trace_id,
                turn_id=turn_id,
                task_id=task_id,
            )
            trace_query_metadata = selected_snapshot.query_metadata
            if event_id and not any(
                e.get("event_id") == event_id for e in trace_events
            ):
                raise HTTPException(
                    status_code=404,
                    detail="Selected event is not in the authorized loaded window; narrow the trace selection",
                )
            latest_timestamp = (
                trace_events[-1].get("timestamp") if trace_events else None
            )
            trace_summary = {
                **trace_events.coverage,
                "event_count": len(trace_events),
                "latest_timestamp": _format_trace_timestamp(latest_timestamp),
            }
            source_is_virtual = bool(
                isinstance(selected_agent, dict)
                and selected_agent.get("_is_virtual_trace_source")
            )
            agent_event_count = int(
                trace_metrics.get(selected_agent_id, {}).get("event_count") or 0
            )
            window_truncated = (
                bool(trace_query_metadata.get("truncated"))
                or bool(trace_events.coverage.get("truncated"))
                or bool(
                    not selected_thread_id
                    and not selected_thread_memory_id
                    and not (root_trace_id or turn_id or task_id)
                    and agent_event_count > len(trace_events)
                )
            )
            analysis_scope = (
                "thread" if selected_thread_id or selected_thread_memory_id else "agent"
            )
            try:
                if current_principal.get().restricted:
                    raise PermissionError(
                        "Legacy learning records are outside the scoped trace view"
                    )
                root_trace_ids = {
                    _to_text(event.get("root_trace_id")).strip()
                    for event in trace_events
                    if _to_text(event.get("root_trace_id")).strip()
                }
                observability_store = ObservabilityStore(_state["provider"])
                trace_signals = observability_store.list_signals(
                    root_trace_ids=root_trace_ids,
                    agent_id=(None if source_is_virtual else selected_agent_id),
                    verified_only=True,
                )
                persisted_recommendations = {
                    _to_text((row.get("insight") or {}).get("id")): row
                    for row in observability_store.list_recommendations(
                        agent_id=selected_agent_id
                    )
                    if _to_text((row.get("insight") or {}).get("id"))
                }
            except Exception as exc:
                logger.debug("Unable to load trace learning records: %s", exc)
            trace_analysis = analyze_trace_events(
                trace_events,
                agent_id=selected_agent_id,
                agent_name=_extract_agent_persona_name(selected_agent),
                scope=analysis_scope,
                source_is_virtual=source_is_virtual,
                window_truncated=window_truncated,
                signals=trace_signals,
            )
            analysis_params = {"agent_id": selected_agent_id}
            analysis_params.update(
                {
                    k: request.query_params[k]
                    for k in ("start_time", "end_time", "run_id")
                    if k in request.query_params
                }
            )
            analysis_params.update(
                {
                    k: v
                    for k, v in {
                        "root_trace_id": root_trace_id,
                        "turn_id": turn_id,
                        "task_id": task_id,
                        **scoped_trace_filters(),
                    }.items()
                    if v is not None
                }
            )
            if selected_thread_id:
                analysis_params["thread_id"] = selected_thread_id
            if selected_thread_memory_id:
                analysis_params["thread_memory_id"] = selected_thread_memory_id
            trace_analysis_export_url = (
                f"/traces/analysis.json?{urlencode(analysis_params)}"
            )
            for thread_row in thread_rows_by_agent.get(selected_agent_id, []):
                row_thread_id = _to_text(thread_row.get("thread_id")).strip()
                row_memory_id = _to_text(thread_row.get("memory_id")).strip()
                if selected_thread_memory_id:
                    if row_memory_id != selected_thread_memory_id:
                        continue
                elif selected_thread_id:
                    if (
                        row_thread_id != selected_thread_id
                        and row_memory_id != selected_thread_id
                    ):
                        continue
                else:
                    continue
                selected_thread_row = thread_row
                break
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Failed to load traces for agent %s: %s",
                selected_agent_id,
                exc,
            )
            error = str(exc)

    content_mode = trace_content_mode()
    # Payloads are fetched only after an explicit, permission-gated reveal.
    display_trace_events = redact_trace_events(trace_events, mode="metadata")
    for event in display_trace_events:
        event["anchor"] = event_anchor(event.get("event_id"))
        event.pop("content", None)
    if trace_analysis:
        for insight in trace_analysis["insights"]:
            insight["evidence_anchors"] = [
                event_anchor(value) for value in insight.get("evidence_event_ids", [])
            ]
    audit_trace_view(
        request,
        "trace_timeline" if selected_agent_id else "trace_dashboard",
        agent_id=selected_agent_id,
        thread_id=selected_thread_id,
        result_count=len(display_trace_events),
        content_mode=content_mode,
    )

    return templates.TemplateResponse(
        "observability.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents_nav": _build_agent_nav_items(
                active_agent_id=selected_agent_id,
                agents=agents,
                last_run_by_agent=last_run_by_agent,
            ),
            "active_agent_id": selected_agent_id,
            "active_page": "traces",
            "agent_query": agent_query,
            "agent_rows": agent_rows,
            "selected_agent": selected_agent,
            "selected_agent_name": (
                _extract_agent_persona_name(selected_agent)
                if selected_agent is not None
                else None
            ),
            "selected_agent_id": selected_agent_id,
            "selected_thread_id": selected_thread_id,
            "selected_root_trace_id": root_trace_id,
            "selected_turn_id": turn_id,
            "selected_task_id": task_id,
            "selected_event_id": event_id,
            "selection_urls": _selection_urls(
                request,
                agent_id=selected_agent_id,
                thread_id=selected_thread_id,
                thread_memory_id=selected_thread_memory_id,
                root_trace_id=root_trace_id,
                turn_id=turn_id,
                task_id=task_id,
                event_id=event_id,
            ),
            "trace_capabilities": trace_capabilities(request),
            "selected_thread_memory_id": selected_thread_memory_id,
            "selected_thread_row": selected_thread_row,
            "trace_events": display_trace_events,
            "trace_waterfall": build_causal_waterfall(display_trace_events),
            "trace_inspectors": build_lineage_inspectors(display_trace_events),
            "trace_permissions": {
                permission: current_principal.get().allows(permission)
                for permission in (
                    "trace.reveal",
                    "artifact.lookup",
                    "account.resolve",
                    "replay.create",
                )
            },
            "inspection_scope": scoped_trace_filters(),
            "trace_summary": trace_summary,
            "trace_analysis": trace_analysis,
            "trace_analysis_export_url": trace_analysis_export_url,
            "trace_signals": trace_signals,
            "persisted_recommendations": persisted_recommendations,
            "trace_query_metadata": trace_query_metadata,
            "trace_content_mode": content_mode,
            "ui_read_only": ui_read_only(),
            "error": error,
        },
    )


def _inspection_context(request, **values):
    return {
        "request": request,
        "provider_type": _state.get("provider_type"),
        "connection_info": _state.get("connection_info"),
        "agents_nav": [],
        "active_page": "traces",
        **values,
    }


@router.get("/traces/health", response_class=HTMLResponse)
@router.get("/traces/health.json", response_class=JSONResponse)
async def trace_health_page(
    request: Request,
    agent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = 500,
    thread_memory_id: Optional[str] = None,
    root_trace_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    task_id: Optional[str] = None,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Read-only capture/query diagnostics, explicitly limited to loaded rows."""
    if not _state.get("provider"):
        raise HTTPException(status_code=503, detail="No memory provider connected")
    snapshot, events = _selection_window(
        request=request,
        agent_id=agent_id,
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
        root_trace_id=root_trace_id,
        turn_id=turn_id,
        task_id=task_id,
        limit=limit,
    )
    health = build_trace_health(events)
    health["queries"] = {
        store: {
            key: snapshot.query_metadata.get(store, {}).get(key)
            for key in (
                "provider_native",
                "scanned_count",
                "query_duration_ms",
                "truncated",
            )
        }
        for store in ("conversation", "tool_log", "trace")
    }
    health["failed_queries"] = snapshot.query_metadata.get("query_errors", [])
    from ...observability.pipeline import pipeline_health

    health["pipeline"] = (
        pipeline_health(_state["provider"])
        if not current_principal.get().restricted
        else {"reported": False, "scope": "not_available_for_scoped_accounts"}
    )
    capabilities = getattr(_state["provider"], "observability_capabilities", None)
    try:
        health["index"] = (
            capabilities()
            if callable(capabilities)
            else {"ready": False, "span_index": False}
        )
    except Exception:
        health["index"] = {"ready": False, "state": "unavailable"}
    health["instrumentation_versions"] = trace_coverage(events, events.coverage).get(
        "instrumentation_versions", []
    )
    health["registration"] = {"state": "unknown"}
    if not current_principal.get().restricted:
        try:
            registered = {
                _extract_agent_identifier(row): set(_extract_agent_memory_ids(row))
                for row in _state["provider"].list_memagents()
            }
            health["registration"] = {
                "state": "observed",
                "unregistered_agents": len(
                    {
                        e["agent_id"]
                        for e in events
                        if e.get("agent_id") and e["agent_id"] not in registered
                    }
                ),
                "memory_id_mismatch_events": sum(
                    bool(
                        e.get("agent_id") in registered
                        and registered[e["agent_id"]]
                        and e.get("memory_id")
                        and e["memory_id"] not in registered[e["agent_id"]]
                    )
                    for e in events
                ),
            }
        except Exception:
            pass
    from ...observability.inspection import build_observability_alerts

    health["alerts"] = build_observability_alerts(health)
    audit_trace_view(
        request,
        "trace_health",
        agent_id=agent_id,
        thread_id=thread_id,
        result_count=len(events),
        content_mode="metadata",
    )
    if request.url.path.endswith(".json"):
        return JSONResponse(_json_safe(health))
    return templates.TemplateResponse(
        "observability_health.html",
        _inspection_context(
            request,
            health=health,
            agent_id=agent_id,
            thread_id=thread_id,
            selection_urls=_selection_urls(
                request,
                agent_id=agent_id,
                thread_id=thread_id,
                thread_memory_id=thread_memory_id,
                root_trace_id=root_trace_id,
                turn_id=turn_id,
                task_id=task_id,
                event_id=event_id,
            ),
        ),
    )


@router.get("/traces/compare", response_class=HTMLResponse)
@router.get("/traces/compare.json", response_class=JSONResponse)
async def trace_compare_page(
    request: Request,
    agent_id: Optional[str] = None,
    baseline_root: Optional[str] = None,
    candidate_root: Optional[str] = None,
    baseline_turn: Optional[str] = None,
    candidate_turn: Optional[str] = None,
    thread_id: Optional[str] = None,
    limit: int = 500,
    thread_memory_id: Optional[str] = None,
    root_trace_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    task_id: Optional[str] = None,
    event_id: Optional[str] = None,
    run_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Compare captured structures only; never replay application side effects."""
    if not _state.get("provider"):
        raise HTTPException(status_code=503, detail="No memory provider connected")
    comparison = None
    if agent_id and baseline_root and candidate_root:
        snapshot, events = _selection_window(
            request=request,
            agent_id=agent_id,
            thread_id=thread_id,
            thread_memory_id=thread_memory_id,
            limit=limit,
            event_limit=0,
        )

        def select(root, turn):
            rows = select_trace_events(
                events, root_trace_id=root, turn_id=turn, task_id=task_id, limit=1000
            )
            if not rows:
                raise HTTPException(
                    status_code=404,
                    detail="Requested trace not found in the bounded loaded window; narrow the thread or increase the row limit",
                )
            return TraceEvents(
                redact_trace_events(rows, mode="metadata"), coverage=rows.coverage
            )

        comparison = compare_trace_windows(
            select(baseline_root, baseline_turn), select(candidate_root, candidate_turn)
        )
        audit_trace_view(
            request,
            "trace_compare",
            agent_id=agent_id,
            thread_id=thread_id,
            result_count=comparison["baseline"]["normalized_events"]
            + comparison["candidate"]["normalized_events"],
            content_mode="metadata",
        )
    elif request.url.path.endswith(".json"):
        raise HTTPException(
            status_code=400,
            detail="agent_id, baseline_root and candidate_root are required",
        )
    if request.url.path.endswith(".json"):
        return JSONResponse(_json_safe(comparison))
    return templates.TemplateResponse(
        "observability_compare.html",
        _inspection_context(
            request,
            comparison=comparison,
            agent_id=agent_id,
            thread_id=thread_id,
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            baseline_turn=baseline_turn,
            candidate_turn=candidate_turn,
            selection_urls=_selection_urls(
                request,
                agent_id=agent_id,
                thread_id=thread_id,
                thread_memory_id=thread_memory_id,
                root_trace_id=root_trace_id or baseline_root,
                turn_id=turn_id or baseline_turn,
                task_id=task_id,
                event_id=event_id,
            ),
        ),
    )


@router.get("/traces/analysis.json", response_class=JSONResponse)
async def trace_analysis_export(
    request: Request,
    agent_id: str,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
    root_trace_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """Export a deterministic, read-only improvement report as JSON."""
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(status_code=503, detail="No memory provider connected")

    try:
        agents = (
            []
            if current_principal.get().restricted
            else provider.list_memagents() or []
        )
    except Exception as exc:
        logger.error("Failed to list agents for trace analysis export: %s", exc)
        raise HTTPException(status_code=503, detail="Unable to load agents") from exc

    conversation_docs, tool_log_docs, _overview_metadata = _load_trace_snapshot()
    agents = _discover_runtime_trace_agents(
        agents,
        conversation_docs=conversation_docs,
        tool_log_docs=tool_log_docs,
    )
    selected_agent = next(
        (
            candidate
            for candidate in agents
            if _extract_agent_identifier(candidate) == agent_id
        ),
        None,
    )
    if selected_agent is None and agent_id != _RUNTIME_TRACE_AGENT_ID:
        try:
            selected_agent = (
                None
                if current_principal.get().restricted
                else provider.retrieve_memagent(agent_id)
            )
        except Exception as exc:
            logger.debug("Failed to retrieve trace export agent %s: %s", agent_id, exc)
    if selected_agent is None:
        selected_agent = _targeted_runtime_agent(agent_id, thread_id)
    if not selected_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    selected_thread_id = _to_text(thread_id).strip() or None
    selected_thread_memory_id = _to_text(thread_memory_id).strip() or None
    _snapshot, events = _selection_window(
        request=request,
        agent_id=agent_id,
        selected_agent=selected_agent,
        thread_id=selected_thread_id,
        thread_memory_id=selected_thread_memory_id,
        root_trace_id=root_trace_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    source_is_virtual = bool(
        isinstance(selected_agent, dict)
        and selected_agent.get("_is_virtual_trace_source")
    )
    scope = "thread" if selected_thread_id or selected_thread_memory_id else "agent"
    try:
        if current_principal.get().restricted:
            raise PermissionError(
                "Legacy learning records are outside the scoped trace view"
            )
        signal_store = ObservabilityStore(provider)
        root_trace_ids = {
            _to_text(event.get("root_trace_id")).strip()
            for event in events
            if _to_text(event.get("root_trace_id")).strip()
        }
        signals = signal_store.list_signals(
            root_trace_ids=root_trace_ids,
            agent_id=None if source_is_virtual else agent_id,
            verified_only=True,
        )
    except Exception as exc:
        logger.debug("Unable to load signals for trace export: %s", exc)
        signals = {"feedback": [], "outcomes": []}
    report = analyze_trace_events(
        events,
        agent_id=agent_id,
        agent_name=_extract_agent_persona_name(selected_agent),
        scope=scope,
        source_is_virtual=source_is_virtual,
        window_truncated=bool(events.coverage.get("truncated")),
        signals=signals,
    )
    audit_trace_view(
        request,
        "trace_analysis_export",
        agent_id=agent_id,
        thread_id=selected_thread_id,
        result_count=len(events),
    )
    return JSONResponse(
        report,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/traces/events.json", response_class=JSONResponse)
async def trace_events_page(
    request: Request,
    store: str = "trace",
    view: str = "events",
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
    task_id: Optional[str] = None,
    user_id: Optional[str] = None,
    root_trace_id: Optional[str] = None,
    run_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    resource_ref: Optional[str] = None,
    event_kind: Optional[str] = None,
    status: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    tool_name: Optional[str] = None,
    success: Optional[bool] = None,
    limit: int = 250,
    cursor: Optional[str] = None,
):
    """Return one cursor page for trace explorers and external local tooling."""
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(status_code=503, detail="No memory provider connected")
    from ...enums.memory_type import MemoryType

    if view not in {"events", "bundles"}:
        raise HTTPException(status_code=400, detail="view must be events or bundles")

    if store == "all":
        if (
            view != "events"
            or cursor
            or any(
                (
                    memory_id,
                    resource_ref,
                    event_kind,
                    status,
                    tool_name,
                    success is not None,
                )
            )
        ):
            raise HTTPException(
                status_code=400,
                detail="Combined export is a bounded snapshot; use per-store pagination for additional event filters",
            )
        scoped_trace_filters({"user_id": user_id} if user_id is not None else {})
        _snapshot, events = _selection_window(
            request=request,
            agent_id=agent_id,
            thread_id=thread_id,
            thread_memory_id=thread_memory_id,
            task_id=task_id,
            root_trace_id=root_trace_id,
            turn_id=turn_id,
            limit=500,
        )
        audit_trace_view(
            request,
            "trace_combined_export",
            result_count=len(events),
            content_mode="metadata",
        )
        return JSONResponse(
            _json_safe(
                {
                    **events.coverage,
                    "evidence_coverage": trace_coverage(events, events.coverage),
                    "items": redact_trace_events(events, mode="metadata"),
                    "content_mode": "metadata",
                    "view": "events",
                    "store_selection": ["conversation", "tool_log", "trace"],
                }
            )
        )

    if thread_memory_id is not None or task_id is not None:
        raise HTTPException(
            status_code=400,
            detail="Task and thread-memory selection require store=all",
        )
    memory_type_by_name = {
        "conversation": MemoryType.CONVERSATION_MEMORY,
        "conversation_memory": MemoryType.CONVERSATION_MEMORY,
        "tool": MemoryType.TOOL_LOG,
        "tool_log": MemoryType.TOOL_LOG,
        "trace": MemoryType.SHARED_MEMORY,
        "trace_bundle": MemoryType.SHARED_MEMORY,
    }
    memory_type = memory_type_by_name.get(store.strip().lower())
    if memory_type is None:
        raise HTTPException(
            status_code=400,
            detail="store must be conversation, tool_log, or trace",
        )
    query = getattr(provider, "query_observability_records", None)
    if not callable(query):
        raise HTTPException(
            status_code=501,
            detail="The connected provider does not support paginated trace queries",
        )
    kwargs: Dict[str, Any] = {
        "agent_ids": [agent_id] if agent_id else None,
        "memory_ids": [memory_id] if memory_id else None,
        "thread_id": thread_id,
        "tool_name": tool_name,
        "success": success,
        "limit": limit,
        "cursor": cursor,
    }
    if memory_type == MemoryType.SHARED_MEMORY:
        kwargs["record_type"] = _TRACE_BUNDLE_RECORD_TYPE
    if user_id is not None:
        kwargs["user_id"] = user_id
    kwargs = scoped_trace_filters(kwargs)
    event_filters = {
        "root_trace_id": root_trace_id,
        "run_id": run_id,
        "turn_id": turn_id,
        "resource_refs": [resource_ref] if resource_ref else None,
        "event_kinds": [event_kind] if event_kind else None,
        "statuses": [status] if status else None,
        "start_time": start_time,
        "end_time": end_time,
    }
    if view == "bundles" and any(value is not None for value in event_filters.values()):
        raise HTTPException(status_code=400, detail="Event filters require view=events")
    if view == "events":
        kwargs.update(
            {key: value for key, value in event_filters.items() if value is not None}
        )
    try:
        page = (
            (
                provider.query_trace_events(**kwargs)
                if memory_type == MemoryType.SHARED_MEMORY
                and callable(getattr(provider, "query_trace_events", None))
                else query_trace_events(provider, memory_type, **kwargs)
            )
            if view == "events"
            else query(memory_type, **kwargs)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Trace event query failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Trace query failed; evidence coverage is unavailable",
        ) from exc

    mode = trace_content_mode()
    if mode != "metadata" and not audit_trace_view(
        request,
        "trace_export_requested",
        agent_id=agent_id,
        thread_id=thread_id,
        content_mode=mode,
    ):
        raise HTTPException(
            status_code=503, detail="Audited trace access is unavailable"
        )
    safe_items: List[Dict[str, Any]] = []
    export_items = redact_trace_events(page.get("items") or [], mode=mode)
    for raw_item in export_items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        safe_items.append(_json_safe(item))
    response_payload = {
        key: _json_safe(value) for key, value in page.items() if key != "items"
    }
    response_payload["items"] = safe_items
    response_payload["content_mode"] = mode
    response_payload["view"] = view
    response_payload["store_selection"] = [store]
    if view == "events":
        response_payload["evidence_coverage"] = trace_coverage(
            page.get("items", []), page
        )
    audit_trace_view(
        request,
        "trace_events_page",
        agent_id=agent_id,
        thread_id=thread_id,
        result_count=len(safe_items),
        content_mode=mode,
    )
    return JSONResponse(
        response_payload,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _analysis_for_selection(
    agent_id: str,
    *,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
) -> tuple[Dict[str, Any], List[Dict[str, Any]], Any]:
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(status_code=503, detail="No memory provider connected")
    try:
        agents = provider.list_memagents() or []
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Unable to load agents") from exc
    overview_conversations, overview_tools, _metadata = _load_trace_snapshot()
    agents = _discover_runtime_trace_agents(
        agents,
        conversation_docs=overview_conversations,
        tool_log_docs=overview_tools,
    )
    selected_agent = next(
        (
            candidate
            for candidate in agents
            if _extract_agent_identifier(candidate) == agent_id
        ),
        None,
    )
    if selected_agent is None and agent_id != _RUNTIME_TRACE_AGENT_ID:
        selected_agent = provider.retrieve_memagent(agent_id)
    if not selected_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    conversations, tools, query_metadata = _load_trace_snapshot(
        agent_ids=sorted(_trace_agent_ids(selected_agent)),
        memory_ids=_extract_agent_memory_ids(selected_agent),
        thread_id=thread_id,
        limit=500,
    )
    events = _load_agent_trace_events(
        selected_agent,
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
        conversation_docs=conversations,
        tool_log_docs=tools,
        query_metadata=query_metadata,
    )
    source_is_virtual = bool(
        isinstance(selected_agent, dict)
        and selected_agent.get("_is_virtual_trace_source")
    )
    root_trace_ids = {
        _to_text(event.get("root_trace_id")).strip()
        for event in events
        if _to_text(event.get("root_trace_id")).strip()
    }
    try:
        signals = ObservabilityStore(provider).list_signals(
            root_trace_ids=root_trace_ids,
            agent_id=None if source_is_virtual else agent_id,
            verified_only=True,
        )
    except Exception:
        signals = {"feedback": [], "outcomes": []}
    report = analyze_trace_events(
        events,
        agent_id=agent_id,
        agent_name=_extract_agent_persona_name(selected_agent),
        scope="thread" if thread_id or thread_memory_id else "agent",
        source_is_virtual=source_is_virtual,
        window_truncated=bool(events.coverage.get("truncated")),
        signals=signals,
    )
    return report, events, selected_agent


def _baseline_agent_config(agent: Any) -> Dict[str, Any]:
    if isinstance(agent, dict):
        raw = dict(agent)
    elif hasattr(agent, "model_dump"):
        raw = agent.model_dump()
    elif hasattr(agent, "dict"):
        raw = agent.dict()
    else:
        raw = dict(getattr(agent, "__dict__", {}) or {})
    llm_config = raw.get("llm_config") or {}
    if not isinstance(llm_config, dict):
        llm_config = {}
    return {
        "agent_id": _extract_agent_identifier(agent),
        "application_mode": raw.get("application_mode"),
        "instruction": raw.get("instruction"),
        "tools": raw.get("tools") or [],
        "tool_result_policy": raw.get("tool_result_policy"),
        "context_policy": raw.get("context_policy"),
        "retrieval_policy": raw.get("retrieval_policy"),
        "llm": {
            key: llm_config.get(key)
            for key in ("provider", "model", "temperature", "max_tokens")
            if llm_config.get(key) is not None
        },
    }


@router.post("/traces/recommendations")
async def save_trace_recommendations(request: Request):
    """Persist current findings for explicit operator review."""
    form = await request.form()
    agent_id = _to_text(form.get("agent_id")).strip()
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    thread_id = _to_text(form.get("thread_id")).strip() or None
    thread_memory_id = _to_text(form.get("thread_memory_id")).strip() or None
    report, events, _selected_agent = _analysis_for_selection(
        agent_id,
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
    )
    evidence_refs = [
        {
            key: event.get(key)
            for key in ("root_trace_id", "run_id", "turn_id", "thread_id")
            if event.get(key)
        }
        for event in events
    ]
    try:
        stored = ObservabilityStore(_state["provider"]).sync_recommendations(
            report,
            evidence_refs=evidence_refs,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    audit_trace_view(
        request,
        "trace_recommendations_saved",
        agent_id=agent_id,
        thread_id=thread_id,
        result_count=len(stored),
    )
    params: Dict[str, str] = {"agent_id": agent_id}
    if thread_id:
        params["thread_id"] = thread_id
    if thread_memory_id:
        params["thread_memory_id"] = thread_memory_id
    return RedirectResponse(
        url=f"/traces?{urlencode(params)}#trace-insights", status_code=303
    )


@router.post("/traces/recommendations/{recommendation_id}/review")
async def review_trace_recommendation(recommendation_id: str, request: Request):
    """Persist a review decision and draft an experiment on acceptance."""
    form = await request.form()
    decision = _to_text(form.get("decision")).strip().lower()
    reviewer_id = (
        _to_text(form.get("reviewer_id")).strip()
        or _to_text(os.environ.get("MEMORIZZ_UI_OPERATOR_ID")).strip()
        or "local-operator"
    )
    note = _to_text(form.get("note")).strip() or None
    store = ObservabilityStore(_state.get("provider"))
    recommendation = next(
        (
            row
            for row in store.list_recommendations()
            if row.get("recommendation_id") == recommendation_id
        ),
        None,
    )
    if not recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    agent_id = _to_text(recommendation.get("agent_id")).strip()
    selected_agent = None
    if agent_id:
        try:
            selected_agent = _state["provider"].retrieve_memagent(agent_id)
        except Exception:
            selected_agent = None
    try:
        result = store.review_recommendation(
            recommendation_id,
            decision=decision,
            reviewer_id=reviewer_id,
            note=note,
            baseline_config=(
                _baseline_agent_config(selected_agent) if selected_agent else {}
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_trace_view(
        request,
        "trace_recommendation_reviewed",
        agent_id=agent_id,
        result_count=1 if result else 0,
    )
    return RedirectResponse(
        url=f"/traces?{urlencode({'agent_id': agent_id})}#trace-insights",
        status_code=303,
    )


@router.post("/traces/feedback")
async def record_trace_feedback(request: Request):
    form = await request.form()
    context = {
        key: _to_text(form.get(key)).strip()
        for key in (
            "application_id",
            "agent_id",
            "run_id",
            "turn_id",
            "root_trace_id",
            "memory_id",
            "thread_id",
            "user_id",
        )
        if _to_text(form.get(key)).strip()
    }
    try:
        row = ObservabilityStore(_state.get("provider")).record_feedback(
            trace_context=context,
            rating=float(form.get("rating") or 0),
            verified=True,
            source="operator_ui",
            label=_to_text(form.get("label")).strip() or None,
            comment=_to_text(form.get("comment")).strip() or None,
            include_comment=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "feedback_id": row.get("record_id")})


@router.post("/traces/outcomes")
async def record_trace_outcome(request: Request):
    form = await request.form()
    context = {
        key: _to_text(form.get(key)).strip()
        for key in (
            "application_id",
            "agent_id",
            "run_id",
            "turn_id",
            "root_trace_id",
            "memory_id",
            "thread_id",
            "user_id",
        )
        if _to_text(form.get(key)).strip()
    }
    try:
        row = ObservabilityStore(_state.get("provider")).record_outcome(
            trace_context=context,
            status=_to_text(form.get("status")).strip(),
            verified=True,
            source="operator_ui",
            score=(
                float(form.get("score"))
                if form.get("score") not in (None, "")
                else None
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse({"ok": True, "outcome_id": row.get("record_id")})


def _load_trace_snapshot(
    *,
    agent_ids: Optional[List[str]] = None,
    memory_ids: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    limit: int = 750,
) -> TraceSnapshot:
    """Load bounded trace pages and surface their query/freshness metadata."""
    provider = _state.get("provider")
    if not provider:
        return TraceSnapshot(
            query_metadata={"truncated": False, "provider_native": False}
        )

    from ...enums.memory_type import MemoryType

    def _page(memory_type: MemoryType) -> Dict[str, Any]:
        from ...observability.index import indexed_bundle_rows, read_path

        if memory_type == MemoryType.SHARED_MEMORY and read_path() == "index":
            page = provider.query_trace_events(
                agent_ids=agent_ids,
                memory_ids=memory_ids,
                thread_id=thread_id,
                limit=limit,
                **scoped_trace_filters(),
            )
            if page.get("normalization_errors"):
                raise RuntimeError("Indexed trace coverage is incomplete")
            return {**page, "items": indexed_bundle_rows(page)}
        query = getattr(provider, "query_observability_records", None)
        if callable(query):
            kwargs: Dict[str, Any] = {
                "agent_ids": agent_ids,
                "memory_ids": memory_ids,
                "thread_id": thread_id,
                "limit": limit,
            }
            if memory_type == MemoryType.SHARED_MEMORY:
                kwargs["record_type"] = _TRACE_BUNDLE_RECORD_TYPE
            kwargs = scoped_trace_filters(kwargs)
            return query(
                memory_type,
                **kwargs,
            )
        # Compatibility for provider-like test doubles predating the contract.
        rows = provider.list_all(memory_type) or []
        matching: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            from ...observability.normalization import read_payload, row_identity

            identity = row_identity(row, read_payload(row))
            if any(
                identity.get(key) != value
                for key, value in scoped_trace_filters().items()
            ):
                continue
            if memory_type == MemoryType.SHARED_MEMORY:
                try:
                    payload = json.loads(_to_text(row.get("content")))
                except (TypeError, ValueError):
                    payload = None
                row_record_type = row.get("record_type") or (
                    payload.get("record_type") if isinstance(payload, dict) else None
                )
                if row_record_type != _TRACE_BUNDLE_RECORD_TYPE:
                    continue
            matching.append(row)
        bounded = matching[:limit]
        return {
            "items": bounded,
            "next_cursor": None,
            "truncated": len(matching) > len(bounded),
            "limit": limit,
            "scanned_count": len(rows),
            "query_duration_ms": None,
            "freshness": None,
            "provider_native": False,
        }

    query_errors = []
    try:
        conversation_page = _page(MemoryType.CONVERSATION_MEMORY)
    except Exception as exc:
        logger.warning(
            "Failed to query conversation trace documents (%s)", type(exc).__name__
        )
        conversation_page = {"items": [], "truncated": False}
        query_errors.append("conversation")
    try:
        tool_page = _page(MemoryType.TOOL_LOG)
    except Exception as exc:
        logger.warning("Failed to query tool trace documents (%s)", type(exc).__name__)
        tool_page = {"items": [], "truncated": False}
        query_errors.append("tool_log")
    try:
        trace_page = _page(MemoryType.SHARED_MEMORY)
    except Exception as exc:
        logger.warning("Failed to query private trace bundles (%s)", type(exc).__name__)
        trace_page = {"items": [], "truncated": False}
        query_errors.append("trace")

    conversations = conversation_page.get("items") or []
    tool_logs = tool_page.get("items") or []
    trace_bundles: List[Dict[str, Any]] = []
    for row in trace_page.get("items") or []:
        if not isinstance(row, dict):
            continue
        normalized = dict(row)
        trace_memory_id = _to_text(normalized.get("trace_memory_id")).strip()
        if trace_memory_id:
            normalized["memory_id"] = trace_memory_id
        normalized.setdefault("role", "tool")
        trace_bundles.append(normalized)
    metadata = {
        "query_errors": query_errors,
        "truncated": bool(conversation_page.get("truncated"))
        or bool(tool_page.get("truncated"))
        or bool(trace_page.get("truncated")),
        "provider_native": bool(conversation_page.get("provider_native"))
        and bool(tool_page.get("provider_native"))
        and bool(trace_page.get("provider_native")),
        "conversation": {
            key: conversation_page.get(key)
            for key in (
                "next_cursor",
                "truncated",
                "limit",
                "scanned_count",
                "query_duration_ms",
                "freshness",
                "provider_native",
            )
        },
        "tool_log": {
            key: tool_page.get(key)
            for key in (
                "next_cursor",
                "truncated",
                "limit",
                "scanned_count",
                "query_duration_ms",
                "freshness",
                "provider_native",
            )
        },
        "trace": {
            key: trace_page.get(key)
            for key in (
                "next_cursor",
                "truncated",
                "limit",
                "scanned_count",
                "query_duration_ms",
                "freshness",
                "provider_native",
            )
        },
    }

    return TraceSnapshot(
        conversation_rows=[row for row in conversations if isinstance(row, dict)],
        tool_rows=[row for row in tool_logs if isinstance(row, dict)],
        bundle_rows=trace_bundles,
        query_metadata=metadata,
    )


def _discover_runtime_trace_agents(
    agents: List[Any],
    *,
    conversation_docs: List[Dict[str, Any]],
    tool_log_docs: List[Dict[str, Any]],
) -> List[Any]:
    """Expose durable traces even when an application never saved its agent.

    Older integrations often constructed a runtime ``MemAgent`` but omitted
    ``save()``. Their conversations and tool logs are still valuable. Group
    those unclaimed records into one virtual trace source instead of making
    the observability page look empty or fragmenting it by ephemeral process
    IDs.
    """
    if current_principal.get().restricted:
        from ...observability.normalization import read_payload, row_identity

        scoped_agents = {}
        for row in [*conversation_docs, *tool_log_docs]:
            identity = row_identity(row, read_payload(row))
            if not identity.get("agent_id") or any(
                identity.get(key) != value
                for key, value in scoped_trace_filters().items()
            ):
                continue
            key = identity["agent_id"]
            agent = scoped_agents.setdefault(
                key,
                {
                    "agent_id": key,
                    "name": key,
                    "memory_ids": [],
                    "_is_virtual_trace_source": True,
                    "_runtime_agent_ids": [key],
                },
            )
            if (
                identity.get("memory_id")
                and identity["memory_id"] not in agent["memory_ids"]
            ):
                agent["memory_ids"].append(identity["memory_id"])
        return list(scoped_agents.values())
    result = list(agents or [])
    saved_agent_ids = {
        agent_id
        for agent_id in (_extract_agent_identifier(agent) for agent in result)
        if agent_id
    }
    claimed_memory_ids = {
        memory_id for agent in result for memory_id in _extract_agent_memory_ids(agent)
    }

    runtime_agent_ids = set()
    runtime_memory_ids = set()
    runtime_tool_names = set()
    for doc in [*conversation_docs, *tool_log_docs]:
        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        memory_id = _extract_trace_memory_id(doc, fallback="")
        if memory_id == "—":
            memory_id = ""

        is_unregistered_agent = bool(
            doc_agent_id and doc_agent_id not in saved_agent_ids
        )
        is_unclaimed_memory = bool(memory_id and memory_id not in claimed_memory_ids)
        if is_unregistered_agent:
            runtime_agent_ids.add(doc_agent_id)
        if is_unregistered_agent or (not doc_agent_id and is_unclaimed_memory):
            if memory_id:
                runtime_memory_ids.add(memory_id)
            tool_name = _to_text(doc.get("tool_name")).strip()
            if tool_name:
                runtime_tool_names.add(tool_name)

    if runtime_agent_ids or runtime_memory_ids:
        result.append(
            {
                "agent_id": _RUNTIME_TRACE_AGENT_ID,
                "name": "Unregistered runtime traces",
                "application_mode": "runtime",
                "memory_ids": sorted(runtime_memory_ids),
                "_runtime_agent_ids": sorted(runtime_agent_ids),
                "_runtime_tool_names": sorted(runtime_tool_names),
                "_is_virtual_trace_source": True,
            }
        )
    return result


def _trace_agent_ids(agent: Any) -> set[str]:
    """Return persisted or virtual runtime IDs represented by an agent row."""
    agent_id = _extract_agent_identifier(agent)
    if isinstance(agent, dict) and agent.get("_is_virtual_trace_source"):
        return {
            _to_text(value).strip()
            for value in (agent.get("_runtime_agent_ids") or [])
            if _to_text(value).strip()
        }
    return {agent_id} if agent_id else set()


def _format_trace_timestamp(value: Any) -> str:
    """Format mixed timestamp values into a compact display string."""
    timestamp = _coerce_timestamp(value)
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    text = _to_text(value).strip()
    return text or "—"


def _extract_trace_memory_id(payload: Dict[str, Any], fallback: str = "") -> str:
    """Extract memory ID from trace payloads."""
    if not isinstance(payload, dict):
        return _to_text(fallback).strip()
    memory_id = _to_text(
        payload.get("memory_id") or payload.get("memoryId") or fallback
    ).strip()
    return memory_id or "—"


def _extract_trace_thread_id(payload: Dict[str, Any], fallback: str = "") -> str:
    """Extract thread/conversation ID from trace payloads."""
    if not isinstance(payload, dict):
        return _to_text(fallback).strip() or "—"

    for key in ("thread_id", "threadId", "conversation_id", "conversationId"):
        value = _to_text(payload.get(key)).strip()
        if value:
            return value

    memory_fallback = _extract_trace_memory_id(payload, fallback=fallback)
    return memory_fallback or "—"


def _thread_row_key(thread_id: str, memory_id: str) -> str:
    """Build a stable key for thread rows."""
    return f"{memory_id}:{thread_id or memory_id}"


def _build_agent_thread_rows(
    agents: List[Any],
    *,
    conversation_docs: Optional[List[Dict[str, Any]]] = None,
    tool_log_docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Derive thread counts from the same normalized window as the timeline."""
    result = {}
    for agent in agents:
        events = _load_agent_trace_events(
            agent,
            total_limit=0,
            conversation_docs=conversation_docs,
            tool_log_docs=tool_log_docs or [],
        )
        rows = {}
        for event in events:
            memory_id = event.get("memory_id") or "—"
            thread_id = event.get("thread_id") or memory_id
            key = _thread_row_key(thread_id, memory_id)
            row = rows.setdefault(
                key,
                {
                    "thread_id": thread_id,
                    "memory_id": memory_id,
                    "event_count": 0,
                    "last_ts": 0.0,
                    "last_activity": "—",
                },
            )
            row["event_count"] += 1
            ts = _coerce_timestamp(event.get("timestamp")) or 0.0
            if ts > row["last_ts"]:
                row.update(last_ts=ts, last_activity=_format_trace_timestamp(ts))
        result[_extract_agent_identifier(agent)] = sorted(
            rows.values(),
            key=lambda row: row["last_ts"],
            reverse=True,
        )
    return result


def _build_agent_trace_metrics(
    agents: List[Any],
    *,
    conversation_docs: Optional[List[Dict[str, Any]]] = None,
    tool_log_docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Count child events separately from their storage envelopes."""
    metrics = {}
    for agent in agents:
        events = _load_agent_trace_events(
            agent,
            total_limit=0,
            conversation_docs=conversation_docs,
            tool_log_docs=tool_log_docs or [],
        )
        metrics[_extract_agent_identifier(agent)] = {
            "event_count": len(events),
            "bundle_count": events.coverage.get("bundle_count", 0),
            "coverage": events.coverage.get("coverage", "unknown"),
            "last_ts": max(
                (_coerce_timestamp(e.get("timestamp")) or 0.0 for e in events),
                default=0.0,
            ),
            "search_terms": " ".join(
                str(value)
                for event in events
                for value in (
                    event.get("thread_id", ""),
                    event.get("root_trace_id", ""),
                    event.get("run_id", ""),
                    event.get("error_code", ""),
                    *(
                        ref.get("ref", "")
                        for ref in [
                            *(event.get("input_refs") or []),
                            *(event.get("output_refs") or []),
                        ]
                    ),
                )
            ),
        }
    return metrics


def _build_trace_agent_rows(
    agents: List[Any],
    tool_counts: Dict[str, int],
    trace_metrics: Dict[str, Dict[str, Any]],
    thread_rows_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    search_query: str = "",
) -> List[Dict[str, Any]]:
    """Create filtered rows for the traces agent table."""
    rows: List[Dict[str, Any]] = []
    query = _to_text(search_query).strip().lower()

    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        metrics = trace_metrics.get(agent_id, {})
        last_ts = float(metrics.get("last_ts") or 0.0)
        row = {
            "agent_id": agent_id,
            "name": _extract_agent_persona_name(agent),
            "mode": _to_text(
                getattr(agent, "application_mode", None)
                if not isinstance(agent, dict)
                else agent.get("application_mode")
            ).strip()
            or "assistant",
            "memory_count": len(_extract_agent_memory_ids(agent)),
            "tool_count": int(tool_counts.get(agent_id, 0)),
            "event_count": int(metrics.get("event_count") or 0),
            "bundle_count": int(metrics.get("bundle_count") or 0),
            "coverage": metrics.get("coverage", "unknown"),
            "last_ts": last_ts,
            "last_activity": _format_trace_timestamp(last_ts) if last_ts > 0 else "—",
            "thread_rows": (
                thread_rows_by_agent.get(agent_id, []) if thread_rows_by_agent else []
            ),
        }

        if query:
            searchable = " ".join(
                [
                    row["name"],
                    row["agent_id"],
                    row["mode"],
                    metrics.get("search_terms", ""),
                ]
            ).lower()
            if query not in searchable:
                continue

        rows.append(row)

    rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
    return rows


def _expand_trace_bundle(
    message: Dict[str, Any],
    *,
    memory_id: str,
    thread_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Compatibility shim for integrations importing the old router helper."""
    if not is_bundle(message):
        return None
    return normalize_trace_snapshot(
        TraceSnapshot(
            bundle_rows=[
                {**message, "trace_memory_id": memory_id, "thread_id": thread_id},
            ]
        )
    ).events


def _load_agent_tool_log_events(
    agent: Any,
    memory_ids: List[str],
    *,
    thread_id: str = "",
    thread_memory_id: str = "",
    documents: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    provider = _state.get("provider")
    if documents is None:
        from ...enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.TOOL_LOG) if provider else []
    return normalize_trace_snapshot(
        TraceSnapshot(tool_rows=documents or []),
        agent_ids=_trace_agent_ids(agent),
        memory_ids=set(memory_ids),
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
    ).events


def _load_agent_trace_events(
    agent: Any,
    per_memory_limit: int = 100,
    total_limit: int = 300,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
    conversation_docs: Optional[List[Dict[str, Any]]] = None,
    tool_log_docs: Optional[List[Dict[str, Any]]] = None,
    query_metadata: Optional[Dict[str, Any]] = None,
) -> TraceEvents:
    """Normalize all returned identities, including unregistered memories.

    The provider bounds storage rows; total_limit bounds normalized children.
    per_memory_limit is retained for call compatibility and no longer silently
    discards bundles before their children can be counted.
    """
    memory_ids = _extract_agent_memory_ids(agent)
    if conversation_docs is None:
        snapshot = _load_trace_snapshot(
            agent_ids=sorted(_trace_agent_ids(agent)),
            memory_ids=memory_ids,
            thread_id=thread_id,
        )
        conversation_docs, loaded_tools, loaded_metadata = snapshot
        tool_log_docs = loaded_tools if tool_log_docs is None else tool_log_docs
        query_metadata = loaded_metadata
    return normalize_trace_snapshot(
        TraceSnapshot(
            conversation_rows=conversation_docs or [],
            tool_rows=tool_log_docs or [],
            query_metadata=query_metadata or {},
        ),
        agent_ids=_trace_agent_ids(agent),
        memory_ids=set(memory_ids),
        thread_id=thread_id,
        thread_memory_id=thread_memory_id,
        limit=total_limit,
        **scoped_trace_filters(),
    ).events


def _trace_sort_key(event: Dict[str, Any]):
    """Sort trace events by timestamp when available."""
    timestamp = _coerce_timestamp(event.get("timestamp"))
    if timestamp is not None:
        return (0, timestamp)
    raw = _to_text(event.get("timestamp")).strip()
    return (1, raw)
