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
from ..helpers import (
    _build_agent_nav_items,
    _build_agent_tool_count_map,
    _coerce_timestamp,
    _extract_agent_identifier,
    _extract_agent_memory_ids,
    _extract_agent_persona_name,
    _extract_message_timestamp,
    _load_agent_last_run_map,
    _retrieve_conversation_history,
    _sort_agents_by_last_run_desc,
    _to_text,
)
from ..security import (
    audit_trace_view,
    redact_trace_events,
    redact_value,
    trace_content_mode,
    ui_read_only,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])

_RUNTIME_TRACE_AGENT_ID = "__memorizz_runtime_traces__"
_TRACE_BUNDLE_RECORD_TYPE = "observability_trace_bundle"


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
):
    """Show traces dashboard with searchable agent table and per-agent timeline."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agents: List[Any] = []
    try:
        agents = _state["provider"].list_memagents()
    except Exception as exc:
        logger.error("Failed to list agents for traces: %s", exc)

    conversation_docs, tool_log_docs, trace_query_metadata = _load_trace_snapshot()
    agents = _discover_runtime_trace_agents(
        agents,
        conversation_docs=conversation_docs,
        tool_log_docs=tool_log_docs,
    )

    agent_query = _to_text(q).strip()
    trace_metrics = _build_agent_trace_metrics(
        agents, conversation_docs=conversation_docs
    )
    last_run_by_agent = {
        row_agent_id: float(metrics.get("last_ts") or 0.0)
        for row_agent_id, metrics in trace_metrics.items()
        if float(metrics.get("last_ts") or 0.0) > 0
    }
    if len(last_run_by_agent) < len(agents):
        fallback_last_run = _load_agent_last_run_map(agents)
        for row_agent_id, timestamp in fallback_last_run.items():
            last_run_by_agent.setdefault(row_agent_id, timestamp)
    agents = _sort_agents_by_last_run_desc(agents, last_run_by_agent=last_run_by_agent)

    tool_counts = _build_agent_tool_count_map(agents)
    for agent in agents:
        if not isinstance(agent, dict) or not agent.get("_is_virtual_trace_source"):
            continue
        runtime_tool_names = agent.get("_runtime_tool_names") or []
        tool_counts[_extract_agent_identifier(agent)] = len(runtime_tool_names)
    thread_rows_by_agent = _build_agent_thread_rows(
        agents, conversation_docs=conversation_docs
    )
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
                selected_agent = _state["provider"].retrieve_memagent(selected_agent_id)
            if not selected_agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            (
                selected_conversations,
                selected_tool_logs,
                selected_query_metadata,
            ) = _load_trace_snapshot(
                agent_ids=sorted(_trace_agent_ids(selected_agent)),
                memory_ids=_extract_agent_memory_ids(selected_agent),
                thread_id=selected_thread_id,
                limit=500,
            )
            # Use the targeted page for the timeline. The overview page remains
            # a separate bounded snapshot so an older agent cannot disappear
            # merely because other agents are busier.
            timeline_conversations = selected_conversations
            timeline_tool_logs = selected_tool_logs
            trace_query_metadata = selected_query_metadata

            trace_events = _load_agent_trace_events(
                selected_agent,
                thread_id=selected_thread_id,
                thread_memory_id=selected_thread_memory_id,
                conversation_docs=timeline_conversations,
                tool_log_docs=timeline_tool_logs,
            )
            latest_timestamp = (
                trace_events[-1].get("timestamp") if trace_events else None
            )
            trace_summary = {
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
                or len(trace_events) >= 300
                or bool(
                    not selected_thread_id
                    and not selected_thread_memory_id
                    and agent_event_count > len(trace_events)
                )
            )
            analysis_scope = (
                "thread" if selected_thread_id or selected_thread_memory_id else "agent"
            )
            try:
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
    display_trace_events = redact_trace_events(trace_events, mode=content_mode)
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
            "selected_thread_memory_id": selected_thread_memory_id,
            "selected_thread_row": selected_thread_row,
            "trace_events": display_trace_events,
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


@router.get("/traces/analysis.json", response_class=JSONResponse)
async def trace_analysis_export(
    request: Request,
    agent_id: str,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
):
    """Export a deterministic, read-only improvement report as JSON."""
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(status_code=503, detail="No memory provider connected")

    try:
        agents = provider.list_memagents() or []
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
            selected_agent = provider.retrieve_memagent(agent_id)
        except Exception as exc:
            logger.debug("Failed to retrieve trace export agent %s: %s", agent_id, exc)
    if not selected_agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    selected_thread_id = _to_text(thread_id).strip() or None
    selected_thread_memory_id = _to_text(thread_memory_id).strip() or None
    conversation_docs, tool_log_docs, query_metadata = _load_trace_snapshot(
        agent_ids=sorted(_trace_agent_ids(selected_agent)),
        memory_ids=_extract_agent_memory_ids(selected_agent),
        thread_id=selected_thread_id,
        limit=500,
    )
    events = _load_agent_trace_events(
        selected_agent,
        thread_id=selected_thread_id,
        thread_memory_id=selected_thread_memory_id,
        conversation_docs=conversation_docs,
        tool_log_docs=tool_log_docs,
    )
    source_is_virtual = bool(
        isinstance(selected_agent, dict)
        and selected_agent.get("_is_virtual_trace_source")
    )
    scope = "thread" if selected_thread_id or selected_thread_memory_id else "agent"
    try:
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
        window_truncated=bool(query_metadata.get("truncated")) or len(events) >= 300,
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
    store: str = "conversation",
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
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
    try:
        page = query(memory_type, **kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mode = trace_content_mode()
    safe_items: List[Dict[str, Any]] = []
    for raw_item in page.get("items") or []:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        if mode == "metadata":
            for key in ("content", "arguments", "result", "embedding"):
                if key in item:
                    item[key] = "[hidden]"
        elif mode == "redacted":
            item = redact_value(item)
        if item.get("user_id"):
            item["user_id"] = "[pseudonymized]"
        safe_items.append(_json_safe(item))
    response_payload = {
        key: _json_safe(value) for key, value in page.items() if key != "items"
    }
    response_payload["items"] = safe_items
    response_payload["content_mode"] = mode
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
        window_truncated=bool(query_metadata.get("truncated")) or len(events) >= 300,
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
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load bounded trace pages and surface their query/freshness metadata."""
    provider = _state.get("provider")
    if not provider:
        return [], [], {"truncated": False, "provider_native": False}

    from ...enums.memory_type import MemoryType

    def _page(memory_type: MemoryType) -> Dict[str, Any]:
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

    try:
        conversation_page = _page(MemoryType.CONVERSATION_MEMORY)
    except Exception as exc:
        logger.warning("Failed to query conversation trace documents: %s", exc)
        conversation_page = {"items": [], "truncated": False}
    try:
        tool_page = _page(MemoryType.TOOL_LOG)
    except Exception as exc:
        logger.warning("Failed to query tool trace documents: %s", exc)
        tool_page = {"items": [], "truncated": False}
    try:
        trace_page = _page(MemoryType.SHARED_MEMORY)
    except Exception as exc:
        logger.warning("Failed to query private trace bundles: %s", exc)
        trace_page = {"items": [], "truncated": False}

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

    return (
        [row for row in conversations if isinstance(row, dict)] + trace_bundles,
        [row for row in tool_logs if isinstance(row, dict)],
        metadata,
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
) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-agent thread rows for traces table child rows."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }
    memory_to_agent_ids: Dict[str, set[str]] = {}
    runtime_to_virtual_id: Dict[str, str] = {}
    for agent in agents:
        row_agent_id = _extract_agent_identifier(agent)
        if not row_agent_id:
            continue
        for memory_id in _extract_agent_memory_ids(agent):
            memory_to_agent_ids.setdefault(memory_id, set()).add(row_agent_id)
        if isinstance(agent, dict) and agent.get("_is_virtual_trace_source"):
            for runtime_agent_id in _trace_agent_ids(agent):
                runtime_to_virtual_id[runtime_agent_id] = row_agent_id

    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Fast path: aggregate directly from conversation documents.
    try:
        from ...enums.memory_type import MemoryType

        docs = (
            conversation_docs
            if conversation_docs is not None
            else provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        )
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            memory_id = _extract_trace_memory_id(doc)
            raw_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            associated_agent_ids = set(memory_to_agent_ids.get(memory_id, set()))
            if raw_agent_id in agent_ids:
                associated_agent_ids.add(raw_agent_id)
            virtual_id = runtime_to_virtual_id.get(raw_agent_id)
            if virtual_id:
                associated_agent_ids.add(virtual_id)
            if not associated_agent_ids:
                continue

            thread_id = _extract_trace_thread_id(doc, fallback=memory_id)
            key = _thread_row_key(thread_id, memory_id)
            ts = _extract_message_timestamp(doc) or 0.0

            for associated_agent_id in associated_agent_ids:
                by_thread = aggregate.setdefault(associated_agent_id, {})
                entry = by_thread.setdefault(
                    key,
                    {
                        "thread_id": thread_id,
                        "memory_id": memory_id,
                        "event_count": 0,
                        "last_ts": 0.0,
                    },
                )
                entry["event_count"] += 1
                if ts > entry["last_ts"]:
                    entry["last_ts"] = ts
    except Exception as exc:
        logger.debug("Failed thread aggregation from docs: %s", exc)

    # Fallback for agents missing doc-level association.
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        if agent_id in aggregate and aggregate[agent_id]:
            continue

        by_thread: Dict[str, Dict[str, Any]] = {}
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
            for msg in history:
                thread_id = _extract_trace_thread_id(msg, fallback=memory_id)
                key = _thread_row_key(thread_id, memory_id)
                ts = _extract_message_timestamp(msg) or 0.0
                entry = by_thread.setdefault(
                    key,
                    {
                        "thread_id": thread_id,
                        "memory_id": memory_id,
                        "event_count": 0,
                        "last_ts": 0.0,
                    },
                )
                entry["event_count"] += 1
                if ts > entry["last_ts"]:
                    entry["last_ts"] = ts
        if by_thread:
            aggregate[agent_id] = by_thread

    thread_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for agent_id, by_thread in aggregate.items():
        rows = []
        for entry in by_thread.values():
            rows.append(
                {
                    "thread_id": entry["thread_id"],
                    "memory_id": entry["memory_id"],
                    "event_count": int(entry["event_count"]),
                    "last_ts": float(entry["last_ts"]),
                    "last_activity": (
                        _format_trace_timestamp(entry["last_ts"])
                        if entry["last_ts"] > 0
                        else "—"
                    ),
                }
            )
        rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
        thread_rows_by_agent[agent_id] = rows

    return thread_rows_by_agent


def _build_agent_trace_metrics(
    agents: List[Any],
    *,
    conversation_docs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build per-agent trace metrics for the traces table."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }
    memory_to_agent_ids: Dict[str, set[str]] = {}
    runtime_to_virtual_id: Dict[str, str] = {}
    for agent in agents:
        row_agent_id = _extract_agent_identifier(agent)
        if not row_agent_id:
            continue
        for memory_id in _extract_agent_memory_ids(agent):
            memory_to_agent_ids.setdefault(memory_id, set()).add(row_agent_id)
        if isinstance(agent, dict) and agent.get("_is_virtual_trace_source"):
            for runtime_agent_id in _trace_agent_ids(agent):
                runtime_to_virtual_id[runtime_agent_id] = row_agent_id

    metrics: Dict[str, Dict[str, Any]] = {}

    # Fast path: aggregate from conversation docs in one pass.
    try:
        from ...enums.memory_type import MemoryType

        docs = (
            conversation_docs
            if conversation_docs is not None
            else provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        )
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            memory_id = _extract_trace_memory_id(doc)
            raw_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            associated_agent_ids = set(memory_to_agent_ids.get(memory_id, set()))
            if raw_agent_id in agent_ids:
                associated_agent_ids.add(raw_agent_id)
            virtual_id = runtime_to_virtual_id.get(raw_agent_id)
            if virtual_id:
                associated_agent_ids.add(virtual_id)
            if not associated_agent_ids:
                continue
            ts = _extract_message_timestamp(doc) or 0.0
            for associated_agent_id in associated_agent_ids:
                entry = metrics.setdefault(
                    associated_agent_id, {"event_count": 0, "last_ts": 0.0}
                )
                entry["event_count"] += 1
                if ts > entry["last_ts"]:
                    entry["last_ts"] = ts
    except Exception as exc:
        logger.debug("Failed to aggregate traces from conversation docs: %s", exc)

    # Fallback for agents not covered by doc-level metadata.
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        if agent_id in metrics and metrics[agent_id].get("event_count", 0) > 0:
            continue

        event_count = 0
        last_ts = 0.0
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
            event_count += len(history)
            for message in history:
                ts = _extract_message_timestamp(message) or 0.0
                if ts > last_ts:
                    last_ts = ts
        metrics[agent_id] = {"event_count": event_count, "last_ts": last_ts}

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
                ]
            ).lower()
            if query not in searchable:
                continue

        rows.append(row)

    rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
    return rows


def _expand_trace_bundle(
    message: Dict[str, Any], *, memory_id: str, thread_id: str
) -> Optional[List[Dict[str, Any]]]:
    """Expand MemAgent.run_stream telemetry into first-class timeline events."""
    raw = message.get("content") or message.get("text")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("type") != "trace_bundle":
        return None
    timestamp = _format_trace_timestamp(message.get("timestamp"))
    expanded = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        kind = (
            _to_text(event.get("trace_kind") or event.get("kind") or "trace")
            .strip()
            .lower()
        )
        title = _to_text(event.get("title") or kind or "Trace").strip()
        content = _to_text(event.get("content") or event.get("message")).strip()
        expanded.append(
            {
                "memory_id": memory_id,
                "thread_id": thread_id,
                "role": "tool",
                "kind": kind or "trace",
                "title": title or "Trace",
                "trace_id": _to_text(event.get("trace_id")).strip(),
                "content": content,
                "timestamp": timestamp,
                **{
                    field: event[field]
                    for field in (
                        "schema_version",
                        "application_id",
                        "agent_id",
                        "run_id",
                        "turn_id",
                        "root_trace_id",
                        "span_id",
                        "parent_span_id",
                        "user_id",
                        "tool_name",
                        "logical_tool_name",
                        "model_tool_name",
                        "tool_call_id",
                        "success",
                        "status",
                        "outcome",
                        "outcome_reason_code",
                        "tool_provider",
                        "primary_provider",
                        "fallback_provider",
                        "outcome_retryable",
                        "result_count",
                        "fallback_used",
                        "degraded",
                        "error_code",
                        "duration_ms",
                        "model",
                        "provider",
                        "input_tokens",
                        "output_tokens",
                        "cached_tokens",
                        "cost_usd",
                        "finish_reason",
                        "iteration",
                        "stage",
                        "total_tokens",
                        "request_id",
                        "client_page_type",
                        "client_page_id",
                        "client_title_fingerprint",
                        "canonical_page_type",
                        "canonical_page_id",
                        "canonical_title_fingerprint",
                        "thread_binding_status",
                        "expected_thread_id",
                        "ownership_verified",
                        "request_context_present",
                        "request_context_fingerprint",
                        "request_context_key_count",
                        "content_version",
                        "grounding_status",
                        "grounding_source",
                        "grounding_excerpt_count",
                        "grounding_source_ids",
                        "cache_decision",
                        "cache_enabled",
                        "cache_bypass_reason",
                        "memory_history_count",
                        "memory_candidate_count",
                        "memory_supplied_count",
                        "memory_referenced_count",
                        "memory_injected_chars",
                        "memory_degraded",
                        "memory_fallback_used",
                        "entity_profile_count",
                        "preference_count",
                        "conversation_memory_count",
                        "writing_sample_count",
                    )
                    if event.get(field) is not None
                },
            }
        )
    return expanded


def _load_agent_tool_log_events(
    agent: Any,
    memory_ids: List[str],
    *,
    thread_id: str = "",
    thread_memory_id: str = "",
    documents: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load durable TOOL_LOG executions into the observability timeline."""
    provider = _state.get("provider")
    agent_id = _extract_agent_identifier(agent)
    represented_agent_ids = _trace_agent_ids(agent)
    if not provider or not agent_id:
        return []
    if documents is None:
        try:
            from ...enums.memory_type import MemoryType

            documents = provider.list_all(MemoryType.TOOL_LOG) or []
        except Exception as exc:
            logger.debug("Failed to load TOOL_LOG events for %s: %s", agent_id, exc)
            return []

    events = []
    known_memory_ids = {str(item) for item in memory_ids if item}
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if represented_agent_ids and doc_agent_id not in represented_agent_ids:
            continue
        memory_id = _extract_trace_memory_id(doc, fallback="—")
        event_thread_id = _extract_trace_thread_id(doc, fallback=memory_id)
        if known_memory_ids and memory_id not in known_memory_ids:
            continue
        if thread_id and event_thread_id != thread_id and memory_id != thread_id:
            continue
        if thread_memory_id and memory_id != thread_memory_id:
            continue
        tool_name = _to_text(doc.get("tool_name") or "tool").strip()
        arguments = doc.get("arguments")
        result = doc.get("result")
        outcome_details = doc.get("outcome_details") or {}
        if isinstance(outcome_details, str):
            try:
                outcome_details = json.loads(outcome_details)
            except (TypeError, ValueError):
                outcome_details = {}
        if not isinstance(outcome_details, dict):
            outcome_details = {}
        outcome = _to_text(doc.get("outcome")).strip().lower() or (
            "success" if doc.get("success") is not False else "error"
        )
        content = json.dumps(
            {
                "tool_log_id": doc.get("tool_log_id")
                or doc.get("id")
                or doc.get("_id"),
                "tool_call_id": doc.get("tool_call_id"),
                "arguments": arguments,
                "result": result,
                "success": doc.get("success"),
                "error": doc.get("error"),
                "outcome": outcome,
                "outcome_details": outcome_details,
            },
            ensure_ascii=False,
            default=str,
            indent=2,
        )
        events.append(
            {
                "memory_id": memory_id,
                "thread_id": event_thread_id,
                "role": "tool",
                "kind": "execution_log",
                "title": f"Execution Log · {tool_name}",
                "tool_name": tool_name,
                "logical_tool_name": tool_name,
                "success": doc.get("success"),
                "outcome": outcome,
                "outcome_reason_code": outcome_details.get("reason_code"),
                "tool_provider": outcome_details.get("provider"),
                "primary_provider": outcome_details.get("primary_provider"),
                "fallback_provider": outcome_details.get("fallback_provider"),
                "outcome_retryable": outcome_details.get("retryable"),
                "result_count": outcome_details.get("result_count"),
                "fallback_used": outcome_details.get("fallback_used"),
                "degraded": outcome_details.get("degraded"),
                "error_code": doc.get("error_code") or doc.get("error"),
                "duration_ms": doc.get("duration_ms"),
                "trace_id": _to_text(doc.get("tool_call_id")).strip(),
                "content": content,
                "timestamp": _format_trace_timestamp(doc.get("timestamp")),
            }
        )
    return events


def _load_agent_trace_events(
    agent: Any,
    per_memory_limit: int = 100,
    total_limit: int = 300,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
    conversation_docs: Optional[List[Dict[str, Any]]] = None,
    tool_log_docs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Load trace events for a selected agent."""
    events: List[Dict[str, Any]] = []
    agent_id = _extract_agent_identifier(agent)
    memory_ids = _extract_agent_memory_ids(agent)
    selected_thread_id = _to_text(thread_id).strip()
    selected_thread_memory_id = _to_text(thread_memory_id).strip()

    histories_by_memory: Dict[str, List[Dict[str, Any]]] = {}
    if conversation_docs is not None:
        represented_agent_ids = _trace_agent_ids(agent)
        known_memory_ids = set(memory_ids)
        for doc in conversation_docs:
            if not isinstance(doc, dict):
                continue
            event_memory_id = _extract_trace_memory_id(doc, fallback="")
            if event_memory_id == "—":
                event_memory_id = ""
            doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            if (
                event_memory_id not in known_memory_ids
                and doc_agent_id not in represented_agent_ids
            ):
                continue
            if not event_memory_id:
                event_memory_id = "—"
            histories_by_memory.setdefault(event_memory_id, []).append(doc)
        for history in histories_by_memory.values():
            history.sort(key=lambda row: _extract_message_timestamp(row) or 0.0)

    selected_memory_ids = memory_ids
    if selected_thread_memory_id:
        selected_memory_ids = [
            memory_id
            for memory_id in memory_ids
            if memory_id == selected_thread_memory_id
        ]

    for memory_id in selected_memory_ids:
        history = (
            histories_by_memory.get(memory_id, [])
            if conversation_docs is not None
            else _retrieve_conversation_history(memory_id=memory_id, limit=None)
        )
        if per_memory_limit > 0 and len(history) > per_memory_limit:
            history = history[-per_memory_limit:]
        for msg in history:
            event_thread_id = _extract_trace_thread_id(msg, fallback=memory_id)
            if (
                selected_thread_id
                and event_thread_id != selected_thread_id
                and memory_id != selected_thread_id
            ):
                continue
            if selected_thread_memory_id and memory_id != selected_thread_memory_id:
                continue
            expanded = _expand_trace_bundle(
                msg, memory_id=memory_id, thread_id=event_thread_id
            )
            if expanded is not None:
                events.extend(expanded)
            else:
                events.append(
                    {
                        "memory_id": memory_id,
                        "thread_id": event_thread_id,
                        "role": _to_text(msg.get("role")).strip().lower() or "system",
                        "kind": "conversation",
                        "title": "Conversation event",
                        "content": _to_text(msg.get("content") or msg.get("text")),
                        "timestamp": _format_trace_timestamp(msg.get("timestamp")),
                    }
                )

    # Fallback for agents without memory_ids: load directly by agent_id.
    if not events and agent_id and conversation_docs is None:
        provider = _state.get("provider")
        if provider:
            try:
                from ...enums.memory_type import MemoryType

                docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    doc_agent_id = _to_text(
                        doc.get("agent_id") or doc.get("agentId")
                    ).strip()
                    if doc_agent_id != agent_id:
                        continue
                    event_memory_id = _extract_trace_memory_id(doc, fallback="—")
                    event_thread_id = _extract_trace_thread_id(
                        doc, fallback=event_memory_id
                    )
                    if (
                        selected_thread_id
                        and event_thread_id != selected_thread_id
                        and event_memory_id != selected_thread_id
                    ):
                        continue
                    if (
                        selected_thread_memory_id
                        and event_memory_id != selected_thread_memory_id
                    ):
                        continue
                    expanded = _expand_trace_bundle(
                        doc, memory_id=event_memory_id, thread_id=event_thread_id
                    )
                    if expanded is not None:
                        events.extend(expanded)
                    else:
                        events.append(
                            {
                                "memory_id": event_memory_id,
                                "thread_id": event_thread_id,
                                "role": _to_text(doc.get("role")).strip().lower()
                                or "system",
                                "kind": "conversation",
                                "title": "Conversation event",
                                "content": _to_text(
                                    doc.get("content") or doc.get("text")
                                ),
                                "timestamp": _format_trace_timestamp(
                                    doc.get("timestamp")
                                ),
                            }
                        )
            except Exception as exc:
                logger.debug(
                    "Failed direct trace fallback for agent %s: %s",
                    agent_id,
                    exc,
                )

    events.extend(
        _load_agent_tool_log_events(
            agent,
            memory_ids,
            thread_id=selected_thread_id,
            thread_memory_id=selected_thread_memory_id,
            documents=tool_log_docs,
        )
    )
    events.sort(key=_trace_sort_key)
    if total_limit > 0 and len(events) > total_limit:
        return events[-total_limit:]
    return events


def _trace_sort_key(event: Dict[str, Any]):
    """Sort trace events by timestamp when available."""
    timestamp = _coerce_timestamp(event.get("timestamp"))
    if timestamp is not None:
        return (0, timestamp)
    raw = _to_text(event.get("timestamp")).strip()
    return (1, raw)
