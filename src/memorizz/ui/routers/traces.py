# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Traces / Observability pages.

Extracted verbatim from ``ui/app.py``: GET /observability (legacy redirect)
and GET /traces, plus the trace helpers used only by these routes. Route
paths, response classes, and behavior are unchanged.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

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
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["traces"])


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

    agent_query = _to_text(q).strip()
    last_run_by_agent = _load_agent_last_run_map(agents)
    agents = _sort_agents_by_last_run_desc(agents, last_run_by_agent=last_run_by_agent)

    tool_counts = _build_agent_tool_count_map(agents)
    trace_metrics = _build_agent_trace_metrics(agents)
    thread_rows_by_agent = _build_agent_thread_rows(agents)
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
    selected_thread_row: Optional[Dict[str, Any]] = None
    error = None

    selected_agent_id = _to_text(agent_id).strip() or None
    selected_thread_id = _to_text(thread_id).strip() or None
    selected_thread_memory_id = _to_text(thread_memory_id).strip() or None
    if selected_agent_id:
        try:
            selected_agent = _state["provider"].retrieve_memagent(selected_agent_id)
            if not selected_agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            trace_events = _load_agent_trace_events(
                selected_agent,
                thread_id=selected_thread_id,
                thread_memory_id=selected_thread_memory_id,
            )
            latest_timestamp = (
                trace_events[-1].get("timestamp") if trace_events else None
            )
            trace_summary = {
                "event_count": len(trace_events),
                "latest_timestamp": _format_trace_timestamp(latest_timestamp),
            }
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
            "selected_agent_id": selected_agent_id,
            "selected_thread_id": selected_thread_id,
            "selected_thread_memory_id": selected_thread_memory_id,
            "selected_thread_row": selected_thread_row,
            "trace_events": trace_events,
            "trace_summary": trace_summary,
            "error": error,
        },
    )


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
    _ = thread_id
    return memory_id


def _build_agent_thread_rows(agents: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-agent thread rows for traces table child rows."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }

    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Fast path: aggregate directly from conversation documents.
    try:
        from ...enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            if not agent_id or agent_id not in agent_ids:
                continue

            memory_id = _extract_trace_memory_id(doc)
            key = _thread_row_key(memory_id, memory_id)
            ts = _extract_message_timestamp(doc) or 0.0

            by_thread = aggregate.setdefault(agent_id, {})
            entry = by_thread.setdefault(
                key,
                {
                    "thread_id": memory_id,
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
                key = _thread_row_key(memory_id, memory_id)
                ts = _extract_message_timestamp(msg) or 0.0
                entry = by_thread.setdefault(
                    key,
                    {
                        "thread_id": memory_id,
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


def _build_agent_trace_metrics(agents: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Build per-agent trace metrics for the traces table."""
    provider = _state.get("provider")
    if not provider or not agents:
        return {}

    agent_ids = {
        _extract_agent_identifier(agent)
        for agent in agents
        if _extract_agent_identifier(agent)
    }

    metrics: Dict[str, Dict[str, Any]] = {}

    # Fast path: aggregate from conversation docs in one pass.
    try:
        from ...enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
            if not agent_id or agent_id not in agent_ids:
                continue
            ts = _extract_message_timestamp(doc) or 0.0
            entry = metrics.setdefault(agent_id, {"event_count": 0, "last_ts": 0.0})
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


def _load_agent_trace_events(
    agent: Any,
    per_memory_limit: int = 100,
    total_limit: int = 300,
    thread_id: Optional[str] = None,
    thread_memory_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load trace events for a selected agent."""
    events: List[Dict[str, Any]] = []
    agent_id = _extract_agent_identifier(agent)
    memory_ids = _extract_agent_memory_ids(agent)
    selected_thread_id = _to_text(thread_id).strip()
    selected_thread_memory_id = _to_text(thread_memory_id).strip()

    for memory_id in memory_ids:
        history = _retrieve_conversation_history(memory_id=memory_id, limit=None)
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
            events.append(
                {
                    "memory_id": memory_id,
                    "thread_id": event_thread_id,
                    "role": _to_text(msg.get("role")).strip().lower() or "system",
                    "content": _to_text(msg.get("content") or msg.get("text")),
                    "timestamp": _format_trace_timestamp(msg.get("timestamp")),
                }
            )

    # Fallback for agents without memory_ids: load directly by agent_id.
    if not events and agent_id:
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
                    events.append(
                        {
                            "memory_id": event_memory_id,
                            "thread_id": event_thread_id,
                            "role": _to_text(doc.get("role")).strip().lower()
                            or "system",
                            "content": _to_text(doc.get("content") or doc.get("text")),
                            "timestamp": _format_trace_timestamp(doc.get("timestamp")),
                        }
                    )
            except Exception as exc:
                logger.debug(
                    "Failed direct trace fallback for agent %s: %s",
                    agent_id,
                    exc,
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
