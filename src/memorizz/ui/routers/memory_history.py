"""Memory evolution and captured request inspection under trace permissions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...enums.memory_type import MemoryType
from ...memory_history import MemoryHistory
from ...observability.context_snapshots import ContextSnapshots
from ...observability.store import ObservabilityStore
from ..security import redact_value, trace_content_mode
from ..state import _state, templates
from ..trace_access import (
    current_principal,
    require_trace_permission,
    scoped_trace_filters,
)

router = APIRouter(tags=["memory-history"])


def _provider():
    require_trace_permission("trace.read")
    if not _state.get("provider"):
        raise HTTPException(400, "Not connected")
    return _state["provider"]


def _agent_scope(provider, agent_id, memory_id=None):
    agent = provider.retrieve_memagent(agent_id)
    if agent is None:
        raise HTTPException(404, "Agent not found")
    if memory_id and memory_id not in (getattr(agent, "memory_ids", None) or []):
        raise HTTPException(404, "Conversation not found")
    return agent


def _response(value):
    return JSONResponse(value, headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/traces/memory-evolution")
async def evolution_page(
    request: Request, agent_id: str = "", memory_id: str = "", run_id: str = ""
):
    _provider()
    return templates.TemplateResponse(
        "memory_evolution.html",
        {
            "request": request,
            "agent_id": agent_id,
            "memory_id": memory_id,
            "run_id": run_id,
            "scope": scoped_trace_filters(),
            "archive_available": not current_principal.get().restricted,
        },
    )


@router.get("/traces/memory-history")
async def evolution_data(
    agent_id: str | None = None,
    memory_id: str | None = None,
    run_id: str | None = None,
    memory_type: str | None = None,
    action: str | None = None,
    actor: str | None = None,
    limit: int = 200,
    cursor: str | None = None,
    include_existing: bool = False,
):
    provider = _provider()
    scope = scoped_trace_filters()
    agent = None
    if agent_id:
        # Coding-agent and harness identities need not be saved MemAgents.
        agent = await run_in_threadpool(provider.retrieve_memagent, agent_id)
    history = MemoryHistory(provider)
    page = await run_in_threadpool(
        history.timeline,
        agent_id=agent_id,
        memory_id=memory_id,
        run_id=run_id,
        memory_type=memory_type,
        action=action,
        actor=actor,
        limit=limit,
        cursor=cursor,
        **scope,
    )
    existing = []
    if include_existing and not cursor and not run_id and (agent_id or memory_id):
        recorded = {(e["memory_type"], e["target_record_id"]) for e in page["events"]}
        existing = await run_in_threadpool(
            history.observations,
            agent_id=agent_id,
            memory_ids=[memory_id]
            if memory_id
            else getattr(agent, "memory_ids", [])
            if agent
            else [],
            limit=limit,
            exclude_records=recorded,
            **scope,
        )
        existing = [
            e
            for e in existing
            if (e["memory_type"], e["target_record_id"]) not in recorded
        ]
    # Resolve only identities present in the scoped result, never list agents.
    writers = {}
    resolved = {agent_id: agent} if agent_id and agent else {}
    for identity in {
        e.get(key)
        for e in [*page["events"], *existing]
        for key in ("actor", "agent_id", "initiator")
        if e.get(key)
    }:
        try:
            writer = resolved.get(identity) or await run_in_threadpool(
                provider.retrieve_memagent, identity
            )
            if writer and (
                not scope.get("application_id")
                or writer.application_id == scope["application_id"]
            ):
                writers[identity] = writer.name or identity
                resolved[identity] = writer
        except Exception:
            pass  # Historical/external identities need not be saved agents.
    navigation = {"agent": None, "delegates": []}
    if (
        agent
        and (
            not scope.get("application_id")
            or agent.application_id == scope["application_id"]
        )
        and (not current_principal.get().restricted or page["events"] or existing)
    ):
        navigation["agent"] = {"agent_id": agent_id, "name": agent.name or agent_id}
        # Current configured delegates, not an inferred historical execution tree.
        for identity in dict.fromkeys(agent.delegates or []):
            try:
                delegate = resolved.get(identity) or await run_in_threadpool(
                    provider.retrieve_memagent, identity
                )
                if delegate and (
                    not scope.get("application_id")
                    or delegate.application_id == scope["application_id"]
                ):
                    navigation["delegates"].append(
                        {"agent_id": identity, "name": delegate.name or identity}
                    )
            except Exception:
                pass  # Deleted or unavailable delegates are not invented.
    return _response(
        {
            **page,
            "existing": existing,
            "writers": writers,
            "navigation": navigation,
            "playground_available": not current_principal.get().restricted,
            "scope": {"agent_id": agent_id, "memory_id": memory_id, "run_id": run_id},
            "content_mode": trace_content_mode(),
        }
    )


@router.get("/traces/memory-history/{event_id}/record")
async def current_memory_record(event_id: str, agent_id: str | None = None):
    provider = _provider()
    history = MemoryHistory(provider)
    event = history.get_change(event_id, agent_id=agent_id, **scoped_trace_filters())
    if event is None:
        raise HTTPException(404, "Memory change not found")
    mode = trace_content_mode()
    if mode == "metadata":
        return _response({"state": "hidden", "content_mode": mode})
    # Agent configuration can contain credentials and is edited on its own page.
    if event["memory_type"] == MemoryType.MEMAGENT.value:
        return _response({"state": "configuration", "content_mode": mode})
    result = history.current_record(event)
    row = result.pop("record", {})
    fields = {}
    budget = 16000
    truncated = False
    import json

    for key in (
        "content",
        "text",
        "summary",
        "description",
        "value",
        "facts",
        "observations",
        "messages",
        "steps",
    ):
        if key not in row or row[key] is None:
            continue
        value = row[key]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        # Always remove credential-shaped fields, even in a local full trace view.
        value = redact_value(value)
        text = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, default=str, indent=2)
        )
        if len(text) > budget:
            text = text[:budget]
            truncated = True
        fields[key] = text
        budget -= len(text)
        if budget <= 0:
            truncated = True
            break
    return _response(
        {**result, "fields": fields, "truncated": truncated, "content_mode": mode}
    )


@router.get("/agents/{agent_id}/playground/context-history")
async def context_turns(
    agent_id: str,
    memory_id: str,
    snapshot_cursor: str | None = None,
    trace_cursor: str | None = None,
    skip_snapshots: bool = False,
    skip_traces: bool = False,
    limit: int = Query(200, ge=1, le=1000),
    user_id: str | None = None,
    application_id: str | None = None,
):
    provider = _provider()
    _agent_scope(provider, agent_id, memory_id)
    scope = scoped_trace_filters(
        {
            key: value
            for key, value in {
                "user_id": user_id,
                "application_id": application_id,
            }.items()
            if value is not None
        }
    )
    snapshot_page = (
        {"items": [], "next_cursor": None}
        if skip_snapshots
        else ContextSnapshots(provider).page(
            agent_id=agent_id,
            memory_id=memory_id,
            cursor=snapshot_cursor,
            limit=limit,
            **scope,
        )
    )
    groups = {}
    for snapshot in snapshot_page["items"]:
        key = (
            snapshot.get("turn_id")
            or snapshot.get("root_trace_id")
            or snapshot["record_id"]
        )
        group = groups.setdefault(
            key,
            {
                "turn_id": key,
                "timestamp": snapshot.get("timestamp"),
                "calls": [],
                "status": "recorded",
            },
        )
        group["calls"].append(snapshot)
    page = (
        {"items": [], "next_cursor": None}
        if skip_traces
        else provider.query_observability_records(
            MemoryType.SHARED_MEMORY,
            memory_ids=[memory_id],
            record_type="observability_trace_bundle",
            limit=limit,
            cursor=trace_cursor,
            **scope,
        )
    )
    for row in page.get("items", []):
        bundle = ObservabilityStore._payload(row)
        if (
            not bundle
            or bundle.get("agent_id") != agent_id
            or bundle.get("memory_id") != memory_id
        ):
            continue
        events = bundle.get("events") or []
        # Harness envelopes repeat the native turn in a different event format.
        # They do not describe a separate MemAgent request or cache replay.
        if events and all(e.get("run_id") and e.get("type") for e in events):
            continue
        key = bundle.get("turn_id") or bundle.get("root_trace_id")
        if not key:
            continue
        group = groups.setdefault(
            key,
            {
                "turn_id": key,
                "timestamp": bundle.get("timestamp") or bundle.get("created_at"),
                "calls": [],
                "status": "unavailable",
            },
        )
        calls = [e for e in events if e.get("trace_kind") == "model_call"]
        if not calls and not group["calls"] and events:
            group["status"] = "no_model_request"
            group[
                "reason"
            ] = "This turn made no model request (for example, a semantic-cache replay)."
        elif not group["calls"]:
            group[
                "reason"
            ] = "No captured request is available for this turn. Older turns were recorded before snapshots were enabled."
    turns = sorted(
        groups.values(),
        key=lambda g: (str(g.get("timestamp") or ""), str(g["turn_id"])),
    )
    return _response(
        {
            "turns": turns,
            "next_cursors": {
                "snapshots": snapshot_page.get("next_cursor"),
                "traces": page.get("next_cursor"),
            },
            "capture_boundary": "Inputs after prompt fitting; provider-specific wire transformations are not included.",
            "content_mode": trace_content_mode(),
        }
    )


@router.get("/agents/{agent_id}/playground/context-history/{snapshot_id}")
async def context_snapshot(
    agent_id: str,
    snapshot_id: str,
    memory_id: str,
    user_id: str | None = None,
    application_id: str | None = None,
):
    provider = _provider()
    _agent_scope(provider, agent_id, memory_id)
    value = ContextSnapshots(provider).get(
        snapshot_id,
        agent_id=agent_id,
        memory_id=memory_id,
        **scoped_trace_filters(
            {
                key: value
                for key, value in {
                    "user_id": user_id,
                    "application_id": application_id,
                }.items()
                if value is not None
            }
        ),
    )
    if value is None:
        raise HTTPException(404, "Snapshot not found")
    mode = trace_content_mode()
    if mode == "metadata":
        value = {
            k: v
            for k, v in value.items()
            if k not in {"messages", "tools", "memory_evidence"}
        }
    elif mode == "redacted":
        value = redact_value(value)
    return _response({"snapshot": value, "content_mode": mode})
