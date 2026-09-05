"""Permission-gated incident lookup, explicit reveal and inert replay drafts."""

import inspect
import re

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ...observability import ObservabilityStore
from ...observability.coverage import trace_coverage
from ...observability.evidence import event_scope
from ...observability.inspection import event_anchor
from ...observability.lineage import build_lineage_inspectors
from ...observability.models import ResourceRef
from ...observability.normalization import TraceEvents, select_trace_events
from ...observability.privacy import validate_opaque
from ...observability.references import event_resource_refs
from ..security import audit_trace_view, redact_trace_events, trace_content_mode
from ..state import _state
from ..trace_access import (
    current_principal,
    require_trace_permission,
    scoped_trace_filters,
)

router = APIRouter(tags=["trace-tools"])


class TraceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=240)
    agent_id: str = Field(min_length=1)
    root_trace_id: str = Field(min_length=1)
    thread_id: str | None = None
    user_id: str | None = None
    application_id: str | None = None
    event_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    thread_memory_id: str | None = None
    run_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    _opaque = field_validator(
        "agent_id",
        "root_trace_id",
        "thread_id",
        "user_id",
        "application_id",
        "event_id",
        "turn_id",
        "task_id",
        "thread_memory_id",
        "run_id",
    )(validate_opaque)

    @model_validator(mode="after")
    def valid_time_range(self):
        select_trace_events([], start_time=self.start_time, end_time=self.end_time)
        return self

    def filters(self):
        return scoped_trace_filters(
            self.model_dump(
                exclude_none=True, exclude={"event_id", "task_id", "thread_memory_id"}
            )
        )


class AccountLookup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=320)


def _selection(**values):
    try:
        return TraceSelection(**values)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid trace selection") from None


def _audit_required(request, action):
    if not audit_trace_view(request, action, content_mode="metadata"):
        raise HTTPException(
            status_code=503, detail="Audited trace access is unavailable"
        )


def _provider():
    provider = _state.get("provider")
    if provider is None:
        raise HTTPException(status_code=503, detail="No memory provider connected")
    return provider


def _query(filters, *, limit=250, cursor=None):
    filters = dict(filters)
    if "agent_id" in filters:
        filters["agent_ids"] = [filters.pop("agent_id")]
    try:
        return _provider().query_trace_events(limit=limit, cursor=cursor, **filters)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid trace query or cursor"
        ) from None
    except Exception:
        raise HTTPException(
            status_code=503, detail="Trace evidence is unavailable"
        ) from None


def _collect(selection):
    rows, cursor = [], None
    page_metadata = []
    seen_cursors, seen_events = set(), set()
    incomplete = False
    for _ in range(20):
        filters = selection.filters()
        page = _query(filters, limit=1000, cursor=cursor)
        if (
            page.get("normalization_errors")
            or page.get("errors")
            or page.get("coverage") == "untrusted"
            or page.get("read_completeness") == "untrusted"
            or page.get("query_errors")
        ):
            raise HTTPException(status_code=409, detail="Trace coverage is untrusted")
        page_metadata.append(
            {k: v for k, v in page.items() if k not in {"items", "next_cursor"}}
        )
        for row in page.get("items", []):
            identity = (*event_scope(row), row.get("run_id"), row.get("event_id"))
            if identity in seen_events:
                raise HTTPException(
                    status_code=409, detail="Trace pagination repeated an event"
                )
            seen_events.add(identity)
            rows.append(row)
        previous_cursor = cursor
        cursor = page.get("next_cursor")
        if cursor:
            if cursor in seen_cursors:
                raise HTTPException(
                    status_code=409, detail="Trace pagination did not advance"
                )
            seen_cursors.add(cursor)
        read = page.get("read_completeness", page.get("coverage", "unknown"))
        # A continuation can resolve page truncation, but never an unknown read.
        # Native final pages mark window_complete=False because they start at a
        # cursor; only a known stored-event page can make that bounded claim.
        incomplete |= (
            read not in {"complete", "partial"}
            or (not cursor and (read == "partial" or bool(page.get("truncated"))))
            or (
                page.get("window_complete") is False
                and not cursor
                and not (
                    previous_cursor
                    and page.get("coverage_scope") == "stored_event_page"
                )
            )
        )
        if not cursor:
            if not rows:
                raise HTTPException(
                    status_code=404, detail="Trace not found in authorized scope"
                )
            evidence = TraceEvents(
                rows,
                coverage={
                    "read_completeness": "partial" if incomplete else "complete",
                    "coverage": "partial" if incomplete else "complete",
                    "window_complete": not incomplete,
                    "truncated": incomplete,
                    "normalized_events": len(rows),
                    "normalization_errors": 0,
                    "page_metadata": page_metadata,
                    "store_selection": ["trace"],
                },
            )
            evidence = select_trace_events(
                evidence,
                **selection.model_dump(
                    exclude_none=True,
                    exclude={"agent_id", "user_id", "application_id", "event_id"},
                ),
            )
            if not evidence:
                raise HTTPException(
                    status_code=404, detail="Task not found in authorized scope"
                )
            evidence.coverage = trace_coverage(evidence, evidence.coverage)
            return evidence
    raise HTTPException(
        status_code=409,
        detail="Trace exceeds the inspection limit; narrow the selection",
    )


def trace_capabilities(request):
    """Availability requires both permission and configured host integration."""
    hooks = {
        "account.resolve": "trace_identity_resolver",
        "artifact.lookup": "trace_artifact_resolver",
        "replay.create": "trace_resource_authorizer",
    }
    principal = current_principal.get()
    return {
        permission: {
            "available": principal.allows(permission)
            and callable(getattr(request.app.state, hook, None)),
            "configured": callable(getattr(request.app.state, hook, None)),
            "setup": {
                "account.resolve": "identity_resolver",
                "artifact.lookup": "artifact_resolver",
                "replay.create": "resource_authorizer",
            }[permission],
        }
        for permission, hook in hooks.items()
    }


@router.get("/traces/capabilities.json")
async def capabilities(request: Request):
    require_trace_permission("trace.read")
    return JSONResponse({"capabilities": trace_capabilities(request)})


async def _hook(callback, *args):
    if not callable(callback):
        raise HTTPException(
            status_code=501, detail="Host integration hook is not configured"
        )
    from starlette.concurrency import run_in_threadpool

    try:
        result = (
            await callback(*args)
            if inspect.iscoroutinefunction(callback)
            else await run_in_threadpool(callback, *args)
        )
        return await result if inspect.isawaitable(result) else result
    except Exception:
        raise HTTPException(
            status_code=503, detail="Host lookup is unavailable"
        ) from None


@router.get("/traces/find.json")
async def find_incident(
    request: Request,
    q: str = "",
    agent_id: str | None = None,
    user_id: str | None = None,
    application_id: str | None = None,
    thread_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
):
    require_trace_permission("trace.read")
    try:
        if len(q) > 240:
            raise ValueError()
        validate_opaque(q)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Use opaque IDs; resolve account emails through the account lookup action",
        ) from None
    filters = scoped_trace_filters(
        {
            key: value
            for key, value in {
                "query": q or None,
                "agent_id": agent_id,
                "user_id": user_id,
                "application_id": application_id,
                "thread_id": thread_id,
                "start_time": start_time,
                "end_time": end_time,
            }.items()
            if value is not None
        }
    )
    page = _query(filters, limit=max(1, min(limit, 250)), cursor=cursor)
    items = redact_trace_events(page["items"], mode="metadata")
    for event in items:
        event.pop("content", None)
        event["anchor"] = event_anchor(event.get("event_id"))
        event["result_type"] = (
            "artifact"
            if event.get("output_refs")
            else "job"
            if event.get("job_ref")
            else "error"
            if event.get("error_code")
            else "trace"
        )
    audit_trace_view(
        request, "incident_search", result_count=len(items), content_mode="metadata"
    )
    return JSONResponse({**page, "items": items})


@router.post("/traces/account/resolve")
async def resolve_account(request: Request, lookup: AccountLookup):
    require_trace_permission("account.resolve")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", lookup.email):
        raise HTTPException(status_code=400, detail="Invalid account address")
    _audit_required(request, "account_resolution_requested")
    try:
        result = await _hook(
            request.app.state.trace_identity_resolver,
            lookup.email,
            current_principal.get().filters(),
        )
        if (
            not isinstance(result, dict)
            or not result.get("user_id")
            or set(result) - {"user_id", "application_id"}
        ):
            raise ValueError()
        for value in result.values():
            validate_opaque(value)
            if value is not None and (not isinstance(value, str) or len(value) > 240):
                raise ValueError()
        result = scoped_trace_filters(result)
    except HTTPException:
        audit_trace_view(request, "account_resolution_failed", content_mode="metadata")
        raise
    except Exception:
        audit_trace_view(request, "account_resolution_failed", content_mode="metadata")
        raise HTTPException(
            status_code=404,
            detail="Account could not be resolved in the authorized scope",
        ) from None
    audit_trace_view(
        request, "account_resolution_completed", result_count=1, content_mode="metadata"
    )
    return JSONResponse(result)


@router.get("/traces/inspect.json")
async def inspect_lineage(
    request: Request,
    agent_id: str,
    root_trace_id: str,
    user_id: str | None = None,
    application_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    thread_memory_id: str | None = None,
    run_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
):
    require_trace_permission("artifact.lookup")
    rows = _collect(
        _selection(
            agent_id=agent_id,
            root_trace_id=root_trace_id,
            user_id=user_id,
            application_id=application_id,
            thread_id=thread_id,
            turn_id=turn_id,
            task_id=task_id,
            thread_memory_id=thread_memory_id,
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
        )
    )
    safe = TraceEvents(
        redact_trace_events(rows, mode="metadata"), coverage=rows.coverage
    )
    audit_trace_view(
        request,
        "lineage_inspection",
        agent_id=agent_id,
        result_count=len(rows),
        content_mode="metadata",
    )
    return JSONResponse({"coverage": rows.coverage, **build_lineage_inspectors(safe)})


@router.get("/traces/artifact.json")
async def lookup_artifact(request: Request, resource_type: str, ref: str):
    require_trace_permission("artifact.lookup")
    try:
        resource = ResourceRef(resource_type=resource_type, ref=ref)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid resource reference"
        ) from None
    _audit_required(request, "artifact_lookup_requested")
    result = await _hook(
        request.app.state.trace_artifact_resolver,
        resource.model_dump(exclude_none=True),
        current_principal.get().filters(),
    )
    if not isinstance(result, dict) or result.get("ownership_verified") is not True:
        raise HTTPException(
            status_code=404, detail="Artifact not found in authorized scope"
        )
    safe = {
        key: result.get(key)
        for key in ("exists", "ownership_verified", "version", "title_fingerprint")
        if key in result
    }
    for value in safe.values():
        try:
            validate_opaque(value)
            if type(value) not in (str, bool, type(None)) or (
                isinstance(value, str) and len(value) > 240
            ):
                raise ValueError()
        except ValueError:
            raise HTTPException(
                status_code=502, detail="Host returned unsafe artifact metadata"
            ) from None
    audit_trace_view(
        request, "artifact_lookup_completed", result_count=1, content_mode="metadata"
    )
    return JSONResponse(safe)


@router.post("/traces/reveal")
async def reveal_event(request: Request, selection: TraceSelection):
    require_trace_permission("trace.reveal")
    mode = trace_content_mode()
    if mode == "metadata" or not selection.event_id:
        raise HTTPException(
            status_code=403, detail="Content reveal is disabled or no event is selected"
        )
    _audit_required(request, "trace_reveal_requested")
    rows = _collect(selection)
    matches = [row for row in rows if row.get("event_id") == selection.event_id]
    if not matches:
        raise HTTPException(status_code=404, detail="Event not found")
    if len(matches) != 1:
        raise HTTPException(
            status_code=409,
            detail="Event selection is ambiguous; narrow the turn or run",
        )
    selected = matches[0]
    content = selected.get("content")
    if not content:
        index = _provider().get_observability_index()
        if index is not None and index.ready():
            filters = selection.filters()
            filters["agent_ids"] = [filters.pop("agent_id")]
            content = index.preview(selection.event_id, **filters)
    if not content:
        raise HTTPException(status_code=404, detail="Preview unavailable or expired")
    safe = redact_trace_events([{**selected, "content": content}], mode=mode)[0]
    audit_trace_view(
        request,
        "trace_content_revealed",
        agent_id=selection.agent_id,
        thread_id=selection.thread_id,
        result_count=1,
        content_mode=mode,
    )
    return JSONResponse(
        {
            "event_id": selection.event_id,
            "content": safe.get("content"),
            "content_mode": mode,
        }
    )


async def _replay(request, selection, *, persist):
    from ...observability.index import digest

    principal = require_trace_permission("replay.create")
    if not callable(request.app.state.trace_resource_authorizer):
        raise HTTPException(
            status_code=501, detail="Host resource authorizer is not configured"
        )
    _audit_required(
        request, "replay_draft_requested" if persist else "replay_plan_requested"
    )
    rows = _collect(selection)
    metadata = rows.coverage
    if (
        metadata.get("read_completeness", metadata.get("coverage")) != "complete"
        or metadata.get("truncated")
        or metadata.get("normalization_errors")
        or metadata.get("window_complete") is False
        or len({(e.get("application_id"), e.get("user_id")) for e in rows}) != 1
    ):
        raise HTTPException(
            status_code=409,
            detail="A complete, authorized single-tenant trace is required",
        )
    checked = set()

    async def check_refs():
        validated = [(event, event_resource_refs(event)) for event in rows]
        for event, refs in validated:
            for ref in refs:
                key = digest(ref)
                if key not in checked:
                    allowed = await _hook(
                        request.app.state.trace_resource_authorizer,
                        ref,
                        {
                            "application_id": event.get("application_id"),
                            "user_id": event.get("user_id"),
                        },
                    )
                    if allowed is not True:
                        raise HTTPException(
                            status_code=403, detail="Resource access not verified"
                        )
                    checked.add(key)

    try:
        await check_refs()
    except ValueError:
        raise HTTPException(
            status_code=409, detail="Resource evidence is invalid"
        ) from None
    try:
        draft = ObservabilityStore(_provider()).create_replay_draft(
            rows,
            created_by=principal.principal_id,
            authorize_resource=lambda ref, scope: digest(ref) in checked,
            persist=persist,
        )
    except (ValueError, PermissionError):
        raise HTTPException(
            status_code=409,
            detail="A complete, authorized single-tenant trace is required",
        ) from None
    audit_trace_view(
        request,
        "replay_draft_created" if persist else "replay_plan_exported",
        agent_id=selection.agent_id,
        result_count=len(rows),
        content_mode="metadata",
    )
    # Even privileged callers receive content-free evidence only.
    return JSONResponse(draft)


@router.post("/traces/replays")
async def create_replay(request: Request, selection: TraceSelection):
    return await _replay(request, selection, persist=True)


@router.get("/traces/replay-plan.json")
async def replay_plan(
    request: Request,
    agent_id: str,
    root_trace_id: str,
    user_id: str | None = None,
    application_id: str | None = None,
    thread_id: str | None = None,
    turn_id: str | None = None,
    task_id: str | None = None,
    thread_memory_id: str | None = None,
    run_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
):
    return await _replay(
        request,
        _selection(
            agent_id=agent_id,
            root_trace_id=root_trace_id,
            user_id=user_id,
            application_id=application_id,
            thread_id=thread_id,
            turn_id=turn_id,
            task_id=task_id,
            thread_memory_id=thread_memory_id,
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
        ),
        persist=False,
    )
