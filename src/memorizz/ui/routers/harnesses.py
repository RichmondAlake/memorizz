"""Operator UI and JSON API for the durable MemoRizz meta-harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...metaharness import catalog
from ...metaharness.catalog import fill_reported_models
from ..helpers import (
    _approval_payload,
    _build_agent_nav_items,
    _json_object,
    _list_agents,
)
from ..security import ui_read_only
from ..state import _state, get_meta_harness, templates

logger = logging.getLogger(__name__)
router = APIRouter(tags=["meta-harness"])
# The page's run ledger shows the newest runs; the JSON API pages through more.
RUN_LIMIT = 100
# Plans and comparisons shown on the page.
WORKFLOW_LIMIT = 20


def _service():
    try:
        return get_meta_harness()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Meta-harness UI operation failed")
    return HTTPException(status_code=500, detail="Harness operation failed")


@router.get("/harnesses", response_class=HTMLResponse)
async def harnesses_page(request: Request):
    if not _state.get("provider"):
        return RedirectResponse(url="/connect", status_code=302)
    service = _service()
    try:
        agents = list(_list_agents() or [])
    except Exception:
        agents = []
    try:
        harnesses = await asyncio.to_thread(service.list_harnesses)
        runs = await asyncio.to_thread(service.list_runs, limit=RUN_LIMIT)
        approvals = await asyncio.to_thread(
            service.list_approvals, status="pending", limit=100
        )
        workflows, workflow_runs = await asyncio.to_thread(
            _workflows_with_runs, service, runs
        )
        error = None
    except Exception as exc:
        harnesses, runs, approvals, workflows, workflow_runs = [], [], [], [], {}
        error = str(exc)
    from datetime import datetime, timezone

    from .. import dashboard as dash
    from ..execution_view import build_harness_monitor

    now = datetime.now(timezone.utc)
    monitor = build_harness_monitor(
        harnesses,
        runs,
        approvals,
        now=now,
        run_limit=RUN_LIMIT,
        workflows=workflows,
        workflow_runs=workflow_runs,
    )
    step_rows = list(monitor.get("runs") or [])
    is_scratch = getattr(service, "is_scratch_workspace", None)
    for row in monitor.get("runs") or []:
        row["scratch"] = bool(callable(is_scratch) and is_scratch(row.get("workspace")))
    for workflow in monitor.get("workflows") or []:
        step_rows.extend(step["run"] for step in workflow["steps"] if step.get("run"))
        # A scratch folder was the form's blank workspace; relaunching gets a
        # fresh one, so the form shows it blank again.
        setup = workflow.get("setup") or {}
        if callable(is_scratch) and is_scratch(setup.get("workspace")):
            setup["workspace"] = ""
    counts = (
        await asyncio.to_thread(
            _step_counts, service, [{"run_id": row["run_id"]} for row in step_rows]
        )
        if step_rows
        else {}
    )
    for row in step_rows:
        found = counts.get(row["run_id"]) or {}
        row["steps"] = {kind: found.get(kind, 0) for kind in STEP_TYPES}
        row["step_total"] = sum(row["steps"].values())
    await asyncio.to_thread(fill_reported_models, service, step_rows)
    return templates.TemplateResponse(
        "harnesses.html",
        {
            "request": request,
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "agents_nav": _build_agent_nav_items(agents=agents),
            "active_agent_id": None,
            "active_page": "harnesses",
            "harnesses": harnesses,
            "workspace_roots": list(getattr(service, "allowed_workspace_roots", [])),
            "recent_workspaces": _recent_workspaces(service, runs),
            "agents": agents,
            "agent_label": catalog.agent_name_of,
            "agent_teams": catalog.agent_teams(agents),
            "default_memagent": default_memagent(service, agents),
            "model_choices": await asyncio.to_thread(
                catalog.model_choices, service, harnesses
            ),
            "runs": runs,
            "approvals": approvals,
            "monitor": monitor,
            "generated_at": now,
            "activity": _activity(runs, workflows, approvals),
            **dash.template_helpers(),
            "ui_read_only": ui_read_only(),
            "error": error,
        },
    )


STEP_TYPES = ("message", "command", "tool_call", "file_change")


def _step_counts(service, runs) -> Dict[str, Dict[str, int]]:
    """Messages, commands, tool calls and file changes per ledger run."""
    run_ids = [str(run.get("run_id") or "") for run in runs]
    store = getattr(service, "run_store", None)
    counter = getattr(store, "event_counts", None)
    try:
        if callable(counter):
            return counter(run_ids)
        counts: Dict[str, Dict[str, int]] = {}
        for run_id in run_ids[:RUN_LIMIT]:
            seen: Dict[str, set] = {}
            for event in service.events(run_id, limit=2000):
                data = event.get("data") or {}
                seen.setdefault(event.get("type"), set()).add(
                    str(data.get("id") or event.get("sequence"))
                )
            counts[run_id] = {kind: len(items) for kind, items in seen.items()}
        return counts
    except Exception:
        logger.debug("Harness step counts unavailable", exc_info=True)
        return {}


def _recent_workspaces(service, runs, limit: int = 8) -> list:
    """Project folders of recent runs, for the launch form's suggestions."""
    scratch = getattr(service, "_scratch_root", None)
    scratch_root = str(scratch().resolve()) if callable(scratch) else ""
    found: list = []
    for run in runs:
        workspace = str((run.get("task") or {}).get("workspace") or "")
        if not workspace or workspace in found:
            continue
        if scratch_root and workspace.startswith(scratch_root.rstrip("/") + "/"):
            continue
        found.append(workspace)
        if len(found) >= limit:
            break
    return found


def _workflows_with_runs(service, runs):
    """Recent plans and comparisons, plus every run they reference (a stage
    run can be older than the ledger page)."""
    list_workflows = getattr(service, "list_orchestrations", None)
    if not callable(list_workflows):
        return [], {}
    try:
        workflows = list_workflows(limit=WORKFLOW_LIMIT)
    except TypeError:
        return [], {}
    by_id = {str(run.get("run_id")): run for run in runs}
    for workflow in workflows:
        for step in workflow.get("steps") or []:
            run_id = str(step.get("run_id") or "")
            if run_id and run_id not in by_id:
                run = service.get_run(run_id)
                if run is not None:
                    by_id[run_id] = run
    return workflows, by_id


# ---------------------------------------------------------------------------
# Harness chat: keep talking to a harness with a run's setup.
# ---------------------------------------------------------------------------


def _turn_view(
    run: Dict[str, Any], counts: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    """One conversation turn for the chat page, JSON-safe."""
    from ..execution_view import shape_harness_run

    row = shape_harness_run(run)
    metadata = dict((run.get("task") or {}).get("metadata") or {})
    found = counts.get(row["run_id"]) or {}
    steps = {kind: found.get(kind, 0) for kind in STEP_TYPES}
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "status_label": row["status_label"],
        "group": row["group"],
        "signal": row["signal"],
        "harness": row["harness_label"],
        "model": row["model"],
        "message": str(metadata.get("conversation_message") or row["task"]),
        "answer": row["final_response"],
        "error": row["error"],
        "remediation": row["remediation"],
        "cost_usd": row["cost_usd"],
        "cost_estimated": row["cost_estimated"],
        "duration_ms": row["duration_ms"],
        "tokens": row["tokens"],
        "started_at": row["started_at"].timestamp() if row["started_at"] else None,
        "approval_id": row["approval_id"],
        "steps": steps,
        "step_total": sum(steps.values()),
    }


@router.get("/harnesses/chat", response_class=HTMLResponse)
async def harness_chat_page(request: Request, conversation: Optional[str] = None):
    """Continue a run as a conversation, with its setup carried over."""
    if not _state.get("provider"):
        return RedirectResponse(url="/connect", status_code=302)
    service = _service()
    origin = request.query_params.get("from")
    if not conversation and origin:
        run = await asyncio.to_thread(service.get_run, origin)
        if run is None:
            raise HTTPException(status_code=404, detail="Harness run not found")
        conversation = service.conversation_id_for(run)
    if not conversation:
        return RedirectResponse(url="/harnesses", status_code=302)
    runs = await asyncio.to_thread(service.conversation, conversation)
    if not runs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    try:
        agents = list(_list_agents() or [])
    except Exception:
        agents = []
    harnesses = await asyncio.to_thread(service.list_harnesses)
    counts = await asyncio.to_thread(_step_counts, service, runs)
    return templates.TemplateResponse(
        "harness_chat.html",
        {
            "request": request,
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "agents_nav": _build_agent_nav_items(agents=agents),
            "active_agent_id": None,
            "active_page": "harnesses",
            "conversation_id": conversation,
            "first_run": runs[0],
            "turns": [_turn_view(run, counts) for run in runs],
            "setup": catalog.chat_setup(runs[-1]),
            "harnesses": harnesses,
            "agents": agents,
            "agent_label": catalog.agent_name_of,
            "default_memagent": default_memagent(service, agents),
            "ui_read_only": ui_read_only(),
        },
    )


@router.get("/api/harness-conversations/{conversation_id}")
async def api_harness_conversation(conversation_id: str):
    service = _service()
    runs = await asyncio.to_thread(service.conversation, conversation_id)
    if not runs:
        raise HTTPException(status_code=404, detail="Conversation not found")
    counts = await asyncio.to_thread(_step_counts, service, runs)
    turns = [_turn_view(run, counts) for run in runs]
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "turns": turns,
        "active": any(turn["group"] in {"active", "approval"} for turn in turns),
    }


@router.delete("/api/harness-conversations/{conversation_id}")
async def api_delete_harness_conversation(
    conversation_id: str, keep_scratch: bool = False
):
    """Delete a conversation's finished turns; turns still working are kept."""
    try:
        return await asyncio.to_thread(
            _service().delete_conversation,
            conversation_id,
            remove_scratch=not keep_scratch,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-conversations/{conversation_id}/turns")
async def api_harness_conversation_turn(conversation_id: str, request: Request):
    """Send the next message: a new run with earlier turns as context."""
    if ui_read_only():
        raise HTTPException(status_code=403, detail="This console is read-only")
    payload = await _json_object(request)
    try:
        if not str(payload.get("task") or "").strip():
            raise ValueError("Write a message")
        service = _service()
        if not await asyncio.to_thread(service.conversation, conversation_id):
            raise KeyError("Conversation not found")
        payload, _ = await asyncio.to_thread(
            _with_memagent, payload, service, [payload.get("harness")]
        )
        payload = await asyncio.to_thread(_with_workspace, payload, service)
        task = _task_from_payload(payload, write=bool(payload.get("write")))
        run = await asyncio.to_thread(
            service.continue_conversation, conversation_id, task
        )
        return _run_response(service, run)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/api/harnesses")
async def api_harnesses():
    service = _service()
    values = await asyncio.to_thread(service.list_harnesses)
    return {"ok": True, "harnesses": values, "count": len(values)}


def _task_from_payload(payload: Dict[str, Any], *, write: bool):
    """The launch form's fields as one HarnessTask."""
    from ...metaharness.requests import harness_task

    schema = payload.get("output_schema")
    if isinstance(schema, str) and schema.strip():
        try:
            schema = json.loads(schema)
        except ValueError as exc:
            raise ValueError("Output schema must be valid JSON") from exc
    return harness_task(
        payload.get("task") or "",
        payload.get("workspace") or "",
        harness=payload.get("harness") or "auto",
        model=payload.get("model"),
        agent_id=payload.get("agent_id"),
        memory_id=payload.get("memory_id"),
        user_id=payload.get("user_id"),
        thread_id=payload.get("thread_id"),
        write=write,
        allow_dirty_workspace=bool(payload.get("allow_dirty_workspace", False)),
        network=payload.get("network") or "none",
        mcp_access=payload.get("mcp_access") or "read_only",
        allowed_env=payload.get("allowed_env"),
        allowed_tools=payload.get("allowed_tools"),
        denied_tools=payload.get("denied_tools"),
        allow_subagents=bool(payload.get("allow_subagents", False)),
        timeout_seconds=payload.get("timeout_seconds", 900),
        max_steps=payload.get("max_steps", 80),
        max_cost_usd=payload.get("max_cost_usd"),
        max_input_tokens=payload.get("max_input_tokens"),
        max_output_tokens=payload.get("max_output_tokens"),
        verification_command=payload.get("verification_command"),
        output_schema=schema if isinstance(schema, dict) else None,
        metadata={
            "execution_backend": str(
                payload.get("execution_backend") or "local"
            ).lower(),
            "approval_owner_id": "ui:operator",
            "source": "ui",
        },
    )


def default_memagent(
    service, agents: Optional[list] = None
) -> Optional[Dict[str, str]]:
    """The saved MemAgent to use when a memagent run names none."""
    if agents is None:
        try:
            agents = list(_list_agents() or [])
        except Exception:
            agents = []
    return catalog.default_memagent(service, agents)


def _with_memagent(payload: Dict[str, Any], service, harnesses: list) -> tuple:
    """Fill in a saved MemAgent when memagent would otherwise fail to start."""
    return catalog.with_memagent(payload, harnesses, lambda: default_memagent(service))


def _with_workspace(payload: Dict[str, Any], service) -> Dict[str, Any]:
    """A blank workspace means "no project": use a fresh scratch folder."""
    if str(payload.get("workspace") or "").strip():
        return payload
    make = getattr(service, "scratch_workspace", None)
    if not callable(make):
        raise ValueError("Enter the workspace folder for this run")
    return {**payload, "workspace": make()}


def _run_response(service, run) -> Any:
    """A started run as the API reports it: failed, awaiting approval or queued."""
    run_value = run.to_dict()
    if run.status.value == "failed":
        result = dict(run_value.get("result") or {})
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "status": "failed",
                "run": run_value,
                "error": {
                    "code": result.get("error_code") or "harness_failed",
                    "message": result.get("error") or "Harness startup failed",
                    "remediation": result.get("remediation"),
                    "run_id": run.run_id,
                },
            },
        )
    response: Dict[str, Any] = {"ok": True, "run": run_value}
    if run.approval_proposal_id:
        proposal = service.approval_store.get(run.approval_proposal_id)
        response.update(
            ok=False,
            status="approval_required",
            proposal=proposal.to_dict(include_arguments=True) if proposal else None,
        )
    return response


@router.post("/api/harness-orchestrations")
async def api_start_harness_orchestration(request: Request):
    """Start a staged plan (``kind: plan``) or a comparison (``kind: compare``)."""
    payload = await _json_object(request)
    kind = str(payload.get("kind") or "").strip().lower()
    try:
        if kind not in {"plan", "compare"}:
            raise ValueError("kind must be plan or compare")
        goal = str(payload.get("task") or "").strip()
        if not goal:
            raise ValueError("Describe the task")
        # Validate before a scratch folder is created for a blank workspace.
        stages = None
        if kind == "plan":
            from ...metaharness.requests import plan_stages

            rows = payload.get("stages")
            stages = plan_stages(rows if isinstance(rows, list) else [], goal)
        harnesses = payload.get("harnesses")
        if kind == "compare" and not isinstance(harnesses, list):
            raise ValueError("harnesses must be a list of harness names")
        service = _service()
        involved = (
            list(harnesses)
            if kind == "compare"
            else [
                stage["harness"] for stage in stages or [] if not stage.get("agent_id")
            ]
        )
        payload, chosen = await asyncio.to_thread(
            _with_memagent, payload, service, involved
        )
        payload = await asyncio.to_thread(_with_workspace, payload, service)
        base = _task_from_payload(payload, write=False)
        if kind == "plan":
            value = await asyncio.to_thread(service.start_plan, base, stages)
        else:
            models = payload.get("harness_models")
            value = await asyncio.to_thread(
                service.start_compare,
                base,
                harnesses,
                models if isinstance(models, dict) else None,
            )
        return {"ok": True, "orchestration": value, "memagent_agent": chosen}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/api/harness-orchestrations")
async def api_harness_orchestrations(status: Optional[str] = None, limit: int = 50):
    service = _service()
    values = await asyncio.to_thread(
        service.list_orchestrations,
        status=status,
        limit=max(1, min(int(limit), 500)),
    )
    return {"ok": True, "orchestrations": values, "count": len(values)}


@router.get("/api/harness-orchestrations/{orchestration_id}")
async def api_harness_orchestration(orchestration_id: str):
    service = _service()
    value = await asyncio.to_thread(service.get_orchestration, orchestration_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Harness workflow not found")
    runs = []
    for step in value.get("steps") or []:
        if step.get("run_id"):
            run = await asyncio.to_thread(service.get_run, step["run_id"])
            if run is not None:
                runs.append(run)
    return {"ok": True, "orchestration": value, "runs": runs}


@router.post("/api/harness-orchestrations/{orchestration_id}/rerun")
async def api_rerun_harness_orchestration(orchestration_id: str):
    """Start a plan or comparison again with the settings it ran with."""
    service = _service()
    value = await asyncio.to_thread(service.get_orchestration, orchestration_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Harness workflow not found")
    try:
        task = dict(value.get("task") or {})
        steps = [dict(step) for step in value.get("steps") or []]
        involved = (
            [step.get("harness") for step in steps]
            if value.get("kind") == "compare"
            else [step.get("harness") for step in steps if not step.get("agent_id")]
        )
        # The first run may have failed for want of a saved agent: fill one in.
        filled, chosen = await asyncio.to_thread(
            _with_memagent, {"agent_id": task.get("agent_id")}, service, involved
        )
        started = await asyncio.to_thread(
            service.rerun_orchestration,
            orchestration_id,
            agent_id=filled.get("agent_id") if chosen else None,
        )
        return {"ok": True, "orchestration": started, "memagent_agent": chosen}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-orchestrations/{orchestration_id}/cancel")
async def api_cancel_harness_orchestration(orchestration_id: str):
    try:
        return await asyncio.to_thread(
            _service().cancel_orchestration, orchestration_id
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.delete("/api/harness-orchestrations/{orchestration_id}")
async def api_delete_harness_orchestration(
    orchestration_id: str, keep_scratch: bool = False
):
    """Delete a finished plan or comparison with all its runs."""
    try:
        return await asyncio.to_thread(
            _service().delete_orchestration,
            orchestration_id,
            remove_scratch=not keep_scratch,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/api/harness-activity")
async def api_harness_activity():
    """A cheap change signal for the page's live refresh."""
    service = _service()

    def snapshot():
        runs = service.list_runs(limit=RUN_LIMIT)
        workflows = []
        if callable(getattr(service, "list_orchestrations", None)):
            workflows = service.list_orchestrations(limit=WORKFLOW_LIMIT)
        approvals = service.list_approvals(status="pending", limit=100)
        return runs, workflows, approvals

    runs, workflows, approvals = await asyncio.to_thread(snapshot)
    return {"ok": True, **_activity(runs, workflows, approvals)}


def _activity(runs, workflows, approvals) -> Dict[str, Any]:
    """Fingerprint of what the page shows, and whether anything is running."""
    marks = (
        [[r.get("run_id"), r.get("status"), r.get("updated_at")] for r in runs]
        + [
            [w.get("orchestration_id"), w.get("status"), w.get("updated_at")]
            for w in workflows
        ]
        + [[a.get("proposal_id"), a.get("status")] for a in approvals]
    )
    active_states = {"queued", "running", "pending_approval"}
    return {
        "fingerprint": hashlib.sha256(
            json.dumps(marks, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        "active": any(r.get("status") in active_states for r in runs)
        or any(w.get("status") in active_states for w in workflows),
    }


@router.post("/api/harness-runs")
async def api_start_harness_run(request: Request):
    payload = await _json_object(request)
    try:
        if not str(payload.get("task") or "").strip():
            raise ValueError("Describe the task")
        service = _service()
        payload, _ = await asyncio.to_thread(
            _with_memagent, payload, service, [payload.get("harness")]
        )
        payload = await asyncio.to_thread(_with_workspace, payload, service)
        task = _task_from_payload(payload, write=bool(payload.get("write")))
        run = await asyncio.to_thread(service.start, task)
        return _run_response(service, run)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-runs/{run_id}/retry")
async def api_retry_harness_run(run_id: str):
    """Start a finished run's exact task again."""
    try:
        service = _service()
        run = await asyncio.to_thread(service.retry_start, run_id)
        return _run_response(service, run)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/api/harness-runs")
async def api_harness_runs(status: Optional[str] = None, limit: int = 100):
    service = _service()
    values = await asyncio.to_thread(
        service.list_runs, status=status, limit=max(1, min(int(limit), 1000))
    )
    return {"ok": True, "runs": values, "count": len(values)}


@router.get("/api/harness-runs/{run_id}")
async def api_harness_run(run_id: str, include_events: bool = False):
    service = _service()
    value = await asyncio.to_thread(service.get_run, run_id)
    if value is None:
        raise HTTPException(status_code=404, detail="Harness run not found")
    if include_events:
        value["events"] = await asyncio.to_thread(service.events, run_id)
    return {"ok": True, "run": value}


@router.get("/api/harness-runs/{run_id}/events")
async def api_harness_events(run_id: str, after: int = 0, limit: int = 1000):
    service = _service()
    if await asyncio.to_thread(service.get_run, run_id) is None:
        raise HTTPException(status_code=404, detail="Harness run not found")
    values = await asyncio.to_thread(
        service.events,
        run_id,
        after=max(0, int(after)),
        limit=max(1, min(int(limit), 10_000)),
    )
    return {"ok": True, "events": values, "count": len(values)}


@router.delete("/api/harness-runs/{run_id}")
async def api_delete_harness_run(run_id: str, keep_scratch: bool = False):
    """Delete a finished run, its events and its delegates' runs."""
    try:
        return await asyncio.to_thread(
            _service().delete_run, run_id, remove_scratch=not keep_scratch
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-runs/delete")
async def api_delete_harness_runs(request: Request):
    """Delete several finished runs; runs still working and workflow steps
    are kept and listed with the reason."""
    payload = await _json_object(request)
    run_ids = payload.get("run_ids")
    if not isinstance(run_ids, list) or not run_ids:
        raise HTTPException(
            status_code=400, detail="List the runs to delete in run_ids"
        )
    try:
        return await asyncio.to_thread(
            _service().delete_runs,
            [str(run_id) for run_id in run_ids[:1000]],
            remove_scratch=not bool(payload.get("keep_scratch")),
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-runs/{run_id}/cancel")
async def api_cancel_harness_run(run_id: str):
    try:
        return await asyncio.to_thread(_service().cancel, run_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-approvals/{proposal_id}/approve")
async def api_approve_harness_run(proposal_id: str, request: Request):
    decision = await _approval_payload(request)
    try:
        proposal = await asyncio.to_thread(_service().approve, proposal_id, **decision)
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-approvals/{proposal_id}/reject")
async def api_reject_harness_run(proposal_id: str, request: Request):
    decision = await _approval_payload(request)
    try:
        proposal = await asyncio.to_thread(_service().reject, proposal_id, **decision)
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-approvals/{proposal_id}/resume")
async def api_resume_harness_run(proposal_id: str):
    try:
        run = await asyncio.to_thread(_service().resume_approval_start, proposal_id)
        return {"ok": True, "run": run.to_dict()}
    except Exception as exc:
        raise _http_error(exc) from exc


__all__ = ["router"]
