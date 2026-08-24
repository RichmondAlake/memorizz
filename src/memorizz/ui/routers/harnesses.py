"""Operator UI and JSON API for the durable MemoRizz meta-harness."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..helpers import _build_agent_nav_items
from ..security import ui_read_only
from ..state import _state, get_meta_harness, templates

logger = logging.getLogger(__name__)
router = APIRouter(tags=["meta-harness"])


def _service():
    try:
        return get_meta_harness()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _json_object(request: Request) -> Dict[str, Any]:
    try:
        value = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be JSON"
        ) from exc
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    return value


def _approver(value: Dict[str, Any]) -> tuple[str, Optional[str]]:
    approver_id = str(value.get("approver_id") or "").strip()
    if not approver_id:
        raise HTTPException(status_code=400, detail="approver_id is required")
    reason = str(value.get("reason") or "").strip() or None
    return approver_id, reason


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
        agents = list(_state["provider"].list_memagents() or [])
    except Exception:
        agents = []
    try:
        harnesses = await asyncio.to_thread(service.list_harnesses)
        runs = await asyncio.to_thread(service.list_runs, limit=100)
        approvals = await asyncio.to_thread(
            service.list_approvals, status="pending", limit=100
        )
        error = None
    except Exception as exc:
        harnesses, runs, approvals = [], [], []
        error = str(exc)
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
            "agents": agents,
            "runs": runs,
            "approvals": approvals,
            "ui_read_only": ui_read_only(),
            "error": error,
        },
    )


@router.get("/api/harnesses")
async def api_harnesses():
    service = _service()
    values = await asyncio.to_thread(service.list_harnesses)
    return {"ok": True, "harnesses": values, "count": len(values)}


@router.post("/api/harness-runs")
async def api_start_harness_run(request: Request):
    payload = await _json_object(request)
    try:
        from ...metaharness import (
            HarnessBudget,
            HarnessPermissions,
            HarnessTask,
            VerificationSpec,
        )

        permissions = HarnessPermissions(
            workspace_mode="direct" if bool(payload.get("write")) else "read_only",
            allow_dirty_workspace=bool(payload.get("allow_dirty_workspace", False)),
            network=str(payload.get("network") or "none"),
            mcp_access=str(payload.get("mcp_access") or "read_only"),
            allowed_env=list(payload.get("allowed_env") or []),
        )
        task = HarnessTask(
            task=str(payload.get("task") or ""),
            workspace=str(payload.get("workspace") or ""),
            harness=str(payload.get("harness") or "auto"),
            model=str(payload.get("model") or "").strip() or None,
            memory_id=str(payload.get("memory_id") or "").strip() or None,
            user_id=str(payload.get("user_id") or "").strip() or None,
            thread_id=str(payload.get("thread_id") or "").strip() or None,
            agent_id=str(payload.get("agent_id") or "").strip() or None,
            permissions=permissions,
            budget=HarnessBudget(
                max_wall_time_seconds=payload.get("timeout_seconds", 900),
                max_steps=payload.get("max_steps", 80),
                max_cost_usd=payload.get("max_cost_usd"),
                max_input_tokens=payload.get("max_input_tokens"),
                max_output_tokens=payload.get("max_output_tokens"),
            ),
            verification=VerificationSpec(
                command=str(payload.get("verification_command") or "").strip() or None
            ),
            metadata={
                "execution_backend": str(
                    payload.get("execution_backend") or "local"
                ).lower(),
                "approval_owner_id": "ui:operator",
                "source": "ui",
            },
        )
        service = _service()
        run = await asyncio.to_thread(service.start, task)
        run_value = run.to_dict()
        response: Dict[str, Any] = {"ok": True, "run": run_value}
        if run.status.value == "failed":
            result = dict(run_value.get("result") or {})
            error = {
                "code": result.get("error_code") or "harness_failed",
                "message": result.get("error") or "Harness startup failed",
                "remediation": result.get("remediation"),
                "run_id": run.run_id,
            }
            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "status": "failed",
                    "run": run_value,
                    "error": error,
                },
            )
        if run.approval_proposal_id:
            proposal = service.approval_store.get(run.approval_proposal_id)
            response.update(
                ok=False,
                status="approval_required",
                proposal=(
                    proposal.to_dict(include_arguments=True) if proposal else None
                ),
            )
        return response
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


@router.post("/api/harness-runs/{run_id}/cancel")
async def api_cancel_harness_run(run_id: str):
    try:
        return await asyncio.to_thread(_service().cancel, run_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-approvals/{proposal_id}/approve")
async def api_approve_harness_run(proposal_id: str, request: Request):
    payload = await _json_object(request)
    approver_id, reason = _approver(payload)
    try:
        proposal = await asyncio.to_thread(
            _service().approve,
            proposal_id,
            approver_id=approver_id,
            reason=reason,
        )
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/api/harness-approvals/{proposal_id}/reject")
async def api_reject_harness_run(proposal_id: str, request: Request):
    payload = await _json_object(request)
    approver_id, reason = _approver(payload)
    try:
        proposal = await asyncio.to_thread(
            _service().reject,
            proposal_id,
            approver_id=approver_id,
            reason=reason,
        )
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
