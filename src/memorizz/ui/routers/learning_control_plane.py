"""Small operator surface for learning events, compilation, and forgetting."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...learning import LearningControlPlane
from ..helpers import _build_agent_nav_items, _extract_agent_identifier
from ..security import ui_read_only
from ..state import _state, templates

logger = logging.getLogger(__name__)
router = APIRouter(tags=["learning-control-plane"])
_last_actions: Dict[str, Dict[str, Any]] = {}


def _plane(agent_id: str) -> LearningControlPlane:
    return LearningControlPlane(
        _state["provider"],
        agent_id=agent_id,
        config={"enabled": True, "compile_async": False},
    )


def _redirect(agent_id: str, **scope: Any) -> RedirectResponse:
    query = {"agent_id": agent_id}
    query.update(
        {key: value for key, value in scope.items() if value not in (None, "")}
    )
    return RedirectResponse(
        url="/learning-control-plane?" + urlencode(query), status_code=303
    )


@router.get("/learning-control-plane", response_class=HTMLResponse)
async def learning_control_plane_page(
    request: Request,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        agents = _state["provider"].list_memagents() or []
    except Exception:
        agents = []
    known_ids = [
        value
        for value in (_extract_agent_identifier(agent) for agent in agents)
        if value
    ]
    selected = (
        agent_id if agent_id in known_ids else (known_ids[0] if known_ids else agent_id)
    )
    report = None
    events = []
    artifacts = []
    error = None
    if selected:
        plane = _plane(selected)
        try:
            report = plane.report(
                memory_id=memory_id, user_id=user_id, thread_id=thread_id
            )
            events = plane.store.list_events(
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                limit=100,
            )
            artifacts = plane.store.list_artifacts(
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                limit=100,
            )
        except Exception as exc:
            logger.error("Learning control-plane page failed: %s", exc)
            error = str(exc)
        finally:
            plane.close()
    event_rows = [
        {
            "event_id": item.event_id,
            "event_type": item.event_type.value,
            "timestamp": item.timestamp,
            "run_id": item.run_id,
            "workflow_id": item.workflow_id,
        }
        for item in reversed(events)
    ]
    artifact_rows = [
        {
            "artifact_id": item.get("artifact_id") or item.get("record_id"),
            "kind": item.get("artifact_kind"),
            "verified": bool(item.get("verified")),
            "updated_at": item.get("updated_at") or item.get("timestamp"),
            "utility": item.get("utility"),
        }
        for item in reversed(artifacts)
    ]
    return templates.TemplateResponse(
        "learning_control_plane.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents_nav": _build_agent_nav_items(
                active_agent_id=selected, agents=agents
            ),
            "agents": agents,
            "known_agent_ids": known_ids,
            "active_agent_id": selected,
            "selected_agent_id": selected,
            "memory_id": memory_id or "",
            "user_id": user_id or "",
            "thread_id": thread_id or "",
            "report": report,
            "events": event_rows,
            "artifacts": artifact_rows,
            "action": _last_actions.get(selected or ""),
            "active_page": "learning-control-plane",
            "ui_read_only": ui_read_only(),
            "error": error,
        },
    )


@router.post("/learning-control-plane/compile")
async def compile_memory(
    agent_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        report = plane.compile(
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        ).to_dict()
        _last_actions[agent_id] = {
            "kind": "compile",
            "message": f"Compiled {report['compiled_events']} event(s)",
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/forget-plan")
async def plan_forgetting(
    agent_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        report = plane.plan_forgetting(
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        ).to_dict()
        _last_actions[agent_id] = {
            "kind": "forget-plan",
            "message": f"Dry run found {report['candidate_count']} candidate(s)",
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/forget-apply")
async def apply_forgetting(
    agent_id: str = Form(...),
    plan_id: str = Form(...),
    approved_by: str = Form(...),
    reason: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        plan = plane.get_forgetting_plan(
            plan_id,
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        )
        report = plane.apply_forgetting(
            plan,
            approved_by=approved_by,
            reason=reason,
            scope={
                "memory_id": memory_id or None,
                "user_id": user_id or None,
                "thread_id": thread_id or None,
            },
        ).to_dict()
        _last_actions[agent_id] = {
            "kind": "forget-apply",
            "message": f"Tombstoned {report['tombstoned']} artifact(s)",
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


__all__ = ["router"]
