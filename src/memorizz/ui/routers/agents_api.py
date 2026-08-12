# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Read-only JSON API: connection status + agent listing/details/system-prompt.

First stateful router extracted from ui/app.py on top of the shared ui.state.
These power the frontend's data fetches; they read the connected provider from
``ui.state._state`` and return plain JSON (no templates). Behavior is verbatim
from the former in-create_app handlers.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..state import _state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agents-api"])


async def _load_runtime_agent(agent_id: str):
    """Load an executable agent for host-only control-plane operations."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    from ...memagent import MemAgent

    try:
        return await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )
    except Exception as exc:
        logger.error("Failed to load agent %s: %s", agent_id, exc)
        raise HTTPException(status_code=404, detail="Agent not found") from exc


async def _approval_decision_payload(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    approver_id = str(payload.get("approver_id") or "").strip()
    if not approver_id:
        raise HTTPException(status_code=400, detail="approver_id is required")
    return {
        "approver_id": approver_id,
        "reason": str(payload.get("reason") or "").strip() or None,
    }


def _serialize_agent(agent) -> Dict[str, Any]:
    """Convert agent object to serializable dict."""
    if hasattr(agent, "model_dump"):
        data = agent.model_dump()
    elif hasattr(agent, "__dict__"):
        data = dict(agent.__dict__)
    else:
        data = {"agent_id": str(agent)}

    # Handle persona serialization
    if "persona" in data and data["persona"] and hasattr(data["persona"], "to_dict"):
        data["persona"] = data["persona"].to_dict()

    # Remove non-serializable items
    data.pop("memory_provider", None)
    data.pop("model", None)

    return data


@router.get("/status")
async def api_status():
    """Get current connection status."""
    return {
        "connected": _state["provider"] is not None,
        "provider_type": _state["provider_type"],
        "connection_info": _state["connection_info"],
    }


@router.get("/capabilities")
async def api_capabilities():
    """Expose the stable deployment feature report to the local UI."""
    from ...capabilities import capabilities

    report = capabilities()
    provider = _state.get("provider")
    report["runtime"] = {
        "connected": provider is not None,
        "provider_type": _state.get("provider_type"),
    }
    return report


@router.get("/agents")
async def api_list_agents():
    """API endpoint to list all agents."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    agents = []
    try:
        raw_agents = _state["provider"].list_memagents()
        for agent in raw_agents:
            agents.append(_serialize_agent(agent))
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"agents": agents, "count": len(agents)}


@router.get("/agents/{agent_id}")
async def api_get_agent(agent_id: str):
    """API endpoint to get agent details."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    try:
        agent = _state["provider"].retrieve_memagent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        return _serialize_agent(agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/system-prompt")
async def api_agent_system_prompt(agent_id: str):
    """Return the fully-assembled system prompt for an agent.

    Instantiates the agent via :meth:`MemAgent.load` (no inference is run) and
    calls ``_build_system_prompt`` — the same code path used at request time.
    Used by the playground to preview what the LLM actually sees, including
    persona + version + evolution history + tool descriptions.
    """
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    from ...memagent import MemAgent

    try:
        agent = await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )
    except Exception as exc:
        logger.error("Failed to load agent %s for system-prompt: %s", agent_id, exc)
        raise HTTPException(status_code=404, detail="Agent not found")

    try:
        prompt = agent._build_system_prompt() or ""
    except Exception as exc:
        logger.error("Failed to build system prompt for %s: %s", agent_id, exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to build system prompt: {exc}"
        )

    persona = agent.persona_manager.current_persona if agent.persona_manager else None
    persona_meta: Optional[Dict[str, Any]] = None
    if persona is not None:
        persona_meta = {
            "name": getattr(persona, "name", None),
            "role": getattr(persona, "role", None),
            "version": getattr(persona, "version", 1),
            "history_count": len(getattr(persona, "evolution_history", []) or []),
            "storage_id": getattr(persona, "_storage_id", None),
        }

    tool_names: List[str] = []
    if agent.tool_manager is not None:
        try:
            tool_names = [
                (meta or {}).get("name", "")
                for meta in (agent.tool_manager.get_tool_metadata() or [])
            ]
            tool_names = [n for n in tool_names if n]
        except Exception:
            tool_names = []

    return {
        "agent_id": agent_id,
        "system_prompt": prompt,
        "length": len(prompt),
        "persona": persona_meta,
        "persona_tools_registered": bool(
            getattr(agent, "_persona_tools_registered", False)
        ),
        "tools": tool_names,
    }


@router.get("/agents/{agent_id}/capabilities")
async def api_agent_capabilities(agent_id: str, preflight: bool = False):
    """Return effective agent capabilities, optionally including provider preflight."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")
    from ...memagent import MemAgent

    try:
        agent = await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )
        return await run_in_threadpool(agent.capability_report, preflight=preflight)
    except Exception as exc:
        logger.error("Failed to build capability report for %s: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/agents/{agent_id}/approvals")
async def api_agent_approvals(
    agent_id: str, status: Optional[str] = None, limit: int = 100
):
    """List durable generic tool proposals for a local host operator."""
    agent = await _load_runtime_agent(agent_id)
    try:
        proposals = await run_in_threadpool(
            agent.list_approval_proposals,
            status=status,
            limit=max(1, min(int(limit), 500)),
        )
        return {"ok": True, "count": len(proposals), "approvals": proposals}
    except Exception as exc:
        logger.error("Failed to list approvals for %s: %s", agent_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agents/{agent_id}/approvals/{proposal_id}/approve")
async def api_agent_approve(agent_id: str, proposal_id: str, request: Request):
    """Approve one exact proposal; execution remains paused until resume."""
    decision = await _approval_decision_payload(request)
    agent = await _load_runtime_agent(agent_id)
    try:
        proposal = await run_in_threadpool(agent.approve, proposal_id, **decision)
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        logger.error("Failed to approve proposal %s: %s", proposal_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agents/{agent_id}/approvals/{proposal_id}/reject")
async def api_agent_reject(agent_id: str, proposal_id: str, request: Request):
    """Reject a pending generic tool proposal without executing it."""
    decision = await _approval_decision_payload(request)
    agent = await _load_runtime_agent(agent_id)
    try:
        proposal = await run_in_threadpool(agent.reject, proposal_id, **decision)
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        logger.error("Failed to reject proposal %s: %s", proposal_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agents/{agent_id}/approvals/{proposal_id}/cancel")
async def api_agent_cancel(agent_id: str, proposal_id: str, request: Request):
    """Cancel a pending proposal (explicit host-facing rejection alias)."""
    decision = await _approval_decision_payload(request)
    agent = await _load_runtime_agent(agent_id)
    try:
        proposal = await run_in_threadpool(
            agent.cancel_approval, proposal_id, **decision
        )
        return {"ok": True, "proposal": proposal}
    except Exception as exc:
        logger.error("Failed to cancel proposal %s: %s", proposal_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/agents/{agent_id}/approvals/{proposal_id}/resume")
async def api_agent_resume(
    agent_id: str,
    proposal_id: str,
    continue_model: bool = True,
):
    """Consume an approval and resume its serialized checkpoint exactly once."""
    agent = await _load_runtime_agent(agent_id)
    try:
        result = await (
            run_in_threadpool(agent.resume_approval, proposal_id)
            if continue_model
            else run_in_threadpool(
                agent.resume_approval,
                proposal_id,
                continue_model=False,
            )
        )
        payload = result.to_dict() if hasattr(result, "to_dict") else result
        return {"ok": bool(getattr(result, "ok", True)), "result": payload}
    except Exception as exc:
        logger.error("Failed to resume proposal %s: %s", proposal_id, exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
