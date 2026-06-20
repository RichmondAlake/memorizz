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

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..state import _state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agents-api"])


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
