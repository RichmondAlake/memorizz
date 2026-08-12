# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""MCP connection management UI and JSON API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...mcp import MCPClientError, MCPClientManager
from ..helpers import (
    _build_agent_nav_items,
    _extract_agent_identifier,
    _extract_agent_persona_name,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["mcp"])


def _require_provider():
    provider = _state.get("provider")
    if not provider:
        raise HTTPException(status_code=400, detail="Memory provider is not connected")
    return provider


def _load_agent(agent_id: str):
    provider = _require_provider()
    try:
        agent = provider.retrieve_memagent(agent_id)
    except Exception as exc:
        logger.error("Failed to retrieve agent %s for MCP: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load agent") from exc
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def _agent_servers(agent: Any) -> List[Dict[str, Any]]:
    if isinstance(agent, dict):
        value = agent.get("mcp_servers")
    else:
        value = getattr(agent, "mcp_servers", None)
    return list(value) if isinstance(value, list) else []


def _manager(agent_id: str, agent: Any) -> MCPClientManager:
    return MCPClientManager(owner_id=agent_id, servers=_agent_servers(agent))


def _save_servers(agent: Any, servers: List[Dict[str, Any]]) -> None:
    provider = _require_provider()
    if isinstance(agent, dict):
        agent["mcp_servers"] = servers
    else:
        agent.mcp_servers = servers
    try:
        if hasattr(provider, "update_memagent"):
            provider.update_memagent(agent)
        else:
            provider.store_memagent(agent)
    except Exception as exc:
        logger.error("Failed to persist MCP server configuration: %s", exc)
        raise HTTPException(
            status_code=500, detail="Failed to persist MCP configuration"
        ) from exc
    try:
        from ...cli.mcp_config import save_servers

        owner_id = _extract_agent_identifier(agent)
        if owner_id:
            save_servers(servers, owner_id)
    except Exception as exc:
        logger.warning("Could not sync the shared MCP configuration: %s", exc)


def _raise_client_error(exc: Exception) -> None:
    if isinstance(exc, MCPClientError):
        raise HTTPException(status_code=400, detail=exc.to_dict()) from exc
    raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _manager_call(function, *args, **kwargs):
    try:
        return await asyncio.to_thread(function, *args, **kwargs)
    except Exception as exc:
        _raise_client_error(exc)


@router.get("/mcp", response_class=HTMLResponse)
async def mcp_page(request: Request, agent_id: Optional[str] = None):
    if not _state.get("provider"):
        return RedirectResponse(url="/connect", status_code=302)
    try:
        agents = list(_state["provider"].list_memagents() or [])
    except Exception as exc:
        logger.error("Failed to list agents for MCP page: %s", exc)
        agents = []

    selected_id = str(agent_id or "").strip()
    if not selected_id and agents:
        selected_id = _extract_agent_identifier(agents[0]) or ""
    selected_agent = _load_agent(selected_id) if selected_id else None
    manager = _manager(selected_id, selected_agent) if selected_agent else None
    statuses = manager.connection_status().get("servers", []) if manager else []
    agent_rows = [
        {
            "agent_id": _extract_agent_identifier(agent),
            "name": _extract_agent_persona_name(agent),
        }
        for agent in agents
        if _extract_agent_identifier(agent)
    ]
    return templates.TemplateResponse(
        "mcp.html",
        {
            "request": request,
            "provider_type": _state.get("provider_type"),
            "connection_info": _state.get("connection_info"),
            "agents_nav": _build_agent_nav_items(
                active_agent_id=selected_id, agents=agents
            ),
            "active_agent_id": selected_id,
            "active_page": "mcp",
            "agents": agent_rows,
            "selected_agent": selected_agent,
            "selected_agent_id": selected_id,
            "servers": manager.server_dicts() if manager else [],
            "server_statuses": statuses,
            "oauth_result": request.query_params.get("oauth"),
            "oauth_server": request.query_params.get("server"),
        },
    )


@router.get("/api/mcp/agents/{agent_id}/servers")
async def api_mcp_servers(agent_id: str):
    agent = _load_agent(agent_id)
    manager = _manager(agent_id, agent)
    return {
        "ok": True,
        "servers": manager.server_dicts(),
        "status": manager.connection_status().get("servers", []),
    }


@router.post("/api/mcp/agents/{agent_id}/servers")
async def api_mcp_upsert_server(agent_id: str, request: Request):
    agent = _load_agent(agent_id)
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("Request body must be an object")
        manager = _manager(agent_id, agent)
        server = manager.upsert_server(payload)
        _save_servers(agent, manager.server_dicts())
        return {"ok": True, "server": server}
    except HTTPException:
        raise
    except Exception as exc:
        _raise_client_error(exc)


@router.delete("/api/mcp/agents/{agent_id}/servers/{server_name}")
async def api_mcp_remove_server(agent_id: str, server_name: str):
    agent = _load_agent(agent_id)
    manager = _manager(agent_id, agent)
    if not manager.remove_server(server_name, delete_credentials=True):
        raise HTTPException(status_code=404, detail="MCP server not found")
    _save_servers(agent, manager.server_dicts())
    return {"ok": True, "removed": server_name}


@router.post("/api/mcp/agents/{agent_id}/servers/{server_name}/test")
async def api_mcp_test_server(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    result = await _manager_call(manager.test_connection, server_name)
    if not result.get("ok"):
        return result
    return result


@router.get("/api/mcp/agents/{agent_id}/servers/{server_name}/tools")
async def api_mcp_list_tools(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.list_tools, server_name)


@router.post("/api/mcp/agents/{agent_id}/servers/{server_name}/call")
async def api_mcp_call_tool(agent_id: str, server_name: str, request: Request):
    manager = _manager(agent_id, _load_agent(agent_id))
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    return await _manager_call(
        manager.call_tool,
        server_name,
        str(payload.get("tool_name") or ""),
        payload.get("arguments") or {},
    )


@router.get("/api/mcp/agents/{agent_id}/approvals")
async def api_mcp_approvals(
    agent_id: str, status: Optional[str] = None, limit: int = 100
):
    manager = _manager(agent_id, _load_agent(agent_id))
    approvals = manager.list_tool_call_approvals(status=status, limit=limit)
    return {"ok": True, "count": len(approvals), "approvals": approvals}


@router.post("/api/mcp/agents/{agent_id}/approvals/{proposal_id}/approve")
async def api_mcp_approve(agent_id: str, proposal_id: str, request: Request):
    manager = _manager(agent_id, _load_agent(agent_id))
    payload = await request.json()
    approver_id = str((payload or {}).get("approver_id") or "").strip()
    if not approver_id:
        raise HTTPException(status_code=400, detail="approver_id is required")
    proposal = manager.approve_tool_call(
        proposal_id,
        approver_id=approver_id,
        reason=(payload or {}).get("reason"),
    )
    return {"ok": True, "proposal": proposal}


@router.post("/api/mcp/agents/{agent_id}/approvals/{proposal_id}/reject")
async def api_mcp_reject(agent_id: str, proposal_id: str, request: Request):
    manager = _manager(agent_id, _load_agent(agent_id))
    payload = await request.json()
    approver_id = str((payload or {}).get("approver_id") or "").strip()
    if not approver_id:
        raise HTTPException(status_code=400, detail="approver_id is required")
    proposal = manager.reject_tool_call(
        proposal_id,
        approver_id=approver_id,
        reason=(payload or {}).get("reason"),
    )
    return {"ok": True, "proposal": proposal}


@router.post("/api/mcp/agents/{agent_id}/approvals/{proposal_id}/resume")
async def api_mcp_resume(agent_id: str, proposal_id: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.resume_tool_call, proposal_id)


@router.get("/api/mcp/agents/{agent_id}/servers/{server_name}/resources")
async def api_mcp_resources(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.list_resources, server_name)


@router.get("/api/mcp/agents/{agent_id}/servers/{server_name}/resource-templates")
async def api_mcp_resource_templates(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.list_resource_templates, server_name)


@router.post("/api/mcp/agents/{agent_id}/servers/{server_name}/resources/read")
async def api_mcp_read_resource(agent_id: str, server_name: str, request: Request):
    manager = _manager(agent_id, _load_agent(agent_id))
    payload = await request.json()
    return await _manager_call(
        manager.read_resource, server_name, str(payload.get("uri") or "")
    )


@router.get("/api/mcp/agents/{agent_id}/servers/{server_name}/prompts")
async def api_mcp_prompts(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.list_prompts, server_name)


@router.post("/api/mcp/agents/{agent_id}/servers/{server_name}/prompts/get")
async def api_mcp_get_prompt(agent_id: str, server_name: str, request: Request):
    manager = _manager(agent_id, _load_agent(agent_id))
    payload = await request.json()
    return await _manager_call(
        manager.get_prompt,
        server_name,
        str(payload.get("prompt_name") or ""),
        payload.get("arguments") or {},
    )


@router.post("/api/mcp/agents/{agent_id}/servers/{server_name}/authorize")
async def api_mcp_authorize(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return await _manager_call(manager.begin_oauth, server_name)


@router.delete("/api/mcp/agents/{agent_id}/servers/{server_name}/credentials")
async def api_mcp_disconnect(agent_id: str, server_name: str):
    manager = _manager(agent_id, _load_agent(agent_id))
    return {
        "ok": True,
        "deleted": await _manager_call(manager.disconnect, server_name),
    }


@router.get("/api/mcp/oauth/callback")
async def api_mcp_oauth_callback(
    state: str = "",
    code: Optional[str] = None,
    iss: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    if not state:
        raise HTTPException(status_code=400, detail="OAuth callback is missing state")
    error_value = error_description or error
    result = await asyncio.to_thread(
        MCPClientManager.complete_oauth_callback,
        state=state,
        code=code,
        issuer=iss,
        error=error_value,
    )
    if not result:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or expired")
    outcome = (
        "error"
        if error_value or not result.get("ok")
        else "pending"
        if result.get("pending")
        else "success"
    )
    return RedirectResponse(
        url=(
            f"/mcp?agent_id={quote(str(result['owner_id']))}"
            f"&oauth={outcome}&server={quote(str(result['server_name']))}"
        ),
        status_code=302,
    )
