# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What an agent can't do yet, and one-click ways to enable it.

The playground's enable card reads this state when it renders and after every
action, so it always reflects the agent's current configuration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from ... import capability_gaps
from ...mcp import MCPClientManager, catalog
from ..helpers import (
    _agent_field,
    _build_internet_provider_config,
    _json_object,
    _load_agent,
    _save_agent_fields,
    _validate_internet_provider_choice,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["capabilities"])


def _agent_state(agent_id: str, agent: Any) -> List[Dict[str, Any]]:
    servers = list(_agent_field(agent, "mcp_servers") or [])
    signed_in: Dict[str, bool] = {}
    if servers:
        try:
            manager = MCPClientManager(owner_id=agent_id, servers=servers)
            servers = manager.server_dicts()
            signed_in = {
                str(row.get("name")): bool(row.get("authenticated"))
                for row in manager.connection_status().get("servers") or []
            }
        except Exception as exc:
            logger.debug("MCP status unavailable for %s: %s", agent_id, exc)
    browser = _agent_field(agent, "browser_control")
    if isinstance(browser, dict):
        browser = browser.get("provider") or browser.get("name")
    return capability_gaps.capability_state(
        internet_provider=_agent_field(agent, "internet_access_provider"),
        mcp_servers=servers,
        signed_in=signed_in,
        sandbox_provider=_agent_field(agent, "sandbox_provider"),
        browser_provider=browser,
    )


@router.get("/api/agents/{agent_id}/capabilities")
async def api_agent_capabilities(agent_id: str, ids: Optional[str] = None):
    """Each capability's live status and the ways to enable it."""
    agent = _load_agent(agent_id)
    state = await asyncio.to_thread(_agent_state, agent_id, agent)
    wanted = {item.strip() for item in (ids or "").split(",") if item.strip()}
    if wanted:
        state = [row for row in state if row["id"] in wanted]
    return {"ok": True, "agent_id": agent_id, "capabilities": state}


def _row(agent_id: str, agent: Any, capability_id: str) -> Dict[str, Any]:
    return next(
        row for row in _agent_state(agent_id, agent) if row["id"] == capability_id
    )


@router.post("/api/agents/{agent_id}/capabilities/{capability_id}/enable")
async def api_enable_capability(agent_id: str, capability_id: str, request: Request):
    """Enable a capability through one of the offers the state listed."""
    if capability_id not in capability_gaps.BY_ID:
        raise HTTPException(status_code=404, detail="Unknown capability")
    agent = _load_agent(agent_id)
    payload = await _json_object(request, required=False)
    kind = str(payload.get("kind") or "")

    if kind == "internet_provider" and capability_id == "web_search":
        await asyncio.to_thread(_enable_internet_provider, agent, payload)
    elif kind == "mcp_preset":
        await asyncio.to_thread(
            _attach_preset, agent_id, agent, capability_id, payload, request
        )
    else:
        raise HTTPException(
            status_code=400, detail="This option cannot be enabled from here"
        )
    agent = _load_agent(agent_id)
    return {"ok": True, "capability": _row(agent_id, agent, capability_id)}


def _enable_internet_provider(agent: Any, payload: Dict[str, Any]) -> None:
    from ..._env_io import apply_env_updates

    choice = next(
        (
            item
            for item in capability_gaps.INTERNET_PROVIDERS
            if item["provider"] == str(payload.get("provider") or "")
        ),
        None,
    )
    if choice is None:
        raise HTTPException(status_code=400, detail="Unknown web search provider")
    api_key = str(payload.get("api_key") or "").strip()
    if api_key:
        # The same store the Settings page writes: ~/.memorizz/.env.
        error = apply_env_updates({choice["key_env"]: api_key})
        if error:
            raise HTTPException(
                status_code=500, detail=f"Could not save the key: {error}"
            )
    config = _build_internet_provider_config(choice["provider"]) or {}
    error = _validate_internet_provider_choice(choice["provider"], config)
    if error:
        raise HTTPException(status_code=400, detail=error)
    _save_agent_fields(
        agent,
        internet_access_provider=choice["provider"],
        internet_access_config=config,
    )


def _attach_preset(
    agent_id: str,
    agent: Any,
    capability_id: str,
    payload: Dict[str, Any],
    request: Request,
) -> None:
    from .mcp import _callback_url, _save_servers

    capability = capability_gaps.BY_ID[capability_id]
    if not capability.preset:
        raise HTTPException(status_code=400, detail="No connector for this capability")
    config = catalog.preset_config(
        capability.preset,
        redirect_uri=_callback_url(request),
        client_id=str(payload.get("client_id") or "").strip() or None,
        client_secret=str(payload.get("client_secret") or "").strip() or None,
    )
    manager = MCPClientManager(
        owner_id=agent_id, servers=list(_agent_field(agent, "mcp_servers") or [])
    )
    try:
        manager.upsert_server(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _save_servers(agent, manager.server_dicts())
