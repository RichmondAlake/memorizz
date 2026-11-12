# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Small JSON routes behind the shell: the inbox bell, the store switcher
and the actions a table offers for several selected agents."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from ..state import _state
from ..trace_access import current_principal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["shell"])

MAX_AGENTS_PER_REQUEST = 200


def _require_operator() -> None:
    if not _state.get("provider"):
        raise HTTPException(status_code=400, detail="Not connected")
    if current_principal.get().restricted:
        raise HTTPException(status_code=403, detail="Not available to this account")


@router.get("/inbox")
async def api_inbox(request: Request) -> Dict[str, Any]:
    """What is waiting for a person: approvals, reviews and failures."""
    if not _state.get("provider"):
        return {"count": 0, "items": []}
    from ..app import _get_automation_store_for_ui
    from ..inbox import cached_inbox

    store = _get_automation_store_for_ui()
    items = await run_in_threadpool(cached_inbox, store)
    return {"count": len(items), "items": items}


@router.get("/budget")
async def api_budget(request: Request) -> Dict[str, Any]:
    """This month's known spend against the monthly budget, for the meter at
    the foot of the sidebar. ``set`` is false when there is no budget."""
    if not _state.get("provider") or current_principal.get().restricted:
        return {"set": False}
    from datetime import datetime, timezone

    from ..usage_budget import budget_status, money

    try:
        status = await run_in_threadpool(
            lambda: budget_status(request, now=datetime.now(timezone.utc))
        )
    except Exception:
        logger.debug("Budget status unavailable", exc_info=True)
        status = None
    if not status or status.get("state") == "unknown":
        return {"set": bool(status), "state": "unknown" if status else None}
    return {
        "set": True,
        "state": status["state"],
        "percent": round(float(status["percent"]), 1),
        "spend": money(status["spend"]),
        "budget": money(status["budget"]),
    }


@router.get("/local-stores")
async def api_local_stores() -> Dict[str, Any]:
    """FileSystem stores on this computer, for switching without the
    connect page. Empty for an account that cannot open that page."""
    if current_principal.get().restricted:
        return {"stores": []}
    from pathlib import Path

    from ..local_stores import detect_local_stores

    current = ""
    if _state.get("provider") and _state.get("provider_type") == "filesystem":
        path = str((_state.get("connection_info") or {}).get("path") or "")
        try:
            current = str(Path(path).expanduser().resolve()) if path else ""
        except OSError:
            current = path
    from ..local_stores import _agent_count, display_path

    stores = [
        {**store, "current": store["path"] == current}
        for store in await run_in_threadpool(detect_local_stores)
        if store["exists"]
    ]
    if current and not any(store["current"] for store in stores):
        # Connected to a folder outside MEMORIZZ_HOME: still list it, first.
        folder = Path(current)
        stores.insert(
            0,
            {
                "path": current,
                "display": display_path(folder),
                "agents": _agent_count(folder),
                "exists": True,
                "default": False,
                "current": True,
            },
        )
    return {"stores": stores}


async def _agent_ids(request: Request) -> List[str]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Send a JSON object") from exc
    ids = payload.get("agent_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="List the agents in agent_ids")
    if len(ids) > MAX_AGENTS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_AGENTS_PER_REQUEST} agents in one request",
        )
    return list(dict.fromkeys(str(agent_id) for agent_id in ids))


@router.post("/agents/bulk-delete")
async def api_delete_agents(request: Request) -> Dict[str, Any]:
    """Delete several saved agents. Their memories, conversations and traces
    are kept: only the agent records go."""
    _require_operator()
    ids = await _agent_ids(request)
    provider = _state["provider"]

    def _delete() -> Dict[str, Any]:
        deleted, failed = [], []
        for agent_id in ids:
            try:
                if provider.retrieve_memagent(agent_id) is None:
                    failed.append({"agent_id": agent_id, "reason": "Not found"})
                elif provider.delete_memagent(agent_id, cascade=False):
                    deleted.append(agent_id)
                else:
                    failed.append({"agent_id": agent_id, "reason": "Not deleted"})
            except Exception as exc:
                logger.warning("Could not delete agent %s: %s", agent_id, exc)
                failed.append({"agent_id": agent_id, "reason": str(exc)})
        return {"deleted": deleted, "failed": failed}

    result = await run_in_threadpool(_delete)
    from ..inbox import clear_cache

    clear_cache()
    return {"ok": not result["failed"], **result}


@router.post("/agents/bulk-favorite")
async def api_favorite_agents(request: Request) -> Dict[str, Any]:
    """Mark several agents as favorites, or clear the mark."""
    _require_operator()
    ids = await _agent_ids(request)
    payload = await request.json()
    value = bool(payload.get("is_favorite", True))
    provider = _state["provider"]

    def _update() -> Dict[str, Any]:
        updated, failed = [], []
        for agent_id in ids:
            try:
                existing = provider.retrieve_memagent(agent_id)
                if existing is None:
                    failed.append({"agent_id": agent_id, "reason": "Not found"})
                    continue
                provider.store_memagent(
                    existing.model_copy(
                        update={"agent_id": agent_id, "is_favorite": value}
                    )
                )
                updated.append(agent_id)
            except Exception as exc:
                logger.warning("Could not update agent %s: %s", agent_id, exc)
                failed.append({"agent_id": agent_id, "reason": str(exc)})
        return {"updated": updated, "failed": failed}

    result = await run_in_threadpool(_update)
    return {"ok": not result["failed"], **result}
