# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Connected apps: the cloud apps (Grok Bot, ChatGPT dots) that use this memory.

The Settings page lists them, disconnects them and makes the pairing code that
approves the next one, as ``memorizz mcp apps list|revoke|pair`` do in a
terminal. A pairing code is shown once and never logged; the store keeps
only its hash, and app tokens never leave it.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from ...mcp_server.config import LOCAL_AGENTS_PRINCIPAL, READ_SCOPE, WRITE_SCOPE
from ...mcp_server.oauth_store import PAIRING_TTL_SECONDS, OAuthStore
from ..connection import ui_is_read_only
from ..observability_view import relative_age

logger = logging.getLogger(__name__)

router = APIRouter(tags=["connected-apps"])

_NO_STORE = {"Cache-Control": "no-store"}


def _principal() -> str:
    # The CLI's default, so a code made here or by `memorizz mcp apps pair`
    # connects an app to the same memory.
    from ...cli.mcp_apps import default_principal

    return default_principal()


def _when(timestamp: Optional[float]) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _app_row(grant: Dict[str, Any], now: float) -> Dict[str, Any]:
    read_only = WRITE_SCOPE not in grant["scopes"]
    shared = grant["principal"] == LOCAL_AGENTS_PRINCIPAL
    last_used = grant["last_used_at"]
    return {
        "grant_id": grant["grant_id"],
        "client_id": grant["client_id"],
        "app_name": grant["app_name"],
        "principal": grant["principal"],
        "shared": shared,
        "memory_of": "Your local agents" if shared else grant["principal"],
        "scopes": grant["scopes"],
        "read_only": read_only,
        "can": "Read only" if read_only else "Read and save",
        "created_at": grant["created_at"],
        "connected": _when(grant["created_at"]),
        "last_used_at": last_used,
        "last_used": relative_age(last_used, now) if last_used else "Never",
        "last_used_title": _when(last_used) if last_used else "",
    }


def connected_apps() -> List[Dict[str, Any]]:
    """Apps still connected, newest first, with what the page shows for each."""
    now = time.time()
    return [_app_row(grant, now) for grant in OAuthStore().list_grants()]


def connected_apps_context() -> Dict[str, Any]:
    """The Settings page's Connected apps section."""
    error = None
    try:
        apps = connected_apps()
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Could not read the connected apps: %s", exc)
        apps, error = [], "Could not read the connected apps. Try again."
    return {
        "apps": apps,
        "error": error,
        "principal": _principal(),
        "read_only": ui_is_read_only(),
        "pairing_minutes": PAIRING_TTL_SECONDS // 60,
    }


def _require_writable() -> None:
    # The middleware already refuses changes in a read-only UI; a read-only
    # connection is refused here too, as the page shows it read-only.
    if ui_is_read_only():
        raise HTTPException(
            403, "This UI is read-only, so apps can't be approved or disconnected here."
        )


@router.get("/api/connected-apps")
async def api_connected_apps():
    apps = await run_in_threadpool(connected_apps)
    return JSONResponse({"ok": True, "apps": apps}, headers=_NO_STORE)


@router.post("/api/connected-apps/pairing-code")
async def api_make_pairing_code(request: Request):
    """A one-time code that approves one app on its sign-in page."""
    _require_writable()
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    read_only = payload.get("read_only", False)
    shared = payload.get("shared", False)
    if not isinstance(read_only, bool) or not isinstance(shared, bool):
        raise HTTPException(400, "read_only and shared must be true or false")
    principal = LOCAL_AGENTS_PRINCIPAL if shared else _principal()
    scopes = [READ_SCOPE] if read_only else [READ_SCOPE, WRITE_SCOPE]
    code, expires_at = await run_in_threadpool(
        OAuthStore().create_pairing_code, principal, scopes
    )
    return JSONResponse(
        {
            "ok": True,
            "code": code,
            "principal": principal,
            "expires_at": expires_at,
            "read_only": read_only,
            "shared": shared,
        },
        headers=_NO_STORE,
    )


@router.post("/api/connected-apps/{grant_id}/disconnect")
async def api_disconnect_app(grant_id: str):
    """Cut an app off now; it needs a new pairing code to come back."""
    _require_writable()
    store = OAuthStore()
    grant = await run_in_threadpool(store.grant, grant_id)
    if grant is None or not await run_in_threadpool(store.revoke_grant, grant_id):
        raise HTTPException(404, "No connected app has this ID.")
    return JSONResponse({"ok": True, "app_name": grant["app_name"]}, headers=_NO_STORE)
