# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Oracle-container-via-Docker management endpoints.

Extracted verbatim from ui/app.py — behavior is unchanged. Every handler is a
thin wrapper over the ``memorizz.ui.docker_oracle`` helper (``_do``), so they
carry no app state and are exercised in tests by mocking that helper.
"""

import json
import os
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse, StreamingResponse

router = APIRouter(prefix="/api/docker/oracle", tags=["oracle-docker"])


@router.get("/status")
async def docker_oracle_status():
    """Report discovered Oracle containers and their state.

    Returns all containers using a gvenzl/oracle-free image (not just the
    canonical `memorizz_oracle`) so users who already have a named container like
    `oracle-memorizz` can start that one instead of creating a duplicate.
    """
    from .. import docker_oracle as _do

    if not _do.docker_available():
        return JSONResponse(
            {
                "state": "docker-unavailable",
                "container_name": _do.CONTAINER_NAME,
                "image": _do.IMAGE,
                "existing": [],
            }
        )
    existing = _do.list_oracle_containers()
    if existing:
        any_running = any(c["state"] == "running" for c in existing)
        summary_state = "running" if any_running else "stopped"
    else:
        summary_state = "absent"
    return JSONResponse(
        {
            "state": summary_state,
            "container_name": _do.CONTAINER_NAME,
            "image": _do.IMAGE,
            "existing": existing,
        }
    )


@router.get("/runtime")
async def docker_oracle_runtime():
    """Report whether a container runtime is available on the host.

    Used by the connect page when /status returns docker-unavailable so the UI
    can offer the right next action: launch an installed GUI app, or link to a
    download.
    """
    from .. import docker_oracle as _do

    return JSONResponse(_do.detect_runtime())


@router.post("/runtime/start")
async def docker_oracle_runtime_start():
    """Best-effort: launch the installed runtime and wait for the daemon.

    Synchronous because the cold-start budget (~30-60s for Docker Desktop) fits
    comfortably inside one HTTP request and avoids us having to invent a polling
    endpoint just for this transition.
    """
    from .. import docker_oracle as _do

    ok, message = _do.start_runtime()
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 500)


@router.post("/start")
async def docker_oracle_start(container_name: str = Form(...)):
    """Start an existing stopped Oracle container by name."""
    from .. import docker_oracle as _do

    if not _do.docker_available():
        return JSONResponse(
            {"ok": False, "message": "Docker is not available"},
            status_code=503,
        )
    state = _do.get_container_state(container_name)
    if state == "running":
        return JSONResponse(
            {"ok": True, "message": f"Container '{container_name}' already running"}
        )
    if state == "absent":
        return JSONResponse(
            {
                "ok": False,
                "message": f"Container '{container_name}' does not exist",
            },
            status_code=404,
        )
    ok, message = _do.start_container(container_name)
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 500)


@router.get("/logs/stream")
async def docker_oracle_logs_stream(name: Optional[str] = None, tail: int = 200):
    """Server-Sent Events stream of ``docker logs -f`` for a container.

    Used by the create-modal so users see live Oracle startup output (pull
    progress, initialization banners, "DATABASE IS READY TO USE") instead of a
    blank spinner. Closes when the client disconnects.

    Secrets in logs: gvenzl/oracle-free does not echo the configured passwords,
    but we still redact any ``ORACLE_PASSWORD`` / ``APP_USER_PASSWORD`` env
    values if they happen to appear — belt and braces.
    """
    from .. import docker_oracle as _do

    container = (name or _do.CONTAINER_NAME).strip()
    # Defensive redaction list — scrape any password the current process has in
    # env. Keeps us safe if a future image change or a custom entrypoint starts
    # echoing them.
    redact_needles = [
        v
        for v in (
            os.environ.get("ORACLE_PASSWORD"),
            os.environ.get("APP_USER_PASSWORD"),
        )
        if v
    ]

    def _redact(line: str) -> str:
        for needle in redact_needles:
            if needle and needle in line:
                line = line.replace(needle, "***")
        return line

    def _event(data: str, event: Optional[str] = None) -> str:
        # Standard SSE frame. `data:` lines end with one \n; two \n ends the event.
        prefix = f"event: {event}\n" if event else ""
        # Make multi-line data safe by prefixing each line with `data: `.
        payload = "".join(f"data: {seg}\n" for seg in data.splitlines()) or "data:\n"
        return prefix + payload + "\n"

    def generator():
        yield _event(
            json.dumps({"container": container, "message": "attached"}),
            event="attach",
        )
        try:
            for line in _do.stream_container_logs(container, tail=tail):
                yield _event(json.dumps({"line": _redact(line)}), event="log")
        except Exception as exc:  # pragma: no cover - defensive
            yield _event(
                json.dumps({"error": str(exc)}),
                event="error",
            )
        finally:
            yield _event(json.dumps({"message": "closed"}), event="close")

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering if ever proxied
            "Connection": "keep-alive",
        },
    )


@router.post("/create")
async def docker_oracle_create(
    oracle_user: str = Form(...),
    oracle_password: str = Form(...),
    oracle_dsn: str = Form(...),
):
    """Create the Oracle container from scratch using form credentials."""
    from .. import docker_oracle as _do

    if not _do.docker_available():
        return JSONResponse(
            {"ok": False, "message": "Docker is not available"},
            status_code=503,
        )
    if _do.get_container_state() != "absent":
        return JSONResponse(
            {"ok": False, "message": "Container already exists — use start"},
            status_code=409,
        )
    port = _do.parse_port_from_dsn(oracle_dsn)
    ok, message = _do.create_container(oracle_user, oracle_password, port)
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 500)
