"""Administrator memory archive downloads and reviewed restores."""

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from ...memory_archive import (
    MAX_BYTES,
    TAXONOMY,
    MemoryArchive,
    MemoryArchiveError,
    archive_json,
)
from ..security import ui_read_only
from ..state import _state, templates
from ..trace_access import current_principal

router = APIRouter(tags=["memory-transfer"])


def _provider(*, write=False):
    # Archive files contain actual memory values and agent configuration, not
    # just trace metadata. Trace-only accounts cannot obtain or restore them.
    if current_principal.get().restricted:
        raise HTTPException(
            403, "Memory archives require unrestricted administrator access"
        )
    if not _state.get("provider"):
        raise HTTPException(400, "Connect a memory provider first")
    if write and (ui_read_only() or _state.get("read_only")):
        raise HTTPException(403, "Memory import is disabled in read-only mode")
    return _state["provider"]


@router.get("/memory-transfer")
async def transfer_page(request: Request, agent_id: str = "", memory_id: str = ""):
    if not _state.get("provider"):
        return RedirectResponse("/connect", status_code=302)
    provider = _provider()
    agents = await run_in_threadpool(provider.list_memagents)
    return templates.TemplateResponse(
        "memory_transfer.html",
        {
            "request": request,
            "active_page": "memory-transfer",
            "agent_id": agent_id,
            "memory_id": memory_id,
            "agents": agents or [],
            "taxonomy": TAXONOMY,
            "provider_type": _state.get("provider_type"),
            "read_only": ui_read_only() or _state.get("read_only", False),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/memory-transfer/export")
async def export_download(
    agent_id: str | None = None,
    memory_id: str | None = None,
    memory_types: str | None = None,
    include_delegates: bool = True,
    include_history: bool = True,
    include_context: bool = True,
):
    service = MemoryArchive(_provider())
    try:
        archive = await run_in_threadpool(
            service.export,
            agent_id=agent_id or None,
            memory_id=memory_id or None,
            memory_types=memory_types.split(",") if memory_types is not None else None,
            include_delegates=include_delegates,
            include_history=include_history,
            include_context=include_context,
        )
    except MemoryArchiveError as exc:
        raise HTTPException(400, str(exc)) from exc
    timestamp = archive["manifest"]["created_at"][:19].replace(":", "-")
    return Response(
        archive_json(archive),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="memory-{timestamp}.memorizz.json"',
            "Cache-Control": "no-store",
            "X-Memorizz-Records": str(archive["manifest"]["record_count"]),
        },
    )


@router.post("/api/memory-transfer/import")
async def import_upload(request: Request):
    provider = _provider()
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "Cross-origin memory import denied")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise HTTPException(413, "Archive exceeds 50 MiB")
    try:
        arguments = json.loads(body)
        if not isinstance(arguments, dict) or set(arguments) - {
            "archive",
            "dry_run",
            "conflict",
            "id_strategy",
            "target_memory_id",
        }:
            raise MemoryArchiveError("Unsupported import options")
        dry_run = arguments.get("dry_run", True)
        if not isinstance(dry_run, bool):
            raise MemoryArchiveError("dry_run must be true or false")
        if not dry_run:
            _provider(write=True)
        result = await run_in_threadpool(
            MemoryArchive(provider).import_archive,
            arguments.get("archive"),
            dry_run=dry_run,
            conflict=arguments.get("conflict", "error"),
            id_strategy=arguments.get("id_strategy", "preserve"),
            target_memory_id=arguments.get("target_memory_id"),
        )
    except (MemoryArchiveError, ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
