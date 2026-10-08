# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Agent knowledge-base API: ingest uploads, list entries, delete an entry.

Extracted verbatim from ``ui/app.py``. Route paths, response classes, and
behavior are unchanged. ``_load_agent_knowledge_base`` lives in ``ui.helpers``
because the playground page render in ``app.py`` uses it too.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from ..helpers import _load_agent_knowledge_base
from ..state import _state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["knowledge-base"])

# Largest single file the ingest endpoint accepts (same cap as memory transfer).
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


class _UploadTooLarge(Exception):
    """An upload passed MAX_UPLOAD_BYTES while it was being read."""


async def _read_capped(upload: UploadFile) -> bytes:
    """Read one upload in chunks and stop as soon as it passes the cap."""
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise _UploadTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def _upload_limit_label() -> str:
    mib = 1024 * 1024
    if MAX_UPLOAD_BYTES % mib == 0:
        return f"{MAX_UPLOAD_BYTES // mib} MiB"
    return f"{MAX_UPLOAD_BYTES} bytes"


def _other_agents_referencing(provider: Any, agent_id: str, kb_id: str) -> List[str]:
    """IDs of the other stored agents whose knowledge_base_ids include kb_id.

    Reports a placeholder holder when the agents cannot be listed, so unknown
    ownership keeps the entries instead of deleting another agent's data.
    """
    try:
        records = provider.list_memagents() or []
    except Exception as exc:
        logger.warning("Could not list agents before deleting %s: %s", kb_id, exc)
        return ["unknown"]
    holders: List[str] = []
    for record in records:
        if isinstance(record, dict):
            other_id = record.get("agent_id")
            ids = record.get("knowledge_base_ids") or []
        else:
            other_id = getattr(record, "agent_id", None)
            ids = getattr(record, "knowledge_base_ids", None) or []
        if other_id and str(other_id) != str(agent_id) and kb_id in ids:
            holders.append(str(other_id))
    return holders


@router.post("/api/agents/{agent_id}/knowledge-base/ingest")
async def api_agent_knowledge_base_ingest(
    agent_id: str,
    files: List[UploadFile] = File(...),
    chunking_strategy: str = Form("fixed"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(100),
):
    """Ingest one or more uploaded files into the knowledge base and
    attach the resulting ``knowledge_base_id``s to this agent.

    Each file becomes one ingest call (one ``knowledge_base_id`` per
    file), split into chunks per the requested strategy. Text extraction
    is delegated to :mod:`memorizz.long_term.semantic.extractors` so the
    SDK (:meth:`KnowledgeBase.ingest_file`) and this endpoint share the
    exact same format-handling logic.
    """
    from ...long_term.semantic import EmptyDocumentError, ExtractorError, KnowledgeBase
    from ...memagent import MemAgent

    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        agent = await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )
    except Exception as exc:
        logger.error("Failed to load agent %s for KB ingest: %s", agent_id, exc)
        raise HTTPException(status_code=404, detail="Agent not found")

    kb = KnowledgeBase(memory_provider=_state["provider"])
    results: List[Dict[str, Any]] = []
    for upload in files:
        filename = upload.filename or "upload"
        try:
            raw = await _read_capped(upload)
        except _UploadTooLarge:
            results.append(
                {
                    "filename": filename,
                    "ok": False,
                    "error": f"file exceeds the {_upload_limit_label()} upload limit",
                }
            )
            continue
        except Exception as exc:
            results.append(
                {"filename": filename, "ok": False, "error": f"read failed: {exc}"}
            )
            continue
        if not raw:
            results.append({"filename": filename, "ok": False, "error": "empty file"})
            continue

        try:
            # Extraction, chunking and embedding block; keep them off the loop.
            kb_id = await run_in_threadpool(
                kb.ingest_file,
                raw,
                namespace=filename,
                filename=filename,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except EmptyDocumentError as exc:
            results.append({"filename": filename, "ok": False, "error": str(exc)})
            continue
        except ExtractorError as exc:
            # Covers UnsupportedFileType, MissingExtractorDependency,
            # ExtractionError — each carries an actionable message.
            results.append({"filename": filename, "ok": False, "error": str(exc)})
            continue
        except Exception as exc:
            logger.exception(
                "KB ingest failed for agent=%s file=%s", agent_id, filename
            )
            results.append({"filename": filename, "ok": False, "error": str(exc)})
            continue

        await run_in_threadpool(kb.attach_to_agent, agent, kb_id)
        try:
            chunk_count = len(await run_in_threadpool(kb.retrieve_knowledge, kb_id))
        except Exception:
            chunk_count = 0
        results.append(
            {
                "filename": filename,
                "ok": True,
                "knowledge_base_id": kb_id,
                "chunk_count": chunk_count,
                "bytes": len(raw),
            }
        )

    succeeded = sum(1 for r in results if r.get("ok"))
    return JSONResponse(
        {
            "ok": succeeded == len(results),
            "ingested": succeeded,
            "total": len(results),
            "results": results,
        },
        status_code=200 if succeeded else 400,
    )


@router.get("/api/agents/{agent_id}/knowledge-base")
async def api_agent_knowledge_base(agent_id: str):
    """Return the agent's current knowledge base entries as JSON.

    Used by the playground to refresh the Knowledge Base panel in
    place after an ingest or delete, instead of doing a full page
    reload. Same data shape that ``_load_agent_knowledge_base``
    produces for the initial server render.
    """
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")
    try:
        agent = _state["provider"].retrieve_memagent(agent_id)
    except Exception as exc:
        logger.error("Failed to load agent %s for KB list: %s", agent_id, exc)
        raise HTTPException(status_code=404, detail="Agent not found")
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return JSONResponse(
        {
            "agent_id": agent_id,
            "entries": _load_agent_knowledge_base(agent),
        }
    )


@router.delete("/api/agents/{agent_id}/knowledge-base/{kb_id}")
async def api_agent_knowledge_base_delete(agent_id: str, kb_id: str):
    """Detach a knowledge_base_id from an agent; delete its entries only
    when this agent was their last owner.

    Invoked by the playground's pending-chips UI when the user clicks
    × on a successfully-ingested file. Always returns a JSON result so
    the frontend can update the chip uniformly: 404 when the id is not
    attached to this agent (nothing is touched), 409 when it was detached
    but another agent still lists it (the entries are kept), 200 once the
    entries are gone.
    """
    from ...long_term.semantic import KnowledgeBase
    from ...memagent import MemAgent

    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    try:
        agent = await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )
    except Exception as exc:
        logger.error("Failed to load agent %s for KB delete: %s", agent_id, exc)
        raise HTTPException(status_code=404, detail="Agent not found")

    if kb_id not in (getattr(agent, "knowledge_base_ids", None) or []):
        return JSONResponse(
            {
                "ok": False,
                "knowledge_base_id": kb_id,
                "detached": False,
                "deleted": False,
                "error": "This knowledge base is not attached to the agent",
            },
            status_code=404,
        )

    kb = KnowledgeBase(memory_provider=_state["provider"])
    detached = await run_in_threadpool(kb.detach_from_agent, agent, kb_id)
    if not detached:
        return JSONResponse(
            {
                "ok": False,
                "knowledge_base_id": kb_id,
                "detached": False,
                "deleted": False,
                "error": "The agent could not be updated",
            },
            status_code=500,
        )

    # The entries are shared storage: delete them only once no agent is left
    # that still lists this knowledge base.
    holders = await run_in_threadpool(
        _other_agents_referencing, _state["provider"], agent_id, kb_id
    )
    if holders:
        return JSONResponse(
            {
                "ok": True,
                "knowledge_base_id": kb_id,
                "detached": True,
                "deleted": False,
                "still_referenced_by": holders,
            },
            status_code=409,
        )

    deleted = await run_in_threadpool(kb.delete_knowledge, kb_id)
    return JSONResponse(
        {
            "ok": deleted,
            "knowledge_base_id": kb_id,
            "detached": True,
            "deleted": deleted,
        },
        status_code=200 if deleted else 500,
    )
