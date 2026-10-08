# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memory list pages: GET /memory/{memory_type}.

Extracted verbatim from ``ui/app.py``. Route path, response class, and
behavior are unchanged.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory-pages"])


@router.get("/memory/{memory_type}", response_class=HTMLResponse)
async def memory_list(request: Request, memory_type: str, error: Optional[str] = None):
    """Show list of items for a memory type."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from ...enums.memory_type import MemoryType

    # Map URL path to MemoryType
    type_mapping = {
        "personas": MemoryType.PERSONAS,
        "toolbox": MemoryType.TOOLBOX,
        "conversations": MemoryType.CONVERSATION_MEMORY,
        "workflows": MemoryType.WORKFLOW_MEMORY,
        "knowledge-base": MemoryType.KNOWLEDGE_BASE,
        "short-term": MemoryType.SHORT_TERM_MEMORY,
        "entity": MemoryType.ENTITY_MEMORY,
        "summaries": MemoryType.SUMMARIES,
        "shared": MemoryType.SHARED_MEMORY,
        "cache": MemoryType.SEMANTIC_CACHE,
        "skills": MemoryType.SKILLBOX,
    }

    mem_type = type_mapping.get(memory_type)
    if not mem_type:
        raise HTTPException(status_code=404, detail="Unknown memory type")

    items = []
    total = None
    load_error = None
    provider = _state["provider"]
    try:
        if _state.get("provider_type") == "notion":
            items = await run_in_threadpool(
                provider.retrieve_by_query, {}, mem_type, limit=101
            )
        elif callable(getattr(provider, "list_recent", None)):
            # Database providers: read only the newest page, count from metadata.
            items = await run_in_threadpool(provider.list_recent, mem_type, 101)
            total = await run_in_threadpool(provider.estimate_count, mem_type)
        else:
            items = await run_in_threadpool(provider.list_all, mem_type)
    except Exception as e:
        logger.error("Failed to list %s (%s)", memory_type, type(e).__name__)
        load_error = "Memory could not be loaded. Check provider access and query limits; this is not an empty-result confirmation."

    from ..memory_view import build_memory_view

    # Newest 100 records, shaped for the explorer (no embedding vectors).
    view = await run_in_threadpool(lambda: build_memory_view(items or [], total=total))
    return templates.TemplateResponse(
        "memory_list.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "memory_type": memory_type,
            "memory_type_display": memory_type.replace("-", " ").title(),
            "view": view,
            "items": view["records"],
            "has_more": view["has_more"],
            # Notion reads a bounded page, so its count is a floor, not a total.
            "total_known": _state.get("provider_type") != "notion",
            "load_error": load_error,
            "action_error": (error or "")[:400],
            "active_page": memory_type,
        },
    )
