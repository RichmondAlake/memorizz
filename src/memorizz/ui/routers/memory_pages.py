# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memory list pages: GET /memory/{memory_type}.

Extracted verbatim from ``ui/app.py``. Route path, response class, and
behavior are unchanged.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["memory-pages"])


@router.get("/memory/{memory_type}", response_class=HTMLResponse)
async def memory_list(request: Request, memory_type: str):
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
    try:
        items = _state["provider"].list_all(mem_type)
    except Exception as e:
        logger.error(f"Failed to list {memory_type}: {e}")

    return templates.TemplateResponse(
        "memory_list.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "memory_type": memory_type,
            "memory_type_display": memory_type.replace("-", " ").title(),
            "items": items[:100],  # Limit to 100 items
            "active_page": memory_type,
        },
    )
