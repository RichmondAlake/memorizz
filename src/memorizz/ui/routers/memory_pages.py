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
        # The records themselves (/memory/workflows and /memory/skills are
        # the continual-learning views of the same stores).
        "workflow-memory": MemoryType.WORKFLOW_MEMORY,
        "skillbox": MemoryType.SKILLBOX,
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


# ---------------------------------------------------------------------------
# Edits an operator makes from a record's pane
# ---------------------------------------------------------------------------


def _back(memory_type: str, error: Optional[str] = None) -> RedirectResponse:
    from urllib.parse import quote

    url = f"/memory/{memory_type}"
    if error:
        url += f"?error={quote(str(error)[:300])}"
    return RedirectResponse(url=url, status_code=303)


def _agents_using_persona(provider, persona_id: str) -> list:
    """IDs of the saved agents whose persona is ``persona_id``."""
    from ...learning.persona_learning import persona_of

    found = []
    for agent in provider.list_memagents() or []:
        try:
            persona = persona_of(agent)
        except Exception:
            persona = None
        if persona is not None and str(persona.persona_id) == str(persona_id):
            agent_id = getattr(agent, "agent_id", None)
            if agent_id:
                found.append(str(agent_id))
    return found


@router.post("/memory/personas/{persona_id}/guardrails")
async def set_persona_guardrails(persona_id: str, request: Request):
    """Replace a persona's guardrails. Only this form (an operator) and host
    code can: learning, the model's tool and the refine cycle are refused."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    form = await request.form()
    lines = [
        line.strip()
        for line in str(form.get("guardrails") or "").splitlines()
        if line.strip()
    ]
    provider = _state["provider"]

    def _save() -> None:
        from ...enums.memory_type import MemoryType
        from ...long_term.semantic.persona import Persona
        from ...memagent import MemAgent

        trigger = {"reason": "Guardrails edited in the UI", "source_type": "ui_form"}
        agent_ids = _agents_using_persona(provider, persona_id)
        if agent_ids:
            # Through the agent, so its stored copy of the persona follows.
            for agent_id in agent_ids:
                agent = MemAgent.load(agent_id, memory_provider=provider)
                agent.set_persona_guardrails(
                    lines, trigger["reason"], source_type="ui_form"
                )
            return
        for row in provider.list_all(MemoryType.PERSONAS) or []:
            if isinstance(row, dict) and str(row.get("persona_id")) == str(persona_id):
                Persona.from_dict(row).set_guardrails(lines, trigger, provider)
                return
        raise ValueError("Persona not found")

    try:
        await run_in_threadpool(_save)
    except (ValueError, PermissionError, TypeError) as exc:
        return _back("personas", str(exc))
    except Exception as exc:
        logger.error("Guardrails were not saved (%s)", type(exc).__name__)
        return _back("personas", "The guardrails could not be saved.")
    return _back("personas")


@router.post("/memory/entity/{entity_id}/aliases")
async def set_entity_aliases(entity_id: str, request: Request):
    """Replace an entity's alias list."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    form = await request.form()
    aliases = [
        part.strip()
        for part in str(form.get("aliases") or "").replace("\n", ",").split(",")
        if part.strip()
    ]
    memory_id = str(form.get("memory_id") or "") or None
    user_id = str(form.get("user_id") or "") or None

    def _save() -> None:
        from ...long_term.semantic.entity_memory import EntityMemory

        EntityMemory(_state["provider"]).set_aliases(
            entity_id, aliases, memory_id=memory_id, user_id=user_id
        )

    try:
        await run_in_threadpool(_save)
    except ValueError as exc:
        return _back("entity", str(exc))
    except Exception as exc:
        logger.error("Aliases were not saved (%s)", type(exc).__name__)
        return _back("entity", "The aliases could not be saved.")
    return _back("entity")


@router.post("/memory/entity/{entity_id}/undo-merge")
async def undo_entity_merge(entity_id: str, request: Request):
    """Undo the consolidation plan that merged other records into this one:
    the merged records come back and the aliases the merge added go."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    form = await request.form()
    plan_id = str(form.get("plan_id") or "").strip()
    memory_id = str(form.get("memory_id") or "").strip()
    user_id = str(form.get("user_id") or "") or None
    if not plan_id or not memory_id:
        return _back("entity", "The merge to undo was not named.")

    def _undo() -> dict:
        from ...long_term.semantic.entity_memory import EntityMemory

        return EntityMemory(_state["provider"]).restore_consolidation(
            plan_id, memory_id=memory_id, user_id=user_id
        )

    try:
        await run_in_threadpool(_undo)
    except Exception as exc:
        logger.error("Merge undo failed (%s)", type(exc).__name__)
        return _back("entity", "The merge could not be undone.")
    return _back("entity")
