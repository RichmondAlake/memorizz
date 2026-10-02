# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Agent skills: the /vercel-skills page, marketplace search and fetch, and
attaching a skill to an agent."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..helpers import (
    _agent_field,
    _extract_agent_identifier,
    _extract_agent_persona_name,
    _list_agents,
    _load_agent,
    _save_agent_fields,
    _to_text,
)
from ..integrations_view import skills_marketplace_agents
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vercel-skills"])


@router.get("/vercel-skills", response_class=HTMLResponse)
async def vercel_skills_page(request: Request):
    """Vercel Agent Skills marketplace browser."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    github_token_present = bool(_to_text(os.environ.get("GITHUB_TOKEN", "")).strip())
    try:
        agents = list(_list_agents() or [])
    except Exception as exc:
        logger.warning("Could not list agents for the skills page: %s", exc)
        agents = []

    return templates.TemplateResponse(
        "vercel_skills.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "active_page": "vercel-skills",
            "github_token_present": github_token_present,
            "skill_agents": skills_marketplace_agents(agents),
            "agent_total": len(agents),
            "agents": [
                {
                    "agent_id": _extract_agent_identifier(agent),
                    "name": _extract_agent_persona_name(agent),
                    "skills": len(_skill_paths(agent)),
                }
                for agent in agents
                if _extract_agent_identifier(agent)
            ],
        },
    )


@router.get("/vercel-skills/api/search")
async def vercel_skills_api_search(
    request: Request,
    q: str = "",
    limit: int = 20,
):
    """API: search Vercel skills via GitHub."""
    from ...vercel_skills import VercelSkillsProvider

    query = (q or "").strip()
    if not query:
        return JSONResponse({"ok": False, "error": "Query parameter 'q' is required."})

    try:
        provider = VercelSkillsProvider(config=_provider_config())
        result = await asyncio.to_thread(provider.search, query=query, limit=limit)
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Vercel skills search failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)})


@router.get("/vercel-skills/api/fetch")
async def vercel_skills_api_fetch(
    request: Request,
    repo: str = "",
    skill_name: str = "",
    branch: str = "main",
):
    """API: fetch a SKILL.md from a GitHub repo."""
    from ...vercel_skills import VercelSkillsProvider

    repo = (repo or "").strip()
    if not repo:
        return JSONResponse({"ok": False, "error": "Parameter 'repo' is required."})

    try:
        provider = VercelSkillsProvider(config=_provider_config())
        result = await asyncio.to_thread(
            provider.fetch_skill,
            repo=repo,
            skill_name=skill_name or None,
            branch=branch,
        )
        result.pop("content", None)
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Vercel skill fetch failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)})


def _provider_config() -> Dict[str, Any]:
    token = _to_text(os.environ.get("GITHUB_TOKEN", "")).strip()
    return {"github_token": token} if token else {}


def _skill_paths(agent: Any) -> List[str]:
    value = _agent_field(agent, "skill_paths")
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value or [] if str(item or "").strip()]


def _attached(agent: Any) -> List[Dict[str, Any]]:
    from ...vercel_skills.install import describe_skill_file

    return [describe_skill_file(path) for path in _skill_paths(agent)]


@router.get("/api/agents/{agent_id}/skills")
async def api_agent_skills(agent_id: str):
    """Skill files attached to an agent."""
    return {"ok": True, "skills": _attached(_load_agent(agent_id))}


@router.post("/api/agents/{agent_id}/skills")
async def api_attach_skill(agent_id: str, request: Request):
    """Save a marketplace skill locally and add it to the agent's skills."""
    from ...vercel_skills import VercelSkillsProvider
    from ...vercel_skills.install import install_skill

    agent = _load_agent(agent_id)
    payload = await request.json()
    payload = payload if isinstance(payload, dict) else {}
    repo = str(payload.get("repo") or "").strip()
    if not repo:
        raise HTTPException(status_code=400, detail="repo is required (owner/repo)")
    result = await asyncio.to_thread(
        install_skill,
        repo,
        str(payload.get("skill_name") or "").strip() or None,
        branch=str(payload.get("branch") or "main").strip() or "main",
        provider=VercelSkillsProvider(config=_provider_config()),
    )
    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    paths = _skill_paths(agent)
    if result["path"] not in paths:
        paths.append(result["path"])
    _save_agent_fields(agent, skill_paths=paths)
    return {"ok": True, "skill": result, "skills": _attached(agent)}


@router.delete("/api/agents/{agent_id}/skills")
async def api_detach_skill(agent_id: str, path: str):
    """Remove a skill from the agent. The saved file stays for other agents."""
    agent = _load_agent(agent_id)
    paths = _skill_paths(agent)
    remaining = [item for item in paths if item != path and str(Path(item)) != path]
    if len(remaining) == len(paths):
        raise HTTPException(
            status_code=404, detail="Skill is not attached to this agent"
        )
    _save_agent_fields(agent, skill_paths=remaining)
    return {"ok": True, "skills": _attached(agent)}
