# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Vercel Agent Skills marketplace: the /vercel-skills page + its 3 API routes.

Extracted verbatim from ``ui/app.py``. Route paths, response classes, and
behavior are unchanged.
"""

import logging
import os
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ..helpers import _to_text
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vercel-skills"])


@router.get("/vercel-skills", response_class=HTMLResponse)
async def vercel_skills_page(request: Request):
    """Vercel Agent Skills marketplace browser."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    github_token_present = bool(_to_text(os.environ.get("GITHUB_TOKEN", "")).strip())

    return templates.TemplateResponse(
        "vercel_skills.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "active_page": "vercel-skills",
            "github_token_present": github_token_present,
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

    config: Dict[str, Any] = {}
    github_token = _to_text(os.environ.get("GITHUB_TOKEN", "")).strip()
    if github_token:
        config["github_token"] = github_token

    try:
        provider = VercelSkillsProvider(config=config)
        result = provider.search(query=query, limit=limit)
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

    config: Dict[str, Any] = {}
    github_token = _to_text(os.environ.get("GITHUB_TOKEN", "")).strip()
    if github_token:
        config["github_token"] = github_token

    try:
        provider = VercelSkillsProvider(config=config)
        result = provider.fetch_skill(
            repo=repo,
            skill_name=skill_name or None,
            branch=branch,
        )
        return JSONResponse(result)
    except Exception as exc:
        logger.error("Vercel skill fetch failed: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)})


@router.get("/vercel-skills/api/token-status")
async def vercel_skills_token_status(request: Request):
    """Report whether GITHUB_TOKEN is configured in the environment."""
    token_present = bool(_to_text(os.environ.get("GITHUB_TOKEN", "")).strip())
    return JSONResponse({"ok": True, "github_token_present": token_present})
