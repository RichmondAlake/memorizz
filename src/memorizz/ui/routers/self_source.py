# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A self-aware agent's own source: its versions, its errors, and the coding
model that recommends and makes fixes there."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from .. import agent_status
from ..helpers import (
    _agent_field,
    _build_agent_nav_items,
    _extract_agent_persona_name,
    _json_object,
    _load_agent,
    _save_agent_fields,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["self-source"])

CODING_PROVIDERS = ("ollama", "openrouter", "openai", "anthropic", "gemini")
# The provider menu on the source page; "" is the agent's own model.
CODING_PROVIDER_OPTIONS = (
    {
        "value": "",
        "mark": "Me",
        "title": "The agent's own model",
        "note": "Use the model the agent talks with",
        "key_env": "",
        "example": "",
    },
    {
        "value": "ollama",
        "mark": "Ol",
        "title": "Ollama",
        "note": "A model on this computer, or on a machine you run",
        "key_env": "",
        "example": "qwen2.5:7b",
    },
    {
        "value": "openrouter",
        "mark": "OR",
        "title": "OpenRouter",
        "note": "Many models with one key, such as Flint",
        "key_env": "OPENROUTER_API_KEY",
        "example": "recursive/flint-v1.0:free",
    },
    {
        "value": "openai",
        "mark": "AI",
        "title": "OpenAI",
        "note": "OpenAI, or any endpoint that speaks its protocol",
        "key_env": "OPENAI_API_KEY",
        "example": "gpt-6.1-sol",
    },
    {
        "value": "droid",
        "mark": "Dr",
        "title": "Factory Droid",
        "note": "Factory's coding agent does the fix, with its own model or yours",
        "key_env": "FACTORY_API_KEY",
        "example": "Droid's default",
    },
    {
        "value": "anthropic",
        "mark": "An",
        "title": "Anthropic",
        "note": "Claude models",
        "key_env": "ANTHROPIC_API_KEY",
        "example": "claude-sonnet-5-5",
    },
    {
        "value": "gemini",
        "mark": "Ge",
        "title": "Google Gemini",
        "note": "Gemini models",
        "key_env": "GEMINI_API_KEY",
        "example": "gemini-4-hydrogen",
    },
)
_CODING_KEYS = {
    option["value"]: option["key_env"]
    for option in CODING_PROVIDER_OPTIONS
    if option["key_env"]
}


def _coding_model_catalog() -> Dict[str, Any]:
    """Model names to suggest for each provider in the coding model form."""
    from ...llms.model_catalog import LLM_MODEL_CATALOG

    return {
        name: [str(row["value"]) for row in LLM_MODEL_CATALOG.get(name, [])][:40]
        for name in CODING_PROVIDERS
    }


_jobs_lock = threading.Lock()
# agent_id -> the last recommend or fix run, running or finished.
_jobs: Dict[str, Dict[str, Any]] = {}


def _config(agent: Any) -> Dict[str, Any]:
    config = _agent_field(agent, "self_aware_config")
    return dict(config) if isinstance(config, dict) else {}


def _enabled(agent: Any) -> bool:
    return bool(_agent_field(agent, "self_aware")) and bool(
        _config(agent).get("own_source")
    )


def _source(agent_id: str):
    from ...memagent.self_source import SelfSource

    return SelfSource(agent_id)


def _live_agent(agent_id: str) -> Any:
    from ...memagent import MemAgent

    return MemAgent.load(agent_id, memory_provider=_state["provider"])


def _job_view(agent_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(agent_id)
        return dict(job) if job else None


def _state_payload(agent_id: str, agent: Any) -> Dict[str, Any]:
    from ...memagent.self_repair import ErrorLog

    enabled = _enabled(agent)
    source = _source(agent_id)
    payload: Dict[str, Any] = {
        "ok": True,
        "agent_id": agent_id,
        "enabled": enabled,
        "coding_model": _config(agent).get("coding_model"),
        "coding_agent": _config(agent).get("coding_agent"),
        # Whether Factory's CLI can be found where MemoRizz runs.
        "droid_installed": bool(shutil.which("droid")),
        "coding_key_missing": _coding_key_missing(agent),
        # Which providers have their usual key where MemoRizz runs.
        "coding_keys": {
            name: bool(
                os.environ.get(env, "").strip()
                or (name == "gemini" and os.environ.get("GOOGLE_API_KEY", "").strip())
            )
            for name, env in _CODING_KEYS.items()
        },
        "agent_model": _agent_model_label(agent),
        "job": _job_view(agent_id),
        "preview": _preview_status(agent_id),
        "source": source.status() if source.exists() else {"exists": False},
        "versions": source.versions() if source.exists() else [],
        "errors": list(reversed(ErrorLog(agent_id).recent(30))),
    }
    return payload


def _preview_status(agent_id: str) -> Optional[Dict[str, Any]]:
    from ...memagent import self_preview

    return self_preview.status(agent_id)


def _coding_key_missing(agent: Any) -> Optional[str]:
    """The variable the coding model's key is read from, when it is not set."""
    if _config(agent).get("coding_agent"):
        return (
            None if os.environ.get("FACTORY_API_KEY", "").strip() else "FACTORY_API_KEY"
        )
    model = _config(agent).get("coding_model")
    key_env = model.get("api_key_env") if isinstance(model, dict) else None
    if not key_env and isinstance(model, dict) and not model.get("base_url"):
        key_env = _CODING_KEYS.get(str(model.get("provider") or ""))
    if key_env == "GEMINI_API_KEY" and os.environ.get("GOOGLE_API_KEY", "").strip():
        return None
    if key_env and not os.environ.get(str(key_env), "").strip():
        return str(key_env)
    return None


def _agent_model_label(agent: Any) -> str:
    config = _agent_field(agent, "llm_config")
    if isinstance(config, dict) and config.get("model"):
        return f"{config.get('provider') or 'openai'} · {config['model']}"
    return "the agent's own model"


@router.get("/agents/{agent_id}/source", response_class=HTMLResponse)
async def agent_source_page(request: Request, agent_id: str):
    """The agent's own copy of the source, its versions, errors and fixes."""
    if not _state["provider"]:
        return RedirectResponse(url="/", status_code=302)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    return templates.TemplateResponse(
        "agent_source.html",
        {
            "request": request,
            "active_page": "agents",
            "active_agent_id": agent_id,
            "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
            "agent_id": agent_id,
            "agent_name": _extract_agent_persona_name(agent),
            "coding_provider_options": CODING_PROVIDER_OPTIONS,
            "coding_model_catalog": _coding_model_catalog(),
            "read_only": bool(_state.get("read_only")),
        },
    )


@router.get("/agents/{agent_id}/source/diff", response_class=HTMLResponse)
async def agent_source_diff_page(request: Request, agent_id: str):
    """What the agent changed in its source, file by file and line by line."""
    if not _state["provider"]:
        return RedirectResponse(url="/", status_code=302)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    return templates.TemplateResponse(
        "agent_source_diff.html",
        {
            "request": request,
            "active_page": "agents",
            "active_agent_id": agent_id,
            "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
            "agent_id": agent_id,
            "agent_name": _extract_agent_persona_name(agent),
        },
    )


@router.get("/agents/{agent_id}/source/file", response_class=HTMLResponse)
async def agent_source_file_page(request: Request, agent_id: str):
    """One whole file of the agent's source, with the changed lines marked."""
    if not _state["provider"]:
        return RedirectResponse(url="/", status_code=302)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    return templates.TemplateResponse(
        "agent_source_file.html",
        {
            "request": request,
            "active_page": "agents",
            "active_agent_id": agent_id,
            "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
            "agent_id": agent_id,
            "agent_name": _extract_agent_persona_name(agent),
        },
    )


@router.get("/api/agents/{agent_id}/source")
async def api_agent_source(agent_id: str):
    agent = await asyncio.to_thread(_load_agent, agent_id)
    return await asyncio.to_thread(_state_payload, agent_id, agent)


@router.get("/api/agents/{agent_id}/source/diff")
async def api_agent_source_diff(agent_id: str, version: str = ""):
    from ...memagent.self_source import SelfSourceError, parse_diff

    await asyncio.to_thread(_load_agent, agent_id)
    source = _source(agent_id)
    try:
        diff = await asyncio.to_thread(source.diff, version or None)
    except SelfSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # How this version came about: the errors read, the prompt, the steps.
    story = None
    if version and version not in ("all", "working"):
        story = await asyncio.to_thread(source.story, version)
    return {
        "ok": True,
        "version": version,
        "diff": diff,
        "files": parse_diff(diff),
        "story": story,
    }


@router.get("/api/agents/{agent_id}/source/file")
async def api_agent_source_file(
    agent_id: str, path: str = "", version: str = "", side: str = ""
):
    """One whole file as a change left it, or as it was before (``side=old``)."""
    from ...memagent.self_source import SelfSourceError

    await asyncio.to_thread(_load_agent, agent_id)
    try:
        view = await asyncio.to_thread(
            _source(agent_id).file_view, path, version or None, side == "old"
        )
    except SelfSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **view}


@router.get("/api/agents/{agent_id}/source/patch", response_class=PlainTextResponse)
async def api_agent_source_patch(agent_id: str):
    """Everything the agent has kept, as one patch against the install."""
    from ...memagent.self_source import SelfSourceError

    await asyncio.to_thread(_load_agent, agent_id)
    try:
        patch = await asyncio.to_thread(_source(agent_id).patch)
    except SelfSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PlainTextResponse(
        patch,
        headers={
            "Content-Disposition": f'attachment; filename="memorizz-{agent_id[:8]}.patch"'
        },
    )


def _writable() -> None:
    if _state.get("read_only"):
        raise HTTPException(status_code=403, detail="This store is read-only")


@router.post("/api/agents/{agent_id}/source/enable")
async def api_agent_source_enable(agent_id: str):
    """Make the agent self-aware with its own copy of the source."""
    _writable()
    agent = await asyncio.to_thread(_load_agent, agent_id)
    config = {**_config(agent), "own_source": True}
    await asyncio.to_thread(
        _save_agent_fields, agent, self_aware=True, self_aware_config=config
    )
    await asyncio.to_thread(_source(agent_id).ensure)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    return await asyncio.to_thread(_state_payload, agent_id, agent)


@router.post("/api/agents/{agent_id}/source/coding-model")
async def api_agent_source_coding_model(agent_id: str, request: Request):
    """Choose the model that reads and edits the agent's source."""
    from ...memagent.managers import SelfAwarenessManager

    _writable()
    payload = await _json_object(request)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    provider = str(payload.get("provider") or "ollama").strip().lower()
    if provider == "droid":
        # Factory's coding agent does the fixes; the coding model steps aside.
        outside = SelfAwarenessManager._normalize_coding_agent(
            {"kind": "droid", "model": payload.get("model")}
        )
        config = {**_config(agent), "coding_agent": outside, "coding_model": None}
        await asyncio.to_thread(_save_agent_fields, agent, self_aware_config=config)
        return {"ok": True, "coding_model": None, "coding_agent": outside}
    if provider not in CODING_PROVIDERS:
        raise HTTPException(status_code=400, detail="Unknown model provider")
    host = str(payload.get("host") or "").strip()
    if host and not host.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="The address must start with http:// or https://"
        )
    base_url = str(payload.get("base_url") or "").strip()
    if base_url and not base_url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="The address must start with http:// or https://"
        )
    key_env = str(payload.get("api_key_env") or "").strip()
    if key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key_env):
        raise HTTPException(
            status_code=400,
            detail="Give the name of the variable that holds the key, "
            "such as OPENROUTER_API_KEY, not the key itself",
        )
    chosen = SelfAwarenessManager._normalize_coding_model(
        {"provider": provider, "model": payload.get("model"), "host": host}
        if provider == "ollama"
        else {
            "provider": provider,
            "model": payload.get("model"),
            # Only the OpenAI protocol is spoken by other people's endpoints.
            "base_url": base_url if provider == "openai" else "",
            "api_key_env": key_env if provider == "openai" else "",
        }
    )
    config = {**_config(agent), "coding_model": chosen, "coding_agent": None}
    await asyncio.to_thread(_save_agent_fields, agent, self_aware_config=config)
    return {"ok": True, "coding_model": chosen, "coding_agent": None}


def _store_arguments() -> list:
    """How a preview console opens the store this console has open."""
    info = _state.get("connection_info") or {}
    if _state.get("provider_type") == "filesystem" and info.get("path"):
        return ["--backend", "filesystem", "--path", str(info["path"])]
    # Other stores: the preview connects as the environment says, or shows
    # the connect page.
    return []


@router.post("/api/agents/{agent_id}/source/preview")
async def api_agent_source_preview(agent_id: str, request: Request):
    """Start a read-only console running one kept version of the agent's source."""
    from ...memagent import self_preview
    from ...memagent.self_source import SelfSourceError

    payload = await _json_object(request, required=False)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    version = str(payload.get("version") or "").strip()
    name = _extract_agent_persona_name(agent)
    source = _source(agent_id)
    try:
        if not version:
            version = source.versions()[0]["id"]
        number = next(
            (row["number"] for row in source.versions() if row["id"] == version), "?"
        )
        preview = await asyncio.to_thread(
            lambda: self_preview.start(
                source,
                version,
                store=_store_arguments(),
                label=f"{name}'s v{number}",
            )
        )
    except SelfSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "preview": preview}


@router.delete("/api/agents/{agent_id}/source/preview")
async def api_agent_source_preview_stop(agent_id: str):
    from ...memagent import self_preview

    stopped = await asyncio.to_thread(self_preview.stop, agent_id)
    return {"ok": True, "stopped": stopped}


@router.post("/api/agents/{agent_id}/source/restore")
async def api_agent_source_restore(agent_id: str, request: Request):
    from ...memagent.self_source import SelfSourceError

    _writable()
    payload = await _json_object(request)
    await asyncio.to_thread(_load_agent, agent_id)
    try:
        version = await asyncio.to_thread(
            _source(agent_id).restore, str(payload.get("version") or "")
        )
    except SelfSourceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "version": version}


def _start(agent_id: str, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with _jobs_lock:
        current = _jobs.get(agent_id)
        if current and current["state"] == "running":
            raise HTTPException(
                status_code=409, detail="This agent is already working on its source"
            )
        job = {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "state": "running",
            "step": "Starting",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }
        _jobs[agent_id] = job

    def step(text: str) -> None:
        with _jobs_lock:
            job["step"] = text
        # The sidebar shows the same line beside the agent's name.
        agent_status.note_event(
            agent_id, {"type": "status", "run_id": job["id"], "message": text}
        )

    def work() -> None:
        agent = None
        try:
            agent = _live_agent(agent_id)
            focus = str(payload.get("focus") or "").strip() or None
            if kind == "recommend":
                result = agent.recommend_self_fix(focus, on_step=step)
            else:
                recommendation = str(payload.get("recommendation") or "").strip()
                result = agent.fix_self(recommendation or None, focus, on_step=step)
            with _jobs_lock:
                job.update(state="done", result=result)
        except Exception as exc:
            logger.warning("Self-repair (%s) failed for %s: %s", kind, agent_id, exc)
            with _jobs_lock:
                job.update(state="error", error=f"{type(exc).__name__}: {exc}"[:600])
        finally:
            with _jobs_lock:
                job["finished_at"] = time.time()
            agent_status.end_run(agent_id, job["id"])
            if agent is not None:
                try:
                    agent.close(close_memory_provider=False)
                except Exception:
                    pass

    threading.Thread(
        target=work, name=f"self-{kind}-{agent_id[:8]}", daemon=True
    ).start()
    return dict(job)


@router.post("/api/agents/{agent_id}/source/recommend")
async def api_agent_source_recommend(agent_id: str, request: Request):
    """Have the coding model find the cause of the agent's errors."""
    payload = await _json_object(request, required=False)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    if not _enabled(agent):
        raise HTTPException(
            status_code=409, detail="This agent has no copy of its source"
        )
    return {"ok": True, "job": _start(agent_id, "recommend", payload)}


@router.post("/api/agents/{agent_id}/source/fix")
async def api_agent_source_fix(agent_id: str, request: Request):
    """Have the coding model make a fix in the agent's copy and keep it."""
    _writable()
    payload = await _json_object(request, required=False)
    agent = await asyncio.to_thread(_load_agent, agent_id)
    if not _enabled(agent):
        raise HTTPException(
            status_code=409, detail="This agent has no copy of its source"
        )
    return {"ok": True, "job": _start(agent_id, "fix", payload)}
