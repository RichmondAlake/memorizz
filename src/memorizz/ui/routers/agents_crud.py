# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Agent CRUD pages (list, create, edit, favorite, delete).

Extracted verbatim from ``ui/app.py``: GET /agents, POST
/agents/{agent_id}/favorite, GET+POST /agents/new, GET+POST
/agents/{agent_id}/edit, the GET /agents/{agent_id} playground redirect, and
POST /agents/{agent_id}/delete, plus the form-parsing helpers used only by
these routes. Route paths, response classes, template context keys, and
behavior are unchanged. ``/agents/new`` stays registered before
``/agents/{agent_id}`` so the literal path keeps winning the match.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ..helpers import (
    DEFAULT_LLM_MODEL_BY_PROVIDER,
    DEFAULT_LLM_PROVIDER,
    _agent_created_timestamp,
    _agent_entity_memory_enabled,
    _agent_workflow_memory_enabled,
    _build_agent_nav_items,
    _build_agent_threads,
    _build_agent_tool_count_map,
    _build_browser_control_config,
    _build_internet_provider_config,
    _build_memory_types_for_agent,
    _build_self_aware_config,
    _build_skills_marketplace_provider_config,
    _extract_agent_identifier,
    _extract_agent_memory_ids,
    _extract_agent_tools,
    _get_default_llm_model,
    _get_default_llm_provider,
    _load_agent_last_run_map,
    _load_memagent_created_at_map,
    _normalize_browser_control_provider_name,
    _normalize_internet_provider_name,
    _normalize_llm_provider,
    _normalize_memory_type_values,
    _normalize_skills_marketplace_provider_name,
    _parse_bool,
    _parse_self_aware_root_paths,
    _persist_mcp_configs_to_toolbox,
    _retrieve_conversation_history,
    _sort_agents_by_last_run_desc,
    _to_text,
    _validate_browser_control_choice,
    _validate_internet_provider_choice,
    _validate_sandbox_provider_choice,
    _validate_self_aware_config,
    _validate_skills_marketplace_provider_choice,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents-crud"])

AGENT_SORT_LAST_RUN = "last_run"
AGENT_SORT_LAST_CREATED = "last_created"
AGENT_SORT_OPTIONS = {AGENT_SORT_LAST_RUN, AGENT_SORT_LAST_CREATED}
SKILL_INJECTION_ROLES = {"user", "developer"}


def _normalize_agent_sort_option(sort_by: Optional[str]) -> str:
    """Normalize sort options for the agent list page."""
    value = _to_text(sort_by).strip().lower()
    if value in AGENT_SORT_OPTIONS:
        return value
    return AGENT_SORT_LAST_RUN


def _sort_agents_by_created_at_desc(agents: List[Any]) -> List[Any]:
    """Sort agents by creation time (newest first), preserving ties."""
    if not agents:
        return []

    created_by_agent = _load_memagent_created_at_map()
    indexed = [
        (_agent_created_timestamp(agent, created_by_agent), idx, agent)
        for idx, agent in enumerate(agents)
    ]
    indexed.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in indexed]


def _build_agent_threads_map(agents: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-agent thread metadata for agents page UI controls."""
    thread_rows_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        thread_rows_by_agent[agent_id] = _build_agent_threads(agent)
    return thread_rows_by_agent


def _build_agent_recent_messages(
    agents: List[Any],
    thread_rows_by_agent: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    per_agent_limit: int = 5,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """Collect latest-thread chat previews per agent for the agents page."""
    provider = _state.get("provider")
    if not provider:
        return {}, {}

    previews: Dict[str, List[Dict[str, Any]]] = {}
    default_memory_by_agent: Dict[str, str] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        selected_memory_id = ""
        if thread_rows_by_agent:
            rows = thread_rows_by_agent.get(agent_id, [])
            if rows:
                selected_memory_id = _to_text(rows[0].get("memory_id")).strip()

        if not selected_memory_id:
            memory_ids = _extract_agent_memory_ids(agent)
            if memory_ids:
                selected_memory_id = memory_ids[0]

        if not selected_memory_id:
            continue

        default_memory_by_agent[agent_id] = selected_memory_id
        history = _retrieve_conversation_history(
            memory_id=selected_memory_id, limit=120
        )
        if not history:
            continue

        recent: List[Dict[str, Any]] = []
        for message in history:
            role = _to_text(message.get("role")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = _to_text(
                message.get("content") or message.get("text", "")
            ).strip()
            if not content:
                continue
            if len(content) > 280:
                content = f"{content[:277]}..."
            recent.append({"role": role, "content": content})

        if recent:
            previews[agent_id] = recent[-max(1, per_agent_limit) :]

    return previews, default_memory_by_agent


def _parse_memory_ids(value: Optional[str]) -> List[str]:
    """Parse a comma- or newline-delimited memory ID list."""
    if not value:
        return []
    raw_parts = value.replace("\n", ",").split(",")
    return [part.strip() for part in raw_parts if part.strip()]


def _parse_llm_config(
    provider: str, model: str, raw_json: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build an LLM config dict from form values.

    Special-cases:
    - ``local-openai`` is a UI-only sentinel for OpenAI-compatible local
      servers (llama.cpp's ``llama-server``, LM Studio, vLLM, etc). It
      collapses to ``provider=openai`` plus a ``base_url`` field so the
      saved config loads through the existing ``OpenAI`` provider.
    """
    config: Optional[Dict[str, Any]] = None
    error = None

    provider_value = (provider or "").strip().lower()
    model_value = (model or "").strip()

    is_local_openai = provider_value == "local-openai"
    if is_local_openai:
        provider_value = "openai"

    if provider_value or model_value:
        config = {}
        if model_value and not provider_value:
            provider_value = DEFAULT_LLM_PROVIDER
        if provider_value:
            config["provider"] = provider_value
        if model_value:
            if provider_value == "azure":
                config["deployment_name"] = model_value
            else:
                config["model"] = model_value
        if is_local_openai:
            # Default to llama.cpp's standard port; users can override
            # in the JSON or by including base_url in the form-level
            # config (handled below when raw_json is parsed).
            config.setdefault("base_url", "http://127.0.0.1:8080/v1")

    if raw_json and raw_json.strip():
        try:
            extra = json.loads(raw_json)
            if not isinstance(extra, dict):
                error = "LLM config JSON must be an object."
            else:
                if config is None:
                    config = extra
                else:
                    config.update(extra)
        except Exception as exc:
            error = f"Invalid LLM config JSON: {exc}"

    return config, error


def _build_continual_learning_config(
    skill_injection_role: str,
    require_shadow: bool,
    shadow_evaluation_enabled: bool = False,
    base_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Merge and validate the continual-learning controls exposed by the UI."""
    config = dict(base_config) if isinstance(base_config, dict) else {}
    raw_role = getattr(skill_injection_role, "value", skill_injection_role)
    role = _to_text(raw_role).strip().lower() or "user"
    config["skill_injection_role"] = role
    config["require_shadow"] = bool(require_shadow)
    config["shadow_evaluation_enabled"] = bool(shadow_evaluation_enabled)

    if role not in SKILL_INJECTION_ROLES:
        return config, "Learned skill authority must be user or developer."
    if role == "developer" and not require_shadow:
        return (
            config,
            "Developer-authority learned skills require shadow review before "
            "activation.",
        )
    return config, None


def _build_persona_payload(
    name: str, role: str, goals: str, background: str
) -> Optional[Dict[str, Any]]:
    """Build a minimal persona payload for persistence."""
    name_value = (name or "").strip()
    if not name_value:
        return None
    return {
        "name": name_value,
        "role": (role or "").strip() or "general",
        "goals": (goals or "").strip(),
        "background": (background or "").strip(),
    }


def _resolve_persona_for_agent(
    provider: Any,
    persona_payload: Optional[Dict[str, Any]],
    persona_id: str,
    agent_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Reconcile the form's persona payload with the PERSONAS collection.

    When a hidden ``persona_id`` is supplied (via the "Load saved persona"
    dropdown), this either reuses the existing record unchanged or applies a
    traceable :meth:`Persona.update` with ``source_type='ui_form'``. When no
    id is supplied and the payload has a name, a new record is stored. The
    returned dict (if any) is the canonical snapshot to embed on the agent
    document so the PERSONAS collection and the agent doc stay consistent.
    """
    if not persona_payload:
        return None
    if provider is None:
        return persona_payload

    from ...enums.memory_type import MemoryType
    from ...long_term.semantic.persona.persona import Persona

    persona_id = (persona_id or "").strip()

    existing_doc: Optional[Dict[str, Any]] = None
    if persona_id:
        try:
            existing_doc = provider.retrieve_by_id(
                persona_id, memory_store_type=MemoryType.PERSONAS
            )
        except Exception as exc:
            logger.warning(
                "Failed to retrieve persona %s for reconciliation: %s",
                persona_id,
                exc,
            )
            existing_doc = None

    if existing_doc:
        # Merge the stored record with form values, then diff
        persona = Persona.from_dict(existing_doc)
        if not persona._storage_id:
            persona._storage_id = persona_id

        fields_to_check = ("name", "role", "goals", "background")
        candidate_updates = {
            field: (persona_payload.get(field) or "").strip()
            for field in fields_to_check
        }
        diff = {
            field: value
            for field, value in candidate_updates.items()
            if value and value != (getattr(persona, field, "") or "").strip()
        }

        if not diff:
            # No changes — just re-embed the existing snapshot
            return persona.to_dict()

        try:
            persona.update(
                updates=diff,
                change_trigger={
                    "reason": "Persona edited via agent config form.",
                    "source_type": "ui_form",
                    "agent_id": agent_id,
                },
                provider=provider,
            )
        except Exception as exc:
            logger.error(
                "Failed to apply ui_form persona update for %s: %s",
                persona_id,
                exc,
            )
            return persona_payload
        return persona.to_dict()

    # No existing record — create a fresh Persona and store it
    try:
        persona = Persona(
            name=persona_payload.get("name", ""),
            role=persona_payload.get("role") or "general",
            goals=persona_payload.get("goals", ""),
            background=persona_payload.get("background", ""),
        )
        persona.store_persona(provider)
        return persona.to_dict()
    except Exception as exc:
        logger.warning(
            "Failed to store new persona in PERSONAS collection; embedding raw payload: %s",
            exc,
        )
        return persona_payload


def _extract_agent_id(result: Any, fallback: Optional[str]) -> Optional[str]:
    """Extract agent_id from provider return values."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return (
            result.get("agent_id")
            or result.get("_id")
            or result.get("agentId")
            or fallback
        )
    if hasattr(result, "agent_id"):
        return getattr(result, "agent_id")
    return fallback


def _build_agent_form_data(agent: Any) -> Dict[str, Any]:
    """Prepare agent data for edit form rendering."""
    from ...memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS

    persona = getattr(agent, "persona", None)
    if isinstance(persona, dict):
        persona_name = persona.get("name", "")
        persona_role = persona.get("role", "")
        persona_goals = persona.get("goals", "")
        persona_background = persona.get("background", "")
        persona_id_value = (
            persona.get("storage_id") or persona.get("_id") or persona.get("id") or ""
        )
    else:
        persona_name = getattr(persona, "name", "") if persona else ""
        persona_role = getattr(persona, "role", "") if persona else ""
        persona_goals = getattr(persona, "goals", "") if persona else ""
        persona_background = getattr(persona, "background", "") if persona else ""
        persona_id_value = getattr(persona, "_storage_id", "") if persona else ""
    persona_id_value = str(persona_id_value) if persona_id_value else ""

    llm_config = getattr(agent, "llm_config", None) or {}
    llm_provider = _normalize_llm_provider(llm_config.get("provider", "openai"))
    llm_model = (
        llm_config.get("model")
        or llm_config.get("deployment_name")
        or _get_default_llm_model(llm_provider)
    )
    extra_config = {
        key: value
        for key, value in llm_config.items()
        if key not in {"provider", "model", "deployment_name"}
    }
    llm_config_json = json.dumps(extra_config, indent=2) if extra_config else ""

    memory_ids = getattr(agent, "memory_ids", None) or []

    # Sandbox provider
    sandbox_val = getattr(agent, "sandbox_provider", None) or ""
    if isinstance(sandbox_val, dict):
        sandbox_val = sandbox_val.get("provider", "")
    browser_control_val = _normalize_browser_control_provider_name(
        getattr(agent, "browser_control", None)
    )
    internet_val = _normalize_internet_provider_name(
        getattr(agent, "internet_access_provider", None)
    )
    skills_marketplace_val = _normalize_skills_marketplace_provider_name(
        getattr(agent, "skills_marketplace_provider", None)
    )
    memory_types = _normalize_memory_type_values(getattr(agent, "memory_types", None))
    self_aware_enabled = bool(getattr(agent, "self_aware", False))
    self_aware_config = getattr(agent, "self_aware_config", None)
    if not isinstance(self_aware_config, dict):
        self_aware_config = {}
    self_aware_root_paths = self_aware_config.get("root_paths")
    if not isinstance(self_aware_root_paths, list):
        self_aware_root_paths = []
    self_aware_root_paths_text = "\n".join(
        _to_text(path).strip()
        for path in self_aware_root_paths
        if _to_text(path).strip()
    )
    continual_learning_config = getattr(agent, "continual_learning_config", None)
    if not isinstance(continual_learning_config, dict):
        continual_learning_config = {}
    raw_skill_role = continual_learning_config.get("skill_injection_role", "user")
    raw_skill_role = getattr(raw_skill_role, "value", raw_skill_role)
    skill_injection_role = _to_text(raw_skill_role).strip().lower()
    if skill_injection_role not in SKILL_INJECTION_ROLES:
        skill_injection_role = "user"
    meta_harness_mode = (
        _to_text(getattr(agent, "meta_harness_mode", "")).strip().lower()
    )
    if meta_harness_mode not in {"delegate", "runtime"}:
        meta_harness_mode = ""
    harness_config = getattr(agent, "harness_config", None)
    if not isinstance(harness_config, dict):
        harness_config = {}

    return {
        "agent_id": getattr(agent, "agent_id", ""),
        "agent_name": _to_text(getattr(agent, "name", "")).strip(),
        "instruction": getattr(agent, "instruction", None) or DEFAULT_INSTRUCTION,
        "application_mode": getattr(agent, "application_mode", None) or "assistant",
        "memory_types": memory_types,
        "enable_entity_memory": _agent_entity_memory_enabled(agent),
        "enable_workflow_memory": _agent_workflow_memory_enabled(agent),
        "max_steps": getattr(agent, "max_steps", None)
        if getattr(agent, "max_steps", None) is not None
        else DEFAULT_MAX_STEPS,
        "tool_access": getattr(agent, "tool_access", None) or "private",
        "semantic_cache": bool(getattr(agent, "semantic_cache", False)),
        "continual_learning": bool(getattr(agent, "continual_learning", False)),
        "learning_control_plane": bool(getattr(agent, "learning_control_plane", False)),
        "skill_injection_role": skill_injection_role,
        "continual_learning_require_shadow": bool(
            continual_learning_config.get("require_shadow", False)
        ),
        "continual_learning_shadow_evaluation": bool(
            continual_learning_config.get("shadow_evaluation_enabled", False)
        ),
        "is_favorite": bool(getattr(agent, "is_favorite", False)),
        "memory_ids_raw": ", ".join(memory_ids),
        "persona_id": persona_id_value,
        "persona_name": persona_name,
        "persona_role": persona_role,
        "persona_goals": persona_goals,
        "persona_background": persona_background,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_config_json": llm_config_json,
        "sandbox_provider": sandbox_val,
        "browser_control_provider": browser_control_val,
        "meta_harness_mode": meta_harness_mode,
        "default_harness": _to_text(getattr(agent, "default_harness", "auto")).strip()
        or "auto",
        "harness_workspace": _to_text(harness_config.get("workspace")).strip(),
        "internet_provider": internet_val,
        "skills_marketplace_provider": skills_marketplace_val,
        "skill_paths": getattr(agent, "skill_paths", None) or [],
        "mcp_servers": getattr(agent, "mcp_servers", None) or [],
        "self_aware": self_aware_enabled,
        "self_aware_root_paths": self_aware_root_paths_text,
        "self_aware_allow_writes": bool(self_aware_config.get("allow_writes", False)),
        "self_aware_allow_deletes": bool(self_aware_config.get("allow_deletes", False)),
        "automations_enabled": bool(getattr(agent, "automations_enabled", True)),
        "default_timezone": _to_text(getattr(agent, "default_timezone", "")).strip(),
        "whatsapp_enabled": bool(getattr(agent, "whatsapp_enabled", False)),
        "whatsapp_welcome_message": (getattr(agent, "whatsapp_config", None) or {}).get(
            "welcome_message", ""
        ),
        "agent_tools": _extract_agent_tools(agent),
    }


@router.get("/agents", response_class=HTMLResponse)
async def agents_list(request: Request):
    """Show list of all agents."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    sort_by = _normalize_agent_sort_option(request.query_params.get("sort_by"))
    agents = []
    last_run_by_agent: Dict[str, float] = {}
    try:
        agents = _state["provider"].list_memagents()
        last_run_by_agent = _load_agent_last_run_map(agents)
        if sort_by == AGENT_SORT_LAST_CREATED:
            agents = _sort_agents_by_created_at_desc(agents)
        else:
            agents = _sort_agents_by_last_run_desc(
                agents, last_run_by_agent=last_run_by_agent
            )
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")

    agent_threads = _build_agent_threads_map(agents)
    agent_recent_messages, agent_default_memory_ids = _build_agent_recent_messages(
        agents,
        thread_rows_by_agent=agent_threads,
        per_agent_limit=5,
    )
    agent_tool_counts = _build_agent_tool_count_map(agents)

    # Get active WhatsApp agent
    active_whatsapp_agent_id = None
    try:
        from memorizz.channels.whatsapp.settings import WhatsAppSettings

        settings = WhatsAppSettings(_state["provider"])
        active_whatsapp_agent_id = settings.get_active_agent_id()
    except Exception:
        pass

    return templates.TemplateResponse(
        "agents.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents": agents,
            "agent_recent_messages": agent_recent_messages,
            "agent_default_memory_ids": agent_default_memory_ids,
            "agent_threads": agent_threads,
            "agent_tool_counts": agent_tool_counts,
            "sort_by": sort_by,
            "agents_nav": _build_agent_nav_items(
                active_agent_id=None,
                agents=agents,
                last_run_by_agent=last_run_by_agent,
            ),
            "active_page": "agents",
            "active_whatsapp_agent_id": active_whatsapp_agent_id,
        },
    )


@router.post("/agents/{agent_id}/favorite")
async def agent_toggle_favorite(
    agent_id: str,
    redirect_to: str = Form("/agents"),
    is_favorite: Optional[str] = Form(None),
):
    """Toggle or set an agent favorite flag."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from ...memagent.models import MemAgentModel

    existing = _state["provider"].retrieve_memagent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    if is_favorite is None:
        next_value = not bool(getattr(existing, "is_favorite", False))
    else:
        next_value = _parse_bool(is_favorite)

    updated = MemAgentModel(
        agent_id=agent_id,
        name=getattr(existing, "name", None),
        instruction=getattr(existing, "instruction", None),
        application_mode=getattr(existing, "application_mode", "assistant"),
        memory_types=getattr(existing, "memory_types", None),
        max_steps=getattr(existing, "max_steps", 20),
        tool_access=getattr(existing, "tool_access", "private"),
        semantic_cache=bool(getattr(existing, "semantic_cache", False)),
        memory_ids=getattr(existing, "memory_ids", None),
        persona=getattr(existing, "persona", None),
        llm_config=getattr(existing, "llm_config", None),
        tools=getattr(existing, "tools", None),
        delegates=getattr(existing, "delegates", None),
        embedding_config=getattr(existing, "embedding_config", None),
        semantic_cache_config=getattr(existing, "semantic_cache_config", None),
        tool_result_policy=getattr(existing, "tool_result_policy", None),
        context_policy=getattr(existing, "context_policy", None),
        retrieval_policy=getattr(existing, "retrieval_policy", None),
        delegation_config=getattr(existing, "delegation_config", None),
        skill_retrieval=bool(getattr(existing, "skill_retrieval", False)),
        skill_retrieval_config=getattr(existing, "skill_retrieval_config", None),
        semantic_layer_config=getattr(existing, "semantic_layer_config", None),
        context_window_tokens=getattr(existing, "context_window_tokens", None),
        is_favorite=next_value,
        internet_access_provider=getattr(existing, "internet_access_provider", None),
        internet_access_config=getattr(existing, "internet_access_config", None),
        skills_marketplace_provider=getattr(
            existing, "skills_marketplace_provider", None
        ),
        skills_marketplace_config=getattr(existing, "skills_marketplace_config", None),
        knowledge_base_ids=getattr(existing, "knowledge_base_ids", None),
        sandbox_provider=getattr(existing, "sandbox_provider", None),
        browser_control=getattr(existing, "browser_control", None),
        skill_paths=getattr(existing, "skill_paths", None),
        mcp_servers=getattr(existing, "mcp_servers", None),
        self_aware=bool(getattr(existing, "self_aware", False)),
        self_aware_config=getattr(existing, "self_aware_config", None),
        continual_learning=bool(getattr(existing, "continual_learning", False)),
        continual_learning_config=getattr(existing, "continual_learning_config", None),
        learning_control_plane=bool(getattr(existing, "learning_control_plane", False)),
        learning_control_plane_config=getattr(
            existing, "learning_control_plane_config", None
        ),
        automations_enabled=bool(getattr(existing, "automations_enabled", True)),
        default_timezone=getattr(existing, "default_timezone", None),
        whatsapp_enabled=bool(getattr(existing, "whatsapp_enabled", False)),
        whatsapp_config=getattr(existing, "whatsapp_config", None),
    )

    try:
        _state["provider"].store_memagent(updated)
    except Exception as exc:
        logger.error("Failed to update favorite for agent %s: %s", agent_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update favorite")

    redirect_target = _to_text(redirect_to).strip() or "/agents"
    if not redirect_target.startswith("/"):
        redirect_target = f"/agents/{agent_id}/playground"
    return RedirectResponse(url=redirect_target, status_code=302)


@router.get("/agents/new", response_class=HTMLResponse)
async def agent_create_page(request: Request):
    """Show form to create a new agent."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from ...memagent.constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS

    default_llm_provider = _get_default_llm_provider()
    default_llm_model = _get_default_llm_model(default_llm_provider)
    default_skills_marketplace_provider = (
        _normalize_skills_marketplace_provider_name(
            os.environ.get("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
        )
        or ""
    )

    return templates.TemplateResponse(
        "agent_form.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "active_page": "agents",
            "form_title": "Create Agent",
            "form_action": "/agents/new",
            "is_edit": False,
            "error": None,
            "agent_id": "",
            "agent_name": "",
            "instruction": DEFAULT_INSTRUCTION,
            "application_mode": "assistant",
            "max_steps": DEFAULT_MAX_STEPS,
            "tool_access": "private",
            "semantic_cache": False,
            "continual_learning": False,
            "learning_control_plane": False,
            "skill_injection_role": "user",
            "continual_learning_require_shadow": False,
            "continual_learning_shadow_evaluation": False,
            "memory_ids_raw": "",
            "persona_name": "",
            "persona_role": "",
            "persona_goals": "",
            "persona_background": "",
            "llm_provider": default_llm_provider,
            "llm_model": default_llm_model,
            "llm_config_json": "",
            "sandbox_provider": os.environ.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", ""),
            "browser_control_provider": os.environ.get(
                "MEMORIZZ_BROWSER_CONTROL_PROVIDER", ""
            ),
            "meta_harness_mode": "",
            "default_harness": "auto",
            "harness_workspace": "",
            "internet_provider": os.environ.get(
                "MEMORIZZ_DEFAULT_INTERNET_PROVIDER", ""
            ),
            "skills_marketplace_provider": default_skills_marketplace_provider,
            "enable_entity_memory": True,
            "enable_workflow_memory": False,
            "self_aware": False,
            "self_aware_root_paths": "",
            "self_aware_allow_writes": False,
            "self_aware_allow_deletes": False,
            "automations_enabled": True,
            "default_timezone": _to_text(
                os.environ.get("MEMORIZZ_DEFAULT_TIMEZONE", "")
            ).strip(),
            "agent_tools": [],
        },
    )


@router.post("/agents/new", response_class=HTMLResponse)
async def agent_create_submit(
    request: Request,
    instruction: str = Form(""),
    application_mode: str = Form("assistant"),
    enable_entity_memory: Optional[str] = Form(None),
    enable_workflow_memory: Optional[str] = Form(None),
    max_steps: int = Form(20),
    tool_access: str = Form("private"),
    semantic_cache: Optional[str] = Form(None),
    continual_learning: Optional[str] = Form(None),
    learning_control_plane: Optional[str] = Form(None),
    skill_injection_role: str = Form("user"),
    continual_learning_require_shadow: Optional[str] = Form(None),
    continual_learning_shadow_evaluation: Optional[str] = Form(None),
    memory_ids: str = Form(""),
    agent_name: str = Form(""),
    persona_id: str = Form(""),
    persona_name: str = Form(""),
    persona_role: str = Form(""),
    persona_goals: str = Form(""),
    persona_background: str = Form(""),
    llm_provider: str = Form(DEFAULT_LLM_PROVIDER),
    llm_model: str = Form(DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]),
    llm_config_json: str = Form(""),
    sandbox_provider: str = Form(""),
    browser_control_provider: str = Form(""),
    meta_harness_mode: str = Form(""),
    default_harness: str = Form("auto"),
    harness_workspace: str = Form(""),
    internet_provider: str = Form(""),
    skills_marketplace_provider: str = Form(""),
    self_aware: Optional[str] = Form(None),
    self_aware_root_paths: str = Form(""),
    self_aware_allow_writes: Optional[str] = Form(None),
    self_aware_allow_deletes: Optional[str] = Form(None),
    automations_enabled: Optional[str] = Form(None),
    default_timezone: str = Form(""),
    whatsapp_enabled: Optional[str] = Form(None),
    whatsapp_welcome_message: str = Form(""),
):
    """Create a new agent using the configured memory provider."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from ...memagent.constants import DEFAULT_INSTRUCTION
    from ...memagent.models import MemAgentModel

    error = None
    llm_config, llm_error = _parse_llm_config(llm_provider, llm_model, llm_config_json)
    if llm_error:
        error = llm_error

    persona_payload = _build_persona_payload(
        persona_name, persona_role, persona_goals, persona_background
    )
    agent_name_value = _to_text(agent_name).strip() or None
    memory_id_list = _parse_memory_ids(memory_ids)
    semantic_cache_enabled = _parse_bool(semantic_cache)
    continual_learning_enabled = _parse_bool(continual_learning)
    learning_control_plane_enabled = _parse_bool(learning_control_plane)
    learning_control_plane_config_value = {"enabled": learning_control_plane_enabled}
    continual_learning_require_shadow_value = _parse_bool(
        continual_learning_require_shadow
    )
    continual_learning_shadow_evaluation_value = _parse_bool(
        continual_learning_shadow_evaluation
    )
    (
        continual_learning_config_value,
        continual_learning_config_error,
    ) = _build_continual_learning_config(
        skill_injection_role=skill_injection_role,
        require_shadow=continual_learning_require_shadow_value,
        shadow_evaluation_enabled=continual_learning_shadow_evaluation_value,
    )
    if not error and continual_learning_config_error:
        error = continual_learning_config_error
    enable_entity_memory_value = _parse_bool(enable_entity_memory)
    enable_workflow_memory_value = _parse_bool(enable_workflow_memory)
    self_aware_enabled = _parse_bool(self_aware)
    self_aware_allow_writes_value = _parse_bool(self_aware_allow_writes)
    self_aware_allow_deletes_value = _parse_bool(self_aware_allow_deletes)
    self_aware_root_paths_value = _parse_self_aware_root_paths(self_aware_root_paths)
    self_aware_config_value = _build_self_aware_config(
        root_paths=self_aware_root_paths_value,
        allow_writes=self_aware_allow_writes_value,
        allow_deletes=self_aware_allow_deletes_value,
    )
    automations_enabled_value = _parse_bool(automations_enabled)
    default_timezone_value = _to_text(default_timezone).strip() or None
    if not error and default_timezone_value:
        try:
            from ...automation.schedule import validate_timezone_name

            validate_timezone_name(default_timezone_value)
        except Exception as exc:
            error = str(exc)

    # WhatsApp configuration
    whatsapp_enabled_value = _parse_bool(whatsapp_enabled)
    whatsapp_config_value = None
    if whatsapp_enabled_value:
        whatsapp_config_value = {
            "welcome_message": _to_text(whatsapp_welcome_message).strip() or None,
            "auto_reply_enabled": True,
            "timeout_seconds": 60,
        }

    memory_types_value = _build_memory_types_for_agent(
        application_mode=application_mode or "assistant",
        enable_entity_memory=enable_entity_memory_value,
        enable_workflow_memory=enable_workflow_memory_value,
    )
    skills_marketplace_provider_value = (
        _normalize_skills_marketplace_provider_name(skills_marketplace_provider) or ""
    )
    browser_control_provider_value = _normalize_browser_control_provider_name(
        browser_control_provider
    )
    meta_harness_mode_value = _to_text(meta_harness_mode).strip().lower()
    if meta_harness_mode_value not in {"", "delegate", "runtime"} and not error:
        error = "Meta-harness mode must be disabled, delegate, or runtime"
    default_harness_value = (
        _to_text(default_harness).strip().lower().replace("_", "-") or "auto"
    )
    if (
        default_harness_value
        not in {"auto", "codex", "claude-code", "openhands", "native"}
        and not error
    ):
        error = "Default harness must be auto, codex, claude-code, openhands, or native"
    harness_workspace_value = _to_text(harness_workspace).strip()
    if harness_workspace_value and not error:
        try:
            resolved_harness_workspace = (
                Path(harness_workspace_value).expanduser().resolve(strict=True)
            )
            if not resolved_harness_workspace.is_dir():
                raise ValueError("not a directory")
            harness_workspace_value = str(resolved_harness_workspace)
        except (OSError, ValueError):
            error = "Harness workspace must be an existing directory"
    harness_config_value = (
        {
            "workspace": harness_workspace_value,
            "permissions": {"allowed_roots": [harness_workspace_value]},
        }
        if harness_workspace_value
        else {}
    )

    if error:
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "active_page": "agents",
                "form_title": "Create Agent",
                "form_action": "/agents/new",
                "is_edit": False,
                "error": error,
                "agent_id": "",
                "agent_name": agent_name,
                "instruction": instruction,
                "application_mode": application_mode,
                "max_steps": max_steps,
                "tool_access": tool_access,
                "semantic_cache": semantic_cache_enabled,
                "continual_learning": continual_learning_enabled,
                "learning_control_plane": learning_control_plane_enabled,
                "skill_injection_role": skill_injection_role,
                "continual_learning_require_shadow": (
                    continual_learning_require_shadow_value
                ),
                "continual_learning_shadow_evaluation": (
                    continual_learning_shadow_evaluation_value
                ),
                "memory_ids_raw": memory_ids,
                "persona_id": persona_id,
                "persona_name": persona_name,
                "persona_role": persona_role,
                "persona_goals": persona_goals,
                "persona_background": persona_background,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_config_json": llm_config_json,
                "sandbox_provider": sandbox_provider,
                "browser_control_provider": browser_control_provider_value,
                "meta_harness_mode": meta_harness_mode_value,
                "default_harness": default_harness_value,
                "harness_workspace": harness_workspace_value,
                "internet_provider": internet_provider,
                "skills_marketplace_provider": skills_marketplace_provider_value,
                "enable_entity_memory": enable_entity_memory_value,
                "enable_workflow_memory": enable_workflow_memory_value,
                "self_aware": self_aware_enabled,
                "self_aware_root_paths": self_aware_root_paths,
                "self_aware_allow_writes": self_aware_allow_writes_value,
                "self_aware_allow_deletes": self_aware_allow_deletes_value,
                "automations_enabled": automations_enabled_value,
                "default_timezone": default_timezone,
                "agent_tools": [],
            },
        )

    instruction_value = instruction.strip() if instruction else ""
    sandbox_value = sandbox_provider.strip() if sandbox_provider else None
    browser_control_value = browser_control_provider_value or None
    browser_control_config = _build_browser_control_config(browser_control_value)
    internet_value = _normalize_internet_provider_name(internet_provider) or None
    internet_config = _build_internet_provider_config(internet_value)
    skills_marketplace_value = skills_marketplace_provider_value or None
    skills_marketplace_config = _build_skills_marketplace_provider_config(
        skills_marketplace_value
    )
    self_aware_validation_error = _validate_self_aware_config(self_aware_config_value)
    sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
    browser_control_validation_error = _validate_browser_control_choice(
        browser_control_value, browser_control_config
    )
    internet_validation_error = _validate_internet_provider_choice(
        internet_value, internet_config
    )
    skills_marketplace_validation_error = _validate_skills_marketplace_provider_choice(
        skills_marketplace_value,
        skills_marketplace_config,
    )
    if (
        sandbox_validation_error
        or browser_control_validation_error
        or internet_validation_error
        or skills_marketplace_validation_error
        or self_aware_validation_error
    ):
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "active_page": "agents",
                "form_title": "Create Agent",
                "form_action": "/agents/new",
                "is_edit": False,
                "error": (
                    sandbox_validation_error
                    or browser_control_validation_error
                    or internet_validation_error
                    or skills_marketplace_validation_error
                    or self_aware_validation_error
                ),
                "agent_id": "",
                "agent_name": agent_name,
                "instruction": instruction,
                "application_mode": application_mode,
                "max_steps": max_steps,
                "tool_access": tool_access,
                "semantic_cache": semantic_cache_enabled,
                "continual_learning": continual_learning_enabled,
                "learning_control_plane": learning_control_plane_enabled,
                "skill_injection_role": skill_injection_role,
                "continual_learning_require_shadow": (
                    continual_learning_require_shadow_value
                ),
                "continual_learning_shadow_evaluation": (
                    continual_learning_shadow_evaluation_value
                ),
                "memory_ids_raw": memory_ids,
                "persona_id": persona_id,
                "persona_name": persona_name,
                "persona_role": persona_role,
                "persona_goals": persona_goals,
                "persona_background": persona_background,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_config_json": llm_config_json,
                "sandbox_provider": sandbox_provider,
                "browser_control_provider": browser_control_provider_value,
                "meta_harness_mode": meta_harness_mode_value,
                "default_harness": default_harness_value,
                "harness_workspace": harness_workspace_value,
                "internet_provider": internet_provider,
                "skills_marketplace_provider": skills_marketplace_provider_value,
                "enable_entity_memory": enable_entity_memory_value,
                "enable_workflow_memory": enable_workflow_memory_value,
                "self_aware": self_aware_enabled,
                "self_aware_root_paths": self_aware_root_paths,
                "self_aware_allow_writes": self_aware_allow_writes_value,
                "self_aware_allow_deletes": self_aware_allow_deletes_value,
                "automations_enabled": automations_enabled_value,
                "default_timezone": default_timezone,
                "agent_tools": [],
            },
        )

    # Reconcile persona with PERSONAS collection: reuse/update linked
    # record, or store a new one so it shows up in the saved-personas picker.
    persona_payload = _resolve_persona_for_agent(
        _state["provider"],
        persona_payload,
        persona_id,
        agent_id=None,
    )

    memagent = MemAgentModel(
        name=agent_name_value,
        instruction=instruction_value or DEFAULT_INSTRUCTION,
        application_mode=application_mode or "assistant",
        memory_types=memory_types_value,
        max_steps=max_steps if max_steps is not None else 20,
        tool_access=tool_access or "private",
        semantic_cache=semantic_cache_enabled,
        is_favorite=False,
        memory_ids=memory_id_list or None,
        persona=persona_payload,
        llm_config=llm_config,
        sandbox_provider=sandbox_value,
        browser_control=browser_control_config,
        meta_harness=bool(meta_harness_mode_value),
        meta_harness_mode=meta_harness_mode_value or None,
        default_harness=default_harness_value,
        harness_config=harness_config_value,
        internet_access_provider=internet_value,
        internet_access_config=internet_config,
        skills_marketplace_provider=skills_marketplace_value,
        skills_marketplace_config=skills_marketplace_config,
        self_aware=self_aware_enabled,
        self_aware_config=self_aware_config_value,
        continual_learning=continual_learning_enabled,
        continual_learning_config=continual_learning_config_value,
        learning_control_plane=learning_control_plane_enabled,
        learning_control_plane_config=learning_control_plane_config_value,
        automations_enabled=automations_enabled_value,
        default_timezone=default_timezone_value,
        whatsapp_enabled=whatsapp_enabled_value,
        whatsapp_config=whatsapp_config_value,
    )

    try:
        result = _state["provider"].store_memagent(memagent)
        agent_id = _extract_agent_id(result, memagent.agent_id)
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        error = str(e)
        agent_id = None

    if error or not agent_id:
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "active_page": "agents",
                "form_title": "Create Agent",
                "form_action": "/agents/new",
                "is_edit": False,
                "error": error or "Failed to create agent.",
                "agent_id": "",
                "agent_name": agent_name,
                "instruction": instruction,
                "application_mode": application_mode,
                "max_steps": max_steps,
                "tool_access": tool_access,
                "semantic_cache": semantic_cache_enabled,
                "continual_learning": continual_learning_enabled,
                "learning_control_plane": learning_control_plane_enabled,
                "skill_injection_role": skill_injection_role,
                "continual_learning_require_shadow": (
                    continual_learning_require_shadow_value
                ),
                "continual_learning_shadow_evaluation": (
                    continual_learning_shadow_evaluation_value
                ),
                "memory_ids_raw": memory_ids,
                "persona_id": persona_id,
                "persona_name": persona_name,
                "persona_role": persona_role,
                "persona_goals": persona_goals,
                "persona_background": persona_background,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_config_json": llm_config_json,
                "sandbox_provider": sandbox_provider,
                "browser_control_provider": browser_control_provider_value,
                "meta_harness_mode": meta_harness_mode_value,
                "default_harness": default_harness_value,
                "harness_workspace": harness_workspace_value,
                "internet_provider": internet_provider,
                "skills_marketplace_provider": skills_marketplace_provider_value,
                "enable_entity_memory": enable_entity_memory_value,
                "enable_workflow_memory": enable_workflow_memory_value,
                "self_aware": self_aware_enabled,
                "self_aware_root_paths": self_aware_root_paths,
                "self_aware_allow_writes": self_aware_allow_writes_value,
                "self_aware_allow_deletes": self_aware_allow_deletes_value,
                "automations_enabled": automations_enabled_value,
                "default_timezone": default_timezone,
                "agent_tools": [],
            },
        )

    return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)


@router.get("/agents/{agent_id}/edit", response_class=HTMLResponse)
async def agent_edit_page(request: Request, agent_id: str):
    """Show form to edit an agent."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agent = _state["provider"].retrieve_memagent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    form_data = _build_agent_form_data(agent)

    return templates.TemplateResponse(
        "agent_form.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
            "active_agent_id": agent_id,
            "active_page": "agents",
            "form_title": "Edit Agent",
            "form_action": f"/agents/{agent_id}/edit",
            "is_edit": True,
            "error": None,
            **form_data,
        },
    )


@router.post("/agents/{agent_id}/edit", response_class=HTMLResponse)
async def agent_edit_submit(
    request: Request,
    agent_id: str,
    instruction: str = Form(""),
    application_mode: str = Form("assistant"),
    enable_entity_memory: Optional[str] = Form(None),
    enable_workflow_memory: Optional[str] = Form(None),
    max_steps: int = Form(20),
    tool_access: str = Form("private"),
    semantic_cache: Optional[str] = Form(None),
    continual_learning: Optional[str] = Form(None),
    learning_control_plane: Optional[str] = Form(None),
    skill_injection_role: str = Form("user"),
    continual_learning_require_shadow: Optional[str] = Form(None),
    continual_learning_shadow_evaluation: Optional[str] = Form(None),
    memory_ids: str = Form(""),
    agent_name: str = Form(""),
    persona_id: str = Form(""),
    persona_name: str = Form(""),
    persona_role: str = Form(""),
    persona_goals: str = Form(""),
    persona_background: str = Form(""),
    llm_provider: str = Form(DEFAULT_LLM_PROVIDER),
    llm_model: str = Form(DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]),
    llm_config_json: str = Form(""),
    sandbox_provider: str = Form(""),
    browser_control_provider: str = Form(""),
    meta_harness_mode: str = Form(""),
    default_harness: str = Form("auto"),
    harness_workspace: str = Form(""),
    internet_provider: str = Form(""),
    skills_marketplace_provider: Optional[str] = Form(None),
    self_aware: Optional[str] = Form(None),
    self_aware_root_paths: str = Form(""),
    self_aware_allow_writes: Optional[str] = Form(None),
    self_aware_allow_deletes: Optional[str] = Form(None),
    automations_enabled: Optional[str] = Form(None),
    default_timezone: str = Form(""),
    whatsapp_enabled: Optional[str] = Form(None),
    whatsapp_welcome_message: str = Form(""),
):
    """Update an existing agent."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from ...memagent.constants import DEFAULT_INSTRUCTION
    from ...memagent.models import MemAgentModel

    existing = _state["provider"].retrieve_memagent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    error = None
    llm_config, llm_error = _parse_llm_config(llm_provider, llm_model, llm_config_json)
    if llm_error:
        error = llm_error

    persona_payload = _build_persona_payload(
        persona_name, persona_role, persona_goals, persona_background
    )
    agent_name_value = _to_text(agent_name).strip() or None
    memory_id_list = _parse_memory_ids(memory_ids)
    semantic_cache_enabled = _parse_bool(semantic_cache)
    continual_learning_enabled = _parse_bool(continual_learning)
    learning_control_plane_enabled = _parse_bool(learning_control_plane)
    existing_learning_control_plane_config = getattr(
        existing, "learning_control_plane_config", None
    )
    learning_control_plane_config_value = (
        dict(existing_learning_control_plane_config)
        if isinstance(existing_learning_control_plane_config, dict)
        else {}
    )
    learning_control_plane_config_value["enabled"] = learning_control_plane_enabled
    continual_learning_require_shadow_value = _parse_bool(
        continual_learning_require_shadow
    )
    continual_learning_shadow_evaluation_value = _parse_bool(
        continual_learning_shadow_evaluation
    )
    existing_continual_learning_config = getattr(
        existing, "continual_learning_config", None
    )
    (
        continual_learning_config_value,
        continual_learning_config_error,
    ) = _build_continual_learning_config(
        skill_injection_role=skill_injection_role,
        require_shadow=continual_learning_require_shadow_value,
        shadow_evaluation_enabled=continual_learning_shadow_evaluation_value,
        base_config=existing_continual_learning_config,
    )
    if not error and continual_learning_config_error:
        error = continual_learning_config_error
    enable_entity_memory_value = _parse_bool(enable_entity_memory)
    enable_workflow_memory_value = _parse_bool(enable_workflow_memory)
    self_aware_enabled_value = _parse_bool(self_aware)
    self_aware_allow_writes_value = _parse_bool(self_aware_allow_writes)
    self_aware_allow_deletes_value = _parse_bool(self_aware_allow_deletes)
    self_aware_root_paths_value = _parse_self_aware_root_paths(self_aware_root_paths)
    existing_self_aware_config = getattr(existing, "self_aware_config", None)
    if not isinstance(existing_self_aware_config, dict):
        existing_self_aware_config = None
    self_aware_config_value = _build_self_aware_config(
        root_paths=self_aware_root_paths_value,
        allow_writes=self_aware_allow_writes_value,
        allow_deletes=self_aware_allow_deletes_value,
        base_config=existing_self_aware_config,
    )
    automations_enabled_value = _parse_bool(automations_enabled)
    default_timezone_value = _to_text(default_timezone).strip() or None
    if not error and default_timezone_value:
        try:
            from ...automation.schedule import validate_timezone_name

            validate_timezone_name(default_timezone_value)
        except Exception as exc:
            error = str(exc)

    # WhatsApp configuration
    whatsapp_enabled_value = _parse_bool(whatsapp_enabled)
    whatsapp_config_value = None
    if whatsapp_enabled_value:
        whatsapp_config_value = {
            "welcome_message": _to_text(whatsapp_welcome_message).strip() or None,
            "auto_reply_enabled": True,
            "timeout_seconds": 60,
        }

    memory_types_value = _build_memory_types_for_agent(
        application_mode=application_mode
        or getattr(existing, "application_mode", "assistant"),
        enable_entity_memory=enable_entity_memory_value,
        enable_workflow_memory=enable_workflow_memory_value,
        existing_memory_types=getattr(existing, "memory_types", None),
    )
    self_aware_validation_error = _validate_self_aware_config(self_aware_config_value)
    if skills_marketplace_provider is None:
        skills_marketplace_provider_value = (
            _normalize_skills_marketplace_provider_name(
                getattr(existing, "skills_marketplace_provider", None)
            )
            or ""
        )
    else:
        skills_marketplace_provider_value = (
            _normalize_skills_marketplace_provider_name(skills_marketplace_provider)
            or ""
        )

    skills_marketplace_value = skills_marketplace_provider_value or None
    existing_skills_provider = _normalize_skills_marketplace_provider_name(
        getattr(existing, "skills_marketplace_provider", None)
    )
    skills_marketplace_base_config = (
        getattr(existing, "skills_marketplace_config", None)
        if skills_marketplace_value
        and existing_skills_provider == skills_marketplace_value
        else None
    )
    skills_marketplace_config_value = _build_skills_marketplace_provider_config(
        skills_marketplace_value,
        skills_marketplace_base_config,
    )
    browser_control_provider_value = _normalize_browser_control_provider_name(
        browser_control_provider
    )
    existing_browser_control = getattr(existing, "browser_control", None)
    existing_browser_provider = _normalize_browser_control_provider_name(
        existing_browser_control
    )
    browser_control_config_value = _build_browser_control_config(
        browser_control_provider_value or None,
        existing_browser_control
        if browser_control_provider_value == existing_browser_provider
        and isinstance(existing_browser_control, dict)
        else None,
    )
    meta_harness_mode_value = _to_text(meta_harness_mode).strip().lower()
    if meta_harness_mode_value not in {"", "delegate", "runtime"} and not error:
        error = "Meta-harness mode must be disabled, delegate, or runtime"
    default_harness_value = (
        _to_text(default_harness).strip().lower().replace("_", "-") or "auto"
    )
    if (
        default_harness_value
        not in {"auto", "codex", "claude-code", "openhands", "native"}
        and not error
    ):
        error = "Default harness must be auto, codex, claude-code, openhands, or native"
    harness_workspace_value = _to_text(harness_workspace).strip()
    if harness_workspace_value and not error:
        try:
            resolved_harness_workspace = (
                Path(harness_workspace_value).expanduser().resolve(strict=True)
            )
            if not resolved_harness_workspace.is_dir():
                raise ValueError("not a directory")
            harness_workspace_value = str(resolved_harness_workspace)
        except (OSError, ValueError):
            error = "Harness workspace must be an existing directory"
    existing_harness_config = getattr(existing, "harness_config", None)
    harness_config_value = (
        dict(existing_harness_config)
        if isinstance(existing_harness_config, dict)
        else {}
    )
    if harness_workspace_value:
        harness_config_value["workspace"] = harness_workspace_value
        permissions_value = dict(harness_config_value.get("permissions") or {})
        permissions_value["allowed_roots"] = [harness_workspace_value]
        harness_config_value["permissions"] = permissions_value
    else:
        harness_config_value.pop("workspace", None)
        harness_config_value.pop("permissions", None)
    form_override_data = {
        "instruction": instruction,
        "application_mode": application_mode,
        "max_steps": max_steps,
        "tool_access": tool_access,
        "semantic_cache": semantic_cache_enabled,
        "continual_learning": continual_learning_enabled,
        "learning_control_plane": learning_control_plane_enabled,
        "skill_injection_role": skill_injection_role,
        "continual_learning_require_shadow": (continual_learning_require_shadow_value),
        "continual_learning_shadow_evaluation": (
            continual_learning_shadow_evaluation_value
        ),
        "memory_ids_raw": memory_ids,
        "agent_name": agent_name,
        "persona_id": persona_id,
        "persona_name": persona_name,
        "persona_role": persona_role,
        "persona_goals": persona_goals,
        "persona_background": persona_background,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_config_json": llm_config_json,
        "sandbox_provider": sandbox_provider,
        "browser_control_provider": browser_control_provider_value,
        "meta_harness_mode": meta_harness_mode_value,
        "default_harness": default_harness_value,
        "harness_workspace": harness_workspace_value,
        "internet_provider": internet_provider,
        "skills_marketplace_provider": skills_marketplace_provider_value,
        "enable_entity_memory": enable_entity_memory_value,
        "enable_workflow_memory": enable_workflow_memory_value,
        "self_aware": self_aware_enabled_value,
        "self_aware_root_paths": self_aware_root_paths,
        "self_aware_allow_writes": self_aware_allow_writes_value,
        "self_aware_allow_deletes": self_aware_allow_deletes_value,
        "automations_enabled": automations_enabled_value,
        "default_timezone": default_timezone,
    }

    if error:
        form_data = _build_agent_form_data(existing)
        form_data.update(form_override_data)
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": error,
                **form_data,
            },
        )

    instruction_value = instruction.strip() if instruction else ""
    browser_control_changed = (
        browser_control_provider_value != existing_browser_provider
    )
    if browser_control_changed:
        browser_control_validation_error = _validate_browser_control_choice(
            browser_control_provider_value, browser_control_config_value
        )
        if browser_control_validation_error:
            form_data = _build_agent_form_data(existing)
            form_data.update(form_override_data)
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": browser_control_validation_error,
                    **form_data,
                },
            )
    internet_value = _normalize_internet_provider_name(internet_provider) or None
    internet_config = _build_internet_provider_config(internet_value)
    internet_validation_error = _validate_internet_provider_choice(
        internet_value, internet_config
    )
    if internet_validation_error:
        form_data = _build_agent_form_data(existing)
        form_data.update(form_override_data)
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": internet_validation_error,
                **form_data,
            },
        )

    skills_marketplace_validation_error = _validate_skills_marketplace_provider_choice(
        skills_marketplace_value,
        skills_marketplace_config_value,
    )
    if skills_marketplace_validation_error:
        form_data = _build_agent_form_data(existing)
        form_data.update(form_override_data)
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": skills_marketplace_validation_error,
                **form_data,
            },
        )

    if self_aware_validation_error:
        form_data = _build_agent_form_data(existing)
        form_data.update(form_override_data)
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": self_aware_validation_error,
                **form_data,
            },
        )

    sandbox_value = sandbox_provider.strip() if sandbox_provider else None
    if sandbox_value is None:
        sandbox_value = getattr(existing, "sandbox_provider", None)
    # Only validate sandbox when the user explicitly changed it;
    # carry forward the existing value without blocking unrelated edits.
    sandbox_changed = (
        sandbox_value
        and sandbox_value
        != _to_text(getattr(existing, "sandbox_provider", "") or "").strip()
    )
    if sandbox_changed:
        sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
        if sandbox_validation_error:
            form_data = _build_agent_form_data(existing)
            form_data.update(form_override_data)
            return templates.TemplateResponse(
                "agent_form.html",
                {
                    "request": request,
                    "provider_type": _state["provider_type"],
                    "connection_info": _state["connection_info"],
                    "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                    "active_agent_id": agent_id,
                    "active_page": "agents",
                    "form_title": "Edit Agent",
                    "form_action": f"/agents/{agent_id}/edit",
                    "is_edit": True,
                    "error": sandbox_validation_error,
                    **form_data,
                },
            )

    # Reconcile persona with PERSONAS collection: updates become new
    # versions on the linked record (with change_trigger source_type='ui_form'),
    # new personas are stored so they appear in the saved-persona picker.
    persona_payload = _resolve_persona_for_agent(
        _state["provider"],
        persona_payload,
        persona_id,
        agent_id=agent_id,
    )

    updated = MemAgentModel(
        agent_id=agent_id,
        name=agent_name_value or getattr(existing, "name", None),
        instruction=instruction_value
        or getattr(existing, "instruction", None)
        or DEFAULT_INSTRUCTION,
        application_mode=application_mode
        or getattr(existing, "application_mode", "assistant"),
        memory_types=memory_types_value,
        max_steps=max_steps
        if max_steps is not None
        else getattr(existing, "max_steps", 20),
        tool_access=tool_access or getattr(existing, "tool_access", "private"),
        semantic_cache=semantic_cache_enabled,
        is_favorite=bool(getattr(existing, "is_favorite", False)),
        memory_ids=memory_id_list or getattr(existing, "memory_ids", None),
        persona=persona_payload,
        llm_config=llm_config or getattr(existing, "llm_config", None),
        tools=getattr(existing, "tools", None),
        delegates=getattr(existing, "delegates", None),
        embedding_config=getattr(existing, "embedding_config", None),
        semantic_cache_config=getattr(existing, "semantic_cache_config", None),
        tool_result_policy=getattr(existing, "tool_result_policy", None),
        context_policy=getattr(existing, "context_policy", None),
        retrieval_policy=getattr(existing, "retrieval_policy", None),
        delegation_config=getattr(existing, "delegation_config", None),
        skill_retrieval=bool(getattr(existing, "skill_retrieval", False)),
        skill_retrieval_config=getattr(existing, "skill_retrieval_config", None),
        semantic_layer_config=getattr(existing, "semantic_layer_config", None),
        context_window_tokens=getattr(existing, "context_window_tokens", None),
        internet_access_provider=internet_value,
        internet_access_config=internet_config,
        skills_marketplace_provider=skills_marketplace_value,
        skills_marketplace_config=skills_marketplace_config_value,
        knowledge_base_ids=getattr(existing, "knowledge_base_ids", None),
        sandbox_provider=sandbox_value,
        browser_control=browser_control_config_value,
        meta_harness=bool(meta_harness_mode_value),
        meta_harness_mode=meta_harness_mode_value or None,
        default_harness=default_harness_value,
        harness_config=harness_config_value,
        skill_paths=getattr(existing, "skill_paths", None),
        mcp_servers=getattr(existing, "mcp_servers", None),
        self_aware=self_aware_enabled_value,
        self_aware_config=self_aware_config_value,
        continual_learning=continual_learning_enabled,
        continual_learning_config=continual_learning_config_value,
        learning_control_plane=learning_control_plane_enabled,
        learning_control_plane_config=learning_control_plane_config_value,
        automations_enabled=automations_enabled_value,
        default_timezone=default_timezone_value,
        whatsapp_enabled=whatsapp_enabled_value,
        whatsapp_config=whatsapp_config_value,
    )

    try:
        _state["provider"].store_memagent(updated)
        _persist_mcp_configs_to_toolbox(
            agent_id=agent_id,
            mcp_servers=updated.mcp_servers or [],
            memory_ids=updated.memory_ids or [],
        )
    except Exception as e:
        logger.error(f"Failed to update agent {agent_id}: {e}")
        form_data = _build_agent_form_data(updated)
        return templates.TemplateResponse(
            "agent_form.html",
            {
                "request": request,
                "provider_type": _state["provider_type"],
                "connection_info": _state["connection_info"],
                "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
                "active_agent_id": agent_id,
                "active_page": "agents",
                "form_title": "Edit Agent",
                "form_action": f"/agents/{agent_id}/edit",
                "is_edit": True,
                "error": str(e),
                **form_data,
            },
        )

    return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)


@router.get("/agents/{agent_id}")
async def agent_detail(agent_id: str):
    """The standalone detail page has been retired — the playground is
    the canonical agent view. Redirect so bookmarks and legacy callers
    still land somewhere useful."""
    return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)


@router.post("/agents/{agent_id}/delete")
async def delete_agent(agent_id: str, cascade: bool = False):
    """Delete an agent."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    try:
        success = _state["provider"].delete_memagent(agent_id, cascade=cascade)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
    except Exception as e:
        logger.error(f"Failed to delete agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return RedirectResponse(url="/agents", status_code=302)
