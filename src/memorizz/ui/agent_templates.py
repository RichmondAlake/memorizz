# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Agent templates on the create page: the cards, and a form saved as one.

A template fills in agent-form fields by name. Saving one keeps what describes
the kind of agent (instructions, persona, model, memory and runtime settings)
and leaves out what belongs to one agent or one machine: memory IDs,
delegates, file paths, WhatsApp, and any key or token in the LLM config.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional

from ..memagent.templates import (
    TEMPLATES,
    AgentTemplate,
    get_template,
    load_user_templates,
)

TEXT_FIELDS = (
    "agent_name",
    "instruction",
    "application_mode",
    "max_steps",
    "tool_access",
    "skill_injection_role",
    "entity_scope",
    "consolidation_every_n_turns",
    "forgetting_scoring_recency_decay_per_hour",
    "forgetting_scoring_recency_anchor",
    "forgetting_scoring_alpha_recency",
    "forgetting_scoring_alpha_importance",
    "forgetting_scoring_alpha_relevance",
    "forgetting_retention_enabled",
    "forgetting_retention_min_retention",
    "forgetting_retention_grace_days",
    "llm_provider",
    "llm_model",
    "llm_config_json",
    "background_model",
    "persona_name",
    "persona_role",
    "persona_goals",
    "persona_background",
    "meta_harness_mode",
    "default_harness",
    "harness_model",
    "sandbox_provider",
    "browser_control_provider",
    "internet_provider",
    "skills_marketplace_provider",
    "default_timezone",
)

CHECKBOX_FIELDS = (
    "semantic_cache",
    "tool_cache",
    "enable_entity_memory",
    "enable_workflow_memory",
    "continual_learning",
    "learning_control_plane",
    "sealed_conversations",
    "memory_consolidation",
    "consolidation_auto_apply",
    "continual_learning_require_shadow",
    "continual_learning_shadow_evaluation",
    "automations_enabled",
)

_SECRET_KEY = re.compile(
    r"api[_-]?key|secret|token|password|passphrase|authorization|credential", re.I
)

_INTERNET_TITLES = {"tavily": "Tavily", "firecrawl": "Firecrawl"}


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_secrets(item)
            for key, item in value.items()
            if not _SECRET_KEY.search(str(key))
        }
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return value


def _llm_config_without_secrets(raw: str) -> str:
    """The advanced LLM config with keys and tokens removed. Text that isn't
    a JSON object is dropped: it can't be checked for secrets."""
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    cleaned = _without_secrets(parsed)
    return json.dumps(cleaned) if cleaned else ""


def values_from_form(form: Mapping[str, Any]) -> Dict[str, Any]:
    """The template-worthy fields of a posted agent form."""
    values: Dict[str, Any] = {}
    for name in TEXT_FIELDS:
        raw = form.get(name)
        if raw is None:
            continue
        text = str(raw).strip()
        if name == "llm_config_json":
            text = _llm_config_without_secrets(text)
        values[name] = text
    for name in CHECKBOX_FIELDS:
        values[name] = str(form.get(name) or "").strip().lower() in {
            "1",
            "on",
            "true",
            "yes",
        }
    return values


_INTERNET_KEYS = {"tavily": "TAVILY_API_KEY", "firecrawl": "FIRECRAWL_API_KEY"}


def _available_internet_provider(preferences: List[str]) -> Optional[str]:
    """The first preferred search provider with a key in the environment,
    the same place the provider reads it from when the agent runs."""
    default = os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER", "").strip().lower()
    for name in preferences:
        if os.environ.get(_INTERNET_KEYS.get(name, ""), "").strip():
            return name
        if (
            name == default
            and os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY", "").strip()
        ):
            return name
    return None


def _card(template: AgentTemplate) -> Dict[str, Any]:
    from ..mcp import catalog

    values = template.form_values()
    chips: List[str] = []
    connections = []
    needs_google_client = False
    for key in template.mcp_presets:
        preset = catalog.PRESETS[catalog.preset_key(key)]
        connections.append(preset["title"])
        needs_google_client = needs_google_client or bool(preset["needs_client"])
    chips.extend(connections)
    internet = values.get("internet_provider")
    if isinstance(internet, (list, tuple)):
        chosen = _available_internet_provider([str(name) for name in internet])
        values["internet_provider"] = chosen or ""
        chips.append(
            f"Web search ({_INTERNET_TITLES.get(chosen, chosen)})"
            if chosen
            else "Web search needs a Tavily or Firecrawl key in Settings"
        )
    elif internet:
        chips.append(f"Web search ({_INTERNET_TITLES.get(internet, internet)})")
    if values.get("sealed_conversations"):
        chips.append("Sealed conversations")
    if values.get("memory_consolidation"):
        chips.append("Memory consolidation")
    if not template.builtin and values.get("llm_model"):
        chips.append(
            " · ".join(
                part
                for part in (values.get("llm_provider"), values.get("llm_model"))
                if part
            )
        )
    return {
        "key": template.key,
        "title": template.title,
        "summary": template.summary,
        "category": template.category,
        "builtin": template.builtin,
        "chips": chips,
        "connections": connections,
        "needs_google_client": needs_google_client,
        "persona_name": template.persona.get("name", ""),
        "values": values,
    }


def template_cards() -> List[Dict[str, Any]]:
    """Built-in templates, then the ones the user saved."""
    return [_card(template) for template in TEMPLATES.values()] + [
        _card(template) for template in load_user_templates()
    ]


def find_template(key: Any) -> Optional[AgentTemplate]:
    try:
        return get_template(str(key or ""))
    except KeyError:
        return None


__all__ = [
    "CHECKBOX_FIELDS",
    "TEXT_FIELDS",
    "find_template",
    "template_cards",
    "values_from_form",
]
