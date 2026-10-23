# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Persistence (save/load/refresh/memory-sharing) for MemAgent.

Module-level implementations of ``MemAgent.save`` / ``MemAgent.load`` /
``MemAgent.refresh`` and the memory-id sharing helpers, moved verbatim
from ``memorizz.memagent.core``. The thin methods on ``MemAgent``
delegate here; public signatures, return values, and exceptions are
unchanged.
"""

import logging
import os
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

from ..enums import ApplicationMode, MemoryType
from ..task_decomposition import normalize_delegation_config
from .constants import DEFAULT_INSTRUCTION, DEFAULT_MAX_STEPS

logger = logging.getLogger(__name__)


def _scrub_internet_access_config(config):
    """Drop credential values from an internet provider config.

    Built-in providers already omit the key, but a third-party provider or an
    agent record saved by an older release may still carry one. The key is
    resolved from the environment on restore, never from the record.
    """
    if not isinstance(config, dict):
        return config
    from ..internet_access.base import scrub_secret_config

    return scrub_secret_config(config)


def download_memory(agent, memagent) -> bool:
    """Copy ``memagent``'s memory_ids onto ``agent`` (see MemAgent.download_memory)."""
    try:
        for mid in memagent.memory_ids:
            if mid not in agent.memory_ids:
                agent.memory_ids.append(mid)

        if hasattr(agent.memory_provider, "update_memagent_memory_ids"):
            agent.memory_provider.update_memagent_memory_ids(
                agent.agent_id, agent.memory_ids
            )

        if agent.memory_manager:
            agent.memory_manager.clear_conversation_cache()

        logger.info(
            f"Downloaded memory from agent {memagent.agent_id} to {agent.agent_id}"
        )
        return True
    except Exception as e:
        logger.error(f"Error downloading memory from agent {memagent.agent_id}: {e}")
        return False


def update_memory(agent, memory_ids: List[str]) -> bool:
    """Add memory_ids to ``agent`` (see MemAgent.update_memory)."""
    try:
        for mid in memory_ids:
            if mid not in agent.memory_ids:
                agent.memory_ids.append(mid)

        if hasattr(agent.memory_provider, "update_memagent_memory_ids"):
            agent.memory_provider.update_memagent_memory_ids(
                agent.agent_id, agent.memory_ids
            )

        if agent.memory_manager:
            agent.memory_manager.clear_conversation_cache()

        logger.info(f"Updated memory_ids for agent {agent.agent_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating memory_ids for agent {agent.agent_id}: {e}")
        return False


def save_agent(agent):
    """Store ``agent`` in its memory provider (see MemAgent.save)."""
    if not agent.memory_provider:
        raise ValueError("Cannot save MemAgent: no memory provider configured")

    try:
        # Re-normalize through the MCP manager before serializing so callers
        # that assigned configurations directly cannot persist bearer tokens,
        # OAuth client secrets, custom headers, or stdio environment values.
        if getattr(agent, "mcp_manager", None) is not None:
            agent.mcp_servers = agent.mcp_manager.configure_servers(
                getattr(agent, "mcp_servers", None) or []
            )

        # Serialize tools from tool manager
        tools_to_save = _serialize_tools_for_save(agent)

        # Convert delegates to agent IDs for persistence
        delegate_ids = _serialize_delegates_for_save(agent)

        # Get semantic cache config for saving
        semantic_cache_config_to_save = _serialize_semantic_cache_config(agent)
        llm_config_to_save = None
        # A run on a borrowed model (a harness run's model pick) keeps the
        # agent's own model in the saved record.
        pinned = getattr(agent, "_pinned_llm_config", None)
        if isinstance(pinned, dict):
            llm_config_to_save = dict(pinned)
        elif agent.model and hasattr(agent.model, "get_config"):
            candidate_llm_config = agent.model.get_config()
            if isinstance(candidate_llm_config, dict):
                llm_config_to_save = candidate_llm_config

        # A harness picked for one CLI session (/harness codex) runs on a
        # pinned copy of the agent's own harness settings, which is what
        # the saved record keeps.
        harness = getattr(agent, "_pinned_harness", None)
        if not isinstance(harness, dict):
            harness = {
                "meta_harness": bool(getattr(agent, "meta_harness", None)),
                "meta_harness_mode": getattr(agent, "meta_harness_mode", None),
                "default_harness": getattr(agent, "default_harness", "auto"),
                "harness_config": getattr(agent, "harness_config", None),
            }

        # Create MemAgentModel with current configuration
        from .models import MemAgentModel

        memagent_to_save = MemAgentModel(
            llm_config=llm_config_to_save,
            background_llm_config=_background_llm_config_to_save(agent),
            application_id=getattr(agent, "application_id", None),
            name=agent.name,
            instruction=agent.instruction,
            max_steps=agent.max_steps,
            tool_access=getattr(agent, "tool_access", "private"),
            application_mode=_get_application_mode_value(agent),
            # Types a runtime rule turned on are re-applied on load, not saved.
            # An empty list is a stateless agent and is saved as such; None
            # would reload with the application mode's default types.
            memory_types=(
                None
                if agent.active_memory_types is None
                else [
                    memory_type.value
                    for memory_type in agent.active_memory_types
                    if hasattr(memory_type, "value")
                    and memory_type not in getattr(agent, "_runtime_memory_types", ())
                ]
            ),
            memory_ids=agent.memory_ids,
            knowledge_base_ids=getattr(agent, "knowledge_base_ids", None) or None,
            agent_id=agent.agent_id,
            persona=(
                agent.persona_manager.configuration_persona
                if agent.persona_manager
                else None
            ),
            tools=tools_to_save,
            delegates=delegate_ids if delegate_ids else None,
            semantic_cache=(
                agent.cache_manager.enabled if agent.cache_manager else False
            ),
            semantic_cache_config=semantic_cache_config_to_save,
            tool_result_policy=agent.tool_result_policy.to_dict(),
            tool_cache_config=(
                agent.tool_cache_config.to_dict() if agent.tool_cache_config else None
            ),
            context_policy=agent.context_policy.to_dict(),
            completion_policy=agent.completion_policy.to_dict(),
            retrieval_policy=agent.retrieval_policy.to_dict(),
            memory_consolidation=(
                agent.memory_consolidation.to_dict()
                if agent.memory_consolidation.enabled
                else None
            ),
            memory_routing=(
                agent.memory_routing.to_dict()
                if getattr(agent, "memory_routing", None) is not None
                and agent.memory_routing.worth_saving
                else None
            ),
            reranker=(
                agent.reranker.to_dict()
                if getattr(agent, "reranker", None) is not None
                else None
            ),
            entity_policy=(
                agent.entity_policy.to_dict()
                if getattr(agent, "entity_policy", None) is not None
                and agent.entity_policy.active
                else None
            ),
            delegation_config=normalize_delegation_config(
                agent.delegation_config, for_persistence=True
            ),
            skill_retrieval=bool(agent.skill_retrieval),
            skill_retrieval_config=dict(agent.skill_retrieval_config or {}),
            semantic_layer_config=(
                agent.semantic_layer.to_dict()
                if agent.semantic_layer is not None
                and hasattr(agent.semantic_layer, "to_dict")
                else None
            ),
            context_window_tokens=getattr(agent, "_context_window_cap", None),
            context_window_source=(
                "explicit" if getattr(agent, "_context_window_cap", None) else None
            ),
            thread_titles=dict(getattr(agent, "thread_titles", None) or {}) or None,
            is_favorite=agent.is_favorite,
            internet_access_provider=agent.get_internet_access_provider_name(),
            internet_access_config=(
                _scrub_internet_access_config(
                    agent.internet_access_manager.get_provider_config()
                )
                if agent.has_internet_access()
                else None
            ),
            skills_marketplace_provider=agent.get_skills_marketplace_provider_name(),
            skills_marketplace_config=agent.get_skills_marketplace_config(),
            sandbox_provider=(
                agent.sandbox_manager.get_provider_config()
                if agent.has_sandbox()
                else None
            ),
            browser_control=(
                agent.browser_control_manager.get_provider_config()
                if agent.has_browser_control()
                else getattr(agent, "browser_control_config", None)
            ),
            meta_harness=bool(harness["meta_harness"]),
            meta_harness_mode=harness["meta_harness_mode"],
            default_harness=harness["default_harness"],
            harness_config=dict(harness["harness_config"] or {}),
            skill_paths=agent.skill_paths or None,
            mcp_servers=agent.mcp_servers or None,
            self_aware=bool(agent.self_aware),
            self_aware_config=agent.get_self_aware_config() or None,
            continual_learning=bool(agent.continual_learning),
            continual_learning_config=agent.continual_learning_config or None,
            learning_control_plane=bool(agent.learning_control_plane_enabled),
            learning_control_plane_config=(
                agent.learning_control_plane_config.to_dict()
                if agent.learning_control_plane_config is not None
                else None
            ),
            automations_enabled=bool(getattr(agent, "automations_enabled", True)),
            default_timezone=getattr(agent, "default_timezone", None),
            embedding_provider=getattr(agent, "embedding_provider", None),
            embedding_config=getattr(agent, "embedding_config", None),
            whatsapp_enabled=bool(getattr(agent, "whatsapp_enabled", False)),
            whatsapp_config=getattr(agent, "whatsapp_config", None),
        )
        memagent_to_save = _with_kept_fields(agent, memagent_to_save)

        # Save or update the agent
        if hasattr(agent.memory_provider, "store_memagent"):
            if agent.agent_id and hasattr(agent.memory_provider, "retrieve_memagent"):
                # Check if agent exists for update vs create
                try:
                    existing = agent.memory_provider.retrieve_memagent(agent.agent_id)
                    if existing and hasattr(agent.memory_provider, "update_memagent"):
                        saved_memagent = _write_memagent_compat(
                            agent.memory_provider.update_memagent,
                            memagent_to_save,
                        )
                    else:
                        saved_memagent = _write_memagent_compat(
                            agent.memory_provider.store_memagent,
                            memagent_to_save,
                        )
                except Exception:
                    # Agent doesn't exist, create new
                    saved_memagent = _write_memagent_compat(
                        agent.memory_provider.store_memagent,
                        memagent_to_save,
                    )
            else:
                # New agent
                saved_memagent = _write_memagent_compat(
                    agent.memory_provider.store_memagent,
                    memagent_to_save,
                )

            # Update agent_id if it was generated
            if not agent.agent_id and saved_memagent.get("_id"):
                agent.agent_id = str(saved_memagent["_id"])

                # Update semantic cache with new agent_id
                if agent.cache_manager and agent.cache_manager.enabled:
                    if (
                        hasattr(agent.cache_manager, "cache_instance")
                        and agent.cache_manager.cache_instance
                    ):
                        agent.cache_manager.cache_instance.agent_id = agent.agent_id
                        if agent.memory_ids:
                            agent.cache_manager.cache_instance.memory_id = (
                                agent.memory_ids[0]
                            )

            # The record now exists with these memories: registration only
            # has to patch memory_ids from here on.
            agent._agent_record_stored = True
            agent._registered_memory_ids = {str(mid) for mid in agent.memory_ids if mid}

            _persist_mcp_servers_to_toolbox_memory(agent)

            logger.info(f"MemAgent {agent.agent_id} saved successfully")
            return agent
        else:
            raise ValueError("Memory provider does not support saving MemAgent")

    except Exception as e:
        logger.error(f"Failed to save MemAgent {agent.agent_id}: {e}")
        raise


def _write_memagent_compat(writer, model):
    """Support both model-native and legacy dict-based provider contracts."""
    try:
        return writer(model)
    except (AttributeError, TypeError):
        # First-party providers accept MemAgentModel. A number of established
        # third-party providers predate that contract and call ``.get`` on a
        # plain mapping; retry once with the lossless Pydantic representation.
        if hasattr(model, "model_dump"):
            return writer(model.model_dump())
        raise


def keep_stored_fields(agent, *fields: str) -> None:
    """Make every later save of ``agent`` keep ``fields`` as stored.

    For settings changed for one run only (a scheduled run with automations
    off, a harness run on another model, an unsaved playground instruction):
    a save during the run (a new memory registered, a tool that saves the
    agent) then writes the agent's own values, read from the store at save
    time so edits made elsewhere meanwhile survive too.
    """
    from .models import MemAgentModel

    unknown = sorted(set(fields) - set(MemAgentModel.model_fields))
    if unknown:
        raise ValueError(f"Not saved agent fields: {', '.join(unknown)}")
    kept = set(getattr(agent, "_keep_stored_fields", None) or ())
    agent._keep_stored_fields = kept | set(fields)


def _record_value(record: Any, name: str) -> Any:
    """One field of a stored agent record (a model, or a legacy mapping)."""
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _with_kept_fields(agent, model):
    """``model`` with the fields ``keep_stored_fields`` named taken from the
    stored record; unchanged when the agent was never stored."""
    kept = getattr(agent, "_keep_stored_fields", None)
    provider = agent.memory_provider
    if not kept or not agent.agent_id or not hasattr(provider, "retrieve_memagent"):
        return model
    stored = provider.retrieve_memagent(agent.agent_id)
    if not stored:
        return model
    return model.model_copy(update={name: _record_value(stored, name) for name in kept})


def _get_application_mode_value(agent) -> str:
    """Return the application mode value as a string."""
    if isinstance(agent.application_mode, ApplicationMode):
        return agent.application_mode.value
    if isinstance(agent.application_mode, str):
        return agent.application_mode
    return ApplicationMode.DEFAULT.value


def _serialize_tools_for_save(agent):
    """Serialize tools for saving."""
    if not agent.tool_manager:
        return None

    # An empty list (not None) when the agent has no tools, so a save after
    # removing the last tool clears the rows that mirrored it; None means
    # "this save carries no tools" and leaves them alone.
    tools_metadata = agent.tool_manager.get_tool_metadata()
    if not tools_metadata:
        return []

    # Tools are agent-scoped (shared across threads), so stamp agent_id
    # and leave memory_id unset. The UI toolbox-memory filter treats
    # rows with agent_id but no memory_id as agent-global, which makes
    # them visible on every thread for this agent.
    serializable_tools = []
    for tool_meta in tools_metadata:
        # MCP tools are rebuilt from the server's cached listing on load; a
        # saved copy would outlive the server as a tool with nothing to call.
        if isinstance(tool_meta, dict) and tool_meta.get("source") == "mcp":
            continue
        if isinstance(tool_meta, dict):
            serializable_tool = {
                "_id": tool_meta.get("_id") or tool_meta.get("name"),
                "name": tool_meta.get("name"),
                "description": tool_meta.get("description", ""),
                "signature": tool_meta.get("signature", ""),
                "docstring": tool_meta.get(
                    "docstring", tool_meta.get("description", "")
                ),
                "parameters": tool_meta.get("parameters", {}),
                "required": tool_meta.get("required", []),
                "input_schema": tool_meta.get("input_schema")
                or {
                    "type": "object",
                    "properties": tool_meta.get("parameters", {}),
                    "required": tool_meta.get("required", []),
                    "additionalProperties": False,
                },
                "tool_policy": tool_meta.get("tool_policy", {}),
                "aliases": tool_meta.get("aliases", []),
                "deprecated_arguments": tool_meta.get("deprecated_arguments", {}),
                "queries": tool_meta.get("queries", []),
                "import_reference": tool_meta.get("import_reference"),
                "type": tool_meta.get("type", "function"),
                "agent_id": agent.agent_id,
            }
            serializable_tools.append(serializable_tool)

    return serializable_tools


def _persist_mcp_servers_to_toolbox_memory(agent) -> None:
    """Persist MCP JSON configs into toolbox memory records."""
    if not agent.memory_provider or not agent.mcp_servers:
        return

    try:
        memory_id = agent._current_memory_id or (
            agent.memory_ids[0] if agent.memory_ids else None
        )
        if not memory_id:
            memory_id = str(uuid.uuid4())
            agent.memory_ids.append(memory_id)
            agent._current_memory_id = memory_id

        for server in agent.mcp_servers:
            server_name = str(server.get("name", "")).strip()
            if not server_name:
                continue
            config_doc = {
                "_id": f"{agent.agent_id}:mcp:{server_name}",
                "tool_id": f"{agent.agent_id}:mcp:{server_name}",
                "name": f"mcp::{server_name}",
                "description": f"MCP server config for {server_name}",
                "signature": "mcp_server_config(server_json)",
                "docstring": "Stored MCP server configuration JSON.",
                "tool_type": "mcp_server_config",
                "type": "mcp_server_config",
                "parameters": server,
                "memory_id": memory_id,
                "agent_id": agent.agent_id,
            }
            agent.memory_provider.store(
                config_doc, memory_store_type=MemoryType.TOOLBOX
            )
    except Exception as exc:
        logger.warning(
            "Failed to persist MCP configs to toolbox memory for %s: %s",
            agent.agent_id,
            exc,
        )


def _serialize_delegates_for_save(agent):
    """Serialize delegate agents for saving."""
    # Note: In the new architecture, delegates would be handled differently
    # This is a placeholder for compatibility
    if hasattr(agent, "delegates") and agent.delegates:
        delegate_ids = []
        for delegate in agent.delegates:
            if hasattr(delegate, "agent_id") and delegate.agent_id:
                delegate_ids.append(delegate.agent_id)
                # Ensure delegate is saved
                try:
                    if hasattr(delegate, "save"):
                        delegate.save()
                except Exception as e:
                    logger.warning(f"Failed to save delegate {delegate.agent_id}: {e}")
        return delegate_ids
    return None


def _serialize_semantic_cache_config(agent):
    """Serialize semantic cache configuration for saving."""
    if not agent.cache_manager or not agent.cache_manager.enabled:
        return None

    try:
        if (
            hasattr(agent.cache_manager, "cache_instance")
            and agent.cache_manager.cache_instance
        ):
            if hasattr(agent.cache_manager.cache_instance, "config"):
                config = agent.cache_manager.cache_instance.config
                if hasattr(config, "__dict__"):
                    config_dict = config.__dict__.copy()
                    # Convert enums to strings for serialization
                    for key, value in config_dict.items():
                        if hasattr(value, "value"):  # Enum
                            config_dict[key] = value.value
                    return config_dict
    except Exception as e:
        logger.warning(f"Failed to serialize semantic cache config: {e}")

    return None


def _saved_context_cap(saved_memagent: Any, llm_config: Any) -> Optional[int]:
    """The agent-level context cap to restore on load.

    Older releases saved the effective budget, including a model default,
    as if it were a cap, so it outlived later changes to the model's window.
    Such unmarked values give way to a window the LLM configuration sets.
    """
    cap = getattr(saved_memagent, "context_window_tokens", None)
    if getattr(saved_memagent, "context_window_source", None) == "explicit":
        return cap
    return None if _config_context_window(llm_config) else cap


def _config_context_window(llm_config: Any) -> Optional[int]:
    """The context window an LLM configuration sets itself, if any."""
    if not isinstance(llm_config, dict):
        return None
    extra = llm_config.get("additional_config")
    for value in (
        llm_config.get("context_window_tokens"),
        llm_config.get("max_context_tokens"),
        llm_config.get("context_window"),
        extra.get("num_ctx") if isinstance(extra, dict) else None,
    ):
        if type(value) is int and value > 0:
            return value
    return None


# The agents being loaded along the current delegate chain. A saved delegate
# loop (A delegates to B, B back to A) would otherwise recurse forever.
_LOADING_CHAIN: ContextVar[FrozenSet[str]] = ContextVar(
    "memorizz_loading_delegate_chain", default=frozenset()
)


def _bool_or(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    return value if isinstance(value, dict) else None


def _list_or_empty(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


# Restored settings set on the instance after construction.
_SAVED_ATTRIBUTES = (
    "knowledge_base_ids",
    "thread_titles",
    "whatsapp_enabled",
    "whatsapp_config",
)
# Restored settings load_agent resolves itself instead of passing them on.
_RESOLVED_ON_LOAD = frozenset(
    {
        "llm_config",
        "context_window_tokens",
        "application_mode",
        "delegates",
        "internet_access_provider",
    }
)


def _saved_settings(saved_memagent: Any) -> Dict[str, Any]:
    """The settings a stored agent record restores, normalized.

    Keys are ``MemAgent`` constructor arguments, plus ``_SAVED_ATTRIBUTES``.
    ``load_agent`` builds an agent from them and ``refresh_agent`` applies
    them to a live one, so both restore the same fields the same way. Three
    are references: ``llm_config`` (for the model), delegate agent ids, and
    the internet provider's ``(name, config)``.
    """

    def field(name: str, default: Any = None) -> Any:
        return getattr(saved_memagent, name, default)

    llm_config = _dict_or_none(field("llm_config"))

    # Reconstruct semantic cache config
    semantic_cache_config = None
    if field("semantic_cache_config"):
        try:
            config_dict = dict(field("semantic_cache_config"))
            # Convert string scope back to enum if needed
            if "scope" in config_dict and isinstance(config_dict["scope"], str):
                try:
                    from ..enums.semantic_cache_scope import SemanticCacheScope

                    scope_str = config_dict["scope"].lower()
                    if scope_str == "local":
                        config_dict["scope"] = SemanticCacheScope.LOCAL
                    elif scope_str == "global":
                        config_dict["scope"] = SemanticCacheScope.GLOBAL
                except ImportError:
                    # If enum not available, keep as string
                    pass
            semantic_cache_config = config_dict
        except Exception as e:
            logger.warning(f"Failed to reconstruct semantic cache config: {e}")

    internet = None
    provider_name = field("internet_access_provider")
    provider_config = _dict_or_none(field("internet_access_config"))
    if provider_config is not None:
        # A record saved by an older release may hold the key itself; the
        # provider resolves it from the environment instead.
        provider_config = _scrub_internet_access_config(provider_config)
    if isinstance(provider_name, str) and provider_name:
        internet = (provider_name, provider_config)

    application_mode = field("application_mode")
    if not isinstance(application_mode, str) or not application_mode:
        application_mode = ApplicationMode.DEFAULT.value

    sandbox_provider = field("sandbox_provider")
    if not isinstance(sandbox_provider, (str, dict)):
        sandbox_provider = None

    browser_control = field("browser_control")
    if not isinstance(browser_control, (str, dict)):
        browser_control = None

    meta_harness_mode = field("meta_harness_mode")
    if meta_harness_mode not in {"delegate", "runtime"}:
        meta_harness_mode = None
    default_harness = field("default_harness", "auto")
    if not isinstance(default_harness, str):
        default_harness = "auto"

    skills_marketplace_provider = field("skills_marketplace_provider")
    if isinstance(skills_marketplace_provider, dict):
        skills_marketplace_provider = skills_marketplace_provider.get(
            "provider"
        ) or skills_marketplace_provider.get("name")
    if not isinstance(skills_marketplace_provider, str):
        skills_marketplace_provider = None
    else:
        skills_marketplace_provider = (
            skills_marketplace_provider.strip().lower() or None
        )

    skill_paths = field("skill_paths")
    if isinstance(skill_paths, str):
        skill_paths = [skill_paths]
    elif not isinstance(skill_paths, list):
        skill_paths = None

    mcp_servers = field("mcp_servers")
    if not isinstance(mcp_servers, list):
        mcp_servers = None

    continual_learning = _bool_or(field("continual_learning", False), False)
    learning_control_plane_config = _dict_or_none(
        field("learning_control_plane_config")
    )

    default_timezone = field("default_timezone")
    if not isinstance(default_timezone, str):
        default_timezone = None
    else:
        default_timezone = default_timezone.strip() or None

    semantic_layer = None
    semantic_layer_config = field("semantic_layer_config")
    if isinstance(semantic_layer_config, dict):
        try:
            from ..semantic_layer import SemanticCatalog

            semantic_layer = SemanticCatalog.from_dict(semantic_layer_config)
        except Exception as exc:
            logger.warning("Failed to restore semantic catalog: %s", exc)

    tool_access = field("tool_access", "private")
    if not isinstance(tool_access, str) or tool_access not in {
        "private",
        "public",
        "global",
    }:
        tool_access = "private"

    memory_ids = field("memory_ids", [])
    if isinstance(memory_ids, str):
        memory_ids = [memory_ids]

    titles = field("thread_titles")
    embedding_provider = field("embedding_provider")

    # Ordered as refresh applies them: the model before the context window
    # it bounds, the application mode before the memory types it implies.
    return {
        "llm_config": llm_config,
        "context_window_tokens": _saved_context_cap(saved_memagent, llm_config),
        "background_llm_config": _dict_or_none(field("background_llm_config")),
        "memory_consolidation": _dict_or_none(field("memory_consolidation")),
        "memory_routing": _dict_or_none(field("memory_routing")),
        "reranker": _dict_or_none(field("reranker")),
        "entity_policy": _dict_or_none(field("entity_policy")),
        "tools": field("tools"),
        "persona": field("persona"),
        "name": field("name"),
        "application_id": field("application_id"),
        "instruction": field("instruction"),
        "max_steps": field("max_steps", DEFAULT_MAX_STEPS),
        "tool_access": tool_access,
        "memory_ids": _list_or_empty(memory_ids),
        "application_mode": application_mode,
        "memory_types": field("memory_types"),
        "delegates": _list_or_empty(field("delegates")),
        "semantic_cache": _bool_or(field("semantic_cache", False), False),
        "semantic_cache_config": semantic_cache_config,
        "tool_result_policy": _dict_or_none(field("tool_result_policy")),
        "tool_cache": _dict_or_none(field("tool_cache_config")),
        "context_policy": _dict_or_none(field("context_policy")),
        "completion_policy": _dict_or_none(field("completion_policy")),
        "retrieval_policy": _dict_or_none(field("retrieval_policy")),
        "skill_retrieval": _bool_or(field("skill_retrieval", False), False),
        "skill_retrieval_config": _dict_or_none(field("skill_retrieval_config")),
        "delegation": _dict_or_none(field("delegation_config")),
        "semantic_layer": semantic_layer,
        "internet_access_provider": internet,
        "skills_marketplace_provider": skills_marketplace_provider,
        "skills_marketplace_config": _dict_or_none(field("skills_marketplace_config")),
        "sandbox_provider": sandbox_provider,
        "browser_control": browser_control,
        "meta_harness": _bool_or(field("meta_harness", False), False),
        "meta_harness_mode": meta_harness_mode,
        "default_harness": default_harness,
        "harness_config": _dict_or_none(field("harness_config")),
        "skill_paths": skill_paths,
        "mcp_servers": mcp_servers,
        "automations_enabled": _bool_or(field("automations_enabled", True), True),
        "default_timezone": default_timezone,
        "self_aware": _bool_or(field("self_aware", False), False),
        "self_aware_config": _dict_or_none(field("self_aware_config")),
        "continual_learning": continual_learning,
        "continual_learning_config": _dict_or_none(field("continual_learning_config")),
        "learning_control_plane": (
            learning_control_plane_config
            if learning_control_plane_config is not None
            else _bool_or(field("learning_control_plane", False), False)
        ),
        "is_favorite": _bool_or(field("is_favorite", False), False),
        "embedding_provider": (
            embedding_provider if isinstance(embedding_provider, str) else None
        ),
        "embedding_config": _dict_or_none(field("embedding_config")),
        "knowledge_base_ids": _list_or_empty(field("knowledge_base_ids")),
        "thread_titles": dict(titles) if isinstance(titles, dict) else {},
        "whatsapp_enabled": _bool_or(field("whatsapp_enabled", False), False),
        "whatsapp_config": _dict_or_none(field("whatsapp_config")),
    }


def _internet_provider(saved: Optional[Tuple[str, Optional[Dict[str, Any]]]]):
    """The internet provider a saved ``(name, config)`` describes, or None."""
    if not saved:
        return None
    provider_name, provider_config = saved
    try:
        from ..internet_access import create_internet_access_provider

        return create_internet_access_provider(provider_name, provider_config)
    except Exception as exc:
        logger.warning(
            "Failed to restore internet provider '%s': %s", provider_name, exc
        )
        return None


def _load_delegates(
    cls, agent_id: str, delegate_ids, memory_provider, runtime_memory_provider
) -> List[Any]:
    """Load an agent's delegates, skipping any already in this delegate chain."""
    loaded = []
    chain = _LOADING_CHAIN.get() | {str(agent_id)}
    token = _LOADING_CHAIN.set(chain)
    try:
        for delegate_id in delegate_ids:
            if str(delegate_id) in chain:
                logger.warning(
                    "Skipped delegate %s of agent %s: it would load an agent "
                    "already in this delegate chain",
                    delegate_id,
                    agent_id,
                )
                continue
            try:
                loaded.append(
                    cls.load(
                        delegate_id,
                        memory_provider,
                        runtime_memory_provider=runtime_memory_provider,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not load delegate agent {delegate_id}: {e}")
    finally:
        _LOADING_CHAIN.reset(token)
    return loaded


def load_agent(cls, agent_id: str, memory_provider=None, **overrides):
    """Load a MemAgent from the memory provider (see MemAgent.load)."""
    # ``create_llm_provider`` is resolved through the core module at call
    # time so existing patches of ``memorizz.memagent.core.create_llm_provider``
    # keep working after this pure move. Imported lazily to avoid a circular
    # import at module load.
    from . import core as _core

    if memory_provider is None:
        from ..memory_provider import create_default_memory_provider

        memory_provider = create_default_memory_provider()
    elif memory_provider is False:
        raise ValueError("Cannot load MemAgent with memory_provider=False")

    if not hasattr(memory_provider, "retrieve_memagent"):
        raise ValueError("Memory provider does not support loading MemAgent")

    # Configuration and runtime memory are normally the same provider. An
    # isolated evaluation can load a persisted agent template while directing
    # all benchmark reads/writes to a disposable provider instead.
    runtime_memory_provider = overrides.pop("runtime_memory_provider", memory_provider)
    if runtime_memory_provider is False or runtime_memory_provider is None:
        raise ValueError("runtime_memory_provider must be a memory provider")

    logger.info(f"Loading MemAgent with agent id {agent_id}...")

    # Retrieve the saved agent
    saved_memagent = memory_provider.retrieve_memagent(agent_id)
    if not saved_memagent:
        raise ValueError(
            f"MemAgent with agent id {agent_id} not found in the memory provider"
        )
    settings = _saved_settings(saved_memagent)

    # Reconstruct LLM model. Stash any construction error so we can
    # attach it to the new instance below — otherwise chat would
    # surface a generic "No LLM model configured" instead of the
    # actual cause (e.g. uncached HF repo, network failure).
    resolved_llm_config = overrides.get("llm_config", settings["llm_config"])
    override_model = overrides.get("model") if "model" in overrides else None
    model_to_load = override_model
    load_llm_error: Optional[str] = None
    if override_model is not None:
        if "llm_config" not in overrides and hasattr(override_model, "get_config"):
            try:
                candidate_config = override_model.get_config()
                if isinstance(candidate_config, dict):
                    resolved_llm_config = candidate_config
            except Exception:
                pass
    elif resolved_llm_config:
        try:
            model_to_load = _core.create_llm_provider(resolved_llm_config)
        except Exception as e:
            load_llm_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "Could not load model from config: %s. Model will be None.", e
            )
    elif "llm_config" not in overrides and getattr(saved_memagent, "model", None):
        model_to_load = saved_memagent.model

    loaded_delegates = None
    if settings["delegates"] and "delegates" not in overrides:
        loaded_delegates = _load_delegates(
            cls,
            agent_id,
            settings["delegates"],
            memory_provider,
            runtime_memory_provider,
        )

    restored = {
        key: overrides.get(key, value)
        for key, value in settings.items()
        if key not in _RESOLVED_ON_LOAD and key not in _SAVED_ATTRIBUTES
    }
    # Create new agent instance with loaded configuration
    agent_instance = cls(
        model=model_to_load,
        # Preserve the secret-free provider metadata as well as the hydrated
        # runtime model.  Passing only ``model`` made a successfully reloaded
        # agent report an empty ``llm_config`` (and therefore no provider or
        # model) to the SDK, CLI, MCP server, and UI.
        llm_config=resolved_llm_config,
        # An agent-level cap survives a normal reload. A replacement model or
        # provider configuration gets its own budget unless explicitly capped.
        context_window_tokens=overrides.get(
            "context_window_tokens",
            None
            if "llm_config" in overrides or override_model is not None
            else settings["context_window_tokens"],
        ),
        agent_id=agent_id,
        memory_provider=runtime_memory_provider,
        application_mode=(
            overrides.get("application_mode") or settings["application_mode"]
        ),
        delegates=overrides.get("delegates", loaded_delegates),
        internet_access_provider=(
            overrides["internet_access_provider"]
            if "internet_access_provider" in overrides
            else _internet_provider(settings["internet_access_provider"])
        ),
        workflow_outcome_evaluator=overrides.get("workflow_outcome_evaluator"),
        streaming=overrides.get("streaming", True),
        auto_register=overrides.get("auto_register", True),
        capture_context_snapshots=overrides.get("capture_context_snapshots", False),
        capture_memory_history=overrides.get("capture_memory_history", False),
        **restored,
    )

    # Not constructor arguments. knowledge_base_ids must survive reloads so
    # the `knowledge_base_lookup` tool can scope retrievals to this agent's
    # ingested documents.
    for name in _SAVED_ATTRIBUTES:
        setattr(agent_instance, name, settings[name])
    if "knowledge_base_ids" in overrides:
        agent_instance.knowledge_base_ids = _list_or_empty(
            overrides["knowledge_base_ids"]
        )

    # The record is already in this provider, so registering a new memory
    # patches its memory_ids rather than rewriting it with this instance's
    # settings (which may include run-only overrides).
    if runtime_memory_provider is memory_provider:
        agent_instance._agent_record_stored = True
        agent_instance._registered_memory_ids = {
            str(mid) for mid in settings["memory_ids"] if mid
        }

    # Carry the LLM-init failure forward so chat surfaces the real cause.
    # The constructor receives both the resolved model and its persisted
    # config, so it retains introspection metadata without constructing the
    # provider a second time.
    if load_llm_error and not getattr(agent_instance, "_llm_init_error", None):
        agent_instance._llm_init_error = load_llm_error

    logger.info(f"MemAgent loaded successfully with agent_id: {agent_id}")
    return agent_instance


# --- refresh: apply restored settings to a live agent -----------------------


def _set(attribute: str, normalize: Callable[[Any], Any] = lambda value: value):
    """A refresh step that sets one attribute."""

    def apply(agent, value) -> None:
        setattr(agent, attribute, normalize(value))

    return apply


def _code_owned(agent, tools) -> None:
    """Tools are code. Built-ins register from it on construction, and saved
    metadata never replaces a live tool (ToolManager._register_tool_from_data)."""


def _refresh_model(agent, llm_config, context_cap) -> None:
    """A changed LLM config gets a new model; the context cap follows."""
    from . import core as _core

    config = dict(llm_config or {})
    if isinstance(getattr(agent, "_pinned_llm_config", None), dict):
        # This run borrows another model and keeps it; saves keep the
        # agent's own, now the stored one.
        agent._pinned_llm_config = config
        return
    try:
        live = agent.model.get_config() if agent.model is not None else None
    except Exception:
        live = None
    if config and config != (agent.llm_config or {}) and config != live:
        agent.llm_config = config
        try:
            agent.model = _core.create_llm_provider(config)
            agent._llm_init_error = None
        except Exception as exc:
            agent.model = None
            agent._llm_init_error = f"{type(exc).__name__}: {exc}"
    agent._context_window_cap = (
        context_cap if type(context_cap) is int and context_cap > 0 else None
    )
    agent._context_window_tokens = agent._initialize_context_window_tokens(
        agent._context_window_cap, agent.llm_config
    )


def _background_llm_config_to_save(agent) -> Optional[Dict[str, Any]]:
    """The background model's settings without credentials, or None."""
    model = getattr(agent, "_background_model", None)
    get_config = getattr(model, "get_config", None)
    config = get_config() if callable(get_config) else None
    if not isinstance(config, dict):
        config = getattr(agent, "background_llm_config", None)
    if not config:
        return None
    from ..internet_access.base import scrub_secret_config

    safe = scrub_secret_config(config)
    safe.pop("api_key_set", None)
    return safe or None


def _refresh_background(agent, background_llm_config, consolidation) -> None:
    from ..learning.consolidation import ConsolidationConfig

    stored = dict(background_llm_config or {})
    if stored != (_background_llm_config_to_save(agent) or {}):
        agent.with_background_model(stored or None)
    agent.memory_consolidation = ConsolidationConfig.from_value(consolidation)


def _refresh_memory_models(agent, memory_routing, reranker, entity_policy) -> None:
    from ..reranking import RerankerConfig
    from .entity_policy import EntityPolicy
    from .memory_routing import MemoryRoutingConfig

    agent.memory_routing = MemoryRoutingConfig.from_value(memory_routing)
    agent.reranker = RerankerConfig.from_value(reranker)
    agent.entity_policy = EntityPolicy.from_value(entity_policy)
    agent._register_memory_routing_tools()


def _refresh_persona(agent, persona) -> None:
    # set_persona registers the persona evolution tools (update_persona,
    # read_persona) when a persona is added here, not just via __init__.
    if agent.persona_manager:
        agent.set_persona(persona, save=False)


def _refresh_application_id(agent, application_id) -> None:
    configured = application_id or os.getenv("MEMORIZZ_APPLICATION_ID", "")
    agent.application_id = (
        configured.strip()
        if isinstance(configured, str) and configured.strip()
        else None
    )


def _refresh_memory_types(agent, application_mode, memory_types) -> None:
    (
        agent.application_mode,
        agent._application_mode_explicit,
    ) = agent._resolve_application_mode(application_mode)
    active = agent._initialize_memory_types(
        agent.application_mode if agent._application_mode_explicit else None,
        memory_types,
    )
    # Types a runtime rule turned on (continual learning, MCP) stay on.
    for memory_type in agent._runtime_memory_types:
        if memory_type not in active:
            active.append(memory_type)
    agent.active_memory_types = active
    wanted = MemoryType.ENTITY_MEMORY in active
    if wanted != agent._entity_memory_enabled:
        agent.with_entity_memory(wanted)


def _refresh_delegates(agent, delegate_ids) -> None:
    current = [
        str(delegate.agent_id)
        for delegate in agent.delegates or []
        if getattr(delegate, "agent_id", None)
    ]
    if [str(delegate_id) for delegate_id in delegate_ids] != current:
        agent.delegates = _load_delegates(
            type(agent),
            agent.agent_id,
            delegate_ids,
            agent.memory_provider,
            agent.memory_provider,
        )


def _refresh_semantic_cache(agent, enabled, config) -> None:
    manager = agent.cache_manager
    if not enabled:
        if manager is not None:
            manager.enabled = False
        return
    stored = {
        key: getattr(value, "value", value) for key, value in (config or {}).items()
    }
    if (
        manager is not None
        and manager.enabled
        and stored == (_serialize_semantic_cache_config(agent) or {})
    ):
        return
    from .managers import CacheManager

    agent.cache_manager = CacheManager(
        enabled=True,
        config=config,
        agent_id=agent.agent_id,
        memory_id=agent.memory_ids[0] if agent.memory_ids else None,
        memory_provider=agent.memory_provider,
    )


def _refresh_policies(agent, tool_result, context, completion, retrieval) -> None:
    from ..completion import CompletionPolicy
    from ..retrieval import RetrievalPolicy
    from ..tooling import ContextPolicy, ToolResultPolicy

    agent.tool_result_policy = ToolResultPolicy.from_value(tool_result)
    agent.completion_policy = CompletionPolicy.from_value(completion)
    agent.retrieval_policy = RetrievalPolicy.from_value(retrieval)
    agent.context_policy = policy = ContextPolicy.from_value(context)
    router = getattr(agent, "semantic_tool_router", None)
    if router is not None:
        # The tool router took these from the policy when it was built.
        router.top_k = max(1, min(int(policy.tool_top_k), 50))
        router.enabled = bool(policy.progressive_tool_disclosure)
        router.max_invocations_per_turn = max(
            1, int(policy.max_tool_invocations_per_turn)
        )
        router.max_attempts_per_call = max(1, int(policy.max_tool_attempts_per_call))
        router.sticky_limit = max(0, int(policy.sticky_tool_limit))
        router.stable_under_tokens = max(0, int(policy.stable_tool_list_tokens))


def _refresh_tool_cache(agent, value) -> None:
    from ..tool_cache import ToolCache, ToolCacheConfig

    config = ToolCacheConfig.from_value(value)
    current = agent.tool_cache_config
    if (config.to_dict() if config else None) != (
        current.to_dict() if current else None
    ):
        agent.tool_cache_config = config
        agent.tool_cache = (
            ToolCache(config, agent_id=agent.agent_id) if config else None
        )


def _refresh_skill_retrieval(agent, enabled, config) -> None:
    # Continual learning retrieves skills whatever the switch says.
    agent.skill_retrieval = bool(enabled) or bool(agent.continual_learning)
    agent.skill_retrieval_config = dict(config or {})
    if agent.skill_retrieval and agent.skillbox is None and agent.memory_provider:
        from ..long_term.procedural.skillbox import Skillbox

        agent.skillbox = Skillbox(
            memory_provider=agent.memory_provider,
            llm_provider=agent.background_model,
            agent_id=agent.agent_id,
        )
        agent._register_skill_tools()


def _refresh_semantic_layer(agent, catalog) -> None:
    def as_dict(value):
        return value.to_dict() if hasattr(value, "to_dict") else None

    if as_dict(catalog) != as_dict(agent.semantic_layer):
        # Its tools hold the catalog they were built with.
        _keep_until_reload(agent, "semantic_layer_config")


def _refresh_internet_access(agent, saved) -> None:
    live = None
    live_name = agent.get_internet_access_provider_name()
    if live_name:
        live_config = agent.internet_access_manager.get_provider_config()
        live = (live_name, _scrub_internet_access_config(live_config) or {})
    stored = (saved[0], saved[1] or {}) if saved else None
    if stored == live:
        return
    if stored is None and agent.application_mode == ApplicationMode.DEEP_RESEARCH:
        return  # deep research attaches its default provider itself
    provider = _internet_provider(saved)
    if saved and provider is None:
        return  # not restorable here (logged); the live one stays
    agent.with_internet_access_provider(provider)


def _refresh_skills_marketplace(agent, provider_name, config) -> None:
    if not provider_name:
        # As on construction: no saved provider means the process default.
        provider_name = (
            os.getenv("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
            .strip()
            .lower()
            or None
        )
    live = (
        agent.get_skills_marketplace_provider_name(),
        agent.get_skills_marketplace_config() or {},
    )
    if (provider_name, dict(config or {})) != live:
        agent.with_skills_marketplace_provider(provider_name, config=config)


def _refresh_sandbox(agent, saved) -> None:
    live_name = agent.get_sandbox_provider_name() if agent.has_sandbox() else None
    if saved is None:
        if live_name:
            agent.with_sandbox_provider(None)
        return
    if isinstance(saved, str) and saved.strip().lower() == (live_name or ""):
        return
    if isinstance(saved, dict) and live_name:
        if saved == agent.sandbox_manager.get_provider_config():
            return
    agent.with_sandbox_provider(saved)


def _refresh_browser_control(agent, saved) -> None:
    wanted = {"provider": saved} if isinstance(saved, str) else saved
    if wanted != agent.browser_control_config:
        agent.with_browser_control(saved)


def _refresh_meta_harness(agent, enabled, mode, default_harness, config) -> None:
    stored = {
        "meta_harness": bool(enabled),
        "meta_harness_mode": mode,
        "default_harness": default_harness,
        "harness_config": dict(config or {}),
    }
    if isinstance(getattr(agent, "_pinned_harness", None), dict):
        # This session runs on a harness of its own choosing; saves keep the
        # agent's own settings, now the stored ones.
        agent._pinned_harness = stored
        return
    live = {
        "meta_harness": agent.meta_harness is not None,
        "meta_harness_mode": agent.meta_harness_mode,
        "default_harness": agent.default_harness,
        "harness_config": dict(agent.harness_config or {}),
    }
    if stored != live:
        # A live agent has no way to switch its meta-harness off.
        _keep_until_reload(agent, *stored)


def _refresh_skill_paths(agent, skill_paths) -> None:
    paths = agent._normalize_skill_paths(skill_paths)
    if paths == agent.skill_paths:
        return
    agent.skill_paths = paths
    agent.skills = agent._load_skills()
    if agent.skills or agent.skillbox:
        agent._register_skill_tools()


def _refresh_mcp_servers(agent, servers) -> None:
    if list(servers or []) != list(agent.mcp_servers or []):
        agent.with_mcp_servers(servers or [])


def _refresh_automations(agent, enabled, default_timezone) -> None:
    if (
        enabled == bool(agent.automations_enabled)
        and default_timezone == agent.default_timezone
    ):
        return
    from .managers.automation_manager import AUTOMATION_TOOL_NAMES, AutomationManager

    agent.automations_enabled = enabled
    agent.default_timezone = default_timezone
    for tool_name in AUTOMATION_TOOL_NAMES:
        if tool_name in agent.tool_manager.tools:
            agent.tool_manager.remove_tool(tool_name)
    agent.automation_manager = AutomationManager(agent.memory_provider, enabled=enabled)
    if agent.automation_manager.is_enabled():
        # The tools resolve times in the default timezone they are built with.
        agent.automation_manager.register_tools(
            agent.tool_manager,
            agent_id=agent.agent_id,
            default_timezone=default_timezone,
        )


def _refresh_self_aware(agent, enabled, config) -> None:
    agent.with_self_aware(enabled, config=config)


def _refresh_learning(agent, enabled, config, control_plane) -> None:
    from ..learning import LearningControlPlaneConfig

    stored_plane = LearningControlPlaneConfig.from_value(control_plane, implied=enabled)
    live_plane = agent.learning_control_plane_config
    if (
        enabled != bool(agent.continual_learning)
        or (config or None) != (agent.continual_learning_config or None)
        or stored_plane.to_dict() != (live_plane.to_dict() if live_plane else None)
    ):
        # The learning loop owns workflow capture and the skillbox.
        _keep_until_reload(
            agent,
            "continual_learning",
            "continual_learning_config",
            "learning_control_plane",
            "learning_control_plane_config",
        )


def _keep_until_reload(agent, *fields: str) -> None:
    """A stored change a live agent cannot apply: saves keep it as stored,
    and it takes effect when the agent is next loaded."""
    logger.info(
        "MemAgent %s: %s changed in storage; applies on the next load",
        agent.agent_id,
        ", ".join(fields),
    )
    keep_stored_fields(agent, *fields)


# Every restored setting and how refresh applies it. A setting
# _saved_settings gains needs a step here; a unit test checks the two match.
_REFRESH_STEPS: Tuple[Tuple[Tuple[str, ...], Callable[..., None]], ...] = (
    (("llm_config", "context_window_tokens"), _refresh_model),
    (("background_llm_config", "memory_consolidation"), _refresh_background),
    (("memory_routing", "reranker", "entity_policy"), _refresh_memory_models),
    (("tools",), _code_owned),
    (("persona",), _refresh_persona),
    (
        ("name",),
        _set(
            "name",
            lambda value: value.strip()
            if isinstance(value, str) and value.strip()
            else None,
        ),
    ),
    (("application_id",), _refresh_application_id),
    (("instruction",), _set("instruction", lambda v: v or DEFAULT_INSTRUCTION)),
    (("max_steps",), _set("max_steps")),
    (("tool_access",), _set("tool_access")),
    (("memory_ids",), _set("memory_ids", list)),
    (("application_mode", "memory_types"), _refresh_memory_types),
    (("delegates",), _refresh_delegates),
    (("semantic_cache", "semantic_cache_config"), _refresh_semantic_cache),
    (
        (
            "tool_result_policy",
            "context_policy",
            "completion_policy",
            "retrieval_policy",
        ),
        _refresh_policies,
    ),
    (("tool_cache",), _refresh_tool_cache),
    (("skill_retrieval", "skill_retrieval_config"), _refresh_skill_retrieval),
    (("delegation",), _set("delegation_config", normalize_delegation_config)),
    (("semantic_layer",), _refresh_semantic_layer),
    (("internet_access_provider",), _refresh_internet_access),
    (
        ("skills_marketplace_provider", "skills_marketplace_config"),
        _refresh_skills_marketplace,
    ),
    (("sandbox_provider",), _refresh_sandbox),
    (("browser_control",), _refresh_browser_control),
    (
        ("meta_harness", "meta_harness_mode", "default_harness", "harness_config"),
        _refresh_meta_harness,
    ),
    (("skill_paths",), _refresh_skill_paths),
    (("mcp_servers",), _refresh_mcp_servers),
    (("automations_enabled", "default_timezone"), _refresh_automations),
    (("self_aware", "self_aware_config"), _refresh_self_aware),
    (
        ("continual_learning", "continual_learning_config", "learning_control_plane"),
        _refresh_learning,
    ),
    (("is_favorite",), _set("is_favorite")),
    (("embedding_provider",), _set("embedding_provider")),
    (("embedding_config",), _set("embedding_config")),
    (("knowledge_base_ids",), _set("knowledge_base_ids")),
    (("thread_titles",), _set("thread_titles")),
    (("whatsapp_enabled",), _set("whatsapp_enabled")),
    (("whatsapp_config",), _set("whatsapp_config")),
)


def _apply_saved_settings(agent, settings: Dict[str, Any]) -> None:
    """Apply restored settings to a live agent, one step at a time."""
    kept = set(getattr(agent, "_keep_stored_fields", None) or ())
    for keys, apply in _REFRESH_STEPS:
        if kept.intersection(keys):
            continue  # overridden for this run; saves read the store anyway
        try:
            apply(agent, *(settings[key] for key in keys))
        except Exception as exc:
            logger.warning(
                "Could not refresh %s of MemAgent %s: %s",
                ", ".join(keys),
                agent.agent_id,
                exc,
            )


def refresh_agent(agent):
    """Refresh ``agent`` from its memory provider (see MemAgent.refresh).

    Every saved setting is applied the way ``load_agent`` restores it, so a
    save afterwards does not undo edits made elsewhere (the UI, another
    process). A few a live agent cannot switch (its meta-harness, the
    learning loop, a semantic catalog) are kept as stored by later saves
    and take effect on the next load.
    """
    if not agent.memory_provider:
        logger.error("Cannot refresh MemAgent: no memory provider configured")
        return False

    if not agent.agent_id:
        logger.error("Cannot refresh MemAgent: no agent_id set")
        return False

    try:
        # Load fresh configuration from memory provider
        if hasattr(agent.memory_provider, "retrieve_memagent"):
            saved_memagent = agent.memory_provider.retrieve_memagent(agent.agent_id)
            if saved_memagent:
                _apply_saved_settings(agent, _saved_settings(saved_memagent))
                agent._agent_record_stored = True
                agent._registered_memory_ids = {
                    str(mid) for mid in agent.memory_ids if mid
                }
                logger.info(f"MemAgent {agent.agent_id} refreshed successfully")
                return agent
            else:
                logger.error(f"MemAgent {agent.agent_id} not found in memory provider")
                return False
        else:
            logger.error("Memory provider does not support retrieving MemAgent")
            return False

    except Exception as e:
        logger.error(f"Error refreshing MemAgent {agent.agent_id}: {e}")
        return False
