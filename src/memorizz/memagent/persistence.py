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
import uuid
from typing import List, Optional

from ..enums import ApplicationMode, MemoryType
from .constants import DEFAULT_MAX_STEPS

logger = logging.getLogger(__name__)


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

        # Create MemAgentModel with current configuration
        from .models import MemAgentModel

        memagent_to_save = MemAgentModel(
            llm_config=(
                agent.model.get_config()
                if agent.model and hasattr(agent.model, "get_config")
                else None
            ),
            name=agent.name,
            instruction=agent.instruction,
            max_steps=agent.max_steps,
            application_mode=_get_application_mode_value(agent),
            memory_types=[
                memory_type.value
                for memory_type in (agent.active_memory_types or [])
                if hasattr(memory_type, "value")
            ]
            or None,
            memory_ids=agent.memory_ids,
            knowledge_base_ids=getattr(agent, "knowledge_base_ids", None) or None,
            agent_id=agent.agent_id,
            persona=(
                agent.persona_manager.current_persona if agent.persona_manager else None
            ),
            tools=tools_to_save,
            delegates=delegate_ids if delegate_ids else None,
            semantic_cache=(
                agent.cache_manager.enabled if agent.cache_manager else False
            ),
            semantic_cache_config=semantic_cache_config_to_save,
            tool_result_policy=agent.tool_result_policy.to_dict(),
            context_policy=agent.context_policy.to_dict(),
            delegation_config=dict(agent.delegation_config or {}),
            skill_retrieval=bool(agent.skill_retrieval),
            skill_retrieval_config=dict(agent.skill_retrieval_config or {}),
            semantic_layer_config=(
                agent.semantic_layer.to_dict()
                if agent.semantic_layer is not None
                and hasattr(agent.semantic_layer, "to_dict")
                else None
            ),
            context_window_tokens=agent._context_window_tokens,
            is_favorite=agent.is_favorite,
            internet_access_provider=agent.get_internet_access_provider_name(),
            internet_access_config=(
                agent.internet_access_manager.get_provider_config()
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
            skill_paths=agent.skill_paths or None,
            mcp_servers=agent.mcp_servers or None,
            self_aware=bool(agent.self_aware),
            self_aware_config=agent.get_self_aware_config() or None,
            continual_learning=bool(agent.continual_learning),
            continual_learning_config=agent.continual_learning_config or None,
            automations_enabled=bool(getattr(agent, "automations_enabled", True)),
            default_timezone=getattr(agent, "default_timezone", None),
        )

        # Save or update the agent
        if hasattr(agent.memory_provider, "store_memagent"):
            if agent.agent_id and hasattr(agent.memory_provider, "retrieve_memagent"):
                # Check if agent exists for update vs create
                try:
                    existing = agent.memory_provider.retrieve_memagent(agent.agent_id)
                    if existing and hasattr(agent.memory_provider, "update_memagent"):
                        saved_memagent = agent.memory_provider.update_memagent(
                            memagent_to_save
                        )
                    else:
                        saved_memagent = agent.memory_provider.store_memagent(
                            memagent_to_save
                        )
                except Exception:
                    # Agent doesn't exist, create new
                    saved_memagent = agent.memory_provider.store_memagent(
                        memagent_to_save
                    )
            else:
                # New agent
                saved_memagent = agent.memory_provider.store_memagent(memagent_to_save)

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

            _persist_mcp_servers_to_toolbox_memory(agent)

            logger.info(f"MemAgent {agent.agent_id} saved successfully")
            return agent
        else:
            raise ValueError("Memory provider does not support saving MemAgent")

    except Exception as e:
        logger.error(f"Failed to save MemAgent {agent.agent_id}: {e}")
        raise


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

    tools_metadata = agent.tool_manager.get_tool_metadata()
    if not tools_metadata:
        return None

    # Tools are agent-scoped (shared across threads), so stamp agent_id
    # and leave memory_id unset. The UI toolbox-memory filter treats
    # rows with agent_id but no memory_id as agent-global, which makes
    # them visible on every thread for this agent.
    serializable_tools = []
    for tool_meta in tools_metadata:
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

    return serializable_tools if serializable_tools else None


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


def load_agent(cls, agent_id: str, memory_provider=None, **overrides):
    """Load a MemAgent from the memory provider (see MemAgent.load)."""
    # ``create_llm_provider`` is resolved through the core module at call
    # time so existing patches of ``memorizz.memagent.core.create_llm_provider``
    # keep working after this pure move. Imported lazily to avoid a circular
    # import at module load.
    from . import core as _core

    if not memory_provider:
        # Try to import default memory provider
        try:
            from ..memory_provider import MemoryProvider

            memory_provider = MemoryProvider()
        except ImportError:
            raise ValueError(
                "No memory provider specified and default MemoryProvider not available"
            )

    if not hasattr(memory_provider, "retrieve_memagent"):
        raise ValueError("Memory provider does not support loading MemAgent")

    logger.info(f"Loading MemAgent with agent id {agent_id}...")

    # Retrieve the saved agent
    saved_memagent = memory_provider.retrieve_memagent(agent_id)
    if not saved_memagent:
        raise ValueError(
            f"MemAgent with agent id {agent_id} not found in the memory provider"
        )

    # Reconstruct LLM model. Stash any construction error so we can
    # attach it to the new instance below — otherwise chat would
    # surface a generic "No LLM model configured" instead of the
    # actual cause (e.g. uncached HF repo, network failure).
    model_to_load = None
    load_llm_error: Optional[str] = None
    if hasattr(saved_memagent, "llm_config") and saved_memagent.llm_config:
        try:
            model_to_load = _core.create_llm_provider(saved_memagent.llm_config)
        except Exception as e:
            load_llm_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "Could not load model from config: %s. Model will be None.", e
            )
    elif hasattr(saved_memagent, "model") and saved_memagent.model:
        model_to_load = saved_memagent.model

    # Load delegates if they exist
    loaded_delegates = None
    if hasattr(saved_memagent, "delegates") and saved_memagent.delegates:
        loaded_delegates = []
        for delegate_id in saved_memagent.delegates:
            try:
                delegate_agent = cls.load(delegate_id, memory_provider)
                loaded_delegates.append(delegate_agent)
            except Exception as e:
                logger.warning(f"Could not load delegate agent {delegate_id}: {e}")

    # Reconstruct semantic cache config
    semantic_cache_config_to_load = None
    if (
        hasattr(saved_memagent, "semantic_cache_config")
        and saved_memagent.semantic_cache_config
    ):
        try:
            config_dict = dict(saved_memagent.semantic_cache_config)
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
            semantic_cache_config_to_load = config_dict
        except Exception as e:
            logger.warning(f"Failed to reconstruct semantic cache config: {e}")

    internet_provider_instance = None
    if hasattr(saved_memagent, "internet_access_provider"):
        provider_name = getattr(saved_memagent, "internet_access_provider", None)
        provider_config = getattr(saved_memagent, "internet_access_config", None)
        if not isinstance(provider_name, str):
            provider_name = None
        if provider_config is not None and not isinstance(provider_config, dict):
            provider_config = None
        if provider_name:
            try:
                from ..internet_access import create_internet_access_provider

                internet_provider_instance = create_internet_access_provider(
                    provider_name, provider_config
                )
            except Exception as exc:
                logger.warning(
                    "Failed to restore internet provider '%s': %s",
                    provider_name,
                    exc,
                )

    application_mode_to_use = overrides.get("application_mode")
    if not application_mode_to_use:
        saved_mode = getattr(saved_memagent, "application_mode", None)
        if isinstance(saved_mode, str) and saved_mode:
            application_mode_to_use = saved_mode
        else:
            application_mode_to_use = ApplicationMode.DEFAULT.value

    saved_sandbox_provider = getattr(saved_memagent, "sandbox_provider", None)
    if not isinstance(saved_sandbox_provider, (str, dict)):
        saved_sandbox_provider = None

    saved_browser_control = getattr(saved_memagent, "browser_control", None)
    if not isinstance(saved_browser_control, (str, dict)):
        saved_browser_control = None

    saved_skills_marketplace_provider = getattr(
        saved_memagent, "skills_marketplace_provider", None
    )
    if isinstance(saved_skills_marketplace_provider, dict):
        saved_skills_marketplace_provider = saved_skills_marketplace_provider.get(
            "provider"
        ) or saved_skills_marketplace_provider.get("name")
    if not isinstance(saved_skills_marketplace_provider, str):
        saved_skills_marketplace_provider = None
    else:
        saved_skills_marketplace_provider = (
            saved_skills_marketplace_provider.strip().lower() or None
        )

    saved_skills_marketplace_config = getattr(
        saved_memagent, "skills_marketplace_config", None
    )
    if not isinstance(saved_skills_marketplace_config, dict):
        saved_skills_marketplace_config = None

    saved_skill_paths = getattr(saved_memagent, "skill_paths", None)
    if isinstance(saved_skill_paths, str):
        saved_skill_paths = [saved_skill_paths]
    elif not isinstance(saved_skill_paths, list):
        saved_skill_paths = None

    saved_mcp_servers = getattr(saved_memagent, "mcp_servers", None)
    if not isinstance(saved_mcp_servers, list):
        saved_mcp_servers = None

    raw_self_aware = getattr(saved_memagent, "self_aware", False)
    saved_self_aware = raw_self_aware if isinstance(raw_self_aware, bool) else False
    saved_self_aware_config = getattr(saved_memagent, "self_aware_config", None)
    if not isinstance(saved_self_aware_config, dict):
        saved_self_aware_config = None

    raw_continual_learning = getattr(saved_memagent, "continual_learning", False)
    saved_continual_learning = (
        raw_continual_learning if isinstance(raw_continual_learning, bool) else False
    )
    saved_continual_learning_config = getattr(
        saved_memagent, "continual_learning_config", None
    )
    if not isinstance(saved_continual_learning_config, dict):
        saved_continual_learning_config = None

    raw_automations_enabled = getattr(saved_memagent, "automations_enabled", True)
    saved_automations_enabled = (
        raw_automations_enabled if isinstance(raw_automations_enabled, bool) else True
    )

    saved_default_timezone = getattr(saved_memagent, "default_timezone", None)
    if not isinstance(saved_default_timezone, str):
        saved_default_timezone = None
    else:
        saved_default_timezone = saved_default_timezone.strip() or None

    saved_tool_result_policy = getattr(saved_memagent, "tool_result_policy", None)
    if not isinstance(saved_tool_result_policy, dict):
        saved_tool_result_policy = None

    saved_context_policy = getattr(saved_memagent, "context_policy", None)
    if not isinstance(saved_context_policy, dict):
        saved_context_policy = None

    raw_skill_retrieval = getattr(saved_memagent, "skill_retrieval", False)
    saved_skill_retrieval = (
        raw_skill_retrieval if isinstance(raw_skill_retrieval, bool) else False
    )
    saved_skill_retrieval_config = getattr(
        saved_memagent, "skill_retrieval_config", None
    )
    if not isinstance(saved_skill_retrieval_config, dict):
        saved_skill_retrieval_config = None

    saved_delegation_config = getattr(saved_memagent, "delegation_config", None)
    if not isinstance(saved_delegation_config, dict):
        saved_delegation_config = None

    saved_semantic_layer = None
    semantic_layer_config = getattr(saved_memagent, "semantic_layer_config", None)
    if isinstance(semantic_layer_config, dict):
        try:
            from ..semantic_layer import SemanticCatalog

            saved_semantic_layer = SemanticCatalog.from_dict(semantic_layer_config)
        except Exception as exc:
            logger.warning("Failed to restore semantic catalog: %s", exc)

    # Create new agent instance with loaded configuration
    agent_instance = cls(
        model=overrides.get("model", model_to_load),
        tools=overrides.get("tools", getattr(saved_memagent, "tools", None)),
        persona=overrides.get("persona", getattr(saved_memagent, "persona", None)),
        name=overrides.get("name", getattr(saved_memagent, "name", None)),
        instruction=overrides.get(
            "instruction", getattr(saved_memagent, "instruction", None)
        ),
        max_steps=overrides.get(
            "max_steps", getattr(saved_memagent, "max_steps", DEFAULT_MAX_STEPS)
        ),
        memory_ids=overrides.get(
            "memory_ids", getattr(saved_memagent, "memory_ids", [])
        ),
        agent_id=agent_id,
        memory_provider=memory_provider,
        application_mode=application_mode_to_use,
        memory_types=overrides.get(
            "memory_types", getattr(saved_memagent, "memory_types", None)
        ),
        delegates=overrides.get("delegates", loaded_delegates),
        semantic_cache=overrides.get(
            "semantic_cache", getattr(saved_memagent, "semantic_cache", False)
        ),
        semantic_cache_config=overrides.get(
            "semantic_cache_config", semantic_cache_config_to_load
        ),
        tool_result_policy=overrides.get(
            "tool_result_policy",
            saved_tool_result_policy,
        ),
        context_policy=overrides.get("context_policy", saved_context_policy),
        skill_retrieval=overrides.get("skill_retrieval", saved_skill_retrieval),
        skill_retrieval_config=overrides.get(
            "skill_retrieval_config",
            saved_skill_retrieval_config,
        ),
        delegation=overrides.get("delegation", saved_delegation_config),
        semantic_layer=overrides.get("semantic_layer", saved_semantic_layer),
        internet_access_provider=overrides.get(
            "internet_access_provider", internet_provider_instance
        ),
        skills_marketplace_provider=overrides.get(
            "skills_marketplace_provider", saved_skills_marketplace_provider
        ),
        skills_marketplace_config=overrides.get(
            "skills_marketplace_config", saved_skills_marketplace_config
        ),
        sandbox_provider=overrides.get("sandbox_provider", saved_sandbox_provider),
        browser_control=overrides.get("browser_control", saved_browser_control),
        skill_paths=overrides.get("skill_paths", saved_skill_paths),
        mcp_servers=overrides.get("mcp_servers", saved_mcp_servers),
        automations_enabled=overrides.get(
            "automations_enabled", saved_automations_enabled
        ),
        default_timezone=overrides.get("default_timezone", saved_default_timezone),
        self_aware=overrides.get("self_aware", saved_self_aware),
        self_aware_config=overrides.get("self_aware_config", saved_self_aware_config),
        continual_learning=overrides.get(
            "continual_learning", saved_continual_learning
        ),
        continual_learning_config=overrides.get(
            "continual_learning_config", saved_continual_learning_config
        ),
        workflow_outcome_evaluator=overrides.get("workflow_outcome_evaluator"),
        is_favorite=overrides.get(
            "is_favorite", getattr(saved_memagent, "is_favorite", False)
        ),
        streaming=overrides.get("streaming", False),
    )

    # Hydrate knowledge_base_ids separately — it isn't a constructor arg
    # (yet) but needs to survive reloads so the `knowledge_base_lookup`
    # tool can scope retrievals to this agent's ingested documents.
    _kb_ids = overrides.get(
        "knowledge_base_ids",
        getattr(saved_memagent, "knowledge_base_ids", None),
    )
    agent_instance.knowledge_base_ids = (
        list(_kb_ids) if isinstance(_kb_ids, (list, tuple)) else []
    )

    # Carry the LLM-init failure forward so chat surfaces the real
    # cause. Constructor sets this when *it* tries to build the LLM;
    # here we propagate the error from the load-time create_llm_provider
    # call above (the constructor never sees llm_config in the load
    # path so its own try/except can't catch this case).
    if load_llm_error and not getattr(agent_instance, "_llm_init_error", None):
        agent_instance._llm_init_error = load_llm_error

    logger.info(f"MemAgent loaded successfully with agent_id: {agent_id}")
    return agent_instance


def refresh_agent(agent):
    """Refresh ``agent`` from its memory provider (see MemAgent.refresh)."""
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
                # Update configuration attributes
                if hasattr(saved_memagent, "instruction"):
                    agent.instruction = saved_memagent.instruction
                if hasattr(saved_memagent, "max_steps"):
                    agent.max_steps = saved_memagent.max_steps
                if hasattr(saved_memagent, "memory_ids"):
                    agent.memory_ids = saved_memagent.memory_ids
                if hasattr(saved_memagent, "knowledge_base_ids"):
                    _kb = saved_memagent.knowledge_base_ids
                    agent.knowledge_base_ids = (
                        list(_kb) if isinstance(_kb, (list, tuple)) else []
                    )
                if hasattr(saved_memagent, "name"):
                    agent.name = saved_memagent.name
                if hasattr(saved_memagent, "is_favorite"):
                    agent.is_favorite = bool(saved_memagent.is_favorite)
                if hasattr(saved_memagent, "self_aware"):
                    agent.with_self_aware(
                        bool(getattr(saved_memagent, "self_aware", False)),
                        config=getattr(saved_memagent, "self_aware_config", None),
                    )

                # Update persona if changed. Route through the public
                # ``set_persona`` wrapper so persona evolution tools
                # (update_persona, read_persona) register when a persona
                # is added via refresh(), not just via __init__.
                if hasattr(saved_memagent, "persona") and agent.persona_manager:
                    agent.set_persona(saved_memagent.persona, save=False)

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
