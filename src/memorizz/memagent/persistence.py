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
from ..task_decomposition import normalize_delegation_config
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
        llm_config_to_save = None
        if agent.model and hasattr(agent.model, "get_config"):
            candidate_llm_config = agent.model.get_config()
            if isinstance(candidate_llm_config, dict):
                llm_config_to_save = candidate_llm_config

        # Create MemAgentModel with current configuration
        from .models import MemAgentModel

        memagent_to_save = MemAgentModel(
            llm_config=llm_config_to_save,
            application_id=getattr(agent, "application_id", None),
            name=agent.name,
            instruction=agent.instruction,
            max_steps=agent.max_steps,
            tool_access=getattr(agent, "tool_access", "private"),
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
            completion_policy=agent.completion_policy.to_dict(),
            retrieval_policy=agent.retrieval_policy.to_dict(),
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
            meta_harness=bool(getattr(agent, "meta_harness", None)),
            meta_harness_mode=getattr(agent, "meta_harness_mode", None),
            default_harness=getattr(agent, "default_harness", "auto"),
            harness_config=dict(getattr(agent, "harness_config", None) or {}),
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
        )

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

    # Reconstruct LLM model. Stash any construction error so we can
    # attach it to the new instance below — otherwise chat would
    # surface a generic "No LLM model configured" instead of the
    # actual cause (e.g. uncached HF repo, network failure).
    saved_llm_config = getattr(saved_memagent, "llm_config", None)
    resolved_llm_config = overrides.get("llm_config", saved_llm_config)
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
    elif saved_llm_config:
        try:
            model_to_load = _core.create_llm_provider(saved_llm_config)
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
                delegate_agent = cls.load(
                    delegate_id,
                    memory_provider,
                    runtime_memory_provider=runtime_memory_provider,
                )
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

    saved_meta_harness = bool(getattr(saved_memagent, "meta_harness", False))
    saved_meta_harness_mode = getattr(saved_memagent, "meta_harness_mode", None)
    if saved_meta_harness_mode not in {"delegate", "runtime"}:
        saved_meta_harness_mode = None
    saved_default_harness = getattr(saved_memagent, "default_harness", "auto")
    if not isinstance(saved_default_harness, str):
        saved_default_harness = "auto"
    saved_harness_config = getattr(saved_memagent, "harness_config", None)
    if not isinstance(saved_harness_config, dict):
        saved_harness_config = None

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

    raw_learning_control_plane = getattr(
        saved_memagent, "learning_control_plane", False
    )
    saved_learning_control_plane = (
        raw_learning_control_plane
        if isinstance(raw_learning_control_plane, bool)
        else False
    )
    saved_learning_control_plane_config = getattr(
        saved_memagent, "learning_control_plane_config", None
    )
    if not isinstance(saved_learning_control_plane_config, dict):
        saved_learning_control_plane_config = None

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

    saved_completion_policy = getattr(saved_memagent, "completion_policy", None)
    if not isinstance(saved_completion_policy, dict):
        saved_completion_policy = None

    saved_retrieval_policy = getattr(saved_memagent, "retrieval_policy", None)
    if not isinstance(saved_retrieval_policy, dict):
        saved_retrieval_policy = None

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

    saved_tool_access = getattr(saved_memagent, "tool_access", "private")
    if not isinstance(saved_tool_access, str) or saved_tool_access not in {
        "private",
        "public",
        "global",
    }:
        saved_tool_access = "private"

    # Create new agent instance with loaded configuration
    agent_instance = cls(
        model=model_to_load,
        # Preserve the secret-free provider metadata as well as the hydrated
        # runtime model.  Passing only ``model`` made a successfully reloaded
        # agent report an empty ``llm_config`` (and therefore no provider or
        # model) to the SDK, CLI, MCP server, and UI.
        llm_config=resolved_llm_config,
        tools=overrides.get("tools", getattr(saved_memagent, "tools", None)),
        persona=overrides.get("persona", getattr(saved_memagent, "persona", None)),
        name=overrides.get("name", getattr(saved_memagent, "name", None)),
        application_id=overrides.get(
            "application_id", getattr(saved_memagent, "application_id", None)
        ),
        instruction=overrides.get(
            "instruction", getattr(saved_memagent, "instruction", None)
        ),
        max_steps=overrides.get(
            "max_steps", getattr(saved_memagent, "max_steps", DEFAULT_MAX_STEPS)
        ),
        tool_access=overrides.get("tool_access", saved_tool_access),
        memory_ids=overrides.get(
            "memory_ids", getattr(saved_memagent, "memory_ids", [])
        ),
        agent_id=agent_id,
        memory_provider=runtime_memory_provider,
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
        completion_policy=overrides.get(
            "completion_policy",
            saved_completion_policy,
        ),
        retrieval_policy=overrides.get("retrieval_policy", saved_retrieval_policy),
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
        meta_harness=overrides.get("meta_harness", saved_meta_harness),
        meta_harness_mode=overrides.get("meta_harness_mode", saved_meta_harness_mode),
        default_harness=overrides.get("default_harness", saved_default_harness),
        harness_config=overrides.get("harness_config", saved_harness_config),
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
        learning_control_plane=overrides.get(
            "learning_control_plane",
            saved_learning_control_plane_config
            if saved_learning_control_plane_config is not None
            else saved_learning_control_plane,
        ),
        workflow_outcome_evaluator=overrides.get("workflow_outcome_evaluator"),
        is_favorite=overrides.get(
            "is_favorite", getattr(saved_memagent, "is_favorite", False)
        ),
        streaming=overrides.get("streaming", True),
        auto_register=overrides.get("auto_register", True),
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

    # Carry the LLM-init failure forward so chat surfaces the real cause.
    # The constructor receives both the resolved model and its persisted
    # config, so it retains introspection metadata without constructing the
    # provider a second time.
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
                if hasattr(saved_memagent, "tool_access"):
                    agent.tool_access = saved_memagent.tool_access or "private"
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
