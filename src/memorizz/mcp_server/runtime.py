# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tenant-safe adapter from MCP tools to Memorizz agents and providers."""

from __future__ import annotations

import inspect
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..approval import (
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    default_approval_store,
)
from ..enums.memory_type import MemoryType
from ..mcp.security import redact
from .auth import RequestIdentity
from .config import EXECUTE_SCOPE, READ_SCOPE, WRITE_SCOPE, MemorizzMCPServerConfig

logger = logging.getLogger(__name__)

_TENANT_MEMORY_TYPES = frozenset(
    {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
        MemoryType.TOOL_LOG,
    }
)
_WRITABLE_MEMORY_TYPES = frozenset(
    {MemoryType.KNOWLEDGE_BASE, MemoryType.SHORT_TERM_MEMORY}
)
_SENSITIVE_AGENT_FIELDS = {
    "model",
    "tools",
    "llm_config",
    "embedding_config",
    "internet_access_config",
    "skills_marketplace_config",
    "sandbox_provider",
    "harness_config",
    "whatsapp_config",
    "mcp_servers",
}
_CREATABLE_LLM_PROVIDERS = {
    "anthropic",
    "azure",
    "huggingface",
    "mlx",
    "ollama",
    "openai",
}


class MemorizzServerError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if str(key).lower() == "embedding":
                continue
            result[str(key)] = _json_value(item)
        return redact(result)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _document_field(document: Dict[str, Any], field: str) -> Any:
    if field in document:
        return document.get(field)
    content = document.get("content")
    return content.get(field) if isinstance(content, dict) else None


def _document_user_id(document: Dict[str, Any]) -> Any:
    return _document_field(document, "user_id")


def _provider_name(value: Any) -> Optional[str]:
    """Return a secret-free provider identifier from persisted config."""
    if isinstance(value, str):
        return value.strip().lower() or None
    if isinstance(value, dict):
        raw = value.get("provider") or value.get("name")
        return str(raw).strip().lower() if raw else None
    return None


def _workspace_is_allowed(path: str, roots: Any) -> bool:
    candidate = Path(path).expanduser().resolve()
    return any(
        candidate == root_path or root_path in candidate.parents
        for root_path in (Path(root).expanduser().resolve() for root in (roots or []))
    )


class MemorizzRuntime:
    """Lazy, synchronized access to one Memorizz deployment."""

    def __init__(
        self,
        config: MemorizzMCPServerConfig,
        *,
        provider: Any = None,
        session_builder: Optional[Callable[..., Any]] = None,
        approval_store: Optional[ApprovalStore] = None,
        meta_harness: Any = None,
    ) -> None:
        self.config = config
        self._provider = provider
        self._session_builder = session_builder
        self._provider_lock = threading.RLock()
        self._agents: Dict[str, Any] = {}
        self._agent_locks: Dict[str, threading.RLock] = {}
        self._conversation_state: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self.approval_store = approval_store or default_approval_store()
        self._meta_harness = meta_harness
        self._meta_harness_lock = threading.RLock()

    @staticmethod
    def _approval_owner(identity: RequestIdentity) -> str:
        return f"mcp-server:{identity.principal or 'anonymous'}"

    @property
    def provider(self):
        with self._provider_lock:
            if self._provider is None:
                from ..cli import agent_factory
                from ..cli.config import load_layered_env

                load_layered_env()
                warnings: List[str] = []
                self._provider = agent_factory.detect_memory_provider({}, warnings)
                for warning in warnings:
                    logger.warning("MCP server provider: %s", warning)
            return self._provider

    @property
    def meta_harness(self):
        """Lazily build the shared durable harness service for this server."""
        with self._meta_harness_lock:
            if self._meta_harness is None:
                from ..metaharness import MetaHarness

                self._meta_harness = MetaHarness.from_env(
                    memory_provider=self.provider,
                    approval_store=self.approval_store,
                    allowed_workspace_roots=self.config.harness_workspace_roots,
                )
            return self._meta_harness

    def _require_scope(self, identity: RequestIdentity, scope: str) -> None:
        if scope not in identity.scopes:
            raise MemorizzServerError(
                "insufficient_scope", f"This operation requires the {scope} scope"
            )
        if scope == WRITE_SCOPE and not self.config.allow_writes:
            raise MemorizzServerError(
                "writes_disabled", "Memory writes are disabled by the server operator"
            )
        if scope == EXECUTE_SCOPE and not self.config.allow_agent_execution:
            raise MemorizzServerError(
                "execution_disabled",
                "Agent execution is disabled by the server operator",
            )

    def _normalize_memory_type(
        self, value: str, identity: RequestIdentity, *, writable: bool = False
    ) -> MemoryType:
        try:
            memory_type = MemoryType(str(value or "").strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in MemoryType)
            raise MemorizzServerError(
                "invalid_memory_type", f"Unsupported memory type. Choose: {supported}"
            ) from exc
        if self.config.transport != "stdio" and memory_type not in _TENANT_MEMORY_TYPES:
            raise MemorizzServerError(
                "memory_type_not_tenant_scoped",
                f"Remote access to {memory_type.value} is not tenant scoped",
            )
        if writable and memory_type not in _WRITABLE_MEMORY_TYPES:
            choices = ", ".join(item.value for item in _WRITABLE_MEMORY_TYPES)
            raise MemorizzServerError(
                "memory_type_not_writable", f"Direct writes are limited to: {choices}"
            )
        return memory_type

    def _agent_public(self, agent: Any) -> Dict[str, Any]:
        value = _json_value(agent)
        if not isinstance(value, dict):
            value = {
                field: getattr(agent, field, None)
                for field in (
                    "agent_id",
                    "name",
                    "instruction",
                    "application_mode",
                    "is_favorite",
                    "automations_enabled",
                    "default_timezone",
                )
            }
        value["sandbox_provider_name"] = _provider_name(value.get("sandbox_provider"))
        value["browser_control_provider_name"] = _provider_name(
            value.get("browser_control")
        )
        value["internet_access_provider_name"] = _provider_name(
            value.get("internet_access_provider")
        )
        mcp_servers = value.get("mcp_servers")
        value["mcp_server_count"] = (
            len(mcp_servers) if isinstance(mcp_servers, list) else 0
        )
        llm_config = value.get("llm_config")
        if not isinstance(llm_config, dict):
            llm_config = getattr(agent, "llm_config", None)
        if isinstance(llm_config, dict):
            value["llm_provider"] = llm_config.get("provider")
            value["llm_model"] = llm_config.get("model") or llm_config.get(
                "deployment_name"
            )
        for field in _SENSITIVE_AGENT_FIELDS:
            value.pop(field, None)
        public = {
            key: value.get(key)
            for key in (
                "agent_id",
                "name",
                "instruction",
                "application_mode",
                "max_steps",
                "tool_access",
                "memory_types",
                "is_favorite",
                "semantic_cache",
                "continual_learning",
                "learning_control_plane",
                "skill_retrieval",
                "meta_harness",
                "meta_harness_mode",
                "default_harness",
                "self_aware",
                "automations_enabled",
                "default_timezone",
                "llm_provider",
                "llm_model",
                "sandbox_provider_name",
                "browser_control_provider_name",
                "internet_access_provider_name",
                "mcp_server_count",
            )
            if value.get(key) is not None
        }
        return public

    def _is_agent_exposed(self, agent_id: str, identity: RequestIdentity) -> bool:
        exposed = self.config.exposed_agent_ids
        if self.config.transport == "stdio" and exposed is None:
            return True
        return bool(exposed and agent_id in exposed)

    def _assert_agent_exposed(self, agent_id: str, identity: RequestIdentity) -> None:
        if not self._is_agent_exposed(agent_id, identity):
            raise MemorizzServerError("agent_not_found", "Agent was not found")

    def server_info(self, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        from .. import __version__

        return {
            "ok": True,
            "name": "memorizz",
            "version": __version__,
            "principal": identity.principal,
            "authenticated": identity.authenticated,
            "scopes": sorted(identity.scopes),
            "policy": self.config.public_dict(),
            "memory_types": [
                item.value
                for item in MemoryType
                if self.config.transport == "stdio" or item in _TENANT_MEMORY_TYPES
            ],
            "directly_writable_memory_types": [
                item.value for item in _WRITABLE_MEMORY_TYPES
            ],
            "surface_parity": {
                "agent_lifecycle": [
                    "create",
                    "read",
                    "update",
                    "delete",
                    "execute",
                ],
                "memory": ["list", "search", "read", "store", "forget"],
                "conversation": ["list", "read", "compact"],
                "meta_harness": [
                    "list",
                    "start",
                    "read",
                    "list_runs",
                    "events",
                    "cancel",
                ],
                "operations": [
                    "capabilities",
                    "observability",
                    "semantic_cache",
                    "learning_report",
                    "compile_memory",
                ],
                "trusted_host_only": [
                    "credential_management",
                    "local_path_ingestion",
                    "approval_decisions",
                    "outbound_mcp_configuration",
                ],
            },
        }

    def list_agents(self, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        try:
            agents = list(self.provider.list_memagents() or [])
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not list agents"
            ) from exc
        visible = []
        for agent in agents:
            agent_id = str(
                getattr(agent, "agent_id", None)
                or (agent.get("agent_id") if isinstance(agent, dict) else "")
                or ""
            )
            if agent_id and self._is_agent_exposed(agent_id, identity):
                visible.append(self._agent_public(agent))
        return {"ok": True, "agents": visible, "count": len(visible)}

    def get_agent(self, agent_id: str, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        normalized = str(agent_id or "").strip()
        self._assert_agent_exposed(normalized, identity)
        try:
            agent = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the agent"
            ) from exc
        if not agent:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        return {"ok": True, "agent": self._agent_public(agent)}

    def create_agent(
        self,
        name: str,
        identity: RequestIdentity,
        *,
        instruction: Optional[str] = None,
        application_mode: str = "assistant",
        max_steps: int = 20,
        memory_ids: Optional[List[str]] = None,
        semantic_cache: bool = False,
        continual_learning: bool = False,
        learning_control_plane: bool = False,
        skill_retrieval: bool = False,
        skill_retrieval_top_k: int = 2,
        tool_access: str = "private",
        automations_enabled: bool = True,
        default_timezone: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        meta_harness_mode: Optional[str] = None,
        default_harness: str = "auto",
        harness_workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a validated local agent through the governed MCP write path.

        Remote creation is intentionally disabled until persisted agents have a
        durable tenant-owner field. Treating the server-wide exposed-agent set
        as ownership would allow one authenticated principal to discover an
        agent created by another principal after a restart.
        """
        self._require_scope(identity, WRITE_SCOPE)
        if self.config.transport != "stdio":
            raise MemorizzServerError(
                "agent_creation_local_only",
                "Agent creation is available only over local stdio MCP; create "
                "remote agents through the authenticated host SDK, CLI, or UI",
            )

        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise MemorizzServerError("invalid_agent", "Agent name cannot be empty")
        if len(normalized_name) > 200:
            raise MemorizzServerError(
                "invalid_agent", "Agent name cannot exceed 200 characters"
            )
        normalized_instruction = str(instruction or "").strip() or None
        if (
            normalized_instruction
            and len(normalized_instruction) > self.config.max_text_chars
        ):
            raise MemorizzServerError(
                "invalid_agent",
                f"Agent instruction cannot exceed {self.config.max_text_chars} characters",
            )
        try:
            normalized_steps = int(max_steps)
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError(
                "invalid_agent", "max_steps must be an integer"
            ) from exc
        if not 1 <= normalized_steps <= 1000:
            raise MemorizzServerError(
                "invalid_agent", "max_steps must be between 1 and 1000"
            )

        from ..enums import ApplicationModeConfig

        try:
            mode = ApplicationModeConfig.validate_mode(application_mode).value
        except ValueError as exc:
            raise MemorizzServerError("invalid_agent", str(exc)) from exc

        normalized_tool_access = str(tool_access or "private").strip().lower()
        if normalized_tool_access not in {"private", "public", "global"}:
            raise MemorizzServerError(
                "invalid_agent", "tool_access must be private, public, or global"
            )
        try:
            normalized_skill_top_k = int(skill_retrieval_top_k)
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError(
                "invalid_agent", "skill_retrieval_top_k must be an integer"
            ) from exc
        if not 1 <= normalized_skill_top_k <= 50:
            raise MemorizzServerError(
                "invalid_agent", "skill_retrieval_top_k must be between 1 and 50"
            )
        normalized_timezone = str(default_timezone or "").strip() or None
        if normalized_timezone:
            try:
                from ..automation.schedule import validate_timezone_name

                validate_timezone_name(normalized_timezone)
            except Exception as exc:
                raise MemorizzServerError("invalid_agent", str(exc)) from exc

        normalized_harness_mode = str(meta_harness_mode or "").strip().lower()
        if normalized_harness_mode not in {"", "delegate", "runtime"}:
            raise MemorizzServerError(
                "invalid_agent", "meta_harness_mode must be delegate or runtime"
            )
        if normalized_harness_mode and not self.config.allow_harness_execution:
            raise MemorizzServerError(
                "harness_execution_disabled",
                "Meta-harness agent configuration is disabled by the server operator",
            )
        normalized_default_harness = (
            str(default_harness or "auto").strip().lower().replace("_", "-")
        )
        if normalized_default_harness not in {
            "auto",
            "codex",
            "claude-code",
            "openhands",
            "native",
        }:
            raise MemorizzServerError(
                "invalid_agent",
                "default_harness must be auto, codex, claude-code, openhands, or native",
            )
        normalized_harness_workspace = str(harness_workspace or "").strip()
        harness_config: Dict[str, Any] = {}
        if normalized_harness_workspace:
            normalized_harness_workspace = os.path.realpath(
                os.path.expanduser(normalized_harness_workspace)
            )
            roots = self.config.harness_workspace_roots or set()
            if not _workspace_is_allowed(normalized_harness_workspace, roots):
                raise MemorizzServerError(
                    "invalid_agent",
                    "harness_workspace must be inside an operator-allowed workspace root",
                )
            harness_config["workspace"] = normalized_harness_workspace
            harness_config["permissions"] = {
                "allowed_roots": [normalized_harness_workspace]
            }

        normalized_memory_ids: List[str] = []
        for raw_memory_id in list(memory_ids or []):
            memory_id = str(raw_memory_id or "").strip()
            if not memory_id:
                continue
            if len(memory_id) > 256 or "\x00" in memory_id:
                raise MemorizzServerError(
                    "invalid_agent", "Memory IDs must be at most 256 characters"
                )
            if memory_id not in normalized_memory_ids:
                normalized_memory_ids.append(memory_id)
            if len(normalized_memory_ids) > self.config.max_result_items:
                raise MemorizzServerError(
                    "invalid_agent",
                    f"At most {self.config.max_result_items} memory IDs are allowed",
                )

        from ..memagent.builders import MemAgentBuilder

        normalized_llm_provider = str(llm_provider or "").strip().lower()
        normalized_llm_model = str(llm_model or "").strip()
        if normalized_llm_model and not normalized_llm_provider:
            raise MemorizzServerError(
                "invalid_agent", "llm_model requires llm_provider"
            )
        llm_config = None
        if normalized_llm_provider:
            if normalized_llm_provider not in _CREATABLE_LLM_PROVIDERS:
                choices = ", ".join(sorted(_CREATABLE_LLM_PROVIDERS))
                raise MemorizzServerError(
                    "invalid_agent", f"Unsupported LLM provider. Choose: {choices}"
                )
            from ..cli import agent_factory

            llm_config = agent_factory.config_for_provider(normalized_llm_provider)
            if normalized_llm_model:
                if normalized_llm_provider == "azure":
                    llm_config.pop("model", None)
                    llm_config["deployment_name"] = normalized_llm_model
                else:
                    llm_config["model"] = normalized_llm_model

        builder = (
            MemAgentBuilder()
            .with_name(normalized_name)
            .with_application_mode(mode)
            .with_max_steps(normalized_steps)
            .with_memory_provider(self.provider)
            .with_semantic_cache(bool(semantic_cache))
            .with_continual_learning(bool(continual_learning))
            .with_learning_control_plane(bool(learning_control_plane))
            .with_skill_retrieval(bool(skill_retrieval), top_k=normalized_skill_top_k)
            .with_tool_access(normalized_tool_access)
            .with_automations_enabled(bool(automations_enabled))
        )
        if normalized_timezone:
            builder.with_default_timezone(normalized_timezone)
        if normalized_instruction:
            builder.with_instruction(normalized_instruction)
        if normalized_memory_ids:
            builder.with_memory_ids(normalized_memory_ids)
        if llm_config:
            builder.with_llm_config(llm_config)
        if normalized_harness_mode:
            builder.with_meta_harness(
                self.meta_harness,
                mode=normalized_harness_mode,
                default_harness=normalized_default_harness,
                config=harness_config,
            )
        try:
            agent = builder.build()
            if llm_config and getattr(agent, "_llm_init_error", None):
                raise MemorizzServerError(
                    "agent_configuration_error",
                    "The requested LLM provider could not be initialized; check "
                    "the server credentials and provider configuration",
                )
            agent.save()
        except MemorizzServerError:
            raise
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not create the agent"
            ) from exc

        agent_id = str(getattr(agent, "agent_id", "") or "")
        if not agent_id:
            raise MemorizzServerError(
                "provider_error", "The memory provider returned no agent ID"
            )
        self._agents[agent_id] = agent
        self._agent_locks.setdefault(agent_id, threading.RLock())
        if self.config.exposed_agent_ids is not None:
            self.config.exposed_agent_ids.add(agent_id)
        return {"ok": True, "created": True, "agent": self._agent_public(agent)}

    def _local_agent_record(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        operation: str,
    ) -> Tuple[str, Any]:
        """Resolve a persisted agent for trusted local lifecycle management."""
        self._require_scope(identity, WRITE_SCOPE)
        if self.config.transport != "stdio":
            raise MemorizzServerError(
                "agent_management_local_only",
                f"Agent {operation} is available only over local stdio MCP; use "
                "the authenticated host SDK, CLI, or UI for remote administration",
            )
        normalized = str(agent_id or "").strip()
        if not normalized:
            raise MemorizzServerError("invalid_agent", "agent_id cannot be empty")
        self._assert_agent_exposed(normalized, identity)
        try:
            existing = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the agent"
            ) from exc
        if not existing:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        return normalized, existing

    def _discard_cached_agent(self, agent_id: str) -> None:
        cached = self._agents.pop(agent_id, None)
        self._agent_locks.pop(agent_id, None)
        if cached is not None:
            close = getattr(cached, "close", None)
            if callable(close):
                try:
                    close(close_memory_provider=False)
                except TypeError:
                    close()

    def update_agent(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        name: Optional[str] = None,
        instruction: Optional[str] = None,
        application_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        memory_ids: Optional[List[str]] = None,
        tool_access: Optional[str] = None,
        semantic_cache: Optional[bool] = None,
        continual_learning: Optional[bool] = None,
        learning_control_plane: Optional[bool] = None,
        skill_retrieval: Optional[bool] = None,
        skill_retrieval_top_k: Optional[int] = None,
        automations_enabled: Optional[bool] = None,
        default_timezone: Optional[str] = None,
        is_favorite: Optional[bool] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        clear_llm: bool = False,
        meta_harness_mode: Optional[str] = None,
        default_harness: Optional[str] = None,
        harness_workspace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update the safe persisted-agent surface without accepting credentials."""
        normalized, existing = self._local_agent_record(
            agent_id, identity, operation="updates"
        )
        supplied = any(
            value is not None
            for value in (
                name,
                instruction,
                application_mode,
                max_steps,
                memory_ids,
                tool_access,
                semantic_cache,
                continual_learning,
                learning_control_plane,
                skill_retrieval,
                skill_retrieval_top_k,
                automations_enabled,
                default_timezone,
                is_favorite,
                llm_provider,
                llm_model,
                meta_harness_mode,
                default_harness,
                harness_workspace,
            )
        ) or bool(clear_llm)
        if not supplied:
            raise MemorizzServerError(
                "invalid_agent", "At least one agent field must be supplied"
            )

        from ..memagent.models import MemAgentModel

        if isinstance(existing, MemAgentModel):
            updated = existing.model_copy(deep=True)
        elif isinstance(existing, dict):
            updated = MemAgentModel.model_validate(existing)
        else:
            updated = MemAgentModel(
                **{
                    field: getattr(existing, field, None)
                    for field in MemAgentModel.model_fields
                }
            )
        updated.agent_id = normalized

        if name is not None:
            normalized_name = str(name).strip()
            if not normalized_name or len(normalized_name) > 200:
                raise MemorizzServerError(
                    "invalid_agent", "Agent name must be between 1 and 200 characters"
                )
            updated.name = normalized_name
        if instruction is not None:
            normalized_instruction = str(instruction).strip()
            if not normalized_instruction:
                raise MemorizzServerError(
                    "invalid_agent", "Agent instruction cannot be empty"
                )
            if len(normalized_instruction) > self.config.max_text_chars:
                raise MemorizzServerError(
                    "invalid_agent",
                    f"Agent instruction cannot exceed {self.config.max_text_chars} characters",
                )
            updated.instruction = normalized_instruction
        if application_mode is not None:
            from ..enums import ApplicationModeConfig

            try:
                mode = ApplicationModeConfig.validate_mode(application_mode)
            except ValueError as exc:
                raise MemorizzServerError("invalid_agent", str(exc)) from exc
            updated.application_mode = mode.value
            updated.memory_types = [
                item.value for item in ApplicationModeConfig.get_memory_types(mode)
            ]
        if max_steps is not None:
            try:
                normalized_steps = int(max_steps)
            except (TypeError, ValueError) as exc:
                raise MemorizzServerError(
                    "invalid_agent", "max_steps must be an integer"
                ) from exc
            if not 1 <= normalized_steps <= 1000:
                raise MemorizzServerError(
                    "invalid_agent", "max_steps must be between 1 and 1000"
                )
            updated.max_steps = normalized_steps
        if memory_ids is not None:
            normalized_memory_ids: List[str] = []
            for raw_memory_id in memory_ids:
                memory_id = str(raw_memory_id or "").strip()
                if not memory_id:
                    continue
                if len(memory_id) > 256 or "\x00" in memory_id:
                    raise MemorizzServerError(
                        "invalid_agent", "Memory IDs must be at most 256 characters"
                    )
                if memory_id not in normalized_memory_ids:
                    normalized_memory_ids.append(memory_id)
                if len(normalized_memory_ids) > self.config.max_result_items:
                    raise MemorizzServerError(
                        "invalid_agent",
                        f"At most {self.config.max_result_items} memory IDs are allowed",
                    )
            updated.memory_ids = normalized_memory_ids
        if tool_access is not None:
            normalized_tool_access = str(tool_access).strip().lower()
            if normalized_tool_access not in {"private", "public", "global"}:
                raise MemorizzServerError(
                    "invalid_agent", "tool_access must be private, public, or global"
                )
            updated.tool_access = normalized_tool_access
        if semantic_cache is not None:
            updated.semantic_cache = bool(semantic_cache)
            if semantic_cache and not updated.semantic_cache_config:
                updated.semantic_cache_config = {
                    "similarity_threshold": 0.85,
                    "scope": "session",
                }
        if continual_learning is not None:
            updated.continual_learning = bool(continual_learning)
        if learning_control_plane is not None:
            updated.learning_control_plane = bool(learning_control_plane)
            control_config = dict(updated.learning_control_plane_config or {})
            control_config["enabled"] = bool(learning_control_plane)
            updated.learning_control_plane_config = control_config
        if skill_retrieval is not None:
            updated.skill_retrieval = bool(skill_retrieval)
        if skill_retrieval_top_k is not None:
            try:
                normalized_top_k = int(skill_retrieval_top_k)
            except (TypeError, ValueError) as exc:
                raise MemorizzServerError(
                    "invalid_agent", "skill_retrieval_top_k must be an integer"
                ) from exc
            if not 1 <= normalized_top_k <= 50:
                raise MemorizzServerError(
                    "invalid_agent", "skill_retrieval_top_k must be between 1 and 50"
                )
            retrieval_config = dict(updated.skill_retrieval_config or {})
            retrieval_config["top_k"] = normalized_top_k
            retrieval_config.setdefault("min_similarity", 0.70)
            updated.skill_retrieval_config = retrieval_config
        # Preserve the continual-learning memory surface when another update
        # (for example, changing application mode) recalculates memory types.
        # Looking only at the incoming optional flag could silently remove
        # workflow/skill memory from an already-learning agent.
        if updated.continual_learning:
            updated.skill_retrieval = True
            memory_types = list(updated.memory_types or [])
            for required in (MemoryType.WORKFLOW_MEMORY, MemoryType.SKILLBOX):
                if required.value not in memory_types:
                    memory_types.append(required.value)
            updated.memory_types = memory_types
        if automations_enabled is not None:
            updated.automations_enabled = bool(automations_enabled)
        if default_timezone is not None:
            normalized_timezone = str(default_timezone).strip() or None
            if normalized_timezone:
                try:
                    from ..automation.schedule import validate_timezone_name

                    validate_timezone_name(normalized_timezone)
                except Exception as exc:
                    raise MemorizzServerError("invalid_agent", str(exc)) from exc
            updated.default_timezone = normalized_timezone
        if is_favorite is not None:
            updated.is_favorite = bool(is_favorite)
        if meta_harness_mode is not None:
            normalized_harness_mode = str(meta_harness_mode).strip().lower()
            if normalized_harness_mode not in {"", "delegate", "runtime"}:
                raise MemorizzServerError(
                    "invalid_agent", "meta_harness_mode must be delegate or runtime"
                )
            updated.meta_harness = bool(normalized_harness_mode)
            updated.meta_harness_mode = normalized_harness_mode or None
        if default_harness is not None:
            normalized_default_harness = (
                str(default_harness or "auto").strip().lower().replace("_", "-")
            )
            if normalized_default_harness not in {
                "auto",
                "codex",
                "claude-code",
                "openhands",
                "native",
            }:
                raise MemorizzServerError(
                    "invalid_agent",
                    "default_harness must be auto, codex, claude-code, openhands, or native",
                )
            updated.default_harness = normalized_default_harness
        if harness_workspace is not None:
            normalized_harness_workspace = str(harness_workspace).strip()
            harness_config = dict(updated.harness_config or {})
            if normalized_harness_workspace:
                normalized_harness_workspace = os.path.realpath(
                    os.path.expanduser(normalized_harness_workspace)
                )
                roots = self.config.harness_workspace_roots or set()
                if not _workspace_is_allowed(normalized_harness_workspace, roots):
                    raise MemorizzServerError(
                        "invalid_agent",
                        "harness_workspace must be inside an operator-allowed workspace root",
                    )
                harness_config["workspace"] = normalized_harness_workspace
                permissions = dict(harness_config.get("permissions") or {})
                permissions["allowed_roots"] = [normalized_harness_workspace]
                harness_config["permissions"] = permissions
            else:
                harness_config.pop("workspace", None)
                harness_config.pop("permissions", None)
            updated.harness_config = harness_config

        if clear_llm and (llm_provider is not None or llm_model is not None):
            raise MemorizzServerError(
                "invalid_agent", "clear_llm cannot be combined with LLM fields"
            )
        if clear_llm:
            updated.llm_config = None
        elif llm_provider is not None or llm_model is not None:
            existing_llm = dict(updated.llm_config or {})
            normalized_llm_provider = (
                str(llm_provider or existing_llm.get("provider") or "").strip().lower()
            )
            if not normalized_llm_provider:
                raise MemorizzServerError(
                    "invalid_agent",
                    "llm_model requires an existing or supplied provider",
                )
            if normalized_llm_provider not in _CREATABLE_LLM_PROVIDERS:
                choices = ", ".join(sorted(_CREATABLE_LLM_PROVIDERS))
                raise MemorizzServerError(
                    "invalid_agent", f"Unsupported LLM provider. Choose: {choices}"
                )
            from ..cli import agent_factory

            llm_config = agent_factory.config_for_provider(normalized_llm_provider)
            normalized_model = str(llm_model or "").strip()
            if normalized_model:
                if normalized_llm_provider == "azure":
                    llm_config.pop("model", None)
                    llm_config["deployment_name"] = normalized_model
                else:
                    llm_config["model"] = normalized_model
            elif llm_provider is None:
                old_model = existing_llm.get("model") or existing_llm.get(
                    "deployment_name"
                )
                if old_model:
                    if normalized_llm_provider == "azure":
                        llm_config["deployment_name"] = old_model
                    else:
                        llm_config["model"] = old_model
            try:
                from ..llms.llm_factory import create_llm_provider

                candidate_provider = create_llm_provider(llm_config)
                close = getattr(candidate_provider, "close", None)
                if callable(close):
                    close()
            except Exception as exc:
                raise MemorizzServerError(
                    "agent_configuration_error",
                    "The requested LLM provider could not be initialized; check "
                    "the server credentials and provider configuration",
                ) from exc
            updated.llm_config = llm_config

        try:
            self.provider.store_memagent(updated)
            persisted = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not update the agent"
            ) from exc
        if not persisted:
            raise MemorizzServerError(
                "provider_error", "The memory provider returned no updated agent"
            )
        self._discard_cached_agent(normalized)
        return {
            "ok": True,
            "updated": True,
            "agent": self._agent_public(persisted),
        }

    def delete_agent(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        cascade: bool = False,
    ) -> Dict[str, Any]:
        """Create a durable exact-call proposal for local agent deletion."""
        normalized, _existing = self._local_agent_record(
            agent_id, identity, operation="deletion"
        )
        arguments = {"agent_id": normalized, "cascade": bool(cascade)}
        proposal = self.approval_store.propose(
            owner_id=self._approval_owner(identity),
            tool_name="memorizz_delete_agent",
            arguments=arguments,
            policy_reason=(
                "Agent deletion requires a human host decision; cascade also removes "
                "the agent's attached memory scopes"
            ),
            checkpoint={
                "version": 1,
                "operation": "delete_agent",
                "arguments": arguments,
                "principal": identity.principal,
                "scopes": sorted(identity.scopes),
                "authenticated": identity.authenticated,
            },
            ttl_seconds=self.config.approval_ttl_seconds,
        )
        return {
            "ok": False,
            "status": "approval_required",
            "proposal": proposal.to_dict(include_arguments=True),
        }

    def _delete_agent_authorized(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        cascade: bool,
    ) -> Dict[str, Any]:
        normalized, _existing = self._local_agent_record(
            agent_id, identity, operation="deletion"
        )
        try:
            deleted = bool(self.provider.delete_memagent(normalized, cascade=cascade))
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not delete the agent"
            ) from exc
        if deleted:
            self._discard_cached_agent(normalized)
            if self.config.exposed_agent_ids is not None:
                self.config.exposed_agent_ids.discard(normalized)
            for state_key in list(self._conversation_state):
                if state_key[1] == normalized:
                    self._conversation_state.pop(state_key, None)
        return {
            "ok": deleted,
            "agent_id": normalized,
            "cascade": bool(cascade),
            "deleted": deleted,
        }

    def _default_session(self):
        if self._session_builder:
            return self._session_builder(memory_provider=self.provider)
        from ..cli import agent_factory
        from ..cli.config import load_layered_env

        load_layered_env()
        try:
            llm_config = agent_factory.detect_llm_config()
            return agent_factory.build_session_agent(
                memory_provider=self.provider, llm_config=llm_config
            )
        except Exception as exc:
            raise MemorizzServerError(
                "agent_configuration_error",
                "No runnable Memorizz LLM is configured for agent execution",
            ) from exc

    def _load_runnable_agent(
        self, agent_id: Optional[str], identity: RequestIdentity
    ) -> Any:
        normalized = str(agent_id or "").strip()
        if not normalized and self.config.transport != "stdio":
            exposed = sorted(self.config.exposed_agent_ids or [])
            if len(exposed) == 1:
                normalized = exposed[0]
            else:
                raise MemorizzServerError(
                    "agent_id_required",
                    "Select one of the explicitly exposed agents by agent_id",
                )
        if not normalized:
            session = self._default_session()
            agent = session.agent
            normalized = str(getattr(agent, "agent_id", ""))
            if (
                self.config.transport == "stdio"
                and self.config.exposed_agent_ids is not None
            ):
                self.config.exposed_agent_ids.add(normalized)
        else:
            self._assert_agent_exposed(normalized, identity)
            if normalized in self._agents:
                return self._agents[normalized]
            from ..memagent import MemAgent

            try:
                agent = MemAgent.load(normalized, memory_provider=self.provider)
            except Exception as exc:
                raise MemorizzServerError(
                    "agent_not_runnable", "Agent could not be loaded for execution"
                ) from exc
        self._agents[normalized] = agent
        self._agent_locks.setdefault(normalized, threading.RLock())
        return agent

    def _load_agent_for_inspection(
        self, agent_id: str, identity: RequestIdentity
    ) -> Any:
        normalized = str(agent_id or "").strip()
        if not normalized:
            raise MemorizzServerError("invalid_agent", "agent_id cannot be empty")
        self._assert_agent_exposed(normalized, identity)
        if normalized in self._agents:
            return self._agents[normalized]
        from ..memagent import MemAgent

        try:
            agent = MemAgent.load(normalized, memory_provider=self.provider)
        except Exception as exc:
            raise MemorizzServerError(
                "agent_not_runnable", "Agent could not be loaded for inspection"
            ) from exc
        self._agents[normalized] = agent
        self._agent_locks.setdefault(normalized, threading.RLock())
        return agent

    def query_traces(
        self,
        agent_id,
        identity,
        *,
        thread_id=None,
        root_trace_id=None,
        cursor=None,
        limit=250,
        explain=False,
    ):
        """Opt-in metadata-only inspection; caller cannot supply another tenant."""
        self._require_scope(identity, READ_SCOPE)
        if not self.config.allow_trace_queries:
            raise MemorizzServerError(
                "trace_queries_disabled", "Trace queries are disabled by server policy"
            )
        self._assert_agent_exposed(agent_id, identity)
        from ..observability.index import _opaque_fields, validate_filters
        from ..observability.normalization import TraceEvents
        from ..observability.privacy import pseudonym

        try:
            filters = validate_filters(
                {
                    "agent_ids": [agent_id],
                    "user_id": identity.principal,
                    "thread_id": thread_id,
                    "root_trace_id": root_trace_id,
                }
            )
            page = self.provider.query_trace_events(
                limit=self._bounded_limit(limit), cursor=cursor, **filters
            )
        except ValueError:
            raise MemorizzServerError(
                "invalid_trace_query",
                "Use bounded opaque trace IDs and a valid scoped cursor",
            ) from None
        except Exception:
            raise MemorizzServerError(
                "trace_query_failed", "Trace evidence is unavailable"
            ) from None
        rows = [_opaque_fields(event) for event in page["items"]]
        for row in rows:
            if row.get("user_id"):
                row["user_id"] = pseudonym(row["user_id"])
        result = {"ok": True, **page, "items": rows}
        if explain:
            from ..observability.lineage import build_lineage_inspectors
            from ..observability.trace_analysis import analyze_trace_events

            window = TraceEvents(
                rows,
                coverage={key: value for key, value in page.items() if key != "items"},
            )
            result["lineage"] = build_lineage_inspectors(window)
            result["analysis"] = analyze_trace_events(window, agent_id=agent_id)
        logger.info(
            "MCP metadata trace query: caller=%s events=%d",
            pseudonym(identity.principal or "anonymous"),
            len(rows),
        )
        return result

    def inspect_agent(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_package_capabilities: bool = False,
    ) -> Dict[str, Any]:
        """Return the SDK operational views through one bounded read tool."""
        self._require_scope(identity, READ_SCOPE)
        agent = self._load_agent_for_inspection(agent_id, identity)
        try:
            capability_report = agent.capability_report(preflight=False)
            capabilities = (
                capability_report
                if include_package_capabilities
                else capability_report.get("agent", {})
            )
            learning = agent.learning_report(
                memory_id=memory_id,
                user_id=identity.principal,
                thread_id=thread_id,
            )
            observability = None
            if memory_id:
                observability = agent.observability_summary(
                    memory_id,
                    identity.principal,
                    thread_id=thread_id,
                    limit=self.config.max_result_items,
                )
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "Agent operational state could not be inspected"
            ) from exc
        result = {
            "ok": True,
            "agent": self._agent_public(agent),
            "capabilities": _json_value(capabilities),
            "semantic_cache": _json_value(agent.semantic_cache_stats()),
            "learning_control_plane": _json_value(learning),
        }
        if observability is not None:
            result["observability"] = _json_value(observability)
        return result

    def preview_personalization(
        self,
        agent_id: str,
        query: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        exclude_thread_id: Optional[str] = None,
        include_conversation_recall: bool = False,
        min_relevance_score: float = 0.65,
        max_conversation_memories: int = 2,
        preferences: Optional[Dict[str, Any]] = None,
        writing_samples: Optional[List[Dict[str, Any]]] = None,
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """Build one bounded personalization context without running the model."""
        self._require_scope(identity, READ_SCOPE)
        text = str(query or "").strip()
        if not text:
            raise MemorizzServerError(
                "invalid_query", "Personalization query cannot be empty"
            )
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "query_too_large",
                f"Query exceeds {self.config.max_text_chars} characters",
            )
        agent = self._load_agent_for_inspection(agent_id, identity)
        agent_memory_ids = {
            str(value).strip()
            for value in (getattr(agent, "memory_ids", None) or [])
            if str(value).strip()
        }
        resolved_memory_id = str(memory_id or "").strip()
        if not resolved_memory_id:
            if len(agent_memory_ids) != 1:
                raise MemorizzServerError(
                    "memory_id_required",
                    "Select one explicit memory_id for personalization preview",
                )
            resolved_memory_id = next(iter(agent_memory_ids))
        if resolved_memory_id not in agent_memory_ids:
            raise MemorizzServerError(
                "memory_not_attached",
                "The selected memory is not attached to this agent",
            )

        bounded_preferences = {
            str(key)[:80]: str(value)[:600]
            for key, value in dict(preferences or {}).items()
            if str(key).strip() and value not in (None, "")
        }
        bounded_samples = []
        for raw in writing_samples or []:
            if not isinstance(raw, dict):
                continue
            bounded_samples.append(
                {
                    "sample_id": str(
                        raw.get("sample_id") or raw.get("id") or uuid.uuid4()
                    )[:160],
                    "title": str(raw.get("title") or "Writing sample")[:160],
                    "text": str(
                        raw.get("text")
                        or raw.get("content")
                        or raw.get("excerpt")
                        or ""
                    )[:2000],
                }
            )
            if len(bounded_samples) >= 3:
                break

        try:
            context = agent.build_personalization_context(
                text,
                memory_id=resolved_memory_id,
                user_id=identity.principal,
                preferences=bounded_preferences,
                writing_samples=bounded_samples,
                exclude_thread_id=exclude_thread_id,
                policy={
                    "conversation_recall": bool(include_conversation_recall),
                    "min_relevance_score": float(min_relevance_score),
                    "max_conversation_memories": max(
                        0,
                        min(
                            int(max_conversation_memories),
                            self.config.max_result_items,
                        ),
                    ),
                },
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError(
                "invalid_personalization_policy", str(exc)
            ) from exc
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "Personalization context could not be assembled"
            ) from exc

        result = {
            "ok": True,
            "agent_id": str(agent_id),
            "memory_id": resolved_memory_id,
            "context_evidence": context.trace_summary(),
        }
        if include_content:
            result["context"] = context.to_dict()
            result["prompt_block"] = context.render()
        return _json_value(result)

    def compile_agent_memory(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """Compile tenant-scoped learning events into retrieval artifacts."""
        self._require_scope(identity, WRITE_SCOPE)
        agent = self._load_agent_for_inspection(agent_id, identity)
        normalized_mode = str(mode or "fast").strip().lower()
        if normalized_mode not in {"fast", "full"}:
            raise MemorizzServerError(
                "invalid_compile_mode", "Compilation mode must be fast or full"
            )
        try:
            report = agent.compile_memory(
                memory_id=memory_id,
                user_id=identity.principal,
                thread_id=thread_id,
                mode=normalized_mode,
            )
        except ValueError as exc:
            raise MemorizzServerError("feature_disabled", str(exc)) from exc
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "Agent memory compilation failed"
            ) from exc
        return {"ok": True, "compile_report": _json_value(report)}

    def compact_conversation(
        self,
        agent_id: str,
        memory_id: str,
        identity: RequestIdentity,
        *,
        thread_id: Optional[str] = None,
        days_back: int = 7,
        max_memories_per_summary: int = 50,
    ) -> Dict[str, Any]:
        """Generate tenant-scoped summaries using the configured agent LLM."""
        self._require_scope(identity, EXECUTE_SCOPE)
        self._require_scope(identity, WRITE_SCOPE)
        resolved_memory_id = str(memory_id or "").strip()
        if not resolved_memory_id:
            raise MemorizzServerError("invalid_memory_id", "memory_id is required")
        bounded_days = max(1, min(int(days_back), 3650))
        bounded_memories = max(2, min(int(max_memories_per_summary), 1000))
        agent = self._load_agent_for_inspection(agent_id, identity)
        if getattr(agent, "model", None) is None:
            raise MemorizzServerError(
                "agent_configuration_error",
                "Conversation compaction requires a configured agent LLM",
            )
        try:
            summary_ids = agent.generate_summaries(
                days_back=bounded_days,
                max_memories_per_summary=bounded_memories,
                memory_id=resolved_memory_id,
                user_id=identity.principal,
                thread_id=thread_id,
            )
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "Conversation compaction failed"
            ) from exc
        return {
            "ok": True,
            "agent_id": str(agent_id),
            "memory_id": resolved_memory_id,
            "thread_id": thread_id,
            "summary_ids": [str(value) for value in summary_ids],
            "summary_count": len(summary_ids),
        }

    def execute_agent(
        self,
        message: str,
        identity: RequestIdentity,
        *,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_scope(identity, EXECUTE_SCOPE)
        # Every agent turn persists conversation state, even when the selected
        # agent does not call an external mutating tool.
        self._require_scope(identity, WRITE_SCOPE)
        text = str(message or "").strip()
        if not text:
            raise MemorizzServerError("invalid_message", "Message cannot be empty")
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "message_too_large",
                f"Message exceeds {self.config.max_text_chars} characters",
            )
        agent = self._load_runnable_agent(agent_id, identity)
        resolved_agent_id = str(getattr(agent, "agent_id", "default"))
        state_key = (identity.principal or "local", resolved_agent_id)
        previous = self._conversation_state.get(state_key)
        resolved_memory_id = str(
            memory_id or (previous[0] if previous else uuid.uuid4())
        )
        resolved_thread_id = str(
            thread_id or (previous[1] if previous else uuid.uuid4())
        )
        lock = self._agent_locks.setdefault(resolved_agent_id, threading.RLock())
        with lock:
            try:
                response = agent.run(
                    text,
                    memory_id=resolved_memory_id,
                    thread_id=resolved_thread_id,
                    user_id=identity.principal,
                    context=context or {},
                    tool_context={
                        "user_id": identity.principal,
                        "mcp_principal": identity.principal,
                    },
                )
                if getattr(agent, "memory_provider", None) is not None:
                    agent.save()
            except Exception as exc:
                logger.exception("Memorizz MCP agent execution failed")
                raise MemorizzServerError(
                    "agent_execution_error", "The agent could not complete the request"
                ) from exc
        self._conversation_state[state_key] = (
            resolved_memory_id,
            resolved_thread_id,
        )
        raw_outcomes = getattr(agent, "last_tool_outcomes", [])
        if callable(raw_outcomes):
            raw_outcomes = raw_outcomes()
        tool_outcomes = [
            _json_value(item) for item in (raw_outcomes or []) if isinstance(item, dict)
        ]
        return {
            "ok": True,
            "agent_id": resolved_agent_id,
            "memory_id": resolved_memory_id,
            "thread_id": resolved_thread_id,
            "response": str(response),
            "tool_outcomes": tool_outcomes,
        }

    def execute_agent_events(
        self,
        message,
        identity,
        *,
        agent_id=None,
        memory_id=None,
        thread_id=None,
        context=None,
        delivery_mode=None,
    ):
        """Authorized native stream; the agent lock covers generation AND saving."""
        from ..streaming import agent_event_stream, check_cancelled

        self._require_scope(identity, EXECUTE_SCOPE)
        self._require_scope(identity, WRITE_SCOPE)
        text = str(message or "").strip()
        if not text:
            raise MemorizzServerError("invalid_message", "Message cannot be empty")
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "message_too_large", "Message exceeds configured limit"
            )
        agent = self._load_runnable_agent(agent_id, identity)
        resolved_agent_id = str(agent.agent_id)
        key = (identity.principal or "local", resolved_agent_id)
        # Reserve stable principal-local IDs before another request starts.
        previous = self._conversation_state.setdefault(
            key, (str(uuid.uuid4()), str(uuid.uuid4()))
        )
        resolved_memory_id = str(memory_id or previous[0])
        resolved_thread_id = str(thread_id or previous[1])
        lock = self._agent_locks.setdefault(resolved_agent_id, threading.RLock())
        result = {}

        def prepare(session):
            session.emit("status", stage="waiting_for_agent")
            while not lock.acquire(timeout=0.1):
                check_cancelled()
            session.stack.callback(lock.release)
            check_cancelled()
            return agent, {
                "user_id": identity.principal,
                "context": context or {},
                "tool_context": {
                    "user_id": identity.principal,
                    "mcp_principal": identity.principal,
                },
            }

        def finalize(session):
            saved = "not_configured"
            if getattr(agent, "memory_provider", None) is not None:
                try:
                    value = agent.save()
                    saved = "failed" if value is False else "written"
                except Exception:
                    saved = "failed"
            session.persistence["adapter_state"] = saved
            self._conversation_state[key] = (resolved_memory_id, resolved_thread_id)
            raw = getattr(agent, "last_tool_outcomes", [])
            if callable(raw):
                raw = raw()
            result.update(
                ok=session.outcome == "completed" and saved != "failed",
                status=session.outcome,
                error_code=session.error_code,
                agent_id=resolved_agent_id,
                memory_id=resolved_memory_id,
                thread_id=resolved_thread_id,
                response=session.answer,
                proposal=session.approval,
                persistence=session.persistence,
                tool_outcomes=[
                    _json_value(item) for item in (raw or []) if isinstance(item, dict)
                ],
            )

        stream = agent_event_stream(
            None,
            text,
            _prepare=prepare,
            _finalize=finalize,
            _agent_id=resolved_agent_id,
            memory_id=resolved_memory_id,
            thread_id=resolved_thread_id,
            delivery_mode=delivery_mode,
        )
        stream.result = result
        return stream

    def _bounded_limit(self, limit: int) -> int:
        return max(1, min(int(limit or 20), self.config.max_result_items))

    def _tenant_rows(
        self, rows: Any, identity: RequestIdentity, limit: int
    ) -> List[Dict[str, Any]]:
        values = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            if _document_user_id(row) != identity.principal:
                continue
            values.append(_json_value(row))
            if len(values) >= limit:
                break
        return values

    def list_memories(
        self,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        resolved_limit = self._bounded_limit(limit)
        list_all = self.provider.list_all
        kwargs = {}
        if _accepts_keyword(list_all, "user_id"):
            kwargs["user_id"] = identity.principal
        try:
            rows = list_all(resolved_type, **kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not list memories"
            ) from exc
        if memory_id:
            rows = [
                row
                for row in rows or []
                if isinstance(row, dict) and row.get("memory_id") == memory_id
            ]
        values = self._tenant_rows(rows, identity, resolved_limit)
        return {
            "ok": True,
            "memory_type": resolved_type.value,
            "memories": values,
            "count": len(values),
        }

    def search_memories(
        self,
        query: str,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        text = str(query or "").strip()
        if not text:
            raise MemorizzServerError("invalid_query", "Search query cannot be empty")
        resolved_type = self._normalize_memory_type(memory_type, identity)
        resolved_limit = self._bounded_limit(limit)
        retrieve = self.provider.retrieve_by_query
        kwargs = {
            "memory_store_type": resolved_type,
            "memory_id": memory_id,
            "limit": resolved_limit,
        }
        if _accepts_keyword(retrieve, "user_id"):
            kwargs["user_id"] = identity.principal
        try:
            rows = retrieve(text, **kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not search memories"
            ) from exc
        values = self._tenant_rows(rows, identity, resolved_limit)
        return {
            "ok": True,
            "query": text,
            "memory_type": resolved_type.value,
            "memories": values,
            "count": len(values),
        }

    def get_memory(
        self, record_id: str, memory_type: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        try:
            row = self.provider.retrieve_by_id(record_id, resolved_type)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the memory"
            ) from exc
        if not isinstance(row, dict) or _document_user_id(row) != identity.principal:
            raise MemorizzServerError("memory_not_found", "Memory was not found")
        return {"ok": True, "memory": _json_value(row)}

    def remember(
        self,
        content: str,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        text = str(content or "").strip()
        if not text:
            raise MemorizzServerError(
                "invalid_content", "Memory content cannot be empty"
            )
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "content_too_large",
                f"Memory content exceeds {self.config.max_text_chars} characters",
            )
        resolved_type = self._normalize_memory_type(
            memory_type, identity, writable=True
        )
        safe_metadata = dict(metadata or {})
        for reserved in (
            "_id",
            "id",
            "embedding",
            "user_id",
            "memory_id",
            "memory_type",
        ):
            safe_metadata.pop(reserved, None)
        resolved_memory_id = str(memory_id or uuid.uuid4())
        document = {
            **safe_metadata,
            "content": text,
            "text": text,
            "memory_id": resolved_memory_id,
            "user_id": identity.principal,
            "memory_type": resolved_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            record_id = self.provider.store(
                document,
                memory_store_type=resolved_type,
                memory_id=resolved_memory_id,
            )
        except Exception as exc:
            logger.exception("Memorizz MCP memory store failed")
            raise MemorizzServerError(
                "provider_error", "The memory provider could not store the memory"
            ) from exc
        return {
            "ok": True,
            "record_id": str(record_id),
            "memory_id": resolved_memory_id,
            "memory_type": resolved_type.value,
        }

    def forget(
        self,
        record_id: str,
        memory_type: str,
        identity: RequestIdentity,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        existing = self.get_memory(record_id, resolved_type.value, identity)
        if not existing.get("ok"):
            raise MemorizzServerError("memory_not_found", "Memory was not found")
        arguments = {
            "record_id": str(record_id),
            "memory_type": resolved_type.value,
        }
        proposal = self.approval_store.propose(
            owner_id=self._approval_owner(identity),
            tool_name="memorizz_forget_memory",
            arguments=arguments,
            policy_reason="Permanent deletion requires a human host decision",
            checkpoint={
                "version": 1,
                "operation": "forget_memory",
                "arguments": arguments,
                "principal": identity.principal,
                "scopes": sorted(identity.scopes),
                "authenticated": identity.authenticated,
            },
            ttl_seconds=self.config.approval_ttl_seconds,
        )
        return {
            "ok": False,
            "status": "approval_required",
            "proposal": proposal.to_dict(include_arguments=True),
        }

    def _forget_authorized(
        self,
        record_id: str,
        memory_type: str,
        identity: RequestIdentity,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        self.get_memory(record_id, resolved_type.value, identity)
        try:
            deleted = bool(self.provider.delete_by_id(record_id, resolved_type))
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not delete the memory"
            ) from exc
        return {
            "ok": deleted,
            "record_id": record_id,
            "memory_type": resolved_type.value,
            "deleted": deleted,
        }

    def list_approvals(
        self,
        *,
        status: Optional[str] = None,
        principal: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        owner_id = f"mcp-server:{principal}" if principal else None
        return [
            item.to_dict(include_arguments=True)
            for item in self.approval_store.list(
                owner_id=owner_id, status=status, limit=limit
            )
            if item.owner_id.startswith("mcp-server:")
        ]

    def approve_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        return self.approval_store.approve(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        ).to_dict(include_arguments=True)

    def reject_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        if (proposal.checkpoint or {}).get("operation") == "harness_run":
            return self.meta_harness.reject(
                proposal_id, approver_id=approver_id, reason=reason
            )
        return self.approval_store.reject(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        ).to_dict(include_arguments=True)

    def cancel_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a pending server proposal without executing it."""
        return self.reject_proposal(
            proposal_id,
            approver_id=approver_id,
            reason=reason or "Cancelled by host",
        )

    def resume_proposal(self, proposal_id: str) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        if proposal.status != ApprovalStatus.APPROVED:
            raise ApprovalStateError("Only an approved proposal can be resumed")
        checkpoint = dict(proposal.checkpoint or {})
        operation = checkpoint.get("operation")
        if operation == "harness_run":
            return self.meta_harness.resume_approval(proposal_id).to_dict()
        if operation not in {"forget_memory", "delete_agent"}:
            raise MemorizzServerError(
                "invalid_checkpoint", "Unsupported MCP server approval checkpoint"
            )
        arguments = dict(checkpoint.get("arguments") or {})
        self.approval_store.consume(
            proposal_id,
            expected_tool_name=(
                "memorizz_forget_memory"
                if operation == "forget_memory"
                else "memorizz_delete_agent"
            ),
            expected_arguments=arguments,
        )
        identity = RequestIdentity(
            principal=checkpoint.get("principal"),
            scopes=frozenset(checkpoint.get("scopes") or []),
            authenticated=bool(checkpoint.get("authenticated")),
        )
        if operation == "delete_agent":
            return self._delete_agent_authorized(
                str(arguments.get("agent_id") or ""),
                identity,
                cascade=bool(arguments.get("cascade", False)),
            )
        return self._forget_authorized(
            str(arguments.get("record_id") or ""),
            str(arguments.get("memory_type") or ""),
            identity,
        )

    # --- External agent harness control plane ---

    def _require_harness_execution(self, identity: RequestIdentity) -> None:
        self._require_scope(identity, EXECUTE_SCOPE)
        if not self.config.allow_harness_execution:
            raise MemorizzServerError(
                "harness_execution_disabled",
                "External harness execution is disabled by the server operator",
            )
        if self.config.transport != "stdio" and not identity.principal:
            raise MemorizzServerError(
                "authentication_required",
                "Remote harness execution requires an authenticated tenant principal",
            )

    def list_harnesses(self, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        values = self.meta_harness.list_harnesses()
        return {
            "ok": True,
            "enabled": bool(self.config.allow_harness_execution),
            "harnesses": values,
            "count": len(values),
            "ready_count": sum(1 for value in values if value.get("ready")),
            "authentication_required": [
                value.get("name")
                for value in values
                if value.get("error_code") == "authentication_required"
            ],
        }

    def _harness_run_for_identity(
        self, run_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        value = self.meta_harness.get_run(str(run_id or "").strip())
        if value is None:
            raise MemorizzServerError(
                "harness_run_not_found", "Harness run was not found"
            )
        owner = (value.get("task") or {}).get("user_id")
        if identity.principal is not None and owner != identity.principal:
            raise MemorizzServerError(
                "harness_run_not_found", "Harness run was not found"
            )
        return value

    def start_harness_run(
        self,
        task: str,
        workspace: str,
        identity: RequestIdentity,
        *,
        harness: str = "auto",
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        write: bool = False,
        allow_dirty_workspace: bool = False,
        network: str = "none",
        mcp_access: str = "read_only",
        allowed_env: Optional[List[str]] = None,
        execution_backend: str = "local",
        verification_command: Optional[str] = None,
        timeout_seconds: int = 900,
        max_steps: int = 80,
        max_cost_usd: Optional[float] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start one tenant-bound run; risky envelopes return durable approval."""
        self._require_harness_execution(identity)
        instruction = str(task or "").strip()
        if not instruction:
            raise MemorizzServerError(
                "invalid_harness_task", "Harness task cannot be empty"
            )
        if len(instruction) > self.config.max_text_chars:
            raise MemorizzServerError(
                "invalid_harness_task",
                f"Harness task exceeds {self.config.max_text_chars} characters",
            )
        if write or str(mcp_access).strip().lower() == "governed_write":
            self._require_scope(identity, WRITE_SCOPE)
        normalized_agent_id = str(agent_id or "").strip() or None
        if normalized_agent_id:
            self._assert_agent_exposed(normalized_agent_id, identity)
        env_names: List[str] = []
        for name in list(allowed_env or []):
            normalized = str(name or "").strip()
            if not normalized:
                continue
            if not normalized.replace("_", "").isalnum() or not normalized[0].isalpha():
                raise MemorizzServerError(
                    "invalid_harness_environment",
                    "Allowed environment names must contain only letters, numbers, and underscores",
                )
            if normalized not in env_names:
                env_names.append(normalized)
            if len(env_names) > 32:
                raise MemorizzServerError(
                    "invalid_harness_environment",
                    "At most 32 environment names may be delegated",
                )
        verify = str(verification_command or "").strip() or None
        if verify and len(verify) > 4_000:
            raise MemorizzServerError(
                "invalid_harness_verification",
                "Verification command cannot exceed 4000 characters",
            )

        from ..metaharness import (
            HarnessBudget,
            HarnessPermissions,
            HarnessTask,
            VerificationSpec,
        )

        try:
            request = HarnessTask(
                task=instruction,
                workspace=str(workspace or "").strip(),
                harness=harness,
                model=str(model).strip() if model else None,
                agent_id=normalized_agent_id,
                memory_id=str(memory_id).strip() if memory_id else None,
                thread_id=str(thread_id).strip() if thread_id else None,
                user_id=identity.principal,
                mode="delegate",
                permissions=HarnessPermissions(
                    workspace_mode="direct" if write else "read_only",
                    allowed_roots=sorted(self.config.harness_workspace_roots or []),
                    allow_dirty_workspace=bool(allow_dirty_workspace),
                    network=network,
                    allowed_env=env_names,
                    mcp_access=mcp_access,
                ),
                budget=HarnessBudget(
                    max_wall_time_seconds=timeout_seconds,
                    max_steps=max_steps,
                    max_cost_usd=max_cost_usd,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                ),
                verification=VerificationSpec(command=verify),
                metadata={
                    "execution_backend": str(execution_backend or "local").lower(),
                    "approval_owner_id": self._approval_owner(identity),
                    "approval_ttl_seconds": self.config.approval_ttl_seconds,
                    "source": "mcp-server",
                    "model_initiated": True,
                },
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("invalid_harness_task", str(exc)) from exc
        run = self.meta_harness.start(request)
        current = self.meta_harness.get_run(run.run_id) or run.to_dict()
        if current.get("status") == "failed":
            failure = dict(current.get("result") or {})
            return {
                "ok": False,
                "status": "failed",
                "run": current,
                "error": {
                    "code": failure.get("error_code") or "harness_failed",
                    "message": failure.get("error") or "Harness startup failed",
                    "remediation": failure.get("remediation"),
                    "run_id": run.run_id,
                },
            }
        result: Dict[str, Any] = {"ok": True, "run": current}
        if run.approval_proposal_id:
            proposal = self.approval_store.get(run.approval_proposal_id)
            result.update(
                ok=False,
                status="approval_required",
                proposal=(
                    proposal.to_dict(include_arguments=True) if proposal else None
                ),
            )
        return result

    def get_harness_run(
        self, run_id: str, identity: RequestIdentity, *, include_events: bool = False
    ) -> Dict[str, Any]:
        value = self._harness_run_for_identity(run_id, identity)
        if include_events:
            value["events"] = self.meta_harness.events(
                run_id, limit=self.config.max_result_items
            )
        return {"ok": True, "run": value}

    def list_harness_runs(
        self,
        identity: RequestIdentity,
        *,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        rows = self.meta_harness.list_runs(
            status=status, limit=self._bounded_limit(limit)
        )
        if identity.principal is not None:
            rows = [
                row
                for row in rows
                if (row.get("task") or {}).get("user_id") == identity.principal
            ]
        return {"ok": True, "runs": rows, "count": len(rows)}

    def get_harness_events(
        self,
        run_id: str,
        identity: RequestIdentity,
        *,
        after: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        self._harness_run_for_identity(run_id, identity)
        events = self.meta_harness.events(
            run_id,
            after=max(0, int(after)),
            limit=self._bounded_limit(limit),
        )
        return {"ok": True, "run_id": run_id, "events": events, "count": len(events)}

    def cancel_harness_run(
        self, run_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_harness_execution(identity)
        self._harness_run_for_identity(run_id, identity)
        return self.meta_harness.cancel(run_id)

    def list_conversations(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        normalized = str(agent_id or "").strip()
        self._assert_agent_exposed(normalized, identity)
        try:
            agent = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the agent"
            ) from exc
        if not agent:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        memory_ids = (
            agent.get("memory_ids")
            if isinstance(agent, dict)
            else getattr(agent, "memory_ids", None)
        )
        from ..cli.conversations import discover_conversations

        conversations = discover_conversations(
            self.provider,
            memory_ids or [],
            user_id=identity.principal,
            agent_id=normalized,
        )[: self._bounded_limit(limit)]
        values = [
            {
                "memory_id": item.memory_id,
                "thread_id": item.thread_id,
                "title": item.title,
                "preview": item.preview,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                "message_count": item.message_count,
            }
            for item in conversations
        ]
        return {"ok": True, "conversations": values, "count": len(values)}

    def get_conversation(
        self,
        memory_id: str,
        thread_id: str,
        identity: RequestIdentity,
        *,
        limit: int = 100,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        retrieve = self.provider.retrieve_conversation_history_ordered_by_timestamp
        kwargs = {
            "memory_id": memory_id,
            "memory_type": MemoryType.CONVERSATION_MEMORY,
            "limit": self._bounded_limit(limit),
        }
        if _accepts_keyword(retrieve, "user_id"):
            kwargs["user_id"] = identity.principal
        if _accepts_keyword(retrieve, "thread_id"):
            kwargs["thread_id"] = thread_id
        try:
            rows = retrieve(**kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the conversation"
            ) from exc
        rows = [
            row
            for row in rows or []
            if isinstance(row, dict)
            and _document_field(row, "memory_id") == memory_id
            and _document_field(row, "thread_id") == thread_id
        ]
        values = self._tenant_rows(rows, identity, self._bounded_limit(limit))
        return {
            "ok": True,
            "memory_id": memory_id,
            "thread_id": thread_id,
            "messages": values,
            "count": len(values),
        }
