# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tenant-safe adapter from MCP tools to Memorizz agents and providers."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
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
from ..metaharness.config import DEFAULT_HARNESS_CHOICES
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
        try:
            dumped = value.model_dump(mode="json", exclude_none=True)
        except Exception:
            # A field holds an object Pydantic cannot serialise (an agent's
            # Persona, say); dump it as Python and convert the parts here.
            dumped = value.model_dump(exclude_none=True)
        return _json_value(dumped)
    if callable(getattr(value, "to_dict", None)):
        try:
            return _json_value(value.to_dict())
        except Exception:
            return str(value)
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


# memorizz_ingest limits: a model can't ask for the whole disk at once.
_INGEST_MAX_FILES = 500
_INGEST_MAX_BYTES = 20 * 1024 * 1024
_INGEST_MAX_PATHS = 50
_INGEST_SKIP_DIRS = frozenset(
    {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__"}
)
_INGEST_SECRET_PATTERNS = (
    ".env*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "credentials*",
    "secrets*",
)
_EMBEDDING_STATUS_TTL_SECONDS = 300.0
_EMBEDDING_PROBE_TIMEOUT_SECONDS = 4.0
_EMBEDDINGS_HINT = (
    "To turn semantic search on, run `memorizz config set "
    "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER ollama` and `memorizz config set "
    "MEMORIZZ_DEFAULT_EMBEDDING_MODEL nomic-embed-text` (or set OPENAI_API_KEY)."
)
_NO_EMBEDDINGS_NOTE = (
    "No embedding model is working, so these are keyword matches. " + _EMBEDDINGS_HINT
)
# How far a tenant-scoped harness listing may widen its store query while
# looking for the caller's rows (the SQLite store caps listings at 10,000).
_TENANT_SCAN_LIMIT = 10_000
_UPDATE_DROPPED_FIELDS = frozenset(
    {
        "_id",
        "id",
        "embedding",
        "content",
        "text",
        "timestamp",
        "created_at",
        "updated_at",
        "status",
        "superseded_by",
        "superseded_at",
        "supersedes",
        "supersede_reason",
        "score",
        "has_embedding",
    }
)
_STOPWORDS = frozenset(
    "a an and are as at be but by do does for from how i in is it of on or our "
    "that the this to was we what when where which who why with you your".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _is_superseded(document: Any) -> bool:
    """Replaced by memorizz_update_memory (top-level fields) or by entity
    consolidation (``metadata.superseded_by``)."""
    if not isinstance(document, dict):
        return False
    if str(document.get("status") or "").strip().lower() == "superseded":
        return True
    if str(document.get("superseded_by") or "").strip():
        return True
    metadata = document.get("metadata")
    return isinstance(metadata, dict) and bool(
        str(metadata.get("superseded_by") or "").strip()
    )


def _document_text(document: Dict[str, Any]) -> str:
    content = document.get("content")
    if isinstance(content, dict):
        content = content.get("content") or content.get("text")
    text = str(content or document.get("text") or "")
    if not text and document.get("name"):
        facts = " ".join(
            f"{item.get('name')} {item.get('value')}"
            for item in document.get("attributes") or []
            if isinstance(item, dict)
        )
        text = f"{document.get('name')} {document.get('entity_type') or ''} {facts}"
    return text


def _stem(word: str) -> str:
    for suffix in ("ments", "ment", "ings", "ing", "ies", "es", "ed", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _keywords(text: str) -> set:
    return {
        _stem(word)
        for word in _WORD.findall(str(text or "").lower())
        if len(word) > 1 and word not in _STOPWORDS
    }


def _ingest_skip_reason(path: Path, *, is_dir: bool) -> Optional[str]:
    name = path.name
    lowered = name.lower()
    if is_dir:
        if lowered in _INGEST_SKIP_DIRS:
            return "dependency or version-control folder"
        return "hidden folder" if name.startswith(".") else None
    if any(fnmatch.fnmatch(lowered, pattern) for pattern in _INGEST_SECRET_PATTERNS):
        return "looks like a secret"
    return "hidden file" if name.startswith(".") else None


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
        self._embedding_lock = threading.Lock()
        self._embedding_cache: Optional[Tuple[float, Dict[str, Any]]] = None

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
            from ..memory_history import enable_memory_history

            enable_memory_history(self._provider)
            return self._provider

    def memory_timeline(
        self,
        identity: RequestIdentity,
        *,
        agent_id=None,
        memory_id=None,
        run_id=None,
        memory_type=None,
        action=None,
        actor=None,
        limit=200,
        cursor=None,
    ):
        """Read content-free change history with the caller's exact tenant scope."""
        from ..memory_history import MemoryHistory

        self._require_scope(identity, READ_SCOPE)
        if agent_id:
            self._assert_agent_exposed(agent_id, identity)
        if run_id:
            self._harness_run_for_identity(run_id, identity)
        if memory_type:
            self._normalize_memory_type(memory_type, identity)
        page = MemoryHistory(self.provider).timeline(
            agent_id=agent_id,
            memory_id=memory_id,
            run_id=run_id,
            user_id=identity.principal,
            memory_type=memory_type,
            action=action,
            actor=actor,
            limit=self._bounded_limit(limit),
            cursor=cursor,
        )
        # A configured agent allowlist also applies when no agent was requested.
        page["events"] = [
            e
            for e in page["events"]
            if not e.get("agent_id") or self._is_agent_exposed(e["agent_id"], identity)
        ]
        return {"ok": True, **page}

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
                "memory": [
                    "list",
                    "search",
                    "read",
                    "store",
                    "update",
                    "forget",
                    "ingest",
                    "entities",
                    "status",
                    "export",
                    "import",
                ],
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
        if normalized_default_harness not in DEFAULT_HARNESS_CHOICES:
            raise MemorizzServerError(
                "invalid_agent",
                "default_harness must be auto, codex, claude-code, openhands, deepseek, pi, hermes, or native",
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
            if normalized_default_harness not in DEFAULT_HARNESS_CHOICES:
                raise MemorizzServerError(
                    "invalid_agent",
                    "default_harness must be auto, codex, claude-code, openhands, deepseek, pi, hermes, or native",
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

    def export_memories(
        self,
        identity: RequestIdentity,
        *,
        memory_id=None,
        agent_id=None,
        memory_types=None,
        include_delegates=True,
        include_history=True,
        include_context=True,
    ):
        """Inline portable archive; large/local backups use the CLI file interface."""
        from ..memory_archive import MemoryArchive, MemoryArchiveError, seal_archive

        self._require_scope(identity, READ_SCOPE)
        if agent_id:
            self._assert_agent_exposed(agent_id, identity)
        types = (
            memory_types
            if memory_types is not None
            else [
                t.value
                for t in MemoryType
                if self.config.transport == "stdio" or t in _TENANT_MEMORY_TYPES
            ]
        )
        for kind in types:
            self._normalize_memory_type(kind, identity)
        try:
            archive = MemoryArchive(self.provider).export(
                memory_id=memory_id,
                agent_id=agent_id,
                memory_types=types,
                include_delegates=include_delegates,
                include_history=include_history,
                include_context=include_context,
                user_id=identity.principal,
                max_bytes=self.config.max_request_body_size,
            )
            for records in archive["stores"].values():
                records[:] = [
                    r
                    for r in records
                    if not _document_field(r["data"], "agent_id")
                    or self._is_agent_exposed(
                        _document_field(r["data"], "agent_id"), identity
                    )
                ]
            archive["manifest"]["scope"]["agent_ids"] = [
                identifier
                for identifier in archive["manifest"]["scope"]["agent_ids"]
                if self._is_agent_exposed(identifier, identity)
            ]
            archive["manifest"]["counts"] = {
                k: len(v) for k, v in archive["stores"].items()
            }
            archive["manifest"]["record_count"] = sum(
                archive["manifest"]["counts"].values()
            )
            return {"ok": True, "archive": seal_archive(archive)}
        except MemoryArchiveError as exc:
            raise MemorizzServerError("archive_error", str(exc)) from exc

    def import_memories(
        self,
        archive,
        identity: RequestIdentity,
        *,
        dry_run=True,
        conflict="error",
        id_strategy="preserve",
        target_memory_id=None,
    ):
        from ..memory_archive import MemoryArchive, MemoryArchiveError, validate_archive

        self._require_scope(identity, READ_SCOPE if dry_run else WRITE_SCOPE)
        try:
            archive = validate_archive(
                archive, max_bytes=self.config.max_request_body_size
            )
            for records in archive["stores"].values():
                for record in records:
                    owner = _document_field(record["data"], "agent_id")
                    if owner:
                        self._assert_agent_exposed(owner, identity)
                        if (
                            id_strategy == "new"
                            and self.config.exposed_agent_ids is not None
                        ):
                            raise MemoryArchiveError(
                                "New agent identities require an unrestricted agent allowlist"
                            )
            types = (
                list(MemoryType)
                if self.config.transport == "stdio"
                else _TENANT_MEMORY_TYPES
            )
            return MemoryArchive(self.provider).import_archive(
                archive,
                dry_run=dry_run,
                conflict=conflict,
                id_strategy=id_strategy,
                target_memory_id=target_memory_id,
                user_id=identity.principal,
                allowed_types=types,
            )
        except MemoryArchiveError as exc:
            raise MemorizzServerError("archive_error", str(exc)) from exc

    def list_memories(
        self,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        limit: int = 20,
        include_superseded: bool = False,
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
        if not include_superseded:
            rows = [row for row in rows or [] if not _is_superseded(row)]
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
        include_superseded: bool = False,
    ) -> Dict[str, Any]:
        """Semantic search, falling back to keyword matches when no embedding
        model works or semantic search finds nothing; ``search_mode`` says
        which (``semantic``, ``keyword``, or ``hybrid`` when keyword matches
        fill out a short semantic list)."""
        self._require_scope(identity, READ_SCOPE)
        text = str(query or "").strip()
        if not text:
            raise MemorizzServerError("invalid_query", "Search query cannot be empty")
        resolved_type = self._normalize_memory_type(memory_type, identity)
        resolved_limit = self._bounded_limit(limit)
        embedder = self._embedding_status()
        values: List[Dict[str, Any]] = []
        note: Optional[str] = None
        semantic_failed = False
        if embedder.get("ready"):
            retrieve = self.provider.retrieve_by_query
            kwargs = {
                "memory_store_type": resolved_type,
                "memory_id": memory_id,
                # Room for rows the superseded filter drops.
                "limit": min(self.config.max_result_items, resolved_limit * 3),
            }
            if _accepts_keyword(retrieve, "user_id"):
                kwargs["user_id"] = identity.principal
            try:
                rows = retrieve(text, **kwargs)
            except Exception as exc:
                semantic_failed = True
                logger.warning(
                    "Semantic memory search failed; using keyword matches (%s)",
                    type(exc).__name__,
                )
            else:
                if not include_superseded:
                    rows = [row for row in rows or [] if not _is_superseded(row)]
                values = self._tenant_rows(rows, identity, resolved_limit)
        mode = "semantic"
        if not embedder.get("ready") or semantic_failed:
            values = self._keyword_matches(
                text,
                resolved_type,
                identity,
                memory_id=memory_id,
                limit=resolved_limit,
                include_superseded=include_superseded,
            )
            mode = "keyword"
            note = (
                _NO_EMBEDDINGS_NOTE
                if not embedder.get("ready")
                else "Semantic search failed, so these are keyword matches."
            )
        elif len(values) < resolved_limit:
            # Rows stored before embeddings were on have no vector, so semantic
            # search can't see them; keyword matches fill the gap.
            seen = {str(item.get("_id") or item.get("id")) for item in values}
            extra = [
                item
                for item in self._keyword_matches(
                    text,
                    resolved_type,
                    identity,
                    memory_id=memory_id,
                    limit=resolved_limit,
                    include_superseded=include_superseded,
                )
                if str(item.get("_id") or item.get("id")) not in seen
            ]
            if extra:
                mode = "hybrid" if values else "keyword"
                if not values:
                    note = "No semantic matches, so these are keyword matches."
                values = (values + extra)[:resolved_limit]
        result = {
            "ok": True,
            "query": text,
            "memory_type": resolved_type.value,
            "search_mode": mode,
            "memories": values,
            "count": len(values),
        }
        if note:
            result["note"] = note
        return result

    def _scoped_rows(
        self,
        memory_type: MemoryType,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        list_all = self.provider.list_all
        kwargs = {}
        if _accepts_keyword(list_all, "user_id"):
            kwargs["user_id"] = identity.principal
        return [
            row
            for row in list_all(memory_type, **kwargs) or []
            if isinstance(row, dict)
            and (not memory_id or row.get("memory_id") == memory_id)
            and _document_user_id(row) == identity.principal
        ]

    def _keyword_matches(
        self,
        query: str,
        memory_type: MemoryType,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str],
        limit: int,
        include_superseded: bool = False,
    ) -> List[Dict[str, Any]]:
        """Rows sharing words with the query, best overlap first, newest
        first among equals."""
        wanted = _keywords(query)
        if not wanted:
            return []
        try:
            rows = self._scoped_rows(memory_type, identity, memory_id=memory_id)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not search memories"
            ) from exc
        scored = []
        for row in rows:
            if not include_superseded and _is_superseded(row):
                continue
            overlap = len(wanted & _keywords(_document_text(row)))
            if overlap:
                scored.append(
                    (
                        overlap / len(wanted),
                        str(row.get("timestamp") or row.get("created_at") or ""),
                        row,
                    )
                )
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        values = []
        for score, _when, row in scored[:limit]:
            value = _json_value(row)
            value["score"] = round(score, 3)
            values.append(value)
        return values

    def _probe_embeddings(self) -> Dict[str, Any]:
        from ..memory_provider.base import provider_manages_embeddings
        from ..metaharness.security import redact as redact_text

        status: Dict[str, Any] = {
            "ready": False,
            "provider": None,
            "model": None,
            "managed_by_provider": False,
            "error": None,
        }
        try:
            managed = provider_manages_embeddings(self.provider)
        except Exception:
            managed = False
        if managed:
            status.update(ready=True, managed_by_provider=True)
            return status
        status["provider"] = (
            os.getenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "").strip().lower()
            or "openai"
        )
        try:
            from ..embeddings import get_embedding_manager

            manager = get_embedding_manager()
            info = manager.get_provider_info()
            status["provider"] = info.get("provider")
            status["model"] = info.get("model")
        except Exception as exc:
            status["error"] = redact_text(f"{type(exc).__name__}: {exc}")[:300]
            return status
        outcome: Dict[str, Any] = {}

        def probe() -> None:
            try:
                vector = manager.get_embedding("memorizz embedding check")
                outcome["ready"] = bool(vector)
            except Exception as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(
            target=probe, name="memorizz-embedding-probe", daemon=True
        )
        worker.start()
        worker.join(_EMBEDDING_PROBE_TIMEOUT_SECONDS)
        if worker.is_alive():
            status[
                "error"
            ] = f"The embedding model didn't answer within {_EMBEDDING_PROBE_TIMEOUT_SECONDS:.0f} s"
        elif outcome.get("ready"):
            status["ready"] = True
        else:
            status["error"] = redact_text(
                str(outcome.get("error") or "No embedding returned")
            )[:300]
        return status

    def _embedding_status(self, *, refresh: bool = False) -> Dict[str, Any]:
        """Whether an embedding model works, checked at most every few
        minutes (one tiny embedding call)."""
        with self._embedding_lock:
            cached = self._embedding_cache
            if (
                cached
                and not refresh
                and time.monotonic() - cached[0] < _EMBEDDING_STATUS_TTL_SECONDS
            ):
                return dict(cached[1])
        status = self._probe_embeddings()
        with self._embedding_lock:
            self._embedding_cache = (time.monotonic(), dict(status))
        return status

    def _embed_or_none(self, text: str) -> Optional[List[float]]:
        """An embedding for a record being written, when a model works and
        the provider doesn't embed on its own; never fails the write."""
        status = self._embedding_status()
        if not status.get("ready") or status.get("managed_by_provider"):
            return None
        try:
            from ..embeddings import get_embedding

            vector = get_embedding(text)
            return list(vector) if vector else None
        except Exception as exc:
            logger.warning(
                "Storing memory without an embedding (%s)", type(exc).__name__
            )
            return None

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
        embedding = self._embed_or_none(text)
        if embedding:
            document["embedding"] = embedding
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

    # ------------------------------------------- update, ingest, entities

    def _episodic_scope(self, memory_id: Any, thread_id: Any) -> tuple:
        memory = str(memory_id or "").strip()
        thread = str(thread_id or "").strip()
        if not memory or not thread or len(memory) > 200 or len(thread) > 200:
            raise MemorizzServerError(
                "invalid_conversation_scope",
                "memory_id and thread_id are required (at most 200 characters)",
            )
        return memory, thread

    def _capture_agent_id(
        self, agent_id: Any, identity: RequestIdentity
    ) -> Optional[str]:
        """The agent a captured turn or summary is filed under: optional, but
        when given it must be one this server exposes to the caller."""
        normalized = str(agent_id or "").strip()
        if not normalized:
            return None
        self._assert_agent_exposed(normalized, identity)
        return normalized[:80]

    def record_turn(
        self,
        memory_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        identity: RequestIdentity,
        *,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save one turn of a coding session (request and answer, secrets
        removed) to conversation memory for this caller."""
        from ..episodic_capture import record_turn

        self._require_scope(identity, WRITE_SCOPE)
        memory, thread = self._episodic_scope(memory_id, thread_id)
        for text in (user_message, assistant_message):
            if len(str(text or "")) > self.config.max_text_chars:
                raise MemorizzServerError(
                    "content_too_large",
                    f"A message exceeds {self.config.max_text_chars} characters",
                )
        if (
            not str(user_message or "").strip()
            and not str(assistant_message or "").strip()
        ):
            raise MemorizzServerError("invalid_content", "The turn has no text")
        resolved_agent_id = self._capture_agent_id(agent_id, identity)
        try:
            record_ids = record_turn(
                self.provider,
                memory_id=memory,
                thread_id=thread,
                user_message=user_message,
                assistant_message=assistant_message,
                agent_id=resolved_agent_id,
                user_id=identity.principal,
            )
        except Exception as exc:
            logger.exception("Memorizz MCP turn capture failed")
            raise MemorizzServerError(
                "provider_error", "The memory provider could not store the turn"
            ) from exc
        return {
            "ok": True,
            "memory_id": memory,
            "thread_id": thread,
            "record_ids": record_ids,
        }

    def summarize_session(
        self,
        memory_id: str,
        thread_id: str,
        identity: RequestIdentity,
        *,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Summarize a coding session's recorded turns with the server's
        default model into the summaries store."""
        from ..episodic_capture import default_model, summarize_thread

        self._require_scope(identity, WRITE_SCOPE)
        memory, thread = self._episodic_scope(memory_id, thread_id)
        resolved_agent_id = self._capture_agent_id(agent_id, identity)
        model = default_model()
        if model is None:
            return {
                "ok": False,
                "summary_ids": [],
                "reason": "no_model",
                "note": "The server has no default model (MEMORIZZ_DEFAULT_LLM_PROVIDER / _MODEL).",
            }
        try:
            summary_ids = summarize_thread(
                self.provider,
                memory_id=memory,
                thread_id=thread,
                agent_id=resolved_agent_id,
                user_id=identity.principal,
                model=model,
            )
        except Exception as exc:
            logger.exception("Memorizz MCP session summary failed")
            raise MemorizzServerError(
                "summary_failed", "The session could not be summarized"
            ) from exc
        return {
            "ok": bool(summary_ids),
            "memory_id": memory,
            "thread_id": thread,
            "summary_ids": summary_ids,
        }

    def _mark_superseded(
        self,
        old_id: str,
        new_id: str,
        when: str,
        memory_type: MemoryType,
        reason: Optional[str] = None,
    ) -> bool:
        """Mark a record superseded by another, and check the mark was kept
        (some backends keep only fixed columns)."""
        fields = {
            "status": "superseded",
            "superseded_by": new_id,
            "superseded_at": when,
        }
        if reason:
            fields["supersede_reason"] = reason
        try:
            return bool(
                self.provider.update_by_id(old_id, fields, memory_type)
            ) and _is_superseded(self.provider.retrieve_by_id(old_id, memory_type))
        except Exception:
            logger.exception("Memorizz MCP could not mark a memory superseded")
            return False

    def update_memory(
        self,
        record_id: str,
        content: str,
        identity: RequestIdentity,
        *,
        memory_type: str = "knowledge_base",
        reason: Optional[str] = None,
        duplicate_of: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Replace a memory with a new record and mark the old one superseded,
        or (with ``duplicate_of``) mark it superseded by an existing memory
        that already says it.

        Nothing is deleted: the old record keeps its content and gains
        ``status="superseded"``, ``superseded_by`` and ``superseded_at``; the
        new record carries ``supersedes`` and ``supersede_reason``. List and
        search hide superseded records unless asked for them.
        """
        self._require_scope(identity, WRITE_SCOPE)
        text = str(content or "").strip()
        keep_id = str(duplicate_of or "").strip()
        if bool(text) == bool(keep_id):
            raise MemorizzServerError(
                "invalid_update",
                "Give the new content, or duplicate_of (an existing memory that "
                "already says it), not both",
            )
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "content_too_large",
                f"Memory content exceeds {self.config.max_text_chars} characters",
            )
        why = str(reason or "").strip()[:1000] or None
        resolved_type = self._normalize_memory_type(
            memory_type, identity, writable=True
        )
        old = self.get_memory(record_id, resolved_type.value, identity)["memory"]
        if _is_superseded(old):
            newer = old.get("superseded_by") or (old.get("metadata") or {}).get(
                "superseded_by"
            )
            raise MemorizzServerError(
                "memory_superseded",
                f"Memory {record_id} was already replaced by {newer}; update that one",
            )
        old_id = str(old.get("_id") or old.get("id") or record_id)
        now = datetime.now(timezone.utc).isoformat()
        if keep_id:
            kept = self.get_memory(keep_id, resolved_type.value, identity)["memory"]
            kept_id = str(kept.get("_id") or kept.get("id") or keep_id)
            if kept_id == old_id or kept.get("memory_id") != old.get("memory_id"):
                raise MemorizzServerError(
                    "invalid_update",
                    "duplicate_of must be another memory in the same memory_id",
                )
            if _is_superseded(kept):
                raise MemorizzServerError(
                    "memory_superseded",
                    f"Memory {kept_id} was itself replaced; use the newer one",
                )
            if not self._mark_superseded(old_id, kept_id, now, resolved_type, why):
                raise MemorizzServerError(
                    "supersede_unsupported",
                    "This memory backend could not mark the memory superseded, so "
                    "nothing changed; forget the duplicate instead",
                )
            return {
                "ok": True,
                "record_id": kept_id,
                "memory_id": old.get("memory_id"),
                "memory_type": resolved_type.value,
                "supersede_reason": why,
                "superseded": {
                    "record_id": old_id,
                    "status": "superseded",
                    "superseded_by": kept_id,
                    "superseded_at": now,
                },
            }
        # ``old`` is the caller's view: secrets redacted and values stringified
        # for JSON. The replacement is built from the stored record itself so
        # the update never persists "***" or loses native types.
        try:
            raw = self.provider.retrieve_by_id(old_id, resolved_type)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the memory"
            ) from exc
        if not isinstance(raw, dict) or _document_user_id(raw) != identity.principal:
            raise MemorizzServerError("memory_not_found", "Memory was not found")
        document = {
            key: value
            for key, value in raw.items()
            if key not in _UPDATE_DROPPED_FIELDS
        }
        memory_id = str(raw.get("memory_id") or uuid.uuid4())
        document.update(
            {
                "content": text,
                "text": text,
                "memory_id": memory_id,
                "user_id": identity.principal,
                "memory_type": resolved_type.value,
                "timestamp": now,
                "supersedes": old_id,
            }
        )
        if why:
            document["supersede_reason"] = why
        embedding = self._embed_or_none(text)
        if embedding:
            document["embedding"] = embedding
        try:
            new_id = str(
                self.provider.store(
                    document, memory_store_type=resolved_type, memory_id=memory_id
                )
            )
        except Exception as exc:
            logger.exception("Memorizz MCP memory update failed")
            raise MemorizzServerError(
                "provider_error", "The memory provider could not store the new memory"
            ) from exc
        if not self._mark_superseded(old_id, new_id, now, resolved_type):
            try:
                self.provider.delete_by_id(new_id, resolved_type)
            except Exception:
                logger.exception("Memorizz MCP could not roll back memory %s", new_id)
            raise MemorizzServerError(
                "supersede_unsupported",
                "This memory backend could not mark the old memory superseded, so "
                "nothing changed; store a new memory and forget the old one instead",
            )
        return {
            "ok": True,
            "record_id": new_id,
            "memory_id": memory_id,
            "memory_type": resolved_type.value,
            "supersedes": old_id,
            "supersede_reason": why,
            "superseded": {
                "record_id": old_id,
                "status": "superseded",
                "superseded_by": new_id,
                "superseded_at": now,
            },
        }

    def _ingest_roots(self) -> List[Path]:
        if self.config.ingest_roots is not None:
            return [Path(root) for root in sorted(self.config.ingest_roots)]
        if self.config.transport == "stdio":
            return [Path.home().resolve()]
        return []

    def ingest(
        self,
        paths: List[str],
        memory_id: str,
        identity: RequestIdentity,
        *,
        recursive: bool = True,
        include: Optional[List[str]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Chunk files and folders into the knowledge base under ``memory_id``.

        Reads only inside the allowed folders, never hidden, dependency or
        version-control folders or secret-looking files, and refuses a batch
        over the file or size cap before storing anything. A file ingested
        before is skipped when unchanged; when changed, its old chunks are
        marked superseded.
        """
        from .._env_io import memorizz_home
        from ..long_term.semantic.extractors import ExtractorError, supported_extensions
        from ..long_term.semantic.knowledge_base import KnowledgeBase

        self._require_scope(identity, WRITE_SCOPE)
        resolved_type = self._normalize_memory_type("knowledge_base", identity)
        memory_id = str(memory_id or "").strip()
        if not memory_id or len(memory_id) > 200:
            raise MemorizzServerError(
                "invalid_memory_id", "Give a memory_id (up to 200 characters)"
            )
        requested = [str(item).strip() for item in (paths or []) if str(item).strip()]
        if not requested:
            raise MemorizzServerError(
                "invalid_paths", "Give at least one file or folder path"
            )
        if len(requested) > _INGEST_MAX_PATHS:
            raise MemorizzServerError(
                "invalid_paths", f"Give at most {_INGEST_MAX_PATHS} paths per call"
            )
        patterns = [str(item).strip() for item in (include or []) if str(item).strip()]
        size = int(chunk_size) if chunk_size else None
        overlap = int(chunk_overlap) if chunk_overlap is not None else None
        if size is not None and not 100 <= size <= 20_000:
            raise MemorizzServerError(
                "invalid_chunking", "chunk_size must be 100 to 20000 characters"
            )
        if overlap is not None and not 0 <= overlap < (size or 1000):
            raise MemorizzServerError(
                "invalid_chunking",
                "chunk_overlap must be at least 0 and less than chunk_size",
            )

        roots = self._ingest_roots()
        if not roots:
            raise MemorizzServerError(
                "ingest_disabled",
                "This server has no ingest folders; set MEMORIZZ_MCP_SERVER_INGEST_ROOTS "
                "or pass --ingest-root",
            )
        extensions = {ext.lower() for ext in supported_extensions()}
        own_data = memorizz_home().expanduser().resolve()
        skipped: List[Dict[str, str]] = []
        planned: Dict[str, Path] = {}

        def root_of(path: Path) -> Optional[Path]:
            for root in roots:
                if path == root or root in path.parents:
                    return root
            return None

        def blocked(path: Path, root: Path, *, is_dir: bool) -> Optional[str]:
            if path == own_data or own_data in path.parents:
                return "MemoRizz's own data folder"
            parts = path.relative_to(root).parts
            for index, part in enumerate(parts):
                reason = _ingest_skip_reason(
                    Path(part), is_dir=is_dir or index < len(parts) - 1
                )
                if reason:
                    return reason
            return None

        def consider(path: Path, top: Path) -> None:
            resolved = path.resolve()
            root = root_of(resolved)
            if root is None:
                skipped.append(
                    {"path": str(path), "reason": "links outside the allowed folders"}
                )
                return
            reason = blocked(resolved, root, is_dir=False)
            if not reason and resolved.suffix.lower() not in extensions:
                reason = "unsupported file type"
            if not reason and patterns:
                relative = (
                    path.relative_to(top).as_posix() if top != path else path.name
                )
                if not any(
                    fnmatch.fnmatch(path.name, pattern)
                    or fnmatch.fnmatch(relative, pattern)
                    for pattern in patterns
                ):
                    reason = "not matched by include"
            if reason:
                skipped.append({"path": str(path), "reason": reason})
            else:
                planned.setdefault(str(resolved), resolved)

        for raw in requested:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                raise MemorizzServerError(
                    "invalid_paths", f"Use an absolute path: {raw}"
                )
            if not candidate.exists():
                skipped.append({"path": raw, "reason": "not found"})
                continue
            resolved = candidate.resolve()
            root = root_of(resolved)
            if root is None:
                allowed = ", ".join(str(item) for item in roots)
                raise MemorizzServerError(
                    "path_not_allowed",
                    f"{raw} is outside the folders this server may ingest ({allowed}); "
                    "the operator sets them with MEMORIZZ_MCP_SERVER_INGEST_ROOTS",
                )
            if resolved.is_file():
                consider(candidate, candidate)
                continue
            reason = blocked(resolved, root, is_dir=True)
            if reason:
                skipped.append({"path": raw, "reason": reason})
                continue
            for folder, dirnames, filenames in os.walk(resolved, followlinks=False):
                kept = []
                for name in sorted(dirnames):
                    reason = _ingest_skip_reason(Path(name), is_dir=True)
                    if reason:
                        skipped.append(
                            {"path": os.path.join(folder, name), "reason": reason}
                        )
                    else:
                        kept.append(name)
                dirnames[:] = kept if recursive else []
                for name in sorted(filenames):
                    consider(Path(folder) / name, resolved)
                if len(planned) > _INGEST_MAX_FILES:
                    break

        files = list(planned.values())
        total_bytes = sum(path.stat().st_size for path in files)
        if len(files) > _INGEST_MAX_FILES or total_bytes > _INGEST_MAX_BYTES:
            raise MemorizzServerError(
                "ingest_too_large",
                f"That is {'over ' if len(files) > _INGEST_MAX_FILES else ''}{len(files)} files "
                f"and {total_bytes / 1_048_576:.1f} MB; one call takes at most "
                f"{_INGEST_MAX_FILES} files and {_INGEST_MAX_BYTES // 1_048_576} MB. "
                "Narrow the paths or pass include patterns, then call again",
            )

        try:
            existing = self._scoped_rows(resolved_type, identity, memory_id=memory_id)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error",
                "The memory provider could not list the knowledge base",
            ) from exc
        previous: Dict[str, List[Dict[str, Any]]] = {}
        for row in existing:
            if row.get("source_path") and not _is_superseded(row):
                previous.setdefault(str(row["source_path"]), []).append(row)

        embedder = self._embedding_status()
        mode = "optional" if embedder.get("ready") else "off"
        knowledge = KnowledgeBase(self.provider)
        ingested: List[Dict[str, Any]] = []
        chunks_stored = embedded = replaced = 0
        for path in files:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                skipped.append(
                    {"path": str(path), "reason": f"unreadable ({type(exc).__name__})"}
                )
                continue
            earlier = previous.get(str(path), [])
            if any(row.get("source_sha256") == digest for row in earlier):
                skipped.append(
                    {"path": str(path), "reason": "unchanged since the last ingest"}
                )
                continue
            options: Dict[str, Any] = {}
            if size:
                options["chunk_size"] = size
            if overlap is not None:
                options["chunk_overlap"] = overlap
            try:
                knowledge_base_id = knowledge.ingest_file(
                    path,
                    namespace=memory_id,
                    user_id=identity.principal,
                    metadata={
                        "memory_id": memory_id,
                        "memory_type": resolved_type.value,
                        "source_path": str(path),
                        "source_name": path.name,
                        "source_sha256": digest,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    embeddings=mode,
                    **options,
                )
            except ExtractorError as exc:
                skipped.append(
                    {"path": str(path), "reason": f"couldn't read: {exc}"[:300]}
                )
                continue
            except Exception as exc:
                logger.exception("Memorizz MCP ingest failed for one file")
                skipped.append(
                    {"path": str(path), "reason": f"failed ({type(exc).__name__})"}
                )
                continue
            stats = dict(knowledge.last_ingest or {})
            count = int(stats.get("chunks") or 0)
            if not count:
                skipped.append({"path": str(path), "reason": "no text"})
                continue
            chunks_stored += count
            embedded += int(stats.get("embedded") or 0)
            for row in earlier:
                row_id = row.get("_id") or row.get("id")
                try:
                    if row_id and self.provider.update_by_id(
                        str(row_id),
                        {
                            "status": "superseded",
                            "superseded_by": knowledge_base_id,
                            "superseded_at": datetime.now(timezone.utc).isoformat(),
                        },
                        resolved_type,
                    ):
                        replaced += 1
                except Exception:
                    logger.exception("Memorizz MCP could not supersede an old chunk")
            ingested.append(
                {
                    "path": str(path),
                    "chunks": count,
                    "knowledge_base_id": knowledge_base_id,
                }
            )

        reasons: Dict[str, int] = {}
        for item in skipped:
            key = item["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        result = {
            "ok": True,
            "memory_id": memory_id,
            "memory_type": resolved_type.value,
            "files_ingested": len(ingested),
            "chunks_stored": chunks_stored,
            "chunks_embedded": embedded,
            "old_chunks_superseded": replaced,
            "files_skipped": len(skipped),
            "skipped_by_reason": reasons,
            "files": ingested[:100],
            "skipped": skipped[:100],
        }
        if chunks_stored and embedded < chunks_stored:
            result["note"] = (
                f"{chunks_stored - embedded} chunks were stored without embeddings: "
                "keyword search finds them, semantic search doesn't. "
                + _EMBEDDINGS_HINT
            )
        return result

    def _entity_memory(self):
        from ..long_term.semantic.entity_memory import EntityMemory

        return EntityMemory(self.provider)

    @staticmethod
    def _entity_public(record: Dict[str, Any]) -> Dict[str, Any]:
        return _json_value(record)

    def lookup_entities(
        self,
        identity: RequestIdentity,
        *,
        query: Optional[str] = None,
        name: Optional[str] = None,
        memory_id: Optional[str] = None,
        limit: int = 10,
        include_superseded: bool = False,
    ) -> Dict[str, Any]:
        """Find entities (people, projects, services) by exact name or by
        meaning; ``search_mode`` says how they were found."""
        self._require_scope(identity, READ_SCOPE)
        resolved_type = self._normalize_memory_type("entity_memory", identity)
        text = str(query or "").strip()
        exact = str(name or "").strip()
        if not text and not exact:
            raise MemorizzServerError("invalid_query", "Give a query or a name")
        resolved_limit = self._bounded_limit(limit)
        entities = self._entity_memory()
        found: List[Dict[str, Any]] = []
        mode = "name"
        note = None
        try:
            if exact:
                rows = [
                    row
                    for row in entities.list_entities(
                        memory_id=memory_id,
                        user_id=identity.principal,
                        include_superseded=include_superseded,
                    )
                    if str(row.get("name") or "").strip().lower() == exact.lower()
                ]
                found.extend(rows)
            if text and len(found) < resolved_limit:
                if self._embedding_status().get("ready"):
                    rows, diagnostics = entities.search_entities_with_diagnostics(
                        text,
                        limit=resolved_limit,
                        memory_id=memory_id,
                        user_id=identity.principal,
                    )
                    mode = (
                        "semantic"
                        if diagnostics.get("semantic_match_count")
                        else "keyword"
                    )
                    found.extend(rows)
                if len(found) < resolved_limit:
                    seen = {
                        str(row.get("entity_id") or row.get("_id"))
                        for row in found
                        if isinstance(row, dict)
                    }
                    keyword = [
                        row
                        for row in self._keyword_matches(
                            text,
                            resolved_type,
                            identity,
                            memory_id=memory_id,
                            limit=resolved_limit,
                            include_superseded=include_superseded,
                        )
                        if str(row.get("entity_id") or row.get("_id")) not in seen
                    ]
                    if keyword:
                        mode = "hybrid" if mode == "semantic" else "keyword"
                    found.extend(keyword)
                if not self._embedding_status().get("ready"):
                    note = _NO_EMBEDDINGS_NOTE
        except MemorizzServerError:
            raise
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not look up entities"
            ) from exc
        values: List[Dict[str, Any]] = []
        seen = set()
        for row in found:
            if (
                not isinstance(row, dict)
                or _document_user_id(row) != identity.principal
            ):
                continue
            if not include_superseded and _is_superseded(row):
                continue
            key = str(row.get("entity_id") or row.get("_id") or row.get("id"))
            if key in seen:
                continue
            seen.add(key)
            values.append(self._entity_public(row))
            if len(values) >= resolved_limit:
                break
        result = {
            "ok": True,
            "search_mode": mode if text else "name",
            "entities": values,
            "count": len(values),
        }
        if note:
            result["note"] = note
        return result

    def upsert_entity(
        self,
        name: str,
        identity: RequestIdentity,
        *,
        entity_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an entity or merge facts into the one with this name.

        ``attributes`` maps fact names to values. Each relation names another
        entity by ``target`` (name or entity ID) and a ``relation_type``; a
        target that doesn't exist yet is created with just its name.
        """
        self._require_scope(identity, WRITE_SCOPE)
        self._normalize_memory_type("entity_memory", identity)
        label = str(name or "").strip()
        if not label or len(label) > 200:
            raise MemorizzServerError(
                "invalid_entity", "Give an entity name (up to 200 characters)"
            )
        facts = []
        for key, value in dict(attributes or {}).items():
            key = str(key).strip()
            if not key:
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            text = str(value).strip()
            if len(text) > 2_000:
                raise MemorizzServerError(
                    "invalid_entity", f"Attribute {key} is over 2000 characters"
                )
            facts.append({"name": key, "value": text, "source": "mcp"})
        if len(facts) > 100:
            raise MemorizzServerError(
                "invalid_entity", "Give at most 100 attributes per call"
            )
        entities = self._entity_memory()
        scope = {"memory_id": memory_id, "user_id": identity.principal}
        created_targets: List[str] = []
        links = []
        try:
            for relation in list(relations or [])[:50]:
                target = str(relation.get("target") or "").strip()
                kind = str(relation.get("relation_type") or "").strip()
                if not target or not kind:
                    raise MemorizzServerError(
                        "invalid_entity",
                        "Each relation needs a target and a relation_type",
                    )
                match = entities._fetch_one(
                    {"entity_id": target}, **scope
                ) or entities.get_entity_by_name(target, **scope)
                if match:
                    target_id = str(match.get("entity_id"))
                else:
                    target_id = entities.upsert_entity(name=target, **scope)
                    created_targets.append(target)
                links.append({"entity_id": target_id, "relation_type": kind})
            existed = entities.get_entity_by_name(label, **scope) is not None
            entity_id = entities.upsert_entity(
                name=label,
                entity_type=str(entity_type).strip() if entity_type else None,
                attributes=facts or None,
                relations=links or None,
                **scope,
            )
            record = entities._fetch_one({"entity_id": entity_id}, **scope)
        except MemorizzServerError:
            raise
        except Exception as exc:
            logger.exception("Memorizz MCP entity upsert failed")
            raise MemorizzServerError(
                "provider_error", "The memory provider could not store the entity"
            ) from exc
        return {
            "ok": True,
            "entity_id": entity_id,
            "created": not existed,
            "created_related_entities": created_targets,
            "entity": self._entity_public(
                record or {"entity_id": entity_id, "name": label}
            ),
        }

    def memory_status(
        self, identity: RequestIdentity, *, memory_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """What this server can do right now: backend, policy, embeddings,
        record counts. Never fails because a part is unavailable."""
        self._require_scope(identity, READ_SCOPE)
        from .. import __version__

        backend: Dict[str, Any] = {}
        try:
            provider = self.provider
            backend["name"] = type(provider).__name__
            root = getattr(provider, "root_path", None)
            backend["location"] = (
                str(root) if root else "database (connection details not shown)"
            )
        except Exception as exc:
            backend = {"name": None, "error": f"{type(exc).__name__}"}
        visible = [
            item
            for item in MemoryType
            if self.config.transport == "stdio" or item in _TENANT_MEMORY_TYPES
        ]
        counts: Dict[str, Any] = {}
        if "error" not in backend:
            for item in (
                MemoryType.KNOWLEDGE_BASE,
                MemoryType.SHORT_TERM_MEMORY,
                MemoryType.ENTITY_MEMORY,
                MemoryType.CONVERSATION_MEMORY,
                MemoryType.SUMMARIES,
                MemoryType.WORKFLOW_MEMORY,
            ):
                if item not in visible:
                    continue
                try:
                    rows = self._scoped_rows(item, identity, memory_id=memory_id)
                    superseded = sum(1 for row in rows if _is_superseded(row))
                    counts[item.value] = {
                        "active": len(rows) - superseded,
                        "superseded": superseded,
                    }
                except Exception as exc:
                    counts[item.value] = {"error": type(exc).__name__}
        roots = self._ingest_roots()
        return {
            "ok": True,
            "version": __version__,
            "transport": self.config.transport,
            "principal": identity.principal,
            "policy": {
                "allow_writes": bool(self.config.allow_writes),
                "writes_available": self._writes_allowed(identity),
                "allow_agent_execution": bool(self.config.allow_agent_execution),
                "allow_harness_execution": bool(self.config.allow_harness_execution),
                "allow_trace_queries": bool(self.config.allow_trace_queries),
                "max_result_items": self.config.max_result_items,
            },
            "backend": backend,
            "directly_writable_memory_types": sorted(
                item.value for item in _WRITABLE_MEMORY_TYPES
            ),
            "embeddings": self._embedding_status(refresh=True),
            "memory_id": memory_id,
            "counts": counts,
            "ingest": {
                "available": bool(roots) and self._writes_allowed(identity),
                "roots": [str(item) for item in roots],
                "max_files": _INGEST_MAX_FILES,
                "max_megabytes": _INGEST_MAX_BYTES // 1_048_576,
            },
        }

    # ------------------------------------------------ the agent's MCP servers

    def _connected_manager(self, agent_id: str, identity: RequestIdentity):
        """The agent's own MCP connections, with its stored credentials.

        Harnesses reach Notion, Gmail and the rest only through here: tokens
        stay in MemoRizz, calls are audited, and changes need approval.
        """
        from ..mcp import MCPClientManager

        normalized = str(agent_id or "").strip()
        exposed = self.config.exposed_agent_ids or set()
        if not normalized and len(exposed) == 1:
            # A harness run's server exposes only the agent it works for.
            normalized = next(iter(exposed))
        if not normalized:
            raise MemorizzServerError(
                "invalid_agent",
                "No agent is attached to this server, so pass agent_id "
                "(memorizz_list_agents lists the agents you can use)",
            )
        self._assert_agent_exposed(normalized, identity)
        try:
            record = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError("agent_not_found", "Agent was not found") from exc
        if not record:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        servers = (
            record.get("mcp_servers")
            if isinstance(record, dict)
            else getattr(record, "mcp_servers", None)
        )
        return MCPClientManager(owner_id=normalized, servers=list(servers or []))

    def list_connected_tools(
        self, agent_id: str, identity: RequestIdentity, *, refresh: bool = False
    ) -> Dict[str, Any]:
        """The tools on the agent's MCP servers, and which ones change data."""
        from ..mcp.security import tool_is_mutating

        self._require_scope(identity, READ_SCOPE)
        if (
            not str(agent_id or "").strip()
            and len(self.config.exposed_agent_ids or set()) != 1
        ):
            # Nothing to list rather than an error: a harness run started
            # without an agent simply has no connected tools.
            return {
                "ok": True,
                "agent_id": None,
                "writes_allowed": self._writes_allowed(identity),
                "servers": [],
                "note": (
                    "No agent is attached to this server, so there are no "
                    "connected tools. Pass agent_id to see an agent's tools; "
                    "memorizz_list_agents lists the agents you can use."
                ),
            }
        manager = self._connected_manager(agent_id, identity)
        statuses = {
            row.get("name"): row
            for row in manager.connection_status().get("servers") or []
        }
        rows = []
        for server in manager.server_dicts():
            name = str(server.get("name") or "")
            config = manager.get_server(name)
            tools = manager.cached_tools(name)
            error_code = None
            if refresh or not tools:
                listed = manager.list_tools(name)
                if listed.get("ok"):
                    tools = manager.cached_tools(name)
                else:
                    error_code = listed.get("error_code")
            rows.append(
                {
                    "server_name": name,
                    "signed_in": bool(statuses.get(name, {}).get("authenticated")),
                    "error_code": error_code,
                    "tools": [
                        {
                            "name": tool["name"],
                            "description": tool.get("description") or "",
                            "input_schema": tool.get("inputSchema") or {},
                            "changes_data": tool_is_mutating(
                                tool["name"],
                                tool,
                                read_only_tools=config.read_only_tools,
                                mutation_tools=config.mutation_tools,
                            ),
                        }
                        for tool in tools
                    ],
                }
            )
        return {
            "ok": True,
            "agent_id": manager.owner_id,
            "writes_allowed": self._writes_allowed(identity),
            "servers": rows,
        }

    def _writes_allowed(self, identity: RequestIdentity) -> bool:
        return WRITE_SCOPE in identity.scopes and bool(self.config.allow_writes)

    def call_connected_tool(
        self,
        agent_id: str,
        server_name: str,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        identity: RequestIdentity,
        *,
        read_only: bool = False,
    ) -> Dict[str, Any]:
        """Call a tool on the agent's MCP server under the agent's policy.

        Reads run at once. A tool that changes data needs write access and then
        returns ``approval_required``: a person approves it on MCP connections
        before ``memorizz_resume_connected_tool_call`` runs it. With
        ``read_only`` the call is refused for such tools instead, which lets
        hosts that never prompt (Codex runs) allow the read path.
        """
        from ..mcp.security import tool_is_mutating

        self._require_scope(identity, READ_SCOPE)
        manager = self._connected_manager(agent_id, identity)
        try:
            config = manager.get_server(server_name)
        except Exception as exc:
            raise MemorizzServerError(
                "mcp_server_not_found", "The agent has no MCP server with that name"
            ) from exc
        tools = manager.cached_tools(config.name)
        if not tools:
            # Never listed yet: fetch the server's annotations once so the
            # classification rests on them rather than on the name alone.
            if manager.list_tools(config.name).get("ok"):
                tools = manager.cached_tools(config.name)
        metadata = next(
            (tool for tool in tools if tool["name"] == tool_name),
            None,
        )
        changes_data = tool_is_mutating(
            tool_name,
            metadata,
            read_only_tools=config.read_only_tools,
            mutation_tools=config.mutation_tools,
        )
        if changes_data and read_only:
            raise MemorizzServerError(
                "changes_data",
                "This tool changes data. Use memorizz_call_connected_tool, which "
                "asks a person to approve it.",
            )
        if changes_data and not self._writes_allowed(identity):
            raise MemorizzServerError(
                "read_only",
                "This tool changes data and this connection is read-only. Run "
                "the harness with write access to propose it for approval.",
            )
        return manager.call_tool(
            server_name=config.name,
            tool_name=tool_name,
            arguments=dict(arguments or {}),
        )

    def resume_connected_tool_call(
        self, agent_id: str, proposal_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        """Run a connected-tool call a person approved, exactly as proposed."""
        self._require_scope(identity, WRITE_SCOPE)
        manager = self._connected_manager(agent_id, identity)
        return manager.resume_tool_call(str(proposal_id or ""))

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

    def _harness_request(
        self,
        task: str,
        workspace: Optional[str],
        identity: RequestIdentity,
        *,
        write: bool = False,
        mcp_access: str = "read_only",
        agent_id: Optional[str] = None,
        allowed_env: Optional[List[str]] = None,
        verification_command: Optional[str] = None,
        execution_backend: str = "local",
        **options: Any,
    ):
        """Validate a model-supplied launch and build its tenant-bound task."""
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
        roots = sorted(self.config.harness_workspace_roots or [])
        folder = str(workspace or "").strip()
        if not folder:
            # No project: a fresh empty folder the host creates and allows.
            folder = self.meta_harness.scratch_workspace()
            roots.append(folder)

        from ..metaharness.requests import harness_task

        try:
            return harness_task(
                instruction,
                folder,
                agent_id=normalized_agent_id,
                user_id=identity.principal,
                write=write,
                mcp_access=mcp_access,
                allowed_env=env_names,
                allowed_roots=roots,
                verification_command=verify,
                mode="delegate",
                metadata={
                    "execution_backend": str(execution_backend or "local").lower(),
                    "approval_owner_id": self._approval_owner(identity),
                    "approval_ttl_seconds": self.config.approval_ttl_seconds,
                    "source": "mcp-server",
                    "model_initiated": True,
                },
                **options,
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("invalid_harness_task", str(exc)) from exc

    def _started_run(self, run: Any) -> Dict[str, Any]:
        """A started run as the tools report it: failed, awaiting approval or queued."""
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

    def start_harness_run(
        self,
        task: str,
        workspace: Optional[str],
        identity: RequestIdentity,
        **options: Any,
    ) -> Dict[str, Any]:
        """Start one tenant-bound run; risky envelopes return durable approval.
        A blank workspace runs in a fresh scratch folder."""
        request = self._harness_request(task, workspace, identity, **options)
        return self._started_run(self.meta_harness.start(request))

    def retry_harness_run(
        self, run_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        """Start a finished run's exact task again; risky runs ask again."""
        self._require_harness_execution(identity)
        previous = self._harness_run_for_identity(run_id, identity)
        permissions = (previous.get("task") or {}).get("permissions") or {}
        if (
            permissions.get("workspace_mode") == "direct"
            or permissions.get("mcp_access") == "governed_write"
        ):
            self._require_scope(identity, WRITE_SCOPE)
        try:
            run = self.meta_harness.retry_start(run_id)
        except ValueError as exc:
            raise MemorizzServerError("invalid_harness_retry", str(exc)) from exc
        return self._started_run(run)

    def start_harness_plan(
        self,
        task: str,
        stages: List[Dict[str, Any]],
        workspace: Optional[str],
        identity: RequestIdentity,
        **options: Any,
    ) -> Dict[str, Any]:
        """Start stages in order, each on its own harness; an edit stage waits
        for host approval like any edit run."""
        from ..metaharness.requests import plan_stages

        rows = list(stages or [])
        for row in rows:
            if not isinstance(row, dict):
                raise MemorizzServerError(
                    "invalid_harness_plan", "Each stage must be an object"
                )
            if len(str(row.get("instruction") or "")) > self.config.max_text_chars:
                raise MemorizzServerError(
                    "invalid_harness_plan", "A stage instruction is too long"
                )
            stage_agent_id = str(row.get("agent_id") or "").strip()
            if stage_agent_id:
                self._assert_agent_exposed(stage_agent_id, identity)
        if any(bool(row.get("write")) for row in rows):
            self._require_harness_execution(identity)
            self._require_scope(identity, WRITE_SCOPE)
        # The base task is read-only; each stage sets its own mode.
        request = self._harness_request(task, workspace, identity, **options)
        try:
            value = self.meta_harness.start_plan(
                request, plan_stages(rows, request.task)
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("invalid_harness_plan", str(exc)) from exc
        return {"ok": True, "workflow": value}

    def start_harness_comparison(
        self,
        task: str,
        harnesses: List[str],
        workspace: Optional[str],
        identity: RequestIdentity,
        *,
        harness_models: Optional[Dict[str, str]] = None,
        **options: Any,
    ) -> Dict[str, Any]:
        """Run one read-only task on several harnesses at once, each on its
        own model when ``harness_models`` names one."""
        request = self._harness_request(task, workspace, identity, **options)
        try:
            value = self.meta_harness.start_compare(
                request,
                list(harnesses or []),
                models=dict(harness_models or {}) or None,
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("invalid_harness_comparison", str(exc)) from exc
        return {"ok": True, "workflow": value}

    def rerun_harness_workflow(
        self, workflow_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        """Run a finished plan or comparison again with the same settings."""
        self._require_harness_execution(identity)
        value = self._workflow_for_identity(workflow_id, identity)
        steps = value.get("steps") or []
        permissions = (value.get("task") or {}).get("permissions") or {}
        if any(step.get("workspace_mode") == "direct" for step in steps) or (
            permissions.get("mcp_access") == "governed_write"
        ):
            self._require_scope(identity, WRITE_SCOPE)
        try:
            started = self.meta_harness.rerun_orchestration(value["orchestration_id"])
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("invalid_harness_rerun", str(exc)) from exc
        return {"ok": True, "workflow": started}

    def delete_harness_workflow(
        self, workflow_id: str, identity: RequestIdentity, *, keep_scratch: bool = False
    ) -> Dict[str, Any]:
        """Delete a finished plan or comparison with all its runs."""
        self._require_harness_execution(identity)
        self._require_scope(identity, WRITE_SCOPE)
        value = self._workflow_for_identity(workflow_id, identity)
        try:
            return self.meta_harness.delete_orchestration(
                value["orchestration_id"], remove_scratch=not keep_scratch
            )
        except (TypeError, ValueError) as exc:
            raise MemorizzServerError("harness_workflow_not_deleted", str(exc)) from exc

    def delete_harness_runs(
        self,
        run_ids: List[str],
        identity: RequestIdentity,
        *,
        keep_scratch: bool = False,
    ) -> Dict[str, Any]:
        """Delete this caller's finished runs (and their delegates' runs);
        others' runs, runs still working and workflow steps are kept."""
        self._require_harness_execution(identity)
        self._require_scope(identity, WRITE_SCOPE)
        owned: List[str] = []
        kept: List[Dict[str, Any]] = []
        for run_id in list(run_ids or [])[: self.config.max_result_items]:
            try:
                owned.append(self._harness_run_for_identity(run_id, identity)["run_id"])
            except MemorizzServerError:
                kept.append({"run_id": str(run_id), "reason": "not_found"})
        try:
            result = self.meta_harness.delete_runs(
                owned, remove_scratch=not keep_scratch
            )
        except TypeError as exc:
            raise MemorizzServerError("harness_runs_not_deleted", str(exc)) from exc
        result["kept"] = kept + list(result.get("kept") or [])
        return result

    def _conversation_for_identity(
        self, conversation: str, identity: RequestIdentity
    ) -> tuple:
        self._require_scope(identity, READ_SCOPE)
        value = str(conversation or "").strip()
        if not value.startswith("hxc-"):
            value = self.meta_harness.conversation_id_for(
                self._harness_run_for_identity(value, identity)
            )
        runs = [
            run
            for run in self.meta_harness.conversation(value)
            if identity.principal is None
            or (run.get("task") or {}).get("user_id") == identity.principal
        ]
        if not runs:
            raise MemorizzServerError(
                "harness_conversation_not_found", "Harness conversation was not found"
            )
        return value, runs

    def get_harness_conversation(
        self, conversation: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        """A harness conversation's turns, oldest first."""
        conversation_id, runs = self._conversation_for_identity(conversation, identity)
        turns = [
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "harness": run.get("harness"),
                "request": (run.get("task") or {}).get("task"),
                "answer": (run.get("result") or {}).get("final_response"),
            }
            for run in runs
        ]
        return {"ok": True, "conversation_id": conversation_id, "turns": turns}

    def continue_harness_conversation(
        self,
        conversation: str,
        message: str,
        identity: RequestIdentity,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Start the next turn of a conversation with the latest turn's setup
        (and any overrides); earlier turns go with it as context."""
        from ..metaharness import catalog

        conversation_id, runs = self._conversation_for_identity(conversation, identity)
        setup = catalog.chat_setup(runs[-1])
        options = {
            "harness": setup["harness"],
            "model": setup["model"] or None,
            "agent_id": setup["agent_id"] or None,
            "memory_id": setup["memory_id"] or None,
            "write": setup["write"],
            "allow_dirty_workspace": setup["allow_dirty_workspace"],
            "network": setup["network"],
            "mcp_access": setup["mcp_access"],
            "allow_subagents": setup["allow_subagents"],
            "timeout_seconds": setup["timeout_seconds"],
            "max_steps": setup["max_steps"],
            "max_cost_usd": setup["max_cost_usd"],
            "execution_backend": setup["execution_backend"],
        }
        options.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        request = self._harness_request(
            message, setup["workspace"], identity, **options
        )
        return self._started_run(
            self.meta_harness.continue_conversation(conversation_id, request)
        )

    def _workflow_for_identity(
        self, workflow_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        value = self.meta_harness.get_orchestration(str(workflow_id or "").strip())
        owner = ((value or {}).get("task") or {}).get("user_id")
        if value is None or (
            identity.principal is not None and owner != identity.principal
        ):
            raise MemorizzServerError(
                "harness_workflow_not_found", "Harness workflow was not found"
            )
        return value

    def _tenant_harness_rows(
        self,
        lister: Callable[..., Any],
        identity: RequestIdentity,
        *,
        status: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Up to ``limit`` of the caller's rows, newest first. The owner goes
        into the store query when it accepts one; otherwise the listing is
        widened (bounded) until enough owned rows are found, so a tenant whose
        work is older than other tenants' recent work still sees it."""
        if identity.principal is None:
            return list(lister(status=status, limit=limit) or [])

        def owned(rows: Any) -> List[Dict[str, Any]]:
            return [
                row
                for row in list(rows or [])
                if isinstance(row, dict)
                and (row.get("task") or {}).get("user_id") == identity.principal
            ]

        if _accepts_keyword(lister, "user_id"):
            return owned(
                lister(status=status, limit=limit, user_id=identity.principal)
            )[:limit]
        batch = limit
        while True:
            rows = list(lister(status=status, limit=batch) or [])
            matches = owned(rows)
            if (
                len(matches) >= limit
                or len(rows) < batch
                or batch >= _TENANT_SCAN_LIMIT
            ):
                return matches[:limit]
            batch = min(batch * 4, _TENANT_SCAN_LIMIT)

    def list_harness_workflows(
        self,
        identity: RequestIdentity,
        *,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        rows = self._tenant_harness_rows(
            self.meta_harness.list_orchestrations,
            identity,
            status=status,
            limit=self._bounded_limit(limit),
        )
        return {"ok": True, "workflows": rows, "count": len(rows)}

    def get_harness_workflow(
        self, workflow_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        value = self._workflow_for_identity(workflow_id, identity)
        runs = [
            self.meta_harness.get_run(step["run_id"])
            for step in value.get("steps") or []
            if step.get("run_id")
        ]
        return {"ok": True, "workflow": value, "runs": [run for run in runs if run]}

    def cancel_harness_workflow(
        self, workflow_id: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_harness_execution(identity)
        self._workflow_for_identity(workflow_id, identity)
        return self.meta_harness.cancel_orchestration(workflow_id)

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
        rows = self._tenant_harness_rows(
            self.meta_harness.list_runs,
            identity,
            status=status,
            limit=self._bounded_limit(limit),
        )
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
