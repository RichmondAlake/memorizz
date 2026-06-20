# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ..conversation_history import is_trace_bundle_entry
from ..enums import ApplicationMode, ApplicationModeConfig, MemoryType, Role
from ..internet_access import get_default_internet_access_provider
from ..llms.llm_factory import create_llm_provider
from .constants import (
    BASE_SYSTEM_PROMPT,
    CONTINUOUS_MAX_STEPS,
    DEFAULT_INSTRUCTION,
    DEFAULT_MAX_STEPS,
    DEFAULT_TOOL_ACCESS,
    SANDBOX_SYSTEM_PROMPT,
    TOOL_ITERATION_BUDGET_INSTRUCTION,
)
from .managers import (
    AutomationManager,
    CacheManager,
    EntityMemoryManager,
    InternetAccessManager,
    MemoryManager,
    PersonaManager,
    SandboxManager,
    SelfAwarenessManager,
    ToolManager,
    WorkflowManager,
)

if TYPE_CHECKING:
    from ..automation.models import AutomationJob, AutomationRun
    from ..internet_access import InternetAccessProvider
    from ..sandbox.base import SandboxProvider

logger = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable primitive.

    Oracle ``oracledb`` surfaces CLOB/BLOB columns as LOB objects with a
    ``.read()`` method rather than as strings. These slip through
    conversation-history, knowledge-base, and summary loads into the
    message list we hand to the LLM provider, where the streaming path
    ultimately does ``json.dumps(...)`` and fails with
    ``Object of type LOB is not JSON serializable``.

    Rather than chase every data path individually, we sanitize at the
    prompt-assembly boundary. One helper, used everywhere we serialize
    model input or tool output.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            read_value = reader()
        except Exception:
            return str(value)
        return _to_jsonable(read_value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Tool-log disambiguation helpers
# ---------------------------------------------------------------------------
# The agent-facing tool placeholder needs enough signal for the LLM to pick
# the right ``tool_log_id`` among multiple calls to the same tool in one
# session. These helpers centralize the three improvements:
#   1) args summary so calls to the same tool can be told apart
#   2) field-aware result summary for common tool output shapes
#   3) a single builder used by both streaming and non-streaming paths
_TOOL_LOG_PLACEHOLDER_PREFIX = "[Tool '"


def _summarize_tool_args(arguments: Any, limit: int = 200) -> str:
    """One-line, length-bounded JSON summary of tool arguments.

    Falls back to ``str(arguments)`` for anything non-JSON. Empty mapping
    returns an empty string so the caller can omit the field cleanly.
    """
    if arguments is None:
        return ""
    try:
        if isinstance(arguments, str):
            # Already serialized — just trim.
            text = arguments
        else:
            text = json.dumps(_to_jsonable(arguments), ensure_ascii=False)
    except Exception:
        text = str(arguments)
    text = text.strip()
    if not text or text in ("{}", "null"):
        return ""
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _summarize_tool_result(result: Any, preview_limit: int = 500) -> str:
    """Field-aware summary of a tool's return value.

    Produces a one-line hint based on common tool output shapes plus a
    bounded ``Preview:`` of the most content-rich field. Unknown shapes
    fall through to ``str(result)[:preview_limit]`` so nothing blows up.
    """
    if result is None:
        return ""
    # Errors first — always useful to surface verbatim
    if isinstance(result, dict):
        if "error" in result and result.get("error"):
            err = str(result["error"])
            return f"error: {err[:preview_limit]}"
        # matches = KB lookup, entity lookup, semantic search
        matches = result.get("matches")
        if isinstance(matches, list):
            hints: List[str] = [f"{len(matches)} match(es)"]
            if matches and isinstance(matches[0], dict):
                top = matches[0]
                ns = top.get("namespace") or top.get("name")
                if ns:
                    hints.append(f"top namespace: {ns}")
                first_content = top.get("content") or top.get("text") or ""
                if isinstance(first_content, str) and first_content.strip():
                    snippet = first_content.strip().replace("\n", " ")
                    if len(snippet) > preview_limit:
                        snippet = snippet[: preview_limit - 1] + "…"
                    hints.append(f"top content: {snippet}")
            return " · ".join(hints)
        # generic list-of-entries return shapes
        entries = result.get("entries") or result.get("logs") or result.get("items")
        if isinstance(entries, list):
            return f"{len(entries)} entr(ies)"
        # single-result wrappers
        if "content" in result and isinstance(result["content"], str):
            content = result["content"].strip().replace("\n", " ")
            if len(content) > preview_limit:
                content = content[: preview_limit - 1] + "…"
            return f"content: {content}"
        if "ok" in result:
            return f"ok={result['ok']}"

    text = str(result).strip().replace("\n", " ")
    if len(text) > preview_limit:
        text = text[: preview_limit - 1] + "…"
    return text


def _build_tool_log_placeholder(
    *,
    tool_name: str,
    tool_log_id: str,
    arguments: Any,
    result: Any,
    error_message: Optional[str] = None,
) -> str:
    """Compose the bracketed placeholder that replaces the raw tool result
    in the LLM message list and in conversation_memory.

    Includes (in order) the tool name, a short args summary, the
    tool_log_id + retrieval-call template, and a field-aware preview.
    That's enough signal for the LLM to pick the right id even among
    several calls to the same tool in one session.
    """
    args_summary = _summarize_tool_args(arguments)
    result_summary = _summarize_tool_result(result)
    args_clause = f" args={args_summary}" if args_summary else ""
    if error_message:
        header = (
            f"{_TOOL_LOG_PLACEHOLDER_PREFIX}{tool_name}' failed: {error_message}."
            f"{args_clause} Full output stored as tool_log:{tool_log_id} — "
            f"use retrieve_tool_log_entry('{tool_log_id}') for complete data]"
        )
    else:
        header = (
            f"{_TOOL_LOG_PLACEHOLDER_PREFIX}{tool_name}' executed successfully."
            f"{args_clause} Full output stored as tool_log:{tool_log_id} — "
            f"use retrieve_tool_log_entry('{tool_log_id}') for complete data]"
        )
    return f"{header}\nSummary: {result_summary}" if result_summary else header


def _is_tool_placeholder_content(text: Any) -> bool:
    """True when a stored conversation row's content is a tool-log placeholder.

    Used to filter these rows out of the message list handed to the LLM
    (they'd otherwise appear as orphan tool-role messages without a
    matching tool_call_id and the OpenAI API would reject the batch).
    The placeholders remain visible in the UI and are surfaced to the
    agent as a structured digest in the system prompt instead.
    """
    if not isinstance(text, str):
        return False
    return text.lstrip().startswith(_TOOL_LOG_PLACEHOLDER_PREFIX)


_MIN_HISTORY_LIMIT = 24
_MAX_HISTORY_LIMIT = 120
_DEFAULT_HISTORY_LIMIT = 60
_FALLBACK_HISTORY_MESSAGES = 24
_PROMPT_WINDOW_RATIO = 0.8
_PROMPT_BUFFER_TOKENS = 200


class MemAgent:
    """
    MemAgent class that orchestrates manager components.

    """

    def __init__(
        self,
        model: Optional[Any] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List, Any]] = None,
        persona: Optional[Any] = None,
        instruction: Optional[str] = None,
        application_mode: Optional[str] = None,
        memory_types: Optional[List[Union[str, Any]]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        memory_provider: Optional[Any] = None,
        memory_ids: Optional[Union[str, List[str]]] = None,
        agent_id: Optional[str] = None,
        tool_access: Optional[str] = DEFAULT_TOOL_ACCESS,
        delegates: Optional[List["MemAgent"]] = None,
        verbose: bool = None,
        embedding_provider: Optional[str] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        semantic_cache: bool = False,
        semantic_cache_config: Optional[Union[Any, Dict[str, Any]]] = None,
        context_window_tokens: Optional[int] = None,
        internet_access_provider: Optional["InternetAccessProvider"] = None,
        skills_marketplace_provider: Optional[Union[str, Dict[str, Any]]] = None,
        skills_marketplace_config: Optional[Dict[str, Any]] = None,
        sandbox_provider: Optional[
            Union[str, Dict[str, Any], "SandboxProvider"]
        ] = None,
        skill_paths: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        automations_enabled: bool = True,
        default_timezone: Optional[str] = None,
        self_aware: bool = False,
        self_aware_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        is_favorite: bool = False,
        streaming: bool = False,
    ):
        """Initialize the MemAgent with configuration."""
        self.streaming = streaming
        # Store configuration
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name.strip() if isinstance(name, str) and name.strip() else None
        self.is_favorite = bool(is_favorite)
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_steps = max_steps
        self.memory_provider = memory_provider
        self.memory_ids = (
            memory_ids
            if isinstance(memory_ids, list)
            else [memory_ids]
            if memory_ids
            else []
        )
        # Populated by MemAgent.load() / KnowledgeBase.attach_to_agent().
        # Initialized to [] here so `getattr(self, "knowledge_base_ids")`
        # always returns a real list rather than AttributeError.
        self.knowledge_base_ids: List[str] = []
        self.tools = tools if tools is not None else []
        self.skill_paths = self._normalize_skill_paths(skill_paths)
        self.skills: List[Dict[str, Any]] = []
        self.mcp_servers = self._normalize_mcp_servers(mcp_servers)
        self.automations_enabled = bool(automations_enabled)
        self.default_timezone = (
            default_timezone.strip()
            if isinstance(default_timezone, str) and default_timezone.strip()
            else None
        )
        self.self_aware = bool(self_aware)
        self.self_aware_config = (
            dict(self_aware_config) if isinstance(self_aware_config, dict) else None
        )

        (
            self.application_mode,
            self._application_mode_explicit,
        ) = self._resolve_application_mode(application_mode)

        # Store memory types for tracking which memory systems are active
        self.active_memory_types = self._initialize_memory_types(
            self.application_mode if self._application_mode_explicit else None,
            memory_types,
        )

        # Initialize thread state tracking
        self._current_thread_id = None
        self._current_memory_id = None
        self._thread_ids_by_memory: Dict[str, str] = {}
        self._last_entity_context: List[Dict[str, Any]] = []

        # Initialize LLM. We stash any construction error on the instance
        # so the chat endpoints can surface the real cause (e.g. "model not
        # cached locally", "401 from HuggingFace") instead of the generic
        # "No LLM model configured" — which sends users on a wild goose
        # chase when in reality the model failed to load.
        self.model = model
        # Preserve the resolved LLM config dict so callers / status pages /
        # debug tooling can introspect ``agent.llm_config`` /
        # ``agent.llm_provider`` / ``agent.llm_model`` without having to
        # crack open the underlying model wrapper.
        self.llm_config: Dict[str, Any] = dict(llm_config) if llm_config else {}
        self._llm_init_error: Optional[str] = None
        if not model and llm_config:
            try:
                self.model = create_llm_provider(llm_config)
            except Exception as e:
                self._llm_init_error = f"{type(e).__name__}: {e}"
                logger.warning("Could not create LLM from config: %s", e)

        self._context_window_tokens = self._initialize_context_window_tokens(
            context_window_tokens, llm_config
        )
        self._last_context_window_stats: Optional[Dict[str, Any]] = None

        # Track entity memory state before manager initialization
        self._entity_memory_enabled = False
        self._entity_memory_tools_registered = False
        self._entity_memory_tool_names = (
            "entity_memory_lookup",
            "entity_memory_upsert",
        )
        self._internet_access_tools_registered = False
        self._internet_access_tool_names = ("internet_search", "open_web_page")
        self._persona_tools_registered = False
        self._persona_tool_names = ("update_persona", "read_persona")
        self._skills_marketplace_tools_registered = False
        self._skills_marketplace_tool_names = (
            "skills_marketplace_search",
            "vercel_skills_search",
            "vercel_skill_fetch",
        )
        self._skills_marketplace_provider_name: Optional[str] = None
        self._skills_marketplace_config: Dict[str, Any] = {}
        self._context_tools_registered = False
        self._summary_registry: List[Dict[str, Any]] = []
        self._known_summary_ids: Set[str] = set()
        self._context_summary_trigger = 80.0
        self._context_summary_cooldown = 90.0
        self._last_summary_timestamp = 0.0
        self._internet_access_failure_count = 0
        self._internet_access_disabled_reason: Optional[str] = None
        self._sandbox_tools_registered = False
        self._sandbox_tool_names = (
            "execute_code",
            "sandbox_write_file",
            "sandbox_read_file",
        )
        self._skill_tools_registered = False
        self._skill_tool_names = (
            "list_skills",
            "read_skill",
            "run_skill_code",
            "run_skill_script",
        )
        self._mcp_tools_registered = False
        self._mcp_tool_names = (
            "list_mcp_servers",
            "mcp_list_tools",
            "mcp_call_tool",
        )
        self._self_aware_tools_registered = False
        self._self_aware_tool_names = (
            "self_aware_list_roots",
            "self_aware_list_files",
            "self_aware_read_file",
            "self_aware_search_files",
            "self_aware_write_file",
            "self_aware_delete_path",
            "self_aware_run_command",
        )
        self._stream_event_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self._stream_trace_events: Optional[List[Dict[str, Any]]] = None

        # Initialize manager components
        self._initialize_managers(
            memory_provider=memory_provider,
            semantic_cache=semantic_cache,
            semantic_cache_config=semantic_cache_config,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            internet_access_provider=internet_access_provider,
            sandbox_provider=sandbox_provider,
            self_aware_config=self_aware_config,
        )
        self._register_context_monitor_tools()
        self._register_knowledge_base_tools()

        provider_to_attach = internet_access_provider
        if (
            not provider_to_attach
            and self.application_mode == ApplicationMode.DEEP_RESEARCH
        ):
            try:
                provider_to_attach = get_default_internet_access_provider()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to initialize default internet provider: %s", exc
                )
                provider_to_attach = None

        if provider_to_attach:
            try:
                self.with_internet_access_provider(provider_to_attach)
            except Exception as exc:
                logger.warning("Internet access provider failed to initialize: %s", exc)

        skills_provider_to_attach = skills_marketplace_provider
        skills_provider_config = (
            dict(skills_marketplace_config)
            if isinstance(skills_marketplace_config, dict)
            else None
        )
        if not skills_provider_to_attach:
            default_skills_provider = (
                os.getenv("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
                .strip()
                .lower()
            )
            if default_skills_provider:
                skills_provider_to_attach = default_skills_provider
        if skills_provider_to_attach:
            try:
                self.with_skills_marketplace_provider(
                    skills_provider_to_attach,
                    config=skills_provider_config,
                )
            except Exception as exc:
                logger.warning(
                    "Skills marketplace provider failed to initialize: %s", exc
                )

        # Enable entity memory automatically if configured via application mode
        if MemoryType.ENTITY_MEMORY in self.active_memory_types:
            try:
                self.with_entity_memory(True)
            except Exception as exc:
                logger.warning("Entity memory failed to initialize: %s", exc)

        # Initialize tools if provided
        if tools:
            try:
                self._initialize_tools(tools)
            except Exception as exc:
                logger.warning("Tool initialization failed: %s", exc)

        # Initialize persona if provided
        if persona:
            try:
                self.persona_manager.set_persona(persona, self.agent_id, save=False)
            except Exception as exc:
                logger.warning("Persona initialization failed: %s", exc)

        # Register persona evolution tools (update_persona, read_persona) once a
        # persona is attached. The registration is idempotent and safe to call
        # again via set_persona().
        try:
            self._register_persona_tools()
        except Exception as exc:
            logger.warning("Persona tools registration failed: %s", exc)

        self.skills = self._load_skills()
        if self.skills:
            try:
                self._register_skill_tools()
            except Exception as exc:
                logger.warning("Skill tools registration failed: %s", exc)

        if self.mcp_servers:
            try:
                self._register_mcp_tools()
            except Exception as exc:
                logger.warning("MCP tools registration failed: %s", exc)

        # Configure self-awareness after base tools/managers are initialized.
        try:
            self.with_self_aware(self.self_aware, config=self.self_aware_config)
        except Exception as exc:
            logger.warning("Self-awareness initialization failed: %s", exc)

        logger.info(
            f"MemAgent {self.agent_id} initialized with memory types: {self.active_memory_types}"
        )

    @property
    def llm_provider(self) -> Optional[str]:
        """The configured LLM provider name (``"openai"``, ``"anthropic"``, …).

        Read from the resolved ``llm_config`` dict. Returns ``None`` when
        the agent was constructed with a pre-built ``model=`` instance
        rather than a config dict.
        """
        return self.llm_config.get("provider") if self.llm_config else None

    @property
    def llm_model(self) -> Optional[str]:
        """The configured LLM model name (e.g. ``"gpt-5-mini"``).

        Read from the resolved ``llm_config`` dict. Returns ``None`` when
        the agent was constructed with a pre-built ``model=`` instance.
        """
        return self.llm_config.get("model") if self.llm_config else None

    def _resolve_application_mode(
        self, application_mode: Optional[Union[str, ApplicationMode]]
    ) -> Tuple[ApplicationMode, bool]:
        """Resolve and validate application mode, returning (mode, was_explicit)."""
        if application_mode is not None:
            try:
                mode = ApplicationModeConfig.validate_mode(application_mode)
                return mode, True
            except ValueError:
                logger.warning(f"Invalid application mode: {application_mode}")
        return ApplicationMode.DEFAULT, False

    def _initialize_memory_types(self, application_mode, memory_types):
        """Initialize active memory types based on application mode or explicit memory_types."""
        # If explicit memory_types provided, use them when they are list-like.
        if memory_types is not None:
            raw_values: Optional[List[Any]] = None
            if isinstance(memory_types, str):
                raw_values = [memory_types]
            elif isinstance(memory_types, (list, tuple, set)):
                raw_values = list(memory_types)
            else:
                logger.warning(
                    "Ignoring unsupported memory_types payload type: %s",
                    type(memory_types).__name__,
                )

            if raw_values:
                # Convert string memory types to MemoryType enums if needed.
                result = []
                for mt in raw_values:
                    if isinstance(mt, str):
                        try:
                            result.append(MemoryType[mt.upper()])
                        except KeyError:
                            logger.warning(f"Unknown memory type: {mt}")
                    else:
                        result.append(mt)
                if result:
                    return result

        if application_mode:
            try:
                mode = (
                    application_mode
                    if isinstance(application_mode, ApplicationMode)
                    else ApplicationModeConfig.validate_mode(application_mode)
                )
                return ApplicationModeConfig.get_memory_types(mode)
            except ValueError:
                logger.warning(f"Invalid application mode: {application_mode}")

        # Otherwise, derive from application_mode or use defaults.
        # Default includes summaries so agents can auto-compress long conversations.
        return [
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.WORKFLOW_MEMORY,
            MemoryType.SUMMARIES,
        ]

    def _initialize_managers(
        self,
        memory_provider=None,
        semantic_cache=False,
        semantic_cache_config=None,
        embedding_provider=None,
        embedding_config=None,
        internet_access_provider=None,
        sandbox_provider=None,
        self_aware_config=None,
    ):
        """Initialize all manager components."""
        # Memory Manager
        if memory_provider:
            self.memory_manager = MemoryManager(memory_provider)
        else:
            self.memory_manager = None
            logger.warning("No memory provider - memory functionality disabled")

        # Entity Memory Manager
        if memory_provider:
            self.entity_memory_manager = EntityMemoryManager(memory_provider)
        else:
            self.entity_memory_manager = None

        # Tool Manager
        self.tool_manager = ToolManager(memory_provider)

        # Automation Manager (durable scheduling + run history) when supported by provider.
        self.automation_manager = AutomationManager(
            memory_provider,
            enabled=bool(getattr(self, "automations_enabled", True)),
        )
        if self.automation_manager and self.automation_manager.is_enabled():
            try:
                self.automation_manager.register_tools(
                    self.tool_manager,
                    agent_id=self.agent_id,
                    default_timezone=getattr(self, "default_timezone", None),
                )
            except Exception as exc:
                logger.warning("Failed to register automation tools: %s", exc)

        # Cache Manager
        self.cache_manager = CacheManager(
            enabled=semantic_cache,
            config=semantic_cache_config,
            agent_id=self.agent_id,
            memory_id=self.memory_ids[0] if self.memory_ids else None,
            memory_provider=memory_provider,  # ✅ Pass memory provider for persistence
        )

        # Persona Manager
        self.persona_manager = PersonaManager(memory_provider)

        # Workflow Manager
        self.workflow_manager = WorkflowManager()

        # Internet Access Manager
        self.internet_access_manager = InternetAccessManager(internet_access_provider)

        # Sandbox Manager
        if sandbox_provider:
            try:
                self.sandbox_manager = SandboxManager.from_config(sandbox_provider)
                self._register_sandbox_tools()
            except Exception as exc:
                logger.warning("Sandbox provider failed to initialize: %s", exc)
                self.sandbox_manager = None
        else:
            self.sandbox_manager = None

        # Self-awareness manager (host codebase awareness and guarded file/CLI ops)
        self.self_awareness_manager = SelfAwarenessManager(config=self_aware_config)
        self.self_aware_config = self.self_awareness_manager.get_config()

    def _initialize_context_window_tokens(
        self, explicit_value: Optional[int], llm_config: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """Configure the context window token budget from overrides/config/provider."""
        if explicit_value and explicit_value > 0:
            return explicit_value

        config_value = self._extract_context_window_from_config(llm_config)
        if config_value:
            return config_value

        return self._get_model_context_window()

    def _get_tool_iteration_limit(self) -> int:
        """Return the max tool-calling iterations allowed in one run loop."""
        try:
            configured_limit = int(self.max_steps)
        except (TypeError, ValueError):
            configured_limit = DEFAULT_MAX_STEPS
        if configured_limit == CONTINUOUS_MAX_STEPS:
            return 10_000_000  # Effectively unlimited
        if configured_limit < 1:
            return 1
        return min(configured_limit, 1000)

    def _extract_context_window_from_config(
        self, llm_config: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        if not llm_config:
            return None
        for key in ("context_window_tokens", "max_context_tokens", "context_window"):
            value = llm_config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        return None

    def _get_model_context_window(self) -> Optional[int]:
        if self.model and hasattr(self.model, "get_context_window_tokens"):
            try:
                value = self.model.get_context_window_tokens()
                if value and value > 0:
                    return value
            except Exception as exc:
                logger.debug(f"Could not read context window from provider: {exc}")
        return None

    def _get_last_model_usage(self) -> Optional[Dict[str, int]]:
        if self.model and hasattr(self.model, "get_last_usage"):
            try:
                return self.model.get_last_usage()
            except Exception as exc:
                logger.debug(f"Could not access provider usage stats: {exc}")
        return None

    def _record_context_window_usage(self, stage: str) -> None:
        usage = self._get_last_model_usage()
        if not usage:
            return

        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None

        if total_tokens is None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        context_window = self._context_window_tokens
        percentage_used = (
            (total_tokens / context_window) * 100
            if context_window and total_tokens is not None and context_window > 0
            else None
        )

        stats = {
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "context_window_tokens": context_window,
            "percentage_used": percentage_used,
            "stage": stage,
        }

        self._last_context_window_stats = stats
        self._maybe_generate_context_summary()

        if context_window and percentage_used is not None:
            logger.info(
                "Context window usage (%s): %s/%s tokens (%.2f%%) | prompt=%s completion=%s",
                stage,
                total_tokens,
                context_window,
                percentage_used,
                prompt_tokens,
                completion_tokens,
            )
        else:
            logger.info(
                "Context window usage (%s): total=%s tokens | prompt=%s completion=%s",
                stage,
                total_tokens,
                prompt_tokens,
                completion_tokens,
            )

    def _get_conversation_history_limit(self) -> int:
        """Return a dynamic conversation-history retrieval limit."""
        context_window = self._context_window_tokens or 0
        if context_window >= 200000:
            return _MAX_HISTORY_LIMIT
        if context_window >= 100000:
            return 100
        if context_window >= 64000:
            return 80
        if context_window >= 32000:
            return _DEFAULT_HISTORY_LIMIT
        if context_window >= 16000:
            return 40
        return _MIN_HISTORY_LIMIT

    def _estimate_text_tokens(self, text: Any) -> int:
        """Estimate token count quickly without provider-specific tokenizers."""
        content = str(text or "").strip()
        if not content:
            return 0
        return max(1, int(len(content) / 4))

    def _normalize_history_item(self, item: Any) -> Optional[Dict[str, str]]:
        """Normalize history records from provider formats into chat messages."""
        role = ""
        text = ""

        if isinstance(item, dict):
            if "role" in item and ("content" in item or "text" in item):
                role = str(item.get("role") or "").strip().lower()
                text = str(item.get("content") or item.get("text") or "")
            elif "content" in item and isinstance(item.get("content"), dict):
                nested = item.get("content") or {}
                role = str(nested.get("role") or "user").strip().lower()
                text = str(nested.get("content") or nested.get("text") or "")

        if not text.strip():
            return None

        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        return {"role": role, "content": text}

    def _prepare_history_messages(
        self, history: Any, system_prompt: str, query: str
    ) -> List[Dict[str, str]]:
        """Select a recency-biased history subset that fits the prompt budget."""
        if not isinstance(history, list) or not history:
            return []

        normalized: List[Dict[str, str]] = []
        for item in history:
            parsed = self._normalize_history_item(item)
            if parsed:
                normalized.append(parsed)
        if not normalized:
            return []

        # OpenAI Chat Completions rejects any ``role: "tool"`` message that
        # isn't immediately preceded by an ``role: "assistant"`` message
        # carrying ``tool_calls`` (the LLM matches the tool result to the
        # original call via ``tool_call_id``). Stored conversation history
        # does not preserve ``tool_calls`` metadata on the assistant turn —
        # only the rendered text — so any tool-role entry pulled back from
        # the provider would be an orphan and trip a 400 from the API.
        #
        # The tool placeholder text has already been embedded into the
        # assistant's textual reply via ``_build_tool_log_placeholder``
        # before persistence, so dropping the standalone tool rows here
        # loses no information the model needs.
        normalized = [m for m in normalized if m.get("role") != "tool"]
        if not normalized:
            return []

        history_limit = self._get_conversation_history_limit()
        normalized = normalized[-history_limit:]

        context_window = self._context_window_tokens
        if not context_window or context_window <= 0:
            return normalized[-_FALLBACK_HISTORY_MESSAGES:]

        base_tokens = (
            self._estimate_text_tokens(system_prompt)
            + self._estimate_text_tokens(query)
            + _PROMPT_BUFFER_TOKENS
        )
        prompt_budget = max(512, int(context_window * _PROMPT_WINDOW_RATIO))
        history_budget = max(160, prompt_budget - base_tokens)

        selected_reversed: List[Dict[str, str]] = []
        consumed = 0
        for message in reversed(normalized):
            message_cost = self._estimate_text_tokens(message.get("content")) + 6
            if selected_reversed and (consumed + message_cost) > history_budget:
                break
            selected_reversed.append(message)
            consumed += message_cost
            if consumed >= history_budget:
                break

        if not selected_reversed:
            return normalized[-1:]
        return list(reversed(selected_reversed))

    def _build_prompt_messages(
        self,
        system_prompt: str,
        query: str,
        context: Dict[str, Any],
        request_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build model input messages with bounded conversation history.

        ``request_context`` is the M2 per-call ephemeral context: when
        provided, it's rendered as a system-role message between the main
        system prompt and the conversation history. It is intentionally
        NOT persisted by ``_record_interaction`` (which only stores the
        original ``query`` string), so it does not pollute
        ``conversation_memory``.
        """
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        if request_context:
            try:
                rendered_context = json.dumps(
                    request_context, ensure_ascii=False, indent=2, default=str
                )
            except Exception:
                rendered_context = str(request_context)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "REQUEST CONTEXT (ephemeral, this turn only — do not "
                        "reference unless directly relevant to the user's "
                        "current query):\n" + rendered_context
                    ),
                }
            )

        history = context.get("conversation_history", [])
        messages.extend(self._prepare_history_messages(history, system_prompt, query))
        messages.append({"role": "user", "content": query})
        # Sanitize once at the boundary so any LOB that slipped through from
        # memory loaders (conversation history, KB retrievals, summaries…)
        # doesn't blow up the LLM provider's json.dumps during streaming.
        return _to_jsonable(messages)

    def _register_context_monitor_tools(self):
        """Register tools that expose context stats and summaries."""
        if not self.tool_manager or self._context_tools_registered:
            return

        def context_window_stats_tool() -> Dict[str, Any]:
            """Return latest context window stats."""
            return self.get_context_window_stats() or {}

        def list_summary_registry_tool() -> Dict[str, Any]:
            """List all summaries for the current thread. Each entry contains
            a summary_id and short_description. Use expand_summary to
            reconstruct the original conversation from any summary."""
            if not self.memory_manager or not self._current_memory_id:
                return {"summaries": self.list_context_summaries()}
            summaries = self.memory_manager.load_summaries_for_thread(
                memory_id=self._current_memory_id,
                agent_id=self.agent_id,
                limit=20,
            )
            return {"summaries": summaries}

        def expand_summary(summary_id: str) -> Dict[str, Any]:
            """Expand a summary reference to its full content AND the original
            conversation messages that were compacted into it. Use this when
            you need more detail from a [Summary ID: xxx] reference in the
            context window. This performs just-in-time retrieval — the full
            data is loaded from the database only when you call this tool."""
            if not self.memory_provider:
                return {"error": "No memory provider configured."}

            # Load the summary document
            summary_doc = self.fetch_context_summary(summary_id)
            if not summary_doc:
                return {"error": f"Summary '{summary_id}' not found."}

            result: Dict[str, Any] = {
                "summary_id": summary_id,
                "summary_content": summary_doc.get("content", ""),
                "period_start": summary_doc.get("period_start"),
                "period_end": summary_doc.get("period_end"),
                "memory_units_count": summary_doc.get("memory_units_count", 0),
            }

            # Reconstruct original messages if source IDs are available
            source_ids = summary_doc.get("source_message_ids") or []
            if source_ids and self.memory_manager:
                original_msgs = self.memory_manager.get_messages_by_ids(source_ids)
                if original_msgs:
                    reconstructed = []
                    for msg in original_msgs:
                        content = msg.get("content") or msg
                        if isinstance(content, dict):
                            role = content.get("role", "unknown")
                            text = content.get("content", "")
                        else:
                            role = msg.get("role", "unknown")
                            text = str(content)
                        reconstructed.append({"role": role, "content": text})
                    result["original_messages"] = reconstructed
                    result["original_message_count"] = len(reconstructed)
                else:
                    result["original_messages"] = []
                    result["note"] = (
                        "Original messages could not be retrieved. "
                        "The summary content above is the best available context."
                    )
            else:
                result["original_messages"] = []
                result["note"] = "No source message IDs linked to this summary."

            return result

        def summarize_conversation(
            days_back: int = 1, max_memories_per_summary: int = 20
        ) -> Dict[str, Any]:
            """Summarize unsummarized conversation messages in the current thread.
            This compacts older messages into summary digests, marks the originals
            as summarized (so they won't appear in conversation history), and
            stores each summary with a [Summary ID] for later expansion.
            Use this when conversation memory is getting long or context window
            utilization is high."""
            if not self.memory_provider:
                return {
                    "ok": False,
                    "error": "No memory provider configured.",
                }
            if not self.model:
                detail = getattr(self, "_llm_init_error", None)
                return {
                    "ok": False,
                    "error": (
                        f"LLM failed to initialize — {detail}"
                        if detail
                        else "No LLM model configured."
                    ),
                }
            try:
                safe_days_back = max(1, min(int(days_back or 1), 365))
            except (TypeError, ValueError):
                safe_days_back = 1
            try:
                safe_chunk_size = max(1, min(int(max_memories_per_summary or 20), 200))
            except (TypeError, ValueError):
                safe_chunk_size = 20

            summary_ids = self.generate_summaries(
                days_back=safe_days_back,
                max_memories_per_summary=safe_chunk_size,
            )
            self._track_summary_ids(
                summary_ids,
                token_estimate=(self._last_context_window_stats or {}).get(
                    "total_tokens"
                ),
            )
            return {
                "ok": True,
                "count": len(summary_ids),
                "summary_ids": summary_ids,
                "days_back": safe_days_back,
                "max_memories_per_summary": safe_chunk_size,
                "message": (
                    f"Generated {len(summary_ids)} summaries. "
                    "Original messages are now excluded from conversation history. "
                    "Use expand_summary(summary_id) to reconstruct any summary."
                    if summary_ids
                    else "No unsummarized messages found to compact."
                ),
            }

        def retrieve_tool_log_entry(tool_log_id: str) -> Dict[str, Any]:
            """Retrieve a stored tool log entry by its ID. Use this to access
            full tool output that was offloaded from the context window."""
            if not self.memory_manager:
                return {"error": "No memory manager available."}
            result = self.memory_manager.retrieve_tool_log(tool_log_id)
            return result or {"error": f"Tool log '{tool_log_id}' not found."}

        def list_recent_tool_logs(limit: int = 10) -> Dict[str, Any]:
            """List recent tool execution logs for the current thread.

            Each entry includes a short ``args_summary`` and field-aware
            ``digest`` so the agent can pick the right ``tool_log_id`` to
            unpack without first fetching every candidate.
            """
            if not self.memory_manager or not self._current_memory_id:
                return {"error": "No memory context available.", "logs": []}
            raw_logs = self.memory_manager.list_tool_logs(
                memory_id=self._current_memory_id, limit=limit
            )
            enriched: List[Dict[str, Any]] = []
            for row in raw_logs or []:
                if not isinstance(row, dict):
                    continue
                item = dict(row)
                parsed = row.get("result")
                if isinstance(parsed, str) and parsed.strip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(parsed)
                    except Exception:
                        pass
                item["args_summary"] = _summarize_tool_args(
                    row.get("arguments"), limit=200
                )
                item["digest"] = _summarize_tool_result(parsed, preview_limit=300)
                enriched.append(item)
            return {"count": len(enriched), "logs": enriched}

        self.tool_manager.add_tool(context_window_stats_tool)
        self.tool_manager.add_tool(list_summary_registry_tool)
        self.tool_manager.add_tool(expand_summary)
        self.tool_manager.add_tool(summarize_conversation)
        self.tool_manager.add_tool(retrieve_tool_log_entry)
        self.tool_manager.add_tool(list_recent_tool_logs)
        self._context_tools_registered = True

    def _track_summary_ids(
        self, summary_ids: List[str], token_estimate: Optional[int] = None
    ) -> None:
        """Cache summary metadata for quick lookup in context-monitor tools."""
        if not summary_ids:
            return
        if not self.memory_provider or not hasattr(
            self.memory_provider, "retrieve_by_id"
        ):
            return

        for summary_id in summary_ids:
            if summary_id in self._known_summary_ids:
                continue
            try:
                summary_doc = self.memory_provider.retrieve_by_id(
                    summary_id, MemoryType.SUMMARIES
                )
            except Exception as exc:
                logger.warning(f"Unable to load summary {summary_id}: {exc}")
                summary_doc = None

            short_description = ""
            if summary_doc and summary_doc.get("content"):
                short_description = summary_doc["content"][:160]

            meta = {
                "summary_id": summary_id,
                "short_description": short_description,
            }
            if token_estimate is not None:
                meta["token_estimate"] = token_estimate

            self._summary_registry.append(meta)
            self._known_summary_ids.add(summary_id)

    def _maybe_generate_context_summary(self):
        """Automatically summarize when context window nears limits."""
        if not self.memory_provider or not hasattr(
            self.memory_provider, "retrieve_by_id"
        ):
            return
        if MemoryType.SUMMARIES not in self.active_memory_types:
            return
        stats = self._last_context_window_stats or {}
        percentage_used = stats.get("percentage_used")
        if percentage_used is None or percentage_used < self._context_summary_trigger:
            return

        import time

        current_time = time.time()
        if current_time - self._last_summary_timestamp < self._context_summary_cooldown:
            return

        summary_ids = self.generate_summaries(days_back=1, max_memories_per_summary=20)
        if not summary_ids:
            return

        self._last_summary_timestamp = current_time
        self._track_summary_ids(summary_ids, token_estimate=stats.get("total_tokens"))

    def list_context_summaries(self) -> List[Dict[str, Any]]:
        """Return summary registry entries."""
        return list(self._summary_registry)

    def fetch_context_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored summary document by ID."""
        if not self.memory_provider or not hasattr(
            self.memory_provider, "retrieve_by_id"
        ):
            return None
        try:
            return self.memory_provider.retrieve_by_id(summary_id, MemoryType.SUMMARIES)
        except Exception as exc:
            logger.warning("Failed to fetch summary %s: %s", summary_id, exc)
            return None

    def _initialize_tools(self, tools):
        """Initialize tools using the tool manager."""
        if hasattr(tools, "__iter__") and not isinstance(tools, str):
            # List of tools
            for tool in tools:
                self.tool_manager.add_tool(tool)
        else:
            # Single tool or toolbox
            self.tool_manager.add_tool(tools)

    def _normalize_skill_paths(
        self, skill_paths: Optional[Union[str, List[str]]]
    ) -> List[str]:
        """Normalize skill path inputs into a stable list."""
        if not skill_paths:
            return []
        if isinstance(skill_paths, str):
            raw_values = [skill_paths]
        elif isinstance(skill_paths, list):
            raw_values = skill_paths
        else:
            return []

        normalized: List[str] = []
        seen: Set[str] = set()
        for value in raw_values:
            path_value = str(value or "").strip()
            if not path_value or path_value in seen:
                continue
            seen.add(path_value)
            normalized.append(path_value)
        return normalized

    def _normalize_mcp_servers(
        self, mcp_servers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Normalize and validate MCP server entries."""
        if not mcp_servers:
            return []

        raw_entries: List[Any]
        if isinstance(mcp_servers, dict):
            raw_entries = [mcp_servers]
        elif isinstance(mcp_servers, list):
            raw_entries = mcp_servers
        else:
            return []

        normalized: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()

        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue

            name = str(entry.get("name", "")).strip()
            if not name or name in seen_names:
                continue

            transport = str(entry.get("transport", "stdio")).strip().lower() or "stdio"
            if transport not in {"stdio", "http"}:
                transport = "stdio"

            server: Dict[str, Any] = {"name": name, "transport": transport}
            timeout = entry.get("timeout")
            try:
                timeout_value = int(timeout) if timeout is not None else 30
            except Exception:
                timeout_value = 30
            server["timeout"] = max(1, min(timeout_value, 300))

            if transport == "stdio":
                command = str(entry.get("command", "")).strip()
                if not command:
                    continue
                server["command"] = command
                args = entry.get("args", [])
                if not isinstance(args, list):
                    args = []
                server["args"] = [str(arg) for arg in args if str(arg).strip()]
                env = entry.get("env", {})
                if not isinstance(env, dict):
                    env = {}
                server["env"] = {
                    str(key): str(value)
                    for key, value in env.items()
                    if str(key).strip()
                }
                cwd = str(entry.get("cwd", "")).strip()
                if cwd:
                    server["cwd"] = cwd
            else:
                url = str(entry.get("url", "")).strip()
                if not url:
                    continue
                server["url"] = url
                headers = entry.get("headers", {})
                if isinstance(headers, dict) and headers:
                    server["headers"] = {
                        str(key): str(value)
                        for key, value in headers.items()
                        if str(key).strip()
                    }

            normalized.append(server)
            seen_names.add(name)

        return normalized

    def _normalize_skills_marketplace_provider_name(self, value: Any) -> str:
        """Normalize skills marketplace provider values to a stable lowercase name."""
        if isinstance(value, dict):
            value = value.get("provider") or value.get("name")
        return str(value or "").strip().lower()

    def _build_skills_marketplace_config(
        self,
        provider_name: Optional[str],
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build provider config with API key fallback from environment."""
        normalized_provider = self._normalize_skills_marketplace_provider_name(
            provider_name
        )
        if normalized_provider not in ("skillsmp", "vercel"):
            return {}

        config: Dict[str, Any] = {}
        if isinstance(base_config, dict):
            for key, value in base_config.items():
                if key == "provider":
                    continue
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                config[key] = value

        if normalized_provider == "skillsmp":
            if "api_key" not in config:
                default_key = os.getenv("SKILLSMP_API_KEY", "").strip()
                if default_key:
                    config["api_key"] = default_key
            if "base_url" not in config:
                config["base_url"] = "https://skillsmp.com"
        elif normalized_provider == "vercel":
            if "github_token" not in config:
                default_token = os.getenv("GITHUB_TOKEN", "").strip()
                if default_token:
                    config["github_token"] = default_token
            if not config.get("github_token"):
                from ..vercel_skills.provider import MISSING_GITHUB_TOKEN_MESSAGE

                logger.warning(MISSING_GITHUB_TOKEN_MESSAGE)

        return config

    def _resolve_skill_path(self, raw_path: str) -> Path:
        """Resolve skill file paths relative to current working directory."""
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path(os.getcwd()) / path
        return path.resolve()

    def _extract_skill_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract fenced code blocks from a skill document."""
        code_blocks: List[Dict[str, Any]] = []
        pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
        for index, match in enumerate(pattern.finditer(content)):
            language = (match.group(1) or "python").strip().lower() or "python"
            code = match.group(2).strip()
            if not code:
                continue
            code_blocks.append(
                {
                    "snippet_id": f"snippet_{index + 1}",
                    "language": language,
                    "code": code,
                }
            )
        return code_blocks

    def _extract_skill_script_paths(self, content: str, base_dir: Path) -> List[str]:
        """Extract local script references from markdown content."""
        script_paths: List[str] = []
        seen: Set[str] = set()
        pattern = re.compile(
            r"(?:(?:\(|\s)|^)(\./[\w./-]+\.(?:py|sh|bash|js|ts))(?:\)|\s|$)"
        )
        for match in pattern.findall(content):
            raw_script_path = match.strip()
            if raw_script_path in seen:
                continue
            candidate = (base_dir / raw_script_path).resolve()
            try:
                candidate.relative_to(base_dir.resolve())
            except Exception:
                continue
            if not candidate.exists() or not candidate.is_file():
                continue
            seen.add(raw_script_path)
            script_paths.append(raw_script_path)
        return script_paths

    def _extract_skill_description(self, content: str) -> str:
        """Extract a short description from skill markdown content."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("#"):
                continue
            return line[:240]
        return "Skill file"

    def _load_skills(self) -> List[Dict[str, Any]]:
        """Load configured skill markdown files from disk."""
        loaded_skills: List[Dict[str, Any]] = []
        for raw_path in self.skill_paths:
            try:
                resolved_path = self._resolve_skill_path(raw_path)
                if not resolved_path.exists() or not resolved_path.is_file():
                    logger.warning("Skill file not found: %s", raw_path)
                    continue

                content = resolved_path.read_text(encoding="utf-8")
                heading = re.search(r"^\s*#\s+(.+)$", content, re.MULTILINE)
                default_name = resolved_path.stem.replace(".skills", "").strip()
                skill_name = (
                    heading.group(1).strip() if heading else default_name or raw_path
                )
                code_blocks = self._extract_skill_code_blocks(content)
                skill_doc = {
                    "name": skill_name,
                    "path": str(resolved_path),
                    "base_dir": str(resolved_path.parent),
                    "description": self._extract_skill_description(content),
                    "content": content,
                    "code_blocks": code_blocks,
                    "script_paths": self._extract_skill_script_paths(
                        content, resolved_path.parent
                    ),
                }
                loaded_skills.append(skill_doc)
            except Exception as exc:
                logger.warning("Failed to load skill '%s': %s", raw_path, exc)
        return loaded_skills

    def _get_skill(self, skill_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve a skill by name or file path."""
        if not self.skills:
            return None
        if not skill_name:
            return self.skills[0]
        for skill in self.skills:
            if skill_name in {
                skill.get("name"),
                skill.get("path"),
                Path(skill.get("path", "")).name,
            }:
                return skill
        return None

    def _execute_code_in_sandbox(
        self,
        code: str,
        language: str = "python",
        timeout: int = 60,
        envs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute code through the configured sandbox and return parsed output."""
        if not self.has_sandbox():
            return {
                "ok": False,
                "error": "Sandbox provider is required for skill/MCP code execution.",
            }
        try:
            result = self.sandbox_manager.execute_code(
                code=code, language=language, timeout=timeout, envs=envs
            )
            payload = json.loads(result.to_json())
            payload["ok"] = payload.get("error") in (None, "")
            return payload
        except Exception as exc:
            logger.error("Sandbox execution failed: %s", exc)
            return {"ok": False, "error": str(exc)}

    def _register_skill_tools(self):
        """Register skill discovery and sandbox execution tools."""
        if self._skill_tools_registered or not self.tool_manager:
            return

        def list_skills() -> Dict[str, Any]:
            """List loaded skill files and executable snippets."""
            return {
                "skills": [
                    {
                        "name": skill.get("name"),
                        "path": skill.get("path"),
                        "description": skill.get("description"),
                        "snippet_count": len(skill.get("code_blocks", [])),
                        "script_paths": skill.get("script_paths", []),
                    }
                    for skill in self.skills
                ]
            }

        def read_skill(skill_name: str) -> Dict[str, Any]:
            """Read full markdown content for a loaded skill file."""
            skill = self._get_skill(skill_name)
            if not skill:
                return {"error": f"Skill '{skill_name}' not found."}
            return {
                "name": skill.get("name"),
                "path": skill.get("path"),
                "content": skill.get("content", ""),
                "code_blocks": [
                    {
                        "snippet_id": block.get("snippet_id"),
                        "language": block.get("language"),
                    }
                    for block in skill.get("code_blocks", [])
                ],
            }

        def run_skill_code(
            skill_name: str = "",
            snippet_id: str = "",
            code: str = "",
            language: str = "python",
        ) -> Dict[str, Any]:
            """Run skill code in the configured sandbox (never on host machine)."""
            target_code = (code or "").strip()
            target_language = (language or "python").strip() or "python"

            if not target_code:
                skill = self._get_skill(skill_name or None)
                if not skill:
                    return {"error": "No skill found to execute."}
                snippets = skill.get("code_blocks", [])
                if not snippets:
                    return {
                        "error": f"Skill '{skill.get('name')}' has no fenced code blocks."
                    }
                selected = None
                if snippet_id:
                    selected = next(
                        (
                            block
                            for block in snippets
                            if block.get("snippet_id") == snippet_id
                        ),
                        None,
                    )
                if not selected:
                    selected = snippets[0]
                target_code = selected.get("code", "")
                target_language = selected.get("language") or target_language

            result = self._execute_code_in_sandbox(
                code=target_code, language=target_language
            )
            return {
                "skill_name": skill_name or None,
                "snippet_id": snippet_id or None,
                "language": target_language,
                "execution": result,
            }

        def run_skill_script(
            skill_name: str,
            script_path: str,
            language: str = "",
        ) -> Dict[str, Any]:
            """Run a local script referenced by a skill inside the sandbox."""
            skill = self._get_skill(skill_name)
            if not skill:
                return {"error": f"Skill '{skill_name}' not found."}

            base_dir = Path(skill.get("base_dir", "")).resolve()
            candidate = (base_dir / script_path).resolve()
            try:
                candidate.relative_to(base_dir)
            except Exception:
                return {"error": "Script path must stay inside the skill directory."}
            if not candidate.exists() or not candidate.is_file():
                return {"error": f"Script not found: {script_path}"}

            try:
                script_code = candidate.read_text(encoding="utf-8")
            except Exception as exc:
                return {"error": f"Failed to read script: {exc}"}

            ext = candidate.suffix.lower()
            detected_language = (language or "").strip() or (
                "python"
                if ext == ".py"
                else "bash"
                if ext in {".sh", ".bash"}
                else "javascript"
            )
            execution = self._execute_code_in_sandbox(
                code=script_code, language=detected_language
            )
            return {
                "skill_name": skill.get("name"),
                "script_path": str(candidate),
                "language": detected_language,
                "execution": execution,
            }

        list_skills.__name__ = "list_skills"
        read_skill.__name__ = "read_skill"
        run_skill_code.__name__ = "run_skill_code"
        run_skill_script.__name__ = "run_skill_script"

        self.tool_manager.add_tool(list_skills)
        self.tool_manager.add_tool(read_skill)
        self.tool_manager.add_tool(run_skill_code)
        self.tool_manager.add_tool(run_skill_script)
        self._skill_tools_registered = True

    def _get_mcp_server(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Find an MCP server by configured name."""
        for server in self.mcp_servers:
            if server.get("name") == server_name:
                return server
        return None

    def _run_mcp_request_in_sandbox(
        self, server_name: str, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call an MCP request inside sandboxed Python."""
        server = self._get_mcp_server(server_name)
        if not server:
            return {"ok": False, "error": f"Unknown MCP server '{server_name}'."}
        if server.get("transport") != "stdio":
            return {
                "ok": False,
                "error": (
                    f"MCP server '{server_name}' uses unsupported transport "
                    f"'{server.get('transport')}'."
                ),
            }

        payload = {
            "server": server,
            "method": method,
            "params": params or {},
        }
        escaped_payload = json.dumps(payload)
        sandbox_code = f"""
import json
import os
import select
import subprocess
import time

payload = {escaped_payload}
server = payload["server"]
method = payload["method"]
params = payload.get("params") or {{}}
timeout = int(server.get("timeout", 30))
command = [server.get("command")] + [str(v) for v in server.get("args", [])]
env = {{k: str(v) for k, v in (server.get("env") or {{}}).items()}}
cwd = server.get("cwd") or None
runtime_env = os.environ.copy()
runtime_env.update(env)

def _start_process():
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=runtime_env,
    )

def _read_jsonline(proc, remaining):
    end = time.time() + remaining
    while time.time() < end:
        left = max(0.1, end - time.time())
        ready, _, _ = select.select([proc.stdout], [], [], left)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="ignore").strip()
        if not text:
            continue
        try:
            return json.loads(text)
        except Exception:
            continue
    raise TimeoutError("Timeout waiting for JSON line response")

def _send_jsonline(proc, message):
    body = (json.dumps(message) + "\\n").encode("utf-8")
    proc.stdin.write(body)
    proc.stdin.flush()

def _read_content_length(proc, remaining):
    end = time.time() + remaining
    headers = {{}}
    while time.time() < end:
        left = max(0.1, end - time.time())
        ready, _, _ = select.select([proc.stdout], [], [], left)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="ignore").strip()
        if not text:
            break
        if ":" in text:
            key, value = text.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    if length <= 0:
        raise RuntimeError("Missing Content-Length header in MCP response")

    data = b""
    while len(data) < length and time.time() < end:
        left = max(0.1, end - time.time())
        ready, _, _ = select.select([proc.stdout], [], [], left)
        if not ready:
            continue
        chunk = proc.stdout.read(length - len(data))
        if not chunk:
            break
        data += chunk

    if len(data) < length:
        raise TimeoutError("Incomplete MCP content-length response")
    return json.loads(data.decode("utf-8", errors="ignore"))

def _send_content_length(proc, message):
    raw = json.dumps(message).encode("utf-8")
    header = f"Content-Length: {{len(raw)}}\\r\\n\\r\\n".encode("utf-8")
    proc.stdin.write(header + raw)
    proc.stdin.flush()

def _run_session(mode):
    send = _send_jsonline if mode == "jsonline" else _send_content_length
    read = _read_jsonline if mode == "jsonline" else _read_content_length
    proc = _start_process()
    try:
        init_request = {{
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {{
                "protocolVersion": "2025-06-18",
                "capabilities": {{}},
                "clientInfo": {{"name": "memorizz", "version": "0.0.39"}},
            }},
        }}
        send(proc, init_request)

        init_response = read(proc, timeout)
        if init_response.get("id") != "init-1":
            raise RuntimeError("Unexpected MCP initialize response")

        send(
            proc,
            {{
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {{}},
            }},
        )

        request = {{
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": method,
            "params": params,
        }}
        send(proc, request)
        response = read(proc, timeout)
        if response.get("id") != "req-1":
            raise RuntimeError("Unexpected MCP method response")
        return {{"ok": True, "mode": mode, "response": response}}
    finally:
        try:
            proc.terminate()
        except Exception:
            pass

attempt_errors = []
for protocol_mode in ("jsonline", "content-length"):
    try:
        result = _run_session(protocol_mode)
        print(json.dumps(result))
        raise SystemExit(0)
    except Exception as exc:
        attempt_errors.append({{"mode": protocol_mode, "error": str(exc)}})

print(json.dumps({{"ok": False, "errors": attempt_errors}}))
"""
        execution = self._execute_code_in_sandbox(sandbox_code, language="python")
        stdout_lines = execution.get("stdout") or []
        for line in reversed(stdout_lines):
            line = str(line).strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {
            "ok": False,
            "error": "No MCP response returned from sandbox execution.",
            "execution": execution,
        }

    def _register_mcp_tools(self):
        """Register MCP bridge tools."""
        if self._mcp_tools_registered or not self.tool_manager:
            return

        def list_mcp_servers() -> Dict[str, Any]:
            """List configured MCP servers."""
            return {
                "servers": [
                    {
                        "name": server.get("name"),
                        "transport": server.get("transport", "stdio"),
                        "command": server.get("command"),
                        "args": server.get("args", []),
                        "url": server.get("url"),
                    }
                    for server in self.mcp_servers
                ]
            }

        def mcp_list_tools(server_name: str) -> Dict[str, Any]:
            """List tools available from a configured MCP server."""
            return self._run_mcp_request_in_sandbox(
                server_name=server_name,
                method="tools/list",
                params={},
            )

        def mcp_call_tool(
            server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Call a tool on a configured MCP server from the sandbox."""
            return self._run_mcp_request_in_sandbox(
                server_name=server_name,
                method="tools/call",
                params={"name": tool_name, "arguments": arguments or {}},
            )

        list_mcp_servers.__name__ = "list_mcp_servers"
        mcp_list_tools.__name__ = "mcp_list_tools"
        mcp_call_tool.__name__ = "mcp_call_tool"

        self.tool_manager.add_tool(list_mcp_servers)
        self.tool_manager.add_tool(mcp_list_tools)
        self.tool_manager.add_tool(mcp_call_tool)
        self._mcp_tools_registered = True

    def with_entity_memory(self, enabled: bool = True):
        """
        Enable or disable entity memory support at runtime.

        Args:
            enabled: True to expose entity memory tools/context, False to disable.

        Returns:
            Self for chaining.
        """
        manager_ready = (
            self.entity_memory_manager and self.entity_memory_manager.is_enabled()
        )

        if enabled:
            if not manager_ready:
                logger.warning(
                    "Cannot enable entity memory: memory provider does not support it."
                )
                self._entity_memory_enabled = False
                return self

            if not self._entity_memory_enabled:
                self._entity_memory_enabled = True
                self._register_entity_memory_tools()
        else:
            if self._entity_memory_enabled:
                self._entity_memory_enabled = False
                self._unregister_entity_memory_tools()

        return self

    def with_internet_access_provider(
        self, provider: Optional["InternetAccessProvider"]
    ):
        """
        Attach or detach an internet access provider.

        Args:
            provider: InternetAccessProvider instance or None to disable.

        Returns:
            Self for chaining.
        """
        if not self.internet_access_manager:
            self.internet_access_manager = InternetAccessManager(provider)
        else:
            self.internet_access_manager.set_provider(provider)

        # Reset failure tracking whenever provider changes
        self._internet_access_failure_count = 0
        self._internet_access_disabled_reason = None

        if provider:
            self._register_internet_access_tools()
        else:
            self._unregister_internet_access_tools()

        return self

    def with_skills_marketplace_provider(
        self,
        provider: Optional[Union[str, Dict[str, Any]]],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Attach or detach a skills marketplace provider.

        Args:
            provider: Provider name (currently ``"skillsmp"``), provider config dict,
                or None to disable.
            config: Optional provider config. Used only when ``provider`` is a string.

        Returns:
            Self for chaining.
        """
        provider_name = ""
        provider_config: Dict[str, Any] = {}

        if isinstance(provider, dict):
            provider_name = self._normalize_skills_marketplace_provider_name(provider)
            provider_config = {k: v for k, v in provider.items() if k != "provider"}
            if isinstance(config, dict):
                provider_config.update(config)
        else:
            provider_name = self._normalize_skills_marketplace_provider_name(provider)
            if isinstance(config, dict):
                provider_config = dict(config)

        if not provider_name:
            self._skills_marketplace_provider_name = None
            self._skills_marketplace_config = {}
            self._unregister_skills_marketplace_tools()
            return self

        if provider_name not in ("skillsmp", "vercel"):
            raise ValueError(
                f"Unknown skills marketplace provider '{provider_name}'. "
                "Supported providers: skillsmp, vercel."
            )

        self._skills_marketplace_provider_name = provider_name
        self._skills_marketplace_config = self._build_skills_marketplace_config(
            provider_name,
            provider_config,
        )
        self._register_skills_marketplace_tools()
        return self

    def has_skills_marketplace(self) -> bool:
        """Return True if a skills marketplace provider is configured."""
        return bool(self._skills_marketplace_provider_name)

    def get_skills_marketplace_provider_name(self) -> Optional[str]:
        """Return the active skills marketplace provider name, if any."""
        return self._skills_marketplace_provider_name

    def get_skills_marketplace_config(self) -> Optional[Dict[str, Any]]:
        """Return active skills marketplace config if configured."""
        if not self.has_skills_marketplace():
            return None
        return dict(self._skills_marketplace_config)

    def has_internet_access(self) -> bool:
        """Return True if an internet provider is configured."""
        return bool(
            self.internet_access_manager
            and self.internet_access_manager.is_enabled()
            and not self._internet_access_disabled_reason
        )

    def get_internet_access_provider_name(self) -> Optional[str]:
        """Return the active internet provider name, if any."""
        if not self.internet_access_manager:
            return None
        return self.internet_access_manager.get_provider_name()

    def search_internet(
        self, query: str, max_results: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Expose direct search helper for callers."""
        if not self.has_internet_access():
            reason = (
                self._internet_access_disabled_reason
                or "No internet access provider configured"
            )
            raise ValueError(reason)
        return self.internet_access_manager.search(
            query=query, max_results=max_results, **kwargs
        )

    def fetch_url(self, url: str, **kwargs) -> Dict[str, Any]:
        """Expose direct fetch helper."""
        if not self.has_internet_access():
            reason = (
                self._internet_access_disabled_reason
                or "No internet access provider configured"
            )
            raise ValueError(reason)
        return self.internet_access_manager.fetch_url(url=url, **kwargs)

    # --- Automations (durable scheduling) ---

    def has_automations(self) -> bool:
        """Return True when automation storage + tools are available for this agent."""
        return bool(
            getattr(self, "automation_manager", None)
            and self.automation_manager.is_enabled()
        )

    def _require_automations(self) -> "AutomationManager":
        """Internal: raise ValueError if automations unavailable, else return manager."""
        if not self.has_automations():
            raise ValueError(
                "Automations are not available. Requires an Oracle memory provider "
                "with automations_enabled=True."
            )
        return self.automation_manager

    def create_automation(
        self,
        name: str,
        schedule_type: str,
        *,
        query_template: str,
        cron_expr: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        timezone: Optional[str] = None,
        memory_id: Optional[str] = None,
        whatsapp_to: Optional[List[str]] = None,
        max_run_seconds: int = 900,
        retry_max_attempts: int = 1,
        retry_backoff_seconds: int = 60,
        misfire_policy: str = "skip",
    ) -> "AutomationJob":
        """Create a scheduled automation job for this agent.

        Args:
            name: Human-readable job name (e.g. "Daily briefing").
            schedule_type: One of "cron", "interval", or "one_shot".
            query_template: The prompt template to run. Supports placeholders:
                {today_iso}, {scheduled_for_iso}, {now_utc_iso}, {timezone}.
            cron_expr: 5-field cron expression (required when schedule_type="cron").
            interval_seconds: Seconds between runs (required when schedule_type="interval").
            timezone: IANA timezone name. Falls back to agent's default_timezone
                or MEMORIZZ_DEFAULT_TIMEZONE env var.
            memory_id: Memory ID for the automation's conversation thread.
                Auto-generated if not provided.
            whatsapp_to: List of WhatsApp recipient numbers. When provided,
                results are delivered via WhatsApp/Twilio.
            max_run_seconds: Maximum execution time per run (default 900).
            retry_max_attempts: Number of retry attempts on failure (default 1).
            retry_backoff_seconds: Backoff between retries (default 60).
            misfire_policy: What to do on missed runs: "skip" or "run".

        Returns:
            The created AutomationJob instance.

        Raises:
            ValueError: If automations are unavailable, timezone is invalid,
                or required parameters are missing.

        Example::

            job = agent.create_automation(
                name="Morning News Digest",
                schedule_type="cron",
                cron_expr="0 8 * * *",
                timezone="America/New_York",
                query_template="Give me today's top 5 tech news for {today_iso}",
            )
            print(f"Created job {job.job_id}, next run: {job.next_run_at}")
        """
        mgr = self._require_automations()
        store = mgr.store

        from ..automation.models import AutomationJob as _AutomationJob
        from ..automation.schedule import (
            compute_next_run_at,
            utcnow,
            validate_timezone_name,
        )

        # Resolve timezone: explicit > agent default > env var
        tz_name = (timezone or "").strip()
        if not tz_name:
            tz_name = (getattr(self, "default_timezone", None) or "").strip()
        if not tz_name:
            tz_name = os.environ.get("MEMORIZZ_DEFAULT_TIMEZONE", "").strip()
        if not tz_name:
            raise ValueError(
                "timezone is required. Pass it explicitly, set default_timezone "
                "on the agent, or set the MEMORIZZ_DEFAULT_TIMEZONE env var."
            )
        validate_timezone_name(tz_name)

        if not name or not name.strip():
            raise ValueError("name is required")
        if not query_template or not query_template.strip():
            raise ValueError("query_template is required")

        now_utc = utcnow()
        next_run = compute_next_run_at(
            schedule_type=schedule_type,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            tz_name=tz_name,
            after_utc=now_utc,
        )

        resolved_memory_id = (memory_id or "").strip() or str(uuid.uuid4())

        # Normalize WhatsApp recipients
        to_list: List[str] = []
        if whatsapp_to:
            seen: Set[str] = set()
            items = whatsapp_to if isinstance(whatsapp_to, list) else [whatsapp_to]
            for item in items:
                text = str(item or "").strip()
                if not text:
                    continue
                val = (
                    text if text.lower().startswith("whatsapp:") else f"whatsapp:{text}"
                )
                if val not in seen:
                    seen.add(val)
                    to_list.append(val)

        job = _AutomationJob(
            job_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            name=name.strip(),
            enabled=True,
            schedule_type=schedule_type,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            timezone=tz_name,
            start_at=now_utc,
            next_run_at=next_run,
            misfire_policy=misfire_policy,
            max_run_seconds=max_run_seconds,
            retry_max_attempts=retry_max_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            action_type="agent_query",
            action_config={
                "query_template": query_template.strip(),
                "memory_id": resolved_memory_id,
            },
            delivery_type="whatsapp_twilio" if to_list else "in_chat",
            delivery_config={"whatsapp_to": to_list} if to_list else {},
        )

        return store.create_job(job)

    def list_automations(
        self, *, enabled: Optional[bool] = None
    ) -> List["AutomationJob"]:
        """List automation jobs belonging to this agent.

        Args:
            enabled: Filter by enabled state. None returns all jobs.

        Returns:
            List of AutomationJob instances.
        """
        mgr = self._require_automations()
        return mgr.store.list_jobs(agent_id=self.agent_id, enabled=enabled)

    def get_automation(self, job_id: str) -> Optional["AutomationJob"]:
        """Get a single automation job by ID.

        Args:
            job_id: The job identifier.

        Returns:
            AutomationJob if found, None otherwise.
        """
        mgr = self._require_automations()
        return mgr.store.get_job(job_id)

    def pause_automation(self, job_id: str) -> "AutomationJob":
        """Pause a running automation job.

        Args:
            job_id: The job identifier to pause.

        Returns:
            The updated AutomationJob with enabled=False.
        """
        mgr = self._require_automations()
        return mgr.store.pause_job(job_id)

    def resume_automation(self, job_id: str) -> "AutomationJob":
        """Resume a paused automation job.

        Args:
            job_id: The job identifier to resume.

        Returns:
            The updated AutomationJob with enabled=True.
        """
        mgr = self._require_automations()
        return mgr.store.resume_job(job_id)

    def delete_automation(self, job_id: str) -> bool:
        """Permanently delete an automation job.

        Args:
            job_id: The job identifier to delete.

        Returns:
            True if the job was deleted, False if not found.
        """
        mgr = self._require_automations()
        return mgr.store.delete_job(job_id)

    def trigger_automation(self, job_id: str) -> "AutomationJob":
        """Trigger an immediate run of an automation job.

        Sets next_run_at to now and enables the job so the worker picks
        it up on its next poll cycle.

        Args:
            job_id: The job identifier to trigger.

        Returns:
            The updated AutomationJob.
        """
        mgr = self._require_automations()
        from ..automation.schedule import utcnow

        now = utcnow()
        return mgr.store.update_job(
            job_id,
            {
                "next_run_at": now,
                "enabled": True,
                "locked_by": None,
                "lock_expires_at": None,
            },
        )

    def list_automation_runs(
        self, job_id: str, *, limit: int = 50
    ) -> List["AutomationRun"]:
        """List execution history for a specific automation job.

        Args:
            job_id: The job identifier.
            limit: Maximum number of runs to return (default 50).

        Returns:
            List of AutomationRun instances, most recent first.
        """
        mgr = self._require_automations()
        return mgr.store.list_runs(job_id, limit=limit)

    # --- Sandbox access ---

    def with_sandbox_provider(
        self,
        provider: Optional[Union[str, Dict[str, Any], "SandboxProvider"]],
    ):
        """
        Attach or detach a sandbox code execution provider.

        Args:
            provider: A provider name string (``"e2b"``, ``"daytona"``,
                ``"graalpy"``), a config dict, a SandboxProvider instance,
                or None to disable.

        Returns:
            Self for chaining.
        """
        if provider is not None:
            if not self.sandbox_manager:
                self.sandbox_manager = SandboxManager.from_config(provider)
            else:
                # Re-create from new config
                new_manager = SandboxManager.from_config(provider)
                self.sandbox_manager.set_provider(new_manager.provider)
            self._register_sandbox_tools()
        else:
            self._unregister_sandbox_tools()
            if self.sandbox_manager:
                self.sandbox_manager.close()
                self.sandbox_manager = None

        return self

    def has_sandbox(self) -> bool:
        """Return True if a sandbox provider is configured."""
        return bool(self.sandbox_manager and self.sandbox_manager.is_enabled())

    def get_sandbox_provider_name(self) -> Optional[str]:
        """Return the active sandbox provider name, if any."""
        if not self.sandbox_manager:
            return None
        return self.sandbox_manager.get_provider_name()

    def execute_code(self, code: str, language: str = "python", **kwargs) -> str:
        """Expose direct code execution helper for callers."""
        if not self.has_sandbox():
            raise ValueError("No sandbox provider configured")
        result = self.sandbox_manager.execute_code(
            code=code, language=language, **kwargs
        )
        return result.to_json()

    def _register_sandbox_tools(self):
        """Register tools that expose sandbox code execution."""
        if not (
            self.tool_manager
            and self.sandbox_manager
            and self.sandbox_manager.is_enabled()
        ):
            return

        if self._sandbox_tools_registered:
            return

        for tool_func in self.sandbox_manager.get_tools():
            self.tool_manager.add_tool(tool_func)

        self._sandbox_tools_registered = True
        logger.info(
            "Registered sandbox tools (provider: %s)",
            self.sandbox_manager.get_provider_name(),
        )

    def _unregister_sandbox_tools(self):
        """Remove sandbox tools from the tool manager."""
        if not self._sandbox_tools_registered or not self.tool_manager:
            return

        for tool_name in self._sandbox_tool_names:
            self.tool_manager.remove_tool(tool_name)

        self._sandbox_tools_registered = False
        logger.info("Unregistered sandbox tools")

    # --- Self-awareness access ---

    def with_self_aware(
        self, enabled: bool = True, config: Optional[Dict[str, Any]] = None
    ):
        """
        Enable or disable self-awareness host codebase tools.

        Args:
            enabled: Whether self-aware tools should be exposed to the agent.
            config: Optional self-aware policy/config override.

        Returns:
            Self for chaining.
        """
        if (
            not hasattr(self, "self_awareness_manager")
            or not self.self_awareness_manager
        ):
            self.self_awareness_manager = SelfAwarenessManager(config=config)
        elif isinstance(config, dict):
            self.self_awareness_manager.configure(config)
        elif config is not None:
            logger.warning(
                "Ignoring non-dict self_aware_config payload: %s",
                type(config).__name__,
            )

        self.self_aware = bool(enabled)
        self.self_aware_config = self.self_awareness_manager.get_config()

        if self.self_aware:
            self._register_self_aware_tools()
        else:
            self._unregister_self_aware_tools()

        return self

    def has_self_awareness(self) -> bool:
        """Return True when self-aware tooling is enabled."""
        return bool(
            self.self_aware
            and hasattr(self, "self_awareness_manager")
            and self.self_awareness_manager
        )

    def get_self_aware_config(self) -> Dict[str, Any]:
        """Return normalized self-aware configuration."""
        if (
            not hasattr(self, "self_awareness_manager")
            or not self.self_awareness_manager
        ):
            return {}
        return self.self_awareness_manager.get_config()

    def _register_self_aware_tools(self):
        """Register self-aware host codebase tools."""
        if not self.tool_manager or not self.self_awareness_manager:
            return
        if self._self_aware_tools_registered:
            return

        def self_aware_list_roots() -> Dict[str, Any]:
            """List configured root paths and policy flags for self-awareness."""
            try:
                return self.self_awareness_manager.list_roots()
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_list_files(
            path: str = ".",
            recursive: bool = False,
            include_hidden: bool = False,
            max_entries: int = 500,
        ) -> Dict[str, Any]:
            """List files/directories under allowed self-aware roots."""
            try:
                return self.self_awareness_manager.list_files(
                    path=path,
                    recursive=recursive,
                    include_hidden=include_hidden,
                    max_entries=max_entries,
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_read_file(
            path: str,
            start_line: int = 1,
            end_line: int = 400,
            max_chars: int = 30000,
        ) -> Dict[str, Any]:
            """Read a file under allowed roots with bounds."""
            try:
                return self.self_awareness_manager.read_file(
                    path=path,
                    start_line=start_line,
                    end_line=end_line,
                    max_chars=max_chars,
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_search_files(
            pattern: str,
            path: str = ".",
            glob: str = "",
            max_results: int = 200,
        ) -> Dict[str, Any]:
            """Search files under allowed roots for a text pattern."""
            try:
                return self.self_awareness_manager.search_files(
                    pattern=pattern,
                    path=path,
                    glob=glob,
                    max_results=max_results,
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_write_file(
            path: str, content: str, mode: str = "overwrite"
        ) -> Dict[str, Any]:
            """Write a file under allowed roots when write policy is enabled."""
            try:
                return self.self_awareness_manager.write_file(
                    path=path, content=content, mode=mode
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_delete_path(
            path: str, recursive: bool = False, force: bool = False
        ) -> Dict[str, Any]:
            """Delete a file/path under allowed roots when delete policy is enabled."""
            try:
                return self.self_awareness_manager.delete_path(
                    path=path, recursive=recursive, force=force
                )
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def self_aware_run_command(command: str, cwd: str = ".") -> Dict[str, Any]:
            """Run a guarded allowlisted host command under allowed roots."""
            try:
                return self.self_awareness_manager.run_command(command=command, cwd=cwd)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        self_aware_list_roots.__name__ = "self_aware_list_roots"
        self_aware_list_files.__name__ = "self_aware_list_files"
        self_aware_read_file.__name__ = "self_aware_read_file"
        self_aware_search_files.__name__ = "self_aware_search_files"
        self_aware_write_file.__name__ = "self_aware_write_file"
        self_aware_delete_path.__name__ = "self_aware_delete_path"
        self_aware_run_command.__name__ = "self_aware_run_command"

        self.tool_manager.add_tool(self_aware_list_roots)
        self.tool_manager.add_tool(self_aware_list_files)
        self.tool_manager.add_tool(self_aware_read_file)
        self.tool_manager.add_tool(self_aware_search_files)
        self.tool_manager.add_tool(self_aware_write_file)
        self.tool_manager.add_tool(self_aware_delete_path)
        self.tool_manager.add_tool(self_aware_run_command)
        self._self_aware_tools_registered = True
        logger.info("Registered self-awareness tools")

    def _unregister_self_aware_tools(self):
        """Remove self-aware tools from the tool manager."""
        if not self._self_aware_tools_registered or not self.tool_manager:
            return

        for tool_name in self._self_aware_tool_names:
            self.tool_manager.remove_tool(tool_name)
        self._self_aware_tools_registered = False
        logger.info("Unregistered self-awareness tools")

    def _resolve_execution_state(
        self, memory_id: Optional[str], thread_id: Optional[str]
    ) -> Tuple[str, str]:
        """Resolve active memory/thread IDs and keep thread state isolated."""
        requested_memory_id = str(memory_id).strip() if memory_id else ""

        if requested_memory_id:
            resolved_memory_id = requested_memory_id
        elif self._current_memory_id:
            resolved_memory_id = self._current_memory_id
        elif self.memory_ids:
            resolved_memory_id = str(self.memory_ids[0]).strip()
        else:
            resolved_memory_id = str(uuid.uuid4())

        self._current_memory_id = resolved_memory_id

        if resolved_memory_id and resolved_memory_id not in self.memory_ids:
            self.memory_ids.append(resolved_memory_id)

        requested_thread_id = str(thread_id).strip() if thread_id else ""
        if requested_thread_id:
            resolved_thread_id = requested_thread_id
        else:
            resolved_thread_id = self._thread_ids_by_memory.get(resolved_memory_id)
            if not resolved_thread_id:
                resolved_thread_id = str(uuid.uuid4())
                logger.debug("Started new thread: %s", resolved_thread_id)

        self._thread_ids_by_memory[resolved_memory_id] = resolved_thread_id
        self._current_thread_id = resolved_thread_id

        if self.cache_manager:
            self.cache_manager.update_scope(
                agent_id=self.agent_id, memory_id=resolved_memory_id
            )

        return resolved_memory_id, resolved_thread_id

    def switch_thread(self, memory_id: str, start_new_thread: bool = False) -> str:
        """
        Switch the active thread to a specific memory_id.

        Returns:
            str: Active thread_id for the selected thread.
        """
        normalized_memory_id = str(memory_id or "").strip()
        if not normalized_memory_id:
            raise ValueError("memory_id is required to switch threads.")

        thread_id = None
        if not start_new_thread:
            thread_id = self._thread_ids_by_memory.get(normalized_memory_id)

        resolved_memory_id, resolved_thread_id = self._resolve_execution_state(
            normalized_memory_id, thread_id
        )
        if start_new_thread:
            resolved_thread_id = str(uuid.uuid4())
            self._thread_ids_by_memory[resolved_memory_id] = resolved_thread_id
            self._current_thread_id = resolved_thread_id
            logger.info(
                "Started new thread %s for memory %s",
                resolved_thread_id,
                resolved_memory_id,
            )

        return resolved_thread_id

    def run(
        self,
        query: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Run the agent with the given query using the new manager architecture.

        Args:
            query: The user's query
            memory_id: Optional memory ID to use (if not provided, uses stored or default)
            thread_id: Optional thread ID to use (if not provided, reuses current or creates new)
            user_id: Optional end-user identifier for multi-tenant apps. When
                provided, every memory unit written during this turn is tagged
                with ``user_id`` and every read is restricted to rows matching
                the same scope. ``None`` means anonymous/legacy scope — reads
                return only rows with no ``user_id`` set.
            context: Optional per-call ephemeral context (M2). When provided,
                the dict is rendered as a structured system-role message
                between the main system prompt and conversation history, so
                the agent sees it for *this turn only*. It is intentionally
                NOT persisted to ``conversation_memory`` (only the original
                ``query`` is recorded). Recommended shape for SDK consumers::

                    {
                      "current_page": {"type": "analysis"|"group"|...,
                                       "id": "<id>", "title": "<title>"},
                      "quoted_text": "<optional highlighted snippet>",
                      "highlights": [...],          # optional
                      "notes": [...],               # optional
                      "internet_search_allowed": True|False,
                      "open_threads": [...],        # optional
                    }

                Callers may pass any JSON-serialisable dict; unknown keys
                are included verbatim.
            tool_context: Optional per-call dict made available to tool
                functions via ``memorizz.get_tool_context()`` (M4). Unlike
                ``context`` above, this is NOT sent to the LLM at all — it's
                stored in a ``contextvars.ContextVar`` set before the tool
                loop and reset when the call returns. Use it for per-request
                facts the LLM should not see or be required to pass
                (e.g. ``{"user_id": "..."}`` for tenant-scoped tool queries).

        Returns:
            The agent's response
        """
        logger.info(f"MemAgent {self.agent_id} executing query: {query[:50]}...")

        # M4: scope per-call tool context for tools reading via get_tool_context().
        from ..tool_context import reset_tool_context, set_tool_context

        _tc_token = set_tool_context(tool_context or {})

        try:
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)

            # 2. Check semantic cache first
            cached_response = None
            if self.cache_manager.enabled:
                cached_response = self.cache_manager.get_cached_response(
                    query, thread_id, user_id=user_id
                )
                if cached_response:
                    logger.info("Returning cached response")
                    self._record_interaction(
                        query,
                        cached_response,
                        memory_id,
                        thread_id,
                        user_id=user_id,
                    )
                    return cached_response

            # 3. Build context and prompt
            built_context = self._build_context(query, memory_id, user_id=user_id)
            system_prompt = self._build_system_prompt()

            # 4. Execute with LLM
            response = self._execute_llm_interaction(
                system_prompt,
                query,
                built_context,
                user_id=user_id,
                request_context=context,
            )

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(
                    query, response, thread_id, user_id=user_id
                )

            # 6. Record interaction in memory
            self._record_interaction(
                query, response, memory_id, thread_id, user_id=user_id
            )

            logger.info(f"MemAgent {self.agent_id} completed successfully")
            return response

        except Exception as e:
            logger.error(f"MemAgent execution failed: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return error_response
        finally:
            # M4: always release the per-call tool_context scope.
            reset_tool_context(_tc_token)

    def run_stream(
        self,
        query: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """
        Run the agent with streaming output, yielding text chunks as they arrive.

        This method mirrors ``run()`` but yields partial text tokens so callers
        can display incremental output.  It honours the same caching, memory,
        and tool-calling logic as the synchronous path.

        Args:
            query: The user's query
            memory_id: Optional memory ID
            thread_id: Optional thread ID
            user_id: Optional end-user identifier for multi-tenant scoping. See
                ``run()`` for the full semantics.
            context: Optional per-call ephemeral context (M2). See ``run()``
                for the full semantics and recommended schema. Not persisted
                to ``conversation_memory``.
            tool_context: Optional per-call dict made available to tool
                functions via ``memorizz.get_tool_context()`` (M4). See
                ``run()`` for the full semantics. Not sent to the LLM.

        Yields:
            str: Partial text chunks of the agent's response
        """
        logger.info(f"MemAgent {self.agent_id} streaming query: {query[:50]}...")

        if not self.model or not hasattr(self.model, "generate_stream"):
            # Fallback: run synchronously and yield the full result
            yield self.run(
                query,
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                tool_context=tool_context,
            )
            return

        # M4: scope per-call tool context for tools reading via get_tool_context().
        from ..tool_context import reset_tool_context, set_tool_context

        _tc_token = set_tool_context(tool_context or {})

        # M3: lifecycle event so the consumer (e.g. an SSE translator) can
        # render "thinking…" indicators immediately, before any text chunk.
        self._emit_stream_event(
            "stream_start",
            {
                "agent_id": self.agent_id,
                "memory_id": memory_id,
                "thread_id": thread_id,
                "user_id": user_id,
                "has_request_context": bool(context),
            },
        )

        self._stream_trace_events = []
        try:
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)

            # 2. Check semantic cache first
            if self.cache_manager.enabled:
                cached = self.cache_manager.get_cached_response(
                    query, thread_id, user_id=user_id
                )
                if cached:
                    self._record_interaction(
                        query, cached, memory_id, thread_id, user_id=user_id
                    )
                    self._emit_stream_event(
                        "stream_end",
                        {"reason": "cache_hit", "agent_id": self.agent_id},
                    )
                    yield cached
                    return

            # 3. Build context and prompt
            built_context = self._build_context(query, memory_id, user_id=user_id)
            system_prompt = self._build_system_prompt()

            # 4. Stream the LLM interaction
            full_response = ""
            for chunk in self._execute_llm_interaction_stream(
                system_prompt,
                query,
                built_context,
                user_id=user_id,
                request_context=context,
            ):
                full_response += chunk
                yield chunk

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(
                    query, full_response, thread_id, user_id=user_id
                )

            # 6. Record interaction in memory
            self._record_interaction(
                query, full_response, memory_id, thread_id, user_id=user_id
            )
            self._record_stream_trace_bundle(memory_id, thread_id, user_id=user_id)
            self._emit_stream_event(
                "stream_end",
                {
                    "reason": "completed",
                    "agent_id": self.agent_id,
                    "response_length": len(full_response),
                },
            )

        except Exception as e:
            # Include full traceback so we can pinpoint which step in the
            # streaming pipeline failed (LOB-serialization bugs surface here
            # without a locator, which is useless for debugging).
            logger.exception("MemAgent streaming failed: %s", e)
            # M3: structured error event so SSE translators / UIs can render
            # an inline error banner instead of treating the error as text.
            self._emit_stream_event(
                "error",
                {
                    "message": str(e),
                    "exception_type": type(e).__name__,
                    "recoverable": False,
                    "agent_id": self.agent_id,
                },
            )
            self._emit_stream_event(
                "stream_end",
                {"reason": "error", "agent_id": self.agent_id},
            )
            yield f"I apologize, but I encountered an error: {str(e)}"
        finally:
            self._stream_trace_events = None
            # M4: always release the per-call tool_context scope.
            reset_tool_context(_tc_token)

    def set_stream_event_callback(
        self, callback: Optional[Callable[[Dict[str, Any]], None]]
    ):
        """Attach or clear a callback for structured streaming events.

        The callback receives a single ``dict`` per event with at minimum a
        ``type`` key. The full taxonomy of events emitted during a
        ``run_stream()`` invocation is:

        ============  =================================================================
        type          payload (additional keys)
        ============  =================================================================
        stream_start  ``agent_id``, ``memory_id``, ``thread_id``, ``user_id``,
                      ``has_request_context``. Fired once at the very top of
                      run_stream so consumers can render "thinking…" UX before
                      any text arrives.
        trace         ``trace_kind`` discriminator with values: ``reasoning``,
                      ``tool_call``, ``tool_result``. Plus ``title``, ``content``,
                      ``trace_id``, ``tool_name`` etc. depending on kind.
        error         ``message``, ``exception_type``, ``recoverable``,
                      ``agent_id``. Fired when run_stream catches an exception
                      mid-stream — UIs should render an error banner.
        stream_end    ``reason`` (``completed``/``cache_hit``/``error``),
                      ``agent_id``, optional ``response_length``. Fired once
                      when the stream terminates.
        ============  =================================================================

        Text chunks themselves are yielded by the generator, not surfaced via
        this callback. Translators that need a unified event stream (e.g. an
        SSE bridge) should combine the yielded chunks (as ``content`` events)
        with the callback events.
        """
        self._stream_event_callback = callback
        return self

    def _emit_stream_event(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a structured stream event to the UI callback when configured.

        See :meth:`set_stream_event_callback` for the full event taxonomy.
        """
        if event_type == "trace" and self._stream_trace_events is not None:
            if isinstance(payload, dict):
                self._stream_trace_events.append(dict(payload))

        callback = self._stream_event_callback
        if not callback:
            return
        event: Dict[str, Any] = {"type": event_type}
        if payload:
            event.update(payload)
        try:
            callback(event)
        except Exception as exc:
            logger.debug("Stream event callback failed: %s", exc)

    def _preview_stream_payload(self, value: Any, limit: int = 1800) -> str:
        """Serialize arbitrary event data into a bounded preview string."""
        if value is None:
            return ""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except Exception:
                text = str(value)
        text = text.strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit]}..."

    def _emit_stream_trace_chunks(
        self,
        trace_kind: str,
        title: str,
        content: Any,
        *,
        trace_id: Optional[str] = None,
        chunk_size: int = 420,
        preview_limit: int = 12000,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a trace payload in small chunks so the UI can render incrementally."""
        payload_text = (
            self._preview_stream_payload(content, limit=preview_limit)
            or "(empty result)"
        )
        chunks = [
            payload_text[i : i + chunk_size]
            for i in range(0, len(payload_text), chunk_size)
        ] or ["(empty result)"]

        base_payload: Dict[str, Any] = {
            "trace_kind": trace_kind,
            "title": title,
        }
        if trace_id:
            base_payload["trace_id"] = trace_id
        if extra:
            base_payload.update(extra)

        total_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            payload = dict(base_payload)
            payload["content"] = chunk
            payload["append"] = idx > 0
            payload["complete"] = idx == total_chunks - 1
            self._emit_stream_event("trace", payload)

    def _build_trace_bundle_events(
        self, events: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, str]]:
        """Normalize and compact streamed trace events for persistence."""
        if not events:
            return []

        normalized: List[Dict[str, str]] = []
        by_trace_id: Dict[str, Dict[str, str]] = {}

        for event in events:
            if not isinstance(event, dict):
                continue

            trace_kind = str(event.get("trace_kind") or event.get("kind") or "trace")
            trace_kind = trace_kind.strip().lower() or "trace"
            title = str(event.get("title") or "Trace").strip() or "Trace"
            content = self._preview_stream_payload(
                event.get("content") or event.get("message") or "",
                limit=12000,
            )
            trace_id = str(event.get("trace_id") or "").strip()
            append_mode = bool(event.get("append"))

            if trace_id and append_mode and trace_id in by_trace_id:
                existing = by_trace_id[trace_id]
                existing["content"] = f"{existing.get('content', '')}{content}"[:12000]
                continue

            entry = {
                "trace_kind": trace_kind,
                "title": title[:160],
                "content": content[:12000],
            }
            if trace_id:
                entry["trace_id"] = trace_id
                by_trace_id[trace_id] = entry
            normalized.append(entry)

        return [item for item in normalized if item.get("title") or item.get("content")]

    def _record_stream_trace_bundle(
        self,
        memory_id: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Persist streamed reasoning/tool traces for later Playground reloads."""
        if not self.memory_manager:
            return

        trace_events = self._build_trace_bundle_events(self._stream_trace_events)
        if not trace_events:
            return

        try:
            payload = {
                "type": "trace_bundle",
                "version": 1,
                "events": trace_events,
            }
            trace_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.TOOL,
                content=json.dumps(payload, ensure_ascii=False),
                thread_id=thread_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
            )
            self.memory_manager.save_memory_unit(trace_memory, memory_id)
            logger.debug(
                "Recorded streamed trace bundle in memory: %s events for thread %s",
                len(trace_events),
                memory_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to record streamed trace bundle: {exc}")

    def _format_recent_tool_logs_digest(self, limit: int = 10) -> str:
        """Build a compact, agent-facing digest of this thread's tool_log entries.

        Injected into the system prompt every turn so the agent retains
        pickable ``tool_log_id`` references across turns — the placeholder
        rows themselves are filtered out of the LLM's history to avoid
        orphan tool-role messages. This is the durable recall channel.

        Format (one entry per line):
            • <tool_log_id>  tool=<name>  args=<short>  digest=<one-liner>

        Returns empty string when there's nothing to surface.
        """
        if not self.memory_manager or not self._current_memory_id:
            return ""
        try:
            rows = self.memory_manager.list_tool_logs(
                memory_id=self._current_memory_id, limit=limit
            )
        except Exception as exc:
            logger.debug("Tool-log digest lookup failed: %s", exc)
            return ""
        if not rows:
            return ""
        lines: List[str] = []
        for row in rows[:limit]:
            if not isinstance(row, dict):
                continue
            tlid = row.get("tool_log_id") or row.get("_id") or ""
            if not tlid:
                continue
            tname = row.get("tool_name") or "?"
            args_text = _summarize_tool_args(row.get("arguments"), limit=120)
            # Tool-log result is stored as a string; parse if it looks like JSON
            # so we can still produce a field-aware digest.
            raw_result: Any = row.get("result")
            if isinstance(raw_result, str) and raw_result.strip().startswith(
                ("{", "[")
            ):
                try:
                    raw_result = json.loads(raw_result)
                except Exception:
                    pass
            digest = _summarize_tool_result(raw_result, preview_limit=200)
            bits = [f"tool={tname}"]
            if args_text:
                bits.append(f"args={args_text}")
            if digest:
                bits.append(f"digest={digest}")
            lines.append(f"• {tlid}  " + "  ".join(bits))
        if not lines:
            return ""
        return (
            "Recent tool outputs in this thread (use "
            "`retrieve_tool_log_entry('<id>')` to unpack the full result):\n"
            + "\n".join(lines)
        )

    def _is_trace_bundle_message(self, message: Any) -> bool:
        """Return True when a conversation row is a persisted trace bundle.

        Delegates to the public :func:`memorizz.is_trace_bundle_entry` so the
        detection lives in exactly one place (the same helper external consumers
        use to filter these internal rows out of displayed history).
        """
        return is_trace_bundle_entry(message)

    def _no_llm_message(self) -> str:
        """User-facing message when self.model is missing.

        Surfaces the captured init exception (HF download failure, gated
        repo, missing transformers feature, etc.) so the chat reply names
        the actual cause rather than the generic "No LLM model configured".
        """
        detail = getattr(self, "_llm_init_error", None)
        if detail:
            return f"Error: LLM failed to initialize — {detail}"
        return "Error: No LLM model configured"

    def _execute_llm_interaction_stream(
        self,
        system_prompt: str,
        query: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Execute the LLM interaction with streaming and tool-calling support.

        ``request_context`` is forwarded to :meth:`_build_prompt_messages`
        and is *not* persisted to ``conversation_memory``.
        """
        if not self.model:
            yield self._no_llm_message()
            return

        workflow = None
        WorkflowOutcome = None
        if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            from ..long_term.procedural.workflow.workflow import Workflow
            from ..long_term.procedural.workflow.workflow import (
                WorkflowOutcome as _WorkflowOutcome,
            )

            WorkflowOutcome = _WorkflowOutcome
            workflow = Workflow(
                name="Tool Execution for Query",
                description=f"Workflow tracking tool usage for: {query[:100]}",
                memory_id=self._current_memory_id
                or (self.memory_ids[0] if self.memory_ids else str(uuid.uuid4())),
                agent_id=self.agent_id,
                user_query=query,
            )
            if user_id is not None:
                try:
                    setattr(workflow, "user_id", user_id)
                except Exception:
                    pass
            logger.debug(
                f"Created streaming workflow for tracking: {workflow.workflow_id}"
            )

        workflow_persisted = False

        def _store_workflow_if_needed() -> None:
            nonlocal workflow_persisted
            if workflow_persisted:
                return
            if workflow and workflow.steps:
                try:
                    workflow.store_workflow(self.memory_provider)
                    workflow_persisted = True
                    logger.info(
                        "Stored workflow with %s steps (streaming)",
                        len(workflow.steps),
                    )
                except Exception as exc:
                    logger.error("Error storing streaming workflow: %s", exc)

        # Build messages (same as _execute_llm_interaction)
        messages = self._build_prompt_messages(
            system_prompt, query, context, request_context=request_context
        )

        # Build tools
        tools = None
        if self.tool_manager:
            tool_metadata = self.tool_manager.get_tool_metadata()
            if tool_metadata:
                tools = []
                for meta in tool_metadata:
                    if (
                        "type" in meta
                        and meta["type"] == "function"
                        and "function" in meta
                    ):
                        function_meta = (
                            meta.get("function")
                            if isinstance(meta.get("function"), dict)
                            else {}
                        )
                        function_name = str(function_meta.get("name", "")).strip()
                        if not function_name:
                            logger.warning(
                                "Skipping malformed OpenAI tool metadata without function.name"
                            )
                            continue
                        tools.append(meta)
                    else:
                        tool_name = str(meta.get("name", "")).strip()
                        if not tool_name:
                            logger.warning(
                                "Skipping tool metadata without name; it cannot be exposed for tool calling."
                            )
                            continue
                        parameters = meta.get("parameters", {})
                        if not isinstance(parameters, dict):
                            parameters = {}
                        required = meta.get("required", [])
                        if not isinstance(required, list):
                            required = []
                        openai_tool = {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": meta.get(
                                    "description", "No description"
                                ),
                                "parameters": {
                                    "type": "object",
                                    "properties": parameters,
                                    "required": required,
                                },
                            },
                        }
                        tools.append(openai_tool)

        # Streaming loop with tool calling
        max_iterations = self._get_tool_iteration_limit()
        try:
            for iteration in range(max_iterations):
                # Belt-and-braces: messages get appended inside this loop
                # (assistant tool_calls, tool results). Any LOB that slipped
                # into `result` via ``str(result)`` preserves shape but still
                # risks re-entering on subsequent iterations, so coerce once
                # per LLM call.
                messages = _to_jsonable(messages)
                for event in self.model.generate_stream(messages, tools=tools):
                    event_type = event.get("type")
                    if event_type == "content":
                        yield event.get("content", "")

                    elif event_type == "reasoning":
                        reasoning_preview = self._preview_stream_payload(
                            event.get("content", "")
                        )
                        if reasoning_preview:
                            self._emit_stream_event(
                                "trace",
                                {
                                    "trace_kind": "reasoning",
                                    "title": "Reasoning Trace",
                                    "content": reasoning_preview,
                                },
                            )

                    elif event_type == "tool_calls":
                        self._record_context_window_usage(
                            stage=f"stream_iteration_{iteration + 1}"
                        )
                        # Handle tool calls synchronously, then continue streaming
                        response = event["response"]
                        message = response.choices[0].message

                        messages.append(
                            {
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                    for tc in message.tool_calls
                                ],
                            }
                        )

                        for tool_call in message.tool_calls:
                            tool_name = tool_call.function.name
                            raw_arguments = tool_call.function.arguments
                            tool_trace_id = (
                                f"{tool_name}:{tool_call.id}"
                                if getattr(tool_call, "id", None)
                                else f"{tool_name}:{uuid.uuid4()}"
                            )
                            self._emit_stream_event(
                                "trace",
                                {
                                    "trace_kind": "tool_call",
                                    "title": f"Tool Call: {tool_name}",
                                    "tool_name": tool_name,
                                    "trace_id": f"call:{tool_trace_id}",
                                    "content": self._preview_stream_payload(
                                        raw_arguments or "{}",
                                        limit=1400,
                                    ),
                                },
                            )
                            try:
                                arguments = json.loads(raw_arguments)
                            except (json.JSONDecodeError, Exception):
                                arguments = {}

                            logger.info(f"Streaming: executing tool {tool_name}")
                            error_message = None
                            tool_outcome = (
                                WorkflowOutcome.SUCCESS
                                if workflow and WorkflowOutcome is not None
                                else None
                            )
                            if self.tool_manager:
                                try:
                                    result, _ = self.tool_manager.execute_tool(
                                        tool_name, arguments
                                    )
                                except Exception as e:
                                    result = f"Error executing tool: {str(e)}"
                                    error_message = str(e)
                                    if workflow and WorkflowOutcome is not None:
                                        tool_outcome = WorkflowOutcome.FAILURE
                            else:
                                result = "Error: No tool manager available"
                                error_message = "No tool manager available"
                                if workflow and WorkflowOutcome is not None:
                                    tool_outcome = WorkflowOutcome.FAILURE

                            self._emit_stream_trace_chunks(
                                "tool_result",
                                f"Tool Result: {tool_name}",
                                result,
                                trace_id=f"result:{tool_trace_id}",
                                chunk_size=420,
                                preview_limit=12000,
                                extra={"tool_name": tool_name},
                            )

                            # Offload full tool output to database as a tool log entry
                            tool_log_id = None
                            result_str = str(result)
                            if self.memory_manager and self._current_memory_id:
                                try:
                                    tool_log_id = self.memory_manager.store_tool_log(
                                        tool_name=tool_name,
                                        arguments=arguments,
                                        result=result,
                                        memory_id=self._current_memory_id,
                                        agent_id=self.agent_id,
                                        tool_call_id=getattr(tool_call, "id", None),
                                        success=(error_message is None),
                                        error=error_message,
                                        thread_id=self._current_thread_id,
                                        user_id=user_id,
                                    )
                                except Exception as log_exc:
                                    logger.debug("Tool log storage failed: %s", log_exc)

                            # Use compact reference in context window instead of full output.
                            # The placeholder now includes args + a field-aware summary
                            # so the LLM can disambiguate multiple calls to the same tool.
                            if tool_log_id:
                                compact_result = _build_tool_log_placeholder(
                                    tool_name=tool_name,
                                    tool_log_id=tool_log_id,
                                    arguments=arguments,
                                    result=result,
                                    error_message=error_message,
                                )
                            else:
                                compact_result = result_str

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": compact_result,
                                }
                            )

                            # Persist the placeholder as its own conversation_memory row
                            # so it survives into future turns (UI audit + system-prompt
                            # digest). Filtered from LLM message history via
                            # _is_tool_placeholder_content to avoid orphan tool-role
                            # messages without matching tool_call_ids.
                            if (
                                tool_log_id
                                and self.memory_manager
                                and self._current_memory_id
                            ):
                                try:
                                    placeholder_unit = self.memory_manager.create_conversation_memory_unit(
                                        role=Role.TOOL,
                                        content=compact_result,
                                        thread_id=self._current_thread_id,
                                        memory_id=self._current_memory_id,
                                        agent_id=self.agent_id,
                                        user_id=user_id,
                                    )
                                    self.memory_manager.save_memory_unit(
                                        placeholder_unit, self._current_memory_id
                                    )
                                except Exception as persist_exc:
                                    logger.debug(
                                        "Tool placeholder persist failed: %s",
                                        persist_exc,
                                    )

                            if workflow:
                                tool_entry = {}
                                if self.tool_manager:
                                    tool_metadata = (
                                        self.tool_manager.get_tool_metadata()
                                    )
                                    tool_entry = next(
                                        (
                                            meta
                                            for meta in tool_metadata
                                            if meta.get("name") == tool_name
                                        ),
                                        {},
                                    )

                                workflow.add_step(
                                    f"Step {len(workflow.steps) + 1}: {tool_name}",
                                    {
                                        "_id": (
                                            str(tool_entry.get("_id"))
                                            if tool_entry
                                            else None
                                        ),
                                        "arguments": arguments,
                                        "result": result,
                                        "timestamp": datetime.now().isoformat(),
                                        "error": error_message,
                                    },
                                )
                                if (
                                    tool_outcome is not None
                                    and WorkflowOutcome is not None
                                    and tool_outcome == WorkflowOutcome.FAILURE
                                ):
                                    workflow.outcome = WorkflowOutcome.FAILURE

                        # Continue to next iteration to stream the final response
                        continue

                    elif event_type == "done":
                        self._record_context_window_usage(
                            stage=f"stream_iteration_{iteration + 1}"
                        )
                        # No tool calls, streaming complete for this iteration
                        _store_workflow_if_needed()
                        return

                # If we got here after tool calls, the inner for-loop finished
                # and we need to loop again for the next streaming round
                continue

            # Exhausted iterations
            _store_workflow_if_needed()
            yield (
                "\n\nI reached the maximum number of tool-call iterations "
                f"({max_iterations}). Please try again."
            )
        except Exception:
            _store_workflow_if_needed()
            raise

    def _build_context(
        self,
        query: str,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build context for the query using memory manager."""
        context = {"query": query}

        if self.memory_manager:
            # Load conversation history
            try:
                history_limit = self._get_conversation_history_limit()
                history = self.memory_manager.load_conversation_history(
                    memory_id, limit=history_limit, user_id=user_id
                )

                # Filter two classes of tool-role rows from the LLM's view:
                #   1) trace_bundle rows (UI-only replay data)
                #   2) tool_log placeholder rows (surfaced to the agent as a
                #      structured digest in the system prompt instead, to
                #      avoid orphan tool messages without matching tool_call_ids)
                def _skip(item: Any) -> bool:
                    if self._is_trace_bundle_message(item):
                        return True
                    if isinstance(item, dict):
                        role = str(item.get("role") or "").strip().lower()
                        if role == Role.TOOL.value and _is_tool_placeholder_content(
                            item.get("content")
                        ):
                            return True
                    return False

                history = [item for item in history if not _skip(item)]
                context["conversation_history"] = history

            except Exception as e:
                logger.warning(f"Failed to build memory context: {e}")

            # Retrieve relevant memory snippets across core memory systems.
            try:
                relevant: Dict[str, Any] = {}
                for memory_type, bucket in (
                    (MemoryType.KNOWLEDGE_BASE, "semantic_memories"),
                    (MemoryType.CONVERSATION_MEMORY, "episodic_memories"),
                    (MemoryType.TOOLBOX, "procedural_memories"),
                ):
                    snippets = self.memory_manager.retrieve_relevant_memories(
                        query=query,
                        memory_type=memory_type,
                        memory_id=memory_id,
                        limit=3,
                        user_id=user_id,
                    )
                    if snippets:
                        relevant[bucket] = snippets

                if relevant:
                    context["relevant_memories"] = relevant
            except Exception as e:
                logger.warning(f"Failed to retrieve relevant memories: {e}")

        if (
            self._entity_memory_enabled
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            try:
                entity_context = self.entity_memory_manager.build_context(
                    query=query, memory_id=memory_id, user_id=user_id
                )
                if entity_context:
                    context["entity_memory_profiles"] = entity_context
                self._last_entity_context = entity_context
            except Exception as e:
                logger.warning(f"Failed to retrieve entity memory context: {e}")
                self._last_entity_context = []
        else:
            self._last_entity_context = []

        # Inject existing summary references so the agent knows what can be expanded
        if self.memory_manager:
            try:
                summaries = self.memory_manager.load_summaries_for_thread(
                    memory_id=memory_id,
                    agent_id=self.agent_id,
                    limit=20,
                    user_id=user_id,
                )
                if summaries:
                    context["summaries"] = summaries
            except Exception as e:
                logger.debug("Failed to load summary references: %s", e)

        return context

    def _build_system_prompt(self) -> str:
        """Build system prompt with base preamble, user instruction, and capability sections.

        Structure:
        1. Base system prompt (memory substrate awareness)
        2. Active memory types for this agent
        3. User's custom instruction
        4. Persona
        5. Tool descriptions
        6. Entity memory instructions (if enabled)
        7. Internet access instructions (if enabled)
        8. Skills marketplace instructions (if enabled)
        9. Automations instructions (if enabled)
        10. Sandbox instructions (if enabled)
        11. Self-awareness instructions (if enabled)
        12. Loaded skills (if configured)
        13. Configured MCP servers (if configured)
        """
        prompt_parts = []

        # 1. Base system prompt — always present
        prompt_parts.append(BASE_SYSTEM_PROMPT)
        prompt_parts.append(TOOL_ITERATION_BUDGET_INSTRUCTION)

        # 2. Active memory types and available memory tools for this agent
        if self.active_memory_types:
            active_names = [
                mt.value.replace("_", " ").title() for mt in self.active_memory_types
            ]
            memory_section = (
                "═══════════════════════════════════════════════════════════════\n"
                "ACTIVE MEMORY SYSTEMS FOR THIS SESSION\n"
                "═══════════════════════════════════════════════════════════════\n\n"
                "The following memory partitions are active and available to you:\n"
                + ", ".join(active_names)
                + ".\n\n"
                "Only the memory systems listed above are configured for this session. "
                "Focus your memory retrieval strategy on these active systems.\n"
            )

            # Inject available memory management tools
            memory_tools = []
            active_values = {mt.value for mt in self.active_memory_types}

            if "summaries" in active_values:
                memory_tools.append(
                    "• expand_summary(summary_id) — Reconstruct a compressed summary. "
                    "Pass the summary ID to retrieve the original messages and summary content. "
                    "Use this when you need detailed context from a previously summarized conversation."
                )
                memory_tools.append(
                    "• summarize_conversation(days_back, max_memories_per_summary) — "
                    "Compress older conversation messages into summaries to free context window space. "
                    "Original messages are preserved and can be reconstructed via expand_summary."
                )
                memory_tools.append(
                    "• list_summary_registry — List all available summaries for this thread. "
                    "Each entry includes a summary_id and short description. Use this to discover "
                    "what compressed conversations are available for expansion."
                )

            if "tool_log" in active_values:
                memory_tools.append(
                    "• retrieve_tool_log_entry(tool_log_id) — Retrieve the full output of a "
                    "previously executed tool by its log ID. Tool outputs over 500 characters are "
                    "automatically offloaded to the database; use this to access the complete result."
                )
                memory_tools.append(
                    "• list_recent_tool_logs(limit) — List recent tool execution records "
                    "including tool name, timestamp, and success/failure status."
                )

            if memory_tools:
                memory_section += (
                    "\nAvailable Memory Management Tools:\n"
                    + "\n".join(memory_tools)
                    + "\n"
                )

            prompt_parts.append(memory_section)

        # 3. User's custom instruction
        if self.instruction and self.instruction != DEFAULT_INSTRUCTION:
            prompt_parts.append(f"Instructions:\n{self.instruction}")
        else:
            prompt_parts.append(self.instruction)

        # 4. Persona information (with evolution history + update guidance)
        persona_prompt = self.persona_manager.get_persona_prompt(
            include_history=True, history_limit=5
        )
        if persona_prompt:
            prompt_parts.append(persona_prompt)
            if self._persona_tools_registered:
                prompt_parts.append(
                    "Persona evolution:\n"
                    "- You have two persona tools: `update_persona` and `read_persona`.\n"
                    "- Call `update_persona` ONLY for durable identity shifts (name, role, "
                    "goals, background). One-off user facts belong in entity_memory, not "
                    "your persona. Every call appends a traceable entry to your evolution "
                    "history — maintain continuity with prior versions rather than "
                    "contradicting them silently.\n"
                    "- `update_persona` requires a `reason` (why this change is warranted). "
                    "Provide `source_type` and `source_id` when the change was triggered by "
                    "a specific memory unit (e.g. source_type='conversation_memory', "
                    "source_id=<conversation memory id>) so the evolution is auditable.\n"
                    "- Use `read_persona` to inspect the full evolution history when the "
                    "summary in this prompt is not enough context."
                )

        # 5. Tool descriptions
        tools = self.tool_manager.get_tool_metadata()
        if tools:
            tool_descriptions = [
                f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}"
                for tool in tools
            ]
            if tool_descriptions:
                prompt_parts.append(
                    f"Available tools ({len(tool_descriptions)}):\n"
                    + chr(10).join(tool_descriptions)
                )

        # Knowledge-base instructions (only when the agent has attached KBs)
        kb_ids = getattr(self, "knowledge_base_ids", None) or []
        if kb_ids:
            prompt_parts.append(
                "Knowledge base usage:\n"
                f"- This agent has {len(kb_ids)} ingested document(s) available via "
                "'knowledge_base_lookup'.\n"
                "- Call 'knowledge_base_lookup' with a natural-language query "
                "whenever the user's question could be grounded in previously "
                "uploaded material (PDFs, notes, reference docs).\n"
                "- Prefer citing retrieved chunks verbatim over summarizing from memory."
            )

        # Recent tool outputs — durable digest of this thread's tool_log entries.
        # Without this, tool_log_id references from prior turns would be lost:
        # the placeholder rows are filtered out of LLM message history to avoid
        # orphan tool-role messages. Exposing a structured digest here lets the
        # agent pick the right id via ``retrieve_tool_log_entry`` even turns later.
        recent_digest = self._format_recent_tool_logs_digest(limit=10)
        if recent_digest:
            prompt_parts.append(recent_digest)

        # 6. Entity memory instructions
        if (
            self._entity_memory_enabled
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            if self._last_entity_context:
                entity_summary = self.entity_memory_manager.summarize_for_prompt(
                    self._last_entity_context
                )
                if entity_summary:
                    prompt_parts.append(
                        "Entity memory facts:\n"
                        + entity_summary
                        + "\nUse the entity memory tools to keep these facts up to date."
                    )
            else:
                prompt_parts.append(
                    "Entity memory usage:\n"
                    "- Before answering, call 'entity_memory_lookup' when the user references known people or when recalling prior facts might help.\n"
                    "- After the user shares stable personal info (name, preferences, background, ongoing projects, likes/dislikes), call 'entity_memory_upsert' with the attribute/value pair so it persists.\n"
                    "- Only store verifiable statements; skip speculative or time-sensitive details.\n"
                    "- Always keep JSON fields descriptive (e.g., attribute 'favorite_hobby', value 'hiking in the mountains')."
                )

        # 7. Internet access instructions
        if self.has_internet_access():
            provider_name = (
                self.get_internet_access_provider_name() or "internet provider"
            )
            prompt_parts.append(
                f"Internet access ({provider_name}):\n"
                "- Call 'internet_search' to look up fresh information online.\n"
                "- Call 'open_web_page' when you need to read a specific URL."
            )

        # 8. Skills marketplace instructions
        if self.has_skills_marketplace():
            provider_name = (
                self.get_skills_marketplace_provider_name() or "skills marketplace"
            )
            prompt_parts.append(
                f"Skills marketplace ({provider_name}):\n"
                "- Call 'skills_marketplace_search' to discover skills.\n"
                "- Skills marketplace discovery is handled by backend HTTP requests.\n"
                "- Any skill code/script execution must still run via sandbox tools."
            )

        # 9. Automations instructions (durable scheduling)
        if self.has_automations():
            prompt_parts.append(
                "Automations (scheduling/reminders):\n"
                "- If the user asks to schedule a reminder, recurring message, or timed task, "
                "ask clarifying questions: time/frequency, timezone, recipients, and content.\n"
                "- Use 'automation_create_job' with confirm=false to draft a proposed job and show a concise summary.\n"
                "- Only create or delete jobs after the user explicitly confirms.\n"
                "- For WhatsApp delivery, require recipients; accept E.164 numbers with or without the 'whatsapp:' prefix.\n"
                "- If required capabilities/data are missing, suggest enabling internet access or using "
                "'skills_marketplace_search' to discover a skill."
            )

        # 10. Sandbox code execution instructions
        if self.has_sandbox():
            provider_name = self.get_sandbox_provider_name() or "sandbox"
            prompt_parts.append(
                f"{SANDBOX_SYSTEM_PROMPT}\n" f"Sandbox provider: {provider_name}"
            )

        # 11. Self-awareness instructions (host codebase tools)
        if self.has_self_awareness():
            cfg = self.get_self_aware_config()
            roots = cfg.get("root_paths") or []
            if roots:
                root_lines = "\n".join(f"- {root_path}" for root_path in roots[:20])
            else:
                root_lines = "- (no roots configured)"
            prompt_parts.append(
                "Self-awareness host codebase tools:\n"
                f"Allowed roots:\n{root_lines}\n"
                f"- Writes enabled: {bool(cfg.get('allow_writes', False))}\n"
                f"- Deletes enabled: {bool(cfg.get('allow_deletes', False))}\n"
                "- Use 'self_aware_list_files', 'self_aware_read_file', and "
                "'self_aware_search_files' for inspection.\n"
                "- Use 'self_aware_write_file' for file mutation when writes are enabled.\n"
                "- Use 'self_aware_run_command' only for allowlisted host commands.\n"
                "- Keep operations bounded and within allowed roots."
            )

        # 12. Loaded skills (local files)
        if self.skills:
            skill_lines = []
            for skill in self.skills[:12]:
                skill_lines.append(
                    "- {name}: {description} (snippets={snippet_count}, scripts={script_count})".format(
                        name=skill.get("name", "Unnamed skill"),
                        description=skill.get("description", "Skill file"),
                        snippet_count=len(skill.get("code_blocks", [])),
                        script_count=len(skill.get("script_paths", [])),
                    )
                )
            prompt_parts.append(
                "Loaded skills:\n"
                + "\n".join(skill_lines)
                + "\nUse 'list_skills' or 'read_skill' to inspect details. "
                "Run any skill code/scripts only with 'run_skill_code' or "
                "'run_skill_script' (sandboxed execution)."
            )

        # 13. Configured MCP servers
        if self.mcp_servers:
            mcp_lines = []
            for server in self.mcp_servers[:12]:
                transport = server.get("transport", "stdio")
                endpoint = server.get("command") or server.get("url") or "unknown"
                mcp_lines.append(
                    f"- {server.get('name', 'unnamed')} [{transport}] -> {endpoint}"
                )
            prompt_parts.append(
                "Configured MCP servers:\n"
                + "\n".join(mcp_lines)
                + "\nUse 'list_mcp_servers' to inspect servers, 'mcp_list_tools' "
                "to discover tools, and 'mcp_call_tool' to execute MCP tools "
                "through the sandbox."
            )

        return "\n\n".join(prompt_parts)

    def _execute_llm_interaction(
        self,
        system_prompt: str,
        query: str,
        context: Dict[str, Any],
        user_id: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute the LLM interaction with tool calling support.

        ``request_context`` is forwarded to :meth:`_build_prompt_messages`
        and is *not* persisted to ``conversation_memory``.
        """
        if not self.model:
            return self._no_llm_message()

        try:
            # Initialize workflow tracking if workflow memory is active
            workflow = None
            if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
                from ..long_term.procedural.workflow.workflow import (
                    Workflow,
                    WorkflowOutcome,
                )

                workflow = Workflow(
                    name="Tool Execution for Query",
                    description=f"Workflow tracking tool usage for: {query[:100]}",
                    memory_id=self._current_memory_id
                    or (self.memory_ids[0] if self.memory_ids else str(uuid.uuid4())),
                    agent_id=self.agent_id,
                    user_query=query,
                )
                if user_id is not None:
                    try:
                        setattr(workflow, "user_id", user_id)
                    except Exception:
                        pass
                logger.debug(f"Created workflow for tracking: {workflow.workflow_id}")

            # Build initial messages
            messages = self._build_prompt_messages(
                system_prompt, query, context, request_context=request_context
            )

            # Get tool metadata if available and convert to OpenAI format
            tools = None
            if self.tool_manager:
                tool_metadata = self.tool_manager.get_tool_metadata()
                if tool_metadata:
                    # Convert to OpenAI function calling format
                    tools = []
                    for meta in tool_metadata:
                        # Check if already in correct OpenAI format (has nested 'function' key)
                        if (
                            "type" in meta
                            and meta["type"] == "function"
                            and "function" in meta
                        ):
                            function_meta = (
                                meta.get("function")
                                if isinstance(meta.get("function"), dict)
                                else {}
                            )
                            function_name = str(function_meta.get("name", "")).strip()
                            if not function_name:
                                logger.warning(
                                    "Skipping malformed OpenAI tool metadata without function.name"
                                )
                                continue
                            # Already in OpenAI format
                            tools.append(meta)
                        else:
                            tool_name = str(meta.get("name", "")).strip()
                            if not tool_name:
                                logger.warning(
                                    "Skipping tool metadata without name; it cannot be exposed for tool calling."
                                )
                                continue
                            parameters = meta.get("parameters", {})
                            if not isinstance(parameters, dict):
                                parameters = {}
                            required = meta.get("required", [])
                            if not isinstance(required, list):
                                required = []
                            # Convert to OpenAI format
                            openai_tool = {
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "description": meta.get(
                                        "description", "No description"
                                    ),
                                    "parameters": {
                                        "type": "object",
                                        "properties": parameters,
                                        "required": required,
                                    },
                                },
                            }
                            tools.append(openai_tool)

            # Execute main loop with tool calling
            max_iterations = self._get_tool_iteration_limit()
            for iteration in range(max_iterations):
                # Coerce LOBs to strings before every LLM call; matches the
                # streaming path's guarantee.
                messages = _to_jsonable(messages)
                # Call LLM
                response = self.model.generate(messages, tools=tools)
                self._record_context_window_usage(stage=f"iteration_{iteration + 1}")

                # Check if response is a string (no tool calls)
                if isinstance(response, str):
                    # Store workflow before returning (if it exists and has steps)
                    if workflow and workflow.steps:
                        try:
                            workflow.store_workflow(self.memory_provider)
                            logger.info(
                                f"Stored workflow with {len(workflow.steps)} steps"
                            )
                        except Exception as e:
                            logger.error(f"Error storing workflow: {str(e)}")
                    return response

                # Handle tool calls
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message

                    # If no tool calls, return the content
                    if not message.tool_calls:
                        final_content = (
                            message.content
                            if message.content
                            else "I couldn't generate a response."
                        )
                        # Store workflow before returning (if it exists and has steps)
                        if workflow and workflow.steps:
                            try:
                                workflow.store_workflow(self.memory_provider)
                                logger.info(
                                    f"Stored workflow with {len(workflow.steps)} steps"
                                )
                            except Exception as e:
                                logger.error(f"Error storing workflow: {str(e)}")
                        return final_content

                    # Add assistant message with tool calls to history
                    messages.append(
                        {
                            "role": "assistant",
                            "content": message.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in message.tool_calls
                            ],
                        }
                    )

                    # Execute each tool call
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            import json

                            arguments = json.loads(tool_call.function.arguments)
                        except (json.JSONDecodeError, Exception):
                            arguments = {}

                        logger.info(
                            f"Executing tool: {tool_name} with args: {arguments}"
                        )

                        # Execute the tool
                        error_message = None
                        tool_outcome = WorkflowOutcome.SUCCESS if workflow else None

                        if self.tool_manager:
                            try:
                                result, _ = self.tool_manager.execute_tool(
                                    tool_name, arguments
                                )
                            except Exception as e:
                                result = f"Error executing tool: {str(e)}"
                                error_message = str(e)
                                if workflow:
                                    tool_outcome = WorkflowOutcome.FAILURE
                                logger.error(f"Tool {tool_name} failed: {e}")
                        else:
                            result = "Error: No tool manager available"
                            error_message = "No tool manager available"
                            if workflow:
                                tool_outcome = WorkflowOutcome.FAILURE

                        # Offload full tool output to database as a tool log entry
                        tool_log_id = None
                        result_str = str(result)
                        if self.memory_manager and self._current_memory_id:
                            try:
                                tool_log_id = self.memory_manager.store_tool_log(
                                    tool_name=tool_name,
                                    arguments=arguments,
                                    result=result,
                                    memory_id=self._current_memory_id,
                                    agent_id=self.agent_id,
                                    tool_call_id=getattr(tool_call, "id", None),
                                    success=(error_message is None),
                                    error=error_message,
                                    thread_id=self._current_thread_id,
                                    user_id=user_id,
                                )
                            except Exception as log_exc:
                                logger.debug("Tool log storage failed: %s", log_exc)

                        # Use compact reference in context window instead of full output.
                        # Centralized in ``_build_tool_log_placeholder`` so the
                        # streaming and non-streaming paths stay in lockstep.
                        if tool_log_id:
                            compact_result = _build_tool_log_placeholder(
                                tool_name=tool_name,
                                tool_log_id=tool_log_id,
                                arguments=arguments,
                                result=result,
                                error_message=error_message,
                            )
                        else:
                            compact_result = result_str

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": compact_result,
                            }
                        )

                        # Persist the placeholder as its own conversation_memory row so
                        # it survives into future turns. Filtered out of LLM history
                        # via _is_tool_placeholder_content; the agent sees recent logs
                        # as a digest in the system prompt instead.
                        if (
                            tool_log_id
                            and self.memory_manager
                            and self._current_memory_id
                        ):
                            try:
                                placeholder_unit = (
                                    self.memory_manager.create_conversation_memory_unit(
                                        role=Role.TOOL,
                                        content=compact_result,
                                        thread_id=self._current_thread_id,
                                        memory_id=self._current_memory_id,
                                        agent_id=self.agent_id,
                                        user_id=user_id,
                                    )
                                )
                                self.memory_manager.save_memory_unit(
                                    placeholder_unit, self._current_memory_id
                                )
                            except Exception as persist_exc:
                                logger.debug(
                                    "Tool placeholder persist failed: %s", persist_exc
                                )

                        logger.info(f"Tool {tool_name} returned: {result}")

                        # Track workflow step if workflow memory is active
                        if workflow:
                            # Get tool metadata for the _id
                            tool_entry = {}
                            if self.tool_manager:
                                tool_metadata = self.tool_manager.get_tool_metadata()
                                tool_entry = next(
                                    (
                                        meta
                                        for meta in tool_metadata
                                        if meta.get("name") == tool_name
                                    ),
                                    {},
                                )

                            workflow.add_step(
                                f"Step {len(workflow.steps) + 1}: {tool_name}",
                                {
                                    "_id": (
                                        str(tool_entry.get("_id"))
                                        if tool_entry
                                        else None
                                    ),
                                    "arguments": arguments,
                                    "result": result,
                                    "timestamp": datetime.now().isoformat(),
                                    "error": error_message,
                                },
                            )

                            # Update workflow outcome if any step failed
                            if tool_outcome == WorkflowOutcome.FAILURE:
                                workflow.outcome = WorkflowOutcome.FAILURE

                    # Continue loop to get final response
                    continue

                # Fallback: return any content we got
                fallback_response = "I encountered an unexpected response format."
                # Store workflow before returning (if it exists and has steps)
                if workflow and workflow.steps:
                    try:
                        workflow.store_workflow(self.memory_provider)
                        logger.info(f"Stored workflow with {len(workflow.steps)} steps")
                    except Exception as e:
                        logger.error(f"Error storing workflow: {str(e)}")
                return fallback_response

            # If we exhausted iterations
            final_response = (
                "I reached the maximum number of tool-call iterations "
                f"({max_iterations}). Please try again."
            )

            # Store workflow if it was created and has steps
            if workflow and workflow.steps:
                try:
                    workflow.store_workflow(self.memory_provider)
                    logger.info(f"Stored workflow with {len(workflow.steps)} steps")
                except Exception as e:
                    logger.error(f"Error storing workflow: {str(e)}")
                    # Continue execution even if workflow storage fails

            return final_response

        except Exception as e:
            logger.error(f"LLM interaction failed: {e}")

            # Store workflow even on error if it exists
            if "workflow" in locals() and workflow and workflow.steps:
                try:
                    workflow.store_workflow(self.memory_provider)
                except Exception as workflow_error:
                    logger.error(
                        f"Error storing workflow after exception: {str(workflow_error)}"
                    )

            return f"I encountered an error while processing your request: {str(e)}"

    def _register_entity_memory_tools(self):
        """Register built-in tools that expose entity memory to the LLM."""
        if not (
            self.tool_manager
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            return

        if self._entity_memory_tools_registered:
            return

        def entity_memory_lookup(
            entity_id: str = None,
            name: str = None,
            query: str = None,
            limit: int = 5,
        ) -> Dict[str, Any]:
            """Search structured entity memory by id, name, or semantic query."""
            resolved_memory_id = self._current_memory_id or (
                self.memory_ids[0] if self.memory_ids else None
            )
            matches = self.entity_memory_manager.lookup_entities(
                entity_id=entity_id,
                name=name,
                query=query,
                limit=limit,
                memory_id=resolved_memory_id,
            )
            logger.info(
                "entity_memory_lookup returned %s record(s) (memory_id=%s)",
                len(matches),
                resolved_memory_id,
            )
            return {"matches": matches}

        def _normalize_attributes(attrs):
            """Normalize attributes from various formats to list of dicts with name/value."""
            if not attrs:
                return None
            if isinstance(attrs, str):
                if not attrs.strip():
                    return None
                try:
                    parsed = json.loads(attrs)
                    if isinstance(parsed, dict):
                        return [{"name": k, "value": str(v)} for k, v in parsed.items()]
                    return parsed if isinstance(parsed, list) else None
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(attrs, dict):
                return [{"name": k, "value": str(v)} for k, v in attrs.items()]
            return attrs if isinstance(attrs, list) else None

        def _normalize_json_field(field, expected_type):
            """Normalize JSON string fields to expected type."""
            if not field:
                return None
            if isinstance(field, str):
                if not field.strip() or field.strip() == "{}":
                    return None
                try:
                    parsed = json.loads(field)
                    return parsed if isinstance(parsed, expected_type) else None
                except (json.JSONDecodeError, TypeError):
                    return None
            return field if isinstance(field, expected_type) else None

        def entity_memory_upsert(
            entity_id: str = None,
            name: str = None,
            entity_type: str = None,
            attributes: Optional[List[Dict[str, Any]]] = None,
            relations: Optional[List[Dict[str, Any]]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            memory_id: str = None,
        ) -> Dict[str, Any]:
            """Insert or update a structured entity record."""
            resolved_memory_id = (
                memory_id
                or self._current_memory_id
                or (self.memory_ids[0] if self.memory_ids else None)
            )
            if resolved_memory_id is None:
                raise ValueError("A memory_id is required to store entity information.")

            new_entity_id = self.entity_memory_manager.upsert_entity_from_tool(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=_normalize_attributes(attributes),
                relations=_normalize_json_field(relations, list),
                metadata=_normalize_json_field(metadata, dict),
                memory_id=resolved_memory_id,
            )
            logger.info(
                "entity_memory_upsert stored entity_id=%s (memory_id=%s)",
                new_entity_id,
                resolved_memory_id,
            )
            return {"entity_id": new_entity_id}

        entity_memory_lookup.__name__ = "entity_memory_lookup"
        entity_memory_upsert.__name__ = "entity_memory_upsert"

        self.tool_manager.add_tool(entity_memory_lookup)
        self.tool_manager.add_tool(entity_memory_upsert)
        self._entity_memory_tools_registered = True

    def _register_knowledge_base_tools(self):
        """Register a semantic-search tool over this agent's knowledge base.

        Registered unconditionally during init. When the agent has no KB
        entries attached (``knowledge_base_ids`` empty) the tool returns
        an empty result — cheaper than querying the store, and it gives
        the LLM a consistent tool surface regardless of ingestion timing.
        """
        if not self.tool_manager or not self.memory_provider:
            return

        def knowledge_base_lookup(
            query: str,
            limit: int = 5,
            namespace: str = None,
        ) -> Dict[str, Any]:
            """Semantic search over knowledge-base documents ingested for this agent.

            Use this whenever the user's question could be answered by
            previously ingested reference material (PDFs, docs, notes).
            Returns the top matching chunks ordered by semantic similarity.

            Parameters:
                query: natural-language question or phrase to search for.
                limit: max number of chunks to return (default 5).
                namespace: optional filter — only return chunks ingested
                    under this namespace (usually the original filename).
            """
            kb_ids = set(getattr(self, "knowledge_base_ids", None) or [])
            if not kb_ids:
                return {
                    "matches": [],
                    "message": "No knowledge base entries are attached to this agent.",
                }
            try:
                from ..long_term.semantic.knowledge_base import KnowledgeBase

                kb = KnowledgeBase(memory_provider=self.memory_provider)
                raw = kb.retrieve_knowledge_by_query(
                    query=query,
                    namespace=namespace,
                    # Over-fetch so we can filter to the agent's KB ids and
                    # still return `limit` results in the common case.
                    limit=max(limit * 4, limit),
                )
            except Exception as exc:
                logger.warning("knowledge_base_lookup failed: %s", exc)
                return {"matches": [], "error": str(exc)}

            # Sanitize via the module-level helper so CLOB/LOB values from
            # the provider never reach the LLM's json.dumps path.
            matches: List[Dict[str, Any]] = []
            for entry in raw:
                if not isinstance(entry, dict):
                    continue
                entry_kb_id = entry.get("knowledge_base_id") or entry.get(
                    "knowledgeBaseId"
                )
                if entry_kb_id and entry_kb_id not in kb_ids:
                    continue
                matches.append(
                    _to_jsonable(
                        {
                            "content": entry.get("content") or "",
                            "namespace": entry.get("namespace"),
                            "knowledge_base_id": entry_kb_id,
                            "chunk_index": entry.get("chunk_index"),
                            "chunk_count": entry.get("chunk_count"),
                        }
                    )
                )
                if len(matches) >= limit:
                    break

            logger.info(
                "knowledge_base_lookup returned %d match(es) (kb_ids=%d, query=%r)",
                len(matches),
                len(kb_ids),
                query[:60],
            )
            return {"matches": matches}

        knowledge_base_lookup.__name__ = "knowledge_base_lookup"
        try:
            self.tool_manager.add_tool(knowledge_base_lookup)
        except Exception as exc:
            logger.debug("Could not register knowledge_base_lookup: %s", exc)

    def _register_internet_access_tools(self):
        """Register tools that expose internet search and browsing."""
        if not (
            self.tool_manager
            and self.internet_access_manager
            and self.internet_access_manager.is_enabled()
        ):
            return

        if self._internet_access_tools_registered:
            return

        def internet_search(query: str, max_results: int = 5) -> Dict[str, Any]:
            """Search the public internet for up-to-date information."""
            try:
                if self._internet_access_disabled_reason:
                    return {
                        "error": f"Internet access disabled: {self._internet_access_disabled_reason}"
                    }
                results = self.internet_access_manager.search(
                    query=query, max_results=max_results
                )
                self._internet_access_failure_count = 0
                return {"results": results}
            except Exception as exc:
                logger.error("internet_search failed: %s", exc)
                return self._handle_internet_access_error(str(exc))

        def open_web_page(url: str) -> Dict[str, Any]:
            """Fetch and summarize the contents of a website."""
            try:
                if self._internet_access_disabled_reason:
                    return {
                        "error": f"Internet access disabled: {self._internet_access_disabled_reason}"
                    }
                page = self.internet_access_manager.fetch_url(url=url)
                self._internet_access_failure_count = 0
                return page
            except Exception as exc:
                logger.error("open_web_page failed: %s", exc)
                return self._handle_internet_access_error(str(exc))

        internet_search.__name__ = "internet_search"
        open_web_page.__name__ = "open_web_page"

        self.tool_manager.add_tool(internet_search)
        self.tool_manager.add_tool(open_web_page)
        self._internet_access_tools_registered = True

    def _register_persona_tools(self) -> None:
        """Register ``update_persona`` and ``read_persona`` tools on the agent.

        These are registered once a persona is attached. The update tool
        appends a traceable entry to the persona's evolution history
        (``change_trigger`` metadata links the change to the memory unit or
        conversation that motivated it) and persists the change in the
        PERSONAS collection, then refreshes the agent's embedded persona
        snapshot via ``save()``.
        """
        if not self.tool_manager:
            return
        if self._persona_tools_registered:
            return
        if self.persona_manager is None or self.persona_manager.current_persona is None:
            return

        def update_persona(
            updates: Dict[str, str],
            reason: str,
            source_type: Optional[str] = None,
            source_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Update your own persona (name, role, goals, or background) with traceable provenance.

            Use this ONLY when a meaningful, durable change to your identity is warranted:
            - The user explicitly asks you to adopt a different persona or role.
            - A sustained pattern across conversations (not a one-off preference) indicates
              your goals or background should shift. One-off user facts belong in entity_memory,
              not in your persona.
            - New information fundamentally changes how you should present yourself long-term.

            Before calling, review the persona block in your system prompt: it includes the
            current version and the most recent evolution entries. Maintain continuity — your
            new values should build on the prior state, not contradict it without explanation.

            Parameters
            ----------
            updates : dict
                Only include the fields you are changing. Valid keys: ``name``, ``role``,
                ``goals``, ``background``. Omitted fields are preserved. Values that match
                the current state are no-ops.
            reason : str
                Required. A clear explanation of *why* this change is warranted. Saved to the
                evolution history so future-you (and human auditors) understand how the
                persona evolved. Be specific — cite the observation that prompted the update.
            source_type : str, optional
                The memory type or trigger class that motivated this change. Prefer a known
                MemoryType value: ``conversation_memory``, ``summaries``, ``entity_memory``,
                ``knowledge_base``, ``tool_log``, ``workflow_memory``. Extension values
                accepted: ``manual``, ``ui_form``, ``user_feedback``, ``llm_inference``,
                ``tool_call``.
            source_id : str, optional
                Storage id of the specific memory unit that triggered the change (e.g. the
                conversation memory id or summary id). Enables traceability back to the
                triggering observation.
            conversation_id : str, optional
                Thread id of the conversation in which this update was decided.

            Returns
            -------
            dict
                ``{"updated": True, "version": int, "history_entry": {...}, "persona": {...}}``
                when fields changed, or ``{"updated": False, "reason": ...}`` when the supplied
                updates matched the current state.
            """
            change_trigger = {
                "reason": reason,
                "source_type": source_type,
                "source_id": source_id,
                "conversation_id": conversation_id,
                "agent_id": self.agent_id,
            }
            try:
                result = self.persona_manager.apply_update(
                    updates=updates or {},
                    change_trigger=change_trigger,
                )
            except Exception as exc:
                logger.error("update_persona tool failed: %s", exc, exc_info=True)
                return {"updated": False, "error": f"Internal error: {exc}"}

            # When the update succeeded, refresh the embedded persona snapshot
            # on the agent document so later reads don't drift from the
            # PERSONAS collection.
            if result.get("updated") and self.memory_provider is not None:
                try:
                    self.save()
                except Exception as exc:
                    logger.warning(
                        "Persona updated in PERSONAS collection but agent save failed: %s",
                        exc,
                    )
                    result["agent_snapshot_warning"] = (
                        "Persona change was persisted to PERSONAS collection, but "
                        "refreshing the agent document failed. The next successful "
                        "save will reconcile the snapshot."
                    )
            return result

        def read_persona() -> Dict[str, Any]:
            """Return your current persona state and its evolution history.

            Use this when you need to inspect the full history of changes beyond the
            recent entries already summarized in your system prompt — for example, before
            calling ``update_persona`` on a nuanced change where older context matters.

            Returns
            -------
            dict
                ``{"persona": {...}, "version": int, "history_count": int}``. The
                ``persona`` dict includes ``evolution_history`` with the full trail.
            """
            persona = (
                self.persona_manager.current_persona if self.persona_manager else None
            )
            if persona is None:
                return {"error": "No persona attached to this agent."}
            payload = persona.to_dict()
            return {
                "persona": payload,
                "version": payload.get("version", 1),
                "history_count": len(payload.get("evolution_history") or []),
            }

        update_persona.__name__ = "update_persona"
        read_persona.__name__ = "read_persona"

        self.tool_manager.add_tool(update_persona)
        self.tool_manager.add_tool(read_persona)
        self._persona_tools_registered = True
        logger.info("Registered persona tools (update_persona, read_persona)")

    def _run_skills_marketplace_request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Call the skills marketplace API directly from the backend runtime."""
        provider_name = self.get_skills_marketplace_provider_name() or "skillsmp"
        config = self.get_skills_marketplace_config() or {}
        api_key = str(config.get("api_key") or "").strip()
        if not api_key:
            return {
                "ok": False,
                "error": (
                    f"Skills marketplace provider '{provider_name}' is missing an API key. "
                    "Set SKILLSMP_API_KEY in Settings."
                ),
            }

        base_url = str(config.get("base_url") or "https://skillsmp.com").strip()
        if not base_url:
            base_url = "https://skillsmp.com"

        endpoint_value = str(endpoint or "")
        if not endpoint_value.startswith("/"):
            endpoint_value = "/" + endpoint_value
        query = urllib.parse.urlencode(params or {}, doseq=True)
        url = base_url.rstrip("/") + endpoint_value + (("?" + query) if query else "")
        request = urllib.request.Request(
            url=url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "Memorizz-SkillsMarketplace/1.0",
            },
            method="GET",
        )
        request_timeout = max(1, min(int(timeout or 30), 120))

        try:
            with urllib.request.urlopen(request, timeout=request_timeout) as response:
                response_text = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(response_text) if response_text else {}
                return {
                    "ok": True,
                    "status_code": int(response.getcode() or 200),
                    "response": parsed,
                }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc else ""
            parsed = None
            try:
                parsed = json.loads(raw) if raw else None
            except Exception:
                parsed = None

            error_code = None
            error_message = ""
            if isinstance(parsed, dict):
                error_block = parsed.get("error")
                if isinstance(error_block, dict):
                    error_code = error_block.get("code")
                    error_message = str(error_block.get("message") or "")

            raw_compact = raw.strip()
            raw_lower = raw_compact.lower()
            if (
                not error_message
                and not isinstance(parsed, dict)
                and "cloudflare" in raw_lower
                and ("error 1010" in raw_lower or "access denied" in raw_lower)
            ):
                error_code = error_code or "CLOUDFLARE_ACCESS_DENIED"
                error_message = (
                    "SkillsMP denied access from this backend runtime "
                    "(Cloudflare 1010 Access Denied). "
                    "Ask SkillsMP support to allowlist this egress route."
                )
            if not error_message and raw_compact:
                error_message = raw_compact[:500]

            return {
                "ok": False,
                "status_code": int(getattr(exc, "code", 0) or 0),
                "error_code": error_code,
                "error": error_message or str(exc),
                "response": parsed,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _run_skills_marketplace_request_in_sandbox(
        self,
        endpoint: str,
        params: Dict[str, Any],
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Backward-compatible wrapper; marketplace requests now run on the backend."""
        return self._run_skills_marketplace_request(
            endpoint=endpoint,
            params=params,
            timeout=timeout,
        )

    def _register_skills_marketplace_tools(self):
        """Register tools that expose skills marketplace search."""
        if not self.tool_manager or self._skills_marketplace_tools_registered:
            return
        if not self.has_skills_marketplace():
            return

        provider_name = self.get_skills_marketplace_provider_name() or ""

        if provider_name == "skillsmp":
            self._register_skillsmp_tools()
        elif provider_name == "vercel":
            self._register_vercel_skills_tools()

        self._skills_marketplace_tools_registered = True

    def _register_skillsmp_tools(self):
        """Register SkillsMP marketplace search tool."""

        def skills_marketplace_search(
            q: str,
            mode: str = "keyword",
            page: int = 1,
            limit: int = 20,
            sort_by: str = "",
        ) -> Dict[str, Any]:
            """Search skills from the configured Skills Marketplace provider.

            Marketplace discovery runs via backend HTTP using the configured provider
            credentials. Skill execution remains sandbox-only through the sandbox tools.

            Args:
                q: Search query string.
                mode: Search mode, either ``keyword`` or ``ai``.
                page: Page number for keyword search (default: 1).
                limit: Number of items for keyword search (default: 20, max: 100).
                sort_by: Optional keyword sort mode (``stars`` or ``recent``).

            Returns:
                Dict with normalized ``skills`` results and provider response metadata.
            """
            query = str(q or "").strip()
            if not query:
                return {"ok": False, "error": "Query 'q' is required."}

            normalized_mode = str(mode or "keyword").strip().lower()
            if normalized_mode in {"ai", "semantic", "ai-search"}:
                endpoint = "/api/v1/skills/ai-search"
                params = {"q": query}
                request_mode = "ai"
            else:
                endpoint = "/api/v1/skills/search"
                try:
                    safe_page = max(1, int(page or 1))
                except (TypeError, ValueError):
                    safe_page = 1
                try:
                    safe_limit = max(1, min(int(limit or 20), 100))
                except (TypeError, ValueError):
                    safe_limit = 20
                params = {"q": query, "page": safe_page, "limit": safe_limit}
                normalized_sort = str(sort_by or "").strip().lower()
                if normalized_sort in {"stars", "recent"}:
                    params["sortBy"] = normalized_sort
                request_mode = "keyword"

            response = self._run_skills_marketplace_request(
                endpoint=endpoint,
                params=params,
                timeout=30,
            )
            if not isinstance(response, dict):
                return {
                    "ok": False,
                    "mode": request_mode,
                    "error": "Unexpected marketplace response format.",
                }

            if not response.get("ok"):
                status_code = response.get("status_code")
                error_code = response.get("error_code")
                error_text = str(response.get("error") or "").strip()
                error_text_lower = error_text.lower()
                if status_code == 403 and (
                    str(error_code or "").upper().startswith("CLOUDFLARE")
                    or "cloudflare" in error_text_lower
                    or "error 1010" in error_text_lower
                ):
                    error_code = "CLOUDFLARE_ACCESS_DENIED"
                    error_text = (
                        "SkillsMP denied access from this backend runtime (Cloudflare 1010). "
                        "Use an allowed backend network route or ask SkillsMP to allowlist it."
                    )
                if error_code and error_text:
                    error_text = f"{error_code}: {error_text}"
                elif not error_text:
                    error_text = "Marketplace request failed."
                return {
                    "ok": False,
                    "mode": request_mode,
                    "error": error_text,
                    "status_code": status_code,
                    "response": response.get("response"),
                }

            payload = response.get("response")
            if not isinstance(payload, dict):
                return {
                    "ok": False,
                    "mode": request_mode,
                    "error": "Marketplace returned invalid JSON payload.",
                    "status_code": response.get("status_code"),
                }

            data = payload.get("data")
            skills: List[Any] = []
            if isinstance(data, dict) and isinstance(data.get("skills"), list):
                skills = data.get("skills") or []

            return {
                "ok": True,
                "mode": request_mode,
                "query": query,
                "skills": skills,
                "count": len(skills),
                "status_code": response.get("status_code"),
                "response": payload,
            }

        skills_marketplace_search.__name__ = "skills_marketplace_search"
        self.tool_manager.add_tool(skills_marketplace_search)

    def _register_vercel_skills_tools(self):
        """Register Vercel Agent Skills tools (search + fetch)."""
        from ..vercel_skills import VercelSkillsProvider
        from ..vercel_skills.provider import MISSING_GITHUB_TOKEN_MESSAGE

        config = self.get_skills_marketplace_config() or {}
        if not str(config.get("github_token") or "").strip():
            logger.warning(MISSING_GITHUB_TOKEN_MESSAGE)
        provider = VercelSkillsProvider(config=config)

        def vercel_skills_search(
            q: str,
            limit: int = 20,
        ) -> Dict[str, Any]:
            """Search Vercel Agent Skills across the open skills ecosystem.

            Searches GitHub for repositories containing SKILL.md files that match
            the query. Use this to discover reusable agent skills for tasks like
            React, Next.js, AI SDK, deployment, design, and more.

            Args:
                q: Search query (e.g. "react best practices", "deployment", "AI SDK").
                limit: Maximum results to return (default: 20, max: 100).

            Returns:
                Dict with ``ok``, ``skills`` list (each with repo, name, description),
                and metadata.
            """
            return provider.search(query=q, limit=limit)

        def vercel_skill_fetch(
            repo: str,
            skill_name: str = "",
            branch: str = "main",
        ) -> Dict[str, Any]:
            """Fetch a Vercel Agent Skill's instructions from a GitHub repository.

            Downloads and parses the SKILL.md file from the given repo. The returned
            instructions tell you exactly how to apply this skill to the current task.

            For repos with multiple skills, provide the skill_name to target a specific one.

            Args:
                repo: GitHub repository in ``owner/repo`` format, or a full GitHub URL.
                skill_name: Optional specific skill within a multi-skill repo.
                branch: Git branch to fetch from (default: ``main``).

            Returns:
                Dict with ``ok``, ``name``, ``description``, ``instructions``
                (the full skill content), and ``metadata``.
            """
            return provider.fetch_skill(
                repo=repo,
                skill_name=skill_name or None,
                branch=branch,
            )

        vercel_skills_search.__name__ = "vercel_skills_search"
        vercel_skill_fetch.__name__ = "vercel_skill_fetch"
        self.tool_manager.add_tool(vercel_skills_search)
        self.tool_manager.add_tool(vercel_skill_fetch)

    def _unregister_entity_memory_tools(self):
        """Remove entity memory tools from the tool manager."""
        if not self.tool_manager or not self._entity_memory_tools_registered:
            return

        for tool_name in self._entity_memory_tool_names:
            self.tool_manager.remove_tool(tool_name)

        self._entity_memory_tools_registered = False

    def _unregister_internet_access_tools(self):
        """Remove internet access tools when provider is disabled."""
        if not self.tool_manager or not self._internet_access_tools_registered:
            return

        for tool_name in self._internet_access_tool_names:
            self.tool_manager.remove_tool(tool_name)

        self._internet_access_tools_registered = False

    def _unregister_skills_marketplace_tools(self):
        """Remove skills marketplace tools when provider is disabled."""
        if not self.tool_manager or not self._skills_marketplace_tools_registered:
            return

        for tool_name in self._skills_marketplace_tool_names:
            self.tool_manager.remove_tool(tool_name)

        self._skills_marketplace_tools_registered = False

    def _handle_internet_access_error(self, message: str) -> Dict[str, Any]:
        """Track repeated provider failures and disable tools after threshold."""
        self._internet_access_failure_count += 1
        if (
            self._internet_access_failure_count >= 3
            and not self._internet_access_disabled_reason
        ):
            self._internet_access_disabled_reason = (
                "Internet access temporarily disabled after repeated failures."
            )
            logger.warning(
                "Disabling internet access tools after %s failures.",
                self._internet_access_failure_count,
            )
        return {
            "error": message
            if not self._internet_access_disabled_reason
            else f"{message} | {self._internet_access_disabled_reason}"
        }

    def _record_interaction(
        self,
        query: str,
        response: str,
        memory_id: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ):
        """Record the interaction in memory."""
        if not self.memory_manager:
            return

        try:
            # Record user query
            user_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.USER,
                content=query,
                thread_id=thread_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
            )
            self.memory_manager.save_memory_unit(user_memory, memory_id)

            # Record assistant response
            assistant_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.ASSISTANT,
                content=response,
                thread_id=thread_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
            )
            self.memory_manager.save_memory_unit(assistant_memory, memory_id)

            logger.debug(f"Recorded interaction in memory: {memory_id}")

        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")

    # Conversation state management methods
    def start_new_thread(self, memory_id: str = None) -> str:
        """
        Start a new thread within the specified (or current) memory scope.

        Returns:
            str: The new thread_id
        """
        target_memory_id = str(memory_id or "").strip()
        if not target_memory_id:
            target_memory_id = (
                self._current_memory_id
                or (str(self.memory_ids[0]).strip() if self.memory_ids else "")
                or str(uuid.uuid4())
            )

        new_thread_id = str(uuid.uuid4())
        self._resolve_execution_state(target_memory_id, new_thread_id)
        logger.info(
            "Started new thread: %s (memory=%s)",
            new_thread_id,
            target_memory_id,
        )
        return new_thread_id

    def get_current_thread_id(self) -> Optional[str]:
        """
        Get the current thread ID.

        Returns:
            str: Current thread_id or None
        """
        return self._current_thread_id

    def get_current_memory_id(self) -> Optional[str]:
        """
        Get the current memory ID.

        Returns:
            str: Current memory_id or None
        """
        return self._current_memory_id

    def reset_thread_state(self):
        """Reset both thread_id and memory_id (start completely fresh)."""
        self._current_thread_id = None
        self._current_memory_id = None
        self._thread_ids_by_memory = {}
        logger.info("Reset thread state")

    def enable_semantic_cache(
        self,
        threshold: float = 0.85,
        scope: str = "local",
        embedding_provider: str = None,
        embedding_config: Dict[str, Any] = None,
    ):
        """
        Enable semantic cache on an already-built agent.

        Args:
            threshold: Similarity threshold for cache hits (0.0-1.0, default: 0.85)
            scope: Cache scope - 'local' (agent-specific) or 'global' (all agents)
            embedding_provider: Optional embedding provider (e.g., 'openai', 'ollama')
            embedding_config: Optional configuration for embedding provider

        Returns:
            Self for method chaining

        Example:
            agent.enable_semantic_cache(threshold=0.9, scope='local')
        """
        try:
            pass

            # Create cache configuration
            cache_config = {"similarity_threshold": threshold, "scope": scope}

            # Add embedding provider if specified
            if embedding_provider:
                cache_config["embedding_provider"] = embedding_provider
            if embedding_config:
                cache_config["embedding_config"] = embedding_config

            # Re-initialize the cache manager with new settings
            if self.cache_manager:
                # Enable the existing cache manager
                self.cache_manager.enabled = True
                self.cache_manager.memory_provider = (
                    self.memory_provider
                )  # Ensure provider is set
                self.cache_manager._initialize_cache(
                    config=cache_config,
                    agent_id=self.agent_id,
                    memory_id=self.memory_ids[0] if self.memory_ids else None,
                )
            else:
                # Create a new cache manager if it doesn't exist
                from .managers import CacheManager

                self.cache_manager = CacheManager(
                    enabled=True,
                    config=cache_config,
                    agent_id=self.agent_id,
                    memory_id=self.memory_ids[0] if self.memory_ids else None,
                    memory_provider=self.memory_provider,  # ✅ Pass memory provider
                )

            logger.info(
                f"Semantic cache enabled for agent {self.agent_id} with threshold={threshold}, scope={scope}"
            )
            return self

        except Exception as e:
            logger.error(f"Failed to enable semantic cache: {e}")
            raise

    def disable_semantic_cache(self):
        """
        Disable semantic cache on the agent.

        Returns:
            Self for method chaining
        """
        if self.cache_manager:
            self.cache_manager.enabled = False
            logger.info(f"Semantic cache disabled for agent {self.agent_id}")
        return self

    # Additional methods for compatibility
    def load_conversation_history(self, memory_id: str = None):
        """Load conversation history (delegated to memory manager)."""
        if self.memory_manager:
            memory_id = (
                memory_id
                or self._current_memory_id
                or (self.memory_ids[0] if self.memory_ids else None)
            )
            if memory_id:
                return self.memory_manager.load_conversation_history(memory_id)
        return []

    def get_context_window_stats(self) -> Optional[Dict[str, Any]]:
        """Return the most recent context window usage snapshot."""
        if not self._last_context_window_stats:
            return None
        return dict(self._last_context_window_stats)

    def add_tool(self, tool, persist: bool = False):
        """Add a tool (delegated to tool manager)."""
        return self.tool_manager.add_tool(tool, persist)

    def set_persona(self, persona, save: bool = True):
        """Set persona (delegated to persona manager) and ensure persona tools are registered."""
        result = self.persona_manager.set_persona(persona, self.agent_id, save)
        try:
            self._register_persona_tools()
        except Exception as exc:
            logger.warning("Persona tools registration failed: %s", exc)
        return result

    def download_memory(self, memagent: "MemAgent") -> bool:
        """
        Download memory from another MemAgent by copying its memory_ids.

        This adds the source agent's memory_ids to this agent's memory_ids,
        giving this agent access to the same conversation history. The source
        agent's memory is not affected.

        Args:
            memagent: The MemAgent to copy memory_ids from.

        Returns:
            True if successful, False otherwise.
        """
        try:
            for mid in memagent.memory_ids:
                if mid not in self.memory_ids:
                    self.memory_ids.append(mid)

            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(
                    self.agent_id, self.memory_ids
                )

            if self.memory_manager:
                self.memory_manager.clear_conversation_cache()

            logger.info(
                f"Downloaded memory from agent {memagent.agent_id} to {self.agent_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error downloading memory from agent {memagent.agent_id}: {e}"
            )
            return False

    def update_memory(self, memory_ids: List[str]) -> bool:
        """
        Add memory_ids to this agent, giving it access to additional conversation history.

        Args:
            memory_ids: List of memory_ids to add.

        Returns:
            True if successful, False otherwise.
        """
        try:
            for mid in memory_ids:
                if mid not in self.memory_ids:
                    self.memory_ids.append(mid)

            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(
                    self.agent_id, self.memory_ids
                )

            if self.memory_manager:
                self.memory_manager.clear_conversation_cache()

            logger.info(f"Updated memory_ids for agent {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating memory_ids for agent {self.agent_id}: {e}")
            return False

    def delete_memory(self) -> bool:
        """
        Delete all memory associated with this agent.

        Deletes the stored conversation data for each memory_id, removes the
        agent's memory_id references from the memory provider, and clears all
        local state (memory_ids, cached memory_id, conversation cache).

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Delete the actual conversation data for each memory_id
            if self.memory_manager:
                for mid in self.memory_ids:
                    self.memory_manager.delete_memory(mid)

            # Remove memory_id references from the agent record
            if hasattr(self.memory_provider, "delete_memagent_memory_ids"):
                self.memory_provider.delete_memagent_memory_ids(self.agent_id)

            # Clear all local state so the next run() starts fresh
            self.memory_ids = []
            self._current_memory_id = None
            self._current_thread_id = None
            self._thread_ids_by_memory.clear()

            logger.info(f"Deleted all memory for agent {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory for agent {self.agent_id}: {e}")
            return False

    def save(self):
        """
        Store the memagent in the memory provider.

        This method stores the memagent configuration and state in the memory provider,
        allowing for persistence and later restoration of the agent.

        Returns:
            MemAgent: Self, for method chaining
        """
        if not self.memory_provider:
            raise ValueError("Cannot save MemAgent: no memory provider configured")

        try:
            # Serialize tools from tool manager
            tools_to_save = self._serialize_tools_for_save()

            # Convert delegates to agent IDs for persistence
            delegate_ids = self._serialize_delegates_for_save()

            # Get semantic cache config for saving
            semantic_cache_config_to_save = self._serialize_semantic_cache_config()

            # Create MemAgentModel with current configuration
            from .models import MemAgentModel

            memagent_to_save = MemAgentModel(
                llm_config=(
                    self.model.get_config()
                    if self.model and hasattr(self.model, "get_config")
                    else None
                ),
                name=self.name,
                instruction=self.instruction,
                max_steps=self.max_steps,
                application_mode=self._get_application_mode_value(),
                memory_types=[
                    memory_type.value
                    for memory_type in (self.active_memory_types or [])
                    if hasattr(memory_type, "value")
                ]
                or None,
                memory_ids=self.memory_ids,
                knowledge_base_ids=getattr(self, "knowledge_base_ids", None) or None,
                agent_id=self.agent_id,
                persona=(
                    self.persona_manager.current_persona
                    if self.persona_manager
                    else None
                ),
                tools=tools_to_save,
                delegates=delegate_ids if delegate_ids else None,
                semantic_cache=(
                    self.cache_manager.enabled if self.cache_manager else False
                ),
                semantic_cache_config=semantic_cache_config_to_save,
                context_window_tokens=self._context_window_tokens,
                is_favorite=self.is_favorite,
                internet_access_provider=self.get_internet_access_provider_name(),
                internet_access_config=(
                    self.internet_access_manager.get_provider_config()
                    if self.has_internet_access()
                    else None
                ),
                skills_marketplace_provider=self.get_skills_marketplace_provider_name(),
                skills_marketplace_config=self.get_skills_marketplace_config(),
                sandbox_provider=(
                    self.sandbox_manager.get_provider_config()
                    if self.has_sandbox()
                    else None
                ),
                skill_paths=self.skill_paths or None,
                mcp_servers=self.mcp_servers or None,
                self_aware=bool(self.self_aware),
                self_aware_config=self.get_self_aware_config() or None,
                automations_enabled=bool(getattr(self, "automations_enabled", True)),
                default_timezone=getattr(self, "default_timezone", None),
            )

            # Save or update the agent
            if hasattr(self.memory_provider, "store_memagent"):
                if self.agent_id and hasattr(self.memory_provider, "retrieve_memagent"):
                    # Check if agent exists for update vs create
                    try:
                        existing = self.memory_provider.retrieve_memagent(self.agent_id)
                        if existing and hasattr(
                            self.memory_provider, "update_memagent"
                        ):
                            saved_memagent = self.memory_provider.update_memagent(
                                memagent_to_save
                            )
                        else:
                            saved_memagent = self.memory_provider.store_memagent(
                                memagent_to_save
                            )
                    except Exception:
                        # Agent doesn't exist, create new
                        saved_memagent = self.memory_provider.store_memagent(
                            memagent_to_save
                        )
                else:
                    # New agent
                    saved_memagent = self.memory_provider.store_memagent(
                        memagent_to_save
                    )

                # Update agent_id if it was generated
                if not self.agent_id and saved_memagent.get("_id"):
                    self.agent_id = str(saved_memagent["_id"])

                    # Update semantic cache with new agent_id
                    if self.cache_manager and self.cache_manager.enabled:
                        if (
                            hasattr(self.cache_manager, "cache_instance")
                            and self.cache_manager.cache_instance
                        ):
                            self.cache_manager.cache_instance.agent_id = self.agent_id
                            if self.memory_ids:
                                self.cache_manager.cache_instance.memory_id = (
                                    self.memory_ids[0]
                                )

                self._persist_mcp_servers_to_toolbox_memory()

                logger.info(f"MemAgent {self.agent_id} saved successfully")
                return self
            else:
                raise ValueError("Memory provider does not support saving MemAgent")

        except Exception as e:
            logger.error(f"Failed to save MemAgent {self.agent_id}: {e}")
            raise

    def _get_application_mode_value(self) -> str:
        """Return the application mode value as a string."""
        if isinstance(self.application_mode, ApplicationMode):
            return self.application_mode.value
        if isinstance(self.application_mode, str):
            return self.application_mode
        return ApplicationMode.DEFAULT.value

    def _serialize_tools_for_save(self):
        """Serialize tools for saving."""
        if not self.tool_manager:
            return None

        tools_metadata = self.tool_manager.get_tool_metadata()
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
                    "type": tool_meta.get("type", "function"),
                    "agent_id": self.agent_id,
                }
                serializable_tools.append(serializable_tool)

        return serializable_tools if serializable_tools else None

    def _persist_mcp_servers_to_toolbox_memory(self) -> None:
        """Persist MCP JSON configs into toolbox memory records."""
        if not self.memory_provider or not self.mcp_servers:
            return

        try:
            memory_id = self._current_memory_id or (
                self.memory_ids[0] if self.memory_ids else None
            )
            if not memory_id:
                memory_id = str(uuid.uuid4())
                self.memory_ids.append(memory_id)
                self._current_memory_id = memory_id

            for server in self.mcp_servers:
                server_name = str(server.get("name", "")).strip()
                if not server_name:
                    continue
                config_doc = {
                    "_id": f"{self.agent_id}:mcp:{server_name}",
                    "tool_id": f"{self.agent_id}:mcp:{server_name}",
                    "name": f"mcp::{server_name}",
                    "description": f"MCP server config for {server_name}",
                    "signature": "mcp_server_config(server_json)",
                    "docstring": "Stored MCP server configuration JSON.",
                    "tool_type": "mcp_server_config",
                    "type": "mcp_server_config",
                    "parameters": server,
                    "memory_id": memory_id,
                    "agent_id": self.agent_id,
                }
                self.memory_provider.store(
                    config_doc, memory_store_type=MemoryType.TOOLBOX
                )
        except Exception as exc:
            logger.warning(
                "Failed to persist MCP configs to toolbox memory for %s: %s",
                self.agent_id,
                exc,
            )

    def _serialize_delegates_for_save(self):
        """Serialize delegate agents for saving."""
        # Note: In the new architecture, delegates would be handled differently
        # This is a placeholder for compatibility
        if hasattr(self, "delegates") and self.delegates:
            delegate_ids = []
            for delegate in self.delegates:
                if hasattr(delegate, "agent_id") and delegate.agent_id:
                    delegate_ids.append(delegate.agent_id)
                    # Ensure delegate is saved
                    try:
                        if hasattr(delegate, "save"):
                            delegate.save()
                    except Exception as e:
                        logger.warning(
                            f"Failed to save delegate {delegate.agent_id}: {e}"
                        )
            return delegate_ids
        return None

    def _serialize_semantic_cache_config(self):
        """Serialize semantic cache configuration for saving."""
        if not self.cache_manager or not self.cache_manager.enabled:
            return None

        try:
            if (
                hasattr(self.cache_manager, "cache_instance")
                and self.cache_manager.cache_instance
            ):
                if hasattr(self.cache_manager.cache_instance, "config"):
                    config = self.cache_manager.cache_instance.config
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

    @classmethod
    def load(cls, agent_id: str, memory_provider=None, **overrides):
        """
        Load a MemAgent from the memory provider.

        Args:
            agent_id (str): The agent ID to load
            memory_provider: Memory provider to use (optional)
            **overrides: Override parameters for the loaded agent

        Returns:
            MemAgent: The loaded agent instance
        """
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
                model_to_load = create_llm_provider(saved_memagent.llm_config)
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

        saved_self_aware = bool(getattr(saved_memagent, "self_aware", False))
        saved_self_aware_config = getattr(saved_memagent, "self_aware_config", None)
        if not isinstance(saved_self_aware_config, dict):
            saved_self_aware_config = None

        saved_automations_enabled = getattr(saved_memagent, "automations_enabled", True)
        if saved_automations_enabled is None:
            saved_automations_enabled = True
        saved_automations_enabled = bool(saved_automations_enabled)

        saved_default_timezone = getattr(saved_memagent, "default_timezone", None)
        if not isinstance(saved_default_timezone, str):
            saved_default_timezone = None
        else:
            saved_default_timezone = saved_default_timezone.strip() or None

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
            skill_paths=overrides.get("skill_paths", saved_skill_paths),
            mcp_servers=overrides.get("mcp_servers", saved_mcp_servers),
            automations_enabled=overrides.get(
                "automations_enabled", saved_automations_enabled
            ),
            default_timezone=overrides.get("default_timezone", saved_default_timezone),
            self_aware=overrides.get("self_aware", saved_self_aware),
            self_aware_config=overrides.get(
                "self_aware_config", saved_self_aware_config
            ),
            is_favorite=overrides.get(
                "is_favorite", getattr(saved_memagent, "is_favorite", False)
            ),
            streaming=overrides.get("streaming", False),
        )

        # Hydrate knowledge_base_ids separately — it isn't a constructor arg
        # (yet) but needs to survive reloads so the `knowledge_base_lookup`
        # tool can scope retrievals to this agent's ingested documents.
        agent_instance.knowledge_base_ids = list(
            overrides.get(
                "knowledge_base_ids",
                getattr(saved_memagent, "knowledge_base_ids", None) or [],
            )
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

    def refresh(self):
        """
        Refresh the MemAgent from the memory provider.

        This method reloads the agent configuration from the memory provider,
        updating the current instance with any changes.

        Returns:
            MemAgent: Self if successful, False if failed
        """
        if not self.memory_provider:
            logger.error("Cannot refresh MemAgent: no memory provider configured")
            return False

        if not self.agent_id:
            logger.error("Cannot refresh MemAgent: no agent_id set")
            return False

        try:
            # Load fresh configuration from memory provider
            if hasattr(self.memory_provider, "retrieve_memagent"):
                saved_memagent = self.memory_provider.retrieve_memagent(self.agent_id)
                if saved_memagent:
                    # Update configuration attributes
                    if hasattr(saved_memagent, "instruction"):
                        self.instruction = saved_memagent.instruction
                    if hasattr(saved_memagent, "max_steps"):
                        self.max_steps = saved_memagent.max_steps
                    if hasattr(saved_memagent, "memory_ids"):
                        self.memory_ids = saved_memagent.memory_ids
                    if hasattr(saved_memagent, "knowledge_base_ids"):
                        self.knowledge_base_ids = (
                            list(saved_memagent.knowledge_base_ids)
                            if saved_memagent.knowledge_base_ids
                            else []
                        )
                    if hasattr(saved_memagent, "name"):
                        self.name = saved_memagent.name
                    if hasattr(saved_memagent, "is_favorite"):
                        self.is_favorite = bool(saved_memagent.is_favorite)
                    if hasattr(saved_memagent, "self_aware"):
                        self.with_self_aware(
                            bool(getattr(saved_memagent, "self_aware", False)),
                            config=getattr(saved_memagent, "self_aware_config", None),
                        )

                    # Update persona if changed. Route through the public
                    # ``set_persona`` wrapper so persona evolution tools
                    # (update_persona, read_persona) register when a persona
                    # is added via refresh(), not just via __init__.
                    if hasattr(saved_memagent, "persona") and self.persona_manager:
                        self.set_persona(saved_memagent.persona, save=False)

                    logger.info(f"MemAgent {self.agent_id} refreshed successfully")
                    return self
                else:
                    logger.error(
                        f"MemAgent {self.agent_id} not found in memory provider"
                    )
                    return False
            else:
                logger.error("Memory provider does not support retrieving MemAgent")
                return False

        except Exception as e:
            logger.error(f"Error refreshing MemAgent {self.agent_id}: {e}")
            return False

    def generate_summaries(
        self, days_back: int = 7, max_memories_per_summary: int = 50
    ) -> List[str]:
        """
        Generate summaries by compressing memory units from a specified time period.

        This method collects memory units from the specified time period,
        uses an LLM to compress them into emotionally and situationally relevant summaries,
        and stores them in the summaries collection.

        Parameters:
        -----------
        days_back : int, optional
            Number of days back to include in the summary (default: 7)
        max_memories_per_summary : int, optional
            Maximum number of memory units to include in each summary (default: 50)

        Returns:
        --------
        List[str]
            List of summary IDs that were created
        """
        try:
            import time

            from ..embeddings import get_embedding

            # Calculate time range (days_back days ago to now)
            current_time = time.time()
            start_time = current_time - (days_back * 24 * 60 * 60)

            logger.info(
                f"Generating summaries for agent {self.agent_id} from {days_back} days back"
            )
            logger.info(f"Agent memory_ids: {self.memory_ids}")
            logger.info(f"Current memory_id: {self._current_memory_id}")
            logger.info(f"Time range: {start_time} to {current_time}")

            # Ensure we have memory IDs to search
            memory_ids_to_search = self.memory_ids or []
            if (
                self._current_memory_id
                and self._current_memory_id not in memory_ids_to_search
            ):
                memory_ids_to_search = [self._current_memory_id] + memory_ids_to_search

            if not memory_ids_to_search:
                logger.warning(
                    f"Agent {self.agent_id} has no memory_ids to search for summaries"
                )
                return []

            logger.info(
                f"Searching {len(memory_ids_to_search)} memory_ids: {memory_ids_to_search}"
            )

            # Collect conversation memories from all memory IDs
            all_memories = []
            for memory_id in memory_ids_to_search:
                logger.info(
                    f"Retrieving conversation history for memory_id: {memory_id}"
                )
                try:
                    # Retrieve all conversation history
                    memories = self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(
                        memory_id=memory_id, include_embedding=False
                    )

                    if memories:
                        logger.info(
                            f"Retrieved {len(memories)} raw memories for memory_id: {memory_id}"
                        )

                        # Filter by time range
                        filtered = []
                        for idx, mem in enumerate(memories):
                            mem_timestamp = mem.get("timestamp")
                            original_timestamp = mem_timestamp

                            # Convert timestamp to float if needed
                            if isinstance(mem_timestamp, str):
                                try:
                                    from datetime import datetime

                                    # Try multiple timestamp formats
                                    if "T" in mem_timestamp:
                                        # ISO format
                                        mem_timestamp = datetime.fromisoformat(
                                            mem_timestamp.replace("Z", "+00:00")
                                        ).timestamp()
                                    else:
                                        # Try parsing as float string
                                        mem_timestamp = float(mem_timestamp)
                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse timestamp '{original_timestamp}' at index {idx}: {e}"
                                    )
                                    continue
                            elif hasattr(mem_timestamp, "timestamp"):
                                # datetime object
                                mem_timestamp = mem_timestamp.timestamp()
                            elif not isinstance(mem_timestamp, (int, float)):
                                logger.warning(
                                    f"Unknown timestamp type at index {idx}: {type(mem_timestamp)} = {original_timestamp}"
                                )
                                continue

                            # Convert to float
                            mem_timestamp = float(mem_timestamp)

                            # Debug first few timestamps
                            if idx < 3:
                                logger.info(
                                    f"Memory {idx}: timestamp={mem_timestamp}, start_time={start_time}, current_time={current_time}, in_range={start_time <= mem_timestamp <= current_time}"
                                )

                            if start_time <= mem_timestamp <= current_time:
                                filtered.append(mem)

                        logger.info(
                            f"Found {len(filtered)} memories within time range (out of {len(memories)} total) for memory_id: {memory_id}"
                        )
                        all_memories.extend(filtered)
                    else:
                        logger.info(f"No memories returned for memory_id: {memory_id}")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve memories for memory_id {memory_id}: {e}"
                    )
                    import traceback

                    logger.debug(traceback.format_exc())

            if not all_memories:
                logger.info(
                    f"No memories found for agent {self.agent_id} in the specified time range"
                )
                return []

            # Sort memories by timestamp
            def get_timestamp(mem):
                ts = mem.get("timestamp", 0)
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        return datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        ).timestamp()
                    except (ValueError, Exception):
                        return 0
                return float(ts) if isinstance(ts, (int, float)) else 0

            all_memories.sort(key=get_timestamp)

            logger.info(f"Found {len(all_memories)} memory units to summarize")

            # Split memories into chunks and create summaries
            summary_ids = []
            for i in range(0, len(all_memories), max_memories_per_summary):
                memory_chunk = all_memories[i : i + max_memories_per_summary]

                # Generate summary for this chunk
                summary_content = self._compress_memories_with_llm(memory_chunk)

                if summary_content:
                    # Get timestamps for period
                    period_start = get_timestamp(memory_chunk[0])
                    period_end = get_timestamp(memory_chunk[-1])

                    # Get the memory_id from the first memory in the chunk
                    chunk_memory_id = memory_chunk[0].get("memory_id")
                    if not chunk_memory_id:
                        # Fallback to current memory_id or first in list
                        chunk_memory_id = self._current_memory_id or (
                            self.memory_ids[0] if self.memory_ids else "default"
                        )

                    # Collect source message IDs for back-reference
                    source_message_ids = []
                    for mem in memory_chunk:
                        msg_id = (
                            mem.get("id") or mem.get("_id") or mem.get("memory_unit_id")
                        )
                        if msg_id:
                            source_message_ids.append(str(msg_id))

                    # Create summary document with source references
                    summary_doc = {
                        "memory_id": chunk_memory_id,
                        "agent_id": self.agent_id,
                        "content": summary_content,
                        "period_start": period_start,
                        "period_end": period_end,
                        "memory_units_count": len(memory_chunk),
                        "source_message_ids": source_message_ids,
                        "summary_type": "automatic",
                        "created_at": current_time,
                        "embedding": get_embedding(summary_content),
                    }

                    # Store summary
                    summary_id = self.memory_provider.store(
                        summary_doc, MemoryType.SUMMARIES
                    )
                    summary_ids.append(summary_id)

                    # Mark original messages as summarized so they are
                    # excluded from conversation history on future loads
                    if source_message_ids and self.memory_manager:
                        try:
                            self.memory_manager.mark_messages_as_summarized(
                                source_message_ids, summary_id
                            )
                            # Clear conversation cache so next load reflects changes
                            self.memory_manager.clear_conversation_cache(
                                chunk_memory_id
                            )
                        except Exception as mark_exc:
                            logger.debug(
                                "Could not mark messages as summarized: %s",
                                mark_exc,
                            )

                    logger.info(
                        f"Created summary {summary_id} for memory_id {chunk_memory_id} covering {len(memory_chunk)} memories"
                    )

            logger.info(
                f"Generated {len(summary_ids)} summaries for agent {self.agent_id}"
            )
            return summary_ids

        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def _compress_memories_with_llm(self, memories: List[Dict]) -> str:
        """
        Use LLM to compress memory units into an emotionally and situationally relevant summary.

        Parameters:
        -----------
        memories : List[Dict]
            List of memory units to compress

        Returns:
        --------
        str
            Compressed summary content
        """
        try:
            # Extract content from memories
            memory_contents = []
            for memory in memories:
                content = memory.get("content", "")
                role = memory.get("role", "")

                if content:
                    if role:
                        memory_contents.append(f"[{role}]: {content}")
                    else:
                        memory_contents.append(content)

            if not memory_contents:
                return ""

            # Create compression prompt
            memories_text = "\n".join(memory_contents)
            compression_prompt = f"""
Analyze the following memory units and create a concise summary that captures:
1. Emotionally significant moments and interactions
2. Situationally relevant context and patterns
3. Key achievements, challenges, or learning experiences
4. Important facts and information learned

Memory Units:
{memories_text}

Provide a comprehensive but concise summary:"""

            # Use the LLM to generate the summary
            if self.model:
                messages = [{"role": "user", "content": compression_prompt}]
                summary = self.model.generate(messages)
                self._record_context_window_usage(stage="memory_compression")
                return summary.strip()
            else:
                logger.warning("No LLM model available for memory compression")
                return ""

        except Exception as e:
            logger.error(f"Error compressing memories with LLM: {e}")
            return ""
