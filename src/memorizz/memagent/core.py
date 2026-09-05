# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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

from ..approval import (
    ApprovalRequired,
    ApprovalResumeResult,
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    default_approval_store,
)
from ..completion import (
    CompletionCandidate,
    CompletionDecision,
    CompletionPolicy,
    CompletionRejectedError,
)
from ..conversation_history import is_trace_bundle_entry
from ..enums import ApplicationMode, ApplicationModeConfig, MemoryType, Role
from ..internet_access import get_default_internet_access_provider
from ..llms.llm_factory import create_llm_provider
from ..long_term.semantic.entity_memory import EntityAttributeInput, EntityRelationInput
from ..streaming import StreamCancelled, check_cancelled, session_for
from ..task_decomposition import normalize_delegation_config
from ..tooling import (
    ContextPolicy,
    SemanticToolRouter,
    ToolOutcome,
    ToolOutcomeStatus,
    ToolResult,
    ToolResultPolicy,
    governed_tool,
    normalize_tool_result,
    serialize_tool_result,
    tool_metadata_to_openai,
)
from . import persistence
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
    BrowserControlManager,
    CacheManager,
    EntityMemoryManager,
    InternetAccessManager,
    MemoryManager,
    PersonaManager,
    SandboxManager,
    SelfAwarenessManager,
    ToolManager,
)
from .utils.context_dedup import dedupe_and_select, filter_skill_covered_workflows
from .utils.tool_log import (  # noqa: F401  — re-exported for compatibility
    _TOOL_LOG_PLACEHOLDER_PREFIX,
    _build_tool_log_placeholder,
    _extract_identifiers,
    _is_tool_placeholder_content,
    _summarize_tool_args,
    _summarize_tool_result,
    _to_jsonable,
)

if TYPE_CHECKING:
    from ..automation.models import AutomationJob, AutomationRun
    from ..browser_control import BrowserControlProvider
    from ..internet_access import InternetAccessProvider
    from ..sandbox.base import SandboxProvider
    from ..short_term_memory.semantic_cache import SemanticCacheInspection

logger = logging.getLogger(__name__)


_MIN_HISTORY_LIMIT = 24
_MAX_HISTORY_LIMIT = 120
_DEFAULT_HISTORY_LIMIT = 60
_FALLBACK_HISTORY_MESSAGES = 24
_PROMPT_WINDOW_RATIO = 0.8
_PROMPT_BUFFER_TOKENS = 200
# History is evicted in chunks of this many messages (rather than sliding by
# one or two messages every turn) so the prompt prefix stays byte-identical
# across turns — the precondition for OpenAI/Anthropic prompt-cache hits.
# A once-per-N-turns full-price request beats a cache miss on every turn.
_HISTORY_EVICTION_CHUNK = 20
# Host applications may attach content-free routing/grounding provenance to a
# run. Only this allowlist is persisted: arbitrary application dictionaries are
# deliberately rejected so observability cannot become a shadow prompt/content
# store. Values are bounded again when the trace bundle is compacted.
_OBSERVABILITY_CONTEXT_FIELDS = frozenset(
    {
        "request_id",
        "client_page_type",
        "client_page_id",
        "client_title_fingerprint",
        "canonical_page_type",
        "canonical_page_id",
        "canonical_title_fingerprint",
        "thread_binding_status",
        "expected_thread_id",
        "ownership_verified",
        "request_context_present",
        "content_version",
        "grounding_status",
        "grounding_source",
        "grounding_excerpt_count",
        "grounding_source_ids",
    }
)


_CONTEXT_LOCAL_MISSING = object()


class _ContextLocal:
    """Descriptor storing one value per thread/async context and agent.

    MemAgent instances are commonly registered as application singletons. A
    plain instance attribute therefore becomes a cross-request race under
    threaded or async servers. This descriptor preserves the existing private
    attribute API while isolating its value with ``ContextVar``.
    """

    def __init__(self, factory: Optional[Callable[[], Any]] = None):
        self.factory = factory or (lambda: None)
        self.name = ""
        self.storage_name = ""

    def __set_name__(self, owner: Any, name: str) -> None:
        self.name = name
        self.storage_name = f"__context_local_{name}"

    def _var(self, instance: Any) -> ContextVar:
        variable = instance.__dict__.get(self.storage_name)
        if variable is None:
            variable = ContextVar(
                f"memorizz_{id(instance)}_{self.name}",
                default=_CONTEXT_LOCAL_MISSING,
            )
            instance.__dict__[self.storage_name] = variable
        return variable

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        if instance is None:
            return self
        variable = self._var(instance)
        value = variable.get()
        if value is _CONTEXT_LOCAL_MISSING:
            value = self.factory()
            variable.set(value)
        return value

    def __set__(self, instance: Any, value: Any) -> None:
        self._var(instance).set(value)


def _provider_error_details(exc: Exception) -> Tuple[bool, bool, Optional[int]]:
    """Return ``(is_provider_error, is_auth_error, status_code)`` safely."""
    current: Optional[BaseException] = exc
    seen: Set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status_value = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_value is None and response is not None:
            status_value = getattr(response, "status_code", None)
        try:
            status_code = int(status_value) if status_value is not None else None
        except (TypeError, ValueError):
            status_code = None
        class_name = type(current).__name__.lower()
        module_name = type(current).__module__.lower()
        message = str(current).lower()
        is_auth = (
            status_code in {401, 403}
            or "authentication" in class_name
            or "unauthorized" in class_name
            or "invalid api key" in message
            or "incorrect api key" in message
        )
        is_provider = bool(
            status_code is not None
            or module_name.startswith(("openai", "anthropic", "google", "cohere"))
            or class_name.endswith(("apierror", "providererror"))
            or is_auth
        )
        if is_provider:
            return True, is_auth, status_code
        current = current.__cause__ or current.__context__
    return False, False, None


class MemAgent:
    """MemAgent class that orchestrates manager components."""

    _current_thread_id = _ContextLocal()
    _current_memory_id = _ContextLocal()
    _current_user_id = _ContextLocal()
    _current_run_id = _ContextLocal()
    _current_turn_id = _ContextLocal()
    _current_root_trace_id = _ContextLocal()
    _current_parent_span_id = _ContextLocal()
    _last_trace_context = _ContextLocal(dict)
    _last_tool_outcomes = _ContextLocal(list)
    _last_retrieved_memories = _ContextLocal(list)
    _last_retrieval_stats = _ContextLocal(dict)
    _last_selection_ledger = _ContextLocal(list)
    _last_memory_context_evidence = _ContextLocal(dict)
    _last_memory_attribution_context = _ContextLocal()
    _thread_ids_by_memory = _ContextLocal(dict)
    _stream_event_callback = _ContextLocal()
    _stream_trace_events = _ContextLocal()
    _turn_had_side_effects = _ContextLocal(lambda: False)
    _turn_had_nondeterministic_tools = _ContextLocal(lambda: False)
    _turn_cache_domains = _ContextLocal(set)
    _cache_bypass_reason = _ContextLocal()
    _activated_skill_ids = _ContextLocal(list)
    _approval_execution_active = _ContextLocal(lambda: False)
    _last_context_window_stats = _ContextLocal()
    _last_completion_decisions = _ContextLocal(list)

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
        browser_control: Optional[
            Union[str, Dict[str, Any], "BrowserControlProvider"]
        ] = None,
        meta_harness: Optional[Any] = None,
        meta_harness_mode: Optional[str] = None,
        default_harness: str = "auto",
        harness_config: Optional[Dict[str, Any]] = None,
        skill_paths: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        automations_enabled: bool = True,
        default_timezone: Optional[str] = None,
        self_aware: bool = False,
        self_aware_config: Optional[Dict[str, Any]] = None,
        continual_learning: bool = False,
        continual_learning_config: Optional[Dict[str, Any]] = None,
        learning_control_plane: Optional[Union[bool, Dict[str, Any], Any]] = None,
        workflow_outcome_evaluator: Optional[Callable[[Any], Any]] = None,
        toolbox: Optional[Any] = None,
        skillbox: Optional[Any] = None,
        authored_skills: Optional[List[Any]] = None,
        skill_retrieval: bool = False,
        skill_retrieval_config: Optional[Dict[str, Any]] = None,
        tool_result_policy: Optional[Union[ToolResultPolicy, Dict[str, Any]]] = None,
        context_policy: Optional[Union[ContextPolicy, Dict[str, Any]]] = None,
        completion_policy: Optional[
            Union[CompletionPolicy, Dict[str, Any], bool]
        ] = None,
        retrieval_policy: Optional[Union[Any, Dict[str, Any], str, bool]] = None,
        approval_store: Optional[ApprovalStore] = None,
        delegation: Optional[Dict[str, Any]] = None,
        semantic_layer: Optional[Any] = None,
        name: Optional[str] = None,
        application_id: Optional[str] = None,
        auto_register: bool = True,
        is_favorite: bool = False,
        streaming: bool = True,
    ):
        """Initialize the MemAgent with configuration."""
        self.streaming = streaming
        self.environment_reports: Dict[str, Any] = {}
        self._last_close_report: Optional[Dict[str, Any]] = None
        self.meta_harness = None
        self.meta_harness_mode = (
            str(meta_harness_mode).strip().lower() if meta_harness_mode else None
        )
        if self.meta_harness_mode not in {None, "delegate", "runtime"}:
            raise ValueError("meta_harness_mode must be delegate, runtime, or None")
        self.default_harness = (
            str(default_harness or "auto").strip().lower().replace("_", "-")
        )
        self.harness_config = dict(harness_config or {})
        self._owns_meta_harness = False
        self._meta_harness_tools_registered = False
        # Store configuration
        self.name = name.strip() if isinstance(name, str) and name.strip() else None
        configured_application_id = application_id or os.getenv(
            "MEMORIZZ_APPLICATION_ID", ""
        )
        self.application_id = (
            configured_application_id.strip()
            if isinstance(configured_application_id, str)
            and configured_application_id.strip()
            else None
        )
        # Explicit IDs always win. For named/application-scoped agents, derive
        # a UUID5 so a process restart cannot fragment traces into a new agent.
        # Anonymous unnamed agents retain the historical random-ID behavior.
        if agent_id:
            self.agent_id = str(agent_id)
        elif self.name or self.application_id:
            identity = f"{self.application_id or 'default'}:{self.name or 'agent'}"
            self.agent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:{identity}"))
        else:
            self.agent_id = str(uuid.uuid4())
        self.auto_register = bool(auto_register)
        self._registration_lock = threading.Lock()
        self._registered_memory_ids: Set[str] = set()
        self._registration_attempted = False
        self.is_favorite = bool(is_favorite)
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_steps = max_steps
        self.tool_access = str(tool_access or DEFAULT_TOOL_ACCESS).strip().lower()
        if self.tool_access not in {"private", "public", "global"}:
            raise ValueError("tool_access must be private, public, or global")
        # MemoRizz is memory-first by default. A bare MemAgent gets a durable,
        # zero-configuration filesystem provider; callers that intentionally
        # need a stateless agent can opt out explicitly with
        # ``memory_provider=False`` (an explicit empty ``memory_types=[]`` only
        # disables active memory systems, not agent/config persistence).
        self.uses_default_memory_provider = memory_provider is None
        if memory_provider is None:
            from ..memory_provider import create_default_memory_provider

            memory_provider = create_default_memory_provider()
        elif memory_provider is False:
            memory_provider = None
            self.uses_default_memory_provider = False
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
        self.toolbox = toolbox
        self.skillbox = skillbox
        self.authored_skills = list(authored_skills or [])
        self.skill_retrieval = bool(skill_retrieval)
        self.skill_retrieval_config = dict(skill_retrieval_config or {})
        self.context_policy = ContextPolicy.from_value(context_policy)
        self.completion_policy = CompletionPolicy.from_value(completion_policy)
        from ..retrieval import RetrievalPolicy

        self.retrieval_policy = RetrievalPolicy.from_value(retrieval_policy)
        self.tool_result_policy = ToolResultPolicy.from_value(tool_result_policy)
        self.approval_store = approval_store
        self.delegation_config = normalize_delegation_config(delegation)
        self.delegates = list(delegates or [])
        self.semantic_layer = semantic_layer
        self._approval_execution_active = False
        self._turn_had_side_effects = False
        self._turn_had_nondeterministic_tools = False
        self._turn_cache_domains: Set[str] = set()
        self._cache_bypass_reason: Optional[str] = None
        self._last_completion_decisions = []
        self._last_retrieved_memories: List[Dict[str, Any]] = []
        self._last_retrieval_stats: Dict[str, Any] = {
            "duration_ms": 0.0,
            "candidate_count": 0,
            "selected_count": 0,
        }
        self._last_memory_context_evidence: Dict[str, Any] = {}
        self._last_memory_attribution_context = None
        self.skill_paths = self._normalize_skill_paths(skill_paths)
        self.skills: List[Dict[str, Any]] = []
        # MCP is a first-class client subsystem. It owns protocol negotiation,
        # encrypted credentials, OAuth, policies, and transport lifecycles;
        # only its secret-free public configuration is persisted on the agent.
        from ..mcp import MCPClientManager

        self.mcp_manager = MCPClientManager(
            owner_id=self.agent_id,
            servers=mcp_servers or [],
        )
        self.mcp_servers = self.mcp_manager.server_dicts()
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
        self.continual_learning = bool(continual_learning) or (
            os.getenv("MEMORIZZ_CONTINUAL_LEARNING", "").strip().lower()
            in ("1", "true", "yes")
        )
        if self.continual_learning:
            self.skill_retrieval = True
        self.continual_learning_config = None
        if isinstance(continual_learning_config, dict):
            from ..long_term.procedural.skillbox import PromotionConfig

            validated_learning_config = PromotionConfig.from_dict(
                continual_learning_config
            )
            self.continual_learning_config = dict(continual_learning_config)
            # Keep persisted agent configs provider/JSON safe even when the
            # programmatic caller supplied SkillInjectionRole.DEVELOPER.
            self.continual_learning_config[
                "skill_injection_role"
            ] = validated_learning_config.skill_injection_role.value
            self.continual_learning_config[
                "require_shadow"
            ] = validated_learning_config.require_shadow
        from ..learning import LearningControlPlaneConfig

        self.learning_control_plane_config = LearningControlPlaneConfig.from_value(
            learning_control_plane, implied=self.continual_learning
        )
        self.learning_control_plane_enabled = bool(
            self.learning_control_plane_config.enabled
        )
        if workflow_outcome_evaluator is not None and not callable(
            workflow_outcome_evaluator
        ):
            raise TypeError("workflow_outcome_evaluator must be callable or None")
        # Runtime-only by design: application business rules are often closures
        # or service objects and cannot be serialized with the agent config.
        self.workflow_outcome_evaluator = workflow_outcome_evaluator

        (
            self.application_mode,
            self._application_mode_explicit,
        ) = self._resolve_application_mode(application_mode)

        # Store memory types for tracking which memory systems are active
        self.active_memory_types = self._initialize_memory_types(
            self.application_mode if self._application_mode_explicit else None,
            memory_types,
        )

        # Continual learning cannot learn without workflow capture, and its
        # skills live in the SKILLBOX partition — force both on when the
        # feature is enabled.
        if self.continual_learning:
            if MemoryType.WORKFLOW_MEMORY not in self.active_memory_types:
                if memory_types is not None:
                    logger.warning(
                        "continual_learning=True but the configured "
                        "memory_types omit workflow_memory — enabling it; "
                        "the learning loop cannot observe runs without it."
                    )
                self.active_memory_types.append(MemoryType.WORKFLOW_MEMORY)
            if MemoryType.SKILLBOX not in self.active_memory_types:
                self.active_memory_types.append(MemoryType.SKILLBOX)

        # Initialize thread state tracking
        self._current_thread_id = None
        self._current_memory_id = None
        self._current_user_id: Optional[str] = None
        self._thread_ids_by_memory: Dict[str, str] = {}

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
        # Context-summary generation runs off the hot path (daemon thread);
        # the flag prevents overlapping runs across turns.
        self._summary_thread_lock = threading.Lock()
        self._summary_generation_in_flight = False
        # Conversation rows are written with embedding=None and backfilled by
        # a single background worker so the hot path never blocks on the
        # embedding API. Disable via MEMORIZZ_DISABLE_CONVERSATION_EMBEDDINGS.
        self._conversation_embedding_enabled = os.getenv(
            "MEMORIZZ_DISABLE_CONVERSATION_EMBEDDINGS", ""
        ).strip().lower() not in ("1", "true", "yes")
        self._embedding_backfill_executor: Optional[ThreadPoolExecutor] = None
        self._internet_access_failure_count = 0
        self._internet_access_disabled_reason: Optional[str] = None
        self._sandbox_tools_registered = False
        self._sandbox_tool_names = (
            "execute_code",
            "sandbox_write_file",
            "sandbox_read_file",
        )
        self._browser_control_tools_registered = False
        self._browser_control_tool_names = ("browser_control",)
        self.browser_control_config: Optional[Dict[str, Any]] = None
        if isinstance(browser_control, str):
            self.browser_control_config = {"provider": browser_control}
        elif isinstance(browser_control, dict):
            self.browser_control_config = dict(browser_control)
        elif browser_control is not None and hasattr(browser_control, "get_config"):
            try:
                self.browser_control_config = dict(browser_control.get_config())
            except Exception:
                self.browser_control_config = None
        self._browser_control_init_error: Optional[str] = None
        self._skill_tools_registered = False
        self._mcp_tools_registered = False
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
        self._last_tool_outcomes: List[Dict[str, Any]] = []
        # Learned skills injected into the CURRENT turn's context — reset at
        # the top of every _build_context so attribution never leaks across
        # runs. Read by the workflow capture paths (skills_activated).
        self._activated_skill_ids: List[str] = []

        # Initialize manager components
        self._initialize_managers(
            memory_provider=memory_provider,
            semantic_cache=semantic_cache,
            semantic_cache_config=semantic_cache_config,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            internet_access_provider=internet_access_provider,
            sandbox_provider=sandbox_provider,
            browser_control=browser_control,
            self_aware_config=self_aware_config,
        )
        if self.toolbox is not None:
            self.tool_manager.initialize_from_toolbox(self.toolbox)
        self.semantic_tool_router = SemanticToolRouter(
            self.tool_manager,
            toolbox=self.toolbox,
            agent_id=self.agent_id,
            top_k=self.context_policy.tool_top_k,
            enabled=self.context_policy.progressive_tool_disclosure,
            always_visible={
                "retrieve_tool_log_entry",
                "expand_summary",
                "list_mcp_servers",
                "mcp_list_tools",
                "mcp_call_tool",
                "semantic_list_models",
                "list_agent_harnesses",
                "run_harness_task",
                "get_harness_run",
            },
            max_invocations_per_turn=self.context_policy.max_tool_invocations_per_turn,
            max_attempts_per_call=self.context_policy.max_tool_attempts_per_call,
        )
        self._register_context_monitor_tools()
        self._register_knowledge_base_tools()
        self._register_semantic_layer_tools()

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
        # Learned (skillbox) skills surface through the same list_skills /
        # read_skill tools, so registration is also needed when continual
        # learning is on even with no file-based skills configured.
        if self.skills or self.skillbox:
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

        if meta_harness:
            try:
                if hasattr(meta_harness, "run") and hasattr(
                    meta_harness, "list_harnesses"
                ):
                    self.meta_harness = meta_harness
                else:
                    from ..metaharness import MetaHarness

                    self.meta_harness = MetaHarness.from_env(
                        memory_provider=self.memory_provider,
                        agent=self,
                        approval_store=self.approval_store,
                    )
                    self._owns_meta_harness = True
                if self.meta_harness_mode is None:
                    self.meta_harness_mode = "delegate"
                if self.meta_harness_mode == "delegate":
                    self._register_meta_harness_tools()
            except Exception as exc:
                logger.warning("Meta-harness initialization failed: %s", exc)
                raise

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

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configured optional subsystems and return a stable report."""
        errors: List[str] = []
        if self.max_steps < 1:
            errors.append("max_steps must be at least 1")
        if self.sandbox_manager and self.sandbox_manager.provider:
            issue = self.sandbox_manager.provider.validate_configuration()
            if issue:
                errors.append(issue)
        if self.browser_control_manager and self.browser_control_manager.provider:
            issue = self.browser_control_manager.provider.validate_configuration()
            if issue:
                errors.append(issue)
        elif self.browser_control_config and self._browser_control_init_error:
            errors.append(self._browser_control_init_error)
        if self.approval_store is not None and not isinstance(
            self.approval_store, ApprovalStore
        ):
            errors.append("approval_store does not implement ApprovalStore")
        if self.delegates and not self.memory_provider:
            errors.append("delegation requires a shared memory provider")
        if self.meta_harness_mode and self.meta_harness is None:
            errors.append("meta_harness_mode requires a configured meta-harness")
        if self.meta_harness is not None and self.default_harness != "auto":
            normalized_harness = {
                "claude": "claude-code",
                "claudecode": "claude-code",
                "open-hands": "openhands",
                "native": "memagent",
                "memorizz": "memagent",
            }.get(self.default_harness, self.default_harness)
            adapters = getattr(self.meta_harness, "adapters", {})
            if normalized_harness not in adapters:
                errors.append(
                    f"default harness {self.default_harness!r} is not registered"
                )
            elif normalized_harness == "memagent":
                try:
                    capability = self.meta_harness.probe(normalized_harness)
                    if (
                        str((capability.get("metadata") or {}).get("agent_id") or "")
                        == self.agent_id
                    ):
                        errors.append(
                            "a MemAgent cannot use itself as its default native harness"
                        )
                except Exception as exc:
                    errors.append(f"default harness validation failed: {exc}")
        report = {"ok": not errors, "errors": errors, "agent_id": self.agent_id}
        if errors:
            raise ValueError("; ".join(errors))
        return report

    def capability_report(self, *, preflight: bool = False) -> Dict[str, Any]:
        """Return package and configured-agent feature/provider states."""
        from ..capabilities import capabilities

        provider = self.memory_provider
        report = capabilities()
        report["agent"] = {
            "agent_id": self.agent_id,
            "llm_provider": self.llm_provider
            or (type(self.model).__name__ if self.model is not None else None),
            "llm_model": self.llm_model or getattr(self.model, "model", None),
            "memory_provider": type(provider).__name__ if provider else None,
            "sandbox_provider": (
                type(self.sandbox_manager.provider).__name__
                if self.sandbox_manager and self.sandbox_manager.provider
                else None
            ),
            "browser_control_provider": (
                type(self.browser_control_manager.provider).__name__
                if self.browser_control_manager
                and self.browser_control_manager.provider
                else None
            ),
            "browser_control_configured": bool(self.browser_control_config),
            "browser_control_error": self._browser_control_init_error,
            "mcp_servers": self.mcp_manager.connection_status(),
            "semantic_cache": self.semantic_cache_stats(),
            "progressive_tool_disclosure": self.semantic_tool_router.enabled,
            "visible_tool_limit": (
                self.semantic_tool_router.top_k
                + len(self.semantic_tool_router.always_visible)
                + 2  # stable discovery and invocation meta-tools
                if self.semantic_tool_router.enabled
                else len(self.tool_manager.list_tools())
            ),
            "registered_tool_count": len(self.tool_manager.list_tools()),
            "skill_retrieval": bool(self.skill_retrieval),
            "continual_learning": bool(self.continual_learning),
            "learning_control_plane": {
                "enabled": self.learning_control_plane is not None,
                "configured": self.learning_control_plane_enabled,
                "evidence_token_budget": (
                    self.learning_control_plane_config.evidence_token_budget
                ),
                "compiler_enabled": self.learning_control_plane_config.compiler_enabled,
                "forgetting_enabled": self.learning_control_plane_config.forgetting_enabled,
            },
            "delegates": [item.agent_id for item in self.delegates],
            "delegation": {
                "enabled": bool(
                    self.delegates and self.delegation_config.get("enabled", True)
                ),
                "mode": self.delegation_config.get("mode"),
                "consolidation_strategy": self.delegation_config.get(
                    "consolidation_strategy", "model"
                ),
                "evidence_context": self.delegation_config.get(
                    "evidence_context", True
                ),
                "verified_read_only_cache_policy": True,
            },
            "durable_approval_store": (
                type(self.approval_store).__name__
                if self.approval_store is not None
                else "SQLiteApprovalStore (lazy)"
            ),
            "semantic_layer": self.semantic_layer is not None,
            "meta_harness": {
                "enabled": self.meta_harness is not None,
                "mode": self.meta_harness_mode,
                "default_harness": self.default_harness,
                "harnesses": (
                    self.meta_harness.list_harnesses()
                    if self.meta_harness is not None
                    else []
                ),
            },
        }
        if (
            preflight
            and provider is not None
            and callable(getattr(provider, "preflight", None))
        ):
            report["agent"]["provider_preflight"] = provider.preflight()
        return report

    def observability_summary(
        self,
        memory_id: str,
        user_id: Optional[str],
        *,
        thread_id: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Return tenant-scoped operational counts without exposing row bodies."""
        if not self.memory_provider:
            raise ValueError("observability_summary requires a memory provider")
        resolved_memory_id = str(memory_id or "").strip()
        if not resolved_memory_id:
            raise ValueError("memory_id is required")
        bounded_limit = max(1, min(int(limit), 10_000))

        def _timestamp(value: Any) -> Optional[float]:
            if value is None:
                return None
            if hasattr(value, "timestamp"):
                try:
                    return float(value.timestamp())
                except Exception:
                    return None
            try:
                return float(value)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(
                        str(value).replace("Z", "+00:00")
                    ).timestamp()
                except (TypeError, ValueError):
                    return None

        def _in_scope(row: Dict[str, Any], *, include_thread: bool = False) -> bool:
            if row.get("memory_id") != resolved_memory_id:
                return False
            if row.get("user_id") != user_id:
                return False
            row_agent = row.get("agent_id")
            if row_agent is not None and str(row_agent) != str(self.agent_id):
                return False
            if include_thread and thread_id is not None:
                row_thread = row.get("thread_id") or row.get("conversation_id") or ""
                if str(row_thread) != str(thread_id):
                    return False
            return True

        retrieve = (
            self.memory_provider.retrieve_conversation_history_ordered_by_timestamp
        )
        conversation_kwargs: Dict[str, Any] = {
            "memory_id": resolved_memory_id,
            "limit": bounded_limit,
        }
        try:
            parameters = inspect.signature(retrieve).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "include_embedding" in parameters:
            conversation_kwargs["include_embedding"] = False
        if "user_id" in parameters:
            conversation_kwargs["user_id"] = user_id
        if thread_id is not None and "thread_id" in parameters:
            conversation_kwargs["thread_id"] = thread_id
        conversation_rows = [
            row
            for row in (retrieve(**conversation_kwargs) or [])
            if isinstance(row, dict) and _in_scope(row, include_thread=True)
        ][:bounded_limit]

        role_counts: Dict[str, int] = {}
        trace_bundle_count = 0
        trace_event_count = 0
        timestamps: List[float] = []
        for row in conversation_rows:
            role = str(row.get("role") or "unknown").lower()
            role_counts[role] = role_counts.get(role, 0) + 1
            timestamp = _timestamp(row.get("timestamp") or row.get("created_at"))
            if timestamp is not None:
                timestamps.append(timestamp)
            if is_trace_bundle_entry(row):
                trace_bundle_count += 1
                try:
                    payload = json.loads(str(row.get("content") or "{}"))
                    trace_event_count += len(payload.get("events") or [])
                except (TypeError, ValueError):
                    pass

        manager = self.memory_manager or MemoryManager(self.memory_provider)
        tool_logs = [
            row
            for row in manager.list_tool_logs(
                resolved_memory_id,
                limit=bounded_limit,
                user_id=user_id,
                thread_id=thread_id,
            )
            if _in_scope(row, include_thread=True)
        ]

        def _scoped_rows(memory_type: MemoryType) -> List[Dict[str, Any]]:
            list_all = self.memory_provider.list_all
            try:
                rows = list_all(memory_type, user_id=user_id) or []
            except TypeError:
                rows = list_all(memory_type) or []
            return [row for row in rows if isinstance(row, dict) and _in_scope(row)][
                :bounded_limit
            ]

        workflows = _scoped_rows(MemoryType.WORKFLOW_MEMORY)
        summaries = _scoped_rows(MemoryType.SUMMARIES)

        approval_counts: Dict[str, int] = {}
        if self.approval_store is not None:
            for proposal in self.approval_store.list(owner_id=self.agent_id, limit=500):
                checkpoint = proposal.checkpoint or {}
                if checkpoint.get("memory_id") != resolved_memory_id:
                    continue
                if checkpoint.get("user_id") != user_id:
                    continue
                if thread_id is not None and str(
                    checkpoint.get("thread_id") or ""
                ) != str(thread_id):
                    continue
                status = proposal.status.value
                approval_counts[status] = approval_counts.get(status, 0) + 1

        workflow_outcomes: Dict[str, int] = {}
        for workflow in workflows:
            outcome = str(
                workflow.get("outcome") or workflow.get("status") or "unknown"
            )
            workflow_outcomes[outcome] = workflow_outcomes.get(outcome, 0) + 1
        tool_failures = sum(
            1
            for row in tool_logs
            if row.get("success") is False or bool(row.get("error"))
        )
        tool_outcomes: Dict[str, int] = {}
        for row in tool_logs:
            outcome = str(
                row.get("outcome")
                or ("success" if row.get("success") is not False else "error")
            ).lower()
            tool_outcomes[outcome] = tool_outcomes.get(outcome, 0) + 1

        return {
            "agent_id": self.agent_id,
            "memory_id": resolved_memory_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "conversation": {
                "row_count": len(conversation_rows),
                "message_count": len(conversation_rows) - trace_bundle_count,
                "role_counts": role_counts,
                "thread_count": len(
                    {
                        str(row.get("thread_id") or row.get("conversation_id") or "")
                        for row in conversation_rows
                    }
                ),
                "summarized_count": sum(
                    1 for row in conversation_rows if row.get("summary_id")
                ),
                "trace_bundle_count": trace_bundle_count,
                "trace_event_count": trace_event_count,
                "first_timestamp": min(timestamps) if timestamps else None,
                "last_timestamp": max(timestamps) if timestamps else None,
            },
            "tool_logs": {
                "count": len(tool_logs),
                "failure_count": tool_failures,
                "success_count": len(tool_logs) - tool_failures,
                "outcomes": tool_outcomes,
            },
            "workflows": {
                "count": len(workflows),
                "outcomes": workflow_outcomes,
            },
            "summaries": {"count": len(summaries)},
            "approvals": {
                "count": sum(approval_counts.values()),
                "statuses": approval_counts,
            },
            "semantic_cache": self.semantic_cache_stats(),
            "context_window": self.get_context_window_stats(),
            "truncated": any(
                len(rows) >= bounded_limit
                for rows in (conversation_rows, tool_logs, workflows, summaries)
            ),
        }

    def learning_report(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return control-plane events, artifacts and latest decisions."""
        if self.learning_control_plane is None:
            return {
                "enabled": False,
                "agent_id": self.agent_id,
                "configured": self.learning_control_plane_enabled,
            }
        return self.learning_control_plane.report(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        )

    def explain_memory_decision(
        self, *, include_content: bool = False
    ) -> Dict[str, Any]:
        """Explain why the last EvidencePack selected or rejected memories."""
        if self.learning_control_plane is None:
            return {"available": False, "reason": "learning control plane disabled"}
        return self.learning_control_plane.explain_last_retrieval(
            include_content=include_content
        )

    def compile_memory(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """Synchronously compile pending learning events into retrieval artifacts."""
        if self.learning_control_plane is None:
            raise ValueError("learning control plane is disabled")
        return self.learning_control_plane.compile(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            mode=mode,
        ).to_dict()

    def plan_forgetting(
        self,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        max_candidates: int = 500,
    ):
        """Create a dry-run, reversible forgetting plan."""
        if self.learning_control_plane is None:
            raise ValueError("learning control plane is disabled")
        return self.learning_control_plane.plan_forgetting(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            max_candidates=max_candidates,
        )

    def apply_forgetting(
        self,
        report,
        *,
        approved_by: str,
        reason: Optional[str] = None,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ):
        """Apply a dry-run forgetting plan with an auditable operator identity."""
        if self.learning_control_plane is None:
            raise ValueError("learning control plane is disabled")
        resolved_user_id = self._current_user_id if user_id is ... else user_id
        return self.learning_control_plane.apply_forgetting(
            report,
            approved_by=approved_by,
            reason=reason,
            scope={
                "memory_id": memory_id or self._current_memory_id,
                "user_id": resolved_user_id,
                "thread_id": thread_id or self._current_thread_id,
            },
        )

    def get_forgetting_plan(
        self,
        plan_id: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Any = ...,
        thread_id: Optional[str] = None,
    ):
        """Load a durable forgetting plan for approval after a restart."""
        if self.learning_control_plane is None:
            raise ValueError("learning control plane is disabled")
        return self.learning_control_plane.get_forgetting_plan(
            plan_id,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )

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

            # An explicitly empty collection means that the caller wants a
            # stateless agent.  Falling through to the defaults here used to
            # re-enable conversation, workflow, and summary memory, which was
            # both surprising and capable of triggering persistence attempts
            # when no provider was configured.
            if raw_values == []:
                return []

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
        browser_control=None,
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

        # Browser Control Manager. Its single model-facing tool is explicitly
        # side-effecting and therefore always enters the durable approval path.
        if browser_control:
            try:
                self.browser_control_manager = BrowserControlManager.from_config(
                    browser_control
                )
                self.browser_control_config = (
                    self.browser_control_manager.get_provider_config()
                )
                self._browser_control_init_error = None
                self._register_browser_control_tools()
            except Exception as exc:
                logger.warning("Browser-control provider failed to initialize: %s", exc)
                self.browser_control_manager = None
                self._browser_control_init_error = str(exc)
        else:
            self.browser_control_manager = None

        # Self-awareness manager (host codebase awareness and guarded file/CLI ops)
        self.self_awareness_manager = SelfAwarenessManager(config=self_aware_config)
        self.self_aware_config = self.self_awareness_manager.get_config()

        # Continual learning manager (workflow → skill promotion loop).
        # Only constructed when the feature flag is on AND a provider
        # exists — the loop is inert without a place to store skills.
        self.continual_learning_manager = None
        if self.continual_learning:
            if memory_provider:
                try:
                    from .managers.continual_learning_manager import (
                        ContinualLearningManager,
                    )

                    self.continual_learning_manager = ContinualLearningManager(
                        memory_provider=memory_provider,
                        llm_provider=self.model,
                        agent_id=self.agent_id,
                        config=self.continual_learning_config,
                        tool_manager=self.tool_manager,
                    )
                except Exception as exc:
                    logger.warning("Continual learning failed to initialize: %s", exc)
            else:
                logger.warning(
                    "continual_learning=True requires a memory provider — "
                    "feature disabled for this agent."
                )

        if self.continual_learning_manager is not None:
            self.skillbox = self.continual_learning_manager.skillbox
        elif (
            self.skillbox is None
            and (self.skill_retrieval or self.authored_skills)
            and memory_provider
        ):
            from ..long_term.procedural.skillbox import Skillbox

            self.skillbox = Skillbox(
                memory_provider=memory_provider,
                llm_provider=self.model,
                agent_id=self.agent_id,
            )
        if self.authored_skills:
            self._persist_authored_skills()

        # A thin event/evidence layer over the existing provider,
        # observability store, and workflow-to-skill loop.
        self.learning_control_plane = None
        if self.learning_control_plane_enabled:
            if not memory_provider:
                logger.warning(
                    "learning_control_plane is enabled but no memory provider is "
                    "available; the control plane is disabled"
                )
            else:
                try:
                    from ..learning import LearningControlPlane

                    self.learning_control_plane = LearningControlPlane(
                        memory_provider,
                        agent_id=self.agent_id,
                        config=self.learning_control_plane_config,
                    )
                    if self.continual_learning_manager is not None:
                        self.continual_learning_manager.control_plane = (
                            self.learning_control_plane
                        )
                except Exception as exc:
                    if not self.learning_control_plane_config.fail_open:
                        raise
                    logger.warning(
                        "Learning control plane failed to initialize: %s", exc
                    )

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

        context_window = self._context_window_tokens
        if not context_window or context_window <= 0:
            ideal_start = max(0, len(normalized) - _FALLBACK_HISTORY_MESSAGES)
            return normalized[self._quantize_history_start(ideal_start, normalized) :]

        base_tokens = (
            self._estimate_text_tokens(system_prompt)
            + self._estimate_text_tokens(query)
            + _PROMPT_BUFFER_TOKENS
        )
        prompt_budget = max(512, int(context_window * _PROMPT_WINDOW_RATIO))
        history_budget = max(160, prompt_budget - base_tokens)

        # Find the earliest message index whose suffix fits the token budget
        # (and the message-count limit).
        total = len(normalized)
        ideal_start = max(0, total - history_limit)
        consumed = 0
        budget_start = total - 1  # always keep at least the newest message
        for index in range(total - 1, -1, -1):
            message_cost = (
                self._estimate_text_tokens(normalized[index].get("content")) + 6
            )
            if index < total - 1 and (consumed + message_cost) > history_budget:
                break
            consumed += message_cost
            budget_start = index
            if consumed >= history_budget:
                break
        ideal_start = max(ideal_start, budget_start)

        return normalized[self._quantize_history_start(ideal_start, normalized) :]

    @staticmethod
    def _quantize_history_start(ideal_start: int, normalized: List[Any]) -> int:
        """Quantize the history window start to eviction-chunk boundaries.

        Recomputing an exact token-fitted window every turn moves the first
        history message every turn, which changes the prompt prefix and
        invalidates the provider prompt cache for the whole conversation.
        Rounding the start UP to the next multiple of ``_HISTORY_EVICTION_CHUNK``
        keeps the window byte-stable for ~a chunk's worth of turns (dropping
        slightly more history than strictly necessary, which the budget's
        ``_PROMPT_WINDOW_RATIO`` headroom absorbs) and then evicts a whole
        chunk at once — one cache miss per chunk instead of one per turn.
        """
        if ideal_start <= 0:
            return 0
        quantized = (
            (ideal_start + _HISTORY_EVICTION_CHUNK - 1)
            // _HISTORY_EVICTION_CHUNK
            * _HISTORY_EVICTION_CHUNK
        )
        # Never quantize away the entire window.
        return min(quantized, max(len(normalized) - 1, 0))

    def _build_prompt_messages(
        self,
        system_prompt: str,
        query: str,
        context: Dict[str, Any],
        request_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build model input messages ordered stable-prefix → volatile-tail.

        Layout (prompt-cache friendly — both OpenAI's automatic prefix cache
        and Anthropic's cache_control breakpoints require the byte-identical
        prefix to come first):

        1. Static system prompt — frozen for the session.
        2. Conversation history — append-only, chunk-evicted.
        3. Optional developer message — reviewed, developer-authority skills.
        4. Final user message — per-turn volatile block (user-authority skills,
           retrieved memories,
           entity facts, tool-log digest, ``request_context``) followed by
           the user's query. Everything that changes per turn lives here, at
           the very end, where it invalidates nothing.

        ``request_context`` (M2 per-call ephemeral context) is rendered
        inside the volatile block. It is intentionally NOT persisted by
        ``_record_interaction`` (which only stores the original ``query``
        string), so it does not pollute ``conversation_memory``.
        """
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        history = context.get("conversation_history", [])
        messages.extend(self._prepare_history_messages(history, system_prompt, query))

        (
            developer_skills,
            user_skills,
            rendered_skills,
        ) = self._render_activated_skill_sections(context)
        if developer_skills:
            messages.append(
                {
                    "role": "developer",
                    "content": (
                        "<memorizz:approved-skills>\n"
                        + developer_skills
                        + "\n</memorizz:approved-skills>"
                    ),
                }
            )

        volatile_block = self._build_volatile_context_block(
            context,
            request_context,
            learned_skills_section=user_skills,
            rendered_skills=rendered_skills,
        )
        if volatile_block:
            user_content = (
                "<memorizz:context>\n"
                + volatile_block
                + "\n</memorizz:context>\n\n"
                + query
            )
        else:
            user_content = query
        messages.append({"role": "user", "content": user_content})
        # Sanitize once at the boundary so any LOB that slipped through from
        # memory loaders (conversation history, KB retrievals, summaries…)
        # doesn't blow up the LLM provider's json.dumps during streaming.
        return _to_jsonable(messages)

    def _render_activated_skill_sections(
        self, context: Dict[str, Any]
    ) -> Tuple[str, str, List[Any]]:
        """Render active skills into developer- and user-authority groups."""
        activated_skills = list(context.get("activated_skills") or [])
        if not activated_skills:
            return "", "", []

        grouped: Dict[str, List[Any]] = {"developer": [], "user": []}
        for scored in activated_skills:
            skill = getattr(scored, "skill", None) or scored
            role = getattr(skill, "injection_role", "user")
            role_value = str(getattr(role, "value", role) or "user").lower()
            grouped["developer" if role_value == "developer" else "user"].append(scored)

        rendered: List[Any] = []
        sections: Dict[str, str] = {"developer": "", "user": ""}
        for role in ("developer", "user"):
            skills = grouped[role]
            if not skills:
                continue
            try:
                if self.continual_learning_manager:
                    section = (
                        self.continual_learning_manager.format_skills_prompt_section(
                            skills, injection_role=role
                        )
                    )
                else:
                    section = "\n\n".join(
                        "### {name}\nWhen to apply: {description}\n{content}".format(
                            name=(getattr(item, "skill", item)).name,
                            description=(getattr(item, "skill", item)).description,
                            content=(getattr(item, "skill", item)).content,
                        )
                        for item in skills
                    )
            except TypeError:
                # Backward-compatible seam for custom managers implementing
                # the pre-authority one-argument formatter.
                section = self.continual_learning_manager.format_skills_prompt_section(
                    skills
                )
            except Exception as exc:
                logger.warning("Failed to render %s learned skills: %s", role, exc)
                continue
            if section:
                sections[role] = section
                rendered.extend(skills)
        return sections["developer"], sections["user"], rendered

    def _build_volatile_context_block(
        self,
        context: Dict[str, Any],
        request_context: Optional[Dict[str, Any]] = None,
        learned_skills_section: Optional[str] = None,
        rendered_skills: Optional[List[Any]] = None,
    ) -> str:
        """Render the per-turn context that must NOT live in the system prompt.

        These sections change turn to turn (retrieved memories, entity facts,
        the tool-log digest, per-call request context). Injected into the
        final user message so the stable prefix — system prompt + history —
        stays byte-identical and prompt caches keep hitting. Content here is
        never persisted to conversation_memory.
        """
        sections: List[str] = []

        # User-authority learned skills lead the volatile block. Developer
        # skills were rendered as a separate message immediately before this
        # final user turn. Direct callers of this helper retain legacy behavior.
        if learned_skills_section is None and rendered_skills is None:
            (
                developer_section,
                learned_skills_section,
                rendered_skills,
            ) = self._render_activated_skill_sections(context)
            # This lower-level helper has no message-role channel of its own.
            # Keep direct-call compatibility by rendering every skill rather
            # than suppressing a covered workflow without its replacement.
            learned_skills_section = "\n\n".join(
                section
                for section in (developer_section, learned_skills_section)
                if section
            )
        if learned_skills_section:
            sections.append(learned_skills_section)
        rendered_skills = list(rendered_skills or [])

        evidence_pack = context.get("evidence_pack")
        if evidence_pack is not None:
            try:
                rendered_evidence = evidence_pack.render()
            except Exception:
                rendered_evidence = ""
            if rendered_evidence:
                sections.append(rendered_evidence)

        # Final context-boundary guard: even callers that manually inject
        # workflow memory cannot co-inject a raw run covered by a skill that
        # was rendered above.
        retrieved = filter_skill_covered_workflows(
            context.get("retrieved_memories") or [],
            rendered_skills,
        )
        if retrieved and evidence_pack is None:
            lines = []
            for item in retrieved:
                source = item.get("source") or "memory"
                source_id = (
                    item.get("parent_source_id")
                    or item.get("source_id")
                    or item.get("id")
                )
                linked_source_ids = list(item.get("linked_source_ids") or [])
                if linked_source_ids:
                    identifier = "; ids=" + ",".join(
                        str(value) for value in linked_source_ids
                    )
                else:
                    identifier = f"; id={source_id}" if source_id else ""
                stamp = ""
                ts = item.get("timestamp")
                if ts:
                    try:
                        stamp = datetime.fromtimestamp(float(ts)).strftime(" %Y-%m-%d")
                    except Exception:
                        stamp = ""
                lines.append(f"• [{source}{stamp}{identifier}] {item.get('text', '')}")
            sections.append(
                "Relevant memories retrieved for this turn (deduplicated; may "
                "be incomplete — use your memory tools for anything deeper):\n"
                + "\n".join(lines)
            )

        entity_profiles = context.get("entity_memory_profiles") or []
        if (
            entity_profiles
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            try:
                entity_summary = self.entity_memory_manager.summarize_for_prompt(
                    entity_profiles
                )
            except Exception:
                entity_summary = ""
            if entity_summary:
                sections.append(
                    "Entity memory facts:\n"
                    + entity_summary
                    + "\nUse the entity memory tools to keep these facts up to date."
                )

        personalization_value = context.get("personalization_context")
        if personalization_value:
            try:
                from ..personalization import PersonalizationContext

                personalization = PersonalizationContext.from_value(
                    personalization_value
                )
                rendered_personalization = personalization.render()
            except Exception as exc:
                logger.warning("Personalization context rendering failed: %s", exc)
                rendered_personalization = ""
            if rendered_personalization:
                sections.append(
                    "Personalization context for this turn:\n"
                    + rendered_personalization
                )

        summaries = context.get("summaries") or []
        if summaries:
            summary_lines = []
            for entry in summaries[:10]:
                sid = entry.get("summary_id") or ""
                desc = entry.get("short_description") or ""
                if sid:
                    summary_lines.append(f"• {sid}: {desc}")
            if summary_lines:
                sections.append(
                    "Compressed conversation summaries available via "
                    "`expand_summary('<summary_id>')`:\n" + "\n".join(summary_lines)
                )

        # Durable digest of this thread's tool_log entries — the placeholder
        # rows are filtered out of LLM history, so this is how tool_log_ids
        # stay pickable across turns.
        recent_digest = self._format_recent_tool_logs_digest(limit=10)
        if recent_digest:
            sections.append(recent_digest)

        if request_context:
            visible_request_context = dict(request_context)
            # The typed block above is already rendered with its own authority
            # and usage rules. Avoid injecting a duplicate raw JSON copy.
            visible_request_context.pop("personalization_context", None)
            try:
                rendered_context = json.dumps(
                    visible_request_context,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                    sort_keys=True,
                )
            except Exception:
                rendered_context = str(request_context)
            sections.append(
                "REQUEST CONTEXT (ephemeral, this turn only — do not "
                "reference unless directly relevant to the user's "
                "current query):\n" + rendered_context
            )

        return "\n\n".join(sections)

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
                user_id=self._current_user_id,
                thread_id=self._current_thread_id,
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
            summary_doc = self.fetch_context_summary(
                summary_id,
                memory_id=self._current_memory_id,
                user_id=self._current_user_id,
                thread_id=self._current_thread_id,
            )
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
            source_ids = (
                summary_doc.get("source_message_ids")
                or summary_doc.get("original_memory_ids")
                or []
            )
            if source_ids and self.memory_manager:
                original_msgs = self.memory_manager.get_messages_by_ids(
                    source_ids,
                    user_id=self._current_user_id,
                    thread_id=self._current_thread_id,
                )
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
                memory_id=self._current_memory_id,
                user_id=self._current_user_id,
                thread_id=self._current_thread_id,
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
                return {
                    "ok": False,
                    "error_code": "tool_log_unavailable",
                    "error": "No memory manager available.",
                    "tool_log_id": tool_log_id,
                }
            result = self.memory_manager.retrieve_tool_log(
                tool_log_id,
                user_id=self._current_user_id,
            )
            if not result:
                return {
                    "ok": False,
                    "error_code": "tool_log_not_found",
                    "error": f"Tool log '{tool_log_id}' not found.",
                    "tool_log_id": tool_log_id,
                }
            return {"ok": True, "tool_log_id": tool_log_id, "tool_log": result}

        def list_recent_tool_logs(limit: int = 10) -> Dict[str, Any]:
            """List recent tool execution logs for the current thread.

            Each entry includes a short ``args_summary`` and field-aware
            ``digest`` so the agent can pick the right ``tool_log_id`` to
            unpack without first fetching every candidate.
            """
            if not self.memory_manager or not self._current_memory_id:
                return {"error": "No memory context available.", "logs": []}
            raw_logs = self.memory_manager.list_tool_logs(
                memory_id=self._current_memory_id,
                limit=limit,
                user_id=self._current_user_id,
                thread_id=self._current_thread_id,
            )
            if not isinstance(raw_logs, (list, tuple)):
                return {
                    "error": "Memory provider returned an invalid tool-log result.",
                    "logs": [],
                }
            enriched: List[Dict[str, Any]] = []
            for row in raw_logs:
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
        """Automatically summarize when context window nears limits.

        Summarization is LLM- and DB-heavy, so it runs on a daemon thread
        ("sleep-time" consolidation) instead of blocking the user-facing
        turn; the in-flight flag prevents overlapping runs.
        """
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

        with self._summary_thread_lock:
            if self._summary_generation_in_flight:
                return
            self._summary_generation_in_flight = True
            # Stamp inside the lock so a burst of turns can't all pass the
            # cooldown check before the worker finishes.
            self._last_summary_timestamp = current_time

        token_estimate = stats.get("total_tokens")
        # Capture the complete request scope before starting the worker. A
        # later concurrent turn may update ``_current_*`` while this thread is
        # waiting to run.
        summary_memory_id = self._current_memory_id
        summary_user_id = self._current_user_id
        summary_thread_id = self._current_thread_id

        def _summarize() -> None:
            try:
                summary_ids = self.generate_summaries(
                    memory_id=summary_memory_id,
                    user_id=summary_user_id,
                    thread_id=summary_thread_id,
                    days_back=1,
                    max_memories_per_summary=20,
                )
                if summary_ids:
                    self._track_summary_ids(summary_ids, token_estimate=token_estimate)
            except Exception as exc:
                logger.warning("Background context summarization failed: %s", exc)
            finally:
                with self._summary_thread_lock:
                    self._summary_generation_in_flight = False

        threading.Thread(
            target=_summarize, daemon=True, name="memorizz-context-summary"
        ).start()

    def list_context_summaries(self) -> List[Dict[str, Any]]:
        """Return summary registry entries."""
        return list(self._summary_registry)

    def fetch_context_summary(
        self,
        summary_id: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a summary only inside the requested tenant/thread scope."""
        if not self.memory_provider or not hasattr(
            self.memory_provider, "retrieve_by_id"
        ):
            return None
        try:
            document = self.memory_provider.retrieve_by_id(
                summary_id, MemoryType.SUMMARIES
            )
            if not isinstance(document, dict):
                return None
            if document.get("user_id") != user_id:
                return None
            if memory_id is not None and str(document.get("memory_id") or "") != str(
                memory_id
            ):
                return None
            if thread_id is not None and self.memory_manager._entry_thread_id(
                document
            ) != str(thread_id):
                return None
            return document
        except Exception as exc:
            logger.warning("Failed to fetch summary %s: %s", summary_id, exc)
            return None

    def _initialize_tools(self, tools):
        """Initialize tools using the tool manager."""
        from ..long_term.procedural.toolbox import Toolbox

        if isinstance(tools, Toolbox):
            self.toolbox = tools
            self.tool_manager.initialize_from_toolbox(tools)
            if getattr(self, "semantic_tool_router", None) is not None:
                self.semantic_tool_router.toolbox = tools
            return
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

    def _persist_authored_skills(self, *, strict: bool = False) -> List[str]:
        """Persist explicitly authored skills without enabling learning machinery."""
        if self.skillbox is None:
            message = "Authored skills require a Skillbox or memory provider"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return []
        from ..long_term.procedural.skillbox import Skill, SkillStatus

        persisted: List[str] = []
        failures: List[str] = []
        for value in self.authored_skills:
            try:
                if isinstance(value, Skill):
                    skill = value
                elif isinstance(value, dict):
                    payload = dict(value)
                    payload.setdefault("status", SkillStatus.ACTIVE.value)
                    payload.setdefault("agent_id", self.agent_id)
                    if "embedding" in payload:
                        skill = Skill.from_dict(payload)
                    else:
                        skill = Skill(
                            name=str(payload.get("name") or ""),
                            description=str(payload.get("description") or ""),
                            content=str(payload.get("content") or ""),
                            preconditions=payload.get("preconditions"),
                            tools_used=payload.get("tools_used"),
                            queries=payload.get("queries"),
                            skill_id=payload.get("skill_id"),
                            agent_id=payload.get("agent_id"),
                            user_id=payload.get("user_id"),
                            status=payload.get("status"),
                            injection_role=payload.get("injection_role", "user"),
                        )
                else:
                    raise TypeError("Authored skills must be Skill instances or dicts")
                skill.agent_id = skill.agent_id or self.agent_id
                skill.status = SkillStatus.ACTIVE
                existing = self.skillbox.get_skill_by_name(skill.name)
                if existing is None:
                    persisted.append(self.skillbox.add_skill(skill))
                else:
                    persisted.append(str(existing.skill_id))
            except Exception as exc:
                logger.warning("Failed to persist authored skill: %s", exc)
                failures.append(str(exc))
        if strict and failures:
            raise RuntimeError(
                "One or more authored skills could not be persisted: "
                + "; ".join(failures)
            )
        return persisted

    def _normalize_mcp_servers(
        self, mcp_servers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Normalize MCP entries through the first-class manager."""
        from ..mcp import MCPClientManager

        manager = MCPClientManager(
            owner_id=getattr(self, "agent_id", "default"),
            servers=mcp_servers or [],
        )
        return manager.server_dicts()

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
        """Resolve a skill by name or file path.

        Falls back to learned (skillbox) skills so ``read_skill`` works on
        promoted skills with zero extra plumbing.
        """
        if self.skills:
            if not skill_name:
                return self.skills[0]
            for skill in self.skills:
                if skill_name in {
                    skill.get("name"),
                    skill.get("path"),
                    Path(skill.get("path", "")).name,
                }:
                    return skill
        if skill_name and self.skillbox:
            try:
                learned = self.skillbox.get_skill_by_name(skill_name)
            except Exception:
                learned = None
            if learned:
                return {
                    "name": learned.name,
                    "path": None,
                    "base_dir": None,
                    "description": learned.description,
                    "content": learned.content,
                    "code_blocks": [],
                    "script_paths": [],
                    "source": "learned",
                    "version": learned.version,
                    "status": learned.status.value,
                }
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
            """List loaded skill files, executable snippets, and learned skills."""
            entries = [
                {
                    "name": skill.get("name"),
                    "path": skill.get("path"),
                    "description": skill.get("description"),
                    "snippet_count": len(skill.get("code_blocks", [])),
                    "script_paths": skill.get("script_paths", []),
                    "source": "file",
                }
                for skill in self.skills
            ]
            if self.skillbox:
                try:
                    from ..long_term.procedural.skillbox import SkillStatus

                    for learned in self.skillbox.list_skills(
                        statuses=(SkillStatus.ACTIVE,)
                    ):
                        entries.append(
                            {
                                "name": learned.name,
                                "path": None,
                                "description": learned.description,
                                "snippet_count": 0,
                                "script_paths": [],
                                "source": "learned",
                                "version": learned.version,
                                "injection_role": learned.injection_role.value,
                            }
                        )
                except Exception as exc:
                    logger.debug("Listing learned skills failed: %s", exc)
            return {"skills": entries}

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
        """Compatibility dispatcher retained for callers of the old private helper."""
        params = params or {}
        if method == "tools/list":
            return self.mcp_manager.list_tools(server_name)
        if method == "tools/call":
            return self.mcp_manager.call_tool(
                server_name=server_name,
                tool_name=str(params.get("name") or ""),
                arguments=params.get("arguments") or {},
            )
        return {
            "ok": False,
            "error": f"Unsupported MCP compatibility method '{method}'.",
            "error_code": "unsupported_method",
        }

    def _register_mcp_tools(self):
        """Register production MCP discovery and invocation tools."""
        if self._mcp_tools_registered or not self.tool_manager:
            return

        def list_mcp_servers() -> Dict[str, Any]:
            """List configured MCP servers and their authentication status."""
            return self.mcp_manager.connection_status()

        def mcp_list_tools(server_name: str) -> Dict[str, Any]:
            """List tools available from a configured MCP server."""
            return self.mcp_manager.list_tools(server_name)

        @governed_tool(deterministic=False, side_effects=False, domains=("mcp",))
        def mcp_call_tool(
            server_name: str,
            tool_name: str,
            arguments: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Call an MCP tool through the agent's durable approval policy."""
            method = (
                self.mcp_manager._call_tool_authorized
                if self._approval_execution_active
                else self.mcp_manager.call_tool
            )
            return method(
                server_name=server_name,
                tool_name=tool_name,
                arguments=arguments or {},
            )

        def mcp_list_resources(server_name: str) -> Dict[str, Any]:
            """List resources exposed by a configured MCP server."""
            return self.mcp_manager.list_resources(server_name)

        def mcp_list_resource_templates(server_name: str) -> Dict[str, Any]:
            """List parameterized resource templates exposed by an MCP server."""
            return self.mcp_manager.list_resource_templates(server_name)

        def mcp_read_resource(server_name: str, uri: str) -> Dict[str, Any]:
            """Read a resource from a configured MCP server."""
            return self.mcp_manager.read_resource(server_name, uri)

        def mcp_list_prompts(server_name: str) -> Dict[str, Any]:
            """List prompts exposed by a configured MCP server."""
            return self.mcp_manager.list_prompts(server_name)

        def mcp_get_prompt(
            server_name: str,
            prompt_name: str,
            arguments: Optional[Dict[str, str]] = None,
        ) -> Dict[str, Any]:
            """Get a prompt template from a configured MCP server."""
            return self.mcp_manager.get_prompt(
                server_name, prompt_name, arguments=arguments or {}
            )

        list_mcp_servers.__name__ = "list_mcp_servers"
        mcp_list_tools.__name__ = "mcp_list_tools"
        mcp_call_tool.__name__ = "mcp_call_tool"
        mcp_list_resources.__name__ = "mcp_list_resources"
        mcp_list_resource_templates.__name__ = "mcp_list_resource_templates"
        mcp_read_resource.__name__ = "mcp_read_resource"
        mcp_list_prompts.__name__ = "mcp_list_prompts"
        mcp_get_prompt.__name__ = "mcp_get_prompt"

        self.tool_manager.add_tool(list_mcp_servers)
        self.tool_manager.add_tool(mcp_list_tools)
        self.tool_manager.add_tool(mcp_call_tool)
        self.tool_manager.add_tool(mcp_list_resources)
        self.tool_manager.add_tool(mcp_list_resource_templates)
        self.tool_manager.add_tool(mcp_read_resource)
        self.tool_manager.add_tool(mcp_list_prompts)
        self.tool_manager.add_tool(mcp_get_prompt)
        self._mcp_tools_registered = True

    def _unregister_mcp_tools(self) -> None:
        """Remove the MCP facade tools when no connection remains configured."""
        if not self.tool_manager:
            return
        for tool_name in (
            "list_mcp_servers",
            "mcp_list_tools",
            "mcp_call_tool",
            "mcp_list_resources",
            "mcp_list_resource_templates",
            "mcp_read_resource",
            "mcp_list_prompts",
            "mcp_get_prompt",
        ):
            try:
                self.tool_manager.remove_tool(tool_name)
            except Exception:
                pass
        self._mcp_tools_registered = False

    def with_mcp_servers(
        self,
        mcp_servers: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
    ):
        """Replace MCP server configuration at runtime and preserve secrets safely."""
        self.mcp_servers = self.mcp_manager.configure_servers(mcp_servers or [])
        if self.mcp_servers:
            self._register_mcp_tools()
        else:
            self._unregister_mcp_tools()
        return self

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
                "Automations are not available. Use a filesystem, MongoDB, or "
                "Oracle memory provider with automations_enabled=True."
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

    # --- Meta-harness access ---

    def with_meta_harness(
        self,
        meta_harness: Any = None,
        *,
        mode: str = "delegate",
        default_harness: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> "MemAgent":
        """Attach or replace the external agent meta-harness at runtime."""
        normalized = str(mode or "delegate").strip().lower()
        if normalized not in {"delegate", "runtime"}:
            raise ValueError("meta-harness mode must be delegate or runtime")
        if meta_harness is None:
            from ..metaharness import MetaHarness

            meta_harness = MetaHarness.from_env(
                memory_provider=self.memory_provider,
                agent=self,
                approval_store=self.approval_store,
            )
            self._owns_meta_harness = True
        self.meta_harness = meta_harness
        self.meta_harness_mode = normalized
        self.default_harness = (
            str(default_harness or "auto").strip().lower().replace("_", "-")
        )
        self.harness_config = dict(config or {})
        if normalized == "delegate":
            self._register_meta_harness_tools()
        return self

    def has_meta_harness(self) -> bool:
        return self.meta_harness is not None

    def run_on_harness(
        self,
        query: str,
        *,
        workspace: Optional[str] = None,
        harness: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        write: Optional[bool] = None,
        verification_command: Optional[str] = None,
        execution_backend: Optional[str] = None,
        model_initiated: bool = False,
    ):
        """Execute a complete turn through a configured harness."""
        if self.meta_harness is None:
            raise ValueError("No meta-harness is configured")
        from ..metaharness import (
            HarnessBudget,
            HarnessPermissions,
            HarnessTask,
            VerificationSpec,
        )

        memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)
        configured = dict(self.harness_config or {})
        resolved_workspace = str(
            workspace
            or (context or {}).get("workspace")
            or configured.get("workspace")
            or os.getcwd()
        )
        permission_config = dict(configured.get("permissions") or {})
        if write is not None:
            permission_config["workspace_mode"] = "direct" if write else "read_only"
        permissions = HarnessPermissions.from_value(permission_config)
        budget = HarnessBudget.from_value(configured.get("budget"))
        verification = VerificationSpec.from_value(
            {
                **dict(configured.get("verification") or {}),
                **(
                    {"command": verification_command, "required": True}
                    if verification_command
                    else {}
                ),
            }
        )
        metadata = dict(configured.get("metadata") or {})
        # A MemAgent must never route a tool or full turn back into the same
        # in-process adapter. The router compares this ID with the wrapped
        # native agent while still allowing an explicitly different MemAgent.
        metadata["origin_agent_id"] = self.agent_id
        metadata["model_initiated"] = bool(model_initiated)
        if execution_backend:
            metadata["execution_backend"] = execution_backend
        task = HarnessTask(
            task=query,
            workspace=resolved_workspace,
            harness=harness or self.default_harness,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            agent_id=self.agent_id,
            model=configured.get("model"),
            mode=self.meta_harness_mode or "delegate",
            permissions=permissions,
            budget=budget,
            verification=verification,
            output_schema=configured.get("output_schema"),
            context=dict(context or {}),
            metadata=metadata,
        )
        result = self.meta_harness.run(task)
        if result.status.value == "succeeded" and result.final_response:
            self._record_interaction(
                query,
                result.final_response,
                memory_id,
                thread_id,
                user_id=user_id,
            )
        return result

    def resume_harness_approval(self, proposal_id: str):
        """Consume an approved run envelope and return structured evidence."""
        if self.meta_harness is None:
            raise ValueError("No meta-harness is configured")
        result = self.meta_harness.resume_approval(proposal_id)
        if result.status.value == "succeeded" and result.final_response:
            run = self.meta_harness.get_run(result.run_id) or {}
            task = dict(run.get("task") or {})
            self._record_interaction(
                str(task.get("task") or ""),
                result.final_response,
                task.get("memory_id"),
                task.get("thread_id"),
                user_id=task.get("user_id"),
            )
        return result

    def _register_meta_harness_tools(self) -> None:
        if not self.meta_harness or self._meta_harness_tools_registered:
            return

        @governed_tool(
            deterministic=False,
            side_effects=True,
            requires_approval=False,
            domains=("workspace", "metaharness"),
        )
        def run_harness_task(
            task: str,
            workspace: str,
            harness: str = "auto",
            write: bool = False,
            verification_command: Optional[str] = None,
            execution_backend: str = "local",
        ) -> Dict[str, Any]:
            """Delegate a bounded workspace task to an external agent harness.

            Write-capable, networked, or secret-bearing runs return a durable
            host approval proposal before execution.
            """
            result = self.run_on_harness(
                task,
                workspace=workspace,
                harness=harness,
                memory_id=self._current_memory_id,
                thread_id=self._current_thread_id,
                user_id=self._current_user_id,
                write=write,
                verification_command=verification_command,
                execution_backend=execution_backend,
                model_initiated=True,
            )
            return result.to_dict()

        @governed_tool(
            deterministic=True,
            side_effects=False,
            domains=("metaharness",),
        )
        def get_harness_run(run_id: str) -> Dict[str, Any]:
            """Read one durable harness run and its verification result."""
            value = self.meta_harness.get_run(run_id)
            return value or {"ok": False, "error_code": "run_not_found"}

        @governed_tool(
            deterministic=True,
            side_effects=False,
            domains=("metaharness",),
        )
        def list_agent_harnesses() -> Dict[str, Any]:
            """List configured harnesses and current capability probes."""
            values = self.meta_harness.list_harnesses()
            return {"ok": True, "harnesses": values, "count": len(values)}

        for tool in (run_harness_task, get_harness_run, list_agent_harnesses):
            self.tool_manager.add_tool(tool)
        self._meta_harness_tools_registered = True

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

    # --- Browser control ---

    def with_browser_control(
        self,
        provider: Optional[Union[str, Dict[str, Any], "BrowserControlProvider"]],
    ):
        """Attach or detach a browser-control provider.

        The registered ``browser_control`` tool is always governed as
        nondeterministic, side-effecting, and approval-required because a
        browser task may click, type, submit, purchase, publish, or delete.
        """
        if provider is not None:
            new_manager = BrowserControlManager.from_config(provider)
            if self.browser_control_manager is None:
                self.browser_control_manager = new_manager
            else:
                self.browser_control_manager.set_provider(new_manager.provider)
            self.browser_control_config = new_manager.get_provider_config()
            self._browser_control_init_error = None
            self._register_browser_control_tools()
        else:
            self._unregister_browser_control_tools()
            if self.browser_control_manager is not None:
                self.browser_control_manager.close()
                self.browser_control_manager = None
            self.browser_control_config = None
            self._browser_control_init_error = None
        return self

    def with_browser_control_provider(
        self,
        provider: Optional[Union[str, Dict[str, Any], "BrowserControlProvider"]],
    ):
        """Compatibility alias for :meth:`with_browser_control`."""
        return self.with_browser_control(provider)

    def has_browser_control(self) -> bool:
        """Return whether a usable browser-control provider is attached."""
        return bool(
            self.browser_control_manager and self.browser_control_manager.is_enabled()
        )

    def get_browser_control_provider_name(self) -> Optional[str]:
        if self.browser_control_manager is None:
            return None
        return self.browser_control_manager.get_provider_name()

    def run_browser_task(
        self,
        task: str,
        *,
        max_steps: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Run a browser task directly as trusted host code.

        Model calls use the separately registered ``browser_control`` tool and
        pass through durable approval. This direct helper is for application
        hosts that have already made their own authorization decision.
        """
        if not self.has_browser_control():
            raise ValueError("No browser-control provider configured")
        return self.browser_control_manager.run_task(
            task, max_steps=max_steps, timeout=timeout
        ).to_json()

    def _register_browser_control_tools(self) -> None:
        if not (
            self.tool_manager
            and self.browser_control_manager
            and self.browser_control_manager.is_enabled()
        ):
            return
        if self._browser_control_tools_registered:
            return
        for tool_func in self.browser_control_manager.get_tools():
            self.tool_manager.add_tool(tool_func)
        self._browser_control_tools_registered = True
        logger.info(
            "Registered browser-control tool (provider: %s)",
            self.browser_control_manager.get_provider_name(),
        )

    def _unregister_browser_control_tools(self) -> None:
        if not self._browser_control_tools_registered or not self.tool_manager:
            return
        for tool_name in self._browser_control_tool_names:
            self.tool_manager.remove_tool(tool_name)
        self._browser_control_tools_registered = False
        logger.info("Unregistered browser-control tool")

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

        # ContextVars copy values when an asyncio task is spawned. Copy before
        # mutation so sibling tasks never share the same inherited dict.
        thread_ids_by_memory = dict(self._thread_ids_by_memory)
        thread_ids_by_memory[resolved_memory_id] = resolved_thread_id
        self._thread_ids_by_memory = thread_ids_by_memory
        self._current_thread_id = resolved_thread_id

        if self.cache_manager:
            self.cache_manager.update_scope(
                agent_id=self.agent_id, memory_id=resolved_memory_id
            )

        self._ensure_agent_registered()

        return resolved_memory_id, resolved_thread_id

    def _ensure_agent_registered(self) -> None:
        """Fail-soft upsert of this logical agent and its current memories.

        Applications should not have to remember a separate ``save()`` call in
        order for observability to work. Registration remains fail-soft so a
        read-only provider or a transient database outage never blocks chat.
        """
        provider = self.memory_provider
        desired_memory_ids = {str(item) for item in self.memory_ids if item}
        if (
            not self.auto_register
            or provider is None
            or not hasattr(provider, "store_memagent")
            or desired_memory_ids.issubset(self._registered_memory_ids)
        ):
            return

        with self._registration_lock:
            desired_memory_ids = {str(item) for item in self.memory_ids if item}
            if desired_memory_ids.issubset(self._registered_memory_ids):
                return
            try:
                persistence.save_agent(self)
                self._registered_memory_ids = desired_memory_ids
                self._registration_attempted = True
            except Exception as exc:
                level = (
                    logging.DEBUG if self._registration_attempted else logging.WARNING
                )
                logger.log(
                    level,
                    "MemAgent auto-registration failed for %s: %s",
                    self.agent_id,
                    exc,
                )
                self._registration_attempted = True

    def _trace_identity_payload(self) -> Dict[str, Any]:
        """Return the canonical, content-free identity envelope for this turn."""
        return {
            "schema_version": 2,
            "application_id": self.application_id,
            "agent_id": self.agent_id,
            "run_id": self._current_run_id,
            "turn_id": self._current_turn_id,
            "root_trace_id": self._current_root_trace_id,
            "memory_id": self._current_memory_id,
            "thread_id": self._current_thread_id,
            "user_id": self._current_user_id,
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        }

    def _begin_trace_turn(
        self,
        user_id: Optional[str],
        *,
        query: Optional[str] = None,
        emit_start: bool = True,
    ) -> None:
        """Initialize an isolated trace collector and durable turn identity."""
        self._current_user_id = user_id
        session = session_for(self)
        self._current_run_id = (
            session.identity["run_id"] if session else str(uuid.uuid4())
        )
        self._current_turn_id = (
            session.identity["turn_id"] if session else str(uuid.uuid4())
        )
        self._current_root_trace_id = (
            session.identity["root_trace_id"] if session else str(uuid.uuid4())
        )
        self._current_parent_span_id = self._current_root_trace_id
        self._stream_trace_events = []
        self._last_tool_outcomes = []
        self._last_memory_context_evidence = {}
        self._last_selection_ledger = []
        self._last_memory_attribution_context = None
        self._last_trace_context = self._trace_identity_payload()
        if query is not None and self.learning_control_plane is not None:
            try:
                self.learning_control_plane.begin_run(
                    query, scope=self._trace_identity_payload()
                )
            except Exception as exc:
                logger.debug("Learning run-start capture failed: %s", exc)
        if emit_start:
            self._emit_trace_turn_start()

    def _emit_trace_turn_start(self) -> None:
        """Emit the root span after stream_start when a callback is present."""
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "turn_start",
                "title": "Agent turn started",
                "trace_id": f"turn:{self._current_turn_id}:start",
                "span_id": self._current_root_trace_id,
                "parent_span_id": None,
                "status": "started",
                "content": "",
            },
        )

    def _sanitize_observability_context(
        self, value: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return the bounded, content-free host provenance allowlist."""
        if not isinstance(value, dict):
            return {}
        safe: Dict[str, Any] = {}
        for key in _OBSERVABILITY_CONTEXT_FIELDS:
            item = value.get(key)
            if item is None:
                continue
            if key == "grounding_source_ids":
                if isinstance(item, (list, tuple, set)):
                    safe[key] = [str(entry)[:240] for entry in list(item)[:16]]
                continue
            if isinstance(item, bool):
                safe[key] = item
            elif isinstance(item, (int, float)):
                safe[key] = item
            elif isinstance(item, str):
                safe[key] = item[:240]
        return safe

    def _emit_context_provenance_trace(
        self,
        observability_context: Optional[Dict[str, Any]],
        request_context: Optional[Dict[str, Any]],
    ) -> None:
        """Persist hashes/status/ids needed to diagnose wrong-context turns.

        The LLM-visible request context itself is never copied into this event.
        A deterministic fingerprint lets operators compare two turns while the
        allowlisted host fields explain the route/thread/grounding decision.
        """
        safe = self._sanitize_observability_context(observability_context)
        safe["request_context_fingerprint"] = self._fingerprint(
            dict(request_context or {})
        )
        safe["request_context_key_count"] = len(dict(request_context or {}))
        safe.setdefault("request_context_present", bool(request_context))
        metadata = {
            key: (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, list)
                else value
            )
            for key, value in safe.items()
        }
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "context_provenance",
                "title": "Request context provenance",
                "trace_id": f"context:{self._current_turn_id}",
                "span_id": str(uuid.uuid4()),
                "parent_span_id": self._current_root_trace_id,
                "status": "verified" if safe.get("ownership_verified") else "recorded",
                "content": json.dumps(
                    safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                **metadata,
            },
        )

    def _attach_personalization_context(
        self,
        built_context: Dict[str, Any],
        request_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Attach a typed host context without duplicating entity profiles."""
        raw = (
            request_context.get("personalization_context")
            if isinstance(request_context, dict)
            else None
        )
        if not raw:
            return built_context
        try:
            from ..personalization import PersonalizationContext

            personalization = PersonalizationContext.from_value(raw)
        except Exception as exc:
            logger.warning("Ignoring invalid personalization context: %s", exc)
            return built_context
        result = dict(built_context)
        result["personalization_context"] = personalization
        # A host-built context may already contain the same entity retrieval.
        # Keep one authoritative rendering and one observable supply record.
        if personalization.policy.include_entity_memory:
            result.pop("entity_memory_profiles", None)
            result.pop("entity_memory_retrieval", None)
        return result

    @staticmethod
    def _host_owns_entity_personalization(
        request_context: Optional[Dict[str, Any]],
    ) -> bool:
        """Avoid a second entity lookup when the host supplied that source."""
        raw = (
            request_context.get("personalization_context")
            if isinstance(request_context, dict)
            else None
        )
        if not raw:
            return False
        try:
            from ..personalization import PersonalizationContext

            return PersonalizationContext.from_value(raw).policy.include_entity_memory
        except Exception:
            return False

    @staticmethod
    def _memory_row_identifier(row: Any, fallback: str) -> str:
        if not isinstance(row, dict):
            return fallback
        for key in (
            "parent_source_id",
            "source_id",
            "entity_id",
            "summary_id",
            "id",
            "_id",
            "thread_id",
        ):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value)
        content = row.get("content")
        if isinstance(content, dict):
            for key in ("id", "thread_id", "summary_id"):
                value = content.get(key)
                if value is not None and str(value).strip():
                    return str(value)
        return fallback

    @staticmethod
    def _memory_row_text(row: Any) -> str:
        if not isinstance(row, dict):
            return ""
        for key in ("text", "summary_content", "content", "excerpt"):
            value = row.get(key)
            if isinstance(value, dict):
                value = value.get("content") or value.get("text")
            if value is not None and str(value).strip():
                return str(value)
        return ""

    def _emit_memory_context_trace(
        self,
        built_context: Dict[str, Any],
    ) -> None:
        """Persist content-free evidence for every memory item sent to the LLM."""
        from ..personalization import PersonalizationContext, PersonalizationPolicy

        history = [
            row
            for row in (built_context.get("conversation_history") or [])
            if isinstance(row, dict)
        ]
        semantic = [
            row
            for row in (built_context.get("retrieved_memories") or [])
            if isinstance(row, dict)
        ]
        entities = [
            row
            for row in (built_context.get("entity_memory_profiles") or [])
            if isinstance(row, dict)
        ]
        summaries = [
            row
            for row in (built_context.get("summaries") or [])
            if isinstance(row, dict)
        ]
        has_personalization = built_context.get("personalization_context") is not None
        personalization = PersonalizationContext.from_value(
            built_context.get("personalization_context")
        )
        personalization_summary = personalization.trace_summary()
        if personalization.entity_profiles:
            entities = list(personalization.entity_profiles)

        entity_attribute_count = 0
        entity_char_count = 0
        entity_refs: List[Dict[str, Any]] = []
        for index, profile in enumerate(entities):
            raw_attributes = profile.get("attributes") or {}
            if isinstance(raw_attributes, dict):
                attribute_names = sorted(str(key) for key in raw_attributes)[:32]
            else:
                attribute_names = sorted(
                    str(item.get("name"))
                    for item in raw_attributes
                    if isinstance(item, dict) and item.get("name")
                )[:32]
            entity_attribute_count += len(attribute_names)
            if isinstance(raw_attributes, dict):
                entity_char_count += sum(
                    len(str(key)) + len(str(value))
                    for key, value in raw_attributes.items()
                )
            else:
                entity_char_count += sum(
                    len(str(item.get("name") or "")) + len(str(item.get("value") or ""))
                    for item in raw_attributes
                    if isinstance(item, dict)
                )
            identifier = self._memory_row_identifier(profile, f"entity:{index}")
            entity_refs.append(
                {
                    "ref": self._fingerprint(identifier),
                    "attribute_names": attribute_names,
                }
            )

        semantic_refs = [
            {
                "ref": self._fingerprint(
                    self._memory_row_identifier(row, f"semantic:{index}")
                ),
                "source": str(row.get("source") or "memory")[:40],
                "score": row.get("score"),
            }
            for index, row in enumerate(semantic)
        ]
        summary_refs = [
            self._fingerprint(self._memory_row_identifier(row, f"summary:{index}"))
            for index, row in enumerate(summaries)
        ]
        source_counts = {
            "history_messages": len(history),
            "semantic_memories": len(semantic),
            "entity_profiles": len(entities),
            "entity_attributes": entity_attribute_count,
            "summaries": len(summaries),
            "preferences": int(
                personalization_summary.get("source_counts", {}).get("preferences", 0)
            ),
            "conversation_memories": int(
                personalization_summary.get("source_counts", {}).get(
                    "conversation_memories", 0
                )
            ),
            "writing_samples": int(
                personalization_summary.get("source_counts", {}).get(
                    "writing_samples", 0
                )
            ),
        }
        supplied_count = sum(
            source_counts[source]
            for source in (
                "history_messages",
                "semantic_memories",
                "entity_attributes",
                "summaries",
                "preferences",
                "conversation_memories",
                "writing_samples",
            )
        )
        entity_retrieval = built_context.get("entity_memory_retrieval") or {}
        personalization_diagnostics = personalization_summary.get("diagnostics") or {}
        degraded = bool(
            entity_retrieval.get("degraded")
            or personalization_diagnostics.get("degraded")
        )
        fallback_used = bool(
            entity_retrieval.get("fallback_used")
            or personalization_diagnostics.get("fallback_used")
        )
        volatile_chars = sum(
            len(self._memory_row_text(row)) for row in [*history, *semantic, *summaries]
        ) + int(personalization_summary.get("rendered_char_count") or 0)
        if personalization.is_empty:
            volatile_chars += entity_char_count
        automatic_entity_candidate_count = int(
            (
                entity_retrieval.get("fallback_candidate_count")
                if entity_retrieval.get("fallback_used")
                else entity_retrieval.get("semantic_match_count")
            )
            or 0
        )
        semantic_candidate_count = int(
            (self._last_retrieval_stats or {}).get("candidate_count") or 0
        )
        personalization_candidate_count = int(
            personalization_diagnostics.get("candidate_count") or 0
        )
        personalization_owns_entity = bool(
            has_personalization and personalization.policy.include_entity_memory
        )
        personalization_selected_count = int(
            personalization_diagnostics.get("selected_count") or 0
        )
        payload = {
            "schema_version": 1,
            "stage": "supplied",
            "source_counts": source_counts,
            "retrieved_candidate_count": (
                semantic_candidate_count
                + automatic_entity_candidate_count
                + personalization_candidate_count
            ),
            "retrieved_source_counts": {
                "semantic": semantic_candidate_count,
                "entity": (
                    int(personalization_diagnostics.get("entity_candidate_count") or 0)
                    if personalization_owns_entity
                    else automatic_entity_candidate_count
                ),
                "conversation_personalization": int(
                    personalization_diagnostics.get("conversation_candidate_count") or 0
                ),
            },
            "retrieved_selected_count": (
                len(semantic)
                + (
                    personalization_selected_count
                    if has_personalization
                    else len(entities)
                )
            ),
            "retrieved_selected_source_counts": {
                "semantic": len(semantic),
                "entity": (
                    int(personalization_diagnostics.get("entity_selected_count") or 0)
                    if personalization_owns_entity
                    else len(entities)
                ),
                "conversation_personalization": int(
                    personalization_diagnostics.get("conversation_selected_count") or 0
                ),
            },
            "supplied_count": supplied_count,
            "injected_char_count": volatile_chars,
            "degraded": degraded,
            "fallback_used": fallback_used,
            "semantic_refs": semantic_refs,
            "entity_refs": entity_refs,
            "summary_refs": summary_refs,
            "preference_keys": personalization_summary.get("preference_keys") or [],
            "conversation_refs": personalization_summary.get("conversation_refs") or [],
            "writing_sample_refs": personalization_summary.get("writing_sample_refs")
            or [],
            "entity_retrieval": {
                key: entity_retrieval.get(key)
                for key in (
                    "retrieval_mode",
                    "semantic_attempted",
                    "semantic_match_count",
                    "fallback_used",
                    "fallback_candidate_count",
                    "match_count",
                    "degraded",
                    "degraded_reason",
                )
                if entity_retrieval.get(key) is not None
            },
            "personalization_retrieval": personalization_diagnostics,
        }
        self._last_memory_context_evidence = dict(payload)
        selection_ledger = getattr(self, "_last_selection_ledger", [])
        if selection_ledger:
            self._emit_stream_event(
                "trace",
                {
                    "trace_kind": "memory_selection",
                    "title": "Memory selection decisions",
                    "trace_id": f"selection:{self._current_turn_id}",
                    "span_id": str(uuid.uuid4()),
                    "parent_span_id": self._current_root_trace_id,
                    "status": "success",
                    "selection_ledger": selection_ledger,
                },
            )

        def _attribution_rows(
            rows: List[Dict[str, Any]], source: str
        ) -> List[Dict[str, Any]]:
            return [{**row, "_memorizz_reference_source": source} for row in rows]

        attribution_memories = _attribution_rows(semantic, "semantic_memory")
        attribution_memories.extend(
            _attribution_rows(
                list(personalization.conversation_memories), "conversation"
            )
        )
        attribution_memories.extend(_attribution_rows(history, "history"))
        attribution_memories.extend(_attribution_rows(summaries, "summary"))
        self._last_memory_attribution_context = PersonalizationContext(
            entity_profiles=entities,
            preferences=personalization.preferences,
            conversation_memories=attribution_memories,
            writing_samples=personalization.writing_samples,
            policy=PersonalizationPolicy(
                max_entity_profiles=max(1, len(entities)),
                max_conversation_memories=max(1, len(attribution_memories)),
                max_preferences=max(1, len(personalization.preferences)),
                max_writing_samples=max(1, len(personalization.writing_samples)),
                max_chars=1,
            ),
        )

        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "memory_context",
                "input_refs": [
                    entry["resource"] for entry in selection_ledger if entry["selected"]
                ][:32],
                "title": "Memory supplied",
                "trace_id": f"memory:{self._current_turn_id}:supplied",
                "span_id": str(uuid.uuid4()),
                "parent_span_id": self._current_root_trace_id,
                "status": "degraded" if degraded else "recorded",
                "content": json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "memory_history_count": len(history),
                "memory_candidate_count": payload["retrieved_candidate_count"],
                "memory_supplied_count": supplied_count,
                "memory_injected_chars": volatile_chars,
                "memory_degraded": degraded,
                "memory_fallback_used": fallback_used,
                "entity_profile_count": len(entities),
                "preference_count": source_counts["preferences"],
                "conversation_memory_count": source_counts["conversation_memories"],
                "writing_sample_count": source_counts["writing_samples"],
            },
        )

    def _emit_memory_reference_trace(self, response: str) -> None:
        """Record conservative post-response attribution without raw content."""
        context = self._last_memory_attribution_context
        supplied = int(
            (self._last_memory_context_evidence or {}).get("supplied_count") or 0
        )
        if context is None or supplied <= 0:
            return
        try:
            payload = {
                "schema_version": 1,
                "stage": "referenced",
                "supplied_count": supplied,
                **context.referenced_by(response),
            }
        except Exception as exc:
            logger.debug("Memory response attribution failed: %s", exc)
            return
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "memory_reference",
                "title": "Memory referenced",
                "trace_id": f"memory:{self._current_turn_id}:referenced",
                "span_id": str(uuid.uuid4()),
                "parent_span_id": self._current_root_trace_id,
                "status": "measured",
                "content": json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "memory_supplied_count": supplied,
                "memory_referenced_count": payload["referenced_count"],
            },
        )

    def _emit_cache_decision_trace(
        self,
        decision: str,
        cache_metadata: Optional[Dict[str, Any]],
        *,
        bypass_reason: Optional[str] = None,
    ) -> None:
        """Record cache routing without query or response content."""
        fingerprints = (cache_metadata or {}).get("fingerprints") or {}
        payload: Dict[str, Any] = {
            "cache_decision": str(decision)[:80],
            "cache_enabled": bool(self.cache_manager.enabled),
            "request_context_fingerprint": str(
                fingerprints.get("request_context") or ""
            )[:240],
        }
        if bypass_reason:
            payload["cache_bypass_reason"] = str(bypass_reason)[:160]
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "cache_decision",
                "title": f"Semantic cache · {decision}",
                "trace_id": f"cache:{self._current_turn_id}:{decision}",
                "span_id": str(uuid.uuid4()),
                "parent_span_id": self._current_root_trace_id,
                "status": str(decision)[:80],
                "content": json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                **payload,
            },
        )

    def _finish_trace_turn(
        self, status: str, *, error_code: Optional[str] = None
    ) -> None:
        """Close the root turn span without persisting exception messages."""
        payload: Dict[str, Any] = {
            "trace_kind": "turn_result",
            "title": "Agent turn completed"
            if status == "success"
            else "Agent turn failed",
            "trace_id": f"turn:{self._current_turn_id}:result",
            "span_id": self._current_root_trace_id,
            "parent_span_id": None,
            "status": status,
            "success": status == "success",
            "content": "",
        }
        if error_code:
            payload["error_code"] = error_code
        self._emit_stream_event("trace", payload)
        self._last_trace_context = self._trace_identity_payload()
        if self.learning_control_plane is not None:
            try:
                self.learning_control_plane.complete_run(
                    "",
                    status=status,
                    metrics={"error_code": error_code} if error_code else {},
                    scope=self._last_trace_context,
                )
            except Exception as exc:
                logger.debug("Learning run-completion capture failed: %s", exc)

    def get_trace_context(self) -> Dict[str, Any]:
        """Return identifiers for linking application outcomes to the last turn."""
        return {
            key: value
            for key, value in dict(self._last_trace_context or {}).items()
            if value is not None
        }

    def observe(self, operation: str, *, trace_context=None, **kwargs):
        """Observe a host action using this agent's provider and turn identity."""
        from ..observability import ObservabilityRecorder

        return ObservabilityRecorder(
            self.memory_provider, trace_context or self.get_trace_context()
        ).start_span(operation, **kwargs)

    @property
    def last_tool_outcomes(self) -> List[Dict[str, Any]]:
        """Structured outcomes from the most recent turn, in execution order."""
        return [dict(item) for item in (self._last_tool_outcomes or [])]

    def record_feedback(
        self,
        rating: float,
        *,
        verified: bool = True,
        source: str = "user",
        label: Optional[str] = None,
        comment: Optional[str] = None,
        include_comment: bool = False,
        trace_context: Optional[Dict[str, Any]] = None,
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Join verified application feedback to a concrete agent trace."""
        if not self.memory_provider:
            raise ValueError("record_feedback requires a memory provider")
        from ..observability import ObservabilityStore

        return ObservabilityStore(self.memory_provider).record_feedback(
            trace_context=trace_context or self.get_trace_context(),
            rating=rating,
            verified=verified,
            source=source,
            label=label,
            comment=comment,
            include_comment=include_comment,
            external_id=external_id,
        )

    def record_task_outcome(
        self,
        status: str,
        *,
        verified: bool = True,
        source: str = "application",
        score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        external_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Join a business/task outcome to a concrete agent trace."""
        if not self.memory_provider:
            raise ValueError("record_task_outcome requires a memory provider")
        resolved_trace_context = trace_context or self.get_trace_context()
        if self.learning_control_plane is not None:
            from ..learning import OutcomeEvidence

            outcome = OutcomeEvidence.from_value(
                status,
                verified=verified,
                source=source,
                score=score,
                metrics=metrics,
            )
            return self.learning_control_plane.record_outcome(
                outcome,
                scope=resolved_trace_context,
                external_id=external_id,
            )
        from ..observability import ObservabilityStore

        return ObservabilityStore(self.memory_provider).record_outcome(
            trace_context=resolved_trace_context,
            status=status,
            verified=verified,
            source=source,
            score=score,
            metrics=metrics,
            external_id=external_id,
        )

    def _model_trace_metadata(self) -> Dict[str, Any]:
        """Extract a bounded provider/model/usage snapshot without credentials."""
        metadata: Dict[str, Any] = {}
        config: Dict[str, Any] = {}
        getter = getattr(self.model, "get_config", None)
        if callable(getter):
            try:
                candidate = getter() or {}
                if isinstance(candidate, dict):
                    config = candidate
            except Exception:
                config = {}
        model_name = (
            config.get("model")
            or config.get("model_name")
            or getattr(self.model, "model", None)
        )
        provider_name = config.get("provider") or config.get("type")
        if not provider_name and self.model is not None:
            provider_name = type(self.model).__name__
        if model_name:
            metadata["model"] = str(model_name)[:240]
        if provider_name:
            metadata["provider"] = str(provider_name)[:120]

        response_getter = getattr(self.model, "get_last_response_metadata", None)
        if callable(response_getter):
            try:
                from ..observability.models import LastResponseMetadata

                response_metadata = response_getter() or {}
                if isinstance(response_metadata, LastResponseMetadata):
                    response_metadata = response_metadata.model_dump(exclude_none=True)
                metadata.update(
                    LastResponseMetadata.model_validate(response_metadata).model_dump(
                        exclude_none=True
                    )
                )
            except Exception:
                logger.debug("Provider returned invalid response metadata")

        usage_getter = getattr(self.model, "get_last_usage", None)
        if callable(usage_getter):
            try:
                usage = usage_getter() or {}
            except Exception:
                usage = {}
            if isinstance(usage, dict):
                field_map = {
                    "prompt_tokens": "input_tokens",
                    "input_tokens": "input_tokens",
                    "completion_tokens": "output_tokens",
                    "output_tokens": "output_tokens",
                    "cached_tokens": "cached_tokens",
                    "cache_read_input_tokens": "cached_tokens",
                    "total_tokens": "total_tokens",
                }
                for source, target in field_map.items():
                    value = usage.get(source)
                    if value is None or target in metadata:
                        continue
                    if isinstance(value, (int, float)):
                        metadata[target] = value
        return metadata

    def _start_model_trace(self, *, iteration: int, stage: str) -> tuple[str, float]:
        span_id = str(uuid.uuid4())
        self._current_parent_span_id = span_id
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "model_call",
                "title": "Model call",
                "trace_id": f"model:{span_id}:call",
                "span_id": span_id,
                "parent_span_id": self._current_root_trace_id,
                "iteration": iteration,
                "stage": stage,
                "status": "started",
                "content": "",
                **{
                    key: value
                    for key, value in self._model_trace_metadata().items()
                    if key in {"model", "provider", "max_output_tokens"}
                },
            },
        )
        return span_id, time.perf_counter()

    def _finish_model_trace(
        self,
        span_id: str,
        started_at: float,
        *,
        iteration: int,
        stage: str,
        error: Optional[BaseException] = None,
        duration_ms: Optional[float] = None,
        response_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        model_metadata = self._model_trace_metadata()
        if error:
            # Optional providers may retain usage from their previous response.
            model_metadata = {
                key: value
                for key, value in model_metadata.items()
                if key in {"model", "provider", "max_output_tokens"}
            }
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "model_result",
                "title": "Model result",
                "trace_id": f"model:{span_id}:result",
                "span_id": span_id,
                "parent_span_id": self._current_root_trace_id,
                "iteration": iteration,
                "stage": stage,
                "status": "error" if error else "success",
                "success": error is None,
                "error_code": type(error).__name__ if error else "",
                "duration_ms": round(
                    duration_ms
                    if duration_ms is not None
                    else (time.perf_counter() - started_at) * 1000,
                    3,
                ),
                "content": "",
                **model_metadata,
                **(response_metadata or {}),
            },
        )
        self._current_parent_span_id = self._current_root_trace_id

    def _generate_with_trace(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        iteration: int,
        stage: str,
    ) -> Any:
        span_id, started_at = self._start_model_trace(iteration=iteration, stage=stage)
        error: Optional[BaseException] = None
        response_metadata = {}
        try:
            response = self.model.generate(messages, tools=tools)
            from ..llms.response_metadata import response_metadata as describe_response

            response_metadata = describe_response(
                response, text=response if isinstance(response, str) else None
            )
            return response
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish_model_trace(
                span_id,
                started_at,
                iteration=iteration,
                stage=stage,
                error=error,
                response_metadata=response_metadata,
            )

    def _generate_stream_with_trace(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        iteration: int,
        stage: str,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield provider events while measuring only time spent in the provider."""
        span_id, started_at = self._start_model_trace(iteration=iteration, stage=stage)
        provider_duration_ms = 0.0
        response_chars = response_bytes = 0
        ttft_ms = None
        terminal_received = False
        stream = None
        error: Optional[BaseException] = None
        try:
            stream = iter(self.model.generate_stream(messages, tools=tools))
            while True:
                check_cancelled()
                next_started_at = time.perf_counter()
                try:
                    event = next(stream)
                except StopIteration:
                    provider_duration_ms += (
                        time.perf_counter() - next_started_at
                    ) * 1000
                    break
                except BaseException:
                    provider_duration_ms += (
                        time.perf_counter() - next_started_at
                    ) * 1000
                    raise
                provider_duration_ms += (time.perf_counter() - next_started_at) * 1000
                if event.get("type") == "content" and isinstance(
                    event.get("content"), str
                ):
                    response_chars += len(event["content"])
                    response_bytes += len(event["content"].encode("utf-8"))
                    if ttft_ms is None:
                        ttft_ms = provider_duration_ms
                        session = session_for(self)
                        if session is not None:
                            session.emit(
                                "status",
                                stage="provider_first_delta",
                                span_id=span_id,
                                attempt=iteration,
                                provider_ttft_ms=round(ttft_ms, 3),
                            )
                if event.get("type") in {"done", "tool_calls"}:
                    terminal_received = True
                yield event
        except GeneratorExit:
            # A consumer returning after the provider's terminal event closes
            # this wrapper at its yield point; that is a successful model call.
            if not terminal_received:
                error = GeneratorExit()
            raise
        except BaseException as exc:
            error = exc
            raise
        finally:
            if stream is not None and callable(getattr(stream, "close", None)):
                try:
                    stream.close()
                except Exception as close_error:
                    if error is None:
                        error = close_error
            self._finish_model_trace(
                span_id,
                started_at,
                iteration=iteration,
                stage=stage,
                error=error,
                duration_ms=provider_duration_ms,
                response_metadata={
                    **(getattr(error, "response_metadata", {}) or {}),
                    "response_chars": response_chars,
                    "response_bytes": response_bytes,
                    "stream_duration_ms": round(provider_duration_ms, 3),
                    **({"ttft_ms": round(ttft_ms, 3)} if ttft_ms is not None else {}),
                },
            )

    def run(
        self,
        query: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        observability_context: Optional[Dict[str, Any]] = None,
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
            observability_context: Optional content-free host provenance for
                the private trace bundle. Only a strict allowlist of routing,
                ownership, grounding, and fingerprint fields is retained; this
                dict is never sent to the LLM or conversation history.

        Returns:
            The agent's response
        """
        if (
            self.meta_harness is not None
            and self.meta_harness_mode == "runtime"
            and not (tool_context or {}).get("_memorizz_harness_native")
        ):
            result = self.run_on_harness(
                query,
                workspace=(tool_context or {}).get("workspace")
                or (context or {}).get("workspace"),
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                write=(tool_context or {}).get("harness_write"),
                verification_command=(tool_context or {}).get(
                    "harness_verification_command"
                ),
                execution_backend=(tool_context or {}).get("harness_execution_backend"),
            )
            if session_for(self) is not None and result.status.value != "succeeded":
                from ..llms.streaming import ProviderStreamError

                raise ProviderStreamError("harness_" + str(result.status.value))
            return (
                result.final_response
                if result.status.value == "succeeded"
                else json.dumps(result.to_dict(), ensure_ascii=False, default=str)
            )
        if (
            self.delegates
            and self.delegation_config.get("enabled", True)
            and self.delegation_config.get("mode", "auto") in {"auto", "deterministic"}
            and not (tool_context or {}).get("_memorizz_skip_delegation")
        ):
            return self.delegate(
                query,
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                tool_context=tool_context,
                return_report=bool(self.delegation_config.get("return_report", False)),
            )
        logger.info(f"MemAgent {self.agent_id} executing query: {query[:50]}...")

        # M4: scope per-call tool context for tools reading via get_tool_context().
        from ..tool_context import reset_tool_context, set_tool_context

        _tc_token = set_tool_context(tool_context or {})
        trace_initialized = False
        turn_status = "error"
        turn_error_code: Optional[str] = None

        try:
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)
            self._begin_trace_turn(user_id, query=query)
            trace_initialized = True
            self._emit_context_provenance_trace(observability_context, context)
            self._turn_had_side_effects = False
            self._turn_had_nondeterministic_tools = False
            self._turn_cache_domains = set()
            self._cache_bypass_reason = None
            self._last_completion_decisions = []
            cache_metadata = self._semantic_cache_metadata(context)
            cache_lookup_bypass = self._semantic_cache_preflight_bypass(
                query, user_id=user_id
            )
            if not self.cache_manager.enabled:
                self._emit_cache_decision_trace("disabled", cache_metadata)
            elif cache_lookup_bypass:
                self._emit_cache_decision_trace(
                    "bypassed", cache_metadata, bypass_reason=cache_lookup_bypass
                )
            if cache_lookup_bypass and self.learning_control_plane is not None:
                try:
                    self.learning_control_plane.record_cache(
                        hit=False,
                        query=query,
                        reason=cache_lookup_bypass,
                        scope=self._trace_identity_payload(),
                    )
                except Exception:
                    pass

            # 2. Check semantic cache first
            cached_response = None
            if self.cache_manager.enabled:
                cached_response = self.cache_manager.get_cached_response(
                    query,
                    thread_id,
                    user_id=user_id,
                    metadata=cache_metadata,
                    bypass_reason=cache_lookup_bypass,
                )
                if not cache_lookup_bypass:
                    self._emit_cache_decision_trace(
                        "hit" if cached_response else "miss", cache_metadata
                    )
                if cached_response:
                    # Cache lookup is not an authorization boundary. Re-run
                    # host completion validation so changing runtime policy,
                    # external acceptance state, or a trusted validator cannot
                    # be bypassed by a response accepted on an earlier turn.
                    cache_decision = self._evaluate_completion_candidate(
                        query=query,
                        response=cached_response,
                        iteration=0,
                        tool_call_count=0,
                    )
                    if cache_decision.accepted:
                        logger.info("Returning host-validated cached response")
                        if self.learning_control_plane is not None:
                            try:
                                self.learning_control_plane.record_cache(
                                    hit=True,
                                    query=query,
                                    reason="exact_or_semantic_cache_hit",
                                    scope=self._trace_identity_payload(),
                                )
                            except Exception:
                                pass
                        self._record_interaction(
                            query,
                            cached_response,
                            memory_id,
                            thread_id,
                            user_id=user_id,
                        )
                        turn_status = "success"
                        return cached_response
                    logger.info(
                        "Cached response rejected by completion policy (%s); "
                        "continuing with a fresh model turn",
                        cache_decision.code,
                    )
                    self._emit_cache_decision_trace(
                        "rejected",
                        cache_metadata,
                        bypass_reason=cache_decision.code,
                    )

            # 3. Build context and prompt
            built_context = self._build_context(
                query,
                memory_id,
                user_id=user_id,
                include_entity_memory=not self._host_owns_entity_personalization(
                    context
                ),
            )
            built_context = self._attach_personalization_context(built_context, context)
            self._emit_memory_context_trace(built_context)
            system_prompt = self._build_system_prompt()

            # 4. Execute with LLM
            response = self._execute_llm_interaction(
                system_prompt,
                query,
                built_context,
                user_id=user_id,
                request_context=context,
            )
            check_cancelled()
            stream_session = session_for(self)
            if stream_session is not None and stream_session.outcome != "completed":
                turn_status = stream_session.outcome
                return ""
            self._emit_memory_reference_trace(response)

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(
                    query,
                    response,
                    thread_id,
                    user_id=user_id,
                    metadata=self._semantic_cache_metadata(context),
                    deterministic=not self._turn_had_nondeterministic_tools,
                    read_only=not self._turn_had_side_effects,
                    bypass_reason=self._cache_bypass_reason,
                )

            # 6. Record interaction in memory
            self._record_interaction(
                query, response, memory_id, thread_id, user_id=user_id
            )

            logger.info(f"MemAgent {self.agent_id} completed successfully")
            turn_status = "success"
            return response

        except CompletionRejectedError:
            turn_error_code = "CompletionRejectedError"
            raise
        except Exception as e:
            logger.error(f"MemAgent execution failed: {e}")
            turn_error_code = type(e).__name__
            if session_for(self) is not None:
                raise
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return error_response
        finally:
            if trace_initialized:
                self._finish_trace_turn(
                    turn_status,
                    error_code=turn_error_code,
                )
                self._record_stream_trace_bundle(
                    memory_id,
                    thread_id,
                    user_id=user_id,
                )
                self._stream_trace_events = None
            # M4: always release the per-call tool_context scope.
            reset_tool_context(_tc_token)

    def delegate(
        self,
        query: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        plan: Any = None,
        trace_id: Optional[str] = None,
        return_report: bool = False,
    ) -> Any:
        """Execute a configured deterministic or model-generated delegation plan."""
        if not self.delegates:
            raise ValueError("This agent has no configured delegates")
        if not self.memory_provider:
            raise ValueError("Delegation requires a shared memory provider")

        from ..multi_agent_orchestrator import MultiAgentOrchestrator

        memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)
        configured_plan = (
            plan if plan is not None else self.delegation_config.get("plan")
        )
        cache_context, cache_bypass_reason = self._delegation_cache_context(
            context, configured_plan
        )
        cache_metadata = self._semantic_cache_metadata(cache_context)
        if self.cache_manager.enabled:
            cached_response = self.cache_manager.get_cached_response(
                query,
                thread_id,
                user_id=user_id,
                metadata=cache_metadata,
                bypass_reason=cache_bypass_reason,
            )
            if cached_response:
                cache_decision = self._evaluate_completion_candidate(
                    query=query,
                    response=cached_response,
                    iteration=0,
                    tool_call_count=0,
                )
                if not cache_decision.accepted:
                    cached_response = None
            if cached_response:
                if self.learning_control_plane is not None:
                    try:
                        self.learning_control_plane.record_cache(
                            hit=True,
                            query=query,
                            reason="verified_read_only_delegation_cache_hit",
                            scope={
                                "memory_id": memory_id,
                                "thread_id": thread_id,
                                "user_id": user_id,
                            },
                        )
                    except Exception:
                        pass
                self._record_interaction(
                    query,
                    cached_response,
                    memory_id,
                    thread_id,
                    user_id=user_id,
                )
                if return_report:
                    return {
                        "ok": True,
                        "partial": False,
                        "cached": True,
                        "workflow_id": None,
                        "trace_id": trace_id,
                        "user_id": user_id,
                        "response": cached_response,
                        "consolidation": {
                            "status": "succeeded",
                            "strategy": "semantic_cache",
                            "model_used": False,
                        },
                        "tasks": [],
                        "failures": [],
                    }
                return cached_response

        inherited_workflow_id = (
            str((tool_context or {}).get("workflow_id") or "").strip() or None
        )
        inherited_trace_id = (
            str((tool_context or {}).get("trace_id") or "").strip() or None
        )
        orchestrator = MultiAgentOrchestrator(
            self,
            self.delegates,
            delegation_plan=configured_plan,
            persist_participants=bool(
                self.delegation_config.get("persist_participants", False)
            ),
            workflow_id=(
                self.delegation_config.get("workflow_id") or inherited_workflow_id
            ),
            consolidation_strategy=self.delegation_config.get(
                "consolidation_strategy", "model"
            ),
            primary_task_id=self.delegation_config.get("primary_task_id"),
            evidence_context=self.delegation_config.get("evidence_context", True),
            max_dependency_context_chars=self.delegation_config.get(
                "max_dependency_context_chars", 12_000
            ),
            max_consolidation_result_chars=self.delegation_config.get(
                "max_consolidation_result_chars", 12_000
            ),
            required_finding_ids=self.delegation_config.get("required_finding_ids", []),
            adaptive_escalation=self.delegation_config.get("adaptive_escalation", {}),
        )
        result = orchestrator.execute_multi_agent_workflow(
            query,
            memory_id,
            thread_id,
            user_id=user_id,
            context=context,
            tool_context=tool_context,
            trace_id=trace_id or inherited_trace_id,
            delegation_plan=configured_plan,
            return_report=return_report,
        )
        response = (
            str(result.get("response") or "")
            if isinstance(result, dict)
            else str(result or "")
        )
        workflow_report = (
            dict(result)
            if isinstance(result, dict)
            else dict(orchestrator.last_workflow_report or {})
        )
        workflow_ok = bool(workflow_report.get("ok", not workflow_report))
        if response and workflow_ok:
            if self.cache_manager.enabled and cache_bypass_reason is None:
                self.cache_manager.cache_response(
                    query,
                    response,
                    thread_id,
                    user_id=user_id,
                    metadata=cache_metadata,
                    deterministic=True,
                    read_only=True,
                )
            self._record_interaction(
                query, response, memory_id, thread_id, user_id=user_id
            )
        self._record_delegation_learning(
            query=query,
            response=response,
            result=workflow_report or result,
            memory_id=memory_id,
            thread_id=thread_id,
            user_id=user_id,
            cache_bypass_reason=cache_bypass_reason,
        )
        return result

    def _delegation_cache_context(
        self,
        context: Optional[Dict[str, Any]],
        configured_plan: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Return cache identity and a fail-closed delegation admission reason."""
        value = dict(context or {})
        value.setdefault("cache_domain", "delegation")
        data_version = value.get("cache_data_version") or value.get("data_version")
        delegation_identity = {
            "plan": configured_plan,
            "configuration": self.delegation_config,
            "delegates": [
                {
                    "agent_id": delegate.agent_id,
                    "instruction": delegate.instruction,
                    "mode": getattr(delegate, "meta_harness_mode", None),
                    "harness": getattr(delegate, "default_harness", None),
                    "harness_config": getattr(delegate, "harness_config", None),
                }
                for delegate in self.delegates
            ],
        }
        value["delegation_fingerprint"] = self._fingerprint(delegation_identity)
        if not self.cache_manager.enabled:
            return value, "semantic_cache_disabled"
        if self.completion_policy.enabled and self.completion_policy.require_tool_calls:
            return value, "completion_policy_requires_fresh_tool_evidence"
        if self.delegation_config.get("mode") != "deterministic" or callable(
            configured_plan
        ):
            return value, "nondeterministic_delegation_plan"
        if not data_version:
            return value, "delegation_data_version_required"

        from ..metaharness import HarnessPermissions, VerificationSpec

        for delegate in self.delegates:
            if getattr(delegate, "meta_harness_mode", None) != "runtime":
                return value, "native_delegate_not_cacheable"
            config = dict(getattr(delegate, "harness_config", None) or {})
            permissions = HarnessPermissions.from_value(config.get("permissions"))
            verification = VerificationSpec.from_value(config.get("verification"))
            if (
                permissions.workspace_mode != "read_only"
                or permissions.network != "none"
                or permissions.mcp_access == "governed_write"
            ):
                return value, "side_effecting_delegation"
            if not verification.required:
                return value, "unverified_delegation"
        value["data_version"] = str(data_version)
        return value, None

    def _record_delegation_learning(
        self,
        *,
        query: str,
        response: str,
        result: Any,
        memory_id: str,
        thread_id: str,
        user_id: Optional[str],
        cache_bypass_reason: Optional[str],
    ) -> None:
        """Capture the coordinated workflow as one continual-learning outcome."""
        if self.learning_control_plane is None:
            return
        report = dict(result) if isinstance(result, dict) else {}
        workflow_id = str(report.get("workflow_id") or uuid.uuid4())
        ok = not report or bool(report.get("ok"))
        scope = {
            "memory_id": memory_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "workflow_id": workflow_id,
            "trace_id": report.get("trace_id"),
            "run_id": workflow_id,
        }
        try:
            self.learning_control_plane.begin_run(query, scope=scope)
            self.learning_control_plane.complete_run(
                response,
                status="success" if ok else "failure",
                metrics={
                    "task_count": len(report.get("tasks") or []),
                    "failure_count": len(report.get("failures") or []),
                    "consolidation": dict(report.get("consolidation") or {}),
                    "cache_bypass_reason": cache_bypass_reason,
                },
                scope=scope,
            )
            self.learning_control_plane.record_workflow(
                workflow_id=workflow_id,
                outcome="success" if ok else "failure",
                canonical_hash=self._fingerprint(
                    {
                        "query": query,
                        "delegation": self.delegation_config,
                    }
                ),
                step_count=len(report.get("tasks") or []),
                skills_activated=[],
                scope=scope,
            )
        except Exception:
            logger.debug("Delegation learning capture failed", exc_info=True)

    def list_approval_proposals(
        self, *, status: Optional[Union[ApprovalStatus, str]] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List durable approval proposals owned by this agent."""
        return [
            proposal.to_dict(include_arguments=True)
            for proposal in self._get_approval_store().list(
                owner_id=self.agent_id, status=status, limit=limit
            )
        ]

    def approve(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a host-side approval decision without executing the tool."""
        proposal = self._get_approval_store().approve(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        return proposal.to_dict(include_arguments=True)

    def reject(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a host-side rejection decision."""
        proposal = self._get_approval_store().reject(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        )
        return proposal.to_dict(include_arguments=True)

    def cancel_approval(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a pending proposal (host-friendly alias for rejection)."""
        return self.reject(
            proposal_id,
            approver_id=approver_id,
            reason=reason or "Cancelled by host",
        )

    def resume_approval(
        self,
        proposal_id: str,
        *,
        continue_model: bool = True,
    ) -> ApprovalResumeResult:
        """Execute an approved checkpoint once and return structured evidence.

        Set ``continue_model=False`` for deterministic host workflows that need
        the exact tool result without asking an LLM to comment on it.
        """
        store = self._get_approval_store()
        proposal = store.get(proposal_id)
        if proposal is None:
            return ApprovalResumeResult(
                proposal=None,
                consumed=False,
                ok=False,
                error_code="approval_not_found",
                error=f"Unknown approval proposal '{proposal_id}'",
                requested_proposal_id=proposal_id,
            )
        if proposal.owner_id != self.agent_id:
            return ApprovalResumeResult(
                proposal=proposal,
                consumed=False,
                ok=False,
                error_code="approval_owner_mismatch",
                error="This proposal belongs to another agent",
            )
        if proposal.status != ApprovalStatus.APPROVED:
            return ApprovalResumeResult(
                proposal=proposal,
                consumed=proposal.status == ApprovalStatus.CONSUMED,
                ok=False,
                error_code="invalid_approval_state",
                error=(
                    "Only an approved proposal can be resumed "
                    f"(current state: {proposal.status.value})"
                ),
            )

        checkpoint = dict(proposal.checkpoint or {})
        logical_tool_name = str(
            checkpoint.get("logical_tool_name") or proposal.tool_name
        )
        logical_arguments = checkpoint.get("logical_arguments") or proposal.arguments
        if not isinstance(logical_arguments, dict):
            raise ApprovalStateError("Approval checkpoint arguments are invalid")
        consumed_proposal = store.consume(
            proposal_id,
            expected_tool_name=logical_tool_name,
            expected_arguments=logical_arguments,
        )

        self._current_memory_id = checkpoint.get("memory_id")
        self._current_thread_id = checkpoint.get("thread_id")
        user_id = checkpoint.get("user_id")
        self._current_user_id = user_id
        query = str(checkpoint.get("query") or "")
        messages = list(checkpoint.get("messages") or [])
        router = getattr(self, "semantic_tool_router", None)
        if router is not None:
            router.restore_state(checkpoint.get("router_state"))

        model_tool_name = str(checkpoint.get("model_tool_name") or logical_tool_name)
        model_arguments: Dict[str, Any] = logical_arguments
        if router is not None and model_tool_name == router.INVOCATION_TOOL:
            model_arguments = {
                "tool_name": logical_tool_name,
                "arguments": logical_arguments,
            }
        tool_call = SimpleNamespace(
            id=str(checkpoint.get("tool_call_id") or f"approval-{proposal_id}"),
            function=SimpleNamespace(
                name=model_tool_name,
                arguments=json.dumps(model_arguments, ensure_ascii=False),
            ),
        )
        workflow = self._init_workflow_capture(query, user_id)
        self._approval_execution_active = True
        raw_tool_result: Any = None
        try:
            try:
                raw_tool_result = self._execute_and_record_tool_call(
                    tool_call,
                    messages,
                    workflow,
                    user_id,
                    streaming=False,
                    query=query,
                )
            except Exception as exc:
                self._persist_workflow_run(workflow)
                return ApprovalResumeResult(
                    proposal=consumed_proposal,
                    consumed=True,
                    ok=False,
                    error_code="approved_tool_execution_failed",
                    error=str(exc),
                )
        finally:
            self._approval_execution_active = False

        raw_tool_result = _to_jsonable(raw_tool_result)
        if isinstance(raw_tool_result, str):
            try:
                raw_tool_result = json.loads(raw_tool_result)
            except (TypeError, ValueError):
                pass

        if not continue_model or not self.model:
            self._persist_workflow_run(workflow)
            return ApprovalResumeResult(
                proposal=consumed_proposal,
                tool_result=raw_tool_result,
                assistant_response=None,
                consumed=True,
            )

        tools = self._build_llm_tools(query, user_id=user_id)
        self._set_provider_cache_scope()
        try:
            for iteration in range(self._get_tool_iteration_limit()):
                response = self._generate_with_trace(
                    _to_jsonable(messages),
                    tools=tools,
                    iteration=iteration + 1,
                    stage="approval_resume",
                )
                self._record_context_window_usage(
                    stage=f"approval_resume_{iteration + 1}"
                )
                if isinstance(response, str):
                    self._persist_workflow_run(workflow)
                    return ApprovalResumeResult(
                        proposal=consumed_proposal,
                        tool_result=raw_tool_result,
                        assistant_response=response,
                        consumed=True,
                    )
                if hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message
                    if not message.tool_calls:
                        self._persist_workflow_run(workflow)
                        return ApprovalResumeResult(
                            proposal=consumed_proposal,
                            tool_result=raw_tool_result,
                            assistant_response=(
                                message.content or "I couldn't generate a response."
                            ),
                            consumed=True,
                        )
                    self._append_assistant_tool_calls(messages, message)
                    for next_call in message.tool_calls:
                        self._execute_and_record_tool_call(
                            next_call,
                            messages,
                            workflow,
                            user_id,
                            streaming=False,
                            query=query,
                        )
                    continue
                break
        except ApprovalRequired as approval:
            self._persist_workflow_run(workflow)
            return ApprovalResumeResult(
                proposal=consumed_proposal,
                tool_result=raw_tool_result,
                assistant_response=self._approval_required_payload(approval.proposal),
                consumed=True,
            )
        self._persist_workflow_run(workflow)
        return ApprovalResumeResult(
            proposal=consumed_proposal,
            tool_result=raw_tool_result,
            assistant_response=(
                "I reached the maximum number of tool-call iterations while resuming."
            ),
            consumed=True,
        )

    def run_stream_events(
        self, query, *, delivery_mode=None, cancellation=None, queue_size=64, **kwargs
    ):
        """Return a closeable, ordered v1 event stream (no private reasoning)."""
        from ..streaming import agent_event_stream

        return agent_event_stream(
            self,
            query,
            delivery_mode=delivery_mode,
            cancellation=cancellation,
            queue_size=queue_size,
            **kwargs,
        )

    async def arun_stream_events(self, query, **kwargs):
        """Async event iterator backed by one owned execution context/worker."""
        stream = self.run_stream_events(query, **kwargs)
        async for event in stream.async_events():
            yield event

    def run_stream(
        self,
        query: str,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        raise_on_provider_error: bool = False,
        event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        observability_context: Optional[Dict[str, Any]] = None,
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
            raise_on_provider_error: Re-raise LLM/provider API failures after
                emitting typed terminal events. Defaults to ``False`` for
                compatibility with UI streams that render inline failures.
            event_callback: Optional callback scoped to this stream only. This
                is concurrency-safe and preferred over mutating a shared agent
                with ``set_stream_event_callback`` immediately before a run.
            observability_context: Optional content-free host provenance. See
                ``run()``; the same strict persistence allowlist applies.

        Yields:
            str: Partial text chunks of the agent's response
        """
        logger.info(f"MemAgent {self.agent_id} streaming query: {query[:50]}...")

        if (
            self.meta_harness is not None
            and self.meta_harness_mode == "runtime"
            and not (tool_context or {}).get("_memorizz_harness_native")
        ):
            yield self.run(
                query,
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                tool_context=tool_context,
                observability_context=observability_context,
            )
            return

        if (
            self.delegates
            and self.delegation_config.get("enabled", True)
            and self.delegation_config.get("mode", "auto") in {"auto", "deterministic"}
            and not (tool_context or {}).get("_memorizz_skip_delegation")
        ):
            result = self.delegate(
                query,
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                tool_context=tool_context,
            )
            yield serialize_tool_result(result)
            return

        from ..llms.streaming import streaming_capabilities

        if self.model is None and session_for(self) is not None:
            from ..llms.streaming import ProviderStreamError

            raise ProviderStreamError("model_not_configured")
        if not self.model or not streaming_capabilities(self.model)["text_deltas"]:
            # Fallback: run synchronously and yield the full result
            yield self.run(
                query,
                memory_id=memory_id,
                thread_id=thread_id,
                user_id=user_id,
                context=context,
                tool_context=tool_context,
                observability_context=observability_context,
            )
            return

        # M4: scope per-call tool context for tools reading via get_tool_context().
        from ..tool_context import reset_tool_context, set_tool_context

        _tc_token = set_tool_context(tool_context or {})
        previous_event_callback = self._stream_event_callback
        if event_callback is not None:
            self._stream_event_callback = event_callback
        trace_initialized = False
        trace_finished = False
        turn_status = "error"
        turn_error_code: Optional[str] = None
        session = session_for(self)
        try:
            check_cancelled()
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, thread_id = self._resolve_execution_state(memory_id, thread_id)
            self._begin_trace_turn(user_id, query=query, emit_start=False)
            trace_initialized = True
            # M3: the lifecycle event is emitted after ID resolution so every
            # callback receives the same durable identity as persisted spans.
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
            self._emit_trace_turn_start()
            self._emit_context_provenance_trace(observability_context, context)
            self._turn_had_side_effects = False
            self._turn_had_nondeterministic_tools = False
            self._turn_cache_domains = set()
            self._cache_bypass_reason = None
            self._last_completion_decisions = []
            cache_metadata = self._semantic_cache_metadata(context)
            cache_lookup_bypass = self._semantic_cache_preflight_bypass(
                query, user_id=user_id
            )
            if not self.cache_manager.enabled:
                self._emit_cache_decision_trace("disabled", cache_metadata)
            elif cache_lookup_bypass:
                self._emit_cache_decision_trace(
                    "bypassed", cache_metadata, bypass_reason=cache_lookup_bypass
                )
            if cache_lookup_bypass and self.learning_control_plane is not None:
                try:
                    self.learning_control_plane.record_cache(
                        hit=False,
                        query=query,
                        reason=cache_lookup_bypass,
                        scope=self._trace_identity_payload(),
                    )
                except Exception:
                    pass

            # 2. Check semantic cache first
            if self.cache_manager.enabled:
                cached = self.cache_manager.get_cached_response(
                    query,
                    thread_id,
                    user_id=user_id,
                    metadata=cache_metadata,
                    bypass_reason=cache_lookup_bypass,
                )
                if not cache_lookup_bypass:
                    self._emit_cache_decision_trace(
                        "hit" if cached else "miss", cache_metadata
                    )
                if cached:
                    cache_decision = self._evaluate_completion_candidate(
                        query=query,
                        response=cached,
                        iteration=0,
                        tool_call_count=0,
                    )
                    if cache_decision.accepted:
                        yield cached
                        if session:
                            session.answer_done()
                        check_cancelled()
                        if self.learning_control_plane is not None:
                            try:
                                self.learning_control_plane.record_cache(
                                    hit=True,
                                    query=query,
                                    reason="exact_or_semantic_cache_hit",
                                    scope=self._trace_identity_payload(),
                                )
                            except Exception:
                                pass
                        self._record_interaction(
                            query, cached, memory_id, thread_id, user_id=user_id
                        )
                        turn_status = "success"
                        self._finish_trace_turn(turn_status)
                        trace_finished = True
                        self._emit_stream_event(
                            "stream_end",
                            {"reason": "cache_hit", "agent_id": self.agent_id},
                        )
                        return
                    logger.info(
                        "Cached stream response rejected by completion policy "
                        "(%s); continuing with a fresh model turn",
                        cache_decision.code,
                    )
                    self._emit_cache_decision_trace(
                        "rejected",
                        cache_metadata,
                        bypass_reason=cache_decision.code,
                    )

            # 3. Build context and prompt
            built_context = self._build_context(
                query,
                memory_id,
                user_id=user_id,
                include_entity_memory=not self._host_owns_entity_personalization(
                    context
                ),
            )
            built_context = self._attach_personalization_context(built_context, context)
            self._emit_memory_context_trace(built_context)
            system_prompt = self._build_system_prompt()
            if session:
                session.emit("status", stage="context_ready")

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
            if session and session.outcome != "completed":
                turn_status = session.outcome
                turn_error_code = session.error_code
                return
            check_cancelled()
            if session:
                session.answer_done()
            self._emit_memory_reference_trace(full_response)

            # 5. Cache the response
            if self.cache_manager.enabled:
                if session:
                    session.persistence["cache"] = "unknown"
                self.cache_manager.cache_response(
                    query,
                    full_response,
                    thread_id,
                    user_id=user_id,
                    metadata=self._semantic_cache_metadata(context),
                    deterministic=not self._turn_had_nondeterministic_tools,
                    read_only=not self._turn_had_side_effects,
                    bypass_reason=self._cache_bypass_reason,
                )

            # 6. Record interaction in memory
            self._record_interaction(
                query, full_response, memory_id, thread_id, user_id=user_id
            )
            turn_status = "success"
            self._finish_trace_turn(turn_status)
            trace_finished = True
            self._emit_stream_event(
                "stream_end",
                {
                    "reason": "completed",
                    "agent_id": self.agent_id,
                    "response_length": len(full_response),
                },
            )

        except StreamCancelled:
            turn_status, turn_error_code = "cancelled", "cancelled"
            raise
        except Exception as e:
            # Include full traceback so we can pinpoint which step in the
            # streaming pipeline failed (LOB-serialization bugs surface here
            # without a locator, which is useless for debugging).
            logger.exception("MemAgent streaming failed: %s", e)
            # M3: structured error event so SSE translators / UIs can render
            # an inline error banner instead of treating the error as text.
            is_provider_error, is_auth_error, provider_status = _provider_error_details(
                e
            )
            error_code = (
                "provider_authentication_failed"
                if is_auth_error
                else "provider_error"
                if is_provider_error
                else "stream_error"
            )
            turn_error_code = error_code
            if trace_initialized:
                self._finish_trace_turn("error", error_code=turn_error_code)
                trace_finished = True
            self._emit_stream_event(
                "error",
                {
                    "message": str(e),
                    "exception_type": type(e).__name__,
                    "recoverable": False,
                    "terminal": True,
                    "error_code": error_code,
                    "provider_status_code": provider_status,
                    "agent_id": self.agent_id,
                },
            )
            self._emit_stream_event(
                "stream_end",
                {
                    "reason": "error",
                    "error_code": error_code,
                    "agent_id": self.agent_id,
                },
            )
            if isinstance(e, CompletionRejectedError):
                raise
            if session:
                raise
            if raise_on_provider_error and is_provider_error:
                raise
            yield f"I apologize, but I encountered an error: {str(e)}"
        finally:
            if trace_initialized:
                if not trace_finished:
                    self._finish_trace_turn(
                        turn_status,
                        error_code=turn_error_code,
                    )
                self._record_stream_trace_bundle(
                    memory_id,
                    thread_id,
                    user_id=user_id,
                )
            self._stream_trace_events = None
            if event_callback is not None:
                self._stream_event_callback = previous_event_callback
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
                      ``terminal``, ``error_code``, optional
                      ``provider_status_code``, and ``agent_id``. Fired when
                      run_stream catches an exception mid-stream — UIs should
                      render an error banner.
        stream_end    ``reason`` (``completed``/``cache_hit``/``error``),
                      ``agent_id``, optional ``response_length``. Fired once
                      when the stream terminates.
        completion_check  Candidate acceptance, code, digest and attempt evidence.
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
        enriched_payload = dict(payload or {})
        if self._current_root_trace_id:
            for key, value in self._trace_identity_payload().items():
                enriched_payload.setdefault(key, value)

        if event_type == "trace" and self._stream_trace_events is not None:
            self._stream_trace_events.append(dict(enriched_payload))

        callback = self._stream_event_callback
        if not callback:
            return
        event: Dict[str, Any] = {"type": event_type}
        if enriched_payload:
            event.update(enriched_payload)
        try:
            callback(event)
        except Exception as exc:
            from ..observability.pipeline import record_pipeline_metric

            record_pipeline_metric(self.memory_provider, "callback_failures")
            logger.debug("Stream event callback failed (%s)", type(exc).__name__)

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
    ) -> List[Dict[str, Any]]:
        """Normalize and compact streamed trace events for persistence."""
        if not events:
            return []

        normalized: List[Dict[str, Any]] = []
        by_trace_id: Dict[str, Dict[str, Any]] = {}
        # Keep a deliberately bounded set of structured attributes. These are
        # needed for reliable analysis in the UI, while arbitrary callback
        # payloads must not become an accidental persistence channel.
        metadata_fields = (
            "schema_version",
            "application_id",
            "agent_id",
            "run_id",
            "turn_id",
            "root_trace_id",
            "span_id",
            "parent_span_id",
            "memory_id",
            "thread_id",
            "user_id",
            "timestamp",
            "tool_name",
            "logical_tool_name",
            "model_tool_name",
            "tool_call_id",
            "success",
            "status",
            "outcome",
            "outcome_reason_code",
            "tool_provider",
            "primary_provider",
            "fallback_provider",
            "outcome_retryable",
            "result_count",
            "fallback_used",
            "degraded",
            "error_code",
            "duration_ms",
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cost_usd",
            "finish_reason",
            "max_output_tokens",
            "response_chars",
            "response_bytes",
            "ttft_ms",
            "stream_duration_ms",
            "retry_count",
            "fallback_count",
            "iteration",
            "stage",
            "total_tokens",
            "request_id",
            "client_page_type",
            "client_page_id",
            "client_title_fingerprint",
            "canonical_page_type",
            "canonical_page_id",
            "canonical_title_fingerprint",
            "thread_binding_status",
            "expected_thread_id",
            "ownership_verified",
            "request_context_present",
            "request_context_fingerprint",
            "request_context_key_count",
            "content_version",
            "grounding_status",
            "grounding_source",
            "grounding_excerpt_count",
            "grounding_source_ids",
            "cache_decision",
            "memory_history_count",
            "memory_candidate_count",
            "memory_supplied_count",
            "memory_referenced_count",
            "memory_injected_chars",
            "memory_degraded",
            "memory_fallback_used",
            "entity_profile_count",
            "preference_count",
            "conversation_memory_count",
            "writing_sample_count",
            "cache_enabled",
            "cache_bypass_reason",
        )

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
            from ..observability.models import ResourceRef, SelectionDecision

            for field, model, maximum in (
                ("selection_ledger", SelectionDecision, 64),
                ("input_refs", ResourceRef, 32),
                ("output_refs", ResourceRef, 32),
            ):
                if isinstance(event.get(field), list):
                    try:
                        entry[field] = [
                            model.model_validate(value).model_dump(
                                mode="json", exclude_none=True
                            )
                            for value in event[field][:maximum]
                        ]
                    except ValueError:
                        from ..observability.pipeline import record_pipeline_metric

                        record_pipeline_metric(
                            self.memory_provider, "dropped_metadata_fields"
                        )
            for field in metadata_fields:
                value = event.get(field)
                if field == "grounding_source_ids" and isinstance(value, (list, tuple)):
                    entry[field] = [
                        item[:240] for item in value[:16] if isinstance(item, str)
                    ]
                    continue
                if value is None or not isinstance(value, (str, int, float, bool)):
                    continue
                entry[field] = value[:240] if isinstance(value, str) else value
            normalized.append(entry)

        return [item for item in normalized if item.get("title") or item.get("content")]

    def _record_stream_trace_bundle(
        self,
        memory_id: str,
        thread_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Persist reasoning/tool traces outside the conversational recall path."""
        session = session_for(self)
        if not self.memory_provider:
            if session:
                session.persistence["trace"] = "not_configured"
            return
        # A handful of legacy provider-like integrations expose ``store`` but
        # no way to list or query a memory type. Writing trace bundles there
        # would create permanently unreadable data. Full MemoryProvider
        # implementations always expose at least one of these read surfaces.
        if not any(
            callable(getattr(self.memory_provider, method_name, None))
            for method_name in ("query_observability_records", "list_all")
        ):
            return

        trace_events = self._build_trace_bundle_events(self._stream_trace_events)
        if not trace_events:
            return

        try:
            from ..observability import ObservabilityStore

            context = {
                **self._trace_identity_payload(),
                "memory_id": memory_id,
                "thread_id": thread_id,
                "user_id": user_id,
            }
            ObservabilityStore(self.memory_provider).record_trace_bundle(
                trace_context=context,
                events=trace_events,
            )
            if session:
                session.persistence["trace"] = "written"
            logger.debug(
                "Recorded private trace bundle: %s events for thread %s",
                len(trace_events),
                thread_id,
            )
        except Exception as exc:
            logger.warning("Failed to record trace bundle: %s", exc)
            if session:
                session.persistence["trace"] = "failed"

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
                memory_id=self._current_memory_id,
                limit=limit,
                user_id=self._current_user_id,
                thread_id=self._current_thread_id,
            )
        except Exception as exc:
            logger.debug("Tool-log digest lookup failed: %s", exc)
            return ""
        if not isinstance(rows, (list, tuple)) or not rows:
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
            ids = _extract_identifiers(raw_result)
            bits = [f"tool={tname}"]
            if args_text:
                bits.append(f"args={args_text}")
            # ``ids`` BEFORE ``digest``: identifiers are the thing the agent
            # most needs to re-grab next turn, and they must survive even when
            # the shape-aware digest collapses to e.g. ``ok=true``.
            if ids:
                bits.append(f"ids={ids}")
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

    def _build_llm_tools(
        self, query: str = "", user_id: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Build a strict, progressively disclosed tool surface for one turn."""
        if not self.tool_manager:
            return None
        router = getattr(self, "semantic_tool_router", None)
        if router is not None:
            tools = router.schemas_for_turn(query, user_id=user_id)
            return tools or None
        tools = [
            schema
            for metadata in (self.tool_manager.get_tool_metadata() or [])
            if (schema := tool_metadata_to_openai(metadata)) is not None
        ]
        tools.sort(key=lambda item: str(item["function"]["name"]))
        return tools or None

    def _set_provider_cache_scope(self) -> None:
        """Give the LLM provider a stable per-thread prompt-cache key.

        OpenAI routes requests to cache shards by prefix-hash + this key;
        pinning it to the conversation thread keeps every turn of a thread on
        the same shard. Providers without ``set_prompt_cache_key`` (Anthropic,
        Ollama, local servers) are silently skipped.
        """
        setter = getattr(self.model, "set_prompt_cache_key", None)
        if not callable(setter):
            return
        try:
            setter(
                "memorizz:{agent}:{memory}:{thread}".format(
                    agent=self.agent_id or "anon",
                    memory=self._current_memory_id or "default",
                    thread=self._current_thread_id or "main",
                )
            )
        except Exception as exc:
            logger.debug("Failed to set prompt cache key: %s", exc)

    def _persist_workflow_run(self, workflow) -> bool:
        """Store a captured workflow and feed the continual-learning loop.

        Single funnel for BOTH the streaming and non-streaming capture
        paths, so per-run enrichment (canonical hashing inside
        ``store_workflow``, skill attribution, promotion scheduling) can
        never drift between them. Learning hooks are fail-safe: they must
        never break a user-facing run.
        """
        if not (workflow and workflow.steps) or self.memory_provider is None:
            return False
        self._apply_workflow_outcome_evaluator(workflow)
        try:
            record_id = workflow.store_workflow(self.memory_provider)
            logger.info("Stored workflow with %s steps", len(workflow.steps))
        except Exception as exc:
            logger.error("Error storing workflow: %s", exc)
            return False
        if self.continual_learning_manager:
            try:
                self.continual_learning_manager.record_run_outcome(
                    workflow, record_id=record_id
                )
                self.continual_learning_manager.maybe_run_scheduled_cycle()
            except Exception as exc:
                logger.error("Continual-learning post-run hook failed: %s", exc)
        if self.learning_control_plane is not None:
            try:
                outcome_value = getattr(
                    getattr(workflow, "outcome", None), "value", "unknown"
                )
                self.learning_control_plane.record_workflow(
                    workflow_id=str(workflow.workflow_id),
                    outcome=str(outcome_value),
                    canonical_hash=getattr(workflow, "canonical_hash", None),
                    step_count=len(workflow.steps),
                    skills_activated=list(
                        getattr(workflow, "skills_activated", None) or []
                    ),
                    scope={
                        **self._trace_identity_payload(),
                        "workflow_id": str(workflow.workflow_id),
                    },
                )
            except Exception as exc:
                logger.debug("Learning workflow-event capture failed: %s", exc)
        try:
            outcome_value = getattr(getattr(workflow, "outcome", None), "value", None)
            if outcome_value in {"success", "failure"}:
                trace_context = self.get_trace_context()
                trace_context["workflow_id"] = str(workflow.workflow_id)
                self.record_task_outcome(
                    outcome_value,
                    # Without an application evaluator this is execution
                    # health, not verified business success.
                    verified=self.workflow_outcome_evaluator is not None,
                    source=(
                        "workflow_outcome_evaluator"
                        if self.workflow_outcome_evaluator is not None
                        else "tool_execution"
                    ),
                    metrics={"step_count": len(workflow.steps)},
                    trace_context=trace_context,
                    external_id=str(workflow.workflow_id),
                )
        except Exception as exc:
            logger.debug("Workflow outcome trace linkage failed: %s", exc)
        return True

    def _apply_workflow_outcome_evaluator(self, workflow) -> None:
        """Apply an application business-success rubric before persistence.

        Tool execution only proves that a run completed without raising. A
        configured evaluator can inspect the complete captured ``Workflow``
        and return ``True``/``False``, ``WorkflowOutcome``,
        ``"success"``/``"failure"``, or ``None`` to keep the current outcome.
        Evaluation is fail-closed for learning, but never interrupts the
        user-facing run. An execution failure cannot be upgraded to success.
        """
        evaluator = self.workflow_outcome_evaluator
        if evaluator is None:
            return

        from ..long_term.procedural.workflow.workflow import WorkflowOutcome

        execution_failed = workflow.outcome == WorkflowOutcome.FAILURE
        try:
            result = evaluator(workflow)
        except Exception as exc:
            workflow.outcome = WorkflowOutcome.FAILURE
            logger.warning(
                "Workflow outcome evaluator failed; recording learning failure: %s",
                exc,
            )
            return

        if result is None:
            return
        if isinstance(result, WorkflowOutcome):
            evaluated = result
        elif isinstance(result, bool):
            evaluated = WorkflowOutcome.SUCCESS if result else WorkflowOutcome.FAILURE
        elif isinstance(result, str):
            try:
                evaluated = WorkflowOutcome(result.strip().lower())
            except ValueError:
                workflow.outcome = WorkflowOutcome.FAILURE
                logger.warning(
                    "Workflow outcome evaluator returned unsupported value %r; "
                    "recording learning failure",
                    result,
                )
                return
        else:
            workflow.outcome = WorkflowOutcome.FAILURE
            logger.warning(
                "Workflow outcome evaluator returned unsupported type %s; "
                "recording learning failure",
                type(result).__name__,
            )
            return

        if execution_failed and evaluated == WorkflowOutcome.SUCCESS:
            logger.warning(
                "Workflow outcome evaluator attempted to upgrade an execution "
                "failure; preserving failure"
            )
            return
        workflow.outcome = evaluated

    def _init_workflow_capture(self, query: str, user_id: Optional[str]):
        """Create the per-run Workflow tracker when workflow memory is active.

        Shared by the streaming and non-streaming loops so per-run
        enrichment (user scope, skill attribution) can never drift between
        them. A configured memory type alone is not a persistence backend:
        provider-less agents must remain stateless and must not construct an
        embedding-bearing Workflow that can never be stored.
        """
        if (
            self.memory_provider is None
            or MemoryType.WORKFLOW_MEMORY not in self.active_memory_types
        ):
            return None
        from ..long_term.procedural.workflow.workflow import Workflow

        workflow = Workflow(
            name="Tool Execution for Query",
            description=f"Workflow tracking tool usage for: {query[:100]}",
            memory_id=self._current_memory_id
            or (self.memory_ids[0] if self.memory_ids else str(uuid.uuid4())),
            agent_id=self.agent_id,
            user_query=query,
            user_id=user_id,
            skills_activated=list(self._activated_skill_ids or []),
        )
        logger.debug(f"Created workflow for tracking: {workflow.workflow_id}")
        return workflow

    @staticmethod
    def _append_assistant_tool_calls(
        messages: List[Dict[str, Any]], message: Any
    ) -> None:
        """Append the assistant's tool-call message to the running history."""
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

    def _get_approval_store(self) -> ApprovalStore:
        """Create the package-owned durable store only when approval is needed."""
        if self.approval_store is None:
            self.approval_store = default_approval_store()
        return self.approval_store

    def _effective_tool_policy(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve static callable policy plus argument-dependent MCP policy."""
        policy = (
            self.tool_manager.get_tool_policy(tool_name) if self.tool_manager else {}
        )
        value = dict(policy or {})
        if tool_name == "mcp_call_tool":
            try:
                server = self.mcp_manager.get_server(
                    str(arguments.get("server_name") or "")
                )
                remote_name = str(arguments.get("tool_name") or "")
                mutating = self.mcp_manager.tool_requires_approval(
                    server.name, remote_name
                )
                value.update(
                    {
                        "deterministic": False,
                        "side_effects": mutating,
                        "requires_approval": mutating,
                        "approval_reason": (
                            f"MCP tool {server.name}.{remote_name} may change external data"
                            if mutating
                            else None
                        ),
                        "domains": ["mcp", server.name],
                    }
                )
            except Exception:
                pass
        return value

    @staticmethod
    def _approval_required_payload(proposal: Any) -> str:
        return json.dumps(
            {
                "ok": False,
                "status": "approval_required",
                "error_code": "approval_required",
                "proposal": proposal.to_dict(include_arguments=True),
                "message": (
                    "Execution is paused. A host user must approve or reject "
                    "this exact proposal, then resume it by proposal_id."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _tool_result_failed(value: Any) -> bool:
        _payload, outcome = normalize_tool_result(value)
        return not outcome.ok

    def _execute_and_record_tool_call(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        workflow,
        user_id: Optional[str],
        streaming: bool = False,
        query: str = "",
    ) -> Any:
        """Execute one tool call and record every side effect.

        The single shared body for BOTH the streaming and non-streaming
        loops: argument parsing, execution, tool-log offload, compact
        placeholder, message append, placeholder persistence, and workflow
        step capture. These ~150 lines used to exist twice and had already
        drifted once; ``streaming`` only toggles the trace-event emission.
        """
        from ..long_term.procedural.workflow.workflow import WorkflowOutcome

        # Keep context tools and direct/internal execution paths on the same
        # tenant boundary as ``run``/``run_stream``.  Approval resumption and
        # host-driven tool execution both enter through this shared method and
        # must not rely on a prior model turn having populated the field.
        self._current_user_id = user_id
        tool_started_at = time.perf_counter()

        tool_name = tool_call.function.name
        raw_arguments = tool_call.function.arguments

        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, Exception):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        model_tool_name = tool_name
        logical_tool_name = tool_name
        logical_arguments = arguments
        routed_warnings: List[str] = []
        routed_call_hash: Optional[str] = None
        error_message: Optional[str] = None
        result: Any = None
        tool_outcome: Optional[ToolOutcome] = None
        router = getattr(self, "semantic_tool_router", None)

        # A routed model call is named ``invoke_tool``, but that transport name
        # is not useful to application traces. Resolve the requested logical
        # name before emitting the call event while retaining both fields for
        # audit/debugging consumers.
        if router is not None and model_tool_name == router.INVOCATION_TOOL:
            requested_name = str(arguments.get("tool_name") or "").strip()
            if requested_name:
                logical_tool_name = router._resolve_name(requested_name)
            nested_arguments = arguments.get("arguments")
            if isinstance(nested_arguments, dict):
                logical_arguments = nested_arguments

        tool_trace_id = (
            f"{model_tool_name}:{tool_call.id}"
            if getattr(tool_call, "id", None)
            else f"{model_tool_name}:{uuid.uuid4()}"
        )
        tool_span_id = str(uuid.uuid4())
        tool_parent_span_id = (
            self._current_parent_span_id or self._current_root_trace_id
        )
        self._emit_stream_event(
            "trace",
            {
                "trace_kind": "tool_call",
                "title": f"Tool Call: {logical_tool_name}",
                "tool_name": logical_tool_name,
                "logical_tool_name": logical_tool_name,
                "model_tool_name": model_tool_name,
                "tool_call_id": getattr(tool_call, "id", None),
                "trace_id": f"call:{tool_trace_id}",
                "span_id": tool_span_id,
                "parent_span_id": tool_parent_span_id,
                "status": "started",
                "content": self._preview_stream_payload(
                    logical_arguments,
                    limit=1400,
                ),
            },
        )

        if router is not None and model_tool_name == router.DISCOVERY_TOOL:
            result = router.discover_tools(
                str(arguments.get("query") or ""),
                limit=arguments.get("limit", router.top_k),
                user_id=user_id,
            )
        else:
            if router is not None and model_tool_name == router.INVOCATION_TOOL:
                try:
                    (
                        logical_tool_name,
                        logical_arguments,
                        routed_warnings,
                    ) = router.normalize_invocation(
                        str(arguments.get("tool_name") or ""),
                        arguments.get("arguments") or {},
                    )
                except (LookupError, PermissionError, TypeError, ValueError) as exc:
                    result = {
                        "ok": False,
                        "error_code": "invalid_tool_invocation",
                        "error": str(exc),
                    }
                    error_message = str(exc)
            elif router is not None:
                # Direct calls are accepted only for schemas actually disclosed
                # this turn. This closes the gap where a provider could invent a
                # hidden registered function name despite never receiving it.
                try:
                    (
                        logical_tool_name,
                        logical_arguments,
                        routed_warnings,
                    ) = router.normalize_invocation(model_tool_name, arguments)
                except (LookupError, PermissionError, TypeError, ValueError) as exc:
                    result = {
                        "ok": False,
                        "error_code": "invalid_tool_invocation",
                        "error": str(exc),
                    }
                    error_message = str(exc)

            if result is None:
                policy = self._effective_tool_policy(
                    logical_tool_name, logical_arguments
                )
                if policy.get("side_effects"):
                    self._turn_had_side_effects = True
                    self._cache_bypass_reason = "side_effecting_tool"
                if policy.get("deterministic") is False:
                    self._turn_had_nondeterministic_tools = True
                self._turn_cache_domains.update(
                    str(item) for item in (policy.get("domains") or []) if item
                )
                if (
                    policy.get("requires_approval")
                    and not self._approval_execution_active
                ):
                    checkpoint = {
                        "version": 1,
                        "query": query,
                        "messages": _to_jsonable(messages),
                        "model_tool_name": model_tool_name,
                        "logical_tool_name": logical_tool_name,
                        "logical_arguments": _to_jsonable(logical_arguments),
                        "tool_call_id": getattr(tool_call, "id", None),
                        "memory_id": self._current_memory_id,
                        "thread_id": self._current_thread_id,
                        "user_id": user_id,
                        "streaming": bool(streaming),
                        "routed_warnings": routed_warnings,
                        "routed_call_hash": routed_call_hash,
                        "router_state": (
                            router.checkpoint_state() if router is not None else {}
                        ),
                    }
                    proposal = self._get_approval_store().propose(
                        owner_id=self.agent_id,
                        tool_name=logical_tool_name,
                        arguments=logical_arguments,
                        policy_reason=str(
                            policy.get("approval_reason")
                            or "Tool policy requires human approval"
                        ),
                        checkpoint=checkpoint,
                        ttl_seconds=self.context_policy.approval_ttl_seconds,
                    )
                    self._cache_bypass_reason = "approval_required"
                    self._emit_stream_event(
                        "trace",
                        {
                            "trace_kind": "tool_result",
                            "title": f"Tool paused: {logical_tool_name}",
                            "tool_name": logical_tool_name,
                            "logical_tool_name": logical_tool_name,
                            "model_tool_name": model_tool_name,
                            "tool_call_id": getattr(tool_call, "id", None),
                            "trace_id": f"result:{tool_trace_id}",
                            "span_id": tool_span_id,
                            "parent_span_id": tool_parent_span_id,
                            "status": "approval_required",
                            "success": False,
                            "error_code": "approval_required",
                            "duration_ms": round(
                                (time.perf_counter() - tool_started_at) * 1000,
                                3,
                            ),
                            "content": "",
                        },
                    )
                    raise ApprovalRequired(proposal)

                if router is not None:
                    prepared = router.prepare_invocation(
                        logical_tool_name, logical_arguments
                    )
                    if not prepared.get("ok"):
                        result = prepared
                        error_message = str(prepared.get("error") or "Routing failed")
                    else:
                        routed_call_hash = str(prepared["call_hash"])

                if result is None:
                    logger.info("Executing tool: %s", logical_tool_name)
                    if self.tool_manager:
                        result, _ = self.tool_manager.execute_tool(
                            logical_tool_name, logical_arguments
                        )
                    else:
                        result = "Error: No tool manager available"
                        error_message = "No tool manager available"

                result, tool_outcome = normalize_tool_result(result)
                failed = not tool_outcome.ok
                if failed and error_message is None:
                    error_message = (
                        str(
                            result.get("error")
                            or result.get("message")
                            or tool_outcome.reason_code
                            or "Tool execution failed"
                        )
                        if isinstance(result, dict)
                        else str(result)
                    )
                if router is not None and routed_call_hash:
                    router.record_invocation(routed_call_hash, success=not failed)
                if router is not None and model_tool_name == router.INVOCATION_TOOL:
                    result = {
                        "ok": not failed,
                        "tool_name": logical_tool_name,
                        "result": result,
                        "call_hash": routed_call_hash,
                        "warnings": routed_warnings,
                        "outcome": tool_outcome.to_dict(),
                    }

        if tool_outcome is None:
            result, tool_outcome = normalize_tool_result(result)
        tool_failed = not tool_outcome.ok
        if (
            tool_outcome.status is not ToolOutcomeStatus.SUCCESS
            and not self._cache_bypass_reason
        ):
            self._cache_bypass_reason = f"tool_outcome_{tool_outcome.status.value}"
        outcome_payload = tool_outcome.to_dict()
        duration_ms = round((time.perf_counter() - tool_started_at) * 1000, 3)
        error_code = (
            _to_jsonable(result.get("error_code"))
            if isinstance(result, dict) and result.get("error_code") is not None
            else outcome_payload.get("reason_code", "")
        )
        outcome_record = {
            "tool_name": logical_tool_name,
            "model_tool_name": model_tool_name,
            "tool_call_id": getattr(tool_call, "id", None),
            "duration_ms": duration_ms,
            **outcome_payload,
        }
        self._last_tool_outcomes = [
            *list(self._last_tool_outcomes or []),
            outcome_record,
        ]
        self._emit_stream_trace_chunks(
            "tool_result",
            f"Tool Result: {logical_tool_name}",
            result,
            trace_id=f"result:{tool_trace_id}",
            chunk_size=420,
            preview_limit=12000,
            extra={
                "tool_name": logical_tool_name,
                "logical_tool_name": logical_tool_name,
                "model_tool_name": model_tool_name,
                "tool_call_id": getattr(tool_call, "id", None),
                "span_id": tool_span_id,
                "parent_span_id": tool_parent_span_id,
                "status": "error" if tool_failed else "success",
                "outcome": tool_outcome.status.value,
                "outcome_reason_code": tool_outcome.reason_code,
                "tool_provider": tool_outcome.provider,
                "primary_provider": tool_outcome.primary_provider,
                "fallback_provider": tool_outcome.fallback_provider,
                "outcome_retryable": tool_outcome.retryable,
                "result_count": tool_outcome.result_count,
                "fallback_used": tool_outcome.fallback_used,
                "degraded": tool_outcome.degraded,
                "success": not tool_failed,
                "error_code": error_code,
                "duration_ms": duration_ms,
            },
        )

        # Serialize exactly once and decide before persistence. Small results
        # remain inline and create no tool-log row; expansion tools can never
        # create pointer-to-pointer loops.
        tool_log_id = None
        result_str = serialize_tool_result(result)
        is_expansion_tool = (
            logical_tool_name in self.tool_result_policy.expansion_tool_names
        )
        offload_eligible = (
            not is_expansion_tool
            and self.tool_result_policy.should_offload(logical_tool_name, result_str)
        )
        if offload_eligible and self.memory_manager and self._current_memory_id:
            try:
                tool_log_id = self.memory_manager.store_tool_log(
                    tool_name=logical_tool_name,
                    arguments=logical_arguments,
                    result=result_str,
                    memory_id=self._current_memory_id,
                    agent_id=self.agent_id,
                    tool_call_id=getattr(tool_call, "id", None),
                    success=tool_outcome.ok,
                    error=error_message,
                    outcome=tool_outcome.status.value,
                    outcome_details=outcome_payload,
                    thread_id=self._current_thread_id,
                    user_id=user_id,
                )
            except Exception as log_exc:
                logger.debug("Tool log storage failed: %s", log_exc)

        should_offload = bool(tool_log_id)
        if should_offload and tool_log_id:
            compact_result = self.tool_result_policy.pointer(
                tool_name=logical_tool_name,
                tool_log_id=tool_log_id,
                serialized_result=result_str,
                digest=_summarize_tool_result(result, preview_limit=360),
                identifiers=_extract_identifiers(result),
                tool_call_id=getattr(tool_call, "id", None),
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

        # Persist the placeholder as its own conversation_memory row so it
        # survives into future turns (UI audit + system-prompt digest).
        # Filtered from LLM message history via _is_tool_placeholder_content
        # to avoid orphan tool-role messages without matching tool_call_ids.
        if should_offload and self.memory_manager and self._current_memory_id:
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
                logger.debug("Tool placeholder persist failed: %s", persist_exc)

        if workflow:
            tool_entry = {}
            if self.tool_manager:
                tool_metadata = self.tool_manager.get_tool_metadata()
                tool_entry = next(
                    (
                        meta
                        for meta in tool_metadata
                        if meta.get("name") == logical_tool_name
                    ),
                    {},
                )

            workflow.add_step(
                f"Step {len(workflow.steps) + 1}: {logical_tool_name}",
                {
                    "_id": str(tool_entry.get("_id")) if tool_entry else None,
                    "arguments": logical_arguments,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "error": error_message,
                    "tool_outcome": outcome_payload,
                },
            )
            if error_message is not None:
                workflow.outcome = WorkflowOutcome.FAILURE

        if self.learning_control_plane is not None:
            try:
                self.learning_control_plane.record_tool(
                    tool_name=logical_tool_name,
                    arguments=logical_arguments,
                    result=result,
                    success=not tool_failed,
                    outcome=outcome_payload,
                    duration_ms=duration_ms,
                    scope=self._trace_identity_payload(),
                )
            except Exception as exc:
                logger.debug("Learning tool-event capture failed: %s", exc)

        # Return the complete in-process value. Callers that need deterministic
        # execution evidence (notably durable approval resumption) must not
        # reconstruct it from the LLM-facing message, which may deliberately be
        # a size-aware tool-log pointer.
        return result

    def _execute_llm_interaction_stream(
        self, system_prompt, query, context, user_id=None, request_context=None
    ):
        session = session_for(self)
        if session is None:
            yield from self._execute_llm_interaction_stream_legacy(
                system_prompt,
                query,
                context,
                user_id=user_id,
                request_context=request_context,
            )
            return
        from ..llms.streaming import ProviderStreamError, streaming_capabilities

        workflow = self._init_workflow_capture(query, user_id)
        messages = self._build_prompt_messages(
            system_prompt, query, context, request_context=request_context
        )
        if getattr(self, "semantic_tool_router", None) is not None:
            self.semantic_tool_router.begin_turn(user_id=user_id)
        tools = list(self._build_llm_tools(query, user_id=user_id) or [])
        if tools and not streaming_capabilities(self.model)["tool_calls"]:
            raise ProviderStreamError("provider_tools_unsupported")
        self._set_provider_cache_scope()
        finalize = "memorizz_finalize_answer"
        final_phase = not tools
        if session.mode == "final_stream" and tools:
            if any(tool.get("function", {}).get("name") == finalize for tool in tools):
                raise ProviderStreamError("reserved_finalizer_tool_name")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": finalize,
                        "description": "Finish the private tool phase. Call alone after collecting all required evidence; the host then enables public answer generation with tools disabled. Do not draft the answer before this call.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        },
                    },
                }
            )
            messages.append(
                {
                    "role": "system",
                    "content": "Use tools to collect evidence, then call memorizz_finalize_answer alone. Do not draft the public answer until the host accepts finalization.",
                }
            )
        tool_count = rejections = 0
        try:
            for iteration in range(self._get_tool_iteration_limit()):
                check_cancelled()
                session.attempt = iteration + 1
                # Only evidence-only policies are valid for this phase. No final
                # text validator is bypassed to obtain irreversible live output.
                if session.mode == "final_stream" and final_phase:
                    decision = self._evaluate_completion_candidate(
                        query=query,
                        response="",
                        iteration=iteration + 1,
                        tool_call_count=tool_count,
                    )
                    if not decision.accepted:
                        raise CompletionRejectedError(decision, rejections)
                pending, had_tools, terminal = [], False, False
                session.emit(
                    "status",
                    stage="answer_generation" if final_phase else "tool_phase",
                    attempt=iteration + 1,
                )
                provider = self._generate_stream_with_trace(
                    _to_jsonable(messages),
                    tools=None if final_phase else tools,
                    iteration=iteration + 1,
                    stage="agent_stream",
                )
                try:
                    for event in provider:
                        check_cancelled()
                        kind = event.get("type")
                        if kind == "content":
                            delta = event.get("content", "")
                            pending.append(delta)
                            if session.mode == "final_stream" and final_phase:
                                yield delta
                        elif kind == "usage":
                            usage = {
                                k: v
                                for k, v in (event.get("usage") or {}).items()
                                if k
                                in {
                                    "prompt_tokens",
                                    "completion_tokens",
                                    "total_tokens",
                                    "cached_tokens",
                                }
                                and (v is None or type(v) is int)
                            }
                            session.emit("usage", usage=usage, attempt=iteration + 1)
                        elif kind == "tool_calls":
                            self._record_context_window_usage(
                                stage=f"stream_iteration_{iteration + 1}"
                            )
                            terminal, had_tools = True, True
                            if final_phase:
                                raise ProviderStreamError(
                                    "unexpected_tool_call_in_final_answer"
                                )
                            message = event["response"].choices[0].message
                            self._append_assistant_tool_calls(messages, message)
                            for call in message.tool_calls:
                                check_cancelled()
                                if (
                                    call.function.name == finalize
                                    and session.mode == "final_stream"
                                ):
                                    decision = self._evaluate_completion_candidate(
                                        query=query,
                                        response="",
                                        iteration=iteration + 1,
                                        tool_call_count=tool_count,
                                    )
                                    accepted = (
                                        decision.accepted
                                        and len(message.tool_calls) == 1
                                    )
                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": call.id,
                                            "content": "Finalization accepted. Produce the public answer now; tools are disabled."
                                            if accepted
                                            else "Finalization rejected. Complete prerequisites and call this tool alone.",
                                        }
                                    )
                                    final_phase = accepted
                                else:
                                    tool_count += 1
                                    self._execute_and_record_tool_call(
                                        call,
                                        messages,
                                        workflow,
                                        user_id,
                                        streaming=True,
                                        query=query,
                                    )
                        elif kind == "done":
                            self._record_context_window_usage(
                                stage=f"stream_iteration_{iteration + 1}"
                            )
                            terminal = True
                            if had_tools:
                                continue
                            candidate = "".join(pending)
                            if event.get("content", candidate) != candidate:
                                raise ProviderStreamError("provider_delta_mismatch")
                            if session.mode == "final_stream" and not final_phase:
                                messages.append(
                                    {
                                        "role": "system",
                                        "content": "Finish tool evidence collection and call memorizz_finalize_answer alone; do not draft the answer.",
                                    }
                                )
                                break
                            decision = self._evaluate_completion_candidate(
                                query=query,
                                response=candidate,
                                iteration=iteration + 1,
                                tool_call_count=tool_count,
                            )
                            if not decision.accepted:
                                rejections += 1
                                if rejections > self.completion_policy.max_rejections:
                                    raise CompletionRejectedError(decision, rejections)
                                self._append_completion_rejection(
                                    messages,
                                    candidate,
                                    self.completion_policy.retry_message(decision),
                                )
                                break
                            if session.mode == "buffered" and candidate:
                                yield candidate
                            session.answer_done()
                            return
                finally:
                    provider.close()
                if not terminal:
                    raise ProviderStreamError("provider_stream_incomplete")
            raise ProviderStreamError("iteration_limit")
        except ApprovalRequired as approval:
            session.outcome = "approval_required"
            proposal = approval.proposal
            session.emit(
                "approval.required", proposal=proposal.to_dict(include_arguments=False)
            )
        finally:
            session.persistence["workflow"] = (
                "written"
                if self._persist_workflow_run(workflow)
                else "unknown"
                if workflow and workflow.steps
                else "not_requested"
            )

    def _execute_llm_interaction_stream_legacy(
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

        workflow = self._init_workflow_capture(query, user_id)

        workflow_persisted = False

        def _store_workflow_if_needed() -> None:
            nonlocal workflow_persisted
            if workflow_persisted:
                return
            if self._persist_workflow_run(workflow):
                workflow_persisted = True

        # Build messages (same as _execute_llm_interaction)
        messages = self._build_prompt_messages(
            system_prompt, query, context, request_context=request_context
        )

        # Build a scoped progressive tool surface and pin prompt-cache scope.
        if getattr(self, "semantic_tool_router", None) is not None:
            self.semantic_tool_router.begin_turn(user_id=user_id)
        tools = self._build_llm_tools(query, user_id=user_id)
        self._set_provider_cache_scope()

        # Streaming loop with tool calling
        max_iterations = self._get_tool_iteration_limit()
        tool_call_count = 0
        rejection_count = 0
        gate_enabled = self.completion_policy.enabled
        try:
            for iteration in range(max_iterations):
                # Belt-and-braces: messages get appended inside this loop
                # (assistant tool_calls, tool results). Any LOB that slipped
                # into `result` via ``str(result)`` preserves shape but still
                # risks re-entering on subsequent iterations, so coerce once
                # per LLM call.
                messages = _to_jsonable(messages)
                pending_content: List[str] = []
                iteration_had_tool_calls = False
                for event in self._generate_stream_with_trace(
                    messages,
                    tools=tools,
                    iteration=iteration + 1,
                    stage="agent_stream",
                ):
                    event_type = event.get("type")
                    if event_type == "content":
                        content_chunk = event.get("content", "")
                        if gate_enabled:
                            pending_content.append(content_chunk)
                        else:
                            yield content_chunk

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
                        iteration_had_tool_calls = True
                        self._record_context_window_usage(
                            stage=f"stream_iteration_{iteration + 1}"
                        )
                        # Handle tool calls synchronously, then continue streaming
                        response = event["response"]
                        message = response.choices[0].message

                        self._append_assistant_tool_calls(messages, message)
                        for tool_call in message.tool_calls:
                            tool_call_count += 1
                            self._execute_and_record_tool_call(
                                tool_call,
                                messages,
                                workflow,
                                user_id,
                                streaming=True,
                                query=query,
                            )

                        # Continue to next iteration to stream the final response
                        continue

                    elif event_type == "done":
                        self._record_context_window_usage(
                            stage=f"stream_iteration_{iteration + 1}"
                        )
                        if iteration_had_tool_calls:
                            continue
                        # A completion gate must buffer the iteration: exposing
                        # rejected tokens to a client would make enforcement
                        # cosmetic rather than real.
                        if gate_enabled:
                            final_content = "".join(pending_content)
                            decision = self._evaluate_completion_candidate(
                                query=query,
                                response=final_content,
                                iteration=iteration + 1,
                                tool_call_count=tool_call_count,
                            )
                            if not decision.accepted:
                                rejection_count += 1
                                if (
                                    rejection_count
                                    > self.completion_policy.max_rejections
                                ):
                                    raise CompletionRejectedError(
                                        decision, rejection_count
                                    )
                                self._append_completion_rejection(
                                    messages,
                                    final_content,
                                    self.completion_policy.retry_message(decision),
                                )
                                break
                            if final_content:
                                yield final_content
                        # No tool calls, streaming complete for this iteration.
                        _store_workflow_if_needed()
                        return

                # If we got here after tool calls, the inner for-loop finished
                # and we need to loop again for the next streaming round
                continue

            # Exhausted iterations
            _store_workflow_if_needed()
            exhausted = CompletionDecision(
                accepted=False,
                code="iteration_limit",
                reason=(
                    "The tool/completion loop reached its maximum iteration "
                    f"count ({max_iterations}) before an acceptable completion."
                ),
            )
            if gate_enabled:
                raise CompletionRejectedError(exhausted, rejection_count)
            yield (
                "\n\nI reached the maximum number of tool-call iterations "
                f"({max_iterations}). Please try again."
            )
        except ApprovalRequired as approval:
            _store_workflow_if_needed()
            yield self._approval_required_payload(approval.proposal)
            return
        except Exception:
            _store_workflow_if_needed()
            raise

    def _build_context(
        self,
        query: str,
        memory_id: str,
        user_id: Optional[str] = None,
        *,
        include_entity_memory: bool = True,
    ) -> Dict[str, Any]:
        """Build context for the query using memory manager."""
        context = {"query": query}
        self._last_retrieved_memories = []
        self._last_selection_ledger = []
        self._last_retrieval_stats = {
            "duration_ms": 0.0,
            "candidate_count": 0,
            "selected_count": 0,
        }

        # Learned-skill retrieval (continual learning). Reset the turn's
        # attribution first so a retrieval failure — or no match — never
        # inherits the previous run's skill IDs.
        self._activated_skill_ids = []
        if self.skill_retrieval and self.skillbox:
            try:
                if self.continual_learning_manager:
                    scored_skills = (
                        self.continual_learning_manager.retrieve_skills_for_query(
                            query, user_id=user_id
                        )
                    )
                else:
                    scored_skills = self.skillbox.retrieve_skills_by_query(
                        query,
                        limit=max(1, int(self.skill_retrieval_config.get("top_k", 2))),
                        min_similarity=float(
                            self.skill_retrieval_config.get("min_similarity", 0.70)
                        ),
                        user_id=user_id,
                    )
                if scored_skills:
                    context["activated_skills"] = scored_skills
                    self._activated_skill_ids = [
                        scored.skill.skill_id for scored in scored_skills
                    ]
            except Exception as e:
                logger.warning(f"Learned-skill retrieval failed: {e}")

        if self.memory_manager:
            # Load conversation history
            try:
                history_limit = self._get_conversation_history_limit()
                history = self.memory_manager.load_conversation_history(
                    memory_id,
                    limit=history_limit,
                    user_id=user_id,
                    thread_id=self._current_thread_id,
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

            # Pre-inference retrieval across the agent's ACTIVE memory
            # systems. Only sources this agent has enabled are queried (each
            # string query costs one embedding lookup — served from the
            # EmbeddingManager LRU after the first — plus one vector search),
            # and the merged candidate set is deduplicated before injection.
            # The TOOLBOX is deliberately not pre-retrieved: tool schemas
            # already ride along in the request's ``tools`` parameter.
            retrieval_started = time.perf_counter()
            candidates: List[Tuple[str, Any]] = []
            selected: List[Dict[str, Any]] = []
            retrieval_queries = [query]
            try:
                active = set(self.active_memory_types or [])
                candidate_sources = []
                policy = self.retrieval_policy
                if policy.query_expansion:
                    try:
                        from ..retrieval_concepts import expand_query_concepts

                        retrieval_queries.extend(expand_query_concepts(query))
                    except Exception as exc:
                        logger.debug("Query expansion skipped: %s", exc)
                retrieval_queries = retrieval_queries[: policy.max_query_variants]
                if (
                    self.learning_control_plane is not None
                    and self.learning_control_plane_config.retrieval_enabled
                ):
                    history_texts = [
                        str((item.get("content") or {}).get("content") or "")
                        if isinstance(item.get("content"), dict)
                        else str(item.get("content") or "")
                        for item in context.get("conversation_history", [])
                        if isinstance(item, dict)
                    ]
                    try:
                        context[
                            "evidence_pack"
                        ] = self.learning_control_plane.retrieve_evidence(
                            query,
                            memory_id=memory_id,
                            user_id=user_id,
                            thread_id=self._current_thread_id,
                            run_id=self._current_run_id,
                            turn_id=self._current_turn_id,
                            trace_id=self._current_root_trace_id,
                            history_texts=history_texts,
                        )
                    except Exception as exc:
                        logger.warning(
                            "EvidencePack retrieval failed; using legacy retrieval: %s",
                            exc,
                        )
                    else:
                        # EvidencePack owns multi-source retrieval for this
                        # turn; skip the duplicate legacy retrieval path.
                        active = set()
                if (
                    MemoryType.KNOWLEDGE_BASE in active
                    and policy.knowledge_base_scope != "disabled"
                ):
                    if policy.knowledge_base_scope == "namespace":
                        candidate_sources.extend(
                            (
                                MemoryType.KNOWLEDGE_BASE,
                                "knowledge_base",
                                None,
                                namespace,
                            )
                            for namespace in policy.knowledge_base_namespaces
                        )
                    else:
                        candidate_sources.append(
                            (MemoryType.KNOWLEDGE_BASE, "knowledge_base", None, None)
                        )
                if (
                    MemoryType.CONVERSATION_MEMORY in active
                    and policy.conversation_scope != "disabled"
                ):
                    candidate_sources.append(
                        (
                            MemoryType.CONVERSATION_MEMORY,
                            "episodic",
                            self._current_thread_id
                            if policy.conversation_scope == "thread"
                            else None,
                            None,
                        )
                    )

                for (
                    memory_type,
                    source,
                    retrieval_thread_id,
                    namespace,
                ) in candidate_sources:
                    for retrieval_query in retrieval_queries:
                        snippets = self.memory_manager.retrieve_relevant_memories(
                            query=retrieval_query,
                            memory_type=memory_type,
                            memory_id=memory_id,
                            limit=policy.candidate_limit,
                            user_id=user_id,
                            include_embedding=True,
                            thread_id=retrieval_thread_id,
                            namespace=namespace,
                        )
                        for row in snippets or []:
                            candidates.append((source, row))

                if candidates:
                    history_texts = [
                        str((item.get("content") or {}).get("content") or "")
                        if isinstance(item.get("content"), dict)
                        else str(item.get("content") or "")
                        for item in context.get("conversation_history", [])
                        if isinstance(item, dict)
                    ]
                    query_embedding = None
                    try:
                        from ..embeddings import get_embedding

                        # Served from the LRU cache — the retrieval calls
                        # above already embedded this exact query text.
                        query_embedding = get_embedding(query)
                    except Exception:
                        query_embedding = None

                    selected = dedupe_and_select(
                        candidates,
                        history_texts=history_texts,
                        query_embedding=query_embedding,
                        max_items=policy.max_items,
                        dedupe_parent_sources=policy.dedupe_parent_sources,
                        selection_ledger=self._last_selection_ledger,
                    )
                    if selected:
                        context["retrieved_memories"] = selected
            except Exception as e:
                logger.warning(f"Failed to retrieve relevant memories: {e}")
            finally:
                self._last_retrieved_memories = [dict(item) for item in selected]
                self._last_retrieval_stats = {
                    "duration_ms": (time.perf_counter() - retrieval_started) * 1000,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "query_variants": list(retrieval_queries),
                }

        if (
            include_entity_memory
            and self.learning_control_plane is None
            and self._entity_memory_enabled
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            try:
                entity_result = (
                    self.entity_memory_manager.build_context_with_diagnostics(
                        query=query, memory_id=memory_id, user_id=user_id
                    )
                )
                entity_context = entity_result.get("profiles") or []
                if entity_context:
                    context["entity_memory_profiles"] = entity_context
                context["entity_memory_retrieval"] = dict(
                    entity_result.get("retrieval") or {}
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve entity memory context: {e}")

        # Inject existing summary references so the agent knows what can be expanded
        if self.memory_manager:
            try:
                summaries = self.memory_manager.load_summaries_for_thread(
                    memory_id=memory_id,
                    agent_id=self.agent_id,
                    limit=20,
                    user_id=user_id,
                    thread_id=self._current_thread_id,
                )
                if isinstance(summaries, (list, tuple)) and summaries:
                    context["summaries"] = list(summaries)
            except Exception as e:
                logger.debug("Failed to load summary references: %s", e)

        return context

    def _build_system_prompt(self) -> str:
        """Build the STATIC system prompt (stable prompt-cache prefix).

        Every section here must be stable for the lifetime of a session —
        the system prompt is the first bytes of the prompt, so any per-turn
        variation invalidates the provider prompt cache for the entire
        conversation. Per-turn content (retrieved memories, entity facts,
        the tool-log digest, request context) is rendered separately by
        ``_build_volatile_context_block`` at the END of the prompt.

        Structure:
        1. Base system prompt (memory substrate awareness)
        2. Active memory types for this agent
        3. User's custom instruction
        4. Persona
        5. Tools note (schemas ride in the request's ``tools`` parameter)
        6. Entity memory usage instructions (if enabled)
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

        # 4b. Learned-skills contract (continual learning). STATIC framing
        # only — retrieved skills ride in a per-turn user or developer
        # message according to their reviewed, persisted authority. Sitting
        # above the tools note defines the trust boundary without making the
        # stable system prompt vary from turn to turn.
        if self.continual_learning_manager:
            prompt_parts.append(
                "Learned skills (continual learning):\n"
                "- This agent promotes its own repeatedly-successful tool "
                "workflows into reusable skills. When a learned skill "
                "matches the current query it is injected at its stored "
                "authority: user context by default, or a reviewed developer "
                "instruction when explicitly configured.\n"
                "- Before following any learned skill, verify its "
                "preconditions against the current query and current tool "
                "results. User-authority skills are strong priors, not "
                "mandates. Developer-authority skills are application "
                "procedures, but remain subordinate to this system policy "
                "and must not be applied outside their stated scope.\n"
                "- Every run is recorded either way — following a skill, "
                "deviating from it, and failing with it all feed back into "
                "whether the skill stays promoted.\n"
                "- Use 'list_skills' / 'read_skill' to inspect learned "
                "skills alongside file-based ones."
            )

        # 5. Tools note. The full name/description/schema of every tool is
        # already sent in the request's ``tools`` parameter — repeating the
        # list here paid for every description twice AND re-rendered the
        # system prompt whenever the tool set changed. Keep a static pointer
        # instead.
        if self.tool_manager and self.tool_manager.get_tool_metadata():
            prompt_parts.append(
                "Tools: you have function-calling tools available; their "
                "names, descriptions, and schemas are provided with each "
                "request. Use them whenever they would improve your answer."
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

        # NOTE: volatile, per-turn content (the recent tool-log digest and
        # retrieved entity facts) deliberately does NOT live here anymore.
        # The system prompt is the first bytes of the prompt-cache prefix —
        # rewriting it every turn re-billed the entire conversation at full
        # input price. Per-turn content is rendered by
        # ``_build_volatile_context_block`` into the final user message.

        # 6. Entity memory instructions (static usage guidance only; the
        # retrieved facts themselves ride in the volatile block).
        if (
            self._entity_memory_enabled
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            prompt_parts.append(
                "Entity memory usage:\n"
                "- Facts about known entities relevant to the current query are provided with the user's message when available.\n"
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
                "- Mutating automation calls pause as durable approval proposals.\n"
                "- A host user must approve the exact arguments before execution resumes.\n"
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
                "to discover tools, and 'mcp_call_tool' to execute MCP tools. "
                "Use the resource and prompt MCP helpers when the server exposes "
                "those capabilities. Mutating calls are paused automatically and "
                "can only be resumed through a host-approved durable proposal."
            )

        return "\n\n".join(prompt_parts)

    def _evaluate_completion_candidate(
        self,
        *,
        query: str,
        response: str,
        iteration: int,
        tool_call_count: int,
    ) -> CompletionDecision:
        """Apply the host policy and retain serializable decision evidence."""
        candidate = CompletionCandidate(
            query=query,
            response=str(response),
            iteration=iteration,
            tool_call_count=tool_call_count,
            metadata={
                "memory_id": self._current_memory_id,
                "thread_id": self._current_thread_id,
                "user_id": self._current_user_id,
                "run_id": self._current_run_id,
            },
        )
        decision = self.completion_policy.evaluate(candidate)
        decisions = list(self._last_completion_decisions or [])
        decisions.append(decision.to_dict())
        self._last_completion_decisions = decisions
        self._emit_stream_event(
            "completion_check",
            {
                **decision.to_dict(),
                "agent_id": self.agent_id,
                "validator_name": self.completion_policy.validator_name,
            },
        )
        return decision

    def completion_policy_report(self) -> Dict[str, Any]:
        """Return the current policy and evidence from the most recent turn."""
        decisions = list(self._last_completion_decisions or [])
        return {
            "policy": self.completion_policy.to_dict(),
            "decision_count": len(decisions),
            "rejection_count": sum(
                1 for decision in decisions if not decision.get("accepted", False)
            ),
            "accepted": bool(decisions and decisions[-1].get("accepted", False)),
            "decisions": decisions,
        }

    @staticmethod
    def _append_completion_rejection(
        messages: List[Dict[str, Any]],
        response: str,
        retry_message: str,
    ) -> None:
        """Preserve the proposed answer and reject it without rebuilding state."""
        messages.append({"role": "assistant", "content": str(response)})
        messages.append({"role": "developer", "content": retry_message})

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
            workflow = self._init_workflow_capture(query, user_id)

            # Build initial messages
            messages = self._build_prompt_messages(
                system_prompt, query, context, request_context=request_context
            )

            # Build a scoped progressive tool surface and pin prompt-cache scope.
            if getattr(self, "semantic_tool_router", None) is not None:
                self.semantic_tool_router.begin_turn(user_id=user_id)
            tools = self._build_llm_tools(query, user_id=user_id)
            self._set_provider_cache_scope()

            # Execute main loop with tool calling
            max_iterations = self._get_tool_iteration_limit()
            tool_call_count = 0
            rejection_count = 0
            for iteration in range(max_iterations):
                # Coerce LOBs to strings before every LLM call; matches the
                # streaming path's guarantee.
                messages = _to_jsonable(messages)
                # Call LLM
                response = self._generate_with_trace(
                    messages,
                    tools=tools,
                    iteration=iteration + 1,
                    stage="agent_run",
                )
                self._record_context_window_usage(stage=f"iteration_{iteration + 1}")

                # Check if response is a string (no tool calls)
                if isinstance(response, str):
                    final_content = response
                    decision = self._evaluate_completion_candidate(
                        query=query,
                        response=final_content,
                        iteration=iteration + 1,
                        tool_call_count=tool_call_count,
                    )
                    if decision.accepted:
                        self._persist_workflow_run(workflow)
                        return final_content
                    rejection_count += 1
                    if rejection_count > self.completion_policy.max_rejections:
                        raise CompletionRejectedError(decision, rejection_count)
                    self._append_completion_rejection(
                        messages,
                        final_content,
                        self.completion_policy.retry_message(decision),
                    )
                    continue

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
                        decision = self._evaluate_completion_candidate(
                            query=query,
                            response=final_content,
                            iteration=iteration + 1,
                            tool_call_count=tool_call_count,
                        )
                        if decision.accepted:
                            self._persist_workflow_run(workflow)
                            return final_content
                        rejection_count += 1
                        if rejection_count > self.completion_policy.max_rejections:
                            raise CompletionRejectedError(decision, rejection_count)
                        self._append_completion_rejection(
                            messages,
                            final_content,
                            self.completion_policy.retry_message(decision),
                        )
                        continue

                    self._append_assistant_tool_calls(messages, message)
                    for tool_call in message.tool_calls:
                        tool_call_count += 1
                        self._execute_and_record_tool_call(
                            tool_call,
                            messages,
                            workflow,
                            user_id,
                            streaming=False,
                            query=query,
                        )

                    # Continue loop to get final response
                    continue

                # Fallback: return any content we got
                fallback_response = "I encountered an unexpected response format."
                if session_for(self) is not None:
                    from ..llms.streaming import ProviderStreamError

                    raise ProviderStreamError("unexpected_response_format")
                self._persist_workflow_run(workflow)
                return fallback_response

            # If we exhausted iterations
            if session_for(self) is not None:
                from ..llms.streaming import ProviderStreamError

                raise ProviderStreamError("iteration_limit")
            final_response = (
                "I reached the maximum number of tool-call iterations "
                f"({max_iterations}). Please try again."
            )

            # Store workflow if it was created and has steps
            self._persist_workflow_run(workflow)

            return final_response

        except ApprovalRequired as approval:
            if "workflow" in locals():
                self._persist_workflow_run(workflow)
            session = session_for(self)
            if session is not None:
                session.outcome = "approval_required"
                session.emit(
                    "approval.required",
                    proposal=approval.proposal.to_dict(include_arguments=False),
                )
                return ""
            return self._approval_required_payload(approval.proposal)
        except CompletionRejectedError:
            if "workflow" in locals():
                self._persist_workflow_run(workflow)
            raise
        except Exception as e:
            logger.error(f"LLM interaction failed: {e}")
            if session_for(self) is not None:
                raise

            # Store workflow even on error if it exists
            if "workflow" in locals():
                self._persist_workflow_run(workflow)

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
            result = self.entity_memory_manager.lookup_entities_with_diagnostics(
                entity_id=entity_id,
                name=name,
                query=query,
                limit=limit,
                memory_id=resolved_memory_id,
                user_id=self._current_user_id,
            )
            logger.info(
                "entity_memory_lookup returned %s record(s) "
                "(memory_id=%s, mode=%s, degraded=%s, reason=%s)",
                len(result.get("matches") or []),
                resolved_memory_id,
                (result.get("retrieval") or {}).get("retrieval_mode"),
                (result.get("retrieval") or {}).get("degraded"),
                (result.get("retrieval") or {}).get("degraded_reason"),
            )
            return result

        def _normalize_attributes(attrs):
            """Normalize attributes from various formats to list of dicts with name/value."""
            if not attrs:
                return None
            parsed = attrs
            if isinstance(attrs, str):
                if not attrs.strip():
                    return None
                try:
                    parsed = json.loads(attrs)
                except (json.JSONDecodeError, TypeError):
                    return None
            if isinstance(parsed, dict):
                aliases = {"name", "attribute", "attribute_name", "key"}
                if "value" in parsed and aliases.intersection(parsed):
                    parsed = [parsed]
                else:
                    parsed = [
                        {"name": key, "value": str(value)}
                        for key, value in parsed.items()
                    ]
            if not isinstance(parsed, list):
                return None

            normalized = []
            for index, raw in enumerate(parsed):
                if isinstance(raw, EntityAttributeInput):
                    item = raw
                elif isinstance(raw, dict):
                    payload = dict(raw)
                    aliases = {"name", "attribute", "attribute_name", "key"}
                    if "value" not in payload and not aliases.intersection(payload):
                        reserved = {
                            "confidence",
                            "source",
                            "created_at",
                            "updated_at",
                            "metadata",
                        }
                        facts = [
                            (key, value)
                            for key, value in payload.items()
                            if key not in reserved
                        ]
                        if facts:
                            shared = {
                                key: value
                                for key, value in payload.items()
                                if key in reserved
                            }
                            for key, value in facts:
                                fact_payload = {
                                    **shared,
                                    "name": key,
                                    "value": str(value),
                                }
                                fact = EntityAttributeInput.model_validate(fact_payload)
                                normalized.append(fact.model_dump(exclude_none=True))
                            continue
                    if "value" in payload and not isinstance(payload["value"], str):
                        payload["value"] = str(payload["value"])
                    item = EntityAttributeInput.model_validate(payload)
                else:
                    raise ValueError(
                        f"Entity attribute at index {index} must be an object."
                    )
                normalized.append(item.model_dump(exclude_none=True))
            return normalized

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

        def _normalize_relations(relations):
            parsed = _normalize_json_field(relations, list)
            if parsed is None:
                return None
            normalized = []
            for index, raw in enumerate(parsed):
                if isinstance(raw, EntityRelationInput):
                    item = raw
                elif isinstance(raw, dict):
                    item = EntityRelationInput.model_validate(raw)
                else:
                    raise ValueError(
                        f"Entity relation at index {index} must be an object."
                    )
                normalized.append(item.model_dump(exclude_none=True))
            return normalized

        def entity_memory_upsert(
            entity_id: str = None,
            name: str = None,
            entity_type: str = None,
            attributes: Optional[List[EntityAttributeInput]] = None,
            relations: Optional[List[EntityRelationInput]] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Insert or update a structured entity in the active run scope."""
            # Scope is host-owned and deliberately absent from the model-visible
            # schema. The model supplies entity facts, never tenant identifiers.
            resolved_memory_id = self._current_memory_id or (
                self.memory_ids[0] if self.memory_ids else None
            )
            if resolved_memory_id is None:
                raise ValueError("A memory_id is required to store entity information.")

            new_entity_id = self.entity_memory_manager.upsert_entity_from_tool(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=_normalize_attributes(attributes),
                relations=_normalize_relations(relations),
                metadata=_normalize_json_field(metadata, dict),
                memory_id=resolved_memory_id,
                user_id=self._current_user_id,
            )
            logger.info(
                "entity_memory_upsert stored entity_id=%s (memory_id=%s)",
                new_entity_id,
                resolved_memory_id,
            )
            return {
                "entity_id": new_entity_id,
                "storage": {
                    "memory_id_bound": resolved_memory_id is not None,
                    "user_id_bound": self._current_user_id is not None,
                    "scope_override_rejected": False,
                },
            }

        entity_memory_lookup.__name__ = "entity_memory_lookup"
        entity_memory_upsert.__name__ = "entity_memory_upsert"

        self.tool_manager.add_tool(entity_memory_lookup)
        self.tool_manager.add_tool(entity_memory_upsert)
        self._entity_memory_tools_registered = True

    def _register_semantic_layer_tools(self) -> None:
        """Expose governed discovery and planning for an attached catalog."""
        catalog = self.semantic_layer
        if catalog is None or not self.tool_manager:
            return
        required = ("list_models", "describe", "plan")
        if any(not callable(getattr(catalog, name, None)) for name in required):
            raise TypeError(
                "semantic_layer must implement list_models(), describe(), and plan()"
            )

        @governed_tool(deterministic=True, side_effects=False, domains=("semantic",))
        def semantic_list_models() -> Dict[str, Any]:
            """List governed semantic model versions available to this agent."""
            return {"models": catalog.list_models()}

        @governed_tool(deterministic=True, side_effects=False, domains=("semantic",))
        def semantic_describe_model(
            model_name: str, version: Optional[str] = None
        ) -> Dict[str, Any]:
            """Describe entities, measures, dimensions, joins, and policies."""
            return catalog.describe(model_name, version)

        @governed_tool(deterministic=True, side_effects=False, domains=("semantic",))
        def semantic_plan_query(
            model_name: str,
            measures: Optional[List[str]] = None,
            dimensions: Optional[List[str]] = None,
            filters: Optional[List[Dict[str, Any]]] = None,
            order_by: Optional[List[str]] = None,
            limit: int = 100,
            version: Optional[str] = None,
            roles: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Create a policy-checked, parameterized semantic query plan.

            This validates semantic names, relationships, role policies,
            filters, ordering, and limits. It returns a plan; it never executes
            arbitrary SQL.
            """
            plan = catalog.plan(
                model_name,
                {
                    "measures": measures or [],
                    "dimensions": dimensions or [],
                    "filters": filters or [],
                    "order_by": order_by or [],
                    "limit": limit,
                    "roles": roles or [],
                },
                version=version,
            )
            return plan.model_dump(mode="json")

        self.tool_manager.add_tool(semantic_list_models)
        self.tool_manager.add_tool(semantic_describe_model)
        self.tool_manager.add_tool(semantic_plan_query)

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

        def internet_search(query: str, max_results: int = 5) -> Any:
            """Search the public internet for up-to-date information."""
            try:
                if self._internet_access_disabled_reason:
                    message = f"Internet access disabled: {self._internet_access_disabled_reason}"
                    return self._internet_error_result(
                        {
                            "ok": False,
                            "error_code": "provider_unavailable",
                            "error": message,
                        },
                        reason_code="provider_unavailable",
                        retryable=True,
                    )
                results = self.internet_access_manager.search(
                    query=query, max_results=max_results
                )
                self._internet_access_failure_count = 0
                provider_name = self.get_internet_access_provider_name()
                if provider_name == "offline":
                    return self._internet_error_result(
                        {"results": results},
                        reason_code="provider_unavailable",
                        retryable=False,
                        result_count=0,
                    )
                return {"results": results}
            except Exception as exc:
                logger.error("internet_search failed: %s", exc)
                return self._handle_internet_access_error(str(exc))

        def open_web_page(url: str) -> Any:
            """Fetch and summarize the contents of a website."""
            try:
                if self._internet_access_disabled_reason:
                    message = f"Internet access disabled: {self._internet_access_disabled_reason}"
                    return self._internet_error_result(
                        {
                            "ok": False,
                            "error_code": "provider_unavailable",
                            "error": message,
                        },
                        reason_code="provider_unavailable",
                        retryable=True,
                    )
                page = self.internet_access_manager.fetch_url(url=url)
                self._internet_access_failure_count = 0
                provider_name = self.get_internet_access_provider_name()
                if provider_name == "offline":
                    return self._internet_error_result(
                        page,
                        reason_code="provider_unavailable",
                        retryable=False,
                        result_count=0,
                    )
                return page
            except Exception as exc:
                logger.error("open_web_page failed: %s", exc)
                return self._handle_internet_access_error(str(exc))

        internet_search.__name__ = "internet_search"
        open_web_page.__name__ = "open_web_page"

        self.tool_manager.add_tool(internet_search)
        self.tool_manager.add_tool(open_web_page)
        self._internet_access_tools_registered = True

    def _internet_error_result(
        self,
        value: Any,
        *,
        reason_code: str,
        retryable: bool,
        result_count: Optional[int] = None,
    ) -> ToolResult:
        """Attach provider-error evidence without changing model-visible data."""
        return ToolResult(
            value=value,
            outcome=ToolOutcome(
                status=ToolOutcomeStatus.PROVIDER_ERROR,
                reason_code=reason_code,
                provider=self.get_internet_access_provider_name(),
                retryable=retryable,
                result_count=result_count,
            ),
        )

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
            "ok": False,
            "error_code": "provider_error",
            "provider": self.get_internet_access_provider_name(),
            "retryable": not bool(self._internet_access_disabled_reason),
            "error": message
            if not self._internet_access_disabled_reason
            else f"{message} | {self._internet_access_disabled_reason}",
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
        session = session_for(self)
        if not self.memory_manager:
            if session:
                session.persistence["conversation"] = "not_configured"
            return

        try:
            # Write-path dedup: a semantic-cache hit or client retry replays
            # the exact same (query, response) pair back-to-back. Storing it
            # again just bloats the history window with rows the dedup pass
            # then has to filter out — skip the double-write entirely.
            try:
                if self.memory_manager.is_duplicate_of_recent(
                    memory_id,
                    Role.USER.value,
                    query,
                    user_id=user_id,
                    thread_id=thread_id,
                ) and self.memory_manager.is_duplicate_of_recent(
                    memory_id,
                    Role.ASSISTANT.value,
                    response,
                    user_id=user_id,
                    thread_id=thread_id,
                ):
                    logger.debug(
                        "Skipping duplicate interaction write for memory %s",
                        memory_id,
                    )
                    if session:
                        session.persistence["conversation"] = "deduplicated"
                    return
            except Exception:
                pass

            # Record user query
            user_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.USER,
                content=query,
                thread_id=thread_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
            )
            user_unit_id = self.memory_manager.save_memory_unit(user_memory, memory_id)

            # Record assistant response
            assistant_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.ASSISTANT,
                content=response,
                thread_id=thread_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
            )
            assistant_unit_id = self.memory_manager.save_memory_unit(
                assistant_memory, memory_id
            )
            if session:
                session.persistence["conversation"] = (
                    "written" if user_unit_id and assistant_unit_id else "unknown"
                )

            # Backfill embeddings off the hot path so episodic semantic
            # recall (vector search over conversation rows) has vectors to
            # match — rows are created with embedding=None so the user-facing
            # turn never blocks on the embedding API.
            self._schedule_conversation_embedding_backfill(
                [
                    (user_unit_id, query),
                    (assistant_unit_id, response),
                ]
            )

            logger.debug(f"Recorded interaction in memory: {memory_id}")

        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")
            if session:
                session.persistence["conversation"] = "failed"

    def _schedule_conversation_embedding_backfill(
        self, unit_texts: List[Tuple[Optional[str], str]]
    ) -> None:
        """Embed stored conversation rows in the background.

        Uses a single-worker executor so backfills never compete with the
        interactive path and never pile up threads. Failures degrade
        silently — a row without an embedding simply won't participate in
        episodic vector recall (the pre-fix status quo for every row).
        """
        if not self._conversation_embedding_enabled or not self.memory_provider:
            return
        pairs = [
            (unit_id, text)
            for unit_id, text in unit_texts
            if unit_id and text and str(text).strip()
        ]
        if not pairs:
            return

        if self._embedding_backfill_executor is None:
            self._embedding_backfill_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="memorizz-embed-backfill"
            )

        provider = self.memory_provider

        def _backfill() -> None:
            from ..embeddings import get_embedding

            for unit_id, text in pairs:
                try:
                    embedding = get_embedding(str(text))
                    if embedding:
                        provider.update_by_id(
                            unit_id,
                            {"embedding": embedding},
                            MemoryType.CONVERSATION_MEMORY,
                        )
                except Exception as exc:
                    logger.debug(
                        "Conversation embedding backfill failed for %s: %s",
                        unit_id,
                        exc,
                    )

        try:
            self._embedding_backfill_executor.submit(_backfill)
        except RuntimeError:
            # Executor shut down (interpreter teardown) — skip quietly.
            pass

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

    def resume_thread(self, memory_id: str, thread_id: str) -> str:
        """Make an existing memory/thread pair the active conversation.

        The next :meth:`run` / :meth:`run_stream` call may still pass the ids
        explicitly; this method updates the agent immediately for interactive
        clients such as the CLI conversation picker.
        """
        resolved_memory_id = str(memory_id or "").strip()
        resolved_thread_id = str(thread_id or "").strip()
        if not resolved_memory_id or not resolved_thread_id:
            raise ValueError("memory_id and thread_id are required")
        self._resolve_execution_state(resolved_memory_id, resolved_thread_id)
        logger.info(
            "Resumed thread: %s (memory=%s)",
            resolved_thread_id,
            resolved_memory_id,
        )
        return resolved_thread_id

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

    @staticmethod
    def _fingerprint(value: Any) -> str:
        payload = json.dumps(
            _to_jsonable(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _semantic_cache_preflight_bypass(
        self, query: str, *, user_id: Optional[str] = None
    ) -> Optional[str]:
        """Fail safe when routing predicts a side-effecting/nondeterministic tool.

        Cache admission after execution already rejects these turns. This
        preflight closes the other boundary: a semantically similar cached
        read must not prevent a newly requested mutation from reaching the
        model and its durable approval gate.
        """
        # A cached string cannot prove that a tool was executed in this turn.
        # Policies requiring fresh tool evidence therefore bypass lookup.
        if self.completion_policy.enabled and self.completion_policy.require_tool_calls:
            return "completion_policy_requires_fresh_tool_evidence"

        router = getattr(self, "semantic_tool_router", None)
        manager = getattr(self, "tool_manager", None)
        if router is None or manager is None:
            return None
        try:
            candidates = router.preview(query, user_id=user_id)
        except Exception as exc:
            logger.debug("Semantic-cache routing preflight failed: %s", exc)
            return None
        for tool_name in candidates:
            policy = dict(manager.get_tool_policy(tool_name) or {})
            if policy.get("side_effects") or policy.get("requires_approval"):
                return "side_effecting_tool_candidate"
            if policy.get("deterministic") is False:
                return "nondeterministic_tool_candidate"
        return None

    def _semantic_cache_metadata(
        self, request_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = dict(request_context or {})
        tool_metadata = (
            self.tool_manager.get_tool_metadata() if self.tool_manager else []
        )
        model_value = {
            "provider": self.llm_provider,
            "model": self.llm_model or getattr(self.model, "model", None),
            "config": self.llm_config,
        }
        prompt_value = {
            "instruction": self.instruction,
            "application_mode": getattr(
                self.application_mode, "value", self.application_mode
            ),
            "persona": str(
                getattr(self.persona_manager, "current_persona", None)
                if self.persona_manager
                else None
            ),
        }
        data_version = context.get("data_version") or context.get(
            "cache_data_version", "default"
        )
        domains = {
            str(item)
            for item in (
                context.get("cache_domains")
                or (
                    [context.get("cache_domain")] if context.get("cache_domain") else []
                )
            )
            if item is not None
        }
        domains.update(self._turn_cache_domains)
        tags = [str(item) for item in (context.get("cache_tags") or [])]
        # Include all per-request context in cache admission/lookup identity.
        # Session scoping prevents cross-thread reuse; this fingerprint also
        # prevents stale reuse within one thread when page, selection, or other
        # ephemeral grounding changes between otherwise identical questions.
        request_context_fingerprint = self._fingerprint(context)
        return {
            "fingerprints": {
                "model": self._fingerprint(model_value),
                "prompt": self._fingerprint(prompt_value),
                "tool_schema": self._fingerprint(tool_metadata),
                "completion_policy": self._fingerprint(
                    self.completion_policy.to_dict()
                ),
                "data_version": str(data_version),
                "request_context": request_context_fingerprint,
            },
            "domain": sorted(domains)[0] if len(domains) == 1 else None,
            "domains": sorted(domains),
            "tags": sorted(set(tags)),
        }

    def semantic_cache_stats(self) -> Dict[str, Any]:
        """Return real hit/miss/bypass/write/eviction counters and provenance."""
        return (
            self.cache_manager.get_statistics()
            if self.cache_manager
            else {
                "enabled": False,
                "hits": 0,
                "misses": 0,
                "bypasses": 0,
                "writes": 0,
                "evictions": 0,
                "size": 0,
            }
        )

    def inspect_semantic_cache(
        self,
        query: str,
        *,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        bypass_reason: Optional[str] = None,
    ) -> "SemanticCacheInspection":
        """Inspect cache provenance/freshness without returning cached content."""
        return self.cache_manager.inspect(
            query,
            session_id=thread_id,
            user_id=user_id,
            metadata=(
                dict(metadata)
                if metadata is not None
                else self._semantic_cache_metadata(context)
            ),
            bypass_reason=bypass_reason,
        )

    def invalidate_semantic_cache(
        self,
        *,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        data_version: Optional[str] = None,
    ) -> int:
        """Invalidate semantic cache entries by operational domain or tag."""
        if not self.cache_manager:
            return 0
        return self.cache_manager.invalidate(
            domains=domains, tags=tags, data_version=data_version
        )

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
    def load_conversation_history(
        self,
        memory_id: str = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """Load conversation history (delegated to memory manager)."""
        if self.memory_manager:
            memory_id = (
                memory_id
                or self._current_memory_id
                or (self.memory_ids[0] if self.memory_ids else None)
            )
            if memory_id:
                return self.memory_manager.load_conversation_history(
                    memory_id,
                    user_id=user_id,
                    thread_id=thread_id,
                )
        return []

    def get_context_window_stats(self) -> Optional[Dict[str, Any]]:
        """Return the most recent context window usage snapshot."""
        if not self._last_context_window_stats:
            return None
        return dict(self._last_context_window_stats)

    def last_retrieval_evidence(self) -> Dict[str, Any]:
        """Return source-safe evidence and timing from the most recent turn."""

        return {
            "items": [dict(item) for item in self._last_retrieved_memories],
            **dict(self._last_retrieval_stats),
        }

    def last_memory_context_evidence(self) -> Dict[str, Any]:
        """Return the content-free memory supply snapshot for the latest turn."""
        return dict(self._last_memory_context_evidence or {})

    def build_personalization_context(
        self,
        query: str,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        writing_samples: Optional[List[Dict[str, Any]]] = None,
        additional_memories: Optional[List[Dict[str, Any]]] = None,
        policy: Optional[Any] = None,
        exclude_thread_id: Optional[str] = None,
    ):
        """Build explicit, tenant-scoped personalization without running an LLM.

        Cross-thread conversation recall occurs only when the supplied
        :class:`~memorizz.personalization.PersonalizationPolicy` enables it and
        both ``memory_id`` and ``user_id`` are bound.  Host preferences and
        writing samples are accepted as already-authorized application data.
        """
        from ..personalization import (
            PersonalizationContextBuilder,
            PersonalizationPolicy,
        )

        resolved_policy = PersonalizationPolicy.from_value(policy)
        resolved_memory_id = str(
            memory_id
            or self._current_memory_id
            or (self.memory_ids[0] if self.memory_ids else "")
        ).strip()
        diagnostics: Dict[str, Any] = {
            "retrieval_mode": "explicit",
            "candidate_count": 0,
            "selected_count": 0,
            "entity_candidate_count": 0,
            "entity_selected_count": 0,
            "conversation_candidate_count": 0,
            "conversation_selected_count": 0,
            "fallback_used": False,
            "degraded": False,
            "degraded_reason": None,
        }

        entity_profiles: List[Dict[str, Any]] = []
        if (
            resolved_policy.include_entity_memory
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
            and resolved_memory_id
            and resolved_policy.max_entity_profiles > 0
        ):
            try:
                entity_result = (
                    self.entity_memory_manager.build_context_with_diagnostics(
                        query=(
                            f"{query} current user profile role preferences "
                            "audience goals"
                        ),
                        memory_id=resolved_memory_id,
                        limit=resolved_policy.max_entity_profiles,
                        user_id=user_id,
                    )
                )
                entity_profiles = list(entity_result.get("profiles") or [])
                entity_diagnostics = entity_result.get("retrieval") or {}
                entity_candidate_count = int(
                    (
                        entity_diagnostics.get("fallback_candidate_count")
                        if entity_diagnostics.get("fallback_used")
                        else entity_diagnostics.get("semantic_match_count")
                    )
                    or 0
                )
                diagnostics["entity_candidate_count"] = entity_candidate_count
                diagnostics["entity_selected_count"] = len(entity_profiles)
                diagnostics["candidate_count"] += entity_candidate_count
                diagnostics["fallback_used"] = bool(
                    entity_diagnostics.get("fallback_used")
                )
                diagnostics["degraded"] = bool(entity_diagnostics.get("degraded"))
                if entity_diagnostics.get("degraded_reason"):
                    diagnostics["degraded_reason"] = entity_diagnostics.get(
                        "degraded_reason"
                    )
            except Exception as exc:
                diagnostics.update(
                    {
                        "degraded": True,
                        "degraded_reason": "entity_retrieval_failed",
                        "error_type": type(exc).__name__,
                    }
                )

        selected_memories: List[Dict[str, Any]] = []
        if (
            resolved_policy.conversation_recall
            and resolved_policy.max_conversation_memories > 0
            and resolved_policy.conversation_candidate_limit > 0
        ):
            if not resolved_memory_id or not str(user_id or "").strip():
                diagnostics.update(
                    {
                        "degraded": True,
                        "degraded_reason": "unbound_tenant_scope",
                    }
                )
            elif self.memory_manager:
                try:
                    candidates = self.memory_manager.retrieve_relevant_memories(
                        query=query,
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        memory_id=resolved_memory_id,
                        limit=resolved_policy.conversation_candidate_limit,
                        user_id=user_id,
                        include_embedding=True,
                    )
                    if exclude_thread_id:
                        wanted_exclusion = str(exclude_thread_id)

                        def _thread(row: Dict[str, Any]) -> str:
                            content = row.get("content")
                            nested = content if isinstance(content, dict) else {}
                            return str(
                                row.get("thread_id")
                                or row.get("conversation_id")
                                or nested.get("thread_id")
                                or nested.get("conversation_id")
                                or ""
                            )

                        candidates = [
                            row
                            for row in candidates
                            if _thread(row) != wanted_exclusion
                        ]
                    diagnostics["conversation_candidate_count"] = len(candidates)
                    diagnostics["candidate_count"] += len(candidates)
                    thresholded = []
                    for row in candidates:
                        try:
                            score = float(row.get("score") or 0.0)
                        except (TypeError, ValueError):
                            score = 0.0
                        if score >= resolved_policy.min_relevance_score:
                            thresholded.append(("conversation", row))
                    selected_memories = dedupe_and_select(
                        thresholded,
                        max_items=resolved_policy.max_conversation_memories,
                        dedupe_parent_sources=True,
                    )
                    diagnostics["conversation_selected_count"] = len(selected_memories)
                except Exception as exc:
                    diagnostics.update(
                        {
                            "degraded": True,
                            "degraded_reason": "conversation_retrieval_failed",
                            "error_type": type(exc).__name__,
                        }
                    )

        for item in additional_memories or []:
            if isinstance(item, dict) and self._memory_row_text(item):
                selected_memories.append(dict(item))
        selected_memories = selected_memories[
            : resolved_policy.max_conversation_memories
        ]
        diagnostics["conversation_selected_count"] = len(selected_memories)
        diagnostics["selected_count"] = len(entity_profiles) + len(selected_memories)

        return PersonalizationContextBuilder(resolved_policy).build(
            entity_profiles=entity_profiles,
            preferences=preferences,
            conversation_memories=selected_memories,
            writing_samples=writing_samples,
            diagnostics=diagnostics,
        )

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
        return persistence.download_memory(self, memagent)

    def update_memory(self, memory_ids: List[str]) -> bool:
        """
        Add memory_ids to this agent, giving it access to additional conversation history.

        Args:
            memory_ids: List of memory_ids to add.

        Returns:
            True if successful, False otherwise.
        """
        return persistence.update_memory(self, memory_ids)

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
        return persistence.save_agent(self)

    def close(
        self,
        *,
        cleanup_scope: Optional[Dict[str, Any]] = None,
        close_memory_provider: bool = True,
        close_model_provider: bool = False,
    ) -> Dict[str, Any]:
        """Close agent-owned runtime resources and optionally delete a scope.

        ``cleanup_scope`` is forwarded to a provider's ``delete_scope`` before
        connections are closed. It must contain an exact ``memory_id``,
        ``user_id``, and/or ``agent_ids`` boundary; providers retain their own
        fail-closed validation.
        """
        report: Dict[str, Any] = {
            "ok": True,
            "cleanup": None,
            "closed": {},
            "errors": [],
        }
        provider = self.memory_provider
        if cleanup_scope is not None:
            if not isinstance(cleanup_scope, dict):
                raise TypeError("cleanup_scope must be a dictionary or None")
            cleanup = getattr(provider, "delete_scope", None)
            if not callable(cleanup):
                report["ok"] = False
                report["errors"].append(
                    "The configured memory provider does not support delete_scope"
                )
            else:
                try:
                    report["cleanup"] = cleanup(**dict(cleanup_scope))
                except Exception as exc:
                    report["ok"] = False
                    report["errors"].append(f"scoped cleanup failed: {exc}")

        closed_objects: Set[int] = set()

        def _close(label: str, value: Any) -> None:
            if value is None or id(value) in closed_objects:
                return
            closer = getattr(value, "close", None)
            if not callable(closer):
                return
            try:
                closer()
                closed_objects.add(id(value))
                report["closed"][label] = True
            except Exception as exc:
                report["ok"] = False
                report["closed"][label] = False
                report["errors"].append(f"{label} close failed: {exc}")

        _close("sandbox", getattr(self, "sandbox_manager", None))
        _close("browser_control", getattr(self, "browser_control_manager", None))
        if getattr(self, "_owns_meta_harness", False):
            _close("meta_harness", getattr(self, "meta_harness", None))
        internet_manager = getattr(self, "internet_access_manager", None)
        _close("internet_access", getattr(internet_manager, "provider", None))
        _close("approval_store", self.approval_store)
        _close("learning_control_plane", self.learning_control_plane)
        if close_model_provider:
            _close("model_provider", self.model)
        if close_memory_provider:
            _close("memory_provider", provider)

        self._last_close_report = report
        return report

    @contextmanager
    def lifecycle(
        self,
        *,
        cleanup_scope: Optional[Dict[str, Any]] = None,
        close_memory_provider: bool = True,
        close_model_provider: bool = False,
    ):
        """Manage the agent lifecycle with optional exact-scope cleanup."""
        try:
            yield self
        finally:
            self.close(
                cleanup_scope=cleanup_scope,
                close_memory_provider=close_memory_provider,
                close_model_provider=close_model_provider,
            )

    def __enter__(self) -> "MemAgent":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

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
        return persistence.load_agent(cls, agent_id, memory_provider, **overrides)

    def refresh(self):
        """
        Refresh the MemAgent from the memory provider.

        This method reloads the agent configuration from the memory provider,
        updating the current instance with any changes.

        Returns:
            MemAgent: Self if successful, False if failed
        """
        return persistence.refresh_agent(self)

    def generate_summaries(
        self,
        days_back: int = 7,
        max_memories_per_summary: int = 50,
        *,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
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
        memory_id : str, optional
            Exact memory scope. When omitted, the configured/current memory
            IDs are considered for backward compatibility.
        user_id : str, optional
            Exact tenant scope. ``None`` selects only anonymous/legacy rows;
            this method never inherits a user from the most recent run.
        thread_id : str, optional
            Exact conversation thread. When omitted, all threads in the
            selected memory/tenant scope may be compacted independently.

        Returns:
        --------
        List[str]
            List of summary IDs that were created
        """
        manager = self.memory_manager or MemoryManager(self.memory_provider)
        selected_memory_ids = [memory_id] if memory_id else self.memory_ids
        current_memory_id = memory_id or self._current_memory_id
        return manager.generate_summaries(
            model=self.model,
            agent_id=self.agent_id,
            memory_ids=selected_memory_ids,
            current_memory_id=current_memory_id,
            user_id=user_id,
            thread_id=thread_id,
            days_back=days_back,
            max_memories_per_summary=max_memories_per_summary,
            record_context_usage=self._record_context_window_usage,
        )

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
        manager = self.memory_manager or MemoryManager(self.memory_provider)
        return manager.compress_memories_with_llm(
            memories,
            model=self.model,
            record_context_usage=self._record_context_window_usage,
        )
