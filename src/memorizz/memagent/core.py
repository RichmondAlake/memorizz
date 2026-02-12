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

from ..enums import ApplicationMode, ApplicationModeConfig, MemoryType, Role
from ..internet_access import get_default_internet_access_provider
from ..llms.llm_factory import create_llm_provider
from .constants import (
    BASE_SYSTEM_PROMPT,
    DEFAULT_INSTRUCTION,
    DEFAULT_MAX_STEPS,
    DEFAULT_TOOL_ACCESS,
    SANDBOX_SYSTEM_PROMPT,
)
from .managers import (
    CacheManager,
    EntityMemoryManager,
    InternetAccessManager,
    MemoryManager,
    PersonaManager,
    SandboxManager,
    ToolManager,
    WorkflowManager,
)

if TYPE_CHECKING:
    from ..internet_access import InternetAccessProvider
    from ..sandbox.base import SandboxProvider

logger = logging.getLogger(__name__)


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
        self.tools = tools if tools is not None else []
        self.skill_paths = self._normalize_skill_paths(skill_paths)
        self.skills: List[Dict[str, Any]] = []
        self.mcp_servers = self._normalize_mcp_servers(mcp_servers)

        (
            self.application_mode,
            self._application_mode_explicit,
        ) = self._resolve_application_mode(application_mode)

        # Store memory types for tracking which memory systems are active
        self.active_memory_types = self._initialize_memory_types(
            self.application_mode if self._application_mode_explicit else None,
            memory_types,
        )

        # Initialize conversation state tracking
        self._current_conversation_id = None
        self._current_memory_id = None
        self._conversation_ids_by_memory: Dict[str, str] = {}
        self._last_entity_context: List[Dict[str, Any]] = []

        # Initialize LLM
        self.model = model
        if not model and llm_config:
            try:
                self.model = create_llm_provider(llm_config)
            except Exception as e:
                logger.warning(f"Could not create LLM from config: {e}")

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
        self._skills_marketplace_tools_registered = False
        self._skills_marketplace_tool_names = ("skills_marketplace_search",)
        self._skills_marketplace_provider_name: Optional[str] = None
        self._skills_marketplace_config: Dict[str, Any] = {}
        self._context_tools_registered = False
        self._summary_registry: List[Dict[str, Any]] = []
        self._known_summary_ids: Set[str] = set()
        self._context_summary_trigger = 85.0
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
        )
        self._register_context_monitor_tools()

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
            self.with_internet_access_provider(provider_to_attach)

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
            self.with_skills_marketplace_provider(
                skills_provider_to_attach,
                config=skills_provider_config,
            )

        # Enable entity memory automatically if configured via application mode
        if MemoryType.ENTITY_MEMORY in self.active_memory_types:
            self.with_entity_memory(True)

        # Initialize tools if provided
        if tools:
            self._initialize_tools(tools)

        # Initialize persona if provided
        if persona:
            self.persona_manager.set_persona(persona, self.agent_id, save=False)

        self.skills = self._load_skills()
        if self.skills:
            self._register_skill_tools()

        if self.mcp_servers:
            self._register_mcp_tools()

        logger.info(
            f"MemAgent {self.agent_id} initialized with memory types: {self.active_memory_types}"
        )

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
            self.sandbox_manager = SandboxManager.from_config(sandbox_provider)
            self._register_sandbox_tools()
        else:
            self.sandbox_manager = None

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
        self, system_prompt: str, query: str, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build model input messages with bounded conversation history."""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        history = context.get("conversation_history", [])
        messages.extend(self._prepare_history_messages(history, system_prompt, query))
        messages.append({"role": "user", "content": query})
        return messages

    def _register_context_monitor_tools(self):
        """Register tools that expose context stats and summaries."""
        if not self.tool_manager or self._context_tools_registered:
            return

        def context_window_stats_tool() -> Dict[str, Any]:
            """Return latest context window stats."""
            return self.get_context_window_stats() or {}

        def list_summary_registry_tool() -> Dict[str, Any]:
            """List auto-generated summaries."""
            return {"summaries": self.list_context_summaries()}

        def fetch_summary_tool(summary_id: str) -> Dict[str, Any]:
            """Fetch full summary content."""
            summary = self.fetch_context_summary(summary_id)
            return summary or {}

        def autosummarize_conversation(
            days_back: int = 1, max_memories_per_summary: int = 20
        ) -> Dict[str, Any]:
            """Generate memory summaries on demand for recent conversations."""
            if not self.memory_provider:
                return {
                    "ok": False,
                    "error": "No memory provider configured. Cannot generate summaries.",
                }
            if not self.model:
                return {
                    "ok": False,
                    "error": "No LLM model configured. Cannot generate summaries.",
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
            }

        self.tool_manager.add_tool(context_window_stats_tool)
        self.tool_manager.add_tool(list_summary_registry_tool)
        self.tool_manager.add_tool(fetch_summary_tool)
        self.tool_manager.add_tool(autosummarize_conversation)
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
        if normalized_provider != "skillsmp":
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

        if "api_key" not in config:
            default_key = os.getenv("SKILLSMP_API_KEY", "").strip()
            if default_key:
                config["api_key"] = default_key

        if "base_url" not in config:
            config["base_url"] = "https://skillsmp.com"

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

        if provider_name != "skillsmp":
            raise ValueError(
                f"Unknown skills marketplace provider '{provider_name}'. "
                "Supported providers: skillsmp."
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

    def _resolve_execution_state(
        self, memory_id: Optional[str], conversation_id: Optional[str]
    ) -> Tuple[str, str]:
        """Resolve active memory/conversation IDs and keep thread state isolated."""
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

        requested_conversation_id = (
            str(conversation_id).strip() if conversation_id else ""
        )
        if requested_conversation_id:
            resolved_conversation_id = requested_conversation_id
        else:
            resolved_conversation_id = self._conversation_ids_by_memory.get(
                resolved_memory_id
            )
            if not resolved_conversation_id:
                resolved_conversation_id = str(uuid.uuid4())
                logger.debug("Started new conversation: %s", resolved_conversation_id)

        self._conversation_ids_by_memory[resolved_memory_id] = resolved_conversation_id
        self._current_conversation_id = resolved_conversation_id

        if self.cache_manager:
            self.cache_manager.update_scope(
                agent_id=self.agent_id, memory_id=resolved_memory_id
            )

        return resolved_memory_id, resolved_conversation_id

    def switch_thread(
        self, memory_id: str, start_new_conversation: bool = False
    ) -> str:
        """
        Switch the active thread to a specific memory_id.

        Returns:
            str: Active conversation_id for the selected thread.
        """
        normalized_memory_id = str(memory_id or "").strip()
        if not normalized_memory_id:
            raise ValueError("memory_id is required to switch threads.")

        conversation_id = None
        if not start_new_conversation:
            conversation_id = self._conversation_ids_by_memory.get(normalized_memory_id)

        resolved_memory_id, resolved_conversation_id = self._resolve_execution_state(
            normalized_memory_id, conversation_id
        )
        if start_new_conversation:
            resolved_conversation_id = str(uuid.uuid4())
            self._conversation_ids_by_memory[
                resolved_memory_id
            ] = resolved_conversation_id
            self._current_conversation_id = resolved_conversation_id
            logger.info(
                "Started new conversation %s for thread %s",
                resolved_conversation_id,
                resolved_memory_id,
            )

        return resolved_conversation_id

    def run(
        self, query: str, memory_id: str = None, conversation_id: str = None
    ) -> str:
        """
        Run the agent with the given query using the new manager architecture.

        Args:
            query: The user's query
            memory_id: Optional memory ID to use (if not provided, uses stored or default)
            conversation_id: Optional conversation ID to use (if not provided, reuses current or creates new)

        Returns:
            The agent's response
        """
        logger.info(f"MemAgent {self.agent_id} executing query: {query[:50]}...")

        try:
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, conversation_id = self._resolve_execution_state(
                memory_id, conversation_id
            )

            # 2. Check semantic cache first
            cached_response = None
            if self.cache_manager.enabled:
                cached_response = self.cache_manager.get_cached_response(
                    query, conversation_id
                )
                if cached_response:
                    logger.info("Returning cached response")
                    self._record_interaction(
                        query, cached_response, memory_id, conversation_id
                    )
                    return cached_response

            # 3. Build context and prompt
            context = self._build_context(query, memory_id)
            system_prompt = self._build_system_prompt()

            # 4. Execute with LLM
            response = self._execute_llm_interaction(system_prompt, query, context)

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(query, response, conversation_id)

            # 6. Record interaction in memory
            self._record_interaction(query, response, memory_id, conversation_id)

            logger.info(f"MemAgent {self.agent_id} completed successfully")
            return response

        except Exception as e:
            logger.error(f"MemAgent execution failed: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            return error_response

    def run_stream(
        self, query: str, memory_id: str = None, conversation_id: str = None
    ) -> Generator[str, None, None]:
        """
        Run the agent with streaming output, yielding text chunks as they arrive.

        This method mirrors ``run()`` but yields partial text tokens so callers
        can display incremental output.  It honours the same caching, memory,
        and tool-calling logic as the synchronous path.

        Args:
            query: The user's query
            memory_id: Optional memory ID
            conversation_id: Optional conversation ID

        Yields:
            str: Partial text chunks of the agent's response
        """
        logger.info(f"MemAgent {self.agent_id} streaming query: {query[:50]}...")

        if not self.model or not hasattr(self.model, "generate_stream"):
            # Fallback: run synchronously and yield the full result
            yield self.run(query, memory_id=memory_id, conversation_id=conversation_id)
            return

        self._stream_trace_events = []
        try:
            # 1. Prepare IDs with per-thread state isolation.
            memory_id, conversation_id = self._resolve_execution_state(
                memory_id, conversation_id
            )

            # 2. Check semantic cache first
            if self.cache_manager.enabled:
                cached = self.cache_manager.get_cached_response(query, conversation_id)
                if cached:
                    self._record_interaction(query, cached, memory_id, conversation_id)
                    yield cached
                    return

            # 3. Build context and prompt
            context = self._build_context(query, memory_id)
            system_prompt = self._build_system_prompt()

            # 4. Stream the LLM interaction
            full_response = ""
            for chunk in self._execute_llm_interaction_stream(
                system_prompt, query, context
            ):
                full_response += chunk
                yield chunk

            # 5. Cache the response
            if self.cache_manager.enabled:
                self.cache_manager.cache_response(query, full_response, conversation_id)

            # 6. Record interaction in memory
            self._record_interaction(query, full_response, memory_id, conversation_id)
            self._record_stream_trace_bundle(memory_id, conversation_id)

        except Exception as e:
            logger.error(f"MemAgent streaming failed: {e}")
            yield f"I apologize, but I encountered an error: {str(e)}"
        finally:
            self._stream_trace_events = None

    def set_stream_event_callback(
        self, callback: Optional[Callable[[Dict[str, Any]], None]]
    ):
        """Attach or clear a callback for structured streaming events."""
        self._stream_event_callback = callback
        return self

    def _emit_stream_event(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a structured stream event to the UI callback when configured."""
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

    def _record_stream_trace_bundle(self, memory_id: str, conversation_id: str) -> None:
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
                conversation_id=conversation_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
            )
            self.memory_manager.save_memory_unit(trace_memory, memory_id)
            logger.debug(
                "Recorded streamed trace bundle in memory: %s events for thread %s",
                len(trace_events),
                memory_id,
            )
        except Exception as exc:
            logger.warning(f"Failed to record streamed trace bundle: {exc}")

    def _is_trace_bundle_message(self, message: Any) -> bool:
        """Return True when a conversation row is a persisted trace bundle."""
        if not isinstance(message, dict):
            return False
        role = str(message.get("role") or "").strip().lower()
        if role != Role.TOOL.value:
            return False

        raw_content = message.get("content")
        if not isinstance(raw_content, str):
            return False
        content = raw_content.strip()
        if not content:
            return False

        try:
            payload = json.loads(content)
        except Exception:
            return False

        return (
            isinstance(payload, dict)
            and str(payload.get("type") or "").strip().lower() == "trace_bundle"
        )

    def _execute_llm_interaction_stream(
        self, system_prompt: str, query: str, context: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """Execute the LLM interaction with streaming and tool-calling support."""
        if not self.model:
            yield "Error: No LLM model configured"
            return

        workflow = None
        WorkflowOutcome = None
        if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            from ..long_term_memory.procedural.workflow.workflow import Workflow
            from ..long_term_memory.procedural.workflow.workflow import (
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
        messages = self._build_prompt_messages(system_prompt, query, context)

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

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "name": tool_name,
                                    "content": str(result),
                                }
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

    def _build_context(self, query: str, memory_id: str) -> Dict[str, Any]:
        """Build context for the query using memory manager."""
        context = {"query": query}

        if self.memory_manager:
            # Load conversation history
            try:
                history_limit = self._get_conversation_history_limit()
                history = self.memory_manager.load_conversation_history(
                    memory_id, limit=history_limit
                )
                history = [
                    item for item in history if not self._is_trace_bundle_message(item)
                ]
                context["conversation_history"] = history

                # Get relevant memories
                relevant_memories = self.memory_manager.retrieve_relevant_memories(
                    query=query,
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    memory_id=memory_id,
                    limit=5,
                )
                context["relevant_memories"] = relevant_memories

            except Exception as e:
                logger.warning(f"Failed to build memory context: {e}")

        if (
            self._entity_memory_enabled
            and self.entity_memory_manager
            and self.entity_memory_manager.is_enabled()
        ):
            try:
                entity_context = self.entity_memory_manager.build_context(
                    query=query, memory_id=memory_id
                )
                if entity_context:
                    context["entity_memory_profiles"] = entity_context
                self._last_entity_context = entity_context
            except Exception as e:
                logger.warning(f"Failed to retrieve entity memory context: {e}")
                self._last_entity_context = []
        else:
            self._last_entity_context = []

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
        9. Sandbox instructions (if enabled)
        10. Loaded skills (if configured)
        11. Configured MCP servers (if configured)
        """
        prompt_parts = []

        # 1. Base system prompt — always present
        prompt_parts.append(BASE_SYSTEM_PROMPT)

        # 2. Active memory types for this specific agent
        if self.active_memory_types:
            active_names = [
                mt.value.replace("_", " ").title() for mt in self.active_memory_types
            ]
            prompt_parts.append(
                "Active memory systems for this session: "
                + ", ".join(active_names)
                + "."
            )

        # 3. User's custom instruction
        if self.instruction and self.instruction != DEFAULT_INSTRUCTION:
            prompt_parts.append(f"Instructions:\n{self.instruction}")
        else:
            prompt_parts.append(self.instruction)

        # 4. Persona information
        persona_prompt = self.persona_manager.get_persona_prompt()
        if persona_prompt:
            prompt_parts.append(persona_prompt)

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

        # 9. Sandbox code execution instructions
        if self.has_sandbox():
            provider_name = self.get_sandbox_provider_name() or "sandbox"
            prompt_parts.append(
                f"{SANDBOX_SYSTEM_PROMPT}\n" f"Sandbox provider: {provider_name}"
            )

        # 10. Loaded skills (local files)
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

        # 11. Configured MCP servers
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
        self, system_prompt: str, query: str, context: Dict[str, Any]
    ) -> str:
        """Execute the LLM interaction with tool calling support."""
        if not self.model:
            return "Error: No LLM model configured"

        try:
            # Initialize workflow tracking if workflow memory is active
            workflow = None
            if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
                from ..long_term_memory.procedural.workflow.workflow import (
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
                logger.debug(f"Created workflow for tracking: {workflow.workflow_id}")

            # Build initial messages
            messages = self._build_prompt_messages(system_prompt, query, context)

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

                        # Add tool result to messages
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": str(result),
                            }
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
        self._skills_marketplace_tools_registered = True

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
        self, query: str, response: str, memory_id: str, conversation_id: str
    ):
        """Record the interaction in memory."""
        if not self.memory_manager:
            return

        try:
            # Record user query
            user_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.USER,
                content=query,
                conversation_id=conversation_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
            )
            self.memory_manager.save_memory_unit(user_memory, memory_id)

            # Record assistant response
            assistant_memory = self.memory_manager.create_conversation_memory_unit(
                role=Role.ASSISTANT,
                content=response,
                conversation_id=conversation_id,
                memory_id=memory_id,
                agent_id=self.agent_id,
            )
            self.memory_manager.save_memory_unit(assistant_memory, memory_id)

            logger.debug(f"Recorded interaction in memory: {memory_id}")

        except Exception as e:
            logger.warning(f"Failed to record interaction: {e}")

    # Conversation state management methods
    def start_new_conversation(self, memory_id: str = None) -> str:
        """
        Start a new conversation within the specified (or current) thread.

        Returns:
            str: The new conversation_id
        """
        target_memory_id = str(memory_id or "").strip()
        if not target_memory_id:
            target_memory_id = (
                self._current_memory_id
                or (str(self.memory_ids[0]).strip() if self.memory_ids else "")
                or str(uuid.uuid4())
            )

        new_conversation_id = str(uuid.uuid4())
        self._resolve_execution_state(target_memory_id, new_conversation_id)
        logger.info(
            "Started new conversation: %s (thread=%s)",
            new_conversation_id,
            target_memory_id,
        )
        return new_conversation_id

    def get_current_conversation_id(self) -> Optional[str]:
        """
        Get the current conversation ID.

        Returns:
            str: Current conversation_id or None
        """
        return self._current_conversation_id

    def get_current_memory_id(self) -> Optional[str]:
        """
        Get the current memory ID.

        Returns:
            str: Current memory_id or None
        """
        return self._current_memory_id

    def reset_conversation_state(self):
        """Reset both conversation_id and memory_id (start completely fresh)."""
        self._current_conversation_id = None
        self._current_memory_id = None
        self._conversation_ids_by_memory = {}
        logger.info("Reset conversation state")

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
        """Set persona (delegated to persona manager)."""
        return self.persona_manager.set_persona(persona, self.agent_id, save)

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

        # Determine the memory_id to use for tools
        # Priority: current memory_id → first memory_id → generate new
        agent_memory_id = self._current_memory_id or (
            self.memory_ids[0] if self.memory_ids else None
        )

        # If no memory_id exists, generate one and add it to the agent
        if not agent_memory_id:
            agent_memory_id = str(uuid.uuid4())
            self.memory_ids.append(agent_memory_id)
            self._current_memory_id = agent_memory_id
            logger.info(
                f"Generated new memory_id for agent {self.agent_id}: {agent_memory_id}"
            )

        # Convert tool metadata to serializable format with all necessary fields for TOOLBOX table
        serializable_tools = []
        for tool_meta in tools_metadata:
            if isinstance(tool_meta, dict):
                serializable_tool = {
                    "_id": tool_meta.get("_id")
                    or tool_meta.get("name"),  # Use name as fallback ID
                    "name": tool_meta.get("name"),
                    "description": tool_meta.get("description", ""),
                    "signature": tool_meta.get("signature", ""),
                    "docstring": tool_meta.get(
                        "docstring", tool_meta.get("description", "")
                    ),
                    "parameters": tool_meta.get("parameters", {}),
                    "type": tool_meta.get("type", "function"),
                    "memory_id": tool_meta.get("memory_id")
                    or agent_memory_id,  # Use agent's memory_id
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

        # Reconstruct LLM model
        model_to_load = None
        if hasattr(saved_memagent, "llm_config") and saved_memagent.llm_config:
            try:
                model_to_load = create_llm_provider(saved_memagent.llm_config)
            except Exception as e:
                logger.warning(
                    f"Could not load model from config: {e}. Model will be None."
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
            is_favorite=overrides.get(
                "is_favorite", getattr(saved_memagent, "is_favorite", False)
            ),
            streaming=overrides.get("streaming", False),
        )

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
                    if hasattr(saved_memagent, "name"):
                        self.name = saved_memagent.name
                    if hasattr(saved_memagent, "is_favorite"):
                        self.is_favorite = bool(saved_memagent.is_favorite)

                    # Update persona if changed
                    if hasattr(saved_memagent, "persona") and self.persona_manager:
                        self.persona_manager.set_persona(
                            saved_memagent.persona, self.agent_id, save=False
                        )

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

                    # Create summary document
                    summary_doc = {
                        "memory_id": chunk_memory_id,
                        "agent_id": self.agent_id,
                        "content": summary_content,
                        "period_start": period_start,
                        "period_end": period_end,
                        "memory_units_count": len(memory_chunk),
                        "summary_type": "automatic",
                        "created_at": current_time,
                        "embedding": get_embedding(summary_content),
                    }

                    # Store summary
                    summary_id = self.memory_provider.store(
                        summary_doc, MemoryType.SUMMARIES
                    )
                    summary_ids.append(summary_id)

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
