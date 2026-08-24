# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Builder pattern for MemAgent construction."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...enums import ApplicationMode
from ...internet_access import get_default_internet_access_provider
from ..models import MemAgentConfig

if TYPE_CHECKING:
    from ..core import MemAgent

logger = logging.getLogger(__name__)


class MemAgentBuilder:
    """
    Builder pattern for constructing MemAgent instances.

    This provides a fluent interface for complex MemAgent configurations,
    making them more readable and maintainable.
    """

    def __init__(self):
        """Initialize the builder with default configuration."""
        self.config = MemAgentConfig()
        self._model = None
        self._llm_config = None
        self._tools = []
        self._persona = None
        self._name = None
        self._memory_provider = None
        self._memory_ids = []
        self._delegates = []
        self._embedding_provider = None
        self._embedding_config = None
        self._semantic_cache_config = None
        self._entity_memory_enabled = None
        self._internet_access_provider = None
        self._skills_marketplace_provider = None
        self._skills_marketplace_config = None
        self._skill_paths = []
        self._mcp_servers = []
        self._sandbox_provider = None
        self._browser_control = None
        self._meta_harness = None
        self._meta_harness_mode = None
        self._default_harness = "auto"
        self._harness_config = None
        self._toolbox = None
        self._skillbox = None
        self._authored_skills = []
        self._skill_retrieval_enabled = False
        self._skill_retrieval_config = None
        self._tool_result_policy = None
        self._context_policy = None
        self._completion_policy = None
        self._retrieval_policy = None
        self._approval_store = None
        self._delegation_config = None
        self._semantic_layer = None
        self._self_aware_enabled = False
        self._self_aware_config = None
        self._continual_learning_enabled = False
        self._continual_learning_config = None
        self._learning_control_plane = None
        self._workflow_outcome_evaluator = None
        self._automations_enabled = True
        self._auto_register = True
        self._default_timezone = None
        self._is_favorite = False
        self._environment_reports: Dict[str, Any] = {}

    def with_instruction(self, instruction: str) -> "MemAgentBuilder":
        """Set the agent instruction."""
        self.config.instruction = instruction
        return self

    def with_model(self, model: Any) -> "MemAgentBuilder":
        """Set the LLM model."""
        self._model = model
        return self

    def with_llm_config(self, config: Dict[str, Any]) -> "MemAgentBuilder":
        """Set the LLM configuration."""
        self._llm_config = config
        return self

    def with_tools(self, tools: Union[List, Any]) -> "MemAgentBuilder":
        """Add tools to the agent."""
        if isinstance(tools, list):
            self._tools.extend(tools)
        else:
            self._tools.append(tools)
        return self

    def with_tool(self, tool: Any) -> "MemAgentBuilder":
        """Add a single tool to the agent."""
        self._tools.append(tool)
        return self

    def with_persona(
        self, persona: Any = None, name: str = None, expertise: List[str] = None
    ) -> "MemAgentBuilder":
        """Set the agent persona."""
        if persona:
            self._persona = persona
        elif name or expertise:
            # Create a simple persona from parameters
            self._persona = {"name": name or "Assistant", "expertise": expertise or []}
        return self

    def with_name(self, name: str) -> "MemAgentBuilder":
        """Set a display name for the agent."""
        self._name = name
        return self

    def with_favorite(self, is_favorite: bool = True) -> "MemAgentBuilder":
        """Mark the built agent as favorite or non-favorite."""
        self._is_favorite = bool(is_favorite)
        return self

    def with_memory_provider(self, provider: Any) -> "MemAgentBuilder":
        """Set the memory provider."""
        self._memory_provider = provider
        return self

    def with_oracle_from_env(
        self,
        *,
        ensure_ready: bool = True,
        provision_if_missing: Optional[bool] = None,
        preflight: bool = True,
        require_preflight_ok: bool = True,
        **provider_overrides: Any,
    ) -> "MemAgentBuilder":
        """Attach an Oracle provider using the documented environment.

        By default the local Docker runtime is started/readied first and a
        provider preflight is retained on the built agent in
        ``environment_reports["oracle"]``.
        """
        from ...memory_provider.oracle import LocalOracleRuntime, OracleProvider

        runtime_report = None
        if ensure_ready:
            runtime = LocalOracleRuntime.from_env(
                provision_if_missing=provision_if_missing
            )
            runtime_report = runtime.ensure_ready()
        provider = OracleProvider.from_env(
            provision_if_missing=(
                bool(provision_if_missing) if not ensure_ready else False
            ),
            **provider_overrides,
        )
        try:
            preflight_report = provider.preflight() if preflight else None
        except Exception:
            provider.close()
            raise
        if (
            require_preflight_ok
            and preflight_report is not None
            and not preflight_report.get("ok", False)
        ):
            provider.close()
            diagnostics = "; ".join(preflight_report.get("diagnostics") or [])
            raise RuntimeError(f"Oracle preflight failed: {diagnostics or 'not ready'}")
        self._memory_provider = provider
        self._environment_reports["oracle"] = {
            "runtime": runtime_report,
            "preflight": preflight_report,
        }
        return self

    def with_memory_ids(self, memory_ids: Union[str, List[str]]) -> "MemAgentBuilder":
        """Set memory IDs."""
        if isinstance(memory_ids, str):
            self._memory_ids = [memory_ids]
        else:
            self._memory_ids = memory_ids
        return self

    def with_internet_access_provider(self, provider: Any) -> "MemAgentBuilder":
        """Attach an internet access provider."""
        self._internet_access_provider = provider
        return self

    def with_skills_marketplace(
        self,
        provider: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> "MemAgentBuilder":
        """Configure a skills marketplace without post-build mutation."""
        self._skills_marketplace_provider = provider
        self._skills_marketplace_config = dict(config or {})
        return self

    def with_skills_marketplace_provider(
        self,
        provider: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> "MemAgentBuilder":
        """Compatibility alias for :meth:`with_skills_marketplace`."""
        return self.with_skills_marketplace(provider, config)

    def with_skill_paths(self, skill_paths: Union[str, List[str]]) -> "MemAgentBuilder":
        """Attach skill markdown file paths to the agent."""
        if isinstance(skill_paths, str):
            self._skill_paths.append(skill_paths)
        elif isinstance(skill_paths, list):
            self._skill_paths.extend(skill_paths)
        return self

    def with_mcp_servers(
        self, mcp_servers: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> "MemAgentBuilder":
        """Attach MCP server configurations to the agent."""
        if isinstance(mcp_servers, dict):
            self._mcp_servers.append(mcp_servers)
        elif isinstance(mcp_servers, list):
            self._mcp_servers.extend(mcp_servers)
        return self

    def with_sandbox(self, provider: Any) -> "MemAgentBuilder":
        """Attach a sandbox provider instance, name, or provider config."""
        self._sandbox_provider = provider
        return self

    def with_e2b_from_env(
        self,
        *,
        validate: bool = True,
        **provider_config: Any,
    ) -> "MemAgentBuilder":
        """Attach a stateful E2B provider using ``E2B_API_KEY`` by default."""
        from ...sandbox.providers.e2b_provider import E2BSandboxProvider

        provider = E2BSandboxProvider(**provider_config)
        validation_error = provider.validate_configuration() if validate else None
        if validation_error:
            provider.close()
            raise ValueError(validation_error)
        self._sandbox_provider = provider
        self._environment_reports["e2b"] = {
            "ok": True,
            "config": provider.get_config(),
        }
        return self

    def with_sandbox_provider(self, provider: Any) -> "MemAgentBuilder":
        """Backward-compatible alias for :meth:`with_sandbox`."""
        return self.with_sandbox(provider)

    def with_browser_control(self, provider: Any) -> "MemAgentBuilder":
        """Attach a browser-control provider instance, name, or config."""
        self._browser_control = provider
        return self

    def with_browser_control_provider(self, provider: Any) -> "MemAgentBuilder":
        """Compatibility alias for :meth:`with_browser_control`."""
        return self.with_browser_control(provider)

    def with_meta_harness(
        self,
        meta_harness: Any = None,
        *,
        mode: str = "delegate",
        default_harness: str = "auto",
        config: Optional[Dict[str, Any]] = None,
    ) -> "MemAgentBuilder":
        """Attach the memory-first external-harness control plane.

        ``delegate`` exposes bounded harness invocation tools to the native
        MemAgent loop. ``runtime`` makes the selected harness execute the
        complete turn while MemoRizz retains memory, approval, and trace
        ownership. Passing ``None`` lazily builds the standard Codex, Claude
        Code, and OpenHands registry from the local configuration.
        """
        normalized = str(mode or "delegate").strip().lower()
        if normalized not in {"delegate", "runtime"}:
            raise ValueError("meta-harness mode must be delegate or runtime")
        self._meta_harness = True if meta_harness is None else meta_harness
        self._meta_harness_mode = normalized
        self._default_harness = (
            str(default_harness or "auto").strip().lower().replace("_", "-")
        )
        self._harness_config = dict(config or {})
        return self

    def with_execution_harness(
        self,
        harness: str,
        *,
        meta_harness: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> "MemAgentBuilder":
        """Convenience alias for full-runtime meta-harness execution."""
        return self.with_meta_harness(
            meta_harness,
            mode="runtime",
            default_harness=harness,
            config=config,
        )

    def with_toolbox(self, toolbox: Any) -> "MemAgentBuilder":
        """Attach a progressively retrieved Toolbox and its callable bindings."""
        self._toolbox = toolbox
        return self

    def with_skillbox(self, skillbox: Any) -> "MemAgentBuilder":
        """Attach an authored/learned Skillbox independently of learning."""
        self._skillbox = skillbox
        return self

    def with_skills(
        self, skills: Union[List[Any], Any], *, persistence: str = "skillbox"
    ) -> "MemAgentBuilder":
        """Attach authored procedural skills without enabling continual learning."""
        if persistence != "skillbox":
            raise ValueError("Authored skill persistence must be 'skillbox'")
        self._authored_skills.extend(skills if isinstance(skills, list) else [skills])
        self._skill_retrieval_enabled = True
        return self

    def with_skill_retrieval(
        self, enabled: bool = True, top_k: int = 2, min_similarity: float = 0.70
    ) -> "MemAgentBuilder":
        """Configure progressive Skillbox retrieval independently of learning."""
        self._skill_retrieval_enabled = bool(enabled)
        self._skill_retrieval_config = {
            "top_k": max(1, int(top_k)),
            "min_similarity": float(min_similarity),
        }
        return self

    def with_tool_result_policy(self, policy: Any) -> "MemAgentBuilder":
        self._tool_result_policy = policy
        return self

    def with_context_policy(self, policy: Any) -> "MemAgentBuilder":
        self._context_policy = policy
        return self

    def with_completion_policy(self, policy: Any) -> "MemAgentBuilder":
        """Attach a host-enforced final-response acceptance policy."""
        self._completion_policy = policy
        return self

    def with_approval_store(self, store: Any) -> "MemAgentBuilder":
        self._approval_store = store
        return self

    def with_delegation(
        self,
        delegates: Union[List[Any], Any] = None,
        **config: Any,
    ) -> "MemAgentBuilder":
        """Make delegates operational and configure orchestration behavior."""
        from ...task_decomposition import normalize_delegation_config

        if delegates is not None:
            self._delegates = delegates if isinstance(delegates, list) else [delegates]
        self._delegation_config = normalize_delegation_config(config)
        return self

    def with_semantic_layer(self, catalog: Any) -> "MemAgentBuilder":
        """Attach a governed semantic catalog and query planner."""
        self._semantic_layer = catalog
        return self

    def with_self_aware(
        self, enabled: bool = True, config: Dict[str, Any] = None
    ) -> "MemAgentBuilder":
        """Enable/disable self-awareness host codebase tooling."""
        self._self_aware_enabled = bool(enabled)
        if isinstance(config, dict):
            self._self_aware_config = dict(config)
        elif config is None:
            self._self_aware_config = None
        return self

    def with_continual_learning(
        self, enabled: bool = True, config: Dict[str, Any] = None
    ) -> "MemAgentBuilder":
        """Enable/disable the workflow→skill continual learning loop.

        ``config`` keys follow ``PromotionConfig`` (e.g. ``min_executions``,
        ``require_shadow``, ``shadow_evaluation_enabled``,
        ``promotion_every_n_runs``).
        """
        self._continual_learning_enabled = bool(enabled)
        if isinstance(config, dict):
            self._continual_learning_config = dict(config)
        elif config is None:
            self._continual_learning_config = None
        return self

    def with_learning_control_plane(
        self,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
        **settings: Any,
    ) -> "MemAgentBuilder":
        """Enable bounded evidence, event compilation, and governed forgetting."""
        value = dict(config or {})
        value.update(settings)
        value["enabled"] = bool(enabled)
        self._learning_control_plane = value
        return self

    def with_workflow_outcome_evaluator(self, evaluator: Any) -> "MemAgentBuilder":
        """Set a runtime business-success rubric for captured workflows."""
        if evaluator is not None and not callable(evaluator):
            raise TypeError("workflow outcome evaluator must be callable or None")
        self._workflow_outcome_evaluator = evaluator
        return self

    def with_automations_enabled(self, enabled: bool = True) -> "MemAgentBuilder":
        """Enable/disable durable automations tooling (when supported by provider)."""
        self._automations_enabled = bool(enabled)
        return self

    def with_auto_registration(self, enabled: bool = True) -> "MemAgentBuilder":
        """Control fail-soft persistence of agent metadata on the first run."""
        self._auto_register = bool(enabled)
        return self

    def as_ephemeral(self, enabled: bool = True) -> "MemAgentBuilder":
        """Build a run-scoped agent without persisting its tool/config snapshot.

        Memory, trace, verification, and learning evidence still use the configured
        provider. Only automatic agent registration and unused automation tooling are
        disabled, which is appropriate for evaluators and short-lived workers.
        """
        self._auto_register = not bool(enabled)
        if enabled:
            self._automations_enabled = False
        return self

    def with_default_timezone(self, tz: str) -> "MemAgentBuilder":
        """Set a default IANA timezone used by automation tools when omitted."""
        tz_value = str(tz or "").strip()
        self._default_timezone = tz_value or None
        return self

    def with_semantic_cache(
        self, enabled: bool = True, threshold: float = 0.85, scope: str = "session"
    ) -> "MemAgentBuilder":
        """Configure semantic caching (isolated to a session by default)."""
        self.config.semantic_cache = enabled
        if enabled:
            self._semantic_cache_config = {
                "similarity_threshold": threshold,
                "scope": scope,
            }
        return self

    def with_retrieval_policy(self, policy: Any) -> "MemAgentBuilder":
        """Set automatic memory-retrieval scope independently of storage."""
        self._retrieval_policy = policy
        return self

    def with_embedding_provider(
        self, provider: str, config: Dict[str, Any] = None
    ) -> "MemAgentBuilder":
        """Set the embedding provider."""
        self._embedding_provider = provider
        self._embedding_config = config or {}
        return self

    def with_max_steps(self, steps: int) -> "MemAgentBuilder":
        """Set maximum execution steps."""
        self.config.max_steps = steps
        return self

    def with_application_mode(self, mode: str) -> "MemAgentBuilder":
        """Set the application mode."""
        self.config.application_mode = mode
        return self

    def with_entity_memory(self, enabled: bool = True) -> "MemAgentBuilder":
        """Explicitly enable or disable entity memory for the agent."""
        self._entity_memory_enabled = enabled
        return self

    def with_tool_access(self, access: str) -> "MemAgentBuilder":
        """Set tool access level."""
        self.config.tool_access = access
        return self

    def with_delegates(self, delegates: List[Any]) -> "MemAgentBuilder":
        """Set delegates and enable model-generated orchestration by default."""
        self._delegates = delegates
        if self._delegation_config is None:
            self._delegation_config = {"enabled": True, "mode": "auto"}
        return self

    def with_verbose(self, verbose: bool = True) -> "MemAgentBuilder":
        """Enable verbose logging."""
        self.config.verbose = verbose
        return self

    def build(self, validate: bool = True, persist: bool = False) -> "MemAgent":
        """
        Build the MemAgent instance.

        Returns:
            Configured MemAgent instance.
        """
        try:
            # Import here to avoid circular imports
            from ..core import MemAgent

            # Oracle's Skillbox rows retain a real foreign key to the owning
            # agent.  For ``build_and_save`` the parent must therefore be
            # committed before authored skills are inserted.  Delay only
            # those writes; ordinary ``build()`` retains its existing eager
            # authored-skill behavior for provider compatibility.
            deferred_authored_skills = (
                list(self._authored_skills)
                if persist and self._authored_skills and self._memory_provider
                else []
            )

            # Create the agent with all configured parameters
            agent = MemAgent(
                model=self._model,
                llm_config=self._llm_config,
                tools=self._tools if self._tools else None,
                persona=self._persona,
                instruction=self.config.instruction,
                application_mode=getattr(self.config, "application_mode", "assistant"),
                max_steps=self.config.max_steps,
                memory_provider=self._memory_provider,
                memory_ids=self._memory_ids if self._memory_ids else None,
                tool_access=self.config.tool_access,
                name=self._name,
                is_favorite=self._is_favorite,
                delegates=self._delegates if self._delegates else None,
                verbose=getattr(self.config, "verbose", None),
                embedding_provider=self._embedding_provider,
                embedding_config=self._embedding_config,
                semantic_cache=self.config.semantic_cache,
                semantic_cache_config=self._semantic_cache_config,
                retrieval_policy=self._retrieval_policy,
                context_window_tokens=getattr(
                    self.config, "context_window_tokens", None
                ),
                internet_access_provider=self._internet_access_provider,
                skills_marketplace_provider=self._skills_marketplace_provider,
                skills_marketplace_config=self._skills_marketplace_config,
                skill_paths=self._skill_paths if self._skill_paths else None,
                mcp_servers=self._mcp_servers if self._mcp_servers else None,
                sandbox_provider=self._sandbox_provider,
                browser_control=self._browser_control,
                meta_harness=self._meta_harness,
                meta_harness_mode=self._meta_harness_mode,
                default_harness=self._default_harness,
                harness_config=self._harness_config,
                toolbox=self._toolbox,
                skillbox=self._skillbox,
                authored_skills=(
                    None
                    if deferred_authored_skills
                    else (self._authored_skills if self._authored_skills else None)
                ),
                skill_retrieval=self._skill_retrieval_enabled,
                skill_retrieval_config=self._skill_retrieval_config,
                tool_result_policy=self._tool_result_policy,
                context_policy=self._context_policy,
                completion_policy=self._completion_policy,
                approval_store=self._approval_store,
                delegation=self._delegation_config,
                semantic_layer=self._semantic_layer,
                automations_enabled=self._automations_enabled,
                default_timezone=self._default_timezone,
                self_aware=self._self_aware_enabled,
                self_aware_config=self._self_aware_config,
                continual_learning=self._continual_learning_enabled,
                continual_learning_config=self._continual_learning_config,
                learning_control_plane=self._learning_control_plane,
                workflow_outcome_evaluator=self._workflow_outcome_evaluator,
                auto_register=self._auto_register,
            )
            agent.environment_reports = {
                key: dict(value) if isinstance(value, dict) else value
                for key, value in self._environment_reports.items()
            }

            logger.info(f"MemAgent built successfully with {len(self._tools)} tools")

            if self._entity_memory_enabled is not None:
                try:
                    agent.with_entity_memory(self._entity_memory_enabled)
                except Exception as exc:
                    logger.warning(f"Failed to configure entity memory: {exc}")

            if validate:
                agent.validate_configuration()
            if persist:
                agent.save()
                if deferred_authored_skills:
                    agent.authored_skills = deferred_authored_skills
                    agent._persist_authored_skills(strict=True)

            return agent

        except Exception as e:
            logger.error(f"Failed to build MemAgent: {e}")
            raise

    def build_and_save(self, validate: bool = True) -> "MemAgent":
        """Build, validate, and persist the configured agent."""
        return self.build(validate=validate, persist=True)

    def clone(self) -> "MemAgentBuilder":
        """
        Create a copy of this builder.

        Returns:
            New MemAgentBuilder instance with the same configuration.
        """
        new_builder = MemAgentBuilder()

        # Copy configuration
        new_builder.config = MemAgentConfig(**self.config.to_dict())
        new_builder._model = self._model
        new_builder._llm_config = self._llm_config.copy() if self._llm_config else None
        new_builder._tools = self._tools.copy()
        new_builder._internet_access_provider = self._internet_access_provider
        new_builder._skills_marketplace_provider = self._skills_marketplace_provider
        new_builder._skills_marketplace_config = (
            self._skills_marketplace_config.copy()
            if self._skills_marketplace_config
            else None
        )
        new_builder._is_favorite = self._is_favorite
        new_builder._persona = self._persona
        new_builder._name = self._name
        new_builder._memory_provider = self._memory_provider
        new_builder._memory_ids = self._memory_ids.copy()
        new_builder._delegates = self._delegates.copy()
        new_builder._embedding_provider = self._embedding_provider
        new_builder._embedding_config = (
            self._embedding_config.copy() if self._embedding_config else None
        )
        new_builder._semantic_cache_config = (
            self._semantic_cache_config.copy() if self._semantic_cache_config else None
        )
        new_builder._skill_paths = self._skill_paths.copy()
        new_builder._mcp_servers = self._mcp_servers.copy()
        new_builder._sandbox_provider = self._sandbox_provider
        new_builder._browser_control = self._browser_control
        new_builder._meta_harness = self._meta_harness
        new_builder._meta_harness_mode = self._meta_harness_mode
        new_builder._default_harness = self._default_harness
        new_builder._harness_config = (
            self._harness_config.copy() if self._harness_config else None
        )
        new_builder._toolbox = self._toolbox
        new_builder._skillbox = self._skillbox
        new_builder._authored_skills = self._authored_skills.copy()
        new_builder._skill_retrieval_enabled = self._skill_retrieval_enabled
        new_builder._skill_retrieval_config = (
            self._skill_retrieval_config.copy()
            if self._skill_retrieval_config
            else None
        )
        new_builder._tool_result_policy = self._tool_result_policy
        new_builder._context_policy = self._context_policy
        new_builder._completion_policy = self._completion_policy
        new_builder._retrieval_policy = self._retrieval_policy
        new_builder._approval_store = self._approval_store
        new_builder._delegation_config = (
            self._delegation_config.copy() if self._delegation_config else None
        )
        new_builder._semantic_layer = self._semantic_layer
        new_builder._self_aware_enabled = self._self_aware_enabled
        new_builder._self_aware_config = (
            self._self_aware_config.copy() if self._self_aware_config else None
        )
        new_builder._continual_learning_enabled = self._continual_learning_enabled
        new_builder._continual_learning_config = (
            self._continual_learning_config.copy()
            if self._continual_learning_config
            else None
        )
        new_builder._learning_control_plane = (
            self._learning_control_plane.copy()
            if isinstance(self._learning_control_plane, dict)
            else self._learning_control_plane
        )
        new_builder._workflow_outcome_evaluator = self._workflow_outcome_evaluator
        new_builder._automations_enabled = self._automations_enabled
        new_builder._auto_register = self._auto_register
        new_builder._default_timezone = self._default_timezone
        new_builder._environment_reports = {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in self._environment_reports.items()
        }

        return new_builder


# Convenience functions for common patterns
def create_assistant(
    name: str = "Assistant", expertise: List[str] = None
) -> MemAgentBuilder:
    """Create a builder configured for a general assistant."""
    return (
        MemAgentBuilder()
        .with_name(name)
        .with_instruction("You are a helpful AI assistant.")
        .with_persona(name=name, expertise=expertise or [])
        .with_application_mode("assistant")
    )


def create_chatbot(
    personality: str = "friendly", memory_enabled: bool = True
) -> MemAgentBuilder:
    """Create a builder configured for a chatbot."""
    instruction = f"You are a {personality} chatbot focused on engaging conversations."
    builder = (
        MemAgentBuilder().with_instruction(instruction).with_application_mode("chatbot")
    )

    if memory_enabled:
        builder = builder.with_semantic_cache(enabled=True)

    return builder


def create_task_agent(
    task_description: str, tools: List[Any] = None
) -> MemAgentBuilder:
    """Create a builder configured for a task-oriented agent."""
    instruction = (
        f"You are a task-oriented agent. Your primary task: {task_description}"
    )
    builder = (
        MemAgentBuilder()
        .with_instruction(instruction)
        .with_application_mode("agent")
        .with_max_steps(30)
    )

    if tools:
        builder = builder.with_tools(tools)

    return builder


def create_deep_research_agent(
    instruction: str = "You are a deep research agent. Break complex questions into sub-tasks, call tools, and return a sourced synthesis.",
    internet_provider=None,
) -> MemAgentBuilder:
    """Create a builder configured for Deep Research mode with internet tooling."""
    builder = (
        MemAgentBuilder()
        .with_instruction(instruction)
        .with_application_mode(ApplicationMode.DEEP_RESEARCH.value)
    )

    provider = internet_provider or get_default_internet_access_provider()
    if provider:
        builder = builder.with_internet_access_provider(provider)

    return builder
