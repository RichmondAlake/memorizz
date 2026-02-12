"""Builder pattern for MemAgent construction."""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Union

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
        self._skill_paths = []
        self._mcp_servers = []
        self._is_favorite = False

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

    def with_semantic_cache(
        self, enabled: bool = True, threshold: float = 0.85, scope: str = "local"
    ) -> "MemAgentBuilder":
        """Configure semantic caching."""
        self.config.semantic_cache = enabled
        if enabled:
            self._semantic_cache_config = {
                "similarity_threshold": threshold,
                "scope": scope,
            }
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
        """Set delegate agents for multi-agent mode."""
        self._delegates = delegates
        return self

    def with_verbose(self, verbose: bool = True) -> "MemAgentBuilder":
        """Enable verbose logging."""
        self.config.verbose = verbose
        return self

    def build(self) -> "MemAgent":
        """
        Build the MemAgent instance.

        Returns:
            Configured MemAgent instance.
        """
        try:
            # Import here to avoid circular imports
            from ..core import MemAgent

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
                context_window_tokens=getattr(
                    self.config, "context_window_tokens", None
                ),
                internet_access_provider=self._internet_access_provider,
                skill_paths=self._skill_paths if self._skill_paths else None,
                mcp_servers=self._mcp_servers if self._mcp_servers else None,
            )

            logger.info(f"MemAgent built successfully with {len(self._tools)} tools")

            if self._entity_memory_enabled is not None:
                try:
                    agent.with_entity_memory(self._entity_memory_enabled)
                except Exception as exc:
                    logger.warning(f"Failed to configure entity memory: {exc}")

            return agent

        except Exception as e:
            logger.error(f"Failed to build MemAgent: {e}")
            raise

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
