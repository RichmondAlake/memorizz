# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from bson import ObjectId
from pydantic import BaseModel

from .embeddings import configure_embeddings
from .enums import ApplicationMode, ApplicationModeConfig, MemoryType, Role
from .llms.azure import AzureOpenAI
from .llms.llm_factory import create_llm_provider
from .llms.llm_provider import LLMProvider
from .llms.openai import OpenAI
from .long_term_memory.episodic.conversational_memory_unit import ConversationMemoryUnit
from .long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType
from .long_term_memory.procedural.toolbox.toolbox import Toolbox
from .long_term_memory.procedural.workflow.workflow import Workflow, WorkflowOutcome
from .long_term_memory.semantic.knowledge_base import KnowledgeBase
from .long_term_memory.semantic.persona.persona import Persona
from .memory_provider import MemoryProvider
from .memory_unit import MemoryUnit
from .short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig
from .short_term_memory.working_memory.cwm import CWM

# Configure logging based on environment variable
MEMORIZZ_LOG_LEVEL = os.getenv("MEMORIZZ_LOG_LEVEL", "DEBUG").upper()
logger = logging.getLogger(__name__)

# Set package-wide logging level
memorizz_logger = logging.getLogger("src.memorizz")
memorizz_logger.setLevel(getattr(logging, MEMORIZZ_LOG_LEVEL, logging.WARNING))

# Configuration constants
DEFAULT_INSTRUCTION = "You are a helpful assistant."
DEFAULT_MAX_STEPS = 20
DEFAULT_TOOL_ACCESS = "private"


class MemAgentModel(BaseModel):
    model: Optional[LLMProvider] = None
    llm_config: Optional[Dict[str, Any]] = None  # Configuration for the LLM
    agent_id: Optional[str] = None
    tools: Optional[Union[List, Toolbox]] = None
    persona: Optional[Persona] = None
    instruction: Optional[str] = None
    application_mode: Optional[str] = ApplicationMode.DEFAULT.value
    memory_types: Optional[
        List[str]
    ] = None  # Custom memory types that override application_mode defaults
    max_steps: int = DEFAULT_MAX_STEPS
    memory_ids: Optional[List[str]] = None
    tool_access: Optional[str] = DEFAULT_TOOL_ACCESS
    long_term_memory_ids: Optional[List[str]] = None
    delegates: Optional[List[str]] = None  # Store delegate agent IDs
    embedding_config: Optional[Dict[str, Any]] = None
    semantic_cache: Optional[bool] = False  # Enable semantic cache
    semantic_cache_config: Optional[
        Union[SemanticCacheConfig, Dict[str, Any]]
    ] = None  # Semantic cache configuration

    model_config = {
        "arbitrary_types_allowed": True  # Allow arbitrary types like Toolbox
    }


class MemAgent:
    def __init__(
        self,
        model: Optional[LLMProvider] = None,  # LLM to use
        llm_config: Optional[Dict[str, Any]] = None,  # Configuration for the LLM
        tools: Optional[Union[List, Toolbox]] = None,  # List of tools to use or toolbox
        persona: Optional[Persona] = None,  # Persona of the agent
        instruction: Optional[str] = None,  # Instruction of the agent
        application_mode: Optional[
            str
        ] = ApplicationMode.DEFAULT.value,  # Application mode of the agent
        memory_types: Optional[
            List[Union[str, MemoryType]]
        ] = None,  # Custom memory types (overrides application_mode)
        max_steps: int = DEFAULT_MAX_STEPS,  # Maximum steps of the agent
        memory_provider: Optional[
            MemoryProvider
        ] = None,  # Memory provider of the agent
        memory_ids: Optional[Union[str, List[str]]] = None,  # Memory id(s) of the agent
        agent_id: Optional[str] = None,  # Agent id of the agent
        tool_access: Optional[str] = DEFAULT_TOOL_ACCESS,  # Tool access of the agent
        delegates: Optional[
            List["MemAgent"]
        ] = None,  # Delegate agents for multi-agent mode
        verbose: bool = None,  # Control logging verbosity (None=use env var, True=INFO, False=WARNING)
        embedding_provider: Optional[
            str
        ] = None,  # Embedding provider to use (openai, ollama)
        embedding_config: Optional[
            Dict[str, Any]
        ] = None,  # Configuration for the embedding provider
        semantic_cache: bool = False,  # Enable semantic cache for query-response caching
        semantic_cache_config: Optional[
            Union[SemanticCacheConfig, Dict[str, Any]]
        ] = None,  # Configuration for semantic cache
    ):
        # If the memory provider is not provided, then we use the default memory provider
        if memory_provider is None:
            logger.debug("No memory provider specified, using default MemoryProvider")
        self.memory_provider = memory_provider or MemoryProvider()

        # Store direct embedding configuration if provided
        self._direct_embedding_provider = embedding_provider
        self._direct_embedding_config = embedding_config

        # Configure embedding provider if specified
        if embedding_provider is not None:
            try:
                configure_embeddings(embedding_provider, embedding_config)
                logger.info(f"Configured embedding provider: {embedding_provider}")
            except Exception as e:
                logger.error(
                    f"Failed to configure embedding provider '{embedding_provider}': {str(e)}"
                )
                raise

        # Extract and store embedding configuration with proper priority
        logger.debug(
            f"About to extract embedding config. Memory provider: {type(self.memory_provider)}"
        )
        if hasattr(self.memory_provider, "config"):
            logger.debug(f"Memory provider config: {self.memory_provider.config}")
            logger.debug(
                f"Memory provider _embedding_provider: {getattr(self.memory_provider, '_embedding_provider', 'NOT_FOUND')}"
            )
        self.embedding_config = self._extract_embedding_config()
        logger.debug(f"Extracted embedding config: {self.embedding_config}")

        # Configure logging verbosity for this agent instance
        if verbose is not None:
            log_level = logging.INFO if verbose else logging.WARNING
            memorizz_logger.setLevel(log_level)
            logger.setLevel(log_level)

        # Validate and set the application mode (handles both strings and enums)
        try:
            self.application_mode = ApplicationModeConfig.validate_mode(
                application_mode
            )
        except ValueError as e:
            logger.warning(f"{e}. Using default mode 'assistant'.")
            self.application_mode = ApplicationMode.DEFAULT

        # Resolve final memory types (custom memory_types override application_mode defaults)
        if memory_types is not None:
            # Convert string memory types to MemoryType enums
            self.active_memory_types = []
            for mt in memory_types:
                if isinstance(mt, str):
                    try:
                        self.active_memory_types.append(MemoryType(mt.upper()))
                    except ValueError:
                        logger.warning(f"Invalid memory type '{mt}'. Skipping.")
                elif isinstance(mt, MemoryType):
                    self.active_memory_types.append(mt)

            logger.info(
                f"Using custom memory types: {[mt.value for mt in self.active_memory_types]}"
            )
        else:
            # Use default memory types from application mode
            self.active_memory_types = ApplicationModeConfig.get_memory_types(
                self.application_mode
            )
            logger.info(
                f"Using application mode '{self.application_mode.value}' with memory types: {[mt.value for mt in self.active_memory_types]}"
            )

        # Initialize semantic cache if enabled
        self.semantic_cache_instance = None
        if semantic_cache:
            # Handle both dictionary and SemanticCacheConfig inputs
            if semantic_cache_config:
                if isinstance(semantic_cache_config, dict):
                    # Convert dictionary to SemanticCacheConfig with validation
                    try:
                        # Merge agent's embedding settings with provided config
                        config_dict = semantic_cache_config.copy()
                        if embedding_provider:
                            config_dict["embedding_provider"] = embedding_provider
                            logger.debug(
                                f"Forcing semantic cache to use agent's embedding_provider: {embedding_provider}"
                            )
                        if embedding_config:
                            config_dict["embedding_config"] = embedding_config
                            logger.debug(
                                f"Forcing semantic cache to use agent's embedding_config: {embedding_config}"
                            )

                        # Handle string scope values in dictionary
                        if "scope" in config_dict and isinstance(
                            config_dict["scope"], str
                        ):
                            from .enums.semantic_cache_scope import SemanticCacheScope

                            scope_str = config_dict["scope"].lower()
                            if scope_str == "local":
                                config_dict["scope"] = SemanticCacheScope.LOCAL
                            elif scope_str == "global":
                                config_dict["scope"] = SemanticCacheScope.GLOBAL
                            else:
                                logger.warning(
                                    f"Invalid scope value '{config_dict['scope']}', using default LOCAL"
                                )
                                config_dict["scope"] = SemanticCacheScope.LOCAL

                        config = SemanticCacheConfig(**config_dict)
                        logger.debug(
                            "Successfully converted dictionary to SemanticCacheConfig"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to create SemanticCacheConfig from dictionary: {e}"
                        )
                        logger.info(
                            "Using default config with agent's embedding settings"
                        )
                        config = SemanticCacheConfig(
                            embedding_provider=embedding_provider,
                            embedding_config=embedding_config,
                        )
                else:
                    # Already a SemanticCacheConfig object
                    config = semantic_cache_config
                    # IMPORTANT: Always use the agent's embedding configuration for consistency
                    # Override any embedding settings in config to ensure consistency
                    if embedding_provider:
                        config.embedding_provider = embedding_provider
                        logger.debug(
                            f"Forcing semantic cache to use agent's embedding_provider: {embedding_provider}"
                        )
                    if embedding_config:
                        config.embedding_config = embedding_config
                        logger.debug(
                            f"Forcing semantic cache to use agent's embedding_config: {embedding_config}"
                        )
            else:
                # Create default config with agent's embedding settings
                config = SemanticCacheConfig(
                    embedding_provider=embedding_provider,
                    embedding_config=embedding_config,
                )

            # Pass the global embedding manager to ensure perfect consistency
            embedding_manager = None
            try:
                from .embeddings import get_embedding_manager

                embedding_manager = get_embedding_manager()
                logger.debug(
                    "Using global embedding manager for semantic cache consistency"
                )
            except Exception as e:
                logger.warning(f"Could not get global embedding manager: {e}")

            self.semantic_cache_instance = SemanticCache(
                config=config,
                memory_provider=self.memory_provider,
                embedding_manager=embedding_manager,  # Use same embedding manager as agent
                agent_id=None,  # Will be set after agent_id is established
                memory_id=None,  # Will be set if memory_ids are established
            )
            logger.info(
                f"SemanticCache enabled with agent's embedding config for consistency"
            )

        # Initialize the model - honor caller's model if provided, else use default
        # This allows users to specify their own model configuration
        if model:
            self.model = model
        elif llm_config:
            self.model = create_llm_provider(llm_config)
        else:
            self.model = OpenAI()

        # Initialize the memory unit based on the application mode
        self.memory_unit = MemoryUnit(
            self.application_mode.value, self.memory_provider, llm_provider=self.model
        )

        # Multi-agent setup
        self.delegates = delegates or []
        self.is_multi_agent_mode = len(self.delegates) > 0
        self._multi_agent_orchestrator = None

        # Ensure delegates have agent IDs for proper persistence
        if self.delegates:
            for delegate in self.delegates:
                if not hasattr(delegate, "agent_id") or not delegate.agent_id:
                    # Generate an agent ID if not present
                    delegate.agent_id = None  # Will be set during save()
                    logger.info(
                        f"Delegate without agent_id will be saved to get proper ID"
                    )

        # If the memory provider is provided and the agent id is provided, then we load the memagent from the memory provider
        if memory_provider and agent_id:
            try:
                # Load the memagent from the memory provider
                loaded_agent = memory_provider.retrieve_memagent(agent_id)
                if loaded_agent:
                    # Copy all the attributes from the loaded agent to self
                    for key, value in vars(loaded_agent).items():
                        setattr(self, key, value)

                    # If the model is not provided, then we use the default model
                    if loaded_agent.model is None:
                        self.model = OpenAI(model="gpt-4.1")

                    # Load delegate agents if they exist
                    if hasattr(loaded_agent, "delegates") and loaded_agent.delegates:
                        self.delegates = []
                        for delegate_id in loaded_agent.delegates:
                            try:
                                delegate_agent = MemAgent.load(
                                    delegate_id, memory_provider
                                )
                                self.delegates.append(delegate_agent)
                            except Exception as e:
                                logger.warning(
                                    f"Could not load delegate agent {delegate_id}: {e}"
                                )
                        self.is_multi_agent_mode = len(self.delegates) > 0

                    return
                else:
                    logger.info(
                        f"No agent found with id {agent_id}, creating a new one"
                    )
            except Exception as e:
                logger.warning(f"Error loading agent from memory provider: {e}")
                logger.info("Creating a new agent instead")

        # Initialize the memagent
        self.tools = tools
        self.persona = persona
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.max_steps = max_steps
        self.tool_access = tool_access
        # Initialize memory_ids as a list, converting single string if needed
        if memory_ids is None:
            self.memory_ids = []
        elif isinstance(memory_ids, str):
            self.memory_ids = [memory_ids]
        else:
            self.memory_ids = memory_ids

        self.agent_id = agent_id

        # Thread ID persistence: Store current thread_id to reuse across runs
        # This fixes the issue where each run() generates a new thread_id
        self._current_thread_id = None

        # Update semantic cache with agent ID and memory ID if enabled
        if self.semantic_cache_instance:
            self.semantic_cache_instance.agent_id = self.agent_id
            # Use the first memory_id if available for cache scoping
            if self.memory_ids:
                self.semantic_cache_instance.memory_id = self.memory_ids[0]
                logger.debug(
                    f"SemanticCache configured with agent_id={self.agent_id}, memory_id={self.memory_ids[0]}"
                )

        # If tools is a Toolbox, properly initialize it using the same logic as add_tool
        if isinstance(tools, Toolbox):
            self._initialize_tools_from_toolbox(tools)

    def _initialize_multi_agent_orchestrator(self):
        """
        Initialize the multi-agent orchestrator with enhanced hierarchical support.

        This method sets up the orchestrator for both flat and hierarchical scenarios.
        The orchestrator automatically detects if this agent is already part of an
        active shared session and coordinates accordingly.
        """
        if self.is_multi_agent_mode and not self._multi_agent_orchestrator:
            # Import here to avoid circular imports
            from .multi_agent_orchestrator import MultiAgentOrchestrator

            self._multi_agent_orchestrator = MultiAgentOrchestrator(
                self, self.delegates
            )

            logger.info(
                f"Initialized orchestrator for agent {self.agent_id} with {len(self.delegates)} delegates"
            )

    def _check_for_shared_memory_context(self) -> str:
        """
        Enhanced shared memory context checking for hierarchical coordination.

        This method provides comprehensive shared memory context to agents,
        enabling them to understand their place in the agent hierarchy and
        coordinate effectively with other agents at all levels.

        Returns:
            str: Formatted shared memory context for inclusion in agent prompts
        """
        try:
            # Import here to avoid circular imports
            from .coordination.shared_memory.shared_memory import SharedMemory

            shared_memory = SharedMemory(self.memory_provider)

            # Find any active session this agent is participating in
            active_session = shared_memory.find_active_session_for_agent(self.agent_id)

            if not active_session:
                return ""

            # Get comprehensive hierarchy information
            session_id = str(active_session.get("_id"))
            hierarchy = shared_memory.get_agent_hierarchy(session_id)
            recent_entries = shared_memory.get_blackboard_entries(session_id)

            # Build rich context string
            context = "\n\n---------MULTI-AGENT COORDINATION CONTEXT---------\n"
            context += (
                f"You are part of a multi-agent workflow (Session: {session_id})\n\n"
            )

            # Hierarchy information
            context += f"AGENT HIERARCHY:\n"
            context += f"• Root Agent: {hierarchy.get('root_agent')}\n"
            context += f"• Delegate Agents: {', '.join(hierarchy.get('delegate_agents', []))}\n"
            context += f"• Sub-Agents: {', '.join(hierarchy.get('sub_agents', []))}\n"
            context += f"• Total Agents: {hierarchy.get('total_agents')}\n"
            context += f"• Your Role: {'Root' if self.agent_id == hierarchy.get('root_agent') else 'Delegate/Sub-Agent'}\n\n"

            # Recent coordination activities
            if recent_entries:
                context += f"RECENT COORDINATION ACTIVITIES:\n"
                for entry in recent_entries[-5:]:  # Last 5 entries
                    agent_id = entry.get("agent_id")
                    entry_type = entry.get("entry_type")
                    content = entry.get("content", {})

                    # Extract meaningful information based on entry type
                    activity_detail = self._format_blackboard_activity(
                        entry_type, content
                    )
                    context += f"• {agent_id}: {entry_type} - {activity_detail}\n"
                context += "\n"

            context += "Use this information to coordinate effectively with other agents in the workflow.\n"
            context += "-" * 50

            return context

        except Exception as e:
            logger.error(f"Error checking shared memory context: {e}")
            return ""

    def _export_multi_agent_logs(self, system_prompt: str, augmented_query: str):
        """
        Export agent-specific logs when in multi-agent mode.

        Creates separate files for each agent's system prompt and augmented query
        to enable debugging of multi-agent coordination.

        Parameters:
            system_prompt (str): The system prompt for this agent
            augmented_query (str): The augmented query for this agent
        """
        logger.info(f"ENTERING _export_multi_agent_logs for agent {self.agent_id}")
        try:
            # Proper multi-agent detection:
            # 1. Agent has delegates (is root agent)
            # 2. Agent is part of an active shared memory session (true multi-agent coordination)
            has_active_shared_session = self._has_shared_memory_session()

            is_multi_agent = (
                self.is_multi_agent_mode
                or len(self.delegates) > 0
                or has_active_shared_session
            )

            # Debug logging to see what's happening
            logger.info(f"Multi-agent log export check for agent {self.agent_id}:")
            logger.info(f"  - is_multi_agent_mode: {self.is_multi_agent_mode}")
            logger.info(f"  - delegates count: {len(self.delegates)}")
            logger.info(f"  - has_active_shared_session: {has_active_shared_session}")
            logger.info(
                f"  - memory_ids (for context): {getattr(self, 'memory_ids', [])}"
            )
            logger.info(f"  - Final is_multi_agent: {is_multi_agent}")

            if not is_multi_agent:
                logger.info(
                    f"Skipping multi-agent logs for agent {self.agent_id} (single agent mode)"
                )
                return

            import os

            # Create multi_agent_logs directory if it doesn't exist
            log_dir = "multi_agent_logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                logger.info(f"Created multi_agent_logs directory")

            # Get current timestamp for this execution session
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create agent-specific filename
            agent_filename = f"{self.agent_id}_{timestamp}"

            # Determine the correct multi-agent status for this agent
            agent_role = self._get_agent_role_in_workflow()

            # Log agent information for debugging (safer than file writes)
            logger.debug(f"Multi-agent context for agent {self.agent_id}:")
            logger.debug(f"  - Agent Role: {agent_role}")
            logger.debug(f"  - Has Delegates: {len(self.delegates) > 0}")
            logger.debug(f"  - Delegates Count: {len(self.delegates)}")
            logger.debug(f"  - Memory IDs: {getattr(self, 'memory_ids', [])}")
            logger.debug(f"  - System prompt length: {len(system_prompt)} characters")
            logger.debug(
                f"  - Augmented query length: {len(augmented_query)} characters"
            )

            # Note: Removed file writes for better security and to prevent file conflicts.
            # Consider implementing configurable debug file output with proper temp file handling
            # if detailed logging to files is needed for development/debugging purposes.

        except Exception as e:
            logger.error(f"Error exporting multi-agent logs: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def _get_agent_role_in_workflow(self) -> str:
        """
        Determine this agent's role in the multi-agent workflow.

        Returns:
            str: The agent's role (Root Agent, Delegate Agent, Sub-Agent, or Single Agent)
        """
        try:
            # Check if this agent has delegates (is a root agent)
            if len(self.delegates) > 0:
                return "Root Agent"

            # Check if this agent is part of a shared memory session
            if self._has_shared_memory_session():
                from .coordination.shared_memory.shared_memory import SharedMemory

                shared_memory = SharedMemory(self.memory_provider)
                active_session = shared_memory.find_active_session_for_agent(
                    self.agent_id
                )

                if active_session:
                    # Check what role this agent has in the session
                    if active_session.get("root_agent_id") == self.agent_id:
                        return "Root Agent"
                    elif self.agent_id in active_session.get("delegate_agent_ids", []):
                        return "Delegate Agent"
                    elif self.agent_id in active_session.get("sub_agent_ids", []):
                        return "Sub-Agent"
                    else:
                        return "Multi-Agent Participant"

            return "Single Agent"

        except Exception as e:
            logger.error(f"Error determining agent role: {e}")
            return "Unknown Role"

    def _has_shared_memory_session(self) -> bool:
        """Check if this agent is part of any shared memory session."""
        try:
            from .coordination.shared_memory.shared_memory import SharedMemory

            shared_memory = SharedMemory(self.memory_provider)
            return (
                shared_memory.find_active_session_for_agent(self.agent_id) is not None
            )
        except Exception:
            return False

    def _get_application_mode_value(self) -> str:
        """Get the application mode value as a string, handling both enum and string cases."""
        return (
            self.application_mode.value
            if hasattr(self.application_mode, "value")
            else self.application_mode
        )

    def _format_blackboard_activity(
        self, entry_type: str, content: Dict[str, Any]
    ) -> str:
        """
        Format blackboard activity content into readable description.

        Parameters:
            entry_type (str): The type of blackboard entry
            content (Dict[str, Any]): The content of the blackboard entry

        Returns:
            str: Formatted activity description
        """
        try:
            if entry_type == "workflow_start":
                query = content.get("original_query", "Unknown query")[:50]
                orchestrator_type = content.get("orchestrator_type", "unknown")
                delegate_count = content.get("delegate_count", 0)
                return f"Started {orchestrator_type} workflow with {delegate_count} delegates: '{query}...'"

            elif entry_type == "task_decomposition":
                sub_tasks = content.get("sub_tasks", [])
                return f"Decomposed task into {len(sub_tasks)} sub-tasks"

            elif entry_type == "task_start":
                task_id = content.get("task_id", "unknown")
                description = content.get("description", "No description")[:50]
                return f"Started task {task_id}: '{description}...'"

            elif entry_type == "task_completion":
                task_id = content.get("task_id", "unknown")
                result = content.get("result", "No result")
                result_preview = (
                    result[:50] if isinstance(result, str) else str(result)[:50]
                )
                return f"Completed task {task_id}: '{result_preview}...'"

            elif entry_type == "workflow_complete":
                response = content.get("consolidated_response", "No response")
                response_preview = (
                    response[:50] if isinstance(response, str) else str(response)[:50]
                )
                return f"Workflow completed: '{response_preview}...'"

            elif entry_type == "hierarchy_update":
                parent = content.get("parent_agent", "unknown")
                sub_agents = content.get("registered_sub_agents", [])
                return f"Registered {len(sub_agents)} sub-agents under {parent}"

            else:
                # Fallback: try to extract any meaningful content
                if isinstance(content, dict) and content:
                    # Look for common fields that might contain useful info
                    for key in ["description", "message", "action", "task", "result"]:
                        if key in content:
                            value = str(content[key])[:50]
                            return f"{key}: '{value}...'"

                    # If no common fields, just show first key-value pair
                    first_key = next(iter(content))
                    first_value = str(content[first_key])[:50]
                    return f"{first_key}: '{first_value}...'"

                return "Activity recorded"

        except Exception as e:
            logger.error(f"Error formatting blackboard activity: {e}")
            return "Activity details unavailable"

    def _initialize_tools_from_toolbox(self, toolbox: Toolbox):
        """
        Initialize tools from a Toolbox using the same logic as add_tool.
        This ensures proper function reference management and tool metadata handling.

        Parameters:
            toolbox (Toolbox): The Toolbox instance to initialize tools from.
        """
        if not isinstance(toolbox, Toolbox):
            raise TypeError(f"Expected a Toolbox, got {type(toolbox)}")

        # Convert to list format for agent use
        self.tools = []

        for meta in toolbox.list_tools():
            # Use _id for function lookup
            tid = str(meta.get("_id"))
            if tid:
                # Resolve the Python callable from the provided Toolbox
                python_fn = toolbox._tools.get(tid)
                if not callable(python_fn):
                    # fallback: perhaps the stored metadata itself packs a .function field?
                    if meta and callable(meta.get("function")):
                        python_fn = meta["function"]

                if callable(python_fn):
                    # build the new entry
                    new_entry = self._build_entry(meta, python_fn)
                    self.tools.append(new_entry)
                else:
                    # Silently skip tools without functions
                    logger.warning(
                        f"Skipping tool with _id {tid} - no callable function found"
                    )
                    continue

    def _build_entry(self, meta: dict, python_fn: Callable) -> dict:
        """
        Construct the flat OpenAI-style schema entry and
        register the python_fn in our internal lookup.

        Parameters:
            meta (dict): Tool metadata containing function information
            python_fn (Callable): The Python function to register

        Returns:
            dict: Formatted tool entry for OpenAI API
        """
        entry = {
            "_id": meta["_id"],  # Use _id as primary identifier
            "name": meta["function"]["name"],
            "description": meta["function"]["description"],
            "parameters": meta["function"]["parameters"],
            # preserve strict-mode flag if present:
            **({"strict": meta.get("strict", True)}),
        }
        # keep a private map of _id → python function,
        # but don't include it when serializing the agent's 'tools' list
        self._tool_functions = getattr(self, "_tool_functions", {})
        self._tool_functions[str(meta["_id"])] = python_fn

        return entry

    def _generate_system_prompt(self):
        """
        Generate the system prompt for the agent.

        This method generates the system prompt for the agent based on the persona and instruction.
        If both are provided, the persona prompt is prepended to the instruction.
        If only the persona is provided, the persona prompt is used.
        If only the instruction is provided, the instruction is used.

        Returns:
            str: The system prompt for the agent.
        """

        # Generate the system prompt or message from the persona and instruction if provided
        if self.persona and self.instruction:
            # Generate the system prompt from the persona and instruction
            persona_prompt = self.persona.generate_system_prompt_input()

            return f"{persona_prompt}\n\n{self.instruction}"

        elif self.persona:
            return f"{self.persona.generate_system_prompt_input()}"
        else:
            return f"{self.instruction}"

    @staticmethod
    def _format_tool(tool_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the tool.

        This method formats the tool.

        Parameters:
            tool_meta (Dict[str, Any]): The tool meta.

        Returns:
            Dict[str, Any]: The formatted tool.
        """

        # Handle different tool metadata structures
        # Case 1: Tool has proper 'function' metadata structure
        if "function" in tool_meta and isinstance(tool_meta["function"], dict):
            function_data = tool_meta["function"]
            name = function_data.get("name", "unknown_tool")
            description = function_data.get("description", "No description available")
            parameters = function_data.get("parameters", [])
        # Case 2: Tool has flat structure (name, description, parameters at top level)
        elif "description" in tool_meta:
            name = tool_meta.get("name", "unknown_tool")
            description = tool_meta.get("description", "No description available")
            parameters = tool_meta.get("parameters", [])
        # Case 3: Tool has corrupted structure with raw function - skip it
        elif "function" in tool_meta and callable(tool_meta["function"]):
            logger.warning(
                f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata"
            )
            return None
        else:
            # Fallback for unknown structure
            logger.warning(
                f"Unknown tool structure for tool with _id {tool_meta.get('_id')}, using fallback"
            )
            name = "unknown_tool"
            description = "Tool metadata corrupted"
            parameters = []

        # Initialize the properties and required parameters
        props, req = {}, []

        # Format the tool parameters
        if isinstance(parameters, list):
            for p in parameters:
                if not isinstance(p, dict):
                    continue

                # Normalize the parameter type for OpenAI API compatibility
                param_type = p.get("type", "string")

                # Clean up type string - remove any extra text like "(required)"
                if isinstance(param_type, str):
                    param_type = param_type.lower().strip()
                    # Remove any parenthetical content
                    if "(" in param_type:
                        param_type = param_type.split("(")[0].strip()

                    # Normalize numeric types to 'number'
                    if param_type in [
                        "float",
                        "decimal",
                        "double",
                        "numeric",
                        "number",
                    ]:
                        param_type = "number"
                    elif param_type in ["int", "integer"]:
                        param_type = "integer"
                    elif param_type in ["bool", "boolean"]:
                        param_type = "boolean"
                    elif param_type in ["str", "text"]:
                        param_type = "string"
                    # Default to string if unrecognized
                    elif param_type not in [
                        "string",
                        "number",
                        "integer",
                        "boolean",
                        "array",
                        "object",
                    ]:
                        param_type = "string"

                props[p["name"]] = {
                    "type": param_type,
                    "description": p.get("description", ""),
                }
                if p.get("required", False):
                    req.append(p["name"])

        # Return the formatted tool
        return {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": req},
        }

    def _load_tools_from_toolbox(self, query: str):
        """
        Load the tools from the toolbox.

        This method loads the tools from the toolbox based on the user input.

        Parameters:
            query (str): The user input.

        Returns:
            List[Dict[str, Any]]: The list of tools.
        """

        # Load the tools from the toolbox
        tools = self.tools.get_most_similar_tools(query)
        formatted_tools = []

        for t in tools:
            formatted_tool = self._format_tool(t)
            # Skip tools that couldn't be formatted (corrupted metadata)
            if formatted_tool is None:
                continue
            # Preserve the _id for reference (convert ObjectId to string for JSON serialization)
            if "_id" in t:
                formatted_tool["_id"] = str(t["_id"])
            formatted_tools.append(formatted_tool)

        return formatted_tools

    def _load_tools_from_memagent(self) -> List[Dict[str, Any]]:
        """
        Load the tools from the memagent and format them
        for OpenAI function-calling (flat schema with name/description/parameters).
        """
        if not self.tools:
            return []

        # Ensure each tool has the required 'type' field and properly formatted parameters
        if isinstance(self.tools, list):
            formatted_tools = []
            for tool in self.tools:
                # Handle different tool metadata structures
                # Case 1: Tool has proper 'function' metadata structure
                if "function" in tool and isinstance(tool["function"], dict):
                    function_data = tool["function"]
                    name = function_data.get("name", "unknown_tool")
                    description = function_data.get(
                        "description", "No description available"
                    )
                    parameters = function_data.get("parameters", [])
                # Case 2: Tool has flat structure (name, description, parameters at top level)
                elif "name" in tool:
                    name = tool.get("name", "unknown_tool")
                    description = tool.get("description", "No description available")
                    parameters = tool.get("parameters", [])
                # Case 3: Tool has corrupted structure - skip it
                else:
                    logger.warning(
                        f"Skipping tool with _id {tool.get('_id')} - missing name and function structure"
                    )
                    continue

                # Create a properly formatted copy of the tool
                formatted_tool = {
                    "type": "function",
                    "name": name,
                    "description": description,
                }

                # Format parameters according to OpenAI's function calling format
                if isinstance(parameters, list) and len(parameters) > 0:
                    properties = {}
                    required = []

                    for param in parameters:
                        if not isinstance(param, dict):
                            continue

                        param_name = param.get("name")
                        if not param_name:
                            continue

                        # Normalize the parameter type for OpenAI API compatibility
                        param_type = param.get("type", "string")

                        # Clean up type string - remove any extra text like "(required)"
                        if isinstance(param_type, str):
                            param_type = param_type.lower().strip()
                            # Remove any parenthetical content
                            if "(" in param_type:
                                param_type = param_type.split("(")[0].strip()

                            # Normalize numeric types to 'number'
                            if param_type in [
                                "float",
                                "decimal",
                                "double",
                                "numeric",
                                "number",
                            ]:
                                param_type = "number"
                            elif param_type in ["int", "integer"]:
                                param_type = "integer"
                            elif param_type in ["bool", "boolean"]:
                                param_type = "boolean"
                            elif param_type in ["str", "text"]:
                                param_type = "string"
                            # Default to string if unrecognized
                            elif param_type not in [
                                "string",
                                "number",
                                "integer",
                                "boolean",
                                "array",
                                "object",
                            ]:
                                param_type = "string"

                        properties[param_name] = {
                            "type": param_type,
                            "description": param.get("description", ""),
                        }
                        if param.get("required", False):
                            required.append(param_name)

                    formatted_tool["parameters"] = {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                else:
                    # If no parameters or parameters is already in the correct format, provide empty schema
                    formatted_tool["parameters"] = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }

                # Preserve _id for reference (convert ObjectId to string for JSON serialization)
                if "_id" in tool:
                    formatted_tool["_id"] = str(tool["_id"])

                formatted_tools.append(formatted_tool)

            return formatted_tools

        return self.tools

    def run(self, query: str, memory_id: str = None, thread_id: str = None) -> str:
        """
        Run the agent with the given query.

        This method runs the agent with the given query. It loads tools from the toolbox if provided,
        generates the system prompt, and runs the agent with the prompt.

        Parameters:
            query (str): The query to run the agent with.
            memory_id (str): The memory id to use.
            thread_id (str): The thread id to use.

        Returns:
            str: The response from the agent.
        """
        logger.info(
            f"AGENT RUN START: Agent {self.agent_id} executing query: {query[:50]}..."
        )
        logger.info(
            f"AGENT RUN DEBUG: Agent has memory_ids: {getattr(self, 'memory_ids', 'NOT SET')}"
        )

        try:
            # Check if we're in multi-agent mode
            if self.is_multi_agent_mode:
                self._initialize_multi_agent_orchestrator()
                return self._multi_agent_orchestrator.execute_multi_agent_workflow(
                    query, memory_id, thread_id
                )

            # 1) Prepare memory and thread IDs
            memory_id, thread_id = self._prepare_memory_and_ids(memory_id, thread_id)

            # 2) Check semantic cache for similar queries (if enabled)
            logger.debug(
                f"Semantic cache instance available: {self.semantic_cache_instance is not None}"
            )
            if self.semantic_cache_instance:
                logger.debug(f"Checking semantic cache for query: {query[:50]}...")
                logger.debug(
                    f"Cache context - session_id: {thread_id}, agent_id: {self.semantic_cache_instance.agent_id}, memory_id: {self.semantic_cache_instance.memory_id}"
                )

                cached_response = self.semantic_cache_instance.get(
                    query=query, session_id=thread_id
                )

                logger.debug(
                    f"Semantic cache result: {'HIT' if cached_response else 'MISS'}"
                )
                if cached_response:
                    # Record user's query in memory before returning cached response
                    self._record_user_query(query, thread_id, memory_id)

                    # Record the cached response as assistant response in conversation memory
                    if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
                        logger.info(
                            f"Recording cached assistant response to memory - memory_id: {memory_id}, thread_id: {thread_id}"
                        )
                        memory_unit = self._generate_conversational_memory_unit(
                            {
                                "role": Role.ASSISTANT,
                                "content": cached_response,
                                "timestamp": datetime.now().isoformat(),
                                "thread_id": thread_id,
                                "memory_id": memory_id,
                            }
                        )
                        logger.debug(
                            f"Created cached response memory unit: {memory_unit}"
                        )
                    else:
                        logger.warning(
                            f"CONVERSATION_MEMORY not in active memory types: {self.active_memory_types}"
                        )

                    return cached_response

            # 3) Build the complete augmented query with context
            augmented_query = self._build_augmented_query(query, memory_id)

            # 4) Generate system prompt and create initial messages
            system_prompt = self._generate_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query},
            ]

            # 5) Log debug information
            self._log_prompt_debug_info(system_prompt, augmented_query)

            # 6) Record user's query in memory
            self._record_user_query(query, thread_id, memory_id)

            # 7) Execute main interaction loop
            final_response = self._execute_main_loop(
                messages, query, memory_id, thread_id
            )

            # 8) Cache the response for future similar queries (if semantic cache enabled)
            if self.semantic_cache_instance:
                self.semantic_cache_instance.set(
                    query=query,
                    response=final_response,
                    session_id=thread_id,
                    metadata={
                        "memory_id": memory_id,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                    },
                )
                logger.debug(
                    f"SEMANTIC CACHE STORE: Cached response for query: {query[:50]}..."
                )

            return final_response

        except Exception as e:
            logger.error(f"Error running agent {self.agent_id}: {e}")
            return f"An error occurred while running the agent: {e}"

    def _prepare_memory_and_ids(
        self, memory_id: str, thread_id: str
    ) -> tuple[str, str]:
        """
        Prepare and validate memory_id and thread_id for the agent run.

        Parameters:
            memory_id (str): The memory id to use (can be None).
            thread_id (str): The thread id to use (can be None).

        Returns:
            tuple[str, str]: Validated (memory_id, thread_id) pair.
        """
        # 1) Ensure memory_id
        if memory_id is None:
            if self.memory_ids and len(self.memory_ids) > 0:
                # Use the most recent memory_id if none specified
                memory_id = self.memory_ids[-1]
            else:
                # Create a new memory_id if none exist using MongoDB ObjectId
                memory_id = str(ObjectId())
        elif memory_id not in (self.memory_ids or []):
            # Create a new memory_id if specified one doesn't exist using MongoDB ObjectId
            memory_id = str(ObjectId())

        # persist to agent if needed
        if self.agent_id and memory_id not in (self.memory_ids or []):
            self.memory_ids.append(memory_id)
            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(
                    self.agent_id, self.memory_ids
                )

        # 2) Ensure thread_id with persistence
        if thread_id is None:
            # Check if we already have a current thread_id stored
            if self._current_thread_id:
                # Reuse existing thread_id to maintain thread continuity
                thread_id = self._current_thread_id
            else:
                # Generate new thread_id and store it for future reuse
                thread_id = str(ObjectId())
                self._current_thread_id = thread_id
        else:
            # Explicit thread_id provided - store it as current for future runs
            self._current_thread_id = thread_id

        return memory_id, thread_id

    def _build_augmented_query(self, query: str, memory_id: str) -> str:
        """
        Build the complete augmented query with all context and memory.

        Parameters:
            query (str): The original user query.
            memory_id (str): The memory ID to use for context.

        Returns:
            str: The complete augmented query with all context.
        """
        # 3) Augment the user query with the query
        augmented_query = (
            f"This is the query to be answered or key objective to be achieved: {query}"
        )

        # Get the prompt for the memory types from the agent's active memory types
        memory_types = self.active_memory_types.copy()

        # Check if this agent has shared memory context available and include it in memory types
        try:
            from .coordination.shared_memory.shared_memory import SharedMemory

            shared_memory = SharedMemory(self.memory_provider)
            active_session = shared_memory.find_active_session_for_agent(self.agent_id)
            if active_session:
                memory_types.append(MemoryType.SHARED_MEMORY)
        except Exception as e:
            logger.error(
                f"Error checking for shared memory when building memory types: {e}"
            )

        cwm_prompt = CWM.get_prompt_from_memory_types(memory_types)
        augmented_query += f"\n\n{cwm_prompt}"

        # Check for shared memory context (multi-agent coordination)
        shared_memory_context = self._check_for_shared_memory_context()
        if shared_memory_context:
            augmented_query += shared_memory_context

        # 4) Load and integrate workflow memory
        augmented_query = self._add_workflow_memory(augmented_query, query)

        # 5) Add conversation history
        augmented_query = self._add_conversation_history(augmented_query, memory_id)

        # 6) Add memory summaries
        augmented_query = self._add_summaries(augmented_query, query, memory_id)

        # 7) Add relevant memory units
        augmented_query = self._add_relevant_memory_units(
            augmented_query, query, memory_id
        )

        # 8) Add long-term knowledge
        augmented_query = self._add_long_term_knowledge(augmented_query, query)

        # Reinforce the user query and the objective to end the prompt construction
        augmented_query += (
            f"\n\nRemember the user query to address and objective is: {query}"
        )

        return augmented_query

    def _add_workflow_memory(self, augmented_query: str, query: str) -> str:
        """Add workflow memory context to the augmented query."""
        if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            try:
                # Retrieve relevant workflows based on the query
                relevant_workflows = Workflow.retrieve_workflows_by_query(
                    query, self.memory_provider
                )

                if relevant_workflows and len(relevant_workflows) > 0:
                    workflow_context = (
                        "\n\n---------THIS IS YOUR WORFLOW MEMORY---------\n"
                    )
                    workflow_context += "\n\nPrevious workflow executions that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
                    for workflow in relevant_workflows:
                        # Add workflow details including outcome to guide execution
                        workflow_context += (
                            f"- Workflow '{workflow.name}': {workflow.description}\n"
                        )
                        workflow_context += f"  Outcome: {workflow.outcome.value}\n"
                        if workflow.outcome == WorkflowOutcome.FAILURE:
                            workflow_context += f"  Error: {workflow.steps.get('error', 'Unknown error')}\n"

                        # Add detailed step information
                        workflow_context += f"  Steps taken: {len(workflow.steps)}\n"
                        for step_name, step_data in workflow.steps.items():
                            workflow_context += f"    Step: {step_name}\n"
                            workflow_context += (
                                f"      Function: {step_data.get('_id', 'Unknown')}\n"
                            )
                            workflow_context += (
                                f"      Arguments: {step_data.get('arguments', {})}\n"
                            )
                            workflow_context += f"      Result: {step_data.get('result', 'No result')}\n"
                            if step_data.get("error"):
                                workflow_context += (
                                    f"      Error: {step_data.get('error')}\n"
                                )
                            workflow_context += f"      Timestamp: {step_data.get('timestamp', 'Unknown')}\n"
                        workflow_context += "\n"

                    augmented_query += workflow_context

            except Exception as e:
                logger.error(f"Error loading workflow memory: {str(e)}")
                # Continue execution even if workflow memory loading fails

        return augmented_query

    def _add_conversation_history(self, augmented_query: str, memory_id: str) -> str:
        """Add conversation history to the augmented query."""
        # Write a conversational history prompt
        conversational_history_prompt = (
            "---------THIS IS YOUR CONVERSATIONAL HISTORY MEMORY---------\n"
        )
        conversational_history_prompt += "\n\nPrevious conversations that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
        augmented_query += conversational_history_prompt

        # 6) Append past conversation history
        conversation_history = self.load_conversation_history(memory_id)
        logger.debug(
            f"Loaded {len(conversation_history) if conversation_history else 0} conversation history items for memory_id: {memory_id}"
        )

        if conversation_history:
            for conv in conversation_history:
                augmented_query += f"\n\n{conv['role']}: {conv['content']}. "

        return augmented_query

    def _add_summaries(self, augmented_query: str, query: str, memory_id: str) -> str:
        """Add memory summaries to the augmented query."""
        if MemoryType.SUMMARIES in self.active_memory_types:
            try:
                # Retrieve relevant summaries based on the query
                summaries = self._load_relevant_memory_units(
                    query, MemoryType.SUMMARIES, memory_id, limit=3
                )

                if summaries and len(summaries) > 0:
                    summaries_context = (
                        "\n\n---------THIS IS YOUR MEMORY SUMMARIES---------\n"
                    )
                    summaries_context += "\n\nCompressed summaries of past interactions that provide broader context about user preferences, patterns, and historical conversations:\n"

                    for summary in summaries:
                        # Format summary content based on the data structure
                        if isinstance(summary, dict):
                            summary_content = summary.get(
                                "content", summary.get("summary", str(summary))
                            )
                            timestamp = summary.get(
                                "timestamp", summary.get("created_at", "")
                            )
                            memory_count = summary.get("memory_count", "unknown")

                            if timestamp:
                                summaries_context += f"\n\nSummary (from {timestamp}, {memory_count} memories): {summary_content}"
                            else:
                                summaries_context += f"\n\nSummary ({memory_count} memories): {summary_content}"
                        else:
                            summaries_context += f"\n\nSummary: {summary}"

                    augmented_query += summaries_context

            except Exception as e:
                logger.error(f"Error loading summaries: {str(e)}")
                # Continue execution even if summaries loading fails

        return augmented_query

    def _add_relevant_memory_units(
        self, augmented_query: str, query: str, memory_id: str
    ) -> str:
        """Add relevant memory units to the augmented query."""
        # Write relevant memory units prompt
        relevant_memory_units_prompt = (
            "---------THIS IS YOUR RELEVANT MEMORY COMPONENTS---------\n"
        )
        relevant_memory_units_prompt += "\n\nRelevant memory units that may be relevant to ensure you are on the right track, use this information to guide your execution:\n"
        augmented_query += relevant_memory_units_prompt

        # Add conversation memory units
        if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
            for mem in self._load_relevant_memory_units(
                query, MemoryType.CONVERSATION_MEMORY, memory_id, limit=5
            ):
                augmented_query += f"\n\n{mem['role']}: {mem['content']}. "

        return augmented_query

    def _add_long_term_knowledge(self, augmented_query: str, query: str) -> str:
        """Add long-term knowledge to the augmented query."""
        # Add long-term knowledge to the prompt if agent has long_term_memory_ids
        if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
            # Write long term memory prompt
            long_term_memory_prompt = (
                "---------THIS IS YOUR LONG-TERM KNOWLEDGE---------\n"
            )
            long_term_memory_prompt += "\n\nRelevant knowledge from your long-term memory that may help answer the query:\n"
            augmented_query += long_term_memory_prompt

            # Import knowledge base and load relevant knowledge
            kb = KnowledgeBase(self.memory_provider)

            # First, try to retrieve knowledge semantically similar to the query
            semantic_entries = kb.retrieve_knowledge_by_query(query, limit=3)

            # Add semantic matches first
            if semantic_entries:
                augmented_query += "\n\n--- Semantically Relevant Knowledge ---\n"
                for entry in semantic_entries:
                    augmented_query += f"\n\nKnowledge: {entry.get('content', '')}\n"
                    augmented_query += (
                        f"Namespace: {entry.get('namespace', 'general')}\n"
                    )

            # For each memory ID, retrieve and add relevant knowledge
            augmented_query += "\n\n--- Agent's Associated Knowledge ---\n"
            for memory_id in self.long_term_memory_ids:
                knowledge_entries = kb.retrieve_knowledge(memory_id)
                for entry in knowledge_entries:
                    # Skip entries already included in semantic search
                    if entry in semantic_entries:
                        continue
                    augmented_query += f"\n\nKnowledge: {entry.get('content', '')}\n"
                    augmented_query += (
                        f"Namespace: {entry.get('namespace', 'general')}\n"
                    )

        return augmented_query

    def _log_prompt_debug_info(self, system_prompt: str, augmented_query: str):
        """Log debugging information about prompts safely."""
        # Log prompt information for debugging (removed unsafe file writes)
        logger.debug(f"System prompt prepared for agent {self.agent_id}")
        logger.debug(f"Augmented query length: {len(augmented_query)} characters")

        # Multi-agent logging: Only call if actually in multi-agent mode
        if (
            self.is_multi_agent_mode
            or len(self.delegates) > 0
            or self._has_shared_memory_session()
        ):
            logger.info(
                f"ABOUT TO CALL _export_multi_agent_logs for agent {self.agent_id}"
            )
            self._export_multi_agent_logs(system_prompt, augmented_query)
            logger.info(
                f"COMPLETED _export_multi_agent_logs call for agent {self.agent_id}"
            )
        else:
            logger.debug(
                f"Skipping multi-agent logging for single-agent mode (agent {self.agent_id})"
            )

    def _record_user_query(self, query: str, thread_id: str, memory_id: str):
        """Record the user's query in conversational memory."""
        if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
            logger.info(
                f"Recording user query to memory - memory_id: {memory_id}, thread_id: {thread_id}"
            )
            memory_unit = self._generate_conversational_memory_unit(
                {
                    "role": Role.USER,
                    "content": query,
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": thread_id,
                    "memory_id": memory_id,
                }
            )
            # Remove the embedding from the memory unit for logging purposes.
            memory_unit.embedding = None
            logger.info(f"Created user memory unit: {memory_unit}")
        else:
            logger.warning(
                f"CONVERSATION_MEMORY not in active memory types for user query: {self.active_memory_types}"
            )

    def _execute_main_loop(
        self, messages: List[Dict], query: str, memory_id: str, thread_id: str
    ) -> str:
        """
        Execute the main agent interaction loop with tool calls and responses.

        Parameters:
            messages (List[Dict]): The conversation messages.
            query (str): The original user query.
            memory_id (str): The memory ID.
            thread_id (str): The thread ID.

        Returns:
            str: The final response from the agent.
        """
        tool_choice = "auto"

        # 10) Main loop
        for step in range(self.max_steps):
            # a) Build function schema list
            tool_metas, tool_choice = self._prepare_tools(query, tool_choice)

            # b) Call the LLM API using the configured model
            response = self.model.client.responses.create(
                model=self.model.model,  # Use the model name from the configured model instance
                input=messages,
                tools=tool_metas,
                tool_choice=tool_choice,
            )

            # c) See if model called a function
            tool_calls = [
                o
                for o in response.output
                if getattr(o, "type", None) == "function_call"
            ]

            if tool_calls:
                # Handle tool execution
                messages = self._handle_tool_calls(
                    tool_calls, messages, query, memory_id, tool_metas
                )
                continue

            # h) No function calls → final answer
            if response.output_text:
                return self._finalize_response(
                    response.output_text, messages, thread_id, memory_id
                )

        # 11) If we never returned…
        raise RuntimeError("Max steps exceeded without reaching a final answer.")

    def _prepare_tools(
        self, query: str, current_tool_choice: str
    ) -> tuple[List[Dict], str]:
        """
        Prepare the tools metadata for the current interaction.

        Parameters:
            query (str): The user query for tool loading context.
            current_tool_choice (str): The current tool choice setting.

        Returns:
            tuple[List[Dict], str]: (tool_metas, tool_choice)
        """
        if self.tools:
            if isinstance(self.tools, Toolbox):
                if self.tool_access == "global":
                    tool_metas = self._load_tools_from_toolbox(query)
                else:
                    # For private access, convert Toolbox tools to the expected format
                    tool_metas = []
                    for tool_meta in self.tools.list_tools():
                        formatted_tool = self._format_tool(tool_meta)
                        if formatted_tool is not None:
                            tool_metas.append(formatted_tool)
            else:
                tool_metas = self._load_tools_from_memagent()

            if not tool_metas:
                return [], "none"
            return tool_metas, current_tool_choice
        else:
            return [], "none"

    def _handle_tool_calls(
        self,
        tool_calls: List,
        messages: List[Dict],
        query: str,
        memory_id: str,
        tool_metas: List[Dict],
    ) -> List[Dict]:
        """
        Handle execution of tool function calls.

        Parameters:
            tool_calls (List): The tool calls from the LLM.
            messages (List[Dict]): The current conversation messages.
            query (str): The original user query.
            memory_id (str): The memory ID.
            tool_metas (List[Dict]): The available tool metadata.

        Returns:
            List[Dict]: Updated messages with tool results.
        """
        # Create a workflow to track all tool calls
        workflow = Workflow(
            name=f"Tool Execution: {len(tool_calls)} steps",
            description=f"Execution of {len(tool_calls)} tools",
            memory_id=memory_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            outcome=WorkflowOutcome.SUCCESS,
            user_query=query,
        )

        for call in tool_calls:
            name = call.name
            error_message = None  # Initialize error message for this tool call

            try:
                args = json.loads(call.arguments)
                messages.append(
                    {
                        "type": "function_call",
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                )
            except Exception:
                args = {}

            # Execute the tool function
            result, error_message, workflow_outcome = self._execute_single_tool(
                call, name, args, tool_metas
            )

            # Append the result to messages
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": str(result),
                }
            )

            # Add step to workflow
            tool_entry = next((meta for meta in tool_metas if meta["name"] == name), {})
            workflow.add_step(
                f"Step {len(workflow.steps) + 1}: {name}",
                {
                    "_id": str(tool_entry.get("_id")) if tool_entry else None,
                    "arguments": args,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                    "error": error_message,
                },
            )

            # Update workflow outcome if any step failed
            if workflow_outcome == WorkflowOutcome.FAILURE:
                workflow.outcome = WorkflowOutcome.FAILURE

        # Store the complete workflow with all steps
        if MemoryType.WORKFLOW_MEMORY in self.active_memory_types:
            try:
                workflow.store_workflow(self.memory_provider)
            except Exception as e:
                logger.error(f"Error storing workflow: {str(e)}")
                # Continue execution even if workflow storage fails

        return messages

    def _execute_single_tool(
        self, call, name: str, args: Dict, tool_metas: List[Dict]
    ) -> tuple[str, str, WorkflowOutcome]:
        """
        Execute a single tool function call.

        Parameters:
            call: The tool call object from the LLM.
            name (str): The name of the tool to execute.
            args (Dict): The arguments for the tool.
            tool_metas (List[Dict]): The available tool metadata.

        Returns:
            tuple[str, str, WorkflowOutcome]: (result, error_message, outcome)
        """
        # d) Lookup the Python function backing this call
        fn = None
        entry = None

        if isinstance(self.tools, Toolbox):
            # For Toolbox, search in the formatted tools we just created
            for meta in tool_metas:
                if meta["name"] == name:
                    entry = meta
                    # Use the Toolbox's get_function_by_id method
                    fn = self.tools.get_function_by_id(str(meta.get("_id")))
                    break
        elif isinstance(self.tools, list):
            formatted = self._load_tools_from_memagent()
            for t in formatted:
                if t["name"] == name:
                    entry = t
                    for orig in self.tools:
                        # Handle different tool structures
                        orig_name = None
                        if "name" in orig:
                            orig_name = orig["name"]
                        elif "function" in orig and isinstance(orig["function"], dict):
                            orig_name = orig["function"].get("name")

                        if orig_name == name:
                            fn = getattr(self, "_tool_functions", {}).get(
                                str(orig.get("_id"))
                            )
                            break
                    break

        if not entry:
            result = f"Error: Tool '{name}' not found in available tools."
            return result, result, WorkflowOutcome.FAILURE
        elif not callable(fn):
            logger.warning(
                f"Tool '{name}' found but function is not callable. Tool ID: {entry.get('_id')}"
            )
            result = f"Sorry, the tool '{name}' is currently unavailable. It exists in the system but its implementation function is not properly registered."
            return result, result, WorkflowOutcome.FAILURE
        else:
            try:
                # e) Execute and append the function's result
                result = fn(**args)
                return result, None, WorkflowOutcome.SUCCESS
            except Exception as e:
                logger.error(f"Error executing tool {name}: {str(e)}")
                result = f"Error executing tool {name}: {str(e)}"
                return result, str(e), WorkflowOutcome.FAILURE

    def _finalize_response(
        self,
        response_text: str,
        messages: List[Dict],
        thread_id: str,
        memory_id: str,
    ) -> str:
        """
        Finalize the agent response by recording it in memory and returning it.

        Parameters:
            response_text (str): The final response text from the LLM.
            messages (List[Dict]): The conversation messages.
            thread_id (str): The thread ID.
            memory_id (str): The memory ID.

        Returns:
            str: The final response text.
        """
        # Record into memory
        if MemoryType.CONVERSATION_MEMORY in self.active_memory_types:
            logger.info(
                f"Recording assistant response to memory - memory_id: {memory_id}, thread_id: {thread_id}"
            )
            memory_unit = self._generate_conversational_memory_unit(
                {
                    "role": Role.ASSISTANT,
                    "content": response_text,
                    "timestamp": datetime.now().isoformat(),
                    "thread_id": thread_id,
                    "memory_id": memory_id,
                }
            )
            # logger.info(f"Created memory unit: {memory_unit}")
        else:
            logger.warning(
                f"CONVERSATION_MEMORY not in active memory types: {self.active_memory_types}"
            )

        # Append as assistant turn and return
        messages.append({"role": "assistant", "content": response_text})
        return response_text

    def load_conversation_history(self, memory_id: str = None):
        """
        Load the conversation history.

        This method loads the conversation history based on the memory id.

        Parameters:
            memory_id (str): The memory id.

        Returns:
            List[ConversationMemoryUnit]: The conversation history.
        """

        # If the memory id is not provided and we have memory_ids, use the most recent one
        if memory_id is None and self.memory_ids:
            memory_id = self.memory_ids[-1]

        return self.memory_unit.retrieve_memory_units_by_memory_id(
            memory_id, MemoryType.CONVERSATION_MEMORY
        )

    def start_new_thread(self):
        """
        Start a new thread by clearing the current thread ID.

        The next run() call will generate a new thread_id and subsequent
        calls will reuse that new ID, maintaining thread continuity.

        Returns:
            str: The new thread_id that will be generated on next run()
        """
        self._current_thread_id = None
        return "New thread will start on next run()"

    def _load_relevant_memory_units(
        self, query: str, memory_type: MemoryType, memory_id: str = None, limit: int = 5
    ):
        """
        Load the relevant memory units.

        This method loads the relevant memory units based on the query.

        Parameters:
            query (str): The user input.
            memory_id (str): The memory id.
            limit (int): The limit of the memory units to return.

        Returns:
            List[ConversationMemoryUnit]: The conversation history.
        """

        # If the memory id is not provided and we have memory_ids, use the most recent one
        if memory_id is None and self.memory_ids:
            memory_id = self.memory_ids[-1]

        # Load the relevant memory units from the memory provider
        relevant_memory_units = self.memory_unit.retrieve_memory_units_by_query(
            query, memory_id=memory_id, memory_type=memory_type, limit=limit
        )

        # Return the relevant memory units
        return relevant_memory_units

    def _generate_conversational_memory_unit(
        self, content: dict
    ) -> ConversationMemoryUnit:
        """
        Generate the conversational memory unit.

        This method generates the conversational memory unit based on the content.

        Parameters:
            content (dict): The content of the memory unit.

        Returns:
            str: The conversational memory unit.
        """

        # Generate the conversational memory unit
        memory_unit = self.memory_unit.generate_memory_unit(content)
        return memory_unit

    def save(self):
        """
        Store the memagent in the memory provider.

        This method stores the memagent in the memory provider.
        """
        # Convert tools to serializable format if it's a Toolbox object
        tools_to_save = self.tools
        if isinstance(self.tools, Toolbox):
            # Convert Toolbox to list of tool metadata for serialization
            tools_to_save = []
            for tool_meta in self.tools.list_tools():
                # Check if 'function' field contains metadata (dict) or raw function (callable)
                function_field = tool_meta.get("function", {})

                if callable(function_field):
                    # If it's a raw function, skip this tool as it's improperly stored
                    logger.warning(
                        f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata"
                    )
                    continue
                elif isinstance(function_field, dict):
                    # If it's proper metadata, extract serializable tool information
                    serializable_tool = {
                        "_id": tool_meta.get("_id"),
                        "function": {
                            "name": function_field.get("name"),
                            "description": function_field.get("description"),
                            "parameters": function_field.get("parameters", []),
                        },
                        "type": tool_meta.get("type", "function"),
                    }
                    tools_to_save.append(serializable_tool)
                else:
                    # If it's neither dict nor callable, skip with warning
                    logger.warning(
                        f"Skipping tool with _id {tool_meta.get('_id')} - unknown function field type: {type(function_field)}"
                    )
                    continue

        # Convert delegates to agent IDs for persistence
        delegate_ids = []
        if hasattr(self, "delegates") and self.delegates:
            for delegate in self.delegates:
                if hasattr(delegate, "agent_id") and delegate.agent_id:
                    delegate_ids.append(delegate.agent_id)
                    # Ensure delegate is saved before saving the reference
                    try:
                        delegate.save()
                    except Exception as e:
                        logger.warning(
                            f"Failed to save delegate {delegate.agent_id}: {e}"
                        )

        # Ensure embedding_config is properly set before saving
        if self.embedding_config is None:
            logger.warning("Embedding config is None, attempting to refresh...")
            self.refresh_embedding_config()

        # Prepare semantic_cache_config for serialization
        semantic_cache_config_to_save = None
        if hasattr(self, "semantic_cache_instance") and self.semantic_cache_instance:
            # Convert SemanticCacheConfig to dict for storage
            if hasattr(self.semantic_cache_instance, "config"):
                semantic_cache_config_to_save = (
                    self.semantic_cache_instance.config.__dict__.copy()
                )
                # Convert enum to string for JSON serialization
                if "scope" in semantic_cache_config_to_save:
                    semantic_cache_config_to_save[
                        "scope"
                    ] = semantic_cache_config_to_save["scope"].value

        # Create a new MemAgentModel with the current object's attributes
        memagent_to_save = MemAgentModel(
            llm_config=(
                self.model.get_config() if self.model else None
            ),  # ✅ Correctly saves model config
            instruction=self.instruction,
            application_mode=self._get_application_mode_value(),
            memory_types=[mt.value for mt in self.active_memory_types],
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,  # This will be removed in the provider
            persona=self.persona,
            tools=tools_to_save,
            long_term_memory_ids=getattr(self, "long_term_memory_ids", None),
            delegates=delegate_ids if delegate_ids else None,
            embedding_config=self.embedding_config,
            semantic_cache=bool(getattr(self, "semantic_cache_instance", None)),
            semantic_cache_config=semantic_cache_config_to_save,
        )

        # Check if this is a new agent or an existing one
        if self.agent_id is None:
            # New agent - store it
            saved_memagent = self.memory_provider.store_memagent(memagent_to_save)
            # Update the agent_id to the MongoDB _id that was generated
            self.agent_id = str(saved_memagent["_id"])

            # Update semantic cache with the newly generated agent_id
            if self.semantic_cache_instance:
                self.semantic_cache_instance.agent_id = self.agent_id
                if self.memory_ids:
                    self.semantic_cache_instance.memory_id = self.memory_ids[0]
                logger.debug(f"Updated SemanticCache with agent_id={self.agent_id}")
        else:
            # Existing agent - update it
            saved_memagent = self.memory_provider.update_memagent(memagent_to_save)
            # Ensure we add the _id for logging consistency
            if "_id" not in saved_memagent:
                saved_memagent["_id"] = self.agent_id

        # Log the saved memagent
        logger.info(f"Memagent {self.agent_id} saved in the memory provider")
        # Log the details and attributes of the saved memagent
        # Show the logs as a json object
        logger.info(json.dumps(saved_memagent, indent=4, default=str))

        return self

    def _extract_embedding_config(self) -> Optional[Dict[str, Any]]:
        """Extract embedding configuration with proper priority: direct > memory_provider > global."""
        logger.debug("Extracting embedding configuration")

        # Priority 1: Direct MemAgent parameters
        if getattr(self, "_direct_embedding_provider", None) is not None:
            result = {
                "provider": self._direct_embedding_provider,
                "config": getattr(self, "_direct_embedding_config", {}) or {},
            }
            logger.debug(f"Using direct embedding config: {result}")
            return result

        # Priority 2: Memory provider configuration (MongoDB provider)
        logger.debug(
            f"Checking memory provider for embedding config: {type(self.memory_provider).__name__}"
        )

        try:
            # Check if memory provider has the processed embedding provider (MongoDBProvider case)
            if hasattr(self.memory_provider, "_embedding_provider"):
                processed_provider = getattr(
                    self.memory_provider, "_embedding_provider"
                )
                logger.info(
                    f" PRIORITY 2: Processed provider type: {type(processed_provider)}"
                )
                logger.info(
                    f" PRIORITY 2: Processed provider is None: {processed_provider is None}"
                )

                if processed_provider is not None:
                    logger.info(
                        f" PRIORITY 2: Provider has get_provider_info: {hasattr(processed_provider, 'get_provider_info')}"
                    )

                    if hasattr(processed_provider, "get_provider_info"):
                        # It's an EmbeddingManager instance
                        try:
                            provider_type = getattr(
                                processed_provider, "provider_type", None
                            )
                            config = getattr(processed_provider, "config", {}) or {}
                            logger.info(f" PRIORITY 2: Provider type: {provider_type}")
                            logger.info(f" PRIORITY 2: Provider config: {config}")

                            result = {
                                "provider": (
                                    provider_type.value
                                    if hasattr(provider_type, "value")
                                    else str(provider_type)
                                ),
                                "config": config,
                            }
                            logger.info(
                                f" PRIORITY 2 SUCCESS: Extracted from processed embedding provider: {result}"
                            )
                            return result
                        except Exception as e:
                            logger.error(
                                f" PRIORITY 2 ERROR: Failed to extract from processed EmbeddingManager: {e}"
                            )
                            import traceback

                            logger.error(
                                f" PRIORITY 2 TRACEBACK: {traceback.format_exc()}"
                            )
                else:
                    logger.info(" PRIORITY 2 SKIP: Processed provider is None")
            else:
                logger.info(
                    " PRIORITY 2 SKIP: Memory provider has no _embedding_provider"
                )

            # Fallback to original config-based approach
            logger.info(f" PRIORITY 2 FALLBACK: Checking config-based approach")
            logger.info(
                f" PRIORITY 2 FALLBACK: Has config: {hasattr(self.memory_provider, 'config')}"
            )

            if (
                hasattr(self.memory_provider, "config")
                and self.memory_provider.config is not None
                and hasattr(self.memory_provider.config, "embedding_provider")
            ):
                provider = self.memory_provider.config.embedding_provider
                config = (
                    getattr(self.memory_provider.config, "embedding_config", {}) or {}
                )
                logger.info(f" PRIORITY 2 FALLBACK: Config provider: {provider}")
                logger.info(f" PRIORITY 2 FALLBACK: Config config: {config}")

                if provider is not None:
                    if isinstance(provider, str):
                        result = {"provider": provider, "config": config}
                        logger.info(
                            f" PRIORITY 2 FALLBACK SUCCESS: Extracted memory provider string config: {result}"
                        )
                        return result
                    elif hasattr(provider, "get_provider_info"):
                        try:
                            result = {
                                "provider": (
                                    provider.provider_type.value
                                    if hasattr(provider, "provider_type")
                                    else str(provider)
                                ),
                                "config": getattr(provider, "config", {}) or {},
                            }
                            logger.info(
                                f" PRIORITY 2 FALLBACK SUCCESS: Extracted memory provider manager config: {result}"
                            )
                            return result
                        except Exception as e:
                            logger.error(
                                f" PRIORITY 2 FALLBACK ERROR: Failed to extract from EmbeddingManager: {e}"
                            )
                else:
                    logger.info(
                        " PRIORITY 2 FALLBACK SKIP: Memory provider embedding_provider is None"
                    )
            else:
                logger.info(f" PRIORITY 2 FALLBACK SKIP: Config check failed")
        except Exception as e:
            logger.error(
                f" PRIORITY 2 ERROR: Error extracting from memory provider: {e}"
            )
            import traceback

            logger.error(f" PRIORITY 2 TRACEBACK: {traceback.format_exc()}")

        # Priority 3: Global embedding manager (with better error handling)
        logger.info(" PRIORITY 3: Checking global embedding manager")
        try:
            from .embeddings import get_embedding_manager

            manager = get_embedding_manager()
            logger.info(f" PRIORITY 3: Global manager type: {type(manager)}")
            logger.info(f" PRIORITY 3: Global manager is None: {manager is None}")

            if (
                manager
                and hasattr(manager, "provider_type")
                and hasattr(manager, "config")
            ):
                result = {
                    "provider": manager.provider_type.value,
                    "config": manager.config or {},
                }
                logger.info(
                    f" PRIORITY 3 SUCCESS: Extracted global embedding config: {result}"
                )
                return result
            else:
                logger.warning(
                    " PRIORITY 3 SKIP: Global embedding manager is invalid or incomplete"
                )
        except Exception as e:
            logger.error(
                f" PRIORITY 3 ERROR: Failed to get global embedding manager: {e}"
            )
            import traceback

            logger.error(f" PRIORITY 3 TRACEBACK: {traceback.format_exc()}")

        logger.error(
            " EXTRACTION FAILED: Could not extract embedding config from any source"
        )
        return None

    def get_stored_embedding_config(self) -> Optional[Dict[str, Any]]:
        """Get the embedding configuration that was stored with this agent."""
        return getattr(self, "embedding_config", None)

    def validate_embedding_config(self) -> Dict[str, Any]:
        """Validate and debug the embedding configuration extraction."""
        validation_result = {
            "current_embedding_config": getattr(self, "embedding_config", None),
            "memory_provider_type": type(self.memory_provider).__name__,
            "has_memory_provider_config": hasattr(self.memory_provider, "config"),
            "direct_provider": getattr(self, "_direct_embedding_provider", None),
            "direct_config": getattr(self, "_direct_embedding_config", None),
        }

        if hasattr(self.memory_provider, "config"):
            config = self.memory_provider.config
            validation_result.update(
                {
                    "memory_provider_config_type": type(config).__name__,
                    "memory_provider_embedding_provider": getattr(
                        config, "embedding_provider", "NOT_FOUND"
                    ),
                    "memory_provider_embedding_config": getattr(
                        config, "embedding_config", "NOT_FOUND"
                    ),
                }
            )

        # Try to re-extract config to see if it works now
        try:
            re_extracted = self._extract_embedding_config()
            validation_result["re_extracted_config"] = re_extracted
        except Exception as e:
            validation_result["re_extraction_error"] = str(e)

        return validation_result

    def refresh_embedding_config(self) -> bool:
        """Re-extract and update the embedding configuration."""
        try:
            new_config = self._extract_embedding_config()
            if new_config is not None:
                self.embedding_config = new_config
                logger.info(f"Refreshed embedding config: {new_config}")
                return True
            else:
                logger.warning(
                    "Failed to refresh embedding config - extraction returned None"
                )
                return False
        except Exception as e:
            logger.error(f"Error refreshing embedding config: {e}")
            return False

    def update(
        self,
        instruction: Optional[str] = None,
        max_steps: Optional[int] = None,
        memory_ids: Optional[List[str]] = None,
        persona: Optional[Persona] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Update the memagent in the memory provider.

        This method updates various parts of the memagent in the memory provider.

        Parameters:
            instruction (str): The instruction of the memagent.
            max_steps (int): The maximum steps of the memagent.
            memory_ids (List[str]): The memory ids of the memagent.
            persona (Persona): The persona of the memagent.
            tools (List[Dict[str, Any]]): The tools of the memagent.

        Returns:
            MemAgent: The updated memagent.
        """

        # Update the memagent in the memory provider
        if tools:
            self.tools = tools

        if persona:
            self.persona = persona

        if instruction:
            self.instruction = instruction

        if max_steps:
            self.max_steps = max_steps

        if memory_ids:
            self.memory_ids = memory_ids

        # Convert tools to serializable format if it's a Toolbox object
        tools_to_update = self.tools
        if isinstance(self.tools, Toolbox):
            # Convert Toolbox to list of tool metadata for serialization
            tools_to_update = []
            for tool_meta in self.tools.list_tools():
                # Extract serializable tool information
                # Check if 'function' field contains metadata (dict) or raw function (callable)
                function_field = tool_meta.get("function", {})

                if callable(function_field):
                    # If it's a raw function, skip this tool as it's improperly stored
                    logger.warning(
                        f"Skipping tool with _id {tool_meta.get('_id')} - contains raw function instead of metadata"
                    )
                    continue
                elif isinstance(function_field, dict):
                    # If it's proper metadata, extract serializable tool information
                    serializable_tool = {
                        "_id": tool_meta.get("_id"),
                        "function": {
                            "name": function_field.get("name"),
                            "description": function_field.get("description"),
                            "parameters": function_field.get("parameters", []),
                        },
                        "type": tool_meta.get("type", "function"),
                    }
                    tools_to_update.append(serializable_tool)
                else:
                    # If it's neither dict nor callable, skip with warning
                    logger.warning(
                        f"Skipping tool with _id {tool_meta.get('_id')} - unknown function field type: {type(function_field)}"
                    )
                    continue

        # Convert delegates to agent IDs for persistence
        delegate_ids = []
        if hasattr(self, "delegates") and self.delegates:
            for delegate in self.delegates:
                if hasattr(delegate, "agent_id") and delegate.agent_id:
                    delegate_ids.append(delegate.agent_id)
                    # Ensure delegate is saved before saving the reference
                    try:
                        delegate.save()
                    except Exception as e:
                        logger.warning(
                            f"Failed to save delegate {delegate.agent_id}: {e}"
                        )

        memagent_to_update = MemAgentModel(
            llm_config=(
                self.model.get_config() if self.model else None
            ),  # ✅ Correctly updates model config
            instruction=self.instruction,
            application_mode=self._get_application_mode_value(),
            memory_types=[mt.value for mt in self.active_memory_types],
            max_steps=self.max_steps,
            memory_ids=self.memory_ids,
            agent_id=self.agent_id,
            persona=self.persona,
            tools=tools_to_update,
            long_term_memory_ids=getattr(self, "long_term_memory_ids", None),
            delegates=(
                delegate_ids if delegate_ids else None
            ),  # Add delegates persistence
        )

        # Update the memagent in the memory provider
        updated_memagent_dict = self.memory_provider.update_memagent(memagent_to_update)
        logger.info(f"Memagent {self.agent_id} updated in the memory provider")

        return self

    @classmethod
    def load(
        cls,
        agent_id: str,
        memory_provider: Optional[MemoryProvider] = None,
        **overrides,
    ):
        """
        Retrieve the memagent from the memory provider.

        This method retrieves the memagent from the memory provider.

        Parameters:
            agent_id (str): The agent id.

        Returns:
            MemAgent: The memagent.
        """
        logger.info(f"Loading MemAgent with agent id {agent_id}...")

        # If the memory provider is not provided, then we use the default memory provider
        provider = memory_provider or MemoryProvider()

        # Retrieve the memagent from the memory provider
        memagent = provider.retrieve_memagent(agent_id)

        if not memagent:
            raise ValueError(
                f"MemAgent with agent id {agent_id} not found in the memory provider"
            )

        model_to_load = None
        if hasattr(memagent, "llm_config") and memagent.llm_config:
            try:
                model_to_load = create_llm_provider(memagent.llm_config)
            except Exception as e:
                logger.warning(
                    f"Could not load model from config: {e}. Model will be None."
                )
        elif hasattr(memagent, "model") and memagent.model:
            model_to_load = memagent.model

        # Get application_mode and memory_types from stored agent
        application_mode_to_use = None
        memory_types_to_use = None

        if hasattr(memagent, "application_mode") and memagent.application_mode:
            application_mode_to_use = memagent.application_mode
            if hasattr(memagent, "memory_types") and memagent.memory_types:
                memory_types_to_use = memagent.memory_types
        else:
            application_mode_to_use = ApplicationMode.DEFAULT.value

        # Load delegate agents if they exist in the stored data
        loaded_delegates = None
        if hasattr(memagent, "delegates") and memagent.delegates:
            loaded_delegates = []
            for delegate_id in memagent.delegates:
                try:
                    delegate_agent = cls.load(delegate_id, provider)
                    loaded_delegates.append(delegate_agent)
                except Exception as e:
                    logger.warning(f"Could not load delegate agent {delegate_id}: {e}")

        # Handle semantic_cache_config reconstruction
        semantic_cache_config_to_load = None
        if (
            hasattr(memagent, "semantic_cache_config")
            and memagent.semantic_cache_config
        ):
            # Convert the stored dict back to SemanticCacheConfig
            try:
                config_dict = dict(memagent.semantic_cache_config)
                # Convert string scope back to enum if present
                if "scope" in config_dict and isinstance(config_dict["scope"], str):
                    from .enums.semantic_cache_scope import SemanticCacheScope

                    scope_str = config_dict["scope"].lower()
                    if scope_str == "local":
                        config_dict["scope"] = SemanticCacheScope.LOCAL
                    elif scope_str == "global":
                        config_dict["scope"] = SemanticCacheScope.GLOBAL
                    else:
                        logger.warning(
                            f"Unknown scope value '{config_dict['scope']}', using LOCAL"
                        )
                        config_dict["scope"] = SemanticCacheScope.LOCAL

                semantic_cache_config_to_load = SemanticCacheConfig(**config_dict)
                logger.debug(
                    "Successfully reconstructed SemanticCacheConfig from stored data"
                )
            except Exception as e:
                logger.warning(f"Failed to reconstruct SemanticCacheConfig: {e}")
                semantic_cache_config_to_load = None

        # Instantiate with saved parameters (and allow callers to override e.g. model)
        agent_instance = cls(
            model=overrides.get("model", model_to_load),
            tools=overrides.get("tools", getattr(memagent, "tools", None)),
            persona=overrides.get("persona", getattr(memagent, "persona", None)),
            instruction=overrides.get(
                "instruction", getattr(memagent, "instruction", None)
            ),
            application_mode=overrides.get("application_mode", application_mode_to_use),
            memory_types=overrides.get("memory_types", memory_types_to_use),
            max_steps=overrides.get("max_steps", getattr(memagent, "max_steps", None)),
            memory_ids=overrides.get("memory_ids", getattr(memagent, "memory_ids", [])),
            agent_id=agent_id,
            memory_provider=provider,
            delegates=overrides.get(
                "delegates", loaded_delegates
            ),  # Include loaded delegates
            semantic_cache=overrides.get(
                "semantic_cache", getattr(memagent, "semantic_cache", False)
            ),
            semantic_cache_config=overrides.get(
                "semantic_cache_config", semantic_cache_config_to_load
            ),
        )

        # Set long_term_memory_ids if they exist
        if hasattr(memagent, "long_term_memory_ids") and memagent.long_term_memory_ids:
            agent_instance.long_term_memory_ids = memagent.long_term_memory_ids

        # Show the logs as a json object
        logger.info(f"MemAgent loaded with agent_id: {agent_id}")

        return agent_instance

    def refresh(self):
        """
        Refresh the memagent from the memory provider.

        This method refreshes the memagent from the memory provider.

        Returns:
            MemAgent: The refreshed memagent.
        """
        try:
            # Get a fresh copy of the memagent from the memory provider
            memagent = self.memory_provider.retrieve_memagent(self.agent_id)

            # Update the memagent with the fresh copy
            self.__dict__.update(memagent.__dict__)

            return self
        except Exception as e:
            logger.error(f"Error refreshing memagent {self.agent_id}: {e}")
            return False

    @classmethod
    def _do_delete(cls, agent_id: str, cascade: bool, memory_provider: MemoryProvider):
        """
        Delete the memagent from the memory provider.

        This method deletes the memagent from the memory provider.

        Parameters:
            agent_id (str): The agent id.
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes all the memory units associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.
            memory_provider (MemoryProvider): The memory provider to use.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.
        """

        try:
            result = memory_provider.delete_memagent(agent_id, cascade)
            logger.info(f"MemAgent {agent_id} deleted from the memory provider")
            return result
        except Exception as e:
            logger.error(
                f"Error deleting MemAgent {agent_id} from the memory provider: {e}"
            )
            return False

    @classmethod
    def delete_by_id(
        cls,
        agent_id: str,
        cascade: bool = False,
        memory_provider: Optional[MemoryProvider] = None,
    ):
        """
        Delete a memagent from the memory provider by agent ID.

        This class method allows deletion of any agent by ID without requiring an instance.
        Use this when you have an agent_id but no agent instance.

        Parameters:
            agent_id (str): The agent id to delete.
            memory_provider (Optional[MemoryProvider]): The memory provider to use.
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes
                           all the memory units associated with the memagent.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.
        """

        # If the memory provider is not provided, then use the default memory provider
        provider = memory_provider or MemoryProvider()

        try:
            result = cls._do_delete(agent_id, cascade, provider)
            logger.info(f"MemAgent {agent_id} deleted from the memory provider")
            return result
        except Exception as e:
            logger.error(
                f"Error deleting MemAgent {agent_id} from the memory provider: {e}"
            )
            return False

    def delete(self, cascade: bool = False):
        """
        Delete this memagent instance from the memory provider.

        This instance method deletes the current agent using its own agent_id and memory_provider.
        Use this when you have an agent instance and want to delete itself.

        Parameters:
            cascade (bool): Whether to cascade the deletion of the memagent. This deletes
                           all the memory units associated with the memagent.

        Returns:
            bool: True if the memagent was deleted successfully, False otherwise.

        Raises:
            ValueError: If the agent_id is not set on this instance.
        """

        if self.agent_id is None:
            raise ValueError(
                "MemAgent agent_id is not set. Please set the agent_id before deleting the memagent."
            )

        return type(self)._do_delete(self.agent_id, cascade, self.memory_provider)

    # Memory Management Methods

    def download_memory(self, memagent: MemAgentModel):
        """
        Download the memory of the memagent.

        This method downloads the memory of the memagent.
        It takes in a memagent and then adds the memory_ids of the memagent to the memory_ids attribute of the memagent.
        It then updates the memory_ids of the memagent in the memory provider.

        Parameters:
            memagent (MemAgent): The memagent to download the memory from.

        Returns:
            bool: True if the memory was downloaded successfully, False otherwise.
        """

        try:
            # Add the list of the memory_ids to the memory_ids attribute of the memagent
            self.memory_ids = self.memory_ids + memagent.memory_ids

            # Update the memory_ids of the memagent in the memory provider
            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                self.memory_provider.update_memagent_memory_ids(
                    self.agent_id, self.memory_ids
                )
            else:
                raise ValueError(
                    "Memory provider does not have the update_memagent_memory_ids method."
                )
            return True
        except Exception as e:
            logger.error(
                f"Error downloading memory from memagent {memagent.agent_id}: {e}"
            )
            return False

    def update_memory(self, memory_ids: List[str]):
        """
        Update the memory_ids of the memagent.

        This method updates the memory_ids of the memagent.
        It takes in a list of memory_ids and then adds the list to the memory_ids attribute of the memagent.
        It then updates the memory_ids of the memagent in the memory provider.

        Parameters:
            memory_ids (List[str]): The memory_ids to update.

        Returns:
            bool: True if the memory_ids were updated successfully, False otherwise.
        """

        try:
            # Update the memory_ids of the memagent in the memory provider
            if hasattr(self.memory_provider, "update_memagent_memory_ids"):
                # Add the list of memory_ids to the memory_ids attribute of the memagent
                memories_to_add = self.memory_ids + memory_ids
                self.memory_provider.update_memagent_memory_ids(
                    self.agent_id, memories_to_add
                )

                # Update the memory_ids of the memagent in the memagent
                self.memory_ids = memories_to_add
            else:
                raise ValueError(
                    "Memory provider does not have the update_memagent_memory_ids method."
                )

            return True
        except Exception as e:
            logger.error(f"Error updating memory_ids of memagent {self.agent_id}: {e}")
            return False

    def delete_memory(self):
        """
        Delete the memory_ids of the memagent.

        It deletes the memory_ids of the memagent in the memory provider.

        Returns:
            bool: True if the memory_ids were deleted successfully, False otherwise.
        """

        try:
            if hasattr(self.memory_provider, "delete_memagent_memory_ids"):
                # Delete the memory_ids of the memagent in the memory provider
                self.memory_provider.delete_memagent_memory_ids(self.agent_id)

                # Delete the memory_ids of the memagent in the memagent
                self.memory_ids = []
                return True
            else:
                raise ValueError(
                    "Memory provider does not have the delete_memagent_memory_ids method."
                )
        except Exception as e:
            logger.error(f"Error deleting memory_ids of memagent {self.agent_id}: {e}")
            return False

    # Persona Management

    def set_persona(self, persona: Persona, save: bool = True):
        """
        Set the persona of the memagent.

        Parameters:
            persona (Persona): The persona to set.
            save (bool): Whether to save the memagent after setting the persona.

        Returns:
            bool: True if the persona was set successfully, False otherwise.
        """

        self.persona = persona
        if save:
            self.update()

    def set_persona_from_memory_provider(self, persona_id: str, save: bool = True):
        """
        Set the persona of the memagent from the persona memory store within the memory provider.

        Parameters:
            persona_id (str): The persona id.
            save (bool): Whether to save the memagent after setting the persona.

        Returns:
            bool: True if the persona was set successfully, False otherwise.
        """

        # Check if the memory provider has the retrieve_persona method
        if hasattr(self.memory_provider, "retrieve_persona"):
            self.persona = self.memory_provider.retrieve_persona(persona_id)
        else:
            raise ValueError(
                "Memory provider does not have the retrieve_persona method."
            )

        if self.persona:
            if save:
                self.update()
            return True
        else:
            raise ValueError(
                "Persona is not set. Please set the persona before setting it from the memory provider."
            )

    def export_persona(self):
        """
        Export the persona of the memagent to the persona memory store within the memory provider.

        Returns:
            bool: True if the persona was exported successfully, False otherwise.
        """
        if self.persona:
            self.persona.store_persona(self.memory_provider)
        else:
            raise ValueError(
                "Persona is not set. Please set the persona before exporting it."
            )

        return True

    def delete_persona(self, save: bool = True):
        """
        Delete the persona of the memagent.

        Parameters:
            save (bool): Whether to save the memagent after deleting the persona.

        Returns:
            bool: True if the persona was deleted successfully, False otherwise.
        """

        if self.persona:
            self.persona = None
            if save:
                self.update()
        else:
            raise ValueError(
                "Persona is not set. Please set the persona before deleting it."
            )

        return True

    # Toolbox Management Functions

    def add_tool(
        self,
        tool_id: str = None,
        toolbox: Toolbox = None,
        func: Callable = None,
        persist: bool = False,
        load_available_only: bool = True,
    ) -> bool:
        """
        Add or update a tool to this agent from multiple sources.

        You must supply one of:
          • tool_id: the UUID of an existing toolbox entry, or
          • toolbox: a Toolbox instance (for batch import), or
          • func: a decorated function (for direct addition)

        If the tool is already in self.tools, its entry will be overwritten
        with the latest metadata & function reference. Otherwise it will be appended.

        Parameters:
            tool_id (str):    The tool id in the memory-provider's toolbox.
            toolbox (Toolbox): The Toolbox instance to pull the Python function from.
            func (Callable):  A decorated function to add directly to the agent.
            persist (bool):   Whether to persist the function in the memory provider (only applies to func).
            load_available_only (bool): When using toolbox, only load tools with available functions (default: True).

        Returns:
            bool: True if the tool was added or updated successfully.
        """
        # Handle direct function input
        if func is not None:
            if not callable(func):
                raise ValueError("func parameter must be a callable function")

            return self._add_function_directly(func, persist)

        # Handle tool_id input (existing functionality)
        if tool_id:
            # 1) fetch the metadata from memory-provider
            meta = self.memory_provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)
            if not meta:
                raise ValueError(f"No such tool in the toolbox: {tool_id}")

            # 2) resolve the Python callable from the provided Toolbox
            if not isinstance(toolbox, Toolbox):
                raise ValueError(
                    "Need a Toolbox instance to resolve the Python callable"
                )

            # Use _id (which is the same as tool_id now) for function lookup
            python_fn = toolbox._tools.get(tool_id)
            if not callable(python_fn):
                # fallback: perhaps the stored metadata itself packs a .function field?
                tb_meta = next(
                    (m for m in toolbox.list_tools() if str(m.get("_id")) == tool_id),
                    None,
                )
                if tb_meta and callable(tb_meta.get("function")):
                    python_fn = tb_meta["function"]

            if not callable(python_fn):
                # Silently skip tools without functions instead of raising error
                return False

            # 3) build the new entry
            new_entry = self._build_entry(meta, python_fn)

            # 4) Handle different types of self.tools (Toolbox vs list)
            if isinstance(self.tools, Toolbox):
                # If self.tools is a Toolbox, convert to list format for agent use
                self.tools = []

            # Now self.tools should be a list (or None)
            existing_idx = None
            for idx, t in enumerate(self.tools or []):
                if str(t.get("_id")) == tool_id:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                # replace the old entry
                self.tools[existing_idx] = new_entry
            else:
                # append a brand-new entry
                self.tools = self.tools or []
                self.tools.append(new_entry)

            # 5) persist the agent's updated tools list
            self.update(tools=self.tools)
            return True

        # Handle toolbox input (existing functionality)
        if toolbox:
            if not isinstance(toolbox, Toolbox):
                raise TypeError(f"Expected a Toolbox, got {type(toolbox)}")

            # If self.tools is already the same Toolbox, no need to re-add
            if self.tools is toolbox:
                return True

            # Use the shared initialization logic for better performance
            # Convert existing tools to list format if needed
            if isinstance(self.tools, Toolbox):
                self.tools = []

            # Store current tools count to check if any were added
            initial_count = len(self.tools or [])

            if load_available_only:
                # EFFICIENT APPROACH: Only process tools that have functions available in memory
                logger.info(
                    f"Loading {len(toolbox._tools)} available tools from toolbox (skipping database-only tools)"
                )

                for tool_id, python_fn in toolbox._tools.items():
                    if callable(python_fn):
                        # Get metadata for this specific tool
                        meta = toolbox.get_tool_by_id(tool_id)
                        if meta:
                            # Check if tool already exists to avoid duplicates
                            existing_idx = None
                            for idx, t in enumerate(self.tools or []):
                                if str(t.get("_id")) == tool_id:
                                    existing_idx = idx
                                    break

                            # build the new entry
                            new_entry = self._build_entry(meta, python_fn)

                            if existing_idx is not None:
                                # replace the old entry
                                self.tools[existing_idx] = new_entry
                            else:
                                # append a brand-new entry
                                self.tools = self.tools or []
                                self.tools.append(new_entry)
                        else:
                            logger.warning(
                                f"Tool {tool_id} has function but no metadata - skipping"
                            )
            else:
                # LEGACY APPROACH: Process all tools from database (may generate warnings)
                logger.info(
                    "Loading all tools from toolbox database (including metadata-only tools)"
                )

                for meta in toolbox.list_tools():
                    tid = str(meta.get("_id"))
                    if tid:
                        # Resolve the Python callable from the provided Toolbox
                        python_fn = toolbox._tools.get(tid)
                        if not callable(python_fn):
                            # fallback: perhaps the stored metadata itself packs a .function field?
                            if meta and callable(meta.get("function")):
                                python_fn = meta["function"]

                        if callable(python_fn):
                            # Check if tool already exists to avoid duplicates
                            existing_idx = None
                            for idx, t in enumerate(self.tools or []):
                                if str(t.get("_id")) == tid:
                                    existing_idx = idx
                                    break

                            # build the new entry
                            new_entry = self._build_entry(meta, python_fn)

                            if existing_idx is not None:
                                # replace the old entry
                                self.tools[existing_idx] = new_entry
                            else:
                                # append a brand-new entry
                                self.tools = self.tools or []
                                self.tools.append(new_entry)
                        else:
                            # Skip tools without functions and warn
                            logger.warning(
                                f"Skipping tool with _id {tid} - no callable function found"
                            )
                            continue

            # Save the updated tools if any were added
            final_count = len(self.tools or [])
            if final_count > initial_count:
                self.update(tools=self.tools)
                return True

            return (
                final_count > 0
            )  # Return True if we have tools, even if none were newly added

        # --- neither provided: error ---
        raise ValueError("Must supply either a tool_id, toolbox, or func parameter.")

    def _add_function_directly(self, func: Callable, persist: bool = False) -> bool:
        """
        Add a function directly to the agent without requiring pre-registration in a toolbox.

        Parameters:
            func (Callable): The function to add
            persist (bool): Whether to persist the function in the memory provider

        Returns:
            bool: True if the function was added successfully
        """
        try:
            # Generate tool metadata from function introspection
            meta = self._generate_tool_metadata_from_function(func)

            # If persist is True, store in memory provider and get an _id
            if persist:
                # Store in memory provider and get the assigned _id
                stored_id = self.memory_provider.store(meta, MemoryType.TOOLBOX)
                meta["_id"] = stored_id
            else:
                # Generate a temporary UUID for ephemeral tools
                meta["_id"] = str(uuid.uuid4())

            # Build the new entry using existing logic
            new_entry = self._build_entry(meta, func)

            # Convert self.tools to list format if needed
            if isinstance(self.tools, Toolbox):
                self.tools = []

            # Check if tool already exists (by function name)
            existing_idx = None
            func_name = func.__name__
            for idx, t in enumerate(self.tools or []):
                if t.get("function", {}).get("name") == func_name:
                    existing_idx = idx
                    break

            if existing_idx is not None:
                # Replace existing tool with same name
                self.tools[existing_idx] = new_entry
                logger.info(f"Updated existing tool: {func_name}")
            else:
                # Add new tool
                self.tools = self.tools or []
                self.tools.append(new_entry)
                logger.info(f"Added new tool: {func_name}")

            # Update the agent's tools (only if persist is True)
            if persist:
                self.update(tools=self.tools)

            return True

        except Exception as e:
            logger.error(f"Failed to add function {func.__name__}: {e}")
            return False

    def _generate_tool_metadata_from_function(self, func: Callable) -> dict:
        """
        Generate tool metadata from function introspection.

        Parameters:
            func (Callable): The function to analyze

        Returns:
            dict: Tool metadata compatible with ToolSchemaType
        """
        import inspect
        from typing import get_type_hints

        # Get function signature and docstring
        sig = inspect.signature(func)
        docstring = func.__doc__ or f"Tool function: {func.__name__}"

        # Extract parameters
        parameters = []
        required = []

        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == "self":
                continue

            # Determine parameter type
            param_type = "string"  # default
            if param_name in type_hints:
                type_hint = type_hints[param_name]
                if type_hint == int:
                    param_type = "integer"
                elif type_hint == float:
                    param_type = "number"
                elif type_hint == bool:
                    param_type = "boolean"
                elif hasattr(type_hint, "__origin__") and type_hint.__origin__ == list:
                    param_type = "array"

            # Create parameter schema
            param_schema = {
                "name": param_name,
                "description": f"Parameter {param_name}",
                "type": param_type,
                "required": param.default == inspect.Parameter.empty,
            }

            parameters.append(param_schema)

            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Create function schema
        function_schema = {
            "name": func.__name__,
            "description": docstring.strip(),
            "parameters": parameters,
            "required": required,
            "queries": [],  # Empty for direct functions
        }

        # Create tool schema
        tool_schema = {"type": "function", "function": function_schema}

        return tool_schema

    def export_tools(self):
        """
        Export the tools of the memagent to the toolbox within the memory provider.
        Only tools with a tool_id are exported.

        Returns:
            bool: True if the tools were exported successfully, False otherwise.
        """
        # If tools is set and not empty, export the tools to the toolbox within the memory provider
        if self.tools and len(self.tools) > 0:
            for tool in self.tools:
                if tool.get("_id") is None:
                    # Export the tool to the toolbox within the memory provider
                    new_tool_id = self.memory_provider.store(tool, MemoryType.TOOLBOX)
                    self.tools.append({"tool_id": new_tool_id, **tool})
            return True
        else:
            raise ValueError(
                "No tools to export. Please add a tool to the memagent before exporting it."
            )

    def refresh_tools(self, tool_id: str):
        """
        Refresh the tools of the memagent from the toolbox within the memory provider.
        This method refreshes the tools of the memagent from the toolbox within the memory provider.

        Parameters:
            tool_id (str): The tool id.

        Returns:
            bool: True if the tools were refreshed successfully, False otherwise.
        """

        # TODO In this method there is a retriveal of the tool from the memory provider (1 within this method and 2 in the add_tool method), this can be optimized by retrieving the tool from the memory provider once and then adding it to the memagent.
        if tool_id:
            # Retrieve the tool from the toolbox within the memory provider
            tool_meta = self.memory_provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)

            # If the tool is not found, raise an error
            if not tool_meta:
                raise ValueError(
                    f"No such tool: {tool_id} in the toolbox within memory provider"
                )

            # If the tool is found, add it to the memagent
            self.add_tool(tool_id=tool_id)
            return True
        else:
            raise ValueError(
                "Tool id is not set. Please set the tool id before refreshing the tool."
            )

    def delete_tool(self, tool_id: str):
        """
        Delete a tool from the memagent.

        This method deletes a tool from the memagent.

        Parameters:
            tool_id (str): The tool id (_id).

        Returns:
            bool: True if the tool was deleted successfully, False otherwise.
        """
        if self.tools:
            self.tools = [
                tool for tool in self.tools if str(tool.get("_id")) != tool_id
            ]
            self.save()
        else:
            raise ValueError(
                "No tools to delete. Please add a tool to the memagent before deleting it."
            )

    def __str__(self):
        """
        Return a string representation of the memagent.
        """

        # Get a fresh copy of the memagent from the memory provider
        self.refresh()

        return f"MemAgent(agent_id={self.agent_id}, memory_ids={self.memory_ids}, application_mode={self.application_mode.value}, active_memory_types={[mt.value for mt in self.active_memory_types]}, max_steps={self.max_steps}, instruction={self.instruction}, model={self.model}, tools={self.tools}, persona={self.persona})"

    def __repr__(self):
        """
        Return a string representation of the memagent that can be used to recreate the object.
        """
        return f"MemAgent(agent_id={self.agent_id}, memory_provider={self.memory_provider})"

    # Long-term Memory Management

    def add_long_term_memory(
        self, corpus: str, namespace: str = "general"
    ) -> Optional[str]:
        """
        Add long-term memory to the agent.

        Parameters:
            corpus (str): The text content to store in long-term memory.
            namespace (str): A namespace to categorize the knowledge.

        Returns:
            Optional[str]: The ID of the created long-term memory, or None if unsuccessful.
        """
        try:
            kb = KnowledgeBase(self.memory_provider)

            # Ingest the knowledge
            long_term_memory_id = kb.ingest_knowledge(corpus, namespace)

            # Initialize the long_term_memory_ids attribute if it doesn't exist
            if not hasattr(self, "long_term_memory_ids"):
                self.long_term_memory_ids = []

            # Add the ID to the agent's long-term memory IDs
            self.long_term_memory_ids.append(long_term_memory_id)

            # Update the agent (use update for existing agents, save for new ones)
            if self.agent_id is None:
                self.save()
            else:
                self.update()

            return long_term_memory_id
        except Exception as e:
            logger.error(f"Error adding long-term memory to agent {self.agent_id}: {e}")
            return None

    def retrieve_long_term_memory(
        self, long_term_memory_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve long-term memory associated with the agent.

        Parameters:
            long_term_memory_id (Optional[str]): The ID of a specific long-term memory to retrieve.
                                                If None, retrieves all long-term memories associated with the agent.

        Returns:
            List[Dict[str, Any]]: List of knowledge entries.
        """
        try:
            kb = KnowledgeBase(self.memory_provider)

            # Initialize result list
            all_entries = []

            # If a specific ID is provided, retrieve just that one
            if long_term_memory_id:
                return kb.retrieve_knowledge(long_term_memory_id)

            # Otherwise, retrieve all long-term memories associated with the agent
            if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
                for memory_id in self.long_term_memory_ids:
                    entries = kb.retrieve_knowledge(memory_id)
                    all_entries.extend(entries)

            return all_entries
        except Exception as e:
            logger.error(
                f"Error retrieving long-term memory for agent {self.agent_id}: {e}"
            )
            return []

    def delete_long_term_memory(self, long_term_memory_id: str) -> bool:
        """
        Delete a specific long-term memory and remove it from the agent.

        Parameters:
            long_term_memory_id (str): The ID of the long-term memory to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            # Remove the ID from the agent's list
            if hasattr(self, "long_term_memory_ids") and self.long_term_memory_ids:
                if long_term_memory_id in self.long_term_memory_ids:
                    self.long_term_memory_ids.remove(long_term_memory_id)
                    self.update()

            # Delete the knowledge from the memory provider
            kb = KnowledgeBase(self.memory_provider)
            return kb.delete_knowledge(long_term_memory_id)
        except Exception as e:
            logger.error(f"Error deleting long-term memory {long_term_memory_id}: {e}")
            return False

    def update_long_term_memory(self, long_term_memory_id: str, corpus: str) -> bool:
        """
        Update the content of a specific long-term memory.

        Parameters:
            long_term_memory_id (str): The ID of the long-term memory to update.
            corpus (str): The new text content.

        Returns:
            bool: True if update was successful, False otherwise.
        """
        try:
            # Check if this memory is associated with the agent
            if (
                not hasattr(self, "long_term_memory_ids")
                or long_term_memory_id not in self.long_term_memory_ids
            ):
                logger.warning(
                    f"Long-term memory {long_term_memory_id} is not associated with agent {self.agent_id}"
                )
                return False

            # Update the knowledge
            kb = KnowledgeBase(self.memory_provider)
            return kb.update_knowledge(long_term_memory_id, corpus)
        except Exception as e:
            logger.error(f"Error updating long-term memory {long_term_memory_id}: {e}")
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

            from .embeddings import get_embedding

            # Calculate time range (days_back days ago to now)
            current_time = time.time()
            start_time = current_time - (days_back * 24 * 60 * 60)

            logger.info(
                f"Generating summaries for agent {self.agent_id} from {days_back} days back"
            )
            logger.debug(f"Time range: {start_time} to {current_time}")
            logger.debug(f"Agent memory_ids: {self.memory_ids}")
            logger.debug(f"Active memory types: {self.memory_unit.active_memory_types}")

            # Collect memory units from all active memory types
            all_memories = []
            for memory_id in self.memory_ids:
                logger.info(f"Searching memory_id: {memory_id}")
                for memory_type in self.memory_unit.active_memory_types:
                    if memory_type in [
                        MemoryType.CONVERSATION_MEMORY,
                        MemoryType.WORKFLOW_MEMORY,
                        MemoryType.SHORT_TERM_MEMORY,
                        MemoryType.LONG_TERM_MEMORY,
                    ]:
                        try:
                            logger.info(
                                f"Searching {memory_type.value} collection for memory_id: {memory_id}"
                            )
                            # Get memories within the time range
                            memories = self._get_memories_in_time_range(
                                memory_id, memory_type, start_time, current_time
                            )
                            logger.info(
                                f"Found {len(memories)} memories in {memory_type.value}"
                            )
                            all_memories.extend(memories)
                        except Exception as e:
                            logger.warning(
                                f"Could not retrieve {memory_type.value} memories: {e}"
                            )

            if not all_memories:
                logger.info(
                    f"No memories found for agent {self.agent_id} in the specified time range"
                )
                return []

            # Sort memories by timestamp
            all_memories.sort(key=lambda x: x.get("timestamp", 0))

            logger.info(f"Found {len(all_memories)} memory units to summarize")

            # Split memories into chunks and create summaries
            summary_ids = []
            for i in range(0, len(all_memories), max_memories_per_summary):
                memory_chunk = all_memories[i : i + max_memories_per_summary]

                # Generate summary for this chunk
                summary_content = self._compress_memories_with_llm(memory_chunk)

                if summary_content:
                    # Helper function to convert timestamp to float
                    def to_float_timestamp(timestamp_value, fallback_value):
                        if isinstance(timestamp_value, str):
                            try:
                                from datetime import datetime

                                return datetime.fromisoformat(
                                    timestamp_value.replace("Z", "+00:00")
                                ).timestamp()
                            except (ValueError, AttributeError):
                                return fallback_value
                        elif isinstance(timestamp_value, (int, float)):
                            return float(timestamp_value)
                        else:
                            return fallback_value

                    # Create summary document with consistent float timestamps
                    summary_doc = {
                        "memory_id": (
                            self.memory_ids[0] if self.memory_ids else "default"
                        ),
                        "agent_id": self.agent_id,
                        "summary_content": summary_content,
                        "period_start": to_float_timestamp(
                            memory_chunk[0].get("timestamp"), start_time
                        ),
                        "period_end": to_float_timestamp(
                            memory_chunk[-1].get("timestamp"), current_time
                        ),
                        "memory_units_count": len(memory_chunk),
                        "created_at": current_time,
                        "embedding": get_embedding(summary_content),
                    }

                    # Store summary
                    summary_id = self.memory_provider.store(
                        summary_doc, MemoryType.SUMMARIES
                    )
                    summary_ids.append(summary_id)

                    logger.info(
                        f"Created summary {summary_id} covering {len(memory_chunk)} memories"
                    )

            logger.info(
                f"Generated {len(summary_ids)} summaries for agent {self.agent_id}"
            )
            return summary_ids

        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
            return []

    def update_persona_from_summaries(
        self, max_summaries: int = 5, save: bool = True
    ) -> bool:
        """
        Update the agent's persona based on recent summaries.

        This method retrieves recent summaries, uses an LLM to analyze them,
        and updates the agent's persona goals and background accordingly.

        Parameters:
        -----------
        max_summaries : int, optional
            Maximum number of recent summaries to consider (default: 5)
        save : bool, optional
            Whether to save the updated persona (default: True)

        Returns:
        --------
        bool
            True if persona was successfully updated, False otherwise
        """
        try:
            if not self.persona:
                logger.info(f"Agent {self.agent_id} has no persona to update")
                return False

            # Get recent summaries for this agent
            recent_summaries = []
            for memory_id in self.memory_ids:
                summaries = self.memory_provider.get_summaries_by_memory_id(
                    memory_id, max_summaries
                )
                recent_summaries.extend(summaries)

            if not recent_summaries:
                logger.info(f"No summaries found for agent {self.agent_id}")
                return False

            # Sort by creation time and take the most recent
            recent_summaries.sort(key=lambda x: x.get("created_at", 0), reverse=True)
            recent_summaries = recent_summaries[:max_summaries]

            logger.info(
                f"Updating persona for agent {self.agent_id} using {len(recent_summaries)} summaries"
            )

            # Use LLM to analyze summaries and update persona
            updated_persona_data = self._update_persona_with_llm(recent_summaries)

            if updated_persona_data:
                # Update persona attributes
                if "goals" in updated_persona_data:
                    self.persona.goals = updated_persona_data["goals"]
                if "background" in updated_persona_data:
                    self.persona.background = updated_persona_data["background"]

                # Regenerate embedding for updated persona
                self.persona.embedding = self.persona._generate_embedding()

                if save:
                    # Save updated persona
                    self.persona.store_persona(self.memory_provider)
                    logger.info(
                        f"Successfully updated and saved persona for agent {self.agent_id}"
                    )
                else:
                    logger.info(
                        f"Updated persona for agent {self.agent_id} (not saved)"
                    )

                return True
            else:
                logger.warning(
                    f"LLM failed to generate persona updates for agent {self.agent_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating persona from summaries: {e}")
            return False

    def _get_memories_in_time_range(
        self,
        memory_id: str,
        memory_type: MemoryType,
        start_time: float,
        end_time: float,
    ) -> List[Dict]:
        """
        Retrieve memory units within a specific time range.

        Parameters:
        -----------
        memory_id : str
            The memory ID to retrieve components for
        memory_type : MemoryType
            The type of memory to retrieve
        start_time : float
            Start timestamp (Unix epoch)
        end_time : float
            End timestamp (Unix epoch)

        Returns:
        --------
        List[Dict]
            List of memory units within the time range
        """
        try:
            from datetime import datetime

            # Convert float timestamps to ISO format strings for comparison
            start_iso = datetime.fromtimestamp(start_time).isoformat()
            end_iso = datetime.fromtimestamp(end_time).isoformat()

            # Build query for time range using ISO format strings
            query = {
                "memory_id": memory_id,
                "timestamp": {"$gte": start_iso, "$lte": end_iso},
            }

            # Get collection based on memory type
            collection = None
            if memory_type == MemoryType.CONVERSATION_MEMORY:
                collection = self.memory_provider.conversation_memory_collection
            elif memory_type == MemoryType.WORKFLOW_MEMORY:
                collection = self.memory_provider.workflow_memory_collection
            elif memory_type == MemoryType.SHORT_TERM_MEMORY:
                collection = self.memory_provider.short_term_memory_collection
            elif memory_type == MemoryType.LONG_TERM_MEMORY:
                collection = self.memory_provider.long_term_memory_collection

            if collection is not None:
                # Debug logging
                logger.debug(
                    f"Searching {memory_type.value} collection for memory_id: {memory_id}"
                )
                logger.debug(f"Time range: {start_iso} to {end_iso}")
                logger.debug(f"Query: {query}")

                # Check total documents in collection for this memory_id
                total_docs = collection.count_documents({"memory_id": memory_id})
                logger.debug(
                    f"Total documents in {memory_type.value} for memory_id {memory_id}: {total_docs}"
                )

                # Check if any documents exist without time filter
                if total_docs > 0:
                    # Get a sample document to check timestamp format
                    sample_doc = collection.find_one({"memory_id": memory_id})
                    if sample_doc:
                        sample_timestamp = sample_doc.get(
                            "timestamp", "No timestamp field"
                        )
                        logger.debug(
                            f"Sample timestamp: {sample_timestamp}, type: {type(sample_timestamp)}"
                        )

                # Execute the actual query
                results = list(
                    collection.find(query, {"embedding": 0}).sort("timestamp", 1)
                )
                logger.debug(
                    f"Found {len(results)} memories in time range for {memory_type.value}"
                )

                return results
            else:
                logger.warning(
                    f"No collection found for memory type: {memory_type.value}"
                )
                return []

        except Exception as e:
            logger.error(f"Error retrieving memories in time range: {e}")
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
                timestamp = memory.get("timestamp", "")

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
            4. Important relationship dynamics or behavioral patterns

            Focus on information that would be valuable for understanding personality development,
            preferences, and behavioral patterns over time.

            Memory components:
            {memories_text}

            Provide a well-structured summary that captures the essence of these experiences:
            """

            # Use the agent's LLM to generate summary
            summary = self.model.generate_text(
                compression_prompt,
                instructions="Create a concise but comprehensive summary focusing on emotionally and situationally relevant aspects. Keep it under 300 words.",
            )

            return summary.strip()

        except Exception as e:
            logger.error(f"Error compressing memories with LLM: {e}")
            return ""

    def _update_persona_with_llm(self, summaries: List[Dict]) -> Dict[str, str]:
        """
        Use LLM to update persona based on summaries.

        Parameters:
        -----------
        summaries : List[Dict]
            List of summary documents

        Returns:
        --------
        Dict[str, str]
            Dictionary containing updated 'goals' and 'background' fields
        """
        try:
            # Extract summary contents
            summary_contents = [
                s.get("summary_content", "")
                for s in summaries
                if s.get("summary_content")
            ]

            if not summary_contents:
                return {}

            summaries_text = "\n\n".join(summary_contents)

            # Current persona information
            current_goals = self.persona.goals
            current_background = self.persona.background

            # Create persona update prompt
            update_prompt = f"""
            Based on the following recent summaries of experiences and interactions,
            analyze how the persona should evolve and update the goals and background accordingly.

            Current Persona:
            Goals: {current_goals}
            Background: {current_background}

            Recent Experience Summaries:
            {summaries_text}

            Please provide updated persona information that reflects growth, learning, and adaptation based on these experiences.
            Consider:
            - New skills or knowledge gained
            - Evolved priorities or interests
            - Adjusted behavioral patterns
            - Refined understanding of capabilities

            Respond with ONLY a JSON object in this exact format:
            {{
                "goals": "updated goals here",
                "background": "updated background here"
            }}
            """

            # Use the agent's LLM to generate updates
            response = self.model.generate_text(
                update_prompt,
                instructions="Return only a valid JSON object with 'goals' and 'background' fields. No other text.",
            )

            # Parse JSON response
            import json

            try:
                persona_updates = json.loads(response.strip())
                if (
                    isinstance(persona_updates, dict)
                    and "goals" in persona_updates
                    and "background" in persona_updates
                ):
                    return persona_updates
                else:
                    logger.warning("LLM response was not in expected format")
                    return {}
            except json.JSONDecodeError:
                logger.warning("LLM response was not valid JSON")
                return {}

        except Exception as e:
            logger.error(f"Error updating persona with LLM: {e}")
            return {}
