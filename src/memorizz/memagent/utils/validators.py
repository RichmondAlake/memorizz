# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Validation utilities for MemAgent."""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates MemAgent configuration parameters."""

    @staticmethod
    def validate_agent_id(agent_id: Optional[str]) -> str:
        """
        Validate and normalize agent ID.

        Args:
            agent_id: Agent ID to validate

        Returns:
            Valid agent ID

        Raises:
            ValueError: If agent_id is invalid
        """
        if agent_id is None:
            return str(uuid.uuid4())

        if not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a string")

        agent_id = agent_id.strip()
        if not agent_id:
            return str(uuid.uuid4())

        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not re.match(r"^[a-zA-Z0-9_-]+$", agent_id):
            raise ValueError(
                "Agent ID can only contain letters, numbers, hyphens, and underscores"
            )

        if len(agent_id) > 100:
            raise ValueError("Agent ID cannot exceed 100 characters")

        return agent_id

    @staticmethod
    def validate_instruction(instruction: Optional[str]) -> str:
        """
        Validate instruction parameter.

        Args:
            instruction: Instruction to validate

        Returns:
            Valid instruction string
        """
        if not instruction or not isinstance(instruction, str):
            return "You are a helpful assistant."

        instruction = instruction.strip()
        if not instruction:
            return "You are a helpful assistant."

        # Check reasonable length limits
        if len(instruction) > 10000:
            logger.warning("Instruction is very long, truncating to 10000 characters")
            instruction = instruction[:10000] + "..."

        return instruction

    @staticmethod
    def validate_max_steps(max_steps: Union[int, str]) -> int:
        """
        Validate max_steps parameter.

        Args:
            max_steps: Maximum steps to validate

        Returns:
            Valid max steps integer

        Raises:
            ValueError: If max_steps is invalid
        """
        if isinstance(max_steps, str):
            try:
                max_steps = int(max_steps)
            except ValueError:
                raise ValueError("max_steps must be a valid integer")

        if not isinstance(max_steps, int):
            raise ValueError("max_steps must be an integer")

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if max_steps > 1000:
            logger.warning(
                f"max_steps is very high ({max_steps}), consider reducing for performance"
            )

        return max_steps

    @staticmethod
    def validate_memory_ids(
        memory_ids: Optional[Union[str, List[str]]],
    ) -> Optional[List[str]]:
        """
        Validate memory IDs parameter.

        Args:
            memory_ids: Memory IDs to validate

        Returns:
            Valid list of memory IDs or None
        """
        if memory_ids is None:
            return None

        if isinstance(memory_ids, str):
            memory_ids = [memory_ids]

        if not isinstance(memory_ids, list):
            raise ValueError("memory_ids must be a string or list of strings")

        validated_ids = []
        for mem_id in memory_ids:
            if not isinstance(mem_id, str):
                raise ValueError("All memory IDs must be strings")

            mem_id = mem_id.strip()
            if not mem_id:
                continue  # Skip empty strings

            if len(mem_id) > 200:
                raise ValueError("Memory ID cannot exceed 200 characters")

            validated_ids.append(mem_id)

        return validated_ids if validated_ids else None

    @staticmethod
    def validate_llm_config(
        llm_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Validate LLM configuration.

        Args:
            llm_config: LLM config to validate

        Returns:
            Valid LLM config or None
        """
        if llm_config is None:
            return None

        if not isinstance(llm_config, dict):
            raise ValueError("llm_config must be a dictionary")

        # Check for required fields if provider is specified
        if "provider" in llm_config:
            provider = llm_config["provider"]
            if not isinstance(provider, str):
                raise ValueError("LLM provider must be a string")

            # Validate provider-specific requirements
            if provider.lower() == "openai":
                if "api_key" not in llm_config and "model" not in llm_config:
                    logger.warning(
                        "OpenAI provider typically requires api_key or model"
                    )

        return llm_config

    @staticmethod
    def validate_application_mode(mode: Optional[str]) -> str:
        """
        Validate application mode.

        Args:
            mode: Application mode to validate

        Returns:
            Valid application mode
        """
        valid_modes = ["assistant", "chatbot", "agent"]

        if mode is None:
            return "assistant"

        if not isinstance(mode, str):
            raise ValueError("Application mode must be a string")

        mode = mode.lower().strip()
        if mode not in valid_modes:
            logger.warning(f"Unknown application mode '{mode}', using 'assistant'")
            return "assistant"

        return mode


class InputValidator:
    """Validates user inputs and runtime parameters."""

    @staticmethod
    def validate_query(query: Any) -> str:
        """
        Validate user query input.

        Args:
            query: Query to validate

        Returns:
            Valid query string

        Raises:
            ValueError: If query is invalid
        """
        if query is None:
            raise ValueError("Query cannot be None")

        if not isinstance(query, str):
            try:
                query = str(query)
            except Exception:
                raise ValueError("Query must be convertible to string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        if len(query) > 50000:
            raise ValueError("Query is too long (max 50,000 characters)")

        # Check for potential injection patterns (basic security)
        suspicious_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Query contains suspicious pattern, sanitizing")
                query = re.sub(pattern, "[REMOVED]", query, flags=re.IGNORECASE)

        return query

    @staticmethod
    def validate_thread_id(thread_id: Optional[str]) -> Optional[str]:
        """
        Validate thread ID.

        Args:
            thread_id: Thread ID to validate

        Returns:
            Valid thread ID or None
        """
        if thread_id is None:
            return None

        if not isinstance(thread_id, str):
            raise ValueError("Thread ID must be a string")

        thread_id = thread_id.strip()
        if not thread_id:
            return None

        if len(thread_id) > 200:
            raise ValueError("Thread ID cannot exceed 200 characters")

        # Check format (UUID-like or alphanumeric with hyphens/underscores)
        if not re.match(r"^[a-zA-Z0-9_-]+$", thread_id):
            raise ValueError("Thread ID contains invalid characters")

        return thread_id

    @staticmethod
    def validate_memory_id(memory_id: Optional[str]) -> Optional[str]:
        """
        Validate memory ID for runtime use.

        Args:
            memory_id: Memory ID to validate

        Returns:
            Valid memory ID or None
        """
        if memory_id is None:
            return None

        if not isinstance(memory_id, str):
            raise ValueError("Memory ID must be a string")

        memory_id = memory_id.strip()
        if not memory_id:
            return None

        if len(memory_id) > 200:
            raise ValueError("Memory ID cannot exceed 200 characters")

        return memory_id

    @staticmethod
    def sanitize_response(response: str) -> str:
        """
        Sanitize response for safe output.

        Args:
            response: Response to sanitize

        Returns:
            Sanitized response
        """
        if not isinstance(response, str):
            response = str(response)

        # Remove or escape potentially harmful content
        # This is a basic implementation - more sophisticated sanitization may be needed

        # Remove script tags
        response = re.sub(
            r"<script[^>]*>.*?</script>",
            "[SCRIPT_REMOVED]",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )

        # Remove javascript: links
        response = re.sub(
            r"javascript:", "removed-javascript:", response, flags=re.IGNORECASE
        )

        # Remove event handlers
        response = re.sub(
            r'on\w+\s*=\s*["\']?[^"\'>\s]+["\']?', "", response, flags=re.IGNORECASE
        )

        return response

    @staticmethod
    def validate_tool_input(
        tool_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validate tool execution input.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tuple of validated (tool_name, arguments)
        """
        # Validate tool name
        if not isinstance(tool_name, str):
            raise ValueError("Tool name must be a string")

        tool_name = tool_name.strip()
        if not tool_name:
            raise ValueError("Tool name cannot be empty")

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tool_name):
            raise ValueError("Tool name contains invalid characters")

        # Validate arguments
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a dictionary")

        # Check for reasonable argument count and size
        if len(arguments) > 50:
            raise ValueError("Too many tool arguments (max 50)")

        total_size = sum(len(str(k)) + len(str(v)) for k, v in arguments.items())
        if total_size > 100000:
            raise ValueError("Tool arguments are too large")

        return tool_name, arguments
