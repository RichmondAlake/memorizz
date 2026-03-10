# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Configuration builder for MemAgent."""

import logging
from typing import Any, Dict

from ..models import MemAgentConfig

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """
    Builder for MemAgent configuration objects.

    This provides a fluent interface for building complex configurations
    that can be reused across multiple agent instances.
    """

    def __init__(self):
        """Initialize the config builder."""
        self._config_dict = {}

    def instruction(self, text: str) -> "ConfigBuilder":
        """Set the instruction."""
        self._config_dict["instruction"] = text
        return self

    def max_steps(self, steps: int) -> "ConfigBuilder":
        """Set maximum steps."""
        self._config_dict["max_steps"] = steps
        return self

    def tool_access(self, access: str) -> "ConfigBuilder":
        """Set tool access level."""
        self._config_dict["tool_access"] = access
        return self

    def semantic_cache(self, enabled: bool) -> "ConfigBuilder":
        """Enable/disable semantic cache."""
        self._config_dict["semantic_cache"] = enabled
        return self

    def application_mode(self, mode: str) -> "ConfigBuilder":
        """Set application mode."""
        self._config_dict["application_mode"] = mode
        return self

    def verbose(self, enabled: bool) -> "ConfigBuilder":
        """Enable/disable verbose logging."""
        self._config_dict["verbose"] = enabled
        return self

    def custom(self, key: str, value: Any) -> "ConfigBuilder":
        """Add custom configuration parameter."""
        self._config_dict[key] = value
        return self

    def build(self) -> MemAgentConfig:
        """
        Build the configuration object.

        Returns:
            Configured MemAgentConfig instance.
        """
        return MemAgentConfig(**self._config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return self._config_dict.copy()


# Preset configurations
class ConfigPresets:
    """Predefined configuration presets for common use cases."""

    @staticmethod
    def assistant() -> MemAgentConfig:
        """Configuration for general assistant."""
        return (
            ConfigBuilder()
            .instruction("You are a helpful AI assistant.")
            .max_steps(20)
            .application_mode("assistant")
            .semantic_cache(False)
            .build()
        )

    @staticmethod
    def chatbot() -> MemAgentConfig:
        """Configuration for conversational chatbot."""
        return (
            ConfigBuilder()
            .instruction("You are a friendly conversational chatbot.")
            .max_steps(15)
            .application_mode("chatbot")
            .semantic_cache(True)
            .build()
        )

    @staticmethod
    def task_agent() -> MemAgentConfig:
        """Configuration for task-oriented agent."""
        return (
            ConfigBuilder()
            .instruction(
                "You are a task-oriented agent focused on completing specific objectives."
            )
            .max_steps(30)
            .application_mode("agent")
            .tool_access("private")
            .semantic_cache(False)
            .build()
        )

    @staticmethod
    def research_agent() -> MemAgentConfig:
        """Configuration for research and analysis agent."""
        return (
            ConfigBuilder()
            .instruction(
                "You are a research agent specialized in information gathering and analysis."
            )
            .max_steps(25)
            .application_mode("agent")
            .semantic_cache(True)
            .verbose(True)
            .build()
        )
