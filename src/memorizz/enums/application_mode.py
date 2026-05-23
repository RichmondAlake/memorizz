# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from enum import Enum
from typing import List

from .memory_type import MemoryType


class ApplicationMode(Enum):
    """
    Application modes define the environment and context the agent operates within,
    automatically configuring the appropriate memory types for each scenario.
    """

    # Available application modes
    WORKFLOW = "workflow"
    DEEP_RESEARCH = "deep_research"
    ASSISTANT = "assistant"

    # Default mode
    DEFAULT = ASSISTANT


class ApplicationModeConfig:
    """
    Configuration class that maps application modes to their associated memory types
    and provides additional configuration for each mode.
    """

    # Memory type mappings for each application mode
    MEMORY_TYPE_MAPPINGS = {
        ApplicationMode.WORKFLOW: [
            MemoryType.WORKFLOW_MEMORY,
            MemoryType.TOOLBOX,
            MemoryType.KNOWLEDGE_BASE,  # Knowledge base
            MemoryType.SHORT_TERM_MEMORY,  # For intermediate results
            MemoryType.SUMMARIES,  # For context compression
        ],
        ApplicationMode.DEEP_RESEARCH: [
            MemoryType.TOOLBOX,
            MemoryType.SHARED_MEMORY,
            MemoryType.KNOWLEDGE_BASE,  # Research knowledge base
            MemoryType.SHORT_TERM_MEMORY,  # For research sessions
            MemoryType.SUMMARIES,  # For context compression
        ],
        ApplicationMode.ASSISTANT: [
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.KNOWLEDGE_BASE,  # Knowledge base
            MemoryType.PERSONAS,  # For personalization
            MemoryType.ENTITY_MEMORY,  # Structured entity facts
            MemoryType.SHORT_TERM_MEMORY,  # For context
            MemoryType.SUMMARIES,  # For memory compression
        ],
    }

    # Description for each application mode
    MODE_DESCRIPTIONS = {
        ApplicationMode.WORKFLOW: "Optimized for structured task execution and process automation",
        ApplicationMode.DEEP_RESEARCH: "Designed for intensive research with collaboration capabilities",
        ApplicationMode.ASSISTANT: "General-purpose conversational assistant with personalization",
    }

    @classmethod
    def get_memory_types(cls, mode: ApplicationMode) -> List[MemoryType]:
        """
        Get the memory types associated with an application mode.

        Parameters:
        -----------
        mode : ApplicationMode
            The application mode to get memory types for.

        Returns:
        --------
        List[MemoryType]
            List of memory types for the specified mode.
        """
        return cls.MEMORY_TYPE_MAPPINGS.get(
            mode, cls.MEMORY_TYPE_MAPPINGS[ApplicationMode.DEFAULT]
        )

    @classmethod
    def get_description(cls, mode: ApplicationMode) -> str:
        """
        Get the description for an application mode.

        Parameters:
        -----------
        mode : ApplicationMode
            The application mode to get description for.

        Returns:
        --------
        str
            Description of the application mode.
        """
        return cls.MODE_DESCRIPTIONS.get(mode, "General-purpose application mode")

    @classmethod
    def list_all_modes(cls) -> List[tuple]:
        """
        List all available application modes with their descriptions.

        Returns:
        --------
        List[tuple]
            List of (mode, description) tuples.
        """
        return [(mode, cls.get_description(mode)) for mode in ApplicationMode]

    @classmethod
    def validate_mode(cls, mode_input) -> ApplicationMode:
        """
        Validate and convert a string or enum to ApplicationMode enum.

        Parameters:
        -----------
        mode_input : str | ApplicationMode
            String representation or enum of the application mode.

        Returns:
        --------
        ApplicationMode
            The corresponding ApplicationMode enum.

        Raises:
        -------
        ValueError
            If the mode input is not valid.
        """
        # If it's already an ApplicationMode enum, return it directly
        if isinstance(mode_input, ApplicationMode):
            return mode_input

        # If it's a string, convert it
        if isinstance(mode_input, str):
            try:
                return ApplicationMode(mode_input.lower())
            except ValueError:
                valid_modes = [mode.value for mode in ApplicationMode]
                raise ValueError(
                    f"Invalid application mode: '{mode_input}'. Valid modes: {valid_modes}"
                )

        # If it's neither string nor enum, raise an error
        raise ValueError(
            f"Application mode must be a string or ApplicationMode enum, got {type(mode_input)}"
        )
