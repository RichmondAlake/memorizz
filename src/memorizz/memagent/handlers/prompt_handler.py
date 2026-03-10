# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Prompt generation and formatting for MemAgent."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptHandler:
    """
    Handles prompt generation and formatting for MemAgent.

    This class encapsulates all prompt-related logic, making it easier
    to maintain and test prompt generation separately.
    """

    def __init__(self):
        """Initialize the prompt handler."""
        self.prompt_templates = {
            "system": self._default_system_template,
            "conversation": self._default_conversation_template,
            "tool_usage": self._default_tool_template,
        }

    def generate_system_prompt(
        self,
        instruction: str,
        persona_info: Optional[str] = None,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate the system prompt for the agent.

        Args:
            instruction: Base instruction for the agent
            persona_info: Optional persona information
            tools_info: Optional tools information
            context: Optional additional context

        Returns:
            Formatted system prompt
        """
        try:
            prompt_parts = [instruction]

            # Add persona information if available
            if persona_info:
                prompt_parts.append(f"\n## Persona\n{persona_info}")

            # Add tool information if available
            if tools_info and len(tools_info) > 0:
                tools_section = self._format_tools_section(tools_info)
                prompt_parts.append(f"\n## Available Tools\n{tools_section}")

            # Add context information if available
            if context:
                context_section = self._format_context_section(context)
                if context_section:
                    prompt_parts.append(f"\n## Context\n{context_section}")

            # Add current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prompt_parts.append(f"\n## Current Time\n{current_time}")

            return "\n".join(prompt_parts)

        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            return instruction  # Fallback to basic instruction

    def format_conversation_history(
        self, history: List[Dict[str, Any]], max_entries: int = 10
    ) -> List[Dict[str, str]]:
        """
        Format conversation history for LLM consumption.

        Args:
            history: List of conversation entries
            max_entries: Maximum number of entries to include

        Returns:
            Formatted conversation messages
        """
        try:
            formatted_messages = []

            # Process recent history (most recent first, then reverse)
            recent_history = (
                history[-max_entries:] if len(history) > max_entries else history
            )

            for entry in recent_history:
                formatted_msg = self._format_history_entry(entry)
                if formatted_msg:
                    formatted_messages.append(formatted_msg)

            return formatted_messages

        except Exception as e:
            logger.error(f"Failed to format conversation history: {e}")
            return []

    def _format_history_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format a single history entry."""
        try:
            if isinstance(entry, dict) and "content" in entry:
                content = entry["content"]

                if isinstance(content, dict):
                    role = content.get("role", "user")
                    text = content.get("content", "")

                    if role and text:
                        return {"role": role, "content": text}
                elif isinstance(content, str):
                    # Assume it's a user message if no role specified
                    return {"role": "user", "content": content}

            return None

        except Exception as e:
            logger.warning(f"Failed to format history entry: {e}")
            return None

    def _format_tools_section(self, tools_info: List[Dict[str, Any]]) -> str:
        """Format tools information for the system prompt."""
        try:
            tool_descriptions = []

            for tool in tools_info[:10]:  # Limit to first 10 tools
                name = tool.get("name", "Unknown Tool")
                description = tool.get("description", "No description available")

                # Format parameters if available
                params = tool.get("parameters", {})
                if params:
                    param_list = []
                    for param_name, param_info in params.items():
                        param_type = param_info.get("type", "string")
                        param_desc = param_info.get("description", "")
                        param_list.append(
                            f"  - {param_name} ({param_type}): {param_desc}"
                        )

                    if param_list:
                        parameters_text = "\n".join(param_list)
                        tool_descriptions.append(
                            f"**{name}**: {description}\n  Parameters:\n{parameters_text}"
                        )
                    else:
                        tool_descriptions.append(f"**{name}**: {description}")
                else:
                    tool_descriptions.append(f"**{name}**: {description}")

            return "\n\n".join(tool_descriptions)

        except Exception as e:
            logger.error(f"Failed to format tools section: {e}")
            return "Tools available but formatting failed."

    def _format_context_section(self, context: Dict[str, Any]) -> str:
        """Format context information for the system prompt."""
        try:
            context_parts = []

            # Add relevant memories if present
            if "relevant_memories" in context:
                memories = context["relevant_memories"]
                if memories:
                    memory_texts = [
                        self._extract_memory_text(mem) for mem in memories[:5]
                    ]
                    memory_texts = [
                        text for text in memory_texts if text
                    ]  # Filter out None
                    if memory_texts:
                        context_parts.append(
                            "Relevant memories:\n- " + "\n- ".join(memory_texts)
                        )

            # Add any additional context
            for key, value in context.items():
                if key not in ["relevant_memories", "conversation_history", "query"]:
                    if isinstance(value, (str, int, float)) and value:
                        context_parts.append(
                            f"{key.replace('_', ' ').title()}: {value}"
                        )

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Failed to format context section: {e}")
            return ""

    def _extract_memory_text(self, memory: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a memory entry."""
        try:
            if isinstance(memory, dict):
                # Try different possible structures
                if "content" in memory:
                    content = memory["content"]
                    if isinstance(content, dict) and "content" in content:
                        return content["content"]
                    elif isinstance(content, str):
                        return content
                elif "text" in memory:
                    return memory["text"]
                elif "message" in memory:
                    return memory["message"]

            return None

        except Exception:
            return None

    def _default_system_template(self, **kwargs) -> str:
        """Default system prompt template."""
        return kwargs.get("instruction", "You are a helpful assistant.")

    def _default_conversation_template(self, **kwargs) -> str:
        """Default conversation template."""
        return "Continue the conversation naturally."

    def _default_tool_template(self, **kwargs) -> str:
        """Default tool usage template."""
        return "Use the available tools as needed to assist the user."
