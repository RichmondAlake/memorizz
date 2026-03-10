# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Formatting utilities for MemAgent."""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptFormatter:
    """Formats prompts and system messages."""

    @staticmethod
    def format_system_prompt(
        instruction: str,
        persona_info: Optional[str] = None,
        tools_info: Optional[List[Dict[str, Any]]] = None,
        context_info: Optional[str] = None,
    ) -> str:
        """
        Format a complete system prompt.

        Args:
            instruction: Base instruction
            persona_info: Optional persona information
            tools_info: Optional tools information
            context_info: Optional context information

        Returns:
            Formatted system prompt
        """
        sections = [("## Instructions", instruction)]

        if persona_info:
            sections.append(("## Your Persona", persona_info))

        if tools_info:
            tools_text = PromptFormatter._format_tools_list(tools_info)
            sections.append(("## Available Tools", tools_text))

        if context_info:
            sections.append(("## Context", context_info))

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        sections.append(("## Current Time", timestamp))

        # Join sections
        formatted_sections = []
        for title, content in sections:
            if content:
                formatted_sections.append(f"{title}\\n{content}")

        return "\\n\\n".join(formatted_sections)

    @staticmethod
    def format_conversation_context(
        history: List[Dict[str, Any]], max_entries: int = 10
    ) -> str:
        """
        Format conversation history for context.

        Args:
            history: List of conversation entries
            max_entries: Maximum entries to include

        Returns:
            Formatted conversation context
        """
        if not history:
            return ""

        recent_history = (
            history[-max_entries:] if len(history) > max_entries else history
        )

        formatted_entries = []
        for entry in recent_history:
            formatted_entry = PromptFormatter._format_history_entry(entry)
            if formatted_entry:
                formatted_entries.append(formatted_entry)

        if not formatted_entries:
            return ""

        return "## Recent Conversation\\n" + "\\n".join(formatted_entries)

    @staticmethod
    def _format_tools_list(tools_info: List[Dict[str, Any]]) -> str:
        """Format tools information for prompt."""
        if not tools_info:
            return "No tools available."

        tool_descriptions = []
        for tool in tools_info[:10]:  # Limit to 10 tools
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description")

            # Format parameters if available
            params = tool.get("parameters", {})
            if params:
                param_list = []
                required = tool.get("required", [])

                for param_name, param_info in params.items():
                    param_type = param_info.get("type", "string")
                    param_desc = param_info.get("description", "")
                    required_marker = " (required)" if param_name in required else ""

                    param_list.append(
                        f"  - {param_name} ({param_type}){required_marker}: {param_desc}"
                    )

                if param_list:
                    params_text = "\\n".join(param_list)
                    tool_descriptions.append(
                        f"**{name}**: {description}\\n{params_text}"
                    )
                else:
                    tool_descriptions.append(f"**{name}**: {description}")
            else:
                tool_descriptions.append(f"**{name}**: {description}")

        return "\\n\\n".join(tool_descriptions)

    @staticmethod
    def _format_history_entry(entry: Dict[str, Any]) -> Optional[str]:
        """Format a single conversation history entry."""
        try:
            if not isinstance(entry, dict):
                return None

            content = entry.get("content")
            if not content:
                return None

            if isinstance(content, dict):
                role = content.get("role", "user")
                text = content.get("content", "")

                if role and text:
                    role_display = role.title()
                    return f"**{role_display}**: {text}"
            elif isinstance(content, str):
                return f"**User**: {content}"

            return None

        except Exception as e:
            logger.warning(f"Failed to format history entry: {e}")
            return None


class ResponseFormatter:
    """Formats responses and output."""

    @staticmethod
    def format_final_response(
        response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format the final response for output.

        Args:
            response: Raw response text
            metadata: Optional metadata to include

        Returns:
            Formatted response
        """
        if not response:
            return "I apologize, but I don't have a response to provide."

        # Clean up the response
        formatted = ResponseFormatter._clean_response_text(response)

        # Add metadata if requested and available
        if metadata and metadata.get("include_metadata", False):
            metadata_text = ResponseFormatter._format_metadata(metadata)
            if metadata_text:
                formatted += f"\\n\\n{metadata_text}"

        return formatted

    @staticmethod
    def format_error_response(error: Exception, context: Optional[str] = None) -> str:
        """
        Format an error response for user-friendly output.

        Args:
            error: The exception that occurred
            context: Optional context about when the error occurred

        Returns:
            Formatted error response
        """
        error_msg = str(error) if error else "An unknown error occurred"

        if context:
            return f"I encountered an error {context}: {error_msg}. Please try again or rephrase your request."
        else:
            return f"I encountered an error: {error_msg}. Please try again or rephrase your request."

    @staticmethod
    def format_tool_result(tool_name: str, result: Any, success: bool = True) -> str:
        """
        Format tool execution result.

        Args:
            tool_name: Name of the executed tool
            result: Tool execution result
            success: Whether execution was successful

        Returns:
            Formatted tool result
        """
        if success:
            if isinstance(result, (dict, list)):
                try:
                    result_text = json.dumps(result, indent=2)
                    return f"**{tool_name}** result:\\n```json\\n{result_text}\\n```"
                except Exception:
                    return f"**{tool_name}** result: {str(result)}"
            else:
                return f"**{tool_name}** result: {str(result)}"
        else:
            return f"**{tool_name}** failed: {str(result)}"

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Clean and format response text."""
        if not text:
            return text

        # Remove excessive whitespace
        cleaned = re.sub(r"\\n\\s*\\n\\s*\\n+", "\\n\\n", text)
        cleaned = re.sub(r"[ \\t]+", " ", cleaned)

        # Remove leading/trailing whitespace from lines
        lines = [line.rstrip() for line in cleaned.split("\\n")]
        cleaned = "\\n".join(lines)

        # Ensure proper sentence spacing
        cleaned = re.sub(r"([.!?])([A-Z])", r"\\1 \\2", cleaned)

        return cleaned.strip()

    @staticmethod
    def _format_metadata(metadata: Dict[str, Any]) -> str:
        """Format metadata for inclusion in response."""
        try:
            filtered_metadata = {
                k: v
                for k, v in metadata.items()
                if k not in ["include_metadata"] and not k.startswith("_")
            }

            if not filtered_metadata:
                return ""

            metadata_lines = []
            for key, value in filtered_metadata.items():
                key_display = key.replace("_", " ").title()
                if isinstance(value, (dict, list)):
                    value_display = json.dumps(value, indent=2)
                    metadata_lines.append(
                        f"**{key_display}**:\\n```\\n{value_display}\\n```"
                    )
                else:
                    metadata_lines.append(f"**{key_display}**: {value}")

            return "---\\n### Metadata\\n" + "\\n\\n".join(metadata_lines)

        except Exception as e:
            logger.warning(f"Failed to format metadata: {e}")
            return ""


class DataFormatter:
    """Formats data structures and objects."""

    @staticmethod
    def format_memory_unit(memory_unit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a memory unit for storage or retrieval.

        Args:
            memory_unit: Raw memory unit data

        Returns:
            Formatted memory unit
        """
        formatted = memory_unit.copy()

        # Ensure required fields
        if "timestamp" not in formatted:
            formatted["timestamp"] = datetime.now().isoformat()

        if "id" not in formatted:
            import uuid

            formatted["id"] = str(uuid.uuid4())

        # Normalize content structure
        if "content" in formatted:
            content = formatted["content"]
            if isinstance(content, str):
                formatted["content"] = {"text": content}
            elif not isinstance(content, dict):
                formatted["content"] = {"data": content}

        return formatted

    @staticmethod
    def format_tool_metadata(
        tool_func, tool_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Format tool metadata from a function.

        Args:
            tool_func: Tool function to analyze
            tool_name: Optional custom tool name

        Returns:
            Formatted tool metadata
        """
        import inspect

        try:
            # Get function info
            name = tool_name or tool_func.__name__
            doc = inspect.getdoc(tool_func) or "No description available"
            sig = inspect.signature(tool_func)

            # Build parameters info
            parameters = {}
            required_params = []

            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_info = {
                    "type": "string",  # Default
                    "description": f"Parameter {param_name}",
                }

                # Try to infer type from annotation
                if param.annotation != inspect.Parameter.empty:
                    param_type = param.annotation
                    if param_type == int:
                        param_info["type"] = "integer"
                    elif param_type == float:
                        param_info["type"] = "number"
                    elif param_type == bool:
                        param_info["type"] = "boolean"
                    elif param_type == list:
                        param_info["type"] = "array"
                    elif param_type == dict:
                        param_info["type"] = "object"

                # Check if required (no default value)
                if param.default == inspect.Parameter.empty:
                    required_params.append(param_name)
                else:
                    param_info["default"] = param.default

                parameters[param_name] = param_info

            return {
                "name": name,
                "description": doc,
                "parameters": parameters,
                "required": required_params,
            }

        except Exception as e:
            logger.error(f"Failed to format tool metadata: {e}")
            return {
                "name": tool_name or "unknown",
                "description": "Tool metadata unavailable",
                "parameters": {},
                "required": [],
            }
