# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tool management functionality for MemAgent."""

import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Union

from ...long_term_memory.procedural.toolbox.toolbox import Toolbox
from ...long_term_memory.procedural.workflow.workflow import Workflow, WorkflowOutcome

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages tool registration, execution, and lifecycle for MemAgent.

    This class encapsulates all tool-related functionality that was
    previously embedded in the main MemAgent class.
    """

    def __init__(self, memory_provider=None):
        """
        Initialize the tool manager.

        Args:
            memory_provider: Optional memory provider for persistent tool storage.
        """
        self.memory_provider = memory_provider
        self.tools = {}  # In-memory tool registry
        self.toolbox = None  # Optional toolbox instance
        self._tool_metadata_cache = {}

    def initialize_from_toolbox(self, toolbox: Toolbox) -> int:
        """
        Initialize tools from a Toolbox instance.

        Args:
            toolbox: The Toolbox instance containing tools.

        Returns:
            Number of tools successfully loaded.
        """
        try:
            self.toolbox = toolbox
            tools_loaded = 0

            # Extract tools from toolbox
            if hasattr(toolbox, "tools") and toolbox.tools:
                for tool_id, tool_data in toolbox.tools.items():
                    if self._register_tool_from_data(tool_id, tool_data):
                        tools_loaded += 1

            logger.info(f"Loaded {tools_loaded} tools from toolbox")
            return tools_loaded

        except Exception as e:
            logger.error(f"Failed to initialize from toolbox: {e}")
            return 0

    def add_tool(
        self,
        tool: Union[Callable, Dict[str, Any], str],
        persist: bool = False,
        tool_type: str = "function",
    ) -> bool:
        """
        Add a tool to the manager.

        Args:
            tool: The tool to add (function, dict, or tool ID).
            persist: Whether to persist the tool to storage.
            tool_type: Type of tool ("function", "workflow", etc.).

        Returns:
            True if successfully added, False otherwise.
        """
        try:
            # Handle different tool types
            if callable(tool):
                return self._add_function_tool(tool, persist)
            elif isinstance(tool, dict):
                return self._add_dict_tool(tool, persist)
            elif isinstance(tool, str):
                return self._add_tool_by_id(tool)
            else:
                logger.warning(f"Unsupported tool type: {type(tool)}")
                return False

        except Exception as e:
            logger.error(f"Failed to add tool: {e}")
            return False

    def _add_function_tool(self, func: Callable, persist: bool) -> bool:
        """Add a Python function as a tool."""
        try:
            # Generate metadata from function
            metadata = self._generate_tool_metadata(func)
            tool_id = metadata["name"]

            # Register the tool
            self.tools[tool_id] = {
                "metadata": metadata,
                "function": func,
                "type": "function",
            }

            # Cache metadata
            self._tool_metadata_cache[tool_id] = metadata

            # Persist if requested
            if persist and self.memory_provider:
                self._persist_tool(tool_id, metadata)

            logger.info(f"Added function tool: {tool_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add function tool: {e}")
            return False

    def _generate_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Generate metadata for a function tool."""
        try:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or "No description available"

            # Build parameter schema
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                param_info = {
                    "type": "string",  # Default type
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

                parameters[param_name] = param_info

            return {
                "_id": func.__name__,  # Use function name as ID
                "name": func.__name__,
                "description": doc,
                "signature": str(sig),  # Add function signature
                "docstring": doc,  # Add docstring explicitly
                "parameters": parameters,
                # Only parameters without defaults are required. This keeps tool
                # schemas sane for LLM tool calling.
                "required": [
                    param_name
                    for param_name, param in sig.parameters.items()
                    if param_name != "self"
                    and param.default == inspect.Parameter.empty
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ],
                "type": "function",
            }

        except Exception as e:
            logger.error(f"Failed to generate tool metadata: {e}")
            return {
                "_id": func.__name__,
                "name": func.__name__,
                "description": "Tool function",
                "signature": "",
                "docstring": "",
                "parameters": {},
                "required": [],
                "type": "function",
            }

    def execute_tool(
        self, tool_name: str, arguments: Any
    ) -> tuple[Any, Optional[WorkflowOutcome]]:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.

        Returns:
            Tuple of (result, workflow_outcome).
        """
        try:
            normalized_arguments: Dict[str, Any] = {}
            if arguments is None:
                normalized_arguments = {}
            elif isinstance(arguments, str):
                parsed_arguments = json.loads(arguments)
                if parsed_arguments is None:
                    normalized_arguments = {}
                elif isinstance(parsed_arguments, dict):
                    normalized_arguments = parsed_arguments
                else:
                    return (
                        "Error: Tool arguments must decode to a JSON object.",
                        None,
                    )
            elif isinstance(arguments, dict):
                normalized_arguments = arguments
            elif isinstance(arguments, Mapping):
                normalized_arguments = dict(arguments)
            else:
                return "Error: Tool arguments must be a mapping.", None

            if tool_name not in self.tools:
                logger.error(f"Tool not found: {tool_name}")
                return f"Error: Tool '{tool_name}' not found", None

            tool_data = self.tools[tool_name]
            tool_type = tool_data.get("type", "function")

            if tool_type == "function":
                func = tool_data.get("function")
                if func:
                    result = func(**normalized_arguments)
                    return result, None
                else:
                    return "Error: Tool function not available", None

            elif tool_type == "workflow":
                workflow = tool_data.get("workflow")
                if isinstance(workflow, Workflow):
                    outcome = workflow.execute(normalized_arguments)
                    return outcome.result, outcome
                else:
                    return "Error: Workflow not available", None

            else:
                return f"Error: Unknown tool type '{tool_type}'", None

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error executing tool: {str(e)}", None

    def get_tool_metadata(
        self, tool_name: Optional[str] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Get metadata for tools.

        Args:
            tool_name: If provided, get metadata for specific tool.
                      Otherwise, get metadata for all tools.

        Returns:
            Tool metadata dict or list of dicts.
        """
        if tool_name:
            if tool_name in self._tool_metadata_cache:
                return self._tool_metadata_cache[tool_name]
            elif tool_name in self.tools:
                return self.tools[tool_name].get("metadata", {})
            else:
                return {}
        else:
            # Return all tool metadata
            all_metadata = []
            for tool_id, tool_data in self.tools.items():
                # Only expose metadata that can be used for tool calling.
                metadata = (
                    tool_data.get("metadata") if isinstance(tool_data, dict) else None
                )
                if not isinstance(metadata, dict):
                    continue
                if not str(metadata.get("name", "")).strip():
                    continue
                all_metadata.append(metadata)
            return all_metadata

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the manager.

        Args:
            tool_name: Name of the tool to remove.

        Returns:
            True if successfully removed, False otherwise.
        """
        try:
            if tool_name in self.tools:
                del self.tools[tool_name]

                if tool_name in self._tool_metadata_cache:
                    del self._tool_metadata_cache[tool_name]

                logger.info(f"Removed tool: {tool_name}")
                return True
            else:
                logger.warning(f"Tool not found for removal: {tool_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to remove tool: {e}")
            return False

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names.
        """
        return list(self.tools.keys())

    def _register_tool_from_data(self, tool_id: str, tool_data: Dict) -> bool:
        """Register a tool from raw data."""
        try:
            self.tools[tool_id] = tool_data

            if "metadata" in tool_data:
                self._tool_metadata_cache[tool_id] = tool_data["metadata"]

            return True
        except Exception as e:
            logger.error(f"Failed to register tool {tool_id}: {e}")
            return False

    def _add_dict_tool(self, tool_dict: Dict, persist: bool) -> bool:
        """Add a tool from dictionary specification."""
        try:
            tool_id = tool_dict.get("name", tool_dict.get("id"))
            if not tool_id:
                logger.error("Tool dict missing name/id")
                return False

            return self._register_tool_from_data(tool_id, tool_dict)

        except Exception as e:
            logger.error(f"Failed to add dict tool: {e}")
            return False

    def _add_tool_by_id(self, tool_id: str) -> bool:
        """Add a tool by loading it from storage."""
        try:
            if not self.memory_provider:
                logger.error("No memory provider available for loading tools")
                return False

            # Load tool from storage
            tool_data = self.memory_provider.retrieve_by_id(tool_id)
            if tool_data:
                return self._register_tool_from_data(tool_id, tool_data)
            else:
                logger.error(f"Tool not found in storage: {tool_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to load tool {tool_id}: {e}")
            return False

    def _persist_tool(self, tool_id: str, metadata: Dict) -> bool:
        """Persist a tool to storage."""
        try:
            if not self.memory_provider:
                return False

            # Store tool metadata
            self.memory_provider.store(
                memory_id=tool_id, memory_unit={"type": "tool", "metadata": metadata}
            )

            logger.debug(f"Persisted tool {tool_id} to storage")
            return True

        except Exception as e:
            logger.error(f"Failed to persist tool {tool_id}: {e}")
            return False
