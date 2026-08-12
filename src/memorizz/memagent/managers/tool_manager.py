# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tool management functionality for MemAgent."""

import inspect
import json
import logging
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Union

from ...long_term.procedural.toolbox.toolbox import Toolbox
from ...long_term.procedural.workflow.workflow import Workflow, WorkflowOutcome
from ...tooling import callable_json_schema, policy_for_callable

logger = logging.getLogger(__name__)


def _coerce_jsonable(value: Any) -> Any:
    """Recursively coerce a value into a JSON-serializable primitive.

    Tool metadata persisted in Oracle has CLOB ``description`` /
    ``docstring`` fields, which ``oracledb`` returns as LOB objects. If
    these are left in tool definitions and later passed to the OpenAI
    SDK's ``chat.completions.create(tools=...)`` call, the SDK's
    internal ``json.dumps`` fails with
    ``Object of type LOB is not JSON serializable``.

    Mirrors the helper in ``memagent/core.py`` — duplicated here because
    the tool manager shouldn't import from core (circular).
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            read_value = reader()
        except Exception:
            return str(value)
        return _coerce_jsonable(read_value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    return str(value)


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

            for tool_id, tool_data in toolbox.tools.items():
                if callable(tool_data):
                    if self.add_tool(tool_data):
                        tools_loaded += 1
                elif self._register_tool_from_data(tool_id, tool_data):
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
            input_schema = callable_json_schema(func)
            policy = policy_for_callable(func)

            return {
                "_id": func.__name__,  # Use function name as ID
                "name": func.__name__,
                "description": doc,
                "signature": str(sig),  # Add function signature
                "docstring": doc,  # Add docstring explicitly
                "parameters": input_schema["properties"],
                "required": input_schema.get("required", []),
                "input_schema": input_schema,
                "tool_policy": policy.to_dict(),
                "aliases": list(policy.aliases),
                "deprecated_arguments": dict(policy.deprecated_arguments),
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
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
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
                    # Bind before calling so unknown/missing arguments cannot be
                    # silently accepted by a compatibility wrapper.
                    inspect.signature(func).bind(**normalized_arguments)
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
            # Return all tool metadata. Accept both the canonical wrapped
            # shape (``{"metadata": {...}, "function": f}``) and a flat
            # metadata dict — the latter can arise from tool data loaded
            # from persistence before ``_register_tool_from_data`` has had
            # a chance to normalize it.
            all_metadata = []
            for tool_id, tool_data in self.tools.items():
                if not isinstance(tool_data, dict):
                    continue
                metadata = tool_data.get("metadata")
                if not isinstance(metadata, dict):
                    # Flat dict fallback — treat the entry itself as metadata.
                    if "name" in tool_data or "parameters" in tool_data:
                        metadata = tool_data
                    else:
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

    def get_tool_callable(self, tool_name: str) -> Optional[Callable[..., Any]]:
        """Return the trusted in-process callable for a registered tool."""
        value = self.tools.get(str(tool_name))
        if not isinstance(value, dict):
            return None
        function = value.get("function")
        return function if callable(function) else None

    def get_tool_policy(self, tool_name: str) -> Dict[str, Any]:
        """Return explicit governance metadata for one tool."""
        metadata = self.get_tool_metadata(str(tool_name))
        if isinstance(metadata, dict) and isinstance(metadata.get("tool_policy"), dict):
            return dict(metadata["tool_policy"])
        function = self.get_tool_callable(str(tool_name))
        return policy_for_callable(function).to_dict()

    def _register_tool_from_data(self, tool_id: str, tool_data: Dict) -> bool:
        """Register a tool from raw data.

        Normalizes to the canonical wrapped shape
        ``{"metadata": {...}, "function": f | None, "type": "function"}``
        regardless of whether ``tool_data`` is already wrapped (as
        ``_add_function_tool`` produces) or is a flat metadata dict (as
        persisted Toolbox entries look after serialize/deserialize).

        If a callable for this tool was previously registered in-memory,
        we preserve the function — otherwise a later replay of persisted
        flat metadata would clobber the runtime callable and the tool
        would silently stop working.

        Metadata values coming from persistence are also run through the
        LOB coercion helper: when the backing store is Oracle, CLOB
        fields (``description``, ``docstring``, etc.) surface as LOB
        objects. Those are JSON-unserializable and blow up the first
        time the streaming pipeline tries to send tools to the LLM —
        with a generic "Object of type LOB is not JSON serializable"
        that gives no hint about tool metadata being the culprit.
        """
        try:
            if not isinstance(tool_data, dict):
                logger.warning("Tool data for %s is not a dict", tool_id)
                return False

            existing = (
                self.tools.get(tool_id)
                if isinstance(self.tools.get(tool_id), dict)
                else None
            )
            existing_func = existing.get("function") if existing else None

            # Two shapes supported:
            #   wrapped: already has "metadata" key
            #   flat:    the dict itself IS the metadata (name/parameters/...)
            if isinstance(tool_data.get("metadata"), dict):
                metadata = _coerce_jsonable(tool_data["metadata"])
                function = tool_data.get("function", existing_func)
                tool_type = tool_data.get("type", "function")
                workflow = tool_data.get("workflow")
            else:
                metadata = _coerce_jsonable(tool_data)
                function = existing_func
                tool_type = tool_data.get("type", "function")
                workflow = tool_data.get("workflow")

            entry: Dict[str, Any] = {
                "metadata": metadata,
                "function": function,
                "type": tool_type,
            }
            if workflow is not None:
                entry["workflow"] = workflow

            self.tools[tool_id] = entry
            self._tool_metadata_cache[tool_id] = metadata
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
