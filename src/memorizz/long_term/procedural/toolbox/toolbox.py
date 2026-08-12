# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import importlib
import inspect
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....llms.llm_provider import LLMProvider
from ....memory_provider import MemoryProvider
from ....tooling import callable_json_schema, policy_for_callable
from .tool_schema import ToolSchemaType


def get_openai_default() -> LLMProvider:
    """Initializes the default OpenAI client lazily."""
    from ....llms.openai import OpenAI

    return OpenAI()


class Toolbox:
    """A toolbox for managing and retrieving tools using a memory provider."""

    def __init__(
        self,
        memory_provider: MemoryProvider,
        llm_provider: Optional[LLMProvider] = None,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize the toolbox.

        The LLM provider is optional and is created lazily only when a caller
        explicitly requests ``augment=True``.

        Parameters:
        -----------
        memory_provider : MemoryProvider
            The memory provider for storing and retrieving tools.
        llm_provider : LLMProvider, optional
            Optional LLM provider for augmented metadata generation.
        agent_id : str, optional
            Agent to scope stored tools to. When set, every ``register_tool``
            write stamps ``agent_id`` on the TOOLBOX row so the playground's
            toolbox-memory pane can find it.
        """
        self.memory_provider = memory_provider

        self.llm_provider = llm_provider

        self.agent_id = agent_id

        # In-memory storage of functions
        self._tools: Dict[str, Callable] = {}
        self._tools_by_name: Dict[str, Callable] = {}

    @classmethod
    def from_functions(
        cls,
        functions: Iterable[Callable[..., Any]],
        *,
        memory_provider: MemoryProvider,
        llm_provider: Optional[LLMProvider] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        augment: bool = False,
        persist: bool = True,
    ) -> "Toolbox":
        """Register and bind trusted callables without serializing executable code."""
        toolbox = cls(
            memory_provider=memory_provider,
            llm_provider=llm_provider,
            agent_id=agent_id,
        )
        for function in functions:
            toolbox.register_tool(
                function,
                augment=augment,
                persist=persist,
                user_id=user_id,
            )
        return toolbox

    @property
    def tools(self) -> Dict[str, Callable]:
        """Return the callables registered in this process, keyed by tool ID."""
        return dict(self._tools)

    def register_tool(
        self,
        func: Optional[Callable] = None,
        augment: bool = False,
        *,
        persist: bool = True,
        user_id: Optional[str] = None,
        aliases: Optional[Iterable[str]] = None,
        deprecated_arguments: Optional[Mapping[str, str]] = None,
    ) -> Union[str, Callable]:
        """
        Register a function as a tool in the toolbox.

        Parameters:
        -----------
        func : Callable, optional
            The function to register as a tool. If None, returns a decorator.
        augment : bool, optional
            Whether to augment the tool docstring and generate synthetic queries
            using the configured LLM provider.
        Returns:
        --------
        Union[str, Callable]
            If func is provided, returns the tool ID. Otherwise returns a decorator.
        """

        def decorator(f: Callable) -> str:
            docstring = f.__doc__ or ""
            signature = str(inspect.signature(f))
            object_id_str = str(uuid.uuid4())

            if augment:
                augmented_docstring = self._augment_docstring(docstring)
                queries = self._generate_queries(augmented_docstring)
                tool_data = self._normalize_tool_metadata(
                    f,
                    self._get_tool_metadata(f),
                    description=augmented_docstring,
                )
                embedding = get_embedding(
                    f"{f.__name__} {augmented_docstring} {signature} {queries}"
                )
                tool_dict = {
                    "_id": object_id_str,
                    "embedding": embedding,
                    "queries": queries,
                    **tool_data,
                }
            else:
                # Deterministic registration is metadata-only by default. The
                # provider may generate an embedding lazily at persistence or
                # retrieval time; no global LLM/embedding client is constructed
                # merely to bind trusted Python callables.
                tool_dict = {
                    "_id": object_id_str,
                    **self._metadata_from_callable(f),
                }

            if self.agent_id:
                tool_dict["agent_id"] = self.agent_id
            tool_dict["user_id"] = user_id
            declared_aliases = list(aliases or [])
            if declared_aliases:
                tool_dict["aliases"] = [str(item) for item in declared_aliases]
            if deprecated_arguments:
                tool_dict["deprecated_arguments"] = dict(deprecated_arguments)
            import_reference = self._import_reference(f)
            if import_reference:
                tool_dict["import_reference"] = import_reference

            if persist:
                self.memory_provider.store(
                    tool_dict, memory_store_type=MemoryType.TOOLBOX
                )
            self._tools[object_id_str] = f
            self._tools_by_name[f.__name__] = f
            return object_id_str

        if func is None:
            return decorator
        return decorator(func)

    @staticmethod
    def _metadata_from_callable(func: Callable) -> Dict[str, Any]:
        """Build stable tool metadata without spending an LLM call."""
        input_schema = callable_json_schema(func)
        docstring = inspect.getdoc(func) or ""
        policy = policy_for_callable(func)
        return {
            "name": func.__name__,
            "description": docstring,
            "signature": str(inspect.signature(func)),
            "docstring": docstring,
            "tool_type": "function",
            "parameters": input_schema["properties"],
            "required": input_schema.get("required", []),
            "input_schema": input_schema,
            "tool_policy": policy.to_dict(),
            "aliases": list(policy.aliases),
            "deprecated_arguments": dict(policy.deprecated_arguments),
        }

    @staticmethod
    def _import_reference(func: Callable[..., Any]) -> Optional[str]:
        module = str(getattr(func, "__module__", "") or "")
        qualname = str(getattr(func, "__qualname__", "") or "")
        if not module or not qualname or "<locals>" in qualname:
            return None
        return f"{module}:{qualname}"

    @staticmethod
    def _load_import_reference(reference: str) -> Callable[..., Any]:
        module_name, separator, qualname = str(reference).partition(":")
        if not separator or not module_name or not qualname or "<locals>" in qualname:
            raise ValueError("Trusted tool references must use module:qualified_name")
        value: Any = importlib.import_module(module_name)
        for component in qualname.split("."):
            value = getattr(value, component)
        if not callable(value):
            raise TypeError(f"Trusted tool reference '{reference}' is not callable")
        return value

    def bind_callable(
        self,
        *,
        function: Optional[Callable[..., Any]] = None,
        import_reference: Optional[str] = None,
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> Callable[..., Any]:
        """Rebind persisted metadata through an explicit trusted reference."""
        bound = function or self._load_import_reference(str(import_reference or ""))
        if not callable(bound):
            raise TypeError("function must be callable")
        if tool_id:
            self._tools[str(tool_id)] = bound
        self._tools_by_name[str(tool_name or bound.__name__)] = bound
        return bound

    @classmethod
    def _normalize_tool_metadata(
        cls,
        func: Callable,
        metadata: Any,
        *,
        description: str,
    ) -> Dict[str, Any]:
        """Flatten provider-specific metadata into the Toolbox store shape."""
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump()
        if not isinstance(metadata, dict):
            metadata = {}

        function_metadata = metadata.get("function")
        if isinstance(function_metadata, dict):
            metadata = function_metadata

        normalized = cls._metadata_from_callable(func)
        normalized["name"] = str(metadata.get("name") or func.__name__)
        normalized["description"] = str(
            metadata.get("description") or description or normalized["description"]
        )
        normalized["docstring"] = description or normalized["docstring"]

        parameters = metadata.get("parameters")
        if isinstance(parameters, list):
            parameters = {
                str(item.get("name")): {
                    "type": item.get("type", "string"),
                    "description": item.get("description", ""),
                }
                for item in parameters
                if isinstance(item, dict) and item.get("name")
            }
        if isinstance(parameters, dict):
            schema = parameters if parameters.get("type") == "object" else None
            normalized["parameters"] = parameters.get("properties", parameters)
            if schema:
                normalized["input_schema"] = dict(schema)

        required = metadata.get("required")
        if isinstance(required, list):
            normalized["required"] = [str(name) for name in required]
        normalized.setdefault(
            "input_schema",
            {
                "type": "object",
                "properties": normalized["parameters"],
                "required": normalized["required"],
                "additionalProperties": False,
            },
        )
        normalized["input_schema"]["additionalProperties"] = False
        return normalized

    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a single tool by its name.

        Parameters:
        -----------
        name : str
            The name of the tool to retrieve.

        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        """
        # Note: This method only retrieves metadata from the provider.
        # Use get_function_by_id to retrieve the callable function.
        return self.memory_provider.retrieve_by_name(
            name, memory_store_type=MemoryType.TOOLBOX
        )

    def get_tool_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool's metadata by its id.

        Parameters:
        -----------
        id : str
            The id of the tool to retrieve.

        Returns:
        --------
        Dict[str, Any]
            The tool data, or None if not found.
        """
        return self.memory_provider.retrieve_by_id(
            id, memory_store_type=MemoryType.TOOLBOX
        )

    def get_most_similar_tools(
        self,
        query: str,
        limit: int = 5,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get the most similar tools to a query using vector search.

        Parameters:
        -----------
        query : str
            The query to search for.
        limit : int, optional
            The maximum number of tools to return.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of the most similar tool metadata.
        """
        effective_agent_id = agent_id if agent_id is not None else self.agent_id
        kwargs = {"user_id": user_id, "agent_id": effective_agent_id}
        try:
            rows = self.memory_provider.retrieve_by_query(
                query,
                memory_store_type=MemoryType.TOOLBOX,
                limit=max(int(limit), 1),
                **kwargs,
            )
        except TypeError:
            # Compatibility providers may not accept scopes. Deliberately
            # over-fetch, then filter before the final top-k.
            rows = self.memory_provider.retrieve_by_query(
                query,
                memory_store_type=MemoryType.TOOLBOX,
                limit=max(int(limit) * 5, 25),
            )
        filtered = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if row.get("user_id") != user_id:
                continue
            if effective_agent_id is not None and row.get("agent_id") not in {
                None,
                effective_agent_id,
            }:
                continue
            filtered.append(row)
            if len(filtered) >= max(int(limit), 1):
                break
        return filtered

    def delete_tool_by_name(self, name: str) -> bool:
        """
        Delete a tool from the toolbox by name.

        Parameters:
        -----------
        name : str
            The name of the tool to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        tool_data = self.memory_provider.retrieve_by_name(
            name, memory_store_type=MemoryType.TOOLBOX
        )
        if tool_data and "_id" in tool_data:
            tool_id = str(tool_data["_id"])
            if tool_id in self._tools:
                del self._tools[tool_id]

        return self.memory_provider.delete_by_name(
            name, memory_store_type=MemoryType.TOOLBOX
        )

    def delete_tool_by_id(self, id: str) -> bool:
        """
        Delete a tool from the toolbox by id.

        Parameters:
        -----------
        id : str
            The id of the tool to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if id in self._tools:
            del self._tools[id]

        return self.memory_provider.delete_by_id(
            id, memory_store_type=MemoryType.TOOLBOX
        )

    def delete_all(self) -> bool:
        """
        Delete all tools in the toolbox.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        self._tools.clear()
        return self.memory_provider.delete_all(memory_store_type=MemoryType.TOOLBOX)

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools in the toolbox from the memory provider.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of all tool metadata from the memory provider.
        """
        return self.memory_provider.list_all(memory_store_type=MemoryType.TOOLBOX)

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """
        List tools that have both metadata in the database AND are callable in the current session.

        Returns:
        --------
        List[Dict[str, Any]]
            A list of tool metadata for tools with available functions.
        """
        available_tools = []
        for tool_id in self._tools:
            meta = self.get_tool_by_id(tool_id)
            if meta:
                available_tools.append(meta)
        return available_tools

    def get_function_by_id(self, tool_id: str) -> Optional[Callable]:
        """
        Get the actual executable function by its tool ID.

        Parameters:
        -----------
        tool_id : str
            The ID of the tool whose function to retrieve.

        Returns:
        --------
        Optional[Callable]
            The function object, or None if not found in the current session.
        """
        return self._tools.get(tool_id)

    def get_function_by_name(self, tool_name: str) -> Optional[Callable]:
        """Return a callable explicitly bound under a persisted tool name."""
        return self._tools_by_name.get(str(tool_name))

    def update_tool_by_id(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a tool's metadata in the memory provider by id.

        Parameters:
        -----------
        id : str
            The id of the tool to update.
        data : Dict[str, Any]
            The data to update the tool with.

        Returns:
        --------
        bool
            True if the update was successful, False otherwise.
        """
        return self.memory_provider.update_by_id(
            id, data, memory_store_type=MemoryType.TOOLBOX
        )

    # --- Internal methods now use the configured self.llm_provider ---

    def _get_tool_metadata(self, func: Callable) -> ToolSchemaType:
        """Get the metadata for a tool using the configured LLM provider."""
        return self._require_llm_provider().get_tool_metadata(func)

    def _augment_docstring(self, docstring: str) -> str:
        """Augment the docstring using the configured LLM provider."""
        return self._require_llm_provider().augment_docstring(docstring)

    def _generate_queries(self, docstring: str) -> List[str]:
        """Generate queries for the tool using the configured LLM provider."""
        return self._require_llm_provider().generate_queries(docstring)

    def _require_llm_provider(self) -> LLMProvider:
        if self.llm_provider is None:
            self.llm_provider = get_openai_default()
        return self.llm_provider
