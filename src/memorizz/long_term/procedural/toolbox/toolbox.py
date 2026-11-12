# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import uuid
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....llms.llm_provider import LLMProvider
from ....memory_provider import MemoryProvider
from ....memory_provider.base import user_scope_matches
from ....tooling import callable_json_schema, policy_for_callable
from .tool_schema import ToolSchemaType


def get_openai_default() -> LLMProvider:
    """Initializes the default OpenAI client lazily."""
    from ....llms.openai import OpenAI

    return OpenAI()


#: Example requests written for a tool when it is registered with ``augment``.
AUGMENTED_QUERIES = 6
MAX_AUGMENTED_DESCRIPTION = 900

_AUGMENT_PROMPT = """You write the catalogue entry that lets an AI agent find \
the right tool for a request. The entry is searched by meaning, so it has to \
say what the tool is for in the words people use when they ask.

Tool: {name}{signature}
Author's docstring:
{docstring}

Reply with JSON only:
{{"description": "...", "queries": ["...", "..."]}}

Rules:
- description: 40 to 70 words. Say what the tool does and the situations it \
is for, and include the other words people use for the same thing. Only say \
what it is for: never list what it is not for. Do not invent behaviour, \
arguments or return fields that the docstring and signature do not show.
- queries: {count} requests this tool should handle, as a busy colleague \
would type them: short, informal, each worded differently, some without the \
obvious keyword. No ids, names, amounts or other specific values.
"""


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
            object_id_str = str(uuid.uuid4())

            if augment:
                from ....memory_provider.base import provider_manages_embeddings
                from .indexing import search_text

                description, queries = self._augment(f)
                tool_data = self._metadata_from_callable(f)
                tool_data["description"] = description or tool_data["description"]
                tool_dict = {"_id": object_id_str, "queries": queries, **tool_data}
                if queries:
                    # Written here, not learned from real requests: tool
                    # learning lets a learned example take a synthetic one's place.
                    tool_dict["queries_source"] = "augmentation"
                tool_dict["embedding"] = (
                    None
                    if provider_manages_embeddings(self.memory_provider)
                    else get_embedding(search_text(tool_dict))
                )
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
            if not user_scope_matches(row.get("user_id"), user_id, MemoryType.TOOLBOX):
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

    # --- Internal methods now use the configured self.llm_provider ---

    def _get_tool_metadata(self, func: Callable) -> ToolSchemaType:
        """Get the metadata for a tool using the configured LLM provider."""
        return self._require_llm_provider().get_tool_metadata(func)

    def _augment(self, func: Callable) -> tuple:
        """A catalogue description and example requests for ``func``.

        One model call. The reply has to be JSON with a description and a
        list of requests; anything else falls back to the author's docstring
        with no examples, so a bad reply never becomes the tool's entry. The
        argument schema always comes from the code, never from the model.
        """
        import json
        import logging

        from ....learning.consolidation import text_generator

        docstring = inspect.getdoc(func) or ""
        generate = text_generator(self._require_llm_provider())
        if generate is None:
            return docstring, []
        prompt = _AUGMENT_PROMPT.format(
            name=func.__name__,
            signature=str(inspect.signature(func)),
            docstring=docstring or "(none)",
            count=AUGMENTED_QUERIES,
        )
        try:
            reply = str(generate(prompt) or "")
            start, end = reply.find("{"), reply.rfind("}")
            parsed = json.loads(reply[start : end + 1]) if 0 <= start < end else None
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Tool %s could not be augmented: %s", func.__name__, exc
            )
            return docstring, []
        if not isinstance(parsed, dict):
            return docstring, []
        description = " ".join(str(parsed.get("description") or "").split())
        if not description or len(description) > MAX_AUGMENTED_DESCRIPTION:
            description = docstring
        raw = parsed.get("queries")
        queries = []
        for item in raw if isinstance(raw, list) else []:
            text = " ".join(str(item or "").split())
            if 4 <= len(text) <= 160 and text not in queries:
                queries.append(text)
        return description, queries[:AUGMENTED_QUERIES]

    def _require_llm_provider(self) -> LLMProvider:
        if self.llm_provider is None:
            self.llm_provider = get_openai_default()
        return self.llm_provider
