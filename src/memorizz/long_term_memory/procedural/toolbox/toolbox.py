import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....llms.llm_provider import LLMProvider
from ....memory_provider import MemoryProvider
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
    ):
        """
        Initialize the toolbox.

        This constructor is backward-compatible. If `llm_provider` is not specified,
        it defaults to using the OpenAI client.

        Parameters:
        -----------
        memory_provider : MemoryProvider
            The memory provider for storing and retrieving tools.
        llm_provider : LLMProvider, optional
            The LLM provider for metadata generation. Defaults to OpenAI if None.
        """
        self.memory_provider = memory_provider

        # If no provider is passed, create the default OpenAI client.
        if llm_provider is None:
            self.llm_provider = get_openai_default()
        else:
            self.llm_provider = llm_provider

        # In-memory storage of functions
        self._tools: Dict[str, Callable] = {}

    def register_tool(
        self, func: Optional[Callable] = None, augment: bool = False
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
            object_id = uuid.uuid4()
            object_id_str = str(object_id)

            if augment:
                # Use the configured LLM provider for augmentation
                augmented_docstring = self._augment_docstring(docstring)
                queries = self._generate_queries(augmented_docstring)
                embedding = get_embedding(
                    f"{f.__name__} {augmented_docstring} {signature} {queries}"
                )
                tool_data = self._get_tool_metadata(f)

                tool_dict = {
                    "_id": object_id,
                    "embedding": embedding,
                    "queries": queries,
                    **tool_data.model_dump(),
                }
            else:
                embedding = get_embedding(f"{f.__name__} {docstring} {signature}")
                tool_data = self._get_tool_metadata(f)

                tool_dict = {
                    "_id": object_id,
                    "embedding": embedding,
                    **tool_data.model_dump(),
                }

            self.memory_provider.store(tool_dict, memory_store_type=MemoryType.TOOLBOX)
            self._tools[object_id_str] = f
            return object_id_str

        if func is None:
            return decorator
        return decorator(func)

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
        self, query: str, limit: int = 5
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
        return self.memory_provider.retrieve_by_query(
            query, memory_store_type=MemoryType.TOOLBOX, limit=limit
        )

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
        return self.llm_provider.get_tool_metadata(func)

    def _augment_docstring(self, docstring: str) -> str:
        """Augment the docstring using the configured LLM provider."""
        return self.llm_provider.augment_docstring(docstring)

    def _generate_queries(self, docstring: str) -> List[str]:
        """Generate queries for the tool using the configured LLM provider."""
        return self.llm_provider.generate_queries(docstring)
