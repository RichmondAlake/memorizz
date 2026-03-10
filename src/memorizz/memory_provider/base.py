# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from memorizz.memagent import MemAgent


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initialize the memory provider with configuration settings."""

    @abstractmethod
    def store(
        self,
        data: Dict[str, Any] = None,
        memory_store_type: str = None,
        memory_id: str = None,
        memory_unit: Any = None,
    ) -> str:
        """
        Store data in the memory provider.

        Parameters:
        -----------
        data : Dict[str, Any], optional
            Data dictionary to store (legacy parameter)
        memory_store_type : str, optional
            Type of memory store (legacy parameter)
        memory_id : str, optional
            Memory ID to associate with (new parameter)
        memory_unit : MemoryUnit, optional
            Memory unit object to store (new parameter)
        """

    @abstractmethod
    def retrieve_by_query(
        self,
        query: Dict[str, Any],
        memory_store_type: str = None,
        limit: int = 1,
        memory_id: str = None,
        memory_type: str = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from the memory provider.

        Parameters:
        -----------
        query : Dict[str, Any] or str
            Search query (dict for filter queries, str for semantic search)
        memory_store_type : str, optional
            Type of memory store (legacy parameter name)
        memory_type : str or MemoryType, optional
            Type of memory store (new parameter name, takes precedence over memory_store_type)
        memory_id : str, optional
            Filter results to specific memory_id
        limit : int
            Maximum number of results to return
        **kwargs
            Additional provider-specific parameters
        """

    @abstractmethod
    def retrieve_by_id(
        self, id: str, memory_store_type: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider by id."""

    @abstractmethod
    def retrieve_by_name(
        self, name: str, memory_store_type: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider by name."""

    @abstractmethod
    def delete_by_id(self, id: str, memory_store_type: str) -> bool:
        """Delete a document from the memory provider by id."""

    @abstractmethod
    def delete_by_name(self, name: str, memory_store_type: str) -> bool:
        """Delete a document from the memory provider by name."""

    @abstractmethod
    def delete_all(self, memory_store_type: str) -> bool:
        """Delete all documents within a memory store type in the memory provider."""

    @abstractmethod
    def list_all(self, memory_store_type: str) -> List[Dict[str, Any]]:
        """List all documents within a memory store type in the memory provider."""

    @abstractmethod
    def retrieve_conversation_history_ordered_by_timestamp(
        self, memory_id: str, memory_type: str = None, limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The memory ID to retrieve history for
        memory_type : str or MemoryType, optional
            Type of memory (typically CONVERSATION_MEMORY)
        limit : int, optional
            Maximum number of entries to return
        """

    @abstractmethod
    def update_by_id(
        self, id: str, data: Dict[str, Any], memory_store_type: str
    ) -> bool:
        """Update a document in a memory store type in the memory provider by id."""

    @abstractmethod
    def close(self) -> None:
        """Close the connection to the memory provider."""

    @abstractmethod
    def store_memagent(self, memagent: "MemAgent") -> str:
        """Store a memagent in the memory provider."""

    @abstractmethod
    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """Delete a memagent from the memory provider."""

    @abstractmethod
    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """Update the memory_ids of a memagent in the memory provider."""

    @abstractmethod
    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """Delete the memory_ids of a memagent in the memory provider."""

    @abstractmethod
    def list_memagents(self) -> List[Dict[str, Any]]:
        """List all memagents in the memory provider."""
