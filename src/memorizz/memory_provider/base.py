# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from memorizz.memagent import MemAgent


# Sentinel so "user_id not supplied" is distinguishable from an explicit None
# (matching the MongoDB provider's _MONGO_UNSET convention).
_UNSET = object()


def filter_tool_log_rows(
    rows: List[Dict[str, Any]],
    *,
    memory_id: Optional[str] = None,
    user_id: Any = _UNSET,
    thread_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Filter + sort + limit raw tool_log rows, most-recent first.

    Shared by ``MemoryManager.list_tool_logs``'s provider-agnostic fallback and
    by the filesystem / Oracle providers' ``list_tool_logs`` (which read every
    tool_log row via ``list_all`` and then narrow in Python). Centralising it
    keeps the scoping rules — crucially the ``thread_id`` filter that keeps the
    tool-log digest from leaking other threads' tool calls — identical
    everywhere.

    ``thread_id`` matching is exact against the row's stored ``thread_id``
    (``store_tool_log`` writes ``thread_id or ""``); when ``thread_id`` is
    ``None`` the filter is skipped (back-compat: all threads).
    """
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if memory_id and row.get("memory_id") != memory_id:
            continue
        if user_id is not _UNSET and row.get("user_id") != user_id:
            continue
        if thread_id is not None and (row.get("thread_id") or "") != thread_id:
            continue
        out.append(row)
    out.sort(key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    if limit and limit > 0:
        out = out[:limit]
    return out


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    def retrieve_skillbox_candidates(
        self,
        query: str,
        *,
        limit: int,
        statuses: List[str],
        agent_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Optional filtered semantic-retrieval hook for learned skills.

        First-party providers override this so lifecycle and tenant filters
        are applied before vector-search top-k selection. Third-party
        providers inherit this backward-compatible over-fetch path; the
        :class:`Skillbox` applies the same filters again after retrieval.
        """
        overfetch = max(int(limit) * 10, 30)
        try:
            result = self.retrieve_by_query(
                query,
                memory_store_type="skillbox",
                limit=overfetch,
                statuses=list(statuses),
                agent_id=agent_id,
                user_id=user_id,
            )
        except TypeError:
            # A provider written against an older MemoRizz contract may not
            # accept filter kwargs. Preserve compatibility and filter in the
            # caller after deliberately over-fetching.
            result = self.retrieve_by_query(
                query,
                memory_store_type="skillbox",
                limit=overfetch,
            )
        if isinstance(result, dict):
            return [result]
        return list(result or [])

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
            Additional provider-specific parameters. Includes ``user_id`` for
            multi-tenant scoping — when provided, results are restricted to
            rows whose stored ``user_id`` equals that value; when ``None``
            (the default), results are restricted to rows whose ``user_id`` is
            also ``None`` (legacy/anonymous scope).
        """

    @abstractmethod
    def retrieve_by_id(
        self, id: str, memory_store_type: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a document from the memory provider by id.

        When a provider supports multi-tenant scoping via ``user_id`` it may
        accept a ``user_id`` keyword argument; callers that need strict
        isolation should check the returned row's ``user_id`` against their
        expected scope before acting on it.
        """

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
        """List all documents within a memory store type in the memory provider.

        Providers should accept an optional ``user_id`` keyword argument for
        tenant scoping. When provided, only rows matching that scope are
        returned; when ``None`` (the default), only rows with no ``user_id``
        are returned.
        """

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
        user_id : str, optional (keyword)
            Multi-tenant scope. When provided, results are restricted to rows
            whose stored ``user_id`` equals that value; when omitted/``None``,
            results are restricted to rows whose ``user_id`` is also ``None``.
        """

    @abstractmethod
    def update_by_id(
        self, id: str, data: Dict[str, Any], memory_store_type: str
    ) -> bool:
        """Update a document in a memory store type in the memory provider by id."""

    def clear_semantic_cache(
        self, agent_id: Optional[str] = None, memory_id: Optional[str] = None
    ) -> int:
        """Delete semantic-cache entries, optionally scoped to an agent/memory.

        Generic implementation over ``list_all`` + ``delete_by_id`` so every
        provider supports it; providers with a native bulk delete (MongoDB)
        override it. Previously only MongoDB implemented this, and the
        SemanticCache layer's calls raised (and were swallowed) on the
        filesystem and Oracle providers.

        Returns the number of entries deleted.
        """
        from ..enums.memory_type import MemoryType

        deleted = 0
        try:
            rows = self.list_all(memory_store_type=MemoryType.SEMANTIC_CACHE) or []
        except Exception:
            return 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if agent_id and row.get("agent_id") != agent_id:
                continue
            if memory_id and row.get("memory_id") != memory_id:
                continue
            record_id = row.get("_id") or row.get("id") or row.get("cache_key")
            if record_id is None:
                continue
            try:
                if self.delete_by_id(
                    str(record_id), memory_store_type=MemoryType.SEMANTIC_CACHE
                ):
                    deleted += 1
            except Exception:
                continue
        return deleted

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
