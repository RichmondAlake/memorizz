# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import base64
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from memorizz.memagent import MemAgent


# Sentinel so "user_id not supplied" is distinguishable from an explicit None
# (matching the MongoDB provider's _MONGO_UNSET convention).
_UNSET = object()


@dataclass(frozen=True)
class MemoryProviderCapabilities:
    """Portable retrieval and storage behavior exposed by a provider."""

    provider: str
    batch_store: bool = False
    transactional_batch: bool = False
    scoped_search: bool = True
    result_scores: bool = False
    provenance: bool = True
    native_vector_search: bool = False
    native_hybrid_search: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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

    def memory_capabilities(self) -> MemoryProviderCapabilities:
        """Describe the portable contract without probing optional internals."""

        return MemoryProviderCapabilities(provider=type(self).__name__)

    def store_many(
        self,
        rows: List[Dict[str, Any]],
        memory_store_type: Any,
        *,
        memory_id: Optional[str] = None,
    ) -> List[str]:
        """Portable batch contract with a correct per-record fallback."""

        return [
            self.store(
                data=dict(row),
                memory_store_type=memory_store_type,
                memory_id=memory_id,
            )
            for row in rows
        ]

    def search_memory(
        self,
        query: Any,
        memory_store_type: Any,
        *,
        limit: int = 10,
        memory_id: Optional[str] = None,
        **scope: Any,
    ) -> List[Dict[str, Any]]:
        """Return one normalized ranked list through the provider contract."""

        result = self.retrieve_by_query(
            query,
            memory_store_type=memory_store_type,
            limit=max(1, int(limit)),
            memory_id=memory_id,
            **scope,
        )
        rows = [result] if isinstance(result, dict) else list(result or [])
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            document = dict(row)
            identifier = (
                document.get("source_id") or document.get("_id") or document.get("id")
            )
            if identifier is not None:
                document.setdefault("source_id", str(identifier))
            normalized.append(document)
        return normalized

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
            multi-tenant scoping. Omitting it is an administrative/unscoped
            read, explicitly passing ``None`` selects anonymous/legacy rows,
            and a string selects that exact tenant.
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
    def list_all(
        self, memory_store_type: str, user_id: Any = _UNSET
    ) -> List[Dict[str, Any]]:
        """List all documents within a memory store type in the memory provider.

        ``user_id`` follows the shared sentinel contract: omitting it performs
        an administrative/unscoped read, explicitly passing ``None`` selects
        only anonymous/legacy rows, and a string selects that exact tenant.
        """

    def query_observability_records(
        self,
        memory_store_type: Any,
        *,
        agent_ids: Optional[List[str]] = None,
        memory_ids: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        user_id: Any = _UNSET,
        application_id: Optional[str] = None,
        record_type: Optional[str] = None,
        tool_name: Optional[str] = None,
        success: Optional[bool] = None,
        start_time: Any = None,
        end_time: Any = None,
        limit: int = 250,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return one bounded observability page with portable metadata.

        First-party database providers should override this with indexed
        predicates. This compatibility path intentionally remains available
        to third-party and filesystem providers, but it never returns an
        unbounded page to the caller.
        """
        started_at = time.perf_counter()
        safe_limit = max(1, min(int(limit or 250), 1000))
        try:
            rows = self.list_all(memory_store_type=memory_store_type) or []
        except TypeError:
            rows = self.list_all(memory_store_type) or []

        wanted_agents = {str(value) for value in (agent_ids or []) if value}
        wanted_memories = {str(value) for value in (memory_ids or []) if value}
        filtered: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            filter_row = row
            # Oracle and older third-party providers can keep shared-memory
            # metadata only inside the JSON payload. Decode it for filtering
            # without changing the returned provider document.
            if record_type and not row.get("record_type"):
                content = row.get("content")
                if hasattr(content, "read"):
                    try:
                        content = content.read()
                    except Exception:
                        content = None
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                if isinstance(content, str):
                    try:
                        payload = json.loads(content)
                    except (TypeError, ValueError):
                        payload = None
                    if isinstance(payload, dict):
                        filter_row = {**row, **payload}
            if record_type is not None and str(
                filter_row.get("record_type") or ""
            ) != str(record_type):
                continue
            row_agent = str(
                filter_row.get("agent_id") or filter_row.get("agentId") or ""
            )
            row_memory = str(
                filter_row.get("trace_memory_id")
                or filter_row.get("memory_id")
                or filter_row.get("memoryId")
                or ""
            )
            if wanted_agents or wanted_memories:
                if row_agent not in wanted_agents and row_memory not in wanted_memories:
                    continue
            row_thread = str(
                filter_row.get("thread_id") or filter_row.get("conversation_id") or ""
            )
            if thread_id is not None and row_thread != str(thread_id):
                continue
            if user_id is not _UNSET and filter_row.get("user_id") != user_id:
                continue
            if (
                application_id is not None
                and filter_row.get("application_id") != application_id
            ):
                continue
            if tool_name is not None and str(filter_row.get("tool_name") or "") != str(
                tool_name
            ):
                continue
            if success is not None and filter_row.get("success") is not success:
                continue
            timestamp = filter_row.get("timestamp")
            if start_time is not None and str(timestamp or "") < str(start_time):
                continue
            if end_time is not None and str(timestamp or "") > str(end_time):
                continue
            result_row = dict(row)
            for key in (
                "record_type",
                "application_id",
                "agent_id",
                "run_id",
                "turn_id",
                "root_trace_id",
                "trace_memory_id",
                "thread_id",
                "user_id",
                "timestamp",
            ):
                if result_row.get(key) is None and filter_row.get(key) is not None:
                    result_row[key] = filter_row[key]
            filtered.append(result_row)

        def sort_key(item):
            identifier = str(
                item.get("_id") or item.get("id") or item.get("memory_id") or ""
            )
            if not identifier:
                identifier = hashlib.sha256(
                    json.dumps(item, sort_keys=True, default=str).encode()
                ).hexdigest()
            return (str(item.get("timestamp") or ""), identifier)

        filtered.sort(key=sort_key, reverse=True)
        offset = 0
        if cursor:
            try:
                padded = cursor + "=" * (-len(cursor) % 4)
                decoded = json.loads(base64.urlsafe_b64decode(padded))
                if isinstance(decoded, int) and decoded >= 0:
                    offset = decoded  # Older offset cursors remain readable.
                elif (
                    isinstance(decoded, dict)
                    and decoded.get("v") == 2
                    and isinstance(decoded.get("last"), list)
                    and len(decoded["last"]) == 2
                ):
                    boundary = tuple(decoded["last"])
                    filtered = [item for item in filtered if sort_key(item) < boundary]
                else:
                    raise ValueError("Invalid observability cursor")
            except Exception as exc:
                raise ValueError("Invalid observability cursor") from exc
        page = filtered[offset : offset + safe_limit]
        next_offset = offset + len(page)
        has_more = next_offset < len(filtered)
        next_cursor = None
        if has_more:
            next_cursor = (
                base64.urlsafe_b64encode(
                    json.dumps({"v": 2, "last": sort_key(page[-1])}).encode()
                )
                .decode("ascii")
                .rstrip("=")
            )
        return {
            "items": page,
            "next_cursor": next_cursor,
            "truncated": has_more,
            "limit": safe_limit,
            "scanned_count": len(rows),
            "query_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "freshness": str(page[0].get("timestamp") or "") if page else None,
            "provider_native": False,
        }

    def query_trace_events(self, *, limit=250, cursor=None, **filters):
        """Return normalized private trace children with a stable child cursor.

        Uses native bundle queries when available; older providers retain the
        compatibility implementation. This method does not authorize tenants.
        """
        from ..enums.memory_type import MemoryType
        from ..observability.index import read_path
        from ..observability.normalization import query_trace_events

        selected_path = filters.pop("read_path", None) or read_path()
        filters.pop("record_type", None)
        if selected_path not in {"bundles", "index"}:
            raise ValueError("read_path must be bundles or index")
        if selected_path == "index":
            index = self.get_observability_index()
            if index is None:
                raise NotImplementedError("Provider has no native observability index")
            return index.query(limit=limit, cursor=cursor, **filters)

        return query_trace_events(
            self,
            MemoryType.SHARED_MEMORY,
            record_type="observability_trace_bundle",
            limit=limit,
            cursor=cursor,
            **filters,
        )

    def get_observability_index(self):
        """Optional private native span index. Provisioning is always explicit."""
        return None

    def delete_observability_bundle(self, record_id, fingerprint):
        """Atomic compare-and-delete used only by reviewed source-expiry plans."""
        raise NotImplementedError("Provider does not support safe source expiry")

    def observability_capabilities(self):
        index = self.get_observability_index()
        return (
            index.capabilities()
            if index is not None
            else {
                "provider": type(self).__name__,
                "span_index": False,
                "ready": False,
                "compatibility_queries": True,
            }
        )

    @abstractmethod
    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        memory_type: str = None,
        limit: int = None,
        user_id: Any = _UNSET,
        thread_id: Optional[str] = None,
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
            Multi-tenant scope. Omitting it is an administrative/unscoped
            read, explicitly passing ``None`` selects anonymous/legacy rows,
            and a string selects that exact tenant.
        thread_id : str, optional
            When provided, return only rows from that exact conversation
            thread. When omitted, return all threads in the memory scope.
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

    def invalidate_semantic_cache(
        self,
        *,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        data_version: Optional[str] = None,
    ) -> int:
        """Delete persistent cache entries matching governance metadata.

        This provider-neutral fallback keeps domain/tag/data-version
        invalidation operational on filesystem, MongoDB, Oracle, and custom
        providers. Native providers may override it with a bulk predicate.
        """
        from ..enums.memory_type import MemoryType

        wanted_domains = {str(item) for item in (domains or [])}
        wanted_tags = {str(item) for item in (tags or [])}
        if not wanted_domains and not wanted_tags and data_version is None:
            return 0
        try:
            rows = self.list_all(memory_store_type=MemoryType.SEMANTIC_CACHE) or []
        except Exception:
            return 0

        deleted = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if agent_id is not None and row.get("agent_id") != agent_id:
                continue
            if memory_id is not None and row.get("memory_id") != memory_id:
                continue
            metadata = row.get("metadata") or {}
            if hasattr(metadata, "read"):
                metadata = metadata.read()
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except (TypeError, ValueError):
                    metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            entry_domains = {
                str(item)
                for item in [
                    metadata.get("domain"),
                    *(metadata.get("domains") or []),
                ]
                if item is not None
            }
            entry_tags = {str(item) for item in (metadata.get("tags") or [])}
            fingerprints = dict(metadata.get("fingerprints") or {})
            matches = (
                bool(wanted_domains.intersection(entry_domains))
                or bool(wanted_tags.intersection(entry_tags))
                or (
                    data_version is not None
                    and fingerprints.get("data_version") == str(data_version)
                )
            )
            if not matches:
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
