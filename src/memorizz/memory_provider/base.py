# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import base64
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    from memorizz.memagent import MemAgent

logger = logging.getLogger(__name__)


# Sentinel so "user_id not supplied" is distinguishable from an explicit None
# (matching the MongoDB provider's _MONGO_UNSET convention).
_UNSET = object()


def _observation_matches(
    row,
    *,
    agent_id=None,
    memory_ids=None,
    user_id=_UNSET,
    application_id=None,
    exclude_ids=(),
):
    """Exact observation scope; a namespace does not override another owner."""
    if not isinstance(row, dict):
        return False
    owner = str(row.get("agent_id") or row.get("owner_agent_id") or "")
    wanted = set(memory_ids or ())
    if agent_id:
        if owner and owner != str(agent_id):
            return False
        if owner != str(agent_id) and row.get("memory_id") not in wanted:
            return False
    elif wanted and row.get("memory_id") not in wanted:
        return False
    if user_id is not _UNSET and row.get("user_id") != user_id:
        return False
    if application_id is not None and row.get("application_id") != application_id:
        return False
    identifier = str(row.get("_id") or row.get("id") or "")
    return identifier not in exclude_ids


def provider_manages_embeddings(provider) -> bool:
    """Whether helpers should leave embedding generation to the provider."""
    getter = getattr(provider, "memory_capabilities", None)
    return callable(getter) and getattr(getter(), "manages_embeddings", False) is True


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
    vector_store: bool = False
    manages_embeddings: bool = False
    requires_live_read: bool = False

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


class GlobalEmbeddingFallbackMixin:
    """For providers with an optional ``_embedding_provider``: use it, else
    the globally configured embeddings (MongoDB, Oracle)."""

    def _get_embedding_provider(self):
        """The provider's own embedding provider, else the global one."""
        if self._embedding_provider is not None:
            return self._embedding_provider
        from ..embeddings import get_embedding_manager

        return get_embedding_manager()

    def _get_embedding_dimensions_safe(self) -> int:
        """The embedding dimensions, or a RuntimeError saying how to configure them."""
        try:
            if self._embedding_provider is not None:
                return self._embedding_provider.get_dimensions()
            from ..embeddings import get_embedding_dimensions

            return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Failed to get embedding dimensions: {e}")
            raise RuntimeError(
                "Cannot determine embedding dimensions. Please configure embeddings first using:\n"
                "configure_embeddings('openai', {'model': 'text-embedding-3-small', 'dimensions': 512})\n"
                "Or use lazy_vector_indexes=True to defer vector index creation."
            )


class FilteredSkillboxSearchMixin:
    """``retrieve_skillbox_candidates`` for providers whose
    ``retrieve_skillbox_item`` applies the lifecycle and tenant filters inside
    vector search (MongoDB Atlas, Oracle)."""

    def retrieve_skillbox_candidates(
        self,
        query: str,
        *,
        limit: int,
        statuses: List[str],
        agent_id: Optional[str],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        return list(
            self.retrieve_skillbox_item(
                query,
                limit,
                statuses=statuses,
                agent_id=agent_id,
                user_id=user_id,
            )
            or []
        )


def _expiry_epoch(value: Any) -> Optional[float]:
    """Coerce a ``datetime``, ISO-8601 string or epoch number to epoch seconds.

    Providers persist ``expires_at`` differently (MongoDB/Oracle return
    ``datetime``, the filesystem provider an ISO string, custom providers may
    store epoch seconds). Naive datetimes are read as UTC: pymongo encodes
    naive values as UTC and hands them back naive. ``None`` means "no usable
    value".
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return _expiry_epoch(datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    def query_memory_observations(
        self,
        memory_store_type,
        *,
        agent_id=None,
        memory_ids=None,
        user_id=_UNSET,
        application_id=None,
        exclude_ids=(),
        limit=200,
    ):
        """Bounded current records for a timeline; native providers override.

        This compatibility fallback supports third-party providers. First-party
        providers push scopes/limits into their metadata index or database.
        """
        from itertools import islice

        matches = (
            row
            for row in self.list_all(memory_store_type) or []
            if _observation_matches(
                row,
                agent_id=agent_id,
                memory_ids=memory_ids,
                user_id=user_id,
                application_id=application_id,
                exclude_ids=exclude_ids,
            )
        )
        return list(islice(matches, max(1, min(int(limit), 1000))))

    def memory_capabilities(self) -> MemoryProviderCapabilities:
        """Describe the portable contract without probing optional internals."""

        return MemoryProviderCapabilities(provider=type(self).__name__)

    def embed_text(self, text: str) -> List[float]:
        """Embed with this provider's configured model, for vector composition.

        This optional interface does not alter legacy document retrieval. Custom
        providers can override it without exposing their embedding internals.
        """
        from .vectors import validate_vector

        resolver = getattr(self, "_get_embedding_provider", None)
        embedder = resolver() if callable(resolver) else None
        if embedder is None:
            raise NotImplementedError(
                "Configure an embedding model on the semantic provider"
            )
        return validate_vector(embedder.get_embedding(text))

    def semantic_identity(self) -> str:
        """Opaque backend/model identity used to detect index rebuilds.

        Custom vector providers should override this when their connection or
        embedding model is not represented by these conventional fields.
        """
        resolver = getattr(self, "_get_embedding_provider", None)
        embedder = resolver() if callable(resolver) else None
        info_getter = getattr(embedder, "get_provider_info", None)
        info = info_getter() if callable(info_getter) else {}
        info = info if isinstance(info, dict) else {"provider": str(info)}
        config = getattr(self, "config", None)
        identity = {
            "provider_class": type(self).__module__ + "." + type(self).__qualname__,
            "embedder_class": type(embedder).__module__
            + "."
            + type(embedder).__qualname__,
            "embedding": {
                key: info.get(key) for key in ("provider", "model", "dimensions")
            },
            "location": {
                key: str(getattr(config, key, ""))
                for key in ("root_path", "uri", "db_name", "dsn", "schema", "user")
            },
        }
        # Connection strings can contain credentials: persist only the digest.
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, default=str).encode()
        ).hexdigest()

    def upsert_vector(self, namespace, source_id, embedding, *, metadata, scope=None):
        """Persist a vector/reference only; never persist the source memory text."""
        from ..enums.memory_type import MemoryType
        from .vectors import VECTOR_MARKER, vector_document

        if not self.memory_capabilities().vector_store:
            raise NotImplementedError(
                "This provider does not implement the vector-store contract"
            )
        row = vector_document(namespace, source_id, embedding, metadata, scope or {})
        identifier = row["id"]
        existing = self.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
        if existing:
            if (
                existing.get("namespace") != namespace
                or (existing.get("metadata") or {}).get("format") != VECTOR_MARKER
            ):
                raise ValueError("Vector identity collides with an unrelated document")
            if not self.update_by_id(identifier, row, MemoryType.KNOWLEDGE_BASE):
                raise RuntimeError("Vector update did not persist")
            return identifier
        return self.store(row, MemoryType.KNOWLEDGE_BASE)

    def query_vectors(
        self, namespace, embedding, *, limit=10, scope=None, include_embedding=False
    ):
        """Rank vectors after applying scope, returning source IDs and references.

        Implementations must raise on unavailable vector search, not substitute
        keyword matches, unscoped records, or arbitrary recent documents.
        """
        raise NotImplementedError("This provider does not implement vector queries")

    def delete_vector(self, namespace, source_id):
        """Delete only the vector owned by this namespace and source."""
        from ..enums.memory_type import MemoryType
        from .vectors import VECTOR_MARKER, vector_id

        if not self.memory_capabilities().vector_store:
            raise NotImplementedError(
                "This provider does not implement vector deletion"
            )
        identifier = vector_id(namespace, source_id)
        row = self.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
        if row is None:
            return False
        if (
            row.get("namespace") != namespace
            or (row.get("metadata") or {}).get("format") != VECTOR_MARKER
        ):
            raise ValueError("Refusing to delete an unrelated document")
        return self.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)

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

    def _clear_agent_toolbox_rows(self, agent_id: str) -> None:
        """Delete the TOOLBOX rows mirrored for one agent."""
        from ..enums.memory_type import MemoryType

        for doc in list(self.list_all(MemoryType.TOOLBOX) or []):
            if not isinstance(doc, dict):
                continue
            if doc.get("agent_id") != agent_id:
                continue
            doc_id = doc.get("_id") or doc.get("id")
            if doc_id:
                self.delete_by_id(str(doc_id), MemoryType.TOOLBOX)

    def _sync_agent_tools_to_toolbox(
        self, agent_id: str, tools: Optional[List[Dict[str, Any]]]
    ) -> None:
        """Mirror an agent's tool list into the TOOLBOX store.

        Deletes any existing TOOLBOX rows for this ``agent_id`` before
        re-inserting the current set so tools removed from the agent
        don't linger in the playground's toolbox-memory pane.
        ``tools=None`` is treated as "caller didn't include tools in this
        save" and is a no-op — only an explicit empty list clears rows.
        """
        from ..enums.memory_type import MemoryType

        if not agent_id or tools is None:
            return

        try:
            self._clear_agent_toolbox_rows(agent_id)
        except Exception as exc:
            logger.warning(
                "Failed to clear toolbox rows for agent %s: %s", agent_id, exc
            )
            return

        if not tools:
            return

        for tool_meta in tools:
            if not isinstance(tool_meta, dict):
                continue
            raw_id = tool_meta.get("_id") or tool_meta.get("name")
            if not raw_id:
                continue
            tool_doc = {
                "_id": f"{agent_id}:{raw_id}",
                "tool_id": f"{agent_id}:{raw_id}",
                "name": tool_meta.get("name"),
                "description": tool_meta.get("description", ""),
                "signature": tool_meta.get("signature", ""),
                "docstring": tool_meta.get(
                    "docstring", tool_meta.get("description", "")
                ),
                "tool_type": tool_meta.get("type", "function"),
                "parameters": tool_meta.get("parameters", {}),
                "agent_id": agent_id,
            }
            try:
                self.store(tool_doc, memory_store_type=MemoryType.TOOLBOX)
            except Exception as exc:
                logger.warning(
                    "Failed to sync tool %s for agent %s to TOOLBOX: %s",
                    tool_doc.get("name"),
                    agent_id,
                    exc,
                )

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

    def list_archive_records(self, memory_type):
        """Complete portable records for archive export (no silent page truncation)."""
        return self.list_all(memory_type)

    def store_archive_record(
        self, memory_type, record_id, data, *, replace=False, reembed=False
    ):
        """Restore one record without evaluating tool or agent configuration."""
        document = dict(data, _id=record_id, id=record_id)
        if reembed:
            from ..embeddings import get_embedding

            text = (
                document.get("content")
                or document.get("description")
                or document.get("name")
            )
            if isinstance(text, str) and text:
                document["embedding"] = get_embedding(text)
        return self.store(document, memory_store_type=memory_type)

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
        event_filters: Optional[Dict[str, Any]] = None,
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
            if event_filters or (record_type and not row.get("record_type")):
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
                elif isinstance(content, dict):
                    # Oracle's JSON columns are already decoded by the driver.
                    filter_row = {**row, **content}
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
            if any(
                filter_row.get(key) != value
                for key, value in (event_filters or {}).items()
            ):
                continue
            timestamp = (
                filter_row.get("timestamp")
                or filter_row.get("started_at")
                or filter_row.get("created_at")
                or filter_row.get("updated_at")
            )
            if start_time is not None and str(timestamp or "") < str(start_time):
                continue
            if end_time is not None and str(timestamp or "") > str(end_time):
                continue
            result_row = dict(row)
            if result_row.get("timestamp") is None and timestamp is not None:
                result_row["timestamp"] = timestamp
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

    def compare_and_swap_shared_memory(
        self, memory_id: str, expected_content: Any, content: str
    ) -> bool:
        """Atomically replace shared content only if the read snapshot matches.

        False means a missing row or a conflicting writer; storage errors must
        raise. Providers must not emulate this with an unprotected read/write.
        """
        raise NotImplementedError(
            "Provider does not support atomic shared-memory updates"
        )

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

    def touch_many(
        self,
        record_ids: Iterable[str],
        memory_store_type: Any,
        *,
        now: Any = None,
    ) -> int:
        """Record a recall of each memory: bump ``access_count``, set ``last_accessed_at``.

        This is the reinforcement signal of the Generative-Agents retrieval
        score (recency is measured since the last access). The portable
        implementation reads and updates each record; database providers
        should override it with one UPDATE statement. It never touches the
        record's ``timestamp`` or embedding, and failures are logged rather
        than raised because a recall must never fail a turn.
        """
        from datetime import datetime, timezone

        if now is None:
            now_value = datetime.now(timezone.utc).isoformat()
        elif hasattr(now, "isoformat"):
            now_value = now.isoformat()
        else:
            now_value = now
        touched = 0
        for record_id in record_ids:
            identifier = str(record_id or "").strip()
            if not identifier:
                continue
            try:
                row = self.retrieve_by_id(identifier, memory_store_type)
            except Exception:
                row = None
            if not isinstance(row, dict):
                continue
            try:
                count = int(row.get("access_count") or 0)
            except (TypeError, ValueError):
                count = 0
            try:
                if self.update_by_id(
                    identifier,
                    {"last_accessed_at": now_value, "access_count": count + 1},
                    memory_store_type,
                ):
                    touched += 1
            except Exception as exc:
                logger.debug("touch_many could not update %s: %s", identifier, exc)
        return touched

    def purge_expired_semantic_cache(self, now=None) -> int:
        """Delete semantic-cache rows whose ``expires_at`` lies before ``now``.

        Generic ``list_all`` + ``delete_by_id`` implementation so every
        provider purges expired rows; native providers may override it with
        one bulk delete but must keep this exact signature. ``now`` accepts a
        ``datetime``, epoch seconds, or ``None`` for the current time.
        ``expires_at`` may be a ``datetime``, an ISO-8601 string or epoch
        seconds; rows without a parseable value are kept.

        Returns the number of rows deleted.
        """
        from ..enums.memory_type import MemoryType

        cutoff = time.time() if now is None else _expiry_epoch(now)
        if cutoff is None:
            raise ValueError(f"Unsupported value for now: {now!r}")
        try:
            rows = self.list_all(memory_store_type=MemoryType.SEMANTIC_CACHE) or []
        except Exception:
            return 0

        deleted = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            expires_at = _expiry_epoch(row.get("expires_at"))
            if expires_at is None or expires_at >= cutoff:
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
