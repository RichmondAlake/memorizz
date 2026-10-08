# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from bson import ObjectId
from pymongo import InsertOne, MongoClient, ReplaceOne, ReturnDocument
from pymongo.errors import CollectionInvalid, DuplicateKeyError, OperationFailure
from pymongo.operations import SearchIndexModel

from ...embeddings import get_embedding
from ...enums.memory_type import MemoryType
from ...memagent import MemAgentModel
from ..base import (
    FilteredSkillboxSearchMixin,
    GlobalEmbeddingFallbackMixin,
    MemoryProvider,
    MemoryProviderCapabilities,
)

logger = logging.getLogger(__name__)


# Sentinel so "user_id filter not supplied" is distinguishable from
# "user_id is explicitly None" (anonymous/legacy scope).
class _MongoUserIdUnset:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<user_id unset>"


_MONGO_UNSET = _MongoUserIdUnset()


class VectorSearchExecutionError(RuntimeError):
    """A Mongo vector query failed before producing a trustworthy result."""

    def __init__(self, message: str, *, reason: str = "vector_query_failed"):
        super().__init__(message)
        self.reason = reason


class VectorSearchUnavailableError(VectorSearchExecutionError):
    """Mongo vector search is not provisioned for the requested collection."""

    def __init__(self, message: str):
        super().__init__(message, reason="vector_search_unavailable")


_MONGO_USER_SCOPED_TYPES = frozenset(
    {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
        MemoryType.TOOL_LOG,
        MemoryType.SKILLBOX,
    }
)


def _mongo_memory_type_supports_user_id(memory_store_type: Any) -> bool:
    try:
        if isinstance(memory_store_type, str):
            memory_store_type = MemoryType(memory_store_type)
    except Exception:
        return False
    return memory_store_type in _MONGO_USER_SCOPED_TYPES


def _mongo_user_id_predicate(user_id: Any) -> Dict[str, Any]:
    """Build a Mongo filter that matches rows in the given user_id scope.

    ``None`` matches documents whose ``user_id`` is missing or null so that
    legacy rows are still visible to anonymous callers. Any other value
    enforces strict equality — preventing cross-tenant leaks even when the
    stored row predates this change.
    """
    if user_id is None:
        return {"user_id": {"$in": [None]}}
    return {"user_id": user_id}


def _mongo_id_predicate(record_id: Any) -> Dict[str, Any]:
    """Match a row by ``_id`` whether it was stored as a string or an ObjectId.

    The provider writes string primary keys itself (toolbox rows use
    ``"<agent_id>:<tool>"``, immutable trace bundles use their logical id,
    vectors use uuid5 keys and ``store`` keeps any caller-supplied ``_id``),
    so an ObjectId-only lookup silently misses those rows.
    """
    candidates: List[Any] = [record_id]
    if not isinstance(record_id, ObjectId) and ObjectId.is_valid(record_id):
        candidates.append(ObjectId(record_id))
    return {"_id": {"$in": candidates}}


def _as_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime; naive values are taken as UTC (BSON's rule)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_expiry(value: Any) -> Optional[datetime]:
    """Coerce a stored/caller ``expires_at`` to aware UTC; ``None`` if unparsable."""
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return _as_utc(datetime.fromisoformat(text))
        except ValueError:
            return None
    return None


def _lexical_tokens(value: Any) -> set[str]:
    punctuation = ".,:;!?()[]{}\"'"
    return {
        token.strip(punctuation)
        for token in str(value or "").lower().replace("_", " ").split()
        if token.strip(punctuation)
    }


def _lexical_overlap(query_tokens: set[str], content: Any) -> float:
    """Return a deterministic rank for degraded, already-scoped retrieval."""
    return (
        len(query_tokens & _lexical_tokens(content)) / len(query_tokens)
        if query_tokens
        else 0.0
    )


_PRESERVED_MONGO_ID_FIELDS = {
    MemoryType.SEMANTIC_CACHE: frozenset({"agent_id", "memory_id"}),
    MemoryType.SHARED_MEMORY: frozenset({"agent_id", "thread_id", "memory_id"}),
    MemoryType.TOOL_LOG: frozenset({"agent_id", "thread_id", "memory_id"}),
    MemoryType.TOOLBOX: frozenset({"agent_id", "tool_id"}),
    MemoryType.SKILLBOX: frozenset({"agent_id"}),
    MemoryType.WORKFLOW_MEMORY: frozenset({"agent_id", "workflow_id"}),
    # Agent ownership is metadata, not a Mongo primary key. Retain it for
    # account-scoped persona evidence and conversation/trace attribution.
    MemoryType.CONVERSATION_MEMORY: frozenset({"agent_id", "thread_id", "memory_id"}),
    MemoryType.KNOWLEDGE_BASE: frozenset({"knowledge_base_id", "memory_id"}),
    MemoryType.ENTITY_MEMORY: frozenset({"memory_id"}),
}


def _mongo_id_fields_to_strip(memory_type: MemoryType) -> set[str]:
    """Return legacy logical IDs that must not become Mongo primary keys."""
    fields = {
        "persona_id",
        "tool_id",
        "workflow_id",
        "short_term_memory_id",
        "agent_id",
        "thread_id",
        "knowledge_base_id",
        "memory_id",
    }
    fields.difference_update(_PRESERVED_MONGO_ID_FIELDS.get(memory_type, ()))
    return fields


@dataclass
class MongoDBConfig:
    """Configuration for the MongoDB provider."""

    def __init__(
        self,
        uri: str,
        db_name: str = "memorizz",
        lazy_vector_indexes: bool = False,
        read_only: bool = False,
        embedding_provider=None,
        embedding_config: Dict[str, Any] = None,
    ):
        """
        Initialize the MongoDB provider with configuration settings.

        Parameters:
        -----------
        uri : str
            The MongoDB URI.
        db_name : str
            The database name.
        lazy_vector_indexes : bool
            If True, vector indexes are created only when needed (when vector operations are performed).
            If False, vector indexes are created immediately during initialization (requires embedding configuration).
            Default: False (maintains backward compatibility)
        embedding_provider : str or EmbeddingManager, optional
            Embedding provider to use. Can be:
            - EmbeddingManager instance (explicit injection)
            - String provider name ("openai", "ollama", "voyageai")
            - None (uses global embedding configuration)
        embedding_config : Dict[str, Any], optional
            Configuration for the embedding provider. Only used when embedding_provider is a string.
            Example: {"model": "text-embedding-3-small", "dimensions": 512}
        """
        self.uri = uri
        self.db_name = db_name
        self.lazy_vector_indexes = lazy_vector_indexes
        self.read_only = bool(read_only)
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config or {}


class MongoDBProvider(
    GlobalEmbeddingFallbackMixin, FilteredSkillboxSearchMixin, MemoryProvider
):
    """MongoDB implementation of the MemoryProvider interface."""

    def memory_capabilities(self) -> MemoryProviderCapabilities:
        return MemoryProviderCapabilities(
            provider=type(self).__name__,
            batch_store=True,
            transactional_batch=False,
            scoped_search=True,
            result_scores=True,
            provenance=True,
            native_vector_search=True,
            vector_store=True,
            native_hybrid_search=False,
        )

    def upsert_vector(self, namespace, source_id, embedding, *, metadata, scope=None):
        from ..vectors import VECTOR_MARKER, vector_document

        if self.config.read_only:
            raise PermissionError("The semantic provider is read-only")
        row = vector_document(namespace, source_id, embedding, metadata, scope or {})
        # The vector interface owns UUID string IDs. The legacy document API
        # uses ObjectIds, so do not round-trip these through retrieve_by_id.
        self.knowledge_base_collection.replace_one(
            {
                "_id": row["_id"],
                "namespace": namespace,
                "metadata.format": VECTOR_MARKER,
            },
            row,
            upsert=True,
        )
        return row["id"]

    def delete_vector(self, namespace, source_id):
        from ..vectors import VECTOR_MARKER, vector_id

        if self.config.read_only:
            raise PermissionError("The semantic provider is read-only")
        result = self.knowledge_base_collection.delete_one(
            {
                "_id": vector_id(namespace, source_id),
                "namespace": namespace,
                "metadata.format": VECTOR_MARKER,
            }
        )
        return bool(result.deleted_count)

    def query_vectors(
        self, namespace, embedding, *, limit=10, scope=None, include_embedding=False
    ):
        from ..vectors import (
            VECTOR_MARKER,
            VECTOR_SCOPE_FIELDS,
            scope_hash,
            validate_scope,
            validate_vector,
            vector_hit,
        )

        vector = validate_vector(embedding)
        filters = validate_scope(scope or {})
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        index_name = "memorizz_semantic_index"
        if not getattr(self, "_composed_vector_index_ready", False):
            indexes = list(
                self.knowledge_base_collection.list_search_indexes(name=index_name)
            )
            if not any(index.get("queryable") for index in indexes):
                if self.config.read_only or indexes:
                    raise VectorSearchUnavailableError(
                        "The composed semantic index is not queryable"
                    )
                created = self._setup_vector_search_index(
                    self.knowledge_base_collection,
                    index_name=index_name,
                    filter_fields=["namespace", "metadata.format"]
                    + [
                        "metadata.scope_hashes." + field
                        for field in sorted(VECTOR_SCOPE_FIELDS)
                    ],
                )
                if not created:
                    raise VectorSearchUnavailableError(
                        "Could not create the composed semantic index"
                    )
                indexes = list(
                    self.knowledge_base_collection.list_search_indexes(name=index_name)
                )
                if not any(index.get("queryable") for index in indexes):
                    raise VectorSearchUnavailableError(
                        "The composed semantic index is not queryable"
                    )
            self._composed_vector_index_ready = True
        predicate = {"namespace": namespace, "metadata.format": VECTOR_MARKER}
        for key, value in filters.items():
            predicate["metadata.scope_hashes." + key] = (
                {"$in": [scope_hash(item) for item in value]}
                if isinstance(value, list)
                else scope_hash(value)
            )
        pipeline = self._build_vector_search_pipeline(
            vector,
            limit,
            index_name=index_name,
            search_filter=predicate,
            include_embedding=include_embedding,
        )
        # Fail closed: the ordinary KB helper can fall back to recent documents,
        # which is not a valid substitute for the vector-only contract.
        # Atlas reports (1 + cosine) / 2. This optional cross-provider contract
        # returns cosine consistently with filesystem and Oracle; legacy APIs
        # retain their existing provider-specific scores.
        return [
            vector_hit(
                {**row, "score": 2 * float(row["score"]) - 1},
                include_embedding=include_embedding,
            )
            for row in self.knowledge_base_collection.aggregate(pipeline)
        ]

    def __init__(self, config: MongoDBConfig):
        """
        Initialize the MongoDB provider with configuration settings.

        Parameters:
        -----------
        config : MongoDBConfig
            Configuration dictionary containing:
            - 'uri': MongoDB URI
            - 'db_name': Database name
            - 'lazy_vector_indexes': Whether to defer vector index creation
            - 'embedding_provider': Optional explicit embedding provider
        """
        self.config = config
        self.lazy_vector_indexes = bool(config.lazy_vector_indexes)
        self.client = MongoClient(config.uri)
        self.db = self.client[config.db_name]
        self.persona_collection = self.db[MemoryType.PERSONAS.value]
        self.toolbox_collection = self.db[MemoryType.TOOLBOX.value]
        self.skillbox_collection = self.db[MemoryType.SKILLBOX.value]
        self.short_term_memory_collection = self.db[MemoryType.SHORT_TERM_MEMORY.value]
        self.knowledge_base_collection = self.db[MemoryType.KNOWLEDGE_BASE.value]
        self.conversation_memory_collection = self.db[
            MemoryType.CONVERSATION_MEMORY.value
        ]
        self.workflow_memory_collection = self.db[MemoryType.WORKFLOW_MEMORY.value]
        self.entity_memory_collection = self.db[MemoryType.ENTITY_MEMORY.value]
        self.memagent_collection = self.db[MemoryType.MEMAGENT.value]
        self.shared_memory_collection = self.db[MemoryType.SHARED_MEMORY.value]
        self.summaries_collection = self.db[MemoryType.SUMMARIES.value]
        self.semantic_cache_collection = self.db[MemoryType.SEMANTIC_CACHE.value]
        self.tool_log_collection = self.db[MemoryType.TOOL_LOG.value]

        # Track which vector indexes have been created
        self._vector_indexes_created = set()
        # Collections for which vector-search index creation failed because
        # the cluster cannot provision more search indexes (Atlas FTS quota,
        # missing tier, etc.). Consumers of those vector-search paths check
        # this set to short-circuit gracefully instead of hammering the API
        # with queries that will return zero rows.
        self._vector_indexes_unavailable: set = set()
        # Root-cause dedup set — when the same Mongo error hits every
        # collection at boot, only the first one logs a WARNING; the rest
        # are recorded silently. See _handle_index_unavailable.
        self._vector_index_unavailable_root_causes: set = set()
        self._vector_search_status_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        # Process embedding provider configuration
        self._embedding_provider = self._setup_embedding_provider(config)

        # A production inspector should be able to connect with a MongoDB user
        # that has read-only roles. In that mode provider construction must not
        # create collections or indexes as a hidden side effect.
        if not config.read_only:
            self._create_memory_stores()
            self._ensure_unique_agent_ids()

        # Create vector indexes immediately only if not using lazy initialization
        if not config.read_only and not config.lazy_vector_indexes:
            try:
                self._create_vector_indexes_for_memory_stores()
            except Exception as e:
                logger.warning(
                    f"Failed to create vector indexes during initialization: {e}"
                )
                logger.info("Vector indexes will be created lazily when needed")
                # Set lazy mode if immediate creation fails
                self.config.lazy_vector_indexes = True
                self.lazy_vector_indexes = True

    @staticmethod
    def _normalize_legacy_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old conversation_id → thread_id on read for backward compat."""
        if "conversation_id" in doc and "thread_id" not in doc:
            doc["thread_id"] = doc.pop("conversation_id")
        if "associated_conversation_ids" in doc and "associated_thread_ids" not in doc:
            doc["associated_thread_ids"] = doc.pop("associated_conversation_ids")
        return doc

    @staticmethod
    def _is_quota_or_unsupported(exc: Exception) -> bool:
        """Classify a Mongo OperationFailure as "expected on small tiers".

        Atlas free / shared tiers cap the number of search indexes per
        cluster and reject ``createSearchIndex`` calls past the cap with
        ``IllegalOperation`` (code 20). We use this classifier to convert
        those into a single graceful WARNING instead of a stderr stack
        trace at every agent boot — see :meth:`_handle_index_unavailable`.
        Also catches CommandNotFound which is what shared tiers without
        Search enabled return.
        """
        code = getattr(exc, "code", None)
        code_name = (getattr(exc, "details", None) or {}).get("codeName", "")
        text = str(exc).lower()
        if code in (20, 59, 6047401):  # IllegalOperation/CommandNotFound/Community
            return True
        if code_name in {"IllegalOperation", "CommandNotFound"}:
            return True
        # String-based fallbacks for older driver/server combos that don't
        # surface a numeric code on every error path.
        for needle in (
            "maximum number of fts indexes",
            "search is not supported",
            "no such command: 'createsearchindex'",
            "atlas search is not enabled",
            "listsearchindexes stage is only allowed on mongodb atlas",
        ):
            if needle in text:
                return True
        return False

    def _handle_index_unavailable(
        self, collection_name: str, index_name: str, exc: Exception
    ) -> None:
        """Mark a collection's vector index as unavailable and warn once.

        Downstream vector-search helpers (``find_similar_*``,
        ``_knowledge_base_vector_search``) check
        ``self._vector_indexes_unavailable`` before issuing a
        ``$vectorSearch`` aggregation so we don't keep racking up failed
        requests for indexes Atlas won't let us create.

        When the *same* quota/unsupported error hits every collection on
        boot, we want one summary log line rather than twelve. The first
        offender for a given root cause gets a full WARNING with the
        remediation hint; subsequent offenders are silently added to the
        unavailable set so the agent still degrades gracefully but the
        boot log stays readable.
        """
        if collection_name in self._vector_indexes_unavailable:
            return
        # Bucket by root cause so we only emit one structured warning for
        # cluster-wide failures (Atlas quota, Search not enabled, etc.).
        root_cause = self._root_cause_key(exc)
        first_for_root = root_cause not in self._vector_index_unavailable_root_causes
        self._vector_indexes_unavailable.add(collection_name)
        if root_cause is not None:
            self._vector_index_unavailable_root_causes.add(root_cause)
        if first_for_root:
            details = getattr(exc, "details", None) or {}
            logger.warning(
                "MongoDB vector-search index unavailable (first offender: "
                "collection=%r, index=%s). Vector-search consumers will "
                "degrade to no-op for any collection hitting the same root "
                "cause. Likely cause: Atlas Search quota exceeded on this "
                "cluster tier (M0/M2 cap is 3 search indexes) or Atlas "
                "Search not enabled. Error type=%s code=%s code_name=%s",
                collection_name,
                index_name,
                type(exc).__name__,
                getattr(exc, "code", None),
                details.get("codeName"),
            )
        else:
            logger.debug(
                "MongoDB vector index unavailable for %r (same root cause "
                "as previous warning, suppressing)",
                collection_name,
            )

    @staticmethod
    def _root_cause_key(exc: Exception) -> Optional[str]:
        """Stable string for an exception's root cause, used to dedupe logs."""
        code = getattr(exc, "code", None)
        code_name = (getattr(exc, "details", None) or {}).get("codeName")
        if code or code_name:
            return f"code={code}|name={code_name}"
        # Fall back to the exception class + first line of the message.
        text = str(exc).splitlines()[0] if str(exc) else ""
        return f"{type(exc).__name__}|{text[:120]}"

    def _vector_index_unavailable(self, collection_name: str) -> bool:
        """True when the named collection's vector index can't be used."""
        return collection_name in getattr(self, "_vector_indexes_unavailable", set())

    def get_vector_search_status(
        self,
        memory_store_type: Union[str, MemoryType],
        *,
        cache_ttl_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """Return bounded vector-index health without creating an index.

        Semantic consumers use this to distinguish a genuine zero-match result
        from a deployment where vector search is unavailable. The short cache
        avoids an Atlas metadata query on every agent turn.
        """
        try:
            resolved_type = (
                memory_store_type
                if isinstance(memory_store_type, MemoryType)
                else MemoryType(memory_store_type)
            )
        except Exception:
            return {"available": False, "reason": "unsupported_memory_type"}

        collection = self._collection(resolved_type)
        if collection is None:
            return {"available": False, "reason": "collection_unavailable"}
        collection_name = collection.name
        now = time.monotonic()
        cached = self._vector_search_status_cache.get(collection_name)
        if cached and now - cached[0] <= max(0.0, float(cache_ttl_seconds)):
            return dict(cached[1])

        if self._vector_index_unavailable(collection_name):
            status = {
                "available": False,
                "queryable": False,
                "reason": "vector_search_unavailable",
            }
            self._vector_search_status_cache[collection_name] = (now, status)
            return dict(status)

        try:
            indexes = list(collection.list_search_indexes())
        except Exception as exc:
            if self._is_quota_or_unsupported(exc):
                self._handle_index_unavailable(collection_name, "vector_index", exc)
                status = {
                    "available": False,
                    "queryable": False,
                    "reason": "vector_search_unsupported",
                }
            else:
                # Unknown metadata failures must not be mistaken for a
                # confirmed missing index. The caller may still try semantic
                # retrieval, then use its scoped exact fallback on failure.
                status = {
                    "available": None,
                    "reason": "status_check_failed",
                }
            self._vector_search_status_cache[collection_name] = (now, status)
            return dict(status)

        index = next(
            (
                item
                for item in indexes
                if item.get("name") == "vector_index"
                and item.get("type") == "vectorSearch"
            ),
            None,
        )
        if index is None:
            lazy_creation = bool(
                getattr(self, "lazy_vector_indexes", False)
            ) and not bool(getattr(getattr(self, "config", None), "read_only", False))
            status = (
                {
                    "available": None,
                    "queryable": False,
                    "reason": "vector_index_pending_lazy_creation",
                }
                if lazy_creation
                else {
                    "available": False,
                    "queryable": False,
                    "reason": "vector_index_missing",
                }
            )
        else:
            queryable = index.get("queryable") is True
            index_status = str(index.get("status") or "").upper() or None
            status = {
                "available": queryable,
                "queryable": queryable,
                "status": index_status,
                "reason": None if queryable else "vector_index_not_queryable",
            }
        self._vector_search_status_cache[collection_name] = (now, status)
        return dict(status)

    @staticmethod
    def _build_vector_search_pipeline(
        embedding,
        limit: int,
        *,
        index_name: str = "vector_index",
        search_filter: Dict[str, Any] | None = None,
        num_candidates: int | None = None,
        path: str = "embedding",
        include_embedding: bool = False,
    ) -> List[Dict[str, Any]]:
        """Build the canonical 3-stage vector-search aggregation pipeline.

        Atlas ``$vectorSearch`` followed by a single ``$project`` that
        mixes inclusion (``"field": 1``) with exclusion (``"embedding":
        0``) is rejected by MongoDB ("Cannot do exclusion on field
        embedding in inclusion projection"). Splitting the projection
        from the score addition sidesteps the rule entirely:

        1. ``$vectorSearch`` — semantic match
        2. ``$project: {embedding: 0}`` — pure exclusion (allowed)
        3. ``$addFields: {score: {$meta: "vectorSearchScore"}}`` —
           computed field, no inclusion/exclusion mix

        Every consumer (persona, toolbox, workflow, summary,
        conversation, KB, entity, semantic-cache) uses this builder so
        the rule is enforced in exactly one place.
        """
        stage: Dict[str, Any] = {
            "$vectorSearch": {
                "queryVector": embedding,
                "path": path,
                "numCandidates": (
                    num_candidates
                    if num_candidates is not None
                    else max(100, limit * 10)
                ),
                "limit": limit,
                "index": index_name,
            }
        }
        if search_filter:
            stage["$vectorSearch"]["filter"] = search_filter
        pipeline: List[Dict[str, Any]] = [stage]
        # Callers running assembly-time dedup/MMR ask for the stored vectors
        # back (include_embedding=True) so they never have to re-embed rows.
        if not include_embedding:
            pipeline.append({"$project": {"embedding": 0}})
        pipeline.append({"$addFields": {"score": {"$meta": "vectorSearchScore"}}})
        return pipeline

    def _run_vector_search(
        self,
        collection,
        pipeline: List[Dict[str, Any]],
        *,
        label: str,
        raise_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run a vector-search pipeline, honoring the unavailable-index flag.

        Short-circuits to ``[]`` when the collection's vector index is
        known to be unavailable (Atlas quota / Search not enabled), so
        we don't keep racking up failed aggregations once the index
        creator has already flagged the collection.
        """
        if self._vector_index_unavailable(collection.name):
            if raise_on_error:
                raise VectorSearchUnavailableError(
                    f"Vector search is unavailable for {label}"
                )
            return []
        try:
            return list(collection.aggregate(pipeline))
        except Exception as exc:
            if self._is_quota_or_unsupported(exc):
                self._handle_index_unavailable(collection.name, "vector_index", exc)
                cache = getattr(self, "_vector_search_status_cache", None)
                if isinstance(cache, dict):
                    cache[collection.name] = (
                        time.monotonic(),
                        {
                            "available": False,
                            "queryable": False,
                            "reason": "vector_search_unavailable",
                        },
                    )
                if raise_on_error:
                    raise VectorSearchUnavailableError(
                        f"Vector search is unavailable for {label}"
                    ) from exc
                return []
            cache = getattr(self, "_vector_search_status_cache", None)
            if isinstance(cache, dict):
                cache[collection.name] = (
                    time.monotonic(),
                    {"available": None, "reason": "vector_query_failed"},
                )
            logger.warning(
                "Vector search failed for %s (%s)", label, type(exc).__name__
            )
            if raise_on_error:
                raise VectorSearchExecutionError(
                    f"Vector search execution failed for {label}"
                ) from exc
            return []

    def _setup_embedding_provider(self, config: MongoDBConfig):
        """
        Setup the embedding provider based on configuration.

        Parameters:
        -----------
        config : MongoDBConfig
            The MongoDB configuration

        Returns:
        --------
        EmbeddingManager or None
            The configured embedding provider, or None to use global configuration
        """
        if config.embedding_provider is None:
            # No explicit provider - will use global configuration
            return None
        elif isinstance(config.embedding_provider, str):
            # String provider name - create EmbeddingManager
            try:
                from ...embeddings import EmbeddingManager

                provider = EmbeddingManager(
                    config.embedding_provider, config.embedding_config
                )
                logger.info(
                    f"Created embedding provider: {provider.get_provider_info()}"
                )
                return provider
            except Exception as e:
                logger.error(
                    f"Failed to create embedding provider '{config.embedding_provider}': {e}"
                )
                raise
        else:
            # Assume it's already an EmbeddingManager instance
            return config.embedding_provider

    def _ensure_vector_index_for_collection(
        self, collection, collection_name: str, memory_store: bool = False
    ):
        """
        Ensure vector index exists for a collection, creating it lazily if needed.

        Parameters:
        -----------
        collection : pymongo.Collection
            The MongoDB collection
        collection_name : str
            Name of the collection (for tracking)
        memory_store : bool
            Whether this is a memory store collection
        """
        index_key = f"{collection_name}_vector_index"

        if index_key not in self._vector_indexes_created:
            try:
                ensured = self._ensure_vector_index(
                    collection,
                    "vector_index",
                    memory_store,
                    filter_fields=(
                        ["status", "agent_id", "user_id"]
                        if collection_name == MemoryType.SKILLBOX.value
                        else (
                            ["user_id"]
                            if collection_name == MemoryType.ENTITY_MEMORY.value
                            else None
                        )
                    ),
                )
                if ensured and not self._vector_index_unavailable(collection_name):
                    self._vector_indexes_created.add(index_key)
                    cache = getattr(self, "_vector_search_status_cache", None)
                    if isinstance(cache, dict):
                        cache.pop(collection_name, None)
                    logger.info(
                        f"Ensured vector index for collection: {collection_name}"
                    )
            except Exception as e:
                if self._is_quota_or_unsupported(e):
                    self._handle_index_unavailable(collection_name, "vector_index", e)
                    return
                logger.error(
                    f"Failed to create vector index for {collection_name}: {e}"
                )
                raise

    def _create_memory_stores(self) -> None:
        """
        Create all memory stores in MongoDB.
        """
        self._create_memory_store(MemoryType.MEMAGENT)
        self._create_memory_store(MemoryType.PERSONAS)
        self._create_memory_store(MemoryType.TOOLBOX)
        self._create_memory_store(MemoryType.SHORT_TERM_MEMORY)
        self._create_memory_store(MemoryType.KNOWLEDGE_BASE)
        self._create_memory_store(MemoryType.CONVERSATION_MEMORY)
        self._create_memory_store(MemoryType.WORKFLOW_MEMORY)
        self._create_memory_store(MemoryType.SHARED_MEMORY)
        self._create_memory_store(MemoryType.SUMMARIES)
        self._create_memory_store(MemoryType.TOOL_LOG)

    def _create_memory_store(self, memory_store_type: MemoryType) -> None:
        """
        Create a new memory store in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store to create.

        Returns:
        --------
        None
        """

        # Create collection if it doesn't exist within the database/memory provider
        # Check if the collection exists within the database and if it doesn't, create an empty collection
        for memory_store_type in MemoryType:
            if memory_store_type.value not in self.db.list_collection_names():
                try:
                    self.db.create_collection(memory_store_type.value)
                except CollectionInvalid:
                    # Cold-start race: another worker/process created this
                    # collection between the list_collection_names() check
                    # and create_collection() (e.g. concurrent gunicorn
                    # workers each building the provider on their first
                    # request). The collection exists either way, so treat
                    # "already exists" as success rather than letting it
                    # abort provider initialisation.
                    pass
                except OperationFailure as exc:
                    # 48 = NamespaceExists — the same race surfaced
                    # server-side (e.g. Atlas). Anything else is a real
                    # failure and must propagate.
                    if exc.code != 48:
                        raise

        # Standard btree indexes for the hot-path queries (per-thread tool
        # log listing, per-thread conversation history, per-user entity
        # lookup). Vector indexes are created separately via
        # ``_create_vector_indexes_for_memory_stores`` since they need
        # Atlas Search and have a different failure model.
        self._ensure_btree_indexes()

    # Hot-path query coverage. Each (collection, [(field, direction), …])
    # entry becomes one compound index. Compound order matters: the most
    # selective filter goes first, the sort field goes last with the same
    # direction the query asks for, so MongoDB can satisfy filter + sort
    # from the index alone (no in-memory sort, no FETCH for skipped docs).
    _BTREE_INDEX_SPECS = {
        MemoryType.TOOL_LOG: [
            (
                "tool_log_agent_cursor",
                [("agent_id", 1), ("_id", -1)],
            ),
            (
                "tool_log_thread_cursor",
                [("thread_id", 1), ("_id", -1)],
            ),
            (
                "tool_log_tool_success_cursor",
                [("tool_name", 1), ("success", 1), ("_id", -1)],
            ),
            (
                "tool_log_memory_timestamp",
                [("memory_id", 1), ("timestamp", -1)],
            ),
            (
                "tool_log_user_timestamp",
                [("user_id", 1), ("timestamp", -1)],
            ),
            (
                "tool_log_id_unique",
                [("tool_log_id", 1)],
                {"unique": True, "sparse": True},
            ),
        ],
        MemoryType.CONVERSATION_MEMORY: [
            (
                "conv_agent_cursor",
                [("agent_id", 1), ("_id", -1)],
            ),
            (
                "conv_thread_cursor",
                [("thread_id", 1), ("_id", -1)],
            ),
            (
                "conv_memory_thread_ts",
                [("memory_id", 1), ("thread_id", 1), ("timestamp", 1)],
            ),
            (
                "conv_memory_user_ts",
                [("user_id", 1), ("timestamp", 1)],
            ),
        ],
        MemoryType.SHARED_MEMORY: [
            (
                "shared_observability_agent_cursor",
                [("record_type", 1), ("agent_id", 1), ("_id", -1)],
            ),
            (
                "shared_observability_thread_cursor",
                [("record_type", 1), ("thread_id", 1), ("_id", -1)],
            ),
            (
                "shared_observability_memory_cursor",
                [("record_type", 1), ("trace_memory_id", 1), ("_id", -1)],
            ),
            (
                "shared_observability_trace_cursor",
                [("root_trace_id", 1), ("_id", -1)],
            ),
            (
                "shared_observability_record_unique",
                [("memory_id", 1)],
                {"unique": True, "sparse": True},
            ),
        ],
        MemoryType.ENTITY_MEMORY: [
            (
                "entity_memory_scope_name",
                [("memory_id", 1), ("user_id", 1), ("name", 1)],
            ),
            (
                "entity_memory_scope_id",
                [("memory_id", 1), ("user_id", 1), ("entity_id", 1)],
            ),
            (
                "entity_memory_user_updated",
                [("user_id", 1), ("updated_at", -1)],
            ),
        ],
        MemoryType.SUMMARIES: [
            (
                "summaries_memory_period_end",
                [("memory_id", 1), ("period_end", -1)],
            ),
            (
                "summaries_memory_thread_period_end",
                [("memory_id", 1), ("thread_id", 1), ("period_end", -1)],
            ),
        ],
        MemoryType.SEMANTIC_CACHE: [
            # TTL: MongoDB drops a row once its date-typed ``expires_at`` has
            # passed. String timestamps are ignored by TTL indexes; the
            # explicit ``purge_expired_semantic_cache`` handles those.
            (
                "semantic_cache_expires_at_ttl",
                [("expires_at", 1)],
                {"expireAfterSeconds": 0},
            ),
        ],
    }

    def _ensure_unique_agent_ids(self) -> None:
        """One record per agent_id, so concurrent saves cannot duplicate an agent.

        Several processes saving the same agent at start-up (e.g. one per web
        worker) raced a find-then-insert and left duplicate records that could
        disagree on configuration. With this index, atomic upserts in
        ``store_memagent`` stay unique. It cannot be built while duplicates
        exist; saves still upsert, and the warning names the fix.
        """
        try:
            self.memagent_collection.create_index(
                "agent_id",
                name="agent_id_unique",
                unique=True,
                partialFilterExpression={"agent_id": {"$type": "string"}},
            )
        except Exception as exc:
            logger.warning(
                "Agent records are not unique by agent_id (%s). Remove duplicate "
                "agent records so concurrent saves cannot create more.",
                type(exc).__name__,
            )

    def _ensure_btree_indexes(self) -> None:
        """Create missing btree indexes for hot-path queries.

        Idempotent: ``create_index`` is a no-op when an equivalent index
        already exists, and we trap any unexpected failure so a single
        provisioning hiccup never breaks agent boot. (Failures here only
        cost us query speed, not correctness — the Python-side fallback
        in :meth:`MemoryManager.list_tool_logs` still works without
        the index.)
        """
        collections_by_type = {
            MemoryType.TOOL_LOG: self.tool_log_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.ENTITY_MEMORY: self.entity_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection,
            MemoryType.SEMANTIC_CACHE: self.semantic_cache_collection,
        }
        for memory_type, specs in self._BTREE_INDEX_SPECS.items():
            collection = collections_by_type.get(memory_type)
            if collection is None:
                continue
            for spec in specs:
                name = spec[0]
                fields = spec[1]
                options = spec[2] if len(spec) > 2 else {}
                try:
                    collection.create_index(
                        fields, name=name, background=True, **options
                    )
                except Exception as exc:
                    # Most common: duplicate index spec under a different
                    # name (left over from a prior migration). Log once
                    # and move on — the existing index already covers us.
                    logger.debug(
                        "btree index %s on %s skipped: %s",
                        name,
                        collection.name,
                        exc,
                    )

    def _create_vector_indexes_for_memory_stores(self) -> None:
        """
        Create a vector index for each memory store in MongoDB.

        Returns:
        --------
        None
        """
        # Create vector indexes for all memory store types
        for memory_store_type in MemoryType:
            # PERSONAS collection doesn't need memory_id filter since it's not memory-scoped
            memory_store_present = memory_store_type != MemoryType.PERSONAS

            # Semantic cache needs special handling due to different field name
            if memory_store_type == MemoryType.SEMANTIC_CACHE:
                self._ensure_semantic_cache_vector_index()
            else:
                self._ensure_vector_index(
                    collection=self.db[memory_store_type.value],
                    index_name="vector_index",
                    memory_store=memory_store_present,
                    filter_fields=(
                        ["status", "agent_id", "user_id"]
                        if memory_store_type == MemoryType.SKILLBOX
                        else (
                            ["user_id"]
                            if memory_store_type == MemoryType.ENTITY_MEMORY
                            else None
                        )
                    ),
                )

    def _collection(self, memory_store_type: MemoryType):
        """Resolve the pymongo collection for a memory type.

        Every ``MemoryType`` maps to ``db[<enum value>]`` (the same handles
        bound in ``__init__``); this replaces three identical
        collection-mapping dicts and five if/elif chains that had to be
        extended by hand for every new memory type.
        """
        try:
            if not isinstance(memory_store_type, MemoryType):
                memory_store_type = MemoryType(memory_store_type)
        except Exception:
            return None
        return self.db[memory_store_type.value]

    @staticmethod
    def _normalize_store_input(
        data: Optional[Dict[str, Any]],
        memory_store_type: Optional[MemoryType],
        memory_id: Optional[str],
        memory_unit: Any,
    ) -> Tuple[Dict[str, Any], MemoryType]:
        if memory_unit is not None:
            if hasattr(memory_unit, "model_dump"):
                data = memory_unit.model_dump()
            elif hasattr(memory_unit, "dict"):
                data = memory_unit.dict()
            else:
                data = dict(memory_unit.__dict__)
            memory_store_type = getattr(memory_unit, "memory_type", None) or data.get(
                "memory_type", MemoryType.CONVERSATION_MEMORY
            )

        if data is None or memory_store_type is None:
            raise ValueError(
                "Either (data, memory_store_type) or (memory_unit) must be provided"
            )
        normalized_data = dict(data)
        if memory_id is not None:
            if memory_unit is not None:
                normalized_data["memory_id"] = memory_id
            else:
                normalized_data.setdefault("memory_id", memory_id)
        normalized_type = (
            MemoryType(memory_store_type)
            if isinstance(memory_store_type, str)
            else memory_store_type
        )
        return normalized_data, normalized_type

    @staticmethod
    def _write_document(
        collection: Any, data: Dict[str, Any], memory_type: MemoryType
    ) -> Any:
        if (
            memory_type == MemoryType.SHARED_MEMORY
            and data.get("immutable_trace") is True
            and data.get("record_type") == "observability_trace_bundle"
        ):
            from pymongo.errors import DuplicateKeyError

            logical_id = str(data["memory_id"])
            existing = collection.find_one({"memory_id": logical_id}, {"_id": 1})
            if existing:
                return existing["_id"]
            data = {**data, "_id": logical_id}
            try:
                collection.insert_one(data)
            except DuplicateKeyError:
                pass  # The concurrent first writer owns this immutable ID.
            return logical_id
        if (
            memory_type == MemoryType.SHARED_MEMORY
            and str(data.get("record_type") or "").startswith("observability_")
            and data.get("memory_id")
        ):
            logical_id = str(data["memory_id"])
            data.pop("_id", None)
            existing = collection.find_one({"memory_id": logical_id}, {"_id": 1})
            if existing:
                collection.replace_one({"_id": existing["_id"]}, data)
                return existing["_id"]

        if "_id" in data:
            collection.replace_one({"_id": data["_id"]}, data, upsert=True)
            return data["_id"]
        return collection.insert_one(data).inserted_id

    def store(
        self,
        data: Dict[str, Any] = None,
        memory_store_type: MemoryType = None,
        memory_id: str = None,
        memory_unit: Any = None,
    ) -> str:
        """
        Store data in MongoDB using only _id field as primary key.

        Parameters:
        -----------
        data : Dict[str, Any], optional
            The document to be stored (legacy parameter)
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_id : str, optional
            Memory ID to associate with (new parameter)
        memory_unit : MemoryUnit, optional
            Memory unit object to store (new parameter)

        Returns:
        --------
        str
            The ID of the inserted/updated document (MongoDB _id).
        """
        data, memory_store_type = self._normalize_store_input(
            data, memory_store_type, memory_id, memory_unit
        )

        if memory_store_type == MemoryType.MEMAGENT:
            memagent = (
                data
                if isinstance(data, (MemAgentModel, dict))
                else MemAgentModel(**data)
            )
            stored = self.store_memagent(memagent)
            if isinstance(stored, dict) and stored.get("_id"):
                return str(stored["_id"])
            return str(stored)

        # Get the appropriate collection based on memory type
        collection = self._collection(memory_store_type)

        if collection is None:
            raise ValueError(f"Invalid memory store type: {memory_store_type}")

        data_copy = dict(data)
        for field in _mongo_id_fields_to_strip(memory_store_type):
            data_copy.pop(field, None)
        return str(self._write_document(collection, data_copy, memory_store_type))

    def list_archive_records(self, memory_type):
        return self.list_all(memory_type, include_embedding=True)

    def store_archive_record(
        self, memory_type, record_id, data, *, replace=False, reembed=False
    ):
        collection = self._collection(memory_type)
        # Preserve native ObjectId addressing for ordinary MongoDB records.
        native_id = ObjectId(record_id) if ObjectId.is_valid(record_id) else record_id
        document = dict(data, _id=native_id)
        document.pop("id", None)
        if memory_type == MemoryType.MEMAGENT:
            old = collection.find_one({"agent_id": record_id}, {"_id": 1})
            if old:
                native_id = document["_id"] = old["_id"]
        if reembed:
            from ...embeddings import get_embedding

            text = (
                document.get("content")
                or document.get("description")
                or document.get("name")
            )
            if isinstance(text, str) and text:
                document["embedding"] = get_embedding(text)
        if replace:
            collection.replace_one({"_id": native_id}, document, upsert=True)
        else:
            collection.insert_one(document)
        return record_id

    def store_many(
        self,
        rows: List[Dict[str, Any]],
        memory_store_type: Union[str, MemoryType],
        *,
        memory_id: Optional[str] = None,
    ) -> List[str]:
        """Persist ordinary records with one unordered MongoDB bulk write.

        Agent records and shared observability upserts retain their specialized
        single-row semantics. Evaluation corpora and other ordinary memory
        rows avoid one network round trip per chunk.
        """

        normalized_type = (
            MemoryType(memory_store_type)
            if isinstance(memory_store_type, str)
            else memory_store_type
        )
        if normalized_type in {MemoryType.MEMAGENT, MemoryType.SHARED_MEMORY}:
            return super().store_many(rows, normalized_type, memory_id=memory_id)
        collection = self._collection(normalized_type)
        if collection is None:
            raise ValueError(f"Invalid memory store type: {normalized_type}")

        operations = []
        prepared_documents: List[Dict[str, Any]] = []
        identifiers: List[str] = []
        for row in rows:
            data, _ = self._normalize_store_input(row, normalized_type, memory_id, None)
            document = dict(data)
            for field in _mongo_id_fields_to_strip(normalized_type):
                document.pop(field, None)
            record_id = document.get("_id")
            if record_id is None:
                record_id = ObjectId()
                document["_id"] = record_id
                operations.append(InsertOne(document))
            else:
                operations.append(ReplaceOne({"_id": record_id}, document, upsert=True))
            prepared_documents.append(document)
            identifiers.append(str(record_id))
        if operations:
            try:
                collection.bulk_write(operations, ordered=False)
            except TypeError as exc:
                # Some mongomock releases lag PyMongo's private bulk API and
                # reject its newer ``sort`` argument. Preserve compatibility
                # without masking unrelated production failures.
                if "unexpected keyword argument 'sort'" not in str(exc):
                    raise
                for document in prepared_documents:
                    collection.replace_one(
                        {"_id": document["_id"]}, document, upsert=True
                    )
        return identifiers

    def retrieve_by_query(
        self,
        query: Union[Dict[str, Any], str],
        memory_store_type: MemoryType = None,
        limit: int = 1,
        include_embedding: bool = False,
        memory_id: str = None,
        memory_type: Union[str, "MemoryType"] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB.

        Parameters:
        -----------
        query : Union[Dict[str, Any], str]
            The query to use for retrieval. For semantic cache, this is a string (search text).
            For other memory types, this is a MongoDB query dict.
        memory_store_type : MemoryType, optional
            The type of memory store (legacy parameter)
        memory_type : Union[str, MemoryType], optional
            The type of memory store (new parameter, takes precedence)
        memory_id : str, optional
            Filter results to specific memory_id
        limit : int
            The maximum number of documents to return.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        # Handle new calling style: memory_type takes precedence over memory_store_type
        if memory_type is not None:
            if isinstance(memory_type, str):
                memory_store_type = MemoryType(memory_type)
            else:
                memory_store_type = memory_type

        if memory_store_type is None:
            raise ValueError("Either memory_store_type or memory_type must be provided")

        # Extract user_id up-front for consistent scoping across all branches.
        user_id_scope = kwargs.pop("user_id", _MONGO_UNSET)

        # If memory_id filter is provided, add it to the query
        if memory_id is not None:
            if isinstance(query, dict):
                query = {**query, "memory_id": memory_id}
            else:
                # For string queries (semantic search), store memory_id for filtering
                kwargs["memory_id"] = memory_id

        # Fold user_id into the query/kwargs depending on the call shape.
        # For dict queries against user-scoped memory types we push down an
        # equality predicate (None matches missing/null via $in).
        if user_id_scope is not _MONGO_UNSET:
            if isinstance(query, dict) and _mongo_memory_type_supports_user_id(
                memory_store_type
            ):
                query = {**query, **_mongo_user_id_predicate(user_id_scope)}
            kwargs["user_id"] = user_id_scope

        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}

        if memory_store_type == MemoryType.PERSONAS:
            return self.retrieve_persona_by_query(query, limit=limit) or []
        elif memory_store_type == MemoryType.TOOLBOX:
            return self.retrieve_toolbox_item(query, limit) or []
        elif memory_store_type == MemoryType.SKILLBOX:
            return (
                self.retrieve_skillbox_item(
                    query,
                    limit,
                    statuses=kwargs.get("statuses"),
                    agent_id=kwargs.get("agent_id", _MONGO_UNSET),
                    user_id=kwargs.get("user_id", _MONGO_UNSET),
                )
                or []
            )
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.retrieve_workflow_by_query(query, limit) or []
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            # Short-term memory is a scratchpad — dict queries do a filter,
            # string queries from the relevant-memories path have no
            # semantic index here yet, so degrade gracefully to ``[]``
            # rather than crashing PyMongo's ``find()``.
            if isinstance(query, dict):
                return self.short_term_memory_collection.find(query, projection).limit(
                    limit
                )
            return []
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            # Three calling shapes:
            #   1. Plain dict filter (legacy) → ``find()``
            #   2. Dict carrying a pre-computed ``embedding`` (from
            #      :meth:`KnowledgeBase.retrieve_knowledge_by_query`) → vector
            #      search via that embedding
            #   3. Raw string (from MemAgent.retrieve_relevant_memories) →
            #      embed-then-vector-search via the helper
            if isinstance(query, dict):
                if "embedding" in query and isinstance(query["embedding"], list):
                    embedding = query["embedding"]
                    kb_kwargs = dict(kwargs)
                    if "namespace" in query and "namespace" not in kb_kwargs:
                        kb_kwargs["namespace"] = query["namespace"]
                    return self._knowledge_base_vector_search(
                        embedding,
                        limit=query.get("limit", limit),
                        **kb_kwargs,
                    )
                return self.knowledge_base_collection.find(query, projection).limit(
                    limit
                )
            if isinstance(query, str):
                return self.find_similar_knowledge_base_entries(
                    query,
                    limit=limit,
                    include_embedding=include_embedding,
                    **kwargs,
                )
            return []
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            # Dict → straight filter; string → semantic recall via vector
            # index over per-turn embeddings. Previously the string path
            # crashed with ``filter must be an instance of dict`` because
            # PyMongo's ``find()`` rejects raw strings.
            if isinstance(query, dict):
                return self.conversation_memory_collection.find(
                    query, projection
                ).limit(limit)
            if isinstance(query, str):
                return self.find_similar_conversation_entries(
                    query,
                    limit=limit,
                    include_embedding=include_embedding,
                    **kwargs,
                )
            return []
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            if isinstance(query, dict):
                # Shared memory is not in the user-scoped set (list_all stays
                # unscoped, as on the filesystem provider), but an explicit
                # user_id on a dict query must still filter — the filesystem
                # provider applies it to every dict query.
                if user_id_scope is not _MONGO_UNSET:
                    query = {**query, **_mongo_user_id_predicate(user_id_scope)}
                return self.shared_memory_collection.find(query, projection).limit(
                    limit
                )
            return []
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            return (
                self.retrieve_entity_memory_records(
                    query, limit, include_embedding=include_embedding, **kwargs
                )
                or []
            )
        elif memory_store_type == MemoryType.SUMMARIES:
            return self.retrieve_summaries_by_query(query, limit) or []
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            # For semantic cache, we need to handle two different cases:
            # 1. Dict query: Loading existing cache entries (e.g., {"agent_id": "xyz"})
            # 2. String query: Semantic similarity search (e.g., "What is Python?")
            if isinstance(query, dict):
                # This is a filter query for loading existing cache entries.
                # Honour include_embedding: the in-memory cache preload needs
                # the stored vectors back, otherwise it loads zero rows.
                return self.semantic_cache_collection.find(query, projection).limit(
                    limit
                )
            else:
                # This is a text query for semantic similarity search
                return self.find_similar_cache_entries(
                    query, limit=limit, include_embedding=include_embedding, **kwargs
                )
        elif memory_store_type == MemoryType.MEMAGENT:
            if isinstance(query, dict):
                return self.memagent_collection.find(query, projection).limit(limit)
            return []
        elif memory_store_type == MemoryType.TOOL_LOG:
            if isinstance(query, dict):
                return self.tool_log_collection.find(query, projection).limit(limit)
            return []
        # Fall-through: unknown memory type — never return None implicitly,
        # which would break callers that ``len()`` the result.
        return []

    def retrieve_by_id(
        self, id: str, memory_store_type: MemoryType
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by _id.

        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        # Get the appropriate collection
        collection = self._collection(memory_store_type)
        if collection is None:
            return None

        # Set projection to exclude embedding for performance
        projection = (
            {"embedding": 0}
            if memory_store_type
            in [
                MemoryType.PERSONAS,
                MemoryType.TOOLBOX,
                MemoryType.SKILLBOX,
                MemoryType.WORKFLOW_MEMORY,
                MemoryType.SUMMARIES,
            ]
            else None
        )

        # For semantic cache, exclude embedding by default for performance
        if memory_store_type == MemoryType.SEMANTIC_CACHE:
            projection = {"embedding": 0}

        try:
            doc = collection.find_one(_mongo_id_predicate(id), projection)
            # Tool logs also carry a UUID tool_log_id, and that is the ID the
            # recent-logs hint shows the model in later turns.
            if doc is None and memory_store_type == MemoryType.TOOL_LOG:
                doc = collection.find_one({"tool_log_id": str(id)}, projection)
            if doc and memory_store_type == MemoryType.CONVERSATION_MEMORY:
                self._normalize_legacy_fields(doc)
            return doc
        except Exception:
            pass

        return None

    def retrieve_by_name(
        self, name: str, memory_store_type: MemoryType, include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document from MongoDB by name.

        Parameters:
        -----------
        name : str
            The name of the document to retrieve.
        memory_store_type : MemoryType
            The type of memory store to retrieve from.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.

        Returns:
        --------
        Optional[Dict[str, Any]]
            The retrieved document, or None if not found.
        """
        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}

        collection = self._collection(memory_store_type)
        if collection is None:
            return None
        # Two stores address rows by a non-name key:
        if memory_store_type == MemoryType.SHARED_MEMORY:
            return collection.find_one({"memory_id": name}, projection)
        if memory_store_type == MemoryType.SEMANTIC_CACHE:
            return collection.find_one(
                {"$or": [{"cache_key": name}, {"query_text": name}]}, projection
            )
        return collection.find_one({"name": name}, projection)

    def retrieve_persona_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a persona or several personas from MongoDB.
        This function uses a vector search to retrieve the most similar personas.

        Parameters:
        -----------
        query : Dict[str, Any]

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved personas, or None if not found.
        """

        # Get the embedding for the query
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        pipeline = self._build_vector_search_pipeline(embedding, limit)
        results = self._run_vector_search(
            self.persona_collection, pipeline, label="persona"
        )
        return results if results else None

    def retrieve_toolbox_item(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a toolbox item or several items from MongoDB.
        This function uses a vector search to retrieve the most similar toolbox items.
        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of toolbox items to return.

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved toolbox items, or None if not found.
        """

        # Get the embedding for the query
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        pipeline = self._build_vector_search_pipeline(embedding, limit)
        results = self._run_vector_search(
            self.toolbox_collection, pipeline, label="toolbox"
        )
        return results if results else None

    def retrieve_skillbox_item(
        self,
        query: Union[Dict[str, Any], str],
        limit: int = 1,
        *,
        statuses: Optional[List[str]] = None,
        agent_id: Any = _MONGO_UNSET,
        user_id: Any = _MONGO_UNSET,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieve learned skills by vector similarity.

        Skill documents embed applicability semantics (when to apply), so
        this matches query intent rather than tool mechanism. Each result
        carries the ``score`` field from ``$vectorSearch`` — the Skillbox
        layer enforces its own (stricter) similarity threshold on it.
        """
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(
                "Failed to generate a skillbox query embedding (%s)",
                type(e).__name__,
            )
            return []

        if self.lazy_vector_indexes:
            self._ensure_vector_index_for_collection(
                self.skillbox_collection,
                MemoryType.SKILLBOX.value,
                memory_store=True,
            )

        filters = []
        if statuses:
            filters.append({"status": {"$in": list(statuses)}})
        if agent_id is not _MONGO_UNSET:
            filters.append({"agent_id": {"$eq": agent_id}})
        if user_id is not _MONGO_UNSET:
            filters.append({"user_id": {"$eq": user_id}})
        search_filter = (
            {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)
        )
        pipeline = self._build_vector_search_pipeline(
            embedding, limit, search_filter=search_filter
        )
        results = self._run_vector_search(
            self.skillbox_collection, pipeline, label="skillbox"
        )
        return results if results else None

    def retrieve_entity_memory_records(
        self,
        query: Union[Dict[str, Any], str],
        limit: int = 5,
        include_embedding: bool = False,
        **kwargs,
    ):
        """
        Retrieve entity memory records using a filter or semantic query.
        """

        if isinstance(query, dict):
            projection = {} if include_embedding else {"embedding": 0}
            return self.entity_memory_collection.find(query, projection).limit(limit)

        if getattr(self, "lazy_vector_indexes", False):
            # Existing entity indexes also need the user_id prefilter. Lazy
            # deployments must reconcile that definition before issuing a
            # scoped query; unsupported local/community servers are marked
            # unavailable and return through the exact fallback upstream.
            self._ensure_vector_index_for_collection(
                self.entity_memory_collection,
                MemoryType.ENTITY_MEMORY.value,
                memory_store=True,
            )
            if self._vector_index_unavailable(self.entity_memory_collection.name):
                raise VectorSearchUnavailableError(
                    "Vector search is unavailable for entity_memory"
                )

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.warning(
                "Failed to generate an entity query embedding (%s)",
                type(e).__name__,
            )
            raise VectorSearchExecutionError(
                "Entity semantic query embedding failed",
                reason="semantic_embedding_failed",
            ) from e

        search_filter: Dict[str, Any] = {}
        memory_id = kwargs.get("memory_id")
        if memory_id is not None:
            search_filter["memory_id"] = str(memory_id)
        user_id_scope = kwargs.get("user_id", _MONGO_UNSET)
        if user_id_scope is not _MONGO_UNSET:
            search_filter.update(_mongo_user_id_predicate(user_id_scope))
        pipeline = self._build_vector_search_pipeline(
            embedding, limit, search_filter=search_filter or None
        )
        return self._run_vector_search(
            self.entity_memory_collection,
            pipeline,
            label="entity_memory",
            raise_on_error=True,
        )

    def retrieve_workflow_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a workflow or several workflows from MongoDB.
        This function uses a vector search to retrieve the most similar workflows.

        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of workflows to return.

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved workflows, or None if not found.
        """

        # Get the embedding for the query
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        pipeline = self._build_vector_search_pipeline(embedding, limit)
        results = self._run_vector_search(
            self.workflow_memory_collection, pipeline, label="workflow_memory"
        )
        return results if results else None

    def retrieve_summaries_by_query(
        self, query: Dict[str, Any], limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve summaries by query using vector search.

        Parameters:
        -----------
        query : Dict[str, Any]
            The query to use for retrieval.
        limit : int
            The maximum number of summaries to return.

        Returns:
        --------
        Optional[List[Dict[str, Any]]]
            The retrieved summaries, or None if not found.
        """
        # Get the embedding for the query
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            return []

        pipeline = self._build_vector_search_pipeline(embedding, limit)
        results = self._run_vector_search(
            self.summaries_collection, pipeline, label="summaries"
        )
        return results if results else None

    def find_similar_conversation_entries(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar conversation_memory entries using vector search.

        Mirrors :meth:`find_similar_cache_entries` but targets the
        conversation memory collection. Conversation rows store an
        ``embedding`` field per turn (see ``MemoryManager.create_conversation_memory_unit``);
        this method embeds the query string, runs an Atlas ``$vectorSearch``,
        and applies the standard ``memory_id`` / ``user_id`` filters.

        Parameters
        ----------
        query : str
            Free-text query to match against past turns.
        limit : int
            Maximum number of similar entries to return.
        **kwargs : Dict[str, Any]
            Optional ``memory_id``, ``user_id`` filters used by the
            ``MemAgent`` runtime when building relevant-memory context.

        Returns
        -------
        List[Dict[str, Any]]
            Conversation entries with a ``score`` field (cosine similarity
            via vectorSearchScore). Returns ``[]`` on any failure so the
            caller can rely on the return shape.
        """
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for conversation_memory query: {e}"
            )
            return []

        search_filter: Dict[str, Any] = {}
        memory_id = kwargs.get("memory_id")
        if memory_id is not None:
            search_filter["memory_id"] = str(memory_id)
        user_id_scope = kwargs.get("user_id", _MONGO_UNSET)
        if user_id_scope is not _MONGO_UNSET:
            search_filter.update(_mongo_user_id_predicate(user_id_scope))
        thread_id = kwargs.get("thread_id")
        if thread_id is not None:
            search_filter["thread_id"] = str(thread_id)

        pipeline = self._build_vector_search_pipeline(
            embedding,
            limit,
            search_filter=search_filter or None,
            include_embedding=bool(kwargs.get("include_embedding")),
        )
        return self._run_vector_search(
            self.conversation_memory_collection,
            pipeline,
            label="conversation_memory",
        )

    def find_similar_knowledge_base_entries(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar knowledge_base entries using vector search.

        The :class:`KnowledgeBase` helper normally passes a dict carrying a
        pre-computed embedding, but callers like
        ``MemAgent.retrieve_relevant_memories`` pass a raw string. Previously
        the string path crashed on ``.find(<str>)``; this method gives string
        queries a real semantic path.
        """
        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.warning(
                f"Failed to generate embedding for knowledge_base query: {e}"
            )
            return []

        return self._knowledge_base_vector_search(
            embedding,
            limit=limit,
            query_text=query,
            **kwargs,
        )

    def _knowledge_base_scoped_fallback(
        self,
        *,
        limit: int,
        query_text: Optional[str] = None,
        reason: str,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Bounded, provider-local fallback when Atlas vector search is absent.

        This is deliberately not a replacement for vector search.  It keeps a
        memory-first harness useful on MongoDB Community, local Docker and
        lower Atlas tiers by retrieving only the exact memory/user/namespace
        scope, then applying a small lexical rank.  The degraded provenance is
        attached to every row so callers and traces can distinguish it from a
        semantic match.
        """
        search_filter: Dict[str, Any] = {}
        memory_id = kwargs.get("memory_id")
        if memory_id is not None:
            search_filter["memory_id"] = str(memory_id)
        user_id_scope = kwargs.get("user_id", _MONGO_UNSET)
        if user_id_scope is not _MONGO_UNSET:
            search_filter.update(_mongo_user_id_predicate(user_id_scope))
        namespace = kwargs.get("namespace")
        if namespace:
            search_filter["namespace"] = str(namespace)

        bounded_limit = max(1, min(int(limit), 100))
        candidate_limit = min(200, max(32, bounded_limit * 8))
        rows = list(
            self.knowledge_base_collection.find(
                search_filter,
                {"embedding": 0},
            ).limit(candidate_limit)
        )
        query_tokens = _lexical_tokens(query_text)

        def lexical_score(row: Dict[str, Any]) -> float:
            return _lexical_overlap(query_tokens, row.get("content"))

        rows.sort(
            key=lambda row: (
                lexical_score(row),
                float(row.get("importance") or 0.0),
                str(row.get("updated_at") or row.get("created_at") or ""),
            ),
            reverse=True,
        )
        results: List[Dict[str, Any]] = []
        for row in rows[:bounded_limit]:
            item = dict(row)
            item["score"] = lexical_score(item)
            item["retrieval_mode"] = "scoped_lexical_fallback"
            item["retrieval_degraded"] = True
            item["retrieval_reason"] = reason
            results.append(item)
        return results

    def _knowledge_base_vector_search(
        self, embedding: List[float], limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """Internal: run a ``$vectorSearch`` on the KB collection with a
        pre-computed embedding plus optional ``memory_id`` / ``user_id`` /
        ``namespace`` filters."""
        search_filter: Dict[str, Any] = {}
        memory_id = kwargs.get("memory_id")
        if memory_id is not None:
            search_filter["memory_id"] = str(memory_id)
        user_id_scope = kwargs.get("user_id", _MONGO_UNSET)
        if user_id_scope is not _MONGO_UNSET:
            search_filter.update(_mongo_user_id_predicate(user_id_scope))
        namespace = kwargs.get("namespace")
        if namespace:
            search_filter["namespace"] = str(namespace)

        pipeline = self._build_vector_search_pipeline(
            embedding,
            limit,
            search_filter=search_filter or None,
            include_embedding=bool(kwargs.get("include_embedding")),
        )
        try:
            return self._run_vector_search(
                self.knowledge_base_collection,
                pipeline,
                label="knowledge_base",
                raise_on_error=True,
            )
        except VectorSearchExecutionError as exc:
            fallback_kwargs = dict(kwargs)
            query_text = fallback_kwargs.pop("query_text", None)
            return self._knowledge_base_scoped_fallback(
                limit=limit,
                query_text=query_text,
                reason=exc.reason,
                **fallback_kwargs,
            )

    def find_similar_cache_entries(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar cache entries using vector search.

        Parameters:
        -----------
        query : str
            The query to search for
        limit : int
            Maximum number of results
        kwargs : Dict[str, Any]
            Additional filters to apply to the query

        Returns:
        --------
        List[Dict[str, Any]]
            List of similar cache entries with similarity scores
        """

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for semantic cache query: {e}")
            return []

        # Extract the filter from kwargs (agent_id, memory_id, session_id)
        # Build filter conditionally - only include agent_id if present (for LOCAL scope)
        search_filter = {}
        if "agent_id" in kwargs and kwargs["agent_id"] is not None:
            search_filter["agent_id"] = str(kwargs["agent_id"])
        if "memory_id" in kwargs and kwargs["memory_id"] is not None:
            search_filter["memory_id"] = str(kwargs["memory_id"])
        if "session_id" in kwargs and kwargs["session_id"] is not None:
            search_filter["session_id"] = str(kwargs["session_id"])

        # Tenant isolation: only allow matches within the same user_id scope.
        user_id_scope = kwargs.get("user_id", _MONGO_UNSET)
        if user_id_scope is not _MONGO_UNSET:
            search_filter.update(_mongo_user_id_predicate(user_id_scope))

        pipeline = self._build_vector_search_pipeline(
            embedding,
            limit,
            search_filter=search_filter or None,
            num_candidates=100,  # cache hit-rate tuned, not query-volume scaled
            include_embedding=bool(kwargs.get("include_embedding")),
        )
        return self._run_vector_search(
            self.semantic_cache_collection, pipeline, label="semantic_cache"
        )

    def clear_semantic_cache(
        self, agent_id: Optional[str] = None, memory_id: Optional[str] = None
    ) -> int:
        """
        Clear semantic cache entries with optional filtering.

        Parameters:
        -----------
        agent_id : Optional[str]
            Clear only entries for this agent ID
        memory_id : Optional[str]
            Clear only entries for this memory ID

        Returns:
        --------
        int
            Number of entries deleted
        """
        query = {}
        if agent_id:
            query["agent_id"] = agent_id
        if memory_id:
            query["memory_id"] = memory_id

        try:
            result = self.semantic_cache_collection.delete_many(query)
            return result.deleted_count
        except Exception as e:
            logger.warning(f"Failed to clear semantic cache: {e}")
            return 0

    def purge_expired_semantic_cache(self, now: Any = None) -> int:
        """Delete semantic-cache rows whose ``expires_at`` is before ``now``.

        Rows written by :class:`SemanticCache` carry a datetime ``expires_at``;
        rows imported from archives or other providers may carry an ISO-8601
        string instead. Both are honoured. The TTL index on ``expires_at``
        lets MongoDB reap date-typed rows in the background; this is the
        explicit, immediate variant the base contract exposes.

        Parameters:
        -----------
        now : datetime | str | float, optional
            The cutoff; defaults to the current UTC time.

        Returns:
        --------
        int
            Number of rows deleted.
        """
        cutoff = _parse_expiry(now) if now is not None else None
        if cutoff is None:
            cutoff = datetime.now(timezone.utc)

        deleted = 0
        try:
            deleted += self.semantic_cache_collection.delete_many(
                {"expires_at": {"$lt": cutoff}}
            ).deleted_count
            # BSON comparisons are type-bracketed: a datetime cutoff never
            # matches string timestamps, so compare those client-side.
            expired_ids: List[Any] = []
            for row in self.semantic_cache_collection.find(
                {"expires_at": {"$type": "string"}}, {"_id": 1, "expires_at": 1}
            ):
                stamp = _parse_expiry(row.get("expires_at"))
                if stamp is not None and stamp < cutoff:
                    expired_ids.append(row["_id"])
            if expired_ids:
                deleted += self.semantic_cache_collection.delete_many(
                    {"_id": {"$in": expired_ids}}
                ).deleted_count
        except Exception as e:
            logger.warning(f"Failed to purge expired semantic cache: {e}")
        return int(deleted)

    def delete_observability_bundle(self, record_id, fingerprint):
        from ...observability.index import digest
        from ...observability.normalization import read_payload

        row = self.shared_memory_collection.find_one({"memory_id": record_id})
        payload = read_payload(row) if row else None
        if (
            not payload
            or payload.get("record_type") != "observability_trace_bundle"
            or digest(payload) != fingerprint
        ):
            return False
        return (
            self.shared_memory_collection.delete_one(
                {"_id": row["_id"], "content": row["content"]}
            ).deleted_count
            == 1
        )

    def delete_by_id(self, id: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by _id.

        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        # Get the appropriate collection
        collection = self._collection(memory_store_type)
        if collection is None:
            return False

        # Delete using MongoDB _id only (string or ObjectId form)
        try:
            result = collection.delete_one(_mongo_id_predicate(id))
            return result.deleted_count > 0
        except Exception:
            pass

        return False

    def delete_by_name(self, name: str, memory_store_type: MemoryType) -> bool:
        """
        Delete a document from MongoDB by name.

        Parameters:
        -----------
        name : str
            The name of the document to delete.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        collection = self._collection(memory_store_type)
        if collection is None:
            return False
        if memory_store_type == MemoryType.SHARED_MEMORY:
            result = collection.delete_one({"memory_id": name})
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            result = collection.delete_one(
                {"$or": [{"cache_key": name}, {"query_text": name}]}
            )
        else:
            result = collection.delete_one({"name": name})

        return result.deleted_count > 0

    def delete_all(self, memory_store_type: MemoryType) -> bool:
        """
        Delete all documents within a memory store type in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        collection = self._collection(memory_store_type)
        if collection is None:
            return False
        # Success means "the store is now empty", not "rows were removed":
        # an already-empty store is not a failure (filesystem/Notion agree).
        collection.delete_many({})

        return True

    def list_all(
        self,
        memory_store_type: MemoryType,
        include_embedding: bool = False,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """
        List all documents within a memory store type in MongoDB.

        Parameters:
        -----------
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.
        user_id : str, optional
            Multi-tenant scope. When provided, results are restricted to rows
            whose ``user_id`` matches. When the sentinel default is used no
            ``user_id`` filter is applied (legacy callers).

        Returns:
        --------
        List[Dict[str, Any]]
            The list of all documents from MongoDB.
        """
        # Define projection to exclude embeddings by default
        projection = {} if include_embedding else {"embedding": 0}

        mongo_filter: Dict[str, Any] = {}
        if user_id is not _MONGO_UNSET and _mongo_memory_type_supports_user_id(
            memory_store_type
        ):
            mongo_filter.update(_mongo_user_id_predicate(user_id))

        collection = self._collection(memory_store_type)
        if collection is None:
            logger.warning(
                f"Unsupported memory store type for list_all: {memory_store_type}"
            )
            return []
        return list(collection.find(mongo_filter, projection))

    def estimate_count(self, memory_store_type: MemoryType) -> Optional[int]:
        """Approximate document count from collection metadata, without a scan.

        Each memory type has its own collection, so this is the store's size.
        Returns None for an unknown memory type.
        """
        collection = self._collection(memory_store_type)
        if collection is None:
            return None
        return int(collection.estimated_document_count())

    def list_recent(
        self,
        memory_store_type: MemoryType,
        limit: int = 100,
        include_embedding: bool = False,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """Newest documents first, at most ``limit`` of them.

        Ordered by the record ``timestamp`` and then ``_id``: stores mix
        ObjectId and string ids (immutable trace bundles use string ids), so
        ``_id`` alone does not follow time. MongoDB applies the limit during
        the sort, so a page never loads the whole collection.
        """
        projection = {} if include_embedding else {"embedding": 0}
        mongo_filter: Dict[str, Any] = {}
        if user_id is not _MONGO_UNSET and _mongo_memory_type_supports_user_id(
            memory_store_type
        ):
            mongo_filter.update(_mongo_user_id_predicate(user_id))
        collection = self._collection(memory_store_type)
        if collection is None:
            return []
        cursor = collection.find(mongo_filter, projection).sort(
            [("timestamp", -1), ("_id", -1)]
        )
        return list(cursor.limit(max(1, int(limit))))

    # BSON types that sort below ObjectId / string. Pages walk ``_id`` in
    # descending order, and MongoDB range operators only match values of the
    # same type, so a cursor must explicitly include the lower types.
    _NUMERIC_ID_TYPES = ["double", "int", "long", "decimal"]

    @staticmethod
    def _encode_observability_cursor(value: Any) -> str:
        # Tag the id type: a string id can look like ObjectId hex.
        tagged = f"o:{value}" if isinstance(value, ObjectId) else f"s:{value}"
        return (
            base64.urlsafe_b64encode(tagged.encode("utf-8")).decode("ascii").rstrip("=")
        )

    @staticmethod
    def _decode_observability_cursor(value: str) -> Any:
        try:
            padded = value + "=" * (-len(value) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
        except Exception as exc:
            raise ValueError("Invalid observability cursor") from exc
        if decoded.startswith("o:") and ObjectId.is_valid(decoded[2:]):
            return ObjectId(decoded[2:])
        if decoded.startswith("s:"):
            return decoded[2:]
        # Untagged cursors from earlier versions.
        return ObjectId(decoded) if ObjectId.is_valid(decoded) else decoded

    def observability_row_cursor(self, row: Dict[str, Any]) -> str:
        """Cursor that continues ``query_observability_records`` after ``row``."""
        return self._encode_observability_cursor(row["_id"])

    @classmethod
    def _observability_cursor_clause(cls, cursor_value: Any) -> Dict[str, Any]:
        """Everything after ``cursor_value`` in descending ``_id`` order.

        Stores mix ObjectId ids (older rows) and string ids (immutable trace
        bundles). Descending BSON order is ObjectId, then string, then numbers;
        a bare ``$lt`` would stop at the first type boundary and silently skip
        every row of the lower types.
        """
        lower_types = list(cls._NUMERIC_ID_TYPES)
        if isinstance(cursor_value, ObjectId):
            lower_types.insert(0, "string")
        return {
            "$or": [
                {"_id": {"$lt": cursor_value}},
                *({"_id": {"$type": name}} for name in lower_types),
            ]
        }

    def get_observability_index(self):
        from ...observability.mongo_index import MongoSpanIndex

        if not hasattr(self, "_observability_index"):
            self._observability_index = MongoSpanIndex(self.db)
        return self._observability_index

    def query_memory_observations(
        self,
        memory_store_type,
        *,
        agent_id=None,
        memory_ids=None,
        application_id=None,
        exclude_ids=(),
        limit=200,
        **kwargs,
    ):
        from ..base import _UNSET, _observation_matches

        user_id = kwargs.get("user_id", _UNSET)
        kind = MemoryType(memory_store_type)
        clauses = []
        if agent_id:
            owners = [
                {"agent_id": str(agent_id)},
                {"agent_id": None, "owner_agent_id": str(agent_id)},
            ]
            if memory_ids:
                owners.append(
                    {
                        "agent_id": None,
                        "owner_agent_id": None,
                        "memory_id": {"$in": list(memory_ids)},
                    }
                )
            clauses.append({"$or": owners})
        elif memory_ids:
            clauses.append({"memory_id": {"$in": list(memory_ids)}})
        if user_id is not _UNSET:
            clauses.append({"user_id": user_id})
        if application_id is not None:
            clauses.append({"application_id": application_id})
        if exclude_ids:
            excluded = list(exclude_ids)
            excluded.extend(
                ObjectId(value) for value in exclude_ids if ObjectId.is_valid(value)
            )
            clauses.append({"_id": {"$nin": excluded}})
        predicate = {"$and": clauses} if clauses else {}
        rows = (
            self._collection(kind)
            .find(predicate, {"embedding": 0})
            .limit(max(1, min(int(limit), 1000)))
        )
        return [
            row
            for row in rows
            if _observation_matches(
                row,
                agent_id=agent_id,
                memory_ids=memory_ids,
                user_id=user_id,
                application_id=application_id,
                exclude_ids=exclude_ids,
            )
        ]

    def query_observability_records(
        self,
        memory_store_type: Any,
        *,
        agent_ids: Optional[List[str]] = None,
        memory_ids: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        user_id: Any = _MONGO_UNSET,
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
        """Run an indexed, cursor-paginated trace query in MongoDB."""
        started_at = time.perf_counter()
        try:
            resolved_type = (
                memory_store_type
                if isinstance(memory_store_type, MemoryType)
                else MemoryType(memory_store_type)
            )
        except Exception as exc:
            raise ValueError("Unsupported observability memory type") from exc
        if resolved_type not in {
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.TOOL_LOG,
            MemoryType.SHARED_MEMORY,
        }:
            raise ValueError(
                "Observability queries support conversation, tool, and shared logs"
            )

        safe_limit = max(1, min(int(limit or 250), 1000))
        clauses: List[Dict[str, Any]] = []
        wanted_agents = [str(value) for value in (agent_ids or []) if value]
        wanted_memories = [str(value) for value in (memory_ids or []) if value]
        if wanted_agents or wanted_memories:
            identity_clauses: List[Dict[str, Any]] = []
            if wanted_agents:
                identity_clauses.append({"agent_id": {"$in": wanted_agents}})
            if wanted_memories:
                memory_field = (
                    "trace_memory_id"
                    if resolved_type == MemoryType.SHARED_MEMORY
                    else "memory_id"
                )
                identity_clauses.append({memory_field: {"$in": wanted_memories}})
            clauses.append(
                identity_clauses[0]
                if len(identity_clauses) == 1
                else {"$or": identity_clauses}
            )
        if thread_id is not None:
            clauses.append({"thread_id": str(thread_id)})
        if user_id is not _MONGO_UNSET:
            clauses.append(_mongo_user_id_predicate(user_id))
        if application_id is not None:
            clauses.append({"application_id": application_id})
        if record_type is not None:
            clauses.append({"record_type": str(record_type)})
        if tool_name is not None:
            clauses.append({"tool_name": str(tool_name)})
        if success is not None:
            clauses.append({"success": bool(success)})
        for key, value in (event_filters or {}).items():
            field = (
                "trace_memory_id"
                if key == "memory_id"
                else (
                    "history_" + key
                    if key in {"memory_type", "action", "actor"}
                    else key
                )
            )
            # Older journals lack these indexed metadata fields. Keep them
            # eligible and let the reader validate the canonical payload.
            clauses.append({"$or": [{field: value}, {field: {"$exists": False}}]})
        if start_time is not None or end_time is not None:
            timestamp_filter: Dict[str, Any] = {}
            if start_time is not None:
                timestamp_filter["$gte"] = start_time
            if end_time is not None:
                timestamp_filter["$lte"] = end_time
            clauses.append({"timestamp": timestamp_filter})
        if cursor:
            clauses.append(
                self._observability_cursor_clause(
                    self._decode_observability_cursor(cursor)
                )
            )

        mongo_filter: Dict[str, Any]
        if not clauses:
            mongo_filter = {}
        elif len(clauses) == 1:
            mongo_filter = clauses[0]
        else:
            mongo_filter = {"$and": clauses}

        collection = self._collection(resolved_type)
        projection = {"embedding": 0}
        docs = list(
            collection.find(mongo_filter, projection)
            .sort("_id", -1)
            .limit(safe_limit + 1)
        )
        has_more = len(docs) > safe_limit
        page = docs[:safe_limit]
        next_cursor = (
            self._encode_observability_cursor(page[-1]["_id"])
            if has_more and page
            else None
        )
        return {
            "items": page,
            "next_cursor": next_cursor,
            "truncated": has_more,
            "limit": safe_limit,
            "scanned_count": len(docs),
            "scanned_count_is_lower_bound": has_more,
            "query_duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "freshness": str(page[0].get("timestamp") or "") if page else None,
            "provider_native": True,
        }

    def list_tool_logs(
        self,
        memory_id: Optional[str] = None,
        user_id: Any = _MONGO_UNSET,
        limit: int = 20,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return recent tool-log rows for a given memory/user, native-side.

        ``MemoryManager.list_tool_logs`` previously had to ``list_all``
        the entire tool_log collection and filter + sort + slice in
        Python, which becomes O(n) in the user's lifetime tool-call
        count. This method runs the equivalent query as a single
        ``find(...).sort("timestamp", -1).limit(N)`` aggregate; with the
        ``tool_log_memory_timestamp`` / ``tool_log_user_timestamp``
        compound indexes created in :meth:`_ensure_btree_indexes`,
        MongoDB satisfies it from the index without scanning.

        Implemented as an optional method on the provider — MemoryManager
        uses ``hasattr`` to prefer it when available and falls back to
        the in-memory scan for providers that don't ship one. This keeps
        ``MemoryProvider`` provider-agnostic.
        """
        mongo_filter: Dict[str, Any] = {}
        if memory_id is not None:
            mongo_filter["memory_id"] = str(memory_id)
        if user_id is not _MONGO_UNSET:
            mongo_filter.update(_mongo_user_id_predicate(user_id))
        # Thread scoping: keep the tool-log digest to THIS conversation so one
        # thread's tool ids (group_id / doc_id / deck_id …) can't fall off a
        # global last-N digest or leak into another thread.
        if thread_id is not None:
            mongo_filter["thread_id"] = str(thread_id)

        try:
            cursor = (
                self.tool_log_collection.find(mongo_filter, {"embedding": 0})
                .sort("timestamp", -1)
                .limit(max(int(limit), 1) if limit else 0)
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("list_tool_logs query failed: %s", exc)
            return []

    def list_summaries(
        self,
        memory_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Any = _MONGO_UNSET,
        thread_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return recent summary rows scoped to a memory/agent, native-side.

        ``MemoryManager.load_summaries_for_thread`` previously read the whole
        summaries collection via ``list_all`` on every agent turn and
        filtered in Python. This runs the equivalent scoped query as one
        ``find(...).sort("period_end", -1).limit(N)``.

        ``memory_id`` / ``agent_id`` are OR-matched (a summary belongs to the
        thread if either matches), mirroring the Python-side rule in
        ``MemoryManager.load_summaries_for_thread``, which still applies its
        own normalization on top.
        """
        mongo_filter: Dict[str, Any] = {}
        scope_clauses: List[Dict[str, Any]] = []
        if memory_id:
            scope_clauses.append({"memory_id": str(memory_id)})
        if agent_id:
            scope_clauses.append({"agent_id": str(agent_id)})
        if scope_clauses:
            mongo_filter["$or"] = scope_clauses
        if user_id is not _MONGO_UNSET:
            mongo_filter.update(_mongo_user_id_predicate(user_id))
        if thread_id is not None:
            mongo_filter["thread_id"] = str(thread_id)

        try:
            cursor = (
                self.summaries_collection.find(mongo_filter, {"embedding": 0})
                .sort("period_end", -1)
                .limit(max(int(limit), 1) if limit else 0)
            )
            return list(cursor)
        except Exception as exc:
            logger.warning("list_summaries query failed: %s", exc)
            return []

    def compare_and_swap_shared_memory(self, memory_id, expected_content, content):
        collection = self._collection(MemoryType.SHARED_MEMORY)
        if collection is None:
            raise RuntimeError("Shared-memory collection is unavailable")
        result = collection.update_one(
            {**_mongo_id_predicate(memory_id), "content": expected_content},
            {"$set": {"content": content}},
        )
        return result.matched_count == 1

    def update_by_id(
        self, id: str, data: Dict[str, Any], memory_store_type: MemoryType
    ) -> bool:
        """
        Update a document in a memory store type in MongoDB by _id.

        Parameters:
        -----------
        id : str
            The MongoDB _id of the document to update.
        data : Dict[str, Any]
            The data to update the document with.
        memory_store_type : MemoryType
            The type of memory store (e.g., "persona", "toolbox", etc.)

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        # Success means "a row was found", not "a field changed": an
        # idempotent re-save of identical data is still a successful update
        # (matches the filesystem and Notion providers).
        if memory_store_type == MemoryType.MEMAGENT:
            payload, _ = self._prepare_memagent_payload(data)
            payload.pop("_id", None)
            try:
                result = self.memagent_collection.update_one(
                    self._memagent_id_query(id), {"$set": payload}
                )
                success = result.matched_count > 0
                if not success:
                    logger.warning(
                        f"Update operation found no documents to modify for id: {id}"
                    )
                return success
            except Exception as e:
                logger.error(
                    f"Error updating document with id {id}: {e}", exc_info=True
                )
                return False
        if memory_store_type == MemoryType.SEMANTIC_CACHE and not ObjectId.is_valid(id):
            try:
                result = self.semantic_cache_collection.update_one(
                    {"$or": [_mongo_id_predicate(id), {"cache_key": id}]},
                    {"$set": data},
                )
                return result.matched_count > 0
            except Exception as e:
                logger.error(f"Error updating semantic cache with id {id}: {e}")
                return False

        # Get the appropriate collection
        collection = self._collection(memory_store_type)
        if collection is None:
            logger.error(
                f"No collection mapping found for memory store type: {memory_store_type}"
            )
            return False

        # Update using MongoDB _id only (string or ObjectId form)
        try:
            result = collection.update_one(_mongo_id_predicate(id), {"$set": data})
            success = result.matched_count > 0
            if not success:
                logger.warning(
                    f"Update operation found no documents to modify for id: {id}"
                )
            return success
        except Exception as e:
            logger.error(f"Error updating document with id {id}: {e}", exc_info=True)
            return False

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id: str,
        include_embedding: bool = False,
        memory_type: Union[str, "MemoryType"] = None,
        limit: int = None,
        user_id: Any = _MONGO_UNSET,
        exclude_trace_bundles: bool = False,
        thread_id: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the conversation history ordered by timestamp.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to retrieve the conversation history for.
        include_embedding : bool
            Whether to include the embedding field in the results. Default is False for performance.
        memory_type : Union[str, MemoryType], optional
            Type of memory (defaults to CONVERSATION_MEMORY)
        limit : int, optional
            Maximum number of entries to return
        user_id : str, optional
            Multi-tenant scope. When provided, results are restricted to rows
            whose ``user_id`` matches. When the sentinel default is used, no
            ``user_id`` filter is applied (legacy callers).
        exclude_trace_bundles : bool, optional
            When ``True``, drop the internal streamed-trace-bundle rows
            (``role="tool"`` rows holding a ``{"type": "trace_bundle"}`` blob,
            written by ``MemAgent.run_stream`` for Playground replay) from the
            result. Defaults to ``False`` so existing callers — including the
            Playground, which renders those rows — are unaffected. Pass ``True``
            when building a user-facing history view so the telemetry rows don't
            surface as empty, nameless tool entries. See
            :func:`memorizz.strip_trace_bundles`.

        Returns:
        --------
        List[Dict[str, Any]]
            The conversation history ordered by timestamp.
        """
        projection = {} if include_embedding else {"embedding": 0}

        base_filter: Dict[str, Any] = {"memory_id": memory_id}
        if user_id is not _MONGO_UNSET:
            base_filter.update(_mongo_user_id_predicate(user_id))
        if thread_id:
            # Thread-scoped read — hits the (memory_id, thread_id, timestamp)
            # compound index so a single thread's history is fetched directly
            # instead of loading the whole memory_id and filtering in Python.
            base_filter["thread_id"] = thread_id

        if limit is not None:
            # Retrieve newest rows first, then reverse so callers still receive
            # chronological order (oldest -> newest) for prompt construction.
            query = (
                self.conversation_memory_collection.find(base_filter, projection)
                .sort("timestamp", -1)
                .limit(limit)
            )
            results = list(query)
            results.reverse()
        else:
            query = self.conversation_memory_collection.find(
                base_filter, projection
            ).sort("timestamp", 1)
            results = list(query)

        # Backward compat: migrate old conversation_id → thread_id on read
        for doc in results:
            self._normalize_legacy_fields(doc)

        if exclude_trace_bundles:
            from ...conversation_history import strip_trace_bundles

            results = strip_trace_bundles(results)

        logger.debug(
            f"Retrieved {len(results)} conversation items for memory_id: {memory_id}"
        )
        return results

    def _prepare_memagent_payload(
        self, memagent: Union["MemAgentModel", Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Normalize memagent payloads for storage and updates."""
        if isinstance(memagent, dict):
            memagent_dict = dict(memagent)
            agent_id = memagent_dict.get("agent_id")
            persona = memagent_dict.get("persona")
        else:
            memagent_dict = memagent.model_dump()
            agent_id = getattr(memagent, "agent_id", None)
            persona = getattr(memagent, "persona", None) or memagent_dict.get("persona")

        # Persist the custom agent_id: it used to be stripped here, which
        # made agents saved under string IDs (the SDK default is a uuid4)
        # unretrievable — retrieve_memagent only resolved ObjectId _ids.
        memagent_dict.pop("agent_id", None)
        if agent_id is not None:
            memagent_dict["agent_id"] = str(agent_id)

        if persona:
            if hasattr(persona, "to_dict"):
                memagent_dict["persona"] = persona.to_dict()
            elif isinstance(persona, dict):
                memagent_dict["persona"] = dict(persona)
            else:
                memagent_dict["persona"] = {"name": str(persona)}

        tools = memagent_dict.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if (
                    isinstance(tool, dict)
                    and "function" in tool
                    and callable(tool["function"])
                ):
                    tool.pop("function")

        return memagent_dict, agent_id

    def _memagent_id_query(self, agent_id: Any) -> Dict[str, Any]:
        """Query matching an agent by Mongo ``_id`` OR stored ``agent_id``."""
        clauses: List[Dict[str, Any]] = [{"agent_id": str(agent_id)}]
        if ObjectId.is_valid(str(agent_id)):
            clauses.append({"_id": ObjectId(str(agent_id))})
        return {"$or": clauses}

    def store_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Store a memagent in the MongoDB database using only _id field.

        Parameters:
        -----------
        memagent : MemAgentModel
            The memagent to be stored.

        Returns:
        --------
        MemAgentModel
            The stored memagent.
        """
        memagent_dict, agent_id = self._prepare_memagent_payload(memagent)

        # Upsert by custom agent_id when present so repeated save() calls
        # update the agent instead of inserting duplicates (matches the
        # filesystem provider's semantics).
        if agent_id is not None:
            # One atomic replace-or-insert. The previous find-then-insert let
            # processes saving the same agent concurrently each insert a copy.
            payload = {
                key: value for key, value in memagent_dict.items() if key != "_id"
            }
            payload["agent_id"] = str(agent_id)
            for attempt in range(2):
                try:
                    stored = self.memagent_collection.find_one_and_replace(
                        {"agent_id": str(agent_id)},
                        payload,
                        projection={"_id": 1},
                        upsert=True,
                        return_document=ReturnDocument.AFTER,
                    )
                    break
                except DuplicateKeyError:
                    # A concurrent save inserted it first; retrying replaces it.
                    if attempt:
                        raise
            memagent_dict["_id"] = stored["_id"]
            self._sync_agent_tools_to_toolbox(str(agent_id), memagent_dict.get("tools"))
            return memagent_dict

        result = self.memagent_collection.insert_one(memagent_dict)
        memagent_dict["_id"] = result.inserted_id

        self._sync_agent_tools_to_toolbox(
            str(agent_id) if agent_id is not None else str(result.inserted_id),
            memagent_dict.get("tools"),
        )

        return memagent_dict

    def update_memagent(self, memagent: "MemAgentModel") -> "MemAgentModel":
        """
        Update a memagent in the MongoDB database using _id field.
        """
        memagent_dict, agent_id = self._prepare_memagent_payload(memagent)
        doc_id = memagent_dict.pop("_id", None)
        if agent_id is None:
            agent_id = doc_id

        if agent_id is not None:
            self.memagent_collection.update_one(
                self._memagent_id_query(agent_id), {"$set": memagent_dict}
            )

        if agent_id is not None:
            self._sync_agent_tools_to_toolbox(str(agent_id), memagent_dict.get("tools"))

        return memagent_dict

    def _clear_agent_toolbox_rows(self, agent_id: str) -> None:
        self.toolbox_collection.delete_many({"agent_id": agent_id})

    def retrieve_memagent(self, agent_id: str) -> "MemAgentModel":
        """
        Retrieve a memagent from the MongoDB database using _id field.

        Parameters:
        -----------
        agent_id : str
            The agent ID to retrieve (MongoDB _id).

        Returns:
        --------
        MemAgentModel
            The retrieved memagent.
        """
        try:
            document = self.memagent_collection.find_one(
                self._memagent_id_query(agent_id), {"embedding": 0}
            )
        except Exception:
            return None

        if not document:
            return None

        return MemAgentModel.from_document(document)

    def list_memagents(self) -> List["MemAgentModel"]:
        """
        List all memagents in the MongoDB database.

        Returns:
        --------
        List[MemAgentModel]
            The list of memagents.
        """
        documents = self.memagent_collection.find({}, {"embedding": 0})
        return [MemAgentModel.from_document(doc) for doc in documents]

    def supports_entity_memory(self) -> bool:
        """MongoDB provider supports entity memory operations."""
        return True

    def migrate_entity_memory_user_scope(self, *, memory_id: str, user_id: str) -> int:
        """Atomically adopt anonymous entity rows in one MongoDB scope."""
        result = self.entity_memory_collection.update_many(
            {
                "memory_id": str(memory_id),
                **_mongo_user_id_predicate(None),
            },
            {"$set": {"user_id": str(user_id)}},
        )
        return int(result.modified_count)

    def update_memagent_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """
        Update the memory_ids of a memagent in the memory provider using _id field.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update (MongoDB _id).
        memory_ids : List[str]
            The list of memory_ids to update.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        try:
            result = self.memagent_collection.update_one(
                self._memagent_id_query(agent_id),
                {"$set": {"memory_ids": memory_ids}},
            )
            return result.matched_count > 0
        except Exception:
            return False

    def delete_memagent_memory_ids(self, agent_id: str) -> bool:
        """
        Delete the memory_ids of a memagent in the memory provider.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to update (MongoDB _id).

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        try:
            result = self.memagent_collection.update_one(
                self._memagent_id_query(agent_id),
                {"$unset": {"memory_ids": []}},
            )
            return result.matched_count > 0
        except Exception:
            return False

    def delete_memagent(self, agent_id: str, cascade: bool = False) -> bool:
        """
        Delete a memagent from the memory provider by id.

        Parameters:
        -----------
        agent_id : str
            The id of the memagent to delete.
        cascade : bool
            Whether to cascade the deletion of the memagent. This deletes all the memory units associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if cascade:
            # Retrieve the memagent
            memagent = self.retrieve_memagent(agent_id)

            if memagent is None:
                raise ValueError(f"MemAgent with id {agent_id} not found")

            # Delete every memory unit tagged with one of the agent's
            # memory_ids, in every store except the agent store itself (the
            # agent row is deleted below). Same sweep as the filesystem
            # provider: agent-scoped definitions without a memory_id
            # (personas, toolbox entries) are left alone.
            for memory_id in memagent.memory_ids or []:
                for memory_type in MemoryType:
                    if memory_type == MemoryType.MEMAGENT:
                        continue
                    self._delete_memory_units_by_memory_id(memory_id, memory_type)

        try:
            result = self.memagent_collection.delete_one(
                self._memagent_id_query(agent_id)
            )
            return result.deleted_count > 0
        except Exception:
            return False

    def _delete_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ):
        """
        Delete all the memory units associated with the memory_id.

        Resolves the collection through ``_collection`` so every memory type
        (including ones added later) is swept; the previous if/elif chain
        silently skipped entity memory, summaries, semantic cache, shared
        memory and the skillbox.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to delete.
        memory_type : MemoryType
            The type of memory to delete.
        """
        collection = self._collection(memory_type)
        if collection is None:
            return
        collection.delete_many({"memory_id": memory_id})

    def _setup_vector_search_index(
        self,
        collection,
        index_name="vector_index",
        memory_store: bool = False,
        filter_fields: Optional[List[str]] = None,
    ):
        """
        Setup a vector search index for a MongoDB collection and wait for it to become queryable.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """

        # Define the index definition
        vector_index_definition = {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    # Dynamic dimensions based on the configured embedding provider
                    "numDimensions": self._get_embedding_dimensions_safe(),
                    "similarity": "cosine",
                }
            ]
        }

        # If the memory store is true, then we add the memory_id field to the index
        # This is used to prefilter the memory units by memory_id
        # useful to narrow the scope of your semantic search and ensure that not all vectors are considered for comparison.
        # It reduces the number of documents against which to run similarity comparisons, which can decrease query latency and increase the accuracy of search results.
        if memory_store:
            vector_index_definition["fields"].append(
                {
                    "type": "filter",
                    "path": "memory_id",
                }
            )
        existing_filter_paths = {
            field["path"]
            for field in vector_index_definition["fields"]
            if field.get("type") == "filter"
        }
        for path in filter_fields or []:
            if path in existing_filter_paths:
                continue
            vector_index_definition["fields"].append({"type": "filter", "path": path})
            existing_filter_paths.add(path)

        new_vector_search_index_model = SearchIndexModel(
            definition=vector_index_definition, name=index_name, type="vectorSearch"
        )

        # Create the new index
        try:
            result = collection.create_search_index(model=new_vector_search_index_model)

            # Wait for the index to become queryable using polling mechanism
            self._wait_for_index_ready(collection, result, index_name)

            return result

        except Exception as exc:
            # Mark the collection as unavailable for vector search and warn
            # once. Quota / unsupported errors are very common on small
            # Atlas tiers and shouldn't blow up the agent boot path.
            if self._is_quota_or_unsupported(exc):
                self._handle_index_unavailable(collection.name, index_name, exc)
            else:
                logger.warning(
                    "Failed to create vector index '%s' on %s (%s)",
                    index_name,
                    collection.name,
                    type(exc).__name__,
                )
                self._vector_indexes_unavailable.add(collection.name)
            return None

    def _wait_for_index_ready(
        self, collection, index_name_result, display_name="vector_index"
    ):
        """
        Wait for a MongoDB Atlas search index to become queryable using polling.

        Args:
        collection: MongoDB collection object
        index_name_result: The name/result returned from create_search_index
        display_name: Human-readable name for logging (default: "vector_index")
        """

        # Define predicate function to check if index is queryable
        def predicate(index):
            return index.get("queryable") is True

        # Bounded poll — never block a request thread / worker boot forever.
        # If the index isn't queryable within the budget, log and return so the
        # caller degrades to no-op vector search instead of hanging indefinitely.
        max_attempts = 24  # ~120s at 5s intervals
        for _attempt in range(max_attempts):
            try:
                # List search indexes and find the one we just created
                indices = list(collection.list_search_indexes(index_name_result))

                # Check if the index exists and is queryable
                if indices and predicate(indices[0]):
                    return

                # Wait 5 seconds before checking again
                time.sleep(5)

            except Exception:
                # Continue polling even if there's an error
                time.sleep(5)
        logger.warning(
            "%s did not become queryable within the wait budget; continuing "
            "(vector search may be unavailable until provisioning finishes)",
            display_name,
        )

    def _ensure_vector_index(
        self,
        collection,
        index_name="vector_index",
        memory_store: bool = False,
        filter_fields: Optional[List[str]] = None,
    ) -> bool:
        """
        Ensure a vector search index exists for the collection. If it doesn't exist, create it and wait for it to be ready.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """
        search_indexes = list(collection.list_search_indexes())
        existing_index = next(
            (
                index
                for index in search_indexes
                if index.get("name") == index_name
                and index.get("type") == "vectorSearch"
            ),
            None,
        )

        if existing_index is None:
            created = self._setup_vector_search_index(
                collection,
                index_name,
                memory_store,
                filter_fields=filter_fields,
            )
            if created is None:
                return False
            cache = getattr(self, "_vector_search_status_cache", None)
            if isinstance(cache, dict):
                cache.pop(collection.name, None)
            return True

        required = set(filter_fields or [])
        if memory_store:
            required.add("memory_id")
        current_definition = (
            existing_index.get("latestDefinition")
            or existing_index.get("definition")
            or {}
        )
        fields = current_definition.get("fields", [])
        present = {
            field.get("path") for field in fields if field.get("type") == "filter"
        }
        missing = required - present
        if not missing:
            return True

        definition = dict(current_definition)
        updated_fields = list(definition.get("fields") or [])
        updated_fields.extend(
            {"type": "filter", "path": path} for path in sorted(missing)
        )
        definition["fields"] = updated_fields
        try:
            collection.update_search_index(index_name, definition)
            logger.info(
                "Updated vector index %s on %s with filters: %s",
                index_name,
                collection.name,
                ", ".join(sorted(missing)),
            )
            cache = getattr(self, "_vector_search_status_cache", None)
            if isinstance(cache, dict):
                cache.pop(collection.name, None)
            return True
        except Exception as exc:
            logger.warning(
                "Vector index %s on %s lacks required filters %s and could "
                "not be reconciled automatically (%s)",
                index_name,
                collection.name,
                ", ".join(sorted(missing)),
                type(exc).__name__,
            )
            return False

    def _ensure_semantic_cache_vector_index(self) -> None:
        """
        Ensure vector index exists for semantic cache collection with correct field name.
        """
        collection = self.semantic_cache_collection
        index_name = "vector_index"

        # Check if vector index already exists and has correct definition
        search_indexes = list(collection.list_search_indexes())
        existing_index = None
        for index in search_indexes:
            if index.get("name") == index_name and index.get("type") == "vectorSearch":
                existing_index = index
                break

        # Check if index exists and has all required filter fields
        has_correct_index = False
        if existing_index:
            fields = existing_index.get("definition", {}).get("fields", [])
            filter_paths = {
                field.get("path") for field in fields if field.get("type") == "filter"
            }
            required_filters = {"agent_id", "memory_id", "session_id"}
            has_correct_index = required_filters.issubset(filter_paths)

        # If index exists but has wrong definition, log warning but don't recreate
        if existing_index and not has_correct_index:
            logger.warning(
                f"Vector index '{index_name}' exists but has incomplete filter definition. "
                f"Expected filters: agent_id, memory_id, session_id. "
                f"To fix this, manually drop the index in MongoDB Atlas and restart the application."
            )

        has_vector_index = (
            existing_index is not None
        )  # Use existing index even if definition is incomplete

        if not has_vector_index:
            logger.info(
                "Creating semantic cache vector index with filters: agent_id, memory_id, session_id"
            )

            try:
                # Get embedding dimensions
                dimensions = self._get_embedding_dimensions_safe()
                logger.info(f"Using embedding dimensions: {dimensions}")

                # Create vector index definition for embedding field
                vector_index_definition = {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",  # Standard embedding field name
                            "numDimensions": dimensions,
                            "similarity": "cosine",
                        },
                        {
                            "type": "filter",
                            "path": "agent_id",  # Filter by agent_id
                        },
                        {
                            "type": "filter",
                            "path": "memory_id",  # Filter by memory_id
                        },
                        {
                            "type": "filter",
                            "path": "session_id",  # Filter by session_id
                        },
                    ]
                }

                new_vector_search_index_model = SearchIndexModel(
                    definition=vector_index_definition,
                    name=index_name,
                    type="vectorSearch",
                )

                logger.info(
                    f"Creating vector search index '{index_name}' for semantic cache..."
                )
                result = collection.create_search_index(
                    model=new_vector_search_index_model
                )

                # Wait for the index to become queryable
                logger.info(f"Waiting for index '{index_name}' to become ready...")
                self._wait_for_index_ready(collection, result, index_name)

                logger.info(
                    f" Vector index '{index_name}' for semantic cache is ready!"
                )
                return result

            except Exception as e:
                # Quota / unsupported errors are an expected failure on small
                # Atlas tiers — soft-warn and disable the cache writer for
                # this session instead of raising up the boot path.
                if self._is_quota_or_unsupported(e):
                    self._handle_index_unavailable(
                        self.semantic_cache_collection.name, index_name, e
                    )
                    return None
                # Any other failure is unexpected and worth surfacing.
                logger.error(
                    "Failed to create semantic cache vector index '%s': %s",
                    index_name,
                    e,
                )
                self._vector_indexes_unavailable.add(
                    self.semantic_cache_collection.name
                )
                return None
        else:
            logger.info(
                f" Vector index '{index_name}' already exists and has correct definition"
            )

    def close(self) -> None:
        """Close the connection to MongoDB."""
        self.client.close()
