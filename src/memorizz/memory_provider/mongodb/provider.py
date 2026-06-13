# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import CollectionInvalid, OperationFailure
from pymongo.operations import SearchIndexModel

from ...embeddings import get_embedding
from ...enums.memory_type import MemoryType
from ...long_term.semantic.persona.persona import Persona
from ...long_term.semantic.persona.role_type import RoleType
from ...memagent import MemAgentModel
from ..base import MemoryProvider

logger = logging.getLogger(__name__)


# Sentinel so "user_id filter not supplied" is distinguishable from
# "user_id is explicitly None" (anonymous/legacy scope).
class _MongoUserIdUnset:
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<user_id unset>"


_MONGO_UNSET = _MongoUserIdUnset()


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


@dataclass
class MongoDBConfig:
    """Configuration for the MongoDB provider."""

    def __init__(
        self,
        uri: str,
        db_name: str = "memorizz",
        lazy_vector_indexes: bool = False,
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
        self.embedding_provider = embedding_provider
        self.embedding_config = embedding_config or {}


class MongoDBProvider(MemoryProvider):
    """MongoDB implementation of the MemoryProvider interface."""

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
        self.client = MongoClient(config.uri)
        self.db = self.client[config.db_name]
        self.persona_collection = self.db[MemoryType.PERSONAS.value]
        self.toolbox_collection = self.db[MemoryType.TOOLBOX.value]
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

        # Process embedding provider configuration
        self._embedding_provider = self._setup_embedding_provider(config)

        # Create all memory stores in MongoDB.
        self._create_memory_stores()

        # Create vector indexes immediately only if not using lazy initialization
        if not config.lazy_vector_indexes:
            try:
                self._create_vector_indexes_for_memory_stores()
            except Exception as e:
                logger.warning(
                    f"Failed to create vector indexes during initialization: {e}"
                )
                logger.info("Vector indexes will be created lazily when needed")
                # Set lazy mode if immediate creation fails
                self.config.lazy_vector_indexes = True

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
        if code in (20, 59):  # IllegalOperation, CommandNotFound
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
            logger.warning(
                "MongoDB vector-search index unavailable (first offender: "
                "collection=%r, index=%s). Vector-search consumers will "
                "degrade to no-op for any collection hitting the same root "
                "cause. Likely cause: Atlas Search quota exceeded on this "
                "cluster tier (M0/M2 cap is 3 search indexes) or Atlas "
                "Search not enabled. Original error: %s",
                collection_name,
                index_name,
                exc,
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
        return collection_name in self._vector_indexes_unavailable

    @staticmethod
    def _build_vector_search_pipeline(
        embedding,
        limit: int,
        *,
        index_name: str = "vector_index",
        search_filter: Dict[str, Any] | None = None,
        num_candidates: int | None = None,
        path: str = "embedding",
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
        return [
            stage,
            {"$project": {"embedding": 0}},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]

    def _run_vector_search(
        self, collection, pipeline: List[Dict[str, Any]], *, label: str
    ) -> List[Dict[str, Any]]:
        """Run a vector-search pipeline, honoring the unavailable-index flag.

        Short-circuits to ``[]`` when the collection's vector index is
        known to be unavailable (Atlas quota / Search not enabled), so
        we don't keep racking up failed aggregations once the index
        creator has already flagged the collection.
        """
        if self._vector_index_unavailable(collection.name):
            return []
        try:
            return list(collection.aggregate(pipeline))
        except Exception as exc:
            if self._is_quota_or_unsupported(exc):
                self._handle_index_unavailable(collection.name, "vector_index", exc)
                return []
            logger.warning("Vector search failed for %s: %s", label, exc)
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

    def _get_embedding_provider(self):
        """
        Get the embedding provider to use, with fallback logic.

        Returns:
        --------
        EmbeddingManager or function
            The embedding provider to use
        """
        if self._embedding_provider is not None:
            # Use explicitly provided embedding provider
            return self._embedding_provider
        else:
            # Fall back to global embedding configuration
            from ...embeddings import get_embedding_manager

            return get_embedding_manager()

    def _get_embedding_dimensions_safe(self) -> int:
        """
        Safely get embedding dimensions with error handling.

        Returns:
        --------
        int
            The embedding dimensions, or None if not available
        """
        try:
            if self._embedding_provider is not None:
                # Use explicit provider
                return self._embedding_provider.get_dimensions()
            else:
                # Use global configuration
                from ...embeddings import get_embedding_dimensions

                return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Failed to get embedding dimensions: {e}")
            raise RuntimeError(
                "Cannot determine embedding dimensions. Please configure embeddings first using:\n"
                "configure_embeddings('openai', {'model': 'text-embedding-3-small', 'dimensions': 512})\n"
                "Or use lazy_vector_indexes=True to defer vector index creation."
            )

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
                self._setup_vector_search_index(
                    collection, "vector_index", memory_store
                )
                self._vector_indexes_created.add(index_key)
                logger.info(f"Created vector index for collection: {collection_name}")
            except Exception as e:
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
                "conv_memory_thread_ts",
                [("memory_id", 1), ("thread_id", 1), ("timestamp", 1)],
            ),
            (
                "conv_memory_user_ts",
                [("user_id", 1), ("timestamp", 1)],
            ),
        ],
        MemoryType.ENTITY_MEMORY: [
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
        ],
    }

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
            MemoryType.ENTITY_MEMORY: self.entity_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection,
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
                )

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
        # Handle new calling style (memory_unit + memory_id)
        if memory_unit is not None:
            # Convert memory_unit to dict
            if hasattr(memory_unit, "model_dump"):
                data = memory_unit.model_dump()
            elif hasattr(memory_unit, "dict"):
                data = memory_unit.dict()
            else:
                data = memory_unit.__dict__

            # Add memory_id if provided
            if memory_id:
                data["memory_id"] = memory_id

            # Determine memory_store_type from memory_unit
            if hasattr(memory_unit, "memory_type"):
                memory_store_type = memory_unit.memory_type
            elif "memory_type" in data:
                memory_store_type = data["memory_type"]
            else:
                memory_store_type = MemoryType.CONVERSATION_MEMORY

        # Validate we have required parameters
        if data is None or memory_store_type is None:
            raise ValueError(
                "Either (data, memory_store_type) or (memory_unit) must be provided"
            )

        # Ensure memory_store_type is MemoryType enum
        if isinstance(memory_store_type, str):
            memory_store_type = MemoryType(memory_store_type)

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
        collection = None
        if memory_store_type == MemoryType.PERSONAS:
            collection = self.persona_collection
        elif memory_store_type == MemoryType.TOOLBOX:
            collection = self.toolbox_collection
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            collection = self.workflow_memory_collection
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            collection = self.short_term_memory_collection
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            collection = self.knowledge_base_collection
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            collection = self.conversation_memory_collection
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            collection = self.shared_memory_collection
        elif memory_store_type == MemoryType.SUMMARIES:
            collection = self.summaries_collection
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            collection = self.semantic_cache_collection
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            collection = self.entity_memory_collection
        elif memory_store_type == MemoryType.TOOL_LOG:
            collection = self.tool_log_collection

        if collection is None:
            raise ValueError(f"Invalid memory store type: {memory_store_type}")

        # Clean data by removing custom ID fields - only use MongoDB _id
        # Note: thread_id is preserved for CONVERSATION_MEMORY as it serves a functional purpose
        data_copy = data.copy()

        # Remove custom ID fields since we only want to use _id
        custom_id_fields = [
            "persona_id",
            "tool_id",
            "workflow_id",
            "short_term_memory_id",
            "agent_id",
        ]

        # Don't remove thread_id for conversation memory
        if memory_store_type != MemoryType.CONVERSATION_MEMORY:
            custom_id_fields.append("thread_id")

        # Don't remove knowledge_base_id for knowledge base entries as it's needed for knowledge linking
        if memory_store_type != MemoryType.KNOWLEDGE_BASE:
            custom_id_fields.append("knowledge_base_id")

        # Don't remove agent_id and memory_id for semantic cache as they're needed for filtering and scoping
        if memory_store_type == MemoryType.SEMANTIC_CACHE:
            # Remove agent_id from the removal list to preserve it (we used this for scoped agents semantic cache)
            custom_id_fields = [
                field for field in custom_id_fields if field != "agent_id"
            ]
            # Don't add memory_id to removal list for semantic cache
        elif memory_store_type == MemoryType.TOOL_LOG:
            # Tool-log rows carry the context the row was produced under:
            # which thread the agent was responding in, which memory bucket
            # it belonged to, and which agent emitted it. Stripping these
            # would break ``list_recent_tool_logs`` / per-thread audit views
            # and is the source of "tool_log rows have no thread" complaints.
            custom_id_fields = [
                field
                for field in custom_id_fields
                if field not in ("agent_id", "thread_id")
            ]
            # ``memory_id`` is not in the base list yet for TOOL_LOG, so
            # nothing to remove there; this branch just keeps it that way.
        elif memory_store_type == MemoryType.TOOLBOX:
            # Preserve agent_id so the playground toolbox-memory pane can
            # scope rows to the right agent, and tool_id so the UI can show
            # a human-readable identifier. Tools are agent-scoped across
            # threads, so memory_id is still stripped.
            custom_id_fields = [
                field
                for field in custom_id_fields
                if field not in ("agent_id", "tool_id")
            ]
            custom_id_fields.append("memory_id")
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            # Don't remove memory_id for conversation memory as it's needed for conversation history retrieval
            pass  # Keep memory_id for conversation memory
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            # Entity memory relies on memory_id for scoping
            pass
        elif memory_store_type == MemoryType.TOOL_LOG:
            # Tool-log rows need ``memory_id`` to be scopable per
            # conversation thread; stripping it would force every audit
            # view to scan the entire collection.
            pass
        else:
            # For all other memory types, remove memory_id as before
            custom_id_fields.append("memory_id")

        for field in custom_id_fields:
            data_copy.pop(field, None)

        # If document has MongoDB _id, update it
        if "_id" in data_copy:
            result = collection.update_one(
                {"_id": data_copy["_id"]}, {"$set": data_copy}, upsert=True
            )
            return str(data_copy["_id"])
        else:
            # For new documents, let MongoDB generate _id automatically
            result = collection.insert_one(data_copy)
            return str(result.inserted_id)

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
                    query, limit=limit, **kwargs
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
                    query, limit=limit, **kwargs
                )
            return []
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            if isinstance(query, dict):
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
                # This is a filter query for loading existing cache entries
                return self.semantic_cache_collection.find(
                    query, {"embedding": 0}
                ).limit(limit)
            else:
                # This is a text query for semantic similarity search
                return self.find_similar_cache_entries(query, limit=limit, **kwargs)
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
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.KNOWLEDGE_BASE: self.knowledge_base_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection,
            MemoryType.SEMANTIC_CACHE: self.semantic_cache_collection,
            MemoryType.ENTITY_MEMORY: self.entity_memory_collection,
            MemoryType.MEMAGENT: self.memagent_collection,
            MemoryType.TOOL_LOG: self.tool_log_collection,
        }

        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            return None

        # Set projection to exclude embedding for performance
        projection = (
            {"embedding": 0}
            if memory_store_type
            in [
                MemoryType.PERSONAS,
                MemoryType.TOOLBOX,
                MemoryType.WORKFLOW_MEMORY,
                MemoryType.SUMMARIES,
            ]
            else None
        )

        # For semantic cache, exclude embedding by default for performance
        if memory_store_type == MemoryType.SEMANTIC_CACHE:
            projection = {"embedding": 0}

        # Retrieve using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                doc = collection.find_one({"_id": ObjectId(id)}, projection)
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

        if memory_store_type == MemoryType.TOOLBOX:
            return self.toolbox_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.PERSONAS:
            return self.persona_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return self.workflow_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return self.short_term_memory_collection.find_one(
                {"name": name}, projection
            )
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            return self.knowledge_base_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return self.conversation_memory_collection.find_one(
                {"name": name}, projection
            )
        elif memory_store_type == MemoryType.SUMMARIES:
            return self.summaries_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            return self.entity_memory_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            return self.shared_memory_collection.find_one(
                {"memory_id": name}, projection
            )
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            return self.semantic_cache_collection.find_one(
                {"$or": [{"cache_key": name}, {"query_text": name}]}, projection
            )
        elif memory_store_type == MemoryType.MEMAGENT:
            return self.memagent_collection.find_one({"name": name}, projection)
        elif memory_store_type == MemoryType.TOOL_LOG:
            return self.tool_log_collection.find_one({"name": name}, projection)

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

        try:
            embedding = get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to generate embedding for entity query: {e}")
            return []

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
            self.entity_memory_collection, pipeline, label="entity_memory"
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

    def get_summaries_by_memory_id(
        self, memory_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id, ordered by timestamp (most recent first).

        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        limit : int
            The maximum number of summaries to return.

        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries for the memory_id.
        """
        return list(
            self.summaries_collection.find({"memory_id": memory_id}, {"embedding": 0})
            .sort("period_end", -1)
            .limit(limit)
        )

    def get_summaries_by_time_range(
        self, memory_id: str, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id within a time range based on the period they cover.

        NOTE: This filters by the time period that the summary covers (period_start/period_end),
        not when the summary was created. Use get_summaries_by_creation_time() to filter by creation time.

        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        start_time : float
            Start timestamp for the memory period range.
        end_time : float
            End timestamp for the memory period range.

        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries whose covered period falls within the time range.
        """
        from datetime import datetime

        # Convert to ISO string for compatibility with existing string timestamps
        start_iso = datetime.fromtimestamp(start_time).isoformat()
        end_iso = datetime.fromtimestamp(end_time).isoformat()

        # Query supports both float and string timestamps
        return list(
            self.summaries_collection.find(
                {
                    "memory_id": memory_id,
                    "$or": [
                        # Float timestamps (new format)
                        {
                            "period_start": {"$gte": start_time, "$type": "number"},
                            "period_end": {"$lte": end_time, "$type": "number"},
                        },
                        # String timestamps (legacy format)
                        {
                            "period_start": {"$gte": start_iso, "$type": "string"},
                            "period_end": {"$lte": end_iso, "$type": "string"},
                        },
                    ],
                },
                {"embedding": 0},
            ).sort("period_start", 1)
        )

    def get_summaries_by_creation_time(
        self, memory_id: str, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        """
        Retrieve summaries for a specific memory_id created within a time range.

        This filters by when the summary was actually created (created_at timestamp),
        not the time period that the summary covers.

        Parameters:
        -----------
        memory_id : str
            The memory_id to retrieve summaries for.
        start_time : float
            Start timestamp for when summaries were created.
        end_time : float
            End timestamp for when summaries were created.

        Returns:
        --------
        List[Dict[str, Any]]
            List of summaries created within the time range.
        """
        return list(
            self.summaries_collection.find(
                {
                    "memory_id": memory_id,
                    "created_at": {"$gte": start_time, "$lte": end_time},
                },
                {"embedding": 0},
            ).sort("created_at", -1)
        )

    # ===== SEMANTIC CACHE METHODS =====

    def store_semantic_cache_entry(self, cache_entry: Dict[str, Any]) -> str:
        """
        Store a semantic cache entry in the semantic_cache collection.

        Short-circuits when the semantic cache's vector index is unavailable
        (Atlas Search quota exceeded / Search not enabled) — there is no
        point persisting entries that can never be retrieved, and silent
        writes would mask a misconfigured deployment. Returns ``""`` in
        that case so callers can detect the no-op.

        Parameters:
        -----------
        cache_entry : Dict[str, Any]
            The cache entry containing query, response, embedding, and metadata.

        Returns:
        --------
        str
            The ID of the stored cache entry, or ``""`` when the cache is
            disabled due to a missing vector index.
        """
        if self._vector_index_unavailable(self.semantic_cache_collection.name):
            return ""
        return self.store(cache_entry, MemoryType.SEMANTIC_CACHE)

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

        pipeline = self._build_vector_search_pipeline(
            embedding, limit, search_filter=search_filter or None
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

        return self._knowledge_base_vector_search(embedding, limit=limit, **kwargs)

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
            embedding, limit, search_filter=search_filter or None
        )
        return self._run_vector_search(
            self.knowledge_base_collection, pipeline, label="knowledge_base"
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
        )
        return self._run_vector_search(
            self.semantic_cache_collection, pipeline, label="semantic_cache"
        )

    def update_cache_entry_usage(
        self, cache_id: str, usage_count: int, last_accessed: float
    ) -> bool:
        """
        Update usage statistics for a cache entry.

        Parameters:
        -----------
        cache_id : str
            The MongoDB _id of the cache entry
        usage_count : int
            New usage count
        last_accessed : float
            New last accessed timestamp

        Returns:
        --------
        bool
            True if update was successful
        """
        try:
            result = self.semantic_cache_collection.update_one(
                {"_id": ObjectId(cache_id)},
                {"$set": {"usage_count": usage_count, "last_accessed": last_accessed}},
            )
            return result.modified_count > 0
        except Exception as e:
            logger.warning(f"Failed to update cache entry usage: {e}")
            return False

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
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.KNOWLEDGE_BASE: self.knowledge_base_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection,
            MemoryType.SEMANTIC_CACHE: self.semantic_cache_collection,
            MemoryType.ENTITY_MEMORY: self.entity_memory_collection,
            MemoryType.MEMAGENT: self.memagent_collection,
            MemoryType.TOOL_LOG: self.tool_log_collection,
        }

        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            return False

        # Delete using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                result = collection.delete_one({"_id": ObjectId(id)})
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
        if memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            result = self.knowledge_base_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SUMMARIES:
            result = self.summaries_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            result = self.entity_memory_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            result = self.shared_memory_collection.delete_one({"memory_id": name})
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            result = self.semantic_cache_collection.delete_one(
                {"$or": [{"cache_key": name}, {"query_text": name}]}
            )
        elif memory_store_type == MemoryType.MEMAGENT:
            result = self.memagent_collection.delete_one({"name": name})
        elif memory_store_type == MemoryType.TOOL_LOG:
            result = self.tool_log_collection.delete_one({"name": name})
        else:
            return False

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
        if memory_store_type == MemoryType.PERSONAS:
            result = self.persona_collection.delete_many({})
        elif memory_store_type == MemoryType.TOOLBOX:
            result = self.toolbox_collection.delete_many({})
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            result = self.short_term_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            result = self.knowledge_base_collection.delete_many({})
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            result = self.conversation_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            result = self.workflow_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.SUMMARIES:
            result = self.summaries_collection.delete_many({})
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            result = self.entity_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            result = self.shared_memory_collection.delete_many({})
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            result = self.semantic_cache_collection.delete_many({})
        elif memory_store_type == MemoryType.MEMAGENT:
            result = self.memagent_collection.delete_many({})
        elif memory_store_type == MemoryType.TOOL_LOG:
            result = self.tool_log_collection.delete_many({})
        else:
            return False

        return result.deleted_count > 0

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

        if memory_store_type == MemoryType.PERSONAS:
            return list(self.persona_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.TOOLBOX:
            return list(self.toolbox_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.SHORT_TERM_MEMORY:
            return list(
                self.short_term_memory_collection.find(mongo_filter, projection)
            )
        elif memory_store_type == MemoryType.KNOWLEDGE_BASE:
            return list(self.knowledge_base_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.CONVERSATION_MEMORY:
            return list(
                self.conversation_memory_collection.find(mongo_filter, projection)
            )
        elif memory_store_type == MemoryType.WORKFLOW_MEMORY:
            return list(self.workflow_memory_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.SHARED_MEMORY:
            return list(self.shared_memory_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.SUMMARIES:
            return list(self.summaries_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.ENTITY_MEMORY:
            return list(self.entity_memory_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.SEMANTIC_CACHE:
            return list(self.semantic_cache_collection.find(mongo_filter, projection))
        elif memory_store_type == MemoryType.MEMAGENT:
            return list(self.memagent_collection.find({}, projection))
        elif memory_store_type == MemoryType.TOOL_LOG:
            return list(self.tool_log_collection.find(mongo_filter, projection))
        else:
            logger.warning(
                f"Unsupported memory store type for list_all: {memory_store_type}"
            )
            return []

    def list_tool_logs(
        self,
        memory_id: Optional[str] = None,
        user_id: Any = _MONGO_UNSET,
        limit: int = 20,
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
        if memory_store_type == MemoryType.MEMAGENT:
            payload, _ = self._prepare_memagent_payload(data)
            payload.pop("_id", None)
            try:
                if ObjectId.is_valid(id):
                    result = self.memagent_collection.update_one(
                        {"_id": ObjectId(id)}, {"$set": payload}
                    )
                    success = result.modified_count > 0
                    if not success:
                        logger.warning(
                            f"Update operation found no documents to modify for id: {id}"
                        )
                    return success
                logger.error(f"Invalid ObjectId: {id}")
                return False
            except Exception as e:
                logger.error(
                    f"Error updating document with id {id}: {e}", exc_info=True
                )
                return False
        if memory_store_type == MemoryType.SEMANTIC_CACHE and not ObjectId.is_valid(id):
            try:
                result = self.semantic_cache_collection.update_one(
                    {"cache_key": id}, {"$set": data}
                )
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Error updating semantic cache with id {id}: {e}")
                return False

        # Get the appropriate collection
        collection_mapping = {
            MemoryType.PERSONAS: self.persona_collection,
            MemoryType.TOOLBOX: self.toolbox_collection,
            MemoryType.WORKFLOW_MEMORY: self.workflow_memory_collection,
            MemoryType.SHORT_TERM_MEMORY: self.short_term_memory_collection,
            MemoryType.KNOWLEDGE_BASE: self.knowledge_base_collection,
            MemoryType.CONVERSATION_MEMORY: self.conversation_memory_collection,
            MemoryType.SHARED_MEMORY: self.shared_memory_collection,
            MemoryType.SUMMARIES: self.summaries_collection,
            MemoryType.SEMANTIC_CACHE: self.semantic_cache_collection,
            MemoryType.ENTITY_MEMORY: self.entity_memory_collection,
            MemoryType.MEMAGENT: self.memagent_collection,
            MemoryType.TOOL_LOG: self.tool_log_collection,
        }

        collection = collection_mapping.get(memory_store_type)
        if collection is None:
            logger.error(
                f"No collection mapping found for memory store type: {memory_store_type}"
            )
            return False

        # Update using MongoDB _id only
        try:
            if ObjectId.is_valid(id):
                result = collection.update_one({"_id": ObjectId(id)}, {"$set": data})
                success = result.modified_count > 0
                if not success:
                    logger.warning(
                        f"Update operation found no documents to modify for id: {id}"
                    )
                return success
            else:
                logger.error(f"Invalid ObjectId: {id}")
                return False
        except Exception as e:
            logger.error(f"Error updating document with id {id}: {e}", exc_info=True)
            return False

    def update_toolbox_item(self, id: str, data: Dict[str, Any]) -> bool:
        """
        Update a toolbox item in MongoDB by id using optimized queries.
        """

        # Update the embedding if the name, docstring or signature has changed

        # Get the old data
        old_data = self.retrieve_by_id(id, MemoryType.TOOLBOX)
        if not old_data:
            return False

        # Concatenate the name, docstring and signature if any of them have changed
        if old_data.get("name") != data.get("name"):
            data["name"] = data.get("name", old_data.get("name", ""))
        if old_data.get("docstring") != data.get("docstring"):
            data["docstring"] = data.get("docstring", old_data.get("docstring", ""))
        if old_data.get("signature") != data.get("signature"):
            data["signature"] = data.get("signature", old_data.get("signature", ""))

        # Update the embedding
        data["embedding"] = get_embedding(
            data["name"] + " " + data["docstring"] + " " + data["signature"]
        )

        # Use the optimized update_by_id method
        return self.update_by_id(id, data, MemoryType.TOOLBOX)

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

    def delete_conversation_thread(
        self,
        memory_id: str,
        thread_id: str,
        user_id: Any = _MONGO_UNSET,
    ) -> int:
        """Delete every conversation row for one thread in a single indexed
        ``delete_many`` — replaces the consumer's load-all-then-per-row delete
        (an N+1). Returns the number of rows removed.

        Parameters
        ----------
        memory_id : str
            The memory the thread belongs to.
        thread_id : str
            The thread to delete.
        user_id : optional
            Multi-tenant scope; when provided, restricts the delete to rows
            whose ``user_id`` matches (defence against cross-tenant deletes).
        """
        if not memory_id or not thread_id:
            return 0
        flt: Dict[str, Any] = {"memory_id": memory_id, "thread_id": thread_id}
        if user_id is not _MONGO_UNSET:
            flt.update(_mongo_user_id_predicate(user_id))
        try:
            result = self.conversation_memory_collection.delete_many(flt)
            deleted = int(getattr(result, "deleted_count", 0) or 0)
            logger.debug(
                "Deleted %d conversation rows for thread %s (memory_id=%s)",
                deleted,
                thread_id,
                memory_id,
            )
            return deleted
        except Exception:
            logger.exception(
                "delete_conversation_thread failed (memory_id=%s thread_id=%s)",
                memory_id,
                thread_id,
            )
            return 0

    def retrieve_memory_units_by_query(
        self,
        query: str = None,
        query_embedding: list[float] = None,
        memory_id: str = None,
        memory_type: MemoryType = None,
        limit: int = 5,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memory units by query.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        memory_type : MemoryType
            The type of memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.
        user_id : str, optional
            Multi-tenant scope; forwarded to downstream vector-search helpers.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Detect the memory type
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            return self.get_conversation_memory_units(
                query, query_embedding, memory_id, limit, user_id=user_id
            )
        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            return self.get_workflow_memory_units(
                query, query_embedding, memory_id, limit, user_id=user_id
            )
        elif memory_type == MemoryType.SUMMARIES:
            return self.get_summaries_memory_units(
                query, query_embedding, memory_id, limit, user_id=user_id
            )
        else:
            # Return empty list for unsupported memory types
            return []

    def get_conversation_memory_units(
        self,
        query: str = None,
        query_embedding: list[float] = None,
        memory_id: str = None,
        limit: int = 5,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Get the conversation memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.
        user_id : str, optional
            Multi-tenant scope for tenant-isolated vector search.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Ensure vector index exists for conversation memory (lazy creation)
        if self.config.lazy_vector_indexes:
            self._ensure_vector_index_for_collection(
                self.conversation_memory_collection,
                "conversation_memory",
                memory_store=True,
            )

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            try:
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        vs_filter: Dict[str, Any] = {"memory_id": memory_id}
        if user_id is not _MONGO_UNSET:
            vs_filter.update(_mongo_user_id_predicate(user_id))

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": vs_filter,
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        # Execute the pipeline
        results = list(self.conversation_memory_collection.aggregate(pipeline))

        # Return the results
        return results

    def get_summaries_memory_units(
        self,
        query: str = None,
        query_embedding: list[float] = None,
        memory_id: str = None,
        limit: int = 5,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Get the summaries memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.
        user_id : str, optional
            Multi-tenant scope for tenant-isolated vector search.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            try:
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        vs_filter: Dict[str, Any] = {"memory_id": memory_id}
        if user_id is not _MONGO_UNSET:
            vs_filter.update(_mongo_user_id_predicate(user_id))

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": vs_filter,
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "created_at": -1}},
        ]

        # Execute the pipeline
        results = list(self.summaries_collection.aggregate(pipeline))

        # Return the results
        return results

    def get_workflow_memory_units(
        self,
        query: str = None,
        query_embedding: list[float] = None,
        memory_id: str = None,
        limit: int = 5,
        user_id: Any = _MONGO_UNSET,
    ) -> List[Dict[str, Any]]:
        """
        Get the workflow memory units.

        Parameters:
        -----------
        query : str
            The query to use for retrieval.
        query_embedding : list[float]
            The embedding of the query.
        memory_id : str
            The id of the memory to retrieve the memory units for.
        limit : int
            The maximum number of memory units to return.
        user_id : str, optional
            Multi-tenant scope for tenant-isolated vector search.

        Returns:
        --------
        List[Dict[str, Any]]
            The memory units ordered by timestamp.
        """

        # Ensure vector index exists for workflow memory (lazy creation)
        if self.config.lazy_vector_indexes:
            self._ensure_vector_index_for_collection(
                self.workflow_memory_collection, "workflow_memory", memory_store=True
            )

        # If the query embedding is not provided, then we create it
        if query_embedding is None and query is not None:
            try:
                query_embedding = get_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
                return []

        vs_filter: Dict[str, Any] = {"memory_id": memory_id}
        if user_id is not _MONGO_UNSET:
            vs_filter.update(_mongo_user_id_predicate(user_id))

        vector_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": limit,
                "filter": vs_filter,
            }
        }

        # Add the vector stage to the pipeline
        pipeline = [
            vector_stage,
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0}},
            {"$sort": {"score": -1, "timestamp": 1}},
        ]

        # Execute the pipeline
        results = list(self.workflow_memory_collection.aggregate(pipeline))

        # Return the results
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

        memagent_dict.pop("agent_id", None)

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
        memagent_dict, _ = self._prepare_memagent_payload(memagent)

        # Insert the document and let MongoDB generate _id automatically
        result = self.memagent_collection.insert_one(memagent_dict)

        # Add the generated _id to the response
        memagent_dict["_id"] = result.inserted_id

        self._sync_agent_tools_to_toolbox(
            str(result.inserted_id), memagent_dict.get("tools")
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

        # Update the memagent in the MongoDB database using _id
        if isinstance(agent_id, ObjectId):
            self.memagent_collection.update_one(
                {"_id": agent_id}, {"$set": memagent_dict}
            )
        elif agent_id and ObjectId.is_valid(str(agent_id)):
            self.memagent_collection.update_one(
                {"_id": ObjectId(str(agent_id))}, {"$set": memagent_dict}
            )

        if agent_id is not None:
            self._sync_agent_tools_to_toolbox(str(agent_id), memagent_dict.get("tools"))

        return memagent_dict

    def _sync_agent_tools_to_toolbox(
        self, agent_id: str, tools: Optional[List[Dict[str, Any]]]
    ) -> None:
        """Mirror an agent's tool list into the TOOLBOX collection.

        Deletes any existing TOOLBOX rows for this ``agent_id`` before
        re-inserting the current set so tools removed from the agent
        don't linger in the playground's toolbox-memory pane.
        ``tools=None`` is treated as "caller didn't include tools in this
        save" and is a no-op — only an explicit empty list clears rows.
        """
        if not agent_id or tools is None:
            return

        try:
            self.toolbox_collection.delete_many({"agent_id": agent_id})
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
        # Get the document from MongoDB using _id
        try:
            if ObjectId.is_valid(agent_id):
                document = self.memagent_collection.find_one(
                    {"_id": ObjectId(agent_id)}, {"embedding": 0}
                )
            else:
                return None
        except Exception:
            return None

        if not document:
            return None

        # Create a new MemAgent with data from the document
        # Use the MongoDB _id as agent_id since we no longer store agent_id field
        memagent = MemAgentModel(
            name=document.get("name"),
            instruction=document.get("instruction"),
            application_mode=document.get("application_mode", "assistant"),
            memory_types=document.get("memory_types"),
            max_steps=document.get("max_steps"),
            memory_ids=document.get("memory_ids") or [],
            agent_id=str(document.get("_id")),
            is_favorite=bool(document.get("is_favorite", False)),
            tools=document.get("tools"),
            tool_access=document.get("tool_access"),
            knowledge_base_ids=document.get("knowledge_base_ids"),
            delegates=document.get("delegates"),
            llm_config=document.get("llm_config"),
            embedding_config=document.get("embedding_config"),
            semantic_cache=bool(document.get("semantic_cache", False)),
            semantic_cache_config=document.get("semantic_cache_config"),
            context_window_tokens=document.get("context_window_tokens"),
            sandbox_provider=document.get("sandbox_provider"),
            internet_access_provider=document.get("internet_access_provider"),
            internet_access_config=document.get("internet_access_config"),
            skills_marketplace_provider=document.get("skills_marketplace_provider"),
            skills_marketplace_config=document.get("skills_marketplace_config"),
            skill_paths=document.get("skill_paths"),
            mcp_servers=document.get("mcp_servers"),
            self_aware=bool(document.get("self_aware", False)),
            self_aware_config=document.get("self_aware_config"),
            automations_enabled=bool(document.get("automations_enabled", True)),
            default_timezone=document.get("default_timezone"),
            whatsapp_enabled=bool(document.get("whatsapp_enabled", False)),
            whatsapp_config=document.get("whatsapp_config"),
            memory_provider=self,
        )

        # Construct persona if present in the document. Use from_dict so
        # stored goals/background aren't double-merged with role defaults and
        # version/evolution_history/storage_id round-trip correctly.
        if document.get("persona"):
            memagent.persona = Persona.from_dict(document.get("persona"))

        return memagent

    def list_memagents(self) -> List["MemAgentModel"]:
        """
        List all memagents in the MongoDB database.

        Returns:
        --------
        List[MemAgentModel]
            The list of memagents.
        """

        documents = list(self.memagent_collection.find({}, {"embedding": 0}))
        agents = []

        for doc in documents:
            # Use the MongoDB _id as agent_id since we no longer store agent_id field
            agent = MemAgentModel(
                name=doc.get("name"),
                instruction=doc.get("instruction"),
                application_mode=doc.get("application_mode", "assistant"),
                memory_types=doc.get("memory_types"),
                max_steps=doc.get("max_steps"),
                memory_ids=doc.get("memory_ids") or [],
                agent_id=str(doc.get("_id")),
                is_favorite=bool(doc.get("is_favorite", False)),
                tools=doc.get("tools"),  # Include tools from document
                tool_access=doc.get("tool_access"),
                knowledge_base_ids=doc.get("knowledge_base_ids"),
                delegates=doc.get("delegates"),
                llm_config=doc.get("llm_config"),
                embedding_config=doc.get("embedding_config"),
                semantic_cache=bool(doc.get("semantic_cache", False)),
                semantic_cache_config=doc.get("semantic_cache_config"),
                context_window_tokens=doc.get("context_window_tokens"),
                sandbox_provider=doc.get("sandbox_provider"),
                internet_access_provider=doc.get("internet_access_provider"),
                internet_access_config=doc.get("internet_access_config"),
                skills_marketplace_provider=doc.get("skills_marketplace_provider"),
                skills_marketplace_config=doc.get("skills_marketplace_config"),
                skill_paths=doc.get("skill_paths"),
                mcp_servers=doc.get("mcp_servers"),
                self_aware=bool(doc.get("self_aware", False)),
                self_aware_config=doc.get("self_aware_config"),
                automations_enabled=bool(doc.get("automations_enabled", True)),
                default_timezone=doc.get("default_timezone"),
                whatsapp_enabled=bool(doc.get("whatsapp_enabled", False)),
                whatsapp_config=doc.get("whatsapp_config"),
                memory_provider=self,
            )

            # Construct persona if present in the document. Use from_dict so
            # stored goals/background aren't double-merged with role defaults
            # and version/evolution_history/storage_id round-trip correctly.
            if doc.get("persona"):
                agent.persona = Persona.from_dict(doc.get("persona"))

            agents.append(agent)

        return agents

    def supports_entity_memory(self) -> bool:
        """MongoDB provider supports entity memory operations."""
        return True

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
            if ObjectId.is_valid(agent_id):
                result = self.memagent_collection.update_one(
                    {"_id": ObjectId(agent_id)}, {"$set": {"memory_ids": memory_ids}}
                )
                return result.modified_count > 0
            else:
                return False
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
            if ObjectId.is_valid(agent_id):
                result = self.memagent_collection.update_one(
                    {"_id": ObjectId(agent_id)}, {"$unset": {"memory_ids": []}}
                )
                return result.modified_count > 0
            else:
                return False
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

            # Delete all the memory units associated with the memagent by deleting the memory_ids and their corresponding memory store in the memory provider.
            for memory_id in memagent.memory_ids:
                # Loop through all the memory stores and delete records with the memory_ids
                for memory_type in MemoryType:
                    self._delete_memory_units_by_memory_id(memory_id, memory_type)
        else:
            try:
                if ObjectId.is_valid(agent_id):
                    result = self.memagent_collection.delete_one(
                        {"_id": ObjectId(agent_id)}
                    )
                    return result.deleted_count > 0
                else:
                    return False
            except Exception:
                return False

        return True

    def _delete_memory_units_by_memory_id(
        self, memory_id: str, memory_type: MemoryType
    ):
        """
        Delete all the memory units associated with the memory_id.

        Parameters:
        -----------
        memory_id : str
            The id of the memory to delete.
        memory_type : MemoryType
            The type of memory to delete.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        if memory_type == MemoryType.CONVERSATION_MEMORY:
            self.conversation_memory_collection.delete_many({"memory_id": memory_id})

        elif memory_type == MemoryType.WORKFLOW_MEMORY:
            self.workflow_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.SHORT_TERM_MEMORY:
            self.short_term_memory_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.KNOWLEDGE_BASE:
            self.knowledge_base_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.PERSONAS:
            self.persona_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.TOOLBOX:
            self.toolbox_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.MEMAGENT:
            self.memagent_collection.delete_many({"memory_id": memory_id})
        elif memory_type == MemoryType.TOOL_LOG:
            self.tool_log_collection.delete_many({"memory_id": memory_id})

    def _setup_vector_search_index(
        self, collection, index_name="vector_index", memory_store: bool = False
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
                    "Failed to create vector index '%s' on %s: %s",
                    index_name,
                    collection.name,
                    exc,
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
        self, collection, index_name="vector_index", memory_store: bool = False
    ):
        """
        Ensure a vector search index exists for the collection. If it doesn't exist, create it and wait for it to be ready.

        Args:
        collection: MongoDB collection object
        index_name: Name of the index (default: "vector_index")
        memory_store: Whether to add the memory_id field to the index (default: False)
        """
        search_indexes = list(collection.list_search_indexes())
        has_vector_index = any(
            index.get("name") == index_name and index.get("type") == "vectorSearch"
            for index in search_indexes
        )

        if not has_vector_index:
            self._setup_vector_search_index(collection, index_name, memory_store)
        else:
            pass  # Index already exists

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
