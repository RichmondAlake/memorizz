# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except ImportError:
    np = None

from ..embeddings import EmbeddingManager, get_embedding_manager
from ..enums.memory_type import MemoryType
from ..enums.semantic_cache_scope import SemanticCacheScope
from ..memory_provider.base import MemoryProvider
from ..memory_unit.semantic_cache_entry import SemanticCacheEntry

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic cache behavior."""

    similarity_threshold: float = (
        0.78  # Lowered for better matching with similar queries
    )
    max_cache_size: int = 1000
    ttl_hours: float = 24.0
    enable_memory_provider_sync: bool = (
        True  # Enable by default to use memory provider consistently
    )
    enable_usage_tracking: bool = True
    # A conversational cache must not replay a response from another thread by
    # default. Callers can still opt out explicitly for stateless workloads.
    enable_session_scoping: bool = True
    scope: SemanticCacheScope = (
        SemanticCacheScope.LOCAL
    )  # LOCAL filters by agent_id, GLOBAL searches all entries
    embedding_provider: Optional[str] = None
    embedding_config: Optional[Dict[str, Any]] = None
    admission_policy: str = "read_only_deterministic"
    freshness_by_domain: Dict[str, float] = field(
        default_factory=lambda: {"mcp": 300.0, "inventory": 60.0, "calendar": 60.0}
    )
    require_fingerprint_match: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.scope, str):
            normalized_scope = self.scope.strip().lower()
            # ``scope="session"`` was part of the public configuration surface
            # before scope gained its LOCAL/GLOBAL enum.  Preserve that spelling
            # and translate it to the two controls that now express the same
            # policy: local agent isolation plus session isolation.
            if normalized_scope == "session":
                self.scope = SemanticCacheScope.LOCAL
                self.enable_session_scoping = True
            else:
                self.scope = SemanticCacheScope(normalized_scope)
        self.similarity_threshold = max(0.0, min(float(self.similarity_threshold), 1.0))
        self.max_cache_size = max(1, int(self.max_cache_size))
        self.ttl_hours = float(self.ttl_hours)
        self.freshness_by_domain = {
            str(domain): max(0.0, float(seconds))
            for domain, seconds in dict(self.freshness_by_domain or {}).items()
        }


@dataclass(frozen=True)
class SemanticCacheInspection:
    """Structured, response-free evidence for one semantic-cache lookup."""

    lookup_query: str
    hit: bool
    matched_query: Optional[str] = None
    cache_key: Optional[str] = None
    similarity: Optional[float] = None
    ttl_seconds: Optional[float] = None
    expires_in_seconds: Optional[float] = None
    age_seconds: Optional[float] = None
    hit_count: int = 0
    bypass_reason: Optional[str] = None
    invalidation_domains: List[str] = field(default_factory=list)
    invalidation_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookup_query": self.lookup_query,
            "hit": self.hit,
            "matched_query": self.matched_query,
            "cache_key": self.cache_key,
            "similarity": self.similarity,
            "ttl_seconds": self.ttl_seconds,
            "expires_in_seconds": self.expires_in_seconds,
            "age_seconds": self.age_seconds,
            "hit_count": self.hit_count,
            "bypass_reason": self.bypass_reason,
            "invalidation_domains": list(self.invalidation_domains),
            "invalidation_tags": list(self.invalidation_tags),
        }


class SemanticCache:
    """Enhanced semantic cache with vector similarity search and intelligent management."""

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id_var.get()

    @agent_id.setter
    def agent_id(self, value: Optional[str]) -> None:
        self._agent_id_var.set(value)

    @property
    def memory_id(self) -> Optional[str]:
        return self._memory_id_var.get()

    @memory_id.setter
    def memory_id(self, value: Optional[str]) -> None:
        self._memory_id_var.set(value)

    @property
    def last_hit(self) -> Optional[Dict[str, Any]]:
        return self._last_hit_var.get()

    @last_hit.setter
    def last_hit(self, value: Optional[Dict[str, Any]]) -> None:
        self._last_hit_var.set(value)

    @property
    def last_inspection(self) -> Optional[SemanticCacheInspection]:
        return self._last_inspection_var.get()

    @last_inspection.setter
    def last_inspection(self, value: Optional[SemanticCacheInspection]) -> None:
        self._last_inspection_var.set(value)

    def __init__(
        self,
        config: Optional[SemanticCacheConfig] = None,
        memory_provider: Optional[MemoryProvider] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ):
        """
        Initialize semantic cache with advanced features.

        Parameters:
        -----------
        config : Optional[SemanticCacheConfig]
            Configuration for cache behavior
        memory_provider : Optional[MemoryProvider]
            Memory provider for persistence (optional)
        embedding_manager : Optional[EmbeddingManager]
            Custom embedding manager (uses global if None)
        agent_id : Optional[str]
            Agent ID for scoped caching
        memory_id : Optional[str]
            Memory ID for scoped caching (enables memory_id-specific isolation)
        """
        self.config = config or SemanticCacheConfig()
        self.memory_provider = memory_provider
        # One MemAgent (and therefore one SemanticCache) is often shared by a
        # threaded web server. Scope is per execution context, not mutable
        # singleton state.
        self._agent_id_var: ContextVar[Optional[str]] = ContextVar(
            f"memorizz_cache_agent_{id(self)}", default=agent_id
        )
        self._memory_id_var: ContextVar[Optional[str]] = ContextVar(
            f"memorizz_cache_memory_{id(self)}", default=memory_id
        )
        self._last_hit_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
            f"memorizz_cache_last_hit_{id(self)}", default=None
        )
        self._last_inspection_var: ContextVar[
            Optional[SemanticCacheInspection]
        ] = ContextVar(f"memorizz_cache_inspection_{id(self)}", default=None)
        self.agent_id = agent_id
        self.memory_id = memory_id

        # Initialize embedding manager - prioritize passed manager for consistency
        if embedding_manager:
            # Use the provided embedding manager (ensures consistency with agent)
            self.embedding_manager = embedding_manager
            logger.debug("Using provided embedding manager for consistency")
        elif self.config.embedding_provider:
            # Create new embedding manager with specific config
            from ..embeddings import EmbeddingManager

            self.embedding_manager = EmbeddingManager(
                self.config.embedding_provider, self.config.embedding_config
            )
            logger.debug(
                f"Created new embedding manager with provider: {self.config.embedding_provider}"
            )
        else:
            # Use global embedding manager as fallback
            self.embedding_manager = get_embedding_manager()
            logger.debug("Using global embedding manager")

        # In-memory cache
        self.cache: Dict[str, SemanticCacheEntry] = {}
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "bypasses": 0,
            "writes": 0,
            "evictions": 0,
        }
        self._bypass_reasons: Dict[str, int] = {}
        self.last_hit: Optional[Dict[str, Any]] = None
        self.last_inspection: Optional[SemanticCacheInspection] = None

        # Embedding cache to avoid regenerating embeddings for the same query
        self._embedding_cache: Dict[str, List[float]] = {}

        # Load existing entries from memory provider if available
        if self.memory_provider and self.config.enable_memory_provider_sync:
            self._load_from_memory_provider()

        logger.info(
            f"SemanticCache initialized with threshold={self.config.similarity_threshold}, "
            f"agent_id={self.agent_id}, memory_id={self.memory_id}"
        )

    def _get_or_generate_embedding(self, query: str) -> List[float]:
        """
        Get embedding from cache or generate if not present.

        Parameters:
        -----------
        query : str
            The query to get embedding for

        Returns:
        --------
        List[float]
            The query embedding
        """
        if query not in self._embedding_cache:
            self._embedding_cache[query] = self.embedding_manager.get_embedding(query)
        return self._embedding_cache[query]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2) or not vec1:
                return 0.0

            # Numpy is fast for large vectors, but the per-call overhead dominates for
            # tiny vectors (which are common in tests). Prefer a pure-Python path for
            # small dimensions to keep cache lookups predictable.
            if np is not None and len(vec1) > 64:
                vec1_np = np.asarray(vec1, dtype=float)
                vec2_np = np.asarray(vec2, dtype=float)

                norm1 = float(np.linalg.norm(vec1_np))
                norm2 = float(np.linalg.norm(vec2_np))
                if norm1 == 0.0 or norm2 == 0.0:
                    return 0.0

                return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

            import math

            dot_product = 0.0
            norm1 = 0.0
            norm2 = 0.0
            for a, b in zip(vec1, vec2):
                dot_product += float(a) * float(b)
                norm1 += float(a) * float(a)
                norm2 += float(b) * float(b)

            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0

            return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0

    def _should_use_memory_provider(self) -> bool:
        """
        Determine if we should use memory provider for retrieval instead of in-memory matching.
        Only use in-memory when memory provider is None or doesn't support vector search.
        """
        if not self.memory_provider:
            return False

        # Check if this is a known provider with vector search support
        provider_class_name = self.memory_provider.__class__.__name__
        if provider_class_name in (
            "MongoDBProvider",
            "OracleProvider",
            "FileSystemProvider",
        ):
            return True

        # For other providers, check if they have vector search capability
        if hasattr(self.memory_provider, "vector_search") or hasattr(
            self.memory_provider, "similarity_search"
        ):
            return True

        # Default to in-memory for local/basic providers
        return False

    def _search_via_memory_provider(
        self,
        query: str,
        threshold: float,
        session_id: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        lookup_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SemanticCacheEntry]:
        """
        Search for similar cache entries using the memory provider's vector search capabilities.
        """
        try:
            search_filter = {}

            # Apply agent_id filter only if scope is LOCAL
            if self.config.scope == SemanticCacheScope.LOCAL and self.agent_id:
                search_filter["agent_id"] = self.agent_id

            # Apply memory_id filter if present
            if self.memory_id:
                search_filter["memory_id"] = self.memory_id

            # Apply session_id filter only if session scoping is enabled
            if self.config.enable_session_scoping and session_id:
                search_filter["session_id"] = session_id

            # Tenant isolation: always scope the semantic cache by user_id.
            # None means "anonymous/legacy" — it will match rows with no user_id.
            search_filter["user_id"] = user_id

            logger.debug(f"Semantic cache search filter: {search_filter}")
            logger.debug(
                f"Session scoping enabled: {self.config.enable_session_scoping}"
            )

            # Use the proper vector search method for semantic similarity
            results = self.memory_provider.retrieve_by_query(
                query=query,
                memory_store_type=MemoryType.SEMANTIC_CACHE,
                limit=limit,
                **search_filter,  # Pass agent_id, memory_id, session_id as kwargs
            )

            if not results or len(results) == 0:
                return None

            for candidate in results:
                result_score = float(candidate.get("score", 0.0) or 0.0)
                if result_score < threshold:
                    break
                if candidate.get("user_id") != user_id:
                    continue
                if (
                    self.config.enable_session_scoping
                    and session_id
                    and candidate.get("session_id") != session_id
                ):
                    continue
                query_text = candidate.get("query") or candidate.get("query_text")
                timestamp = candidate.get("timestamp")
                if timestamp is None and "created_at" in candidate:
                    created_at = candidate["created_at"]
                    timestamp = (
                        created_at.timestamp()
                        if hasattr(created_at, "timestamp")
                        else time.time()
                    )
                elif timestamp is None:
                    timestamp = time.time()
                best_match = SemanticCacheEntry(
                    query=query_text,
                    response=candidate["response"],
                    embedding=candidate.get("embedding", []),
                    timestamp=timestamp,
                    session_id=candidate.get("session_id"),
                    memory_id=candidate.get("memory_id"),
                    agent_id=candidate.get("agent_id"),
                    user_id=candidate.get("user_id"),
                    usage_count=candidate.get(
                        "usage_count", candidate.get("hit_count", 0)
                    ),
                    last_accessed=candidate.get("last_accessed"),
                    metadata=candidate.get("metadata", {}),
                    cache_key=candidate.get("cache_key"),
                )
                if "_id" in candidate:
                    best_match.metadata["_id"] = str(candidate["_id"])
                if not self._fresh_for_metadata(best_match):
                    continue
                if not self._metadata_matches(best_match, lookup_metadata):
                    continue
                best_match.metadata["similarity"] = result_score
                return best_match
            return None

        except Exception as e:
            logger.error(f"Error searching via memory provider: {e}")
            return None

    def _is_entry_expired(self, entry: SemanticCacheEntry) -> bool:
        """Check if cache entry has expired based on TTL."""
        if self.config.ttl_hours <= 0:
            return False  # No expiration

        expiry_time = entry.timestamp + (self.config.ttl_hours * 3600)
        return time.time() > expiry_time

    def _cleanup_expired_entries(self) -> int:
        """Remove expired entries from cache."""
        expired_keys = [
            key for key, entry in self.cache.items() if self._is_entry_expired(entry)
        ]

        for key in expired_keys:
            del self.cache[key]

        self._stats["evictions"] += len(expired_keys)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def _evict_lru_entries(self) -> int:
        """Evict least recently used entries if cache is full."""
        if len(self.cache) <= self.config.max_cache_size:
            return 0

        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            self.cache.items(), key=lambda x: x[1].last_accessed or 0
        )

        # Remove oldest entries
        evict_count = len(self.cache) - self.config.max_cache_size
        for i in range(evict_count):
            key, _ = sorted_entries[i]
            del self.cache[key]

        if evict_count > 0:
            self._stats["evictions"] += evict_count
            logger.debug(f"Evicted {evict_count} LRU cache entries")

        return evict_count

    def _generate_cache_key(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Generate a unique cache key for the query."""
        key_parts = [query]
        if self.agent_id:
            key_parts.append(f"agent:{self.agent_id}")
        if self.memory_id:
            key_parts.append(f"memory:{self.memory_id}")
        if session_id:
            key_parts.append(f"session:{session_id}")
        key_parts.append(f"user:{user_id if user_id is not None else '<anonymous>'}")

        return str(uuid.uuid5(uuid.NAMESPACE_OID, "|".join(key_parts)))

    def _exact_cache_match(
        self,
        query: str,
        *,
        session_id: Optional[str],
        user_id: Optional[str],
        lookup_metadata: Optional[Dict[str, Any]],
    ) -> Optional[SemanticCacheEntry]:
        """Resolve an exact scoped key before approximate vector search.

        Exact repeats should not depend on floating-point vector normalization
        or a backend-specific score rounding to precisely ``1.0``. Freshness,
        tenant scope, and every configured fingerprint remain mandatory.
        """
        key = self._generate_cache_key(query, session_id, user_id)
        entry = self.cache.get(key)
        if entry is None:
            return None
        if getattr(entry, "user_id", None) != user_id:
            return None
        if self.agent_id and entry.agent_id != self.agent_id:
            return None
        if self.memory_id and entry.memory_id != self.memory_id:
            return None
        if not self._fresh_for_metadata(entry):
            return None
        if not self._metadata_matches(entry, lookup_metadata):
            return None
        if (
            self.config.enable_session_scoping
            and session_id
            and entry.session_id != session_id
        ):
            return None
        return entry

    def record_bypass(self, reason: str) -> None:
        normalized = str(reason or "policy").strip() or "policy"
        self._stats["bypasses"] += 1
        self._bypass_reasons[normalized] = self._bypass_reasons.get(normalized, 0) + 1

    def _metadata_matches(
        self,
        entry: SemanticCacheEntry,
        lookup_metadata: Optional[Dict[str, Any]],
    ) -> bool:
        expected = dict(lookup_metadata or {})
        stored = dict(entry.metadata or {})
        if self.config.require_fingerprint_match:
            expected_fingerprints = dict(expected.get("fingerprints") or {})
            stored_fingerprints = dict(stored.get("fingerprints") or {})
            if expected_fingerprints != stored_fingerprints:
                return False
        expected_domain = expected.get("domain")
        if expected_domain is not None and stored.get("domain") != expected_domain:
            return False
        expected_tags = set(expected.get("tags") or [])
        stored_tags = set(stored.get("tags") or [])
        if expected_tags and not expected_tags.issubset(stored_tags):
            return False
        return True

    def _fresh_for_metadata(self, entry: SemanticCacheEntry) -> bool:
        if self._is_entry_expired(entry):
            return False
        metadata = dict(entry.metadata or {})
        domains = [metadata.get("domain"), *(metadata.get("domains") or [])]
        limits = [
            float(self.config.freshness_by_domain[str(domain)])
            for domain in domains
            if domain is not None and str(domain) in self.config.freshness_by_domain
        ]
        if not limits:
            return True
        return (time.time() - float(entry.timestamp)) <= min(limits)

    def statistics(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "size": len(self.cache),
            "bypass_reasons": dict(self._bypass_reasons),
            "last_hit": dict(self.last_hit) if self.last_hit else None,
            "last_inspection": (
                self.last_inspection.to_dict() if self.last_inspection else None
            ),
        }

    def _inspection(
        self,
        query: str,
        *,
        entry: Optional[SemanticCacheEntry] = None,
        similarity: Optional[float] = None,
        bypass_reason: Optional[str] = None,
    ) -> SemanticCacheInspection:
        metadata = dict(entry.metadata or {}) if entry is not None else {}
        domains = {
            str(item)
            for item in [metadata.get("domain"), *(metadata.get("domains") or [])]
            if item is not None
        }
        tags = {str(item) for item in (metadata.get("tags") or [])}
        ttl_candidates = []
        if self.config.ttl_hours > 0:
            ttl_candidates.append(float(self.config.ttl_hours) * 3600.0)
        ttl_candidates.extend(
            float(self.config.freshness_by_domain[domain])
            for domain in domains
            if domain in self.config.freshness_by_domain
        )
        # Report the effective freshness window, not only the global cache TTL.
        # Domain-specific limits can deliberately expire operational answers
        # sooner than the backing cache entry.
        ttl_seconds = min(ttl_candidates) if ttl_candidates else None
        age_seconds = (
            max(0.0, time.time() - float(entry.timestamp))
            if entry is not None
            else None
        )
        expires_in = (
            max(0.0, ttl_seconds - age_seconds)
            if ttl_seconds is not None and age_seconds is not None
            else None
        )
        return SemanticCacheInspection(
            lookup_query=str(query),
            hit=entry is not None,
            matched_query=entry.query if entry is not None else None,
            cache_key=entry.cache_key if entry is not None else None,
            similarity=float(similarity) if similarity is not None else None,
            ttl_seconds=ttl_seconds,
            expires_in_seconds=expires_in,
            age_seconds=age_seconds,
            hit_count=int(entry.usage_count) if entry is not None else 0,
            bypass_reason=bypass_reason,
            invalidation_domains=sorted(domains),
            invalidation_tags=sorted(tags),
        )

    def inspect(
        self,
        query: str,
        session_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[str] = None,
        lookup_metadata: Optional[Dict[str, Any]] = None,
        bypass_reason: Optional[str] = None,
    ) -> SemanticCacheInspection:
        """Inspect a potential match without returning the cached response.

        Inspection does not increment hit/miss counters or entry hit counts.
        """
        if bypass_reason:
            result = self._inspection(query, bypass_reason=bypass_reason)
            self.last_inspection = result
            return result
        threshold = (
            self.config.similarity_threshold
            if similarity_threshold is None
            else float(similarity_threshold)
        )
        best_match = self._exact_cache_match(
            query,
            session_id=session_id,
            user_id=user_id,
            lookup_metadata=lookup_metadata,
        )
        best_similarity = 1.0 if best_match is not None else 0.0
        if best_match is not None:
            pass
        elif self._should_use_memory_provider():
            best_match = self._search_via_memory_provider(
                query,
                threshold,
                session_id,
                user_id=user_id,
                lookup_metadata=lookup_metadata,
            )
            if best_match is not None and isinstance(best_match.metadata, dict):
                best_similarity = float(best_match.metadata.get("similarity") or 0.0)
        else:
            query_embedding = self._get_or_generate_embedding(query)
            for entry in self.cache.values():
                if getattr(entry, "user_id", None) != user_id:
                    continue
                if self.agent_id and entry.agent_id != self.agent_id:
                    continue
                if self.memory_id and entry.memory_id != self.memory_id:
                    continue
                if not self._fresh_for_metadata(entry):
                    continue
                if not self._metadata_matches(entry, lookup_metadata):
                    continue
                if (
                    self.config.enable_session_scoping
                    and session_id
                    and entry.session_id != session_id
                ):
                    continue
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity >= threshold and similarity > best_similarity:
                    best_match = entry
                    best_similarity = similarity
        result = self._inspection(
            query,
            entry=best_match,
            similarity=best_similarity if best_match is not None else None,
        )
        self.last_inspection = result
        return result

    def get(
        self,
        query: str,
        session_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        user_id: Optional[str] = None,
        lookup_metadata: Optional[Dict[str, Any]] = None,
        bypass_reason: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve cached response for semantically similar queries.
        Uses memory provider for search when available, falls back to in-memory for local providers.

        Parameters:
        -----------
        query : str
            The query to search for
        session_id : Optional[str]
            Session ID for scoped search
        similarity_threshold : Optional[float]
            Override default similarity threshold

        Returns:
        --------
        Optional[str]
            Cached response if found, None otherwise
        """
        try:
            self.last_hit = None
            self.last_inspection = None
            if bypass_reason:
                self.record_bypass(bypass_reason)
                self.last_inspection = self._inspection(
                    query, bypass_reason=bypass_reason
                )
                return None
            threshold = similarity_threshold or self.config.similarity_threshold
            logger.debug(
                f"Semantic cache query: '{query[:50]}...', threshold={threshold}"
            )
            logger.debug(
                f"Cache context - agent_id={self.agent_id}, memory_id={self.memory_id}, session_id={session_id}"
            )

            # Exact scoped repeats are deterministic and should not be sent
            # through a vector backend where score rounding can turn 1.0 into
            # a miss at strict thresholds.
            best_match = self._exact_cache_match(
                query,
                session_id=session_id,
                user_id=user_id,
                lookup_metadata=lookup_metadata,
            )
            exact_match = best_match is not None

            # Decide whether to use memory provider or in-memory search
            should_use_provider = self._should_use_memory_provider()
            logger.debug(f"Using memory provider for search: {should_use_provider}")

            if exact_match:
                best_similarity = 1.0
            elif should_use_provider:
                logger.debug("Using memory provider for semantic cache retrieval")
                best_match = self._search_via_memory_provider(
                    query,
                    threshold,
                    session_id,
                    user_id=user_id,
                    lookup_metadata=lookup_metadata,
                )
            else:
                # Generate query embedding for in-memory search (with caching)
                query_embedding = self._get_or_generate_embedding(query)

                logger.debug("Using in-memory search for semantic cache retrieval")
                # Clean up expired entries first (only for in-memory)
                self._cleanup_expired_entries()

                if not self.cache:
                    self._stats["misses"] += 1
                    self.last_inspection = self._inspection(query)
                    return None

                # In-memory search (original logic)
                best_match = None
                best_similarity = 0.0

                for entry in self.cache.values():
                    # Tenant isolation: skip entries from other users.
                    if getattr(entry, "user_id", None) != user_id:
                        continue
                    if self.agent_id and entry.agent_id != self.agent_id:
                        continue
                    if self.memory_id and entry.memory_id != self.memory_id:
                        continue

                    if not self._fresh_for_metadata(entry):
                        continue
                    if not self._metadata_matches(entry, lookup_metadata):
                        continue

                    # Skip entries that don't match scope (only if session scoping is enabled)
                    if (
                        self.config.enable_session_scoping
                        and session_id
                        and entry.session_id != session_id
                    ):
                        continue

                    similarity = self._cosine_similarity(
                        query_embedding, entry.embedding
                    )

                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = entry

            if best_match:
                # Update usage statistics
                if self.config.enable_usage_tracking:
                    best_match.usage_count += 1
                    best_match.last_accessed = time.time()

                    # If using memory provider, also update in the database
                    if (
                        self._should_use_memory_provider()
                        and self.config.enable_memory_provider_sync
                    ):
                        self._update_usage_in_memory_provider(best_match)

                logger.debug(f"Cache HIT: query: {query[:50]}...")
                self._stats["hits"] += 1
                similarity = None
                if exact_match:
                    similarity = 1.0
                elif isinstance(best_match.metadata, dict):
                    similarity = best_match.metadata.get("similarity")
                if similarity is None and not should_use_provider:
                    similarity = best_similarity
                self.last_hit = {
                    "cache_key": best_match.cache_key,
                    "query": best_match.query,
                    "similarity": similarity,
                    "age_seconds": max(0.0, time.time() - best_match.timestamp),
                    "agent_id": best_match.agent_id,
                    "memory_id": best_match.memory_id,
                    "session_id": best_match.session_id,
                    "user_id": best_match.user_id,
                    "metadata": dict(best_match.metadata or {}),
                }
                self.last_inspection = self._inspection(
                    query,
                    entry=best_match,
                    similarity=similarity,
                )
                return best_match.response
            else:
                logger.debug(f"Cache MISS: no similar query found for: {query[:50]}...")
                self._stats["misses"] += 1
                self.last_inspection = self._inspection(query)
                return None

        except Exception as e:
            logger.error(f"Error in semantic cache get: {e}")
            self._stats["misses"] += 1
            self.last_inspection = self._inspection(
                query, bypass_reason=f"lookup_error:{type(e).__name__}"
            )
            return None

    def _update_usage_in_memory_provider(self, entry: SemanticCacheEntry) -> bool:
        """Update usage statistics in the memory provider."""
        try:
            # If we have the document ID from the cache hit, use it directly
            if entry.metadata and "_id" in entry.metadata:
                # Oracle semantic_cache table only has hit_count, not last_accessed
                update_data = {
                    "hit_count": entry.usage_count,  # Map usage_count to hit_count for Oracle
                }

                return self.memory_provider.update_by_id(
                    id=entry.metadata["_id"],
                    data=update_data,
                    memory_store_type=MemoryType.SEMANTIC_CACHE,
                )
            else:
                # Fallback: use cache_key to find and update
                if entry.cache_key:
                    # Directly update by cache_key without needing _id
                    # Oracle semantic_cache table only has hit_count, not last_accessed or usage_count
                    update_data = {
                        "cache_key": entry.cache_key,
                        "hit_count": entry.usage_count,  # Map usage_count to hit_count for Oracle
                    }

                    # Try update_by_id with cache_key as the identifier
                    try:
                        return self.memory_provider.update_by_id(
                            id=entry.cache_key,
                            data=update_data,
                            memory_store_type=MemoryType.SEMANTIC_CACHE,
                        )
                    except Exception:
                        # If that fails, log and continue
                        logger.debug(
                            f"Could not update usage by cache_key: {entry.cache_key}"
                        )
                        return False

                return False

        except Exception as e:
            logger.warning(f"Failed to update usage statistics in memory provider: {e}")
            return False

    def set(
        self,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Store query-response pair in cache with embedding.

        Parameters:
        -----------
        query : str
            The query text
        response : str
            The response to cache
        session_id : Optional[str]
            Session ID for scoped storage
        metadata : Optional[Dict[str, Any]]
            Additional metadata to store

        Returns:
        --------
        bool
            True if stored successfully, False otherwise
        """
        try:
            metadata_value = dict(metadata or {})
            admission = dict(metadata_value.get("admission") or {})
            if self.config.admission_policy == "read_only_deterministic" and (
                admission.get("read_only", True) is not True
                or admission.get("deterministic", True) is not True
            ):
                self.record_bypass("cache_admission_policy")
                return False
            # Generate embedding for query (with caching to avoid duplication)
            query_embedding = self._get_or_generate_embedding(query)

            # Generate cache key
            cache_key = self._generate_cache_key(query, session_id, user_id)

            # Create cache entry
            entry = SemanticCacheEntry(
                query=query,
                response=response,
                embedding=query_embedding,
                timestamp=time.time(),
                session_id=session_id,
                memory_id=self.memory_id,
                agent_id=self.agent_id,
                user_id=user_id,
                metadata=metadata_value,
                cache_key=cache_key,
            )

            # Store in memory cache
            self.cache[cache_key] = entry

            # Clean up if necessary
            self._evict_lru_entries()

            # Sync to memory provider if enabled
            if self.memory_provider and self.config.enable_memory_provider_sync:
                self._sync_to_memory_provider(cache_key, entry)

            logger.debug(f"Cache SET: stored query: {query[:50]}...")
            self._stats["writes"] += 1
            return True

        except Exception as e:
            logger.error(f"Error in semantic cache set: {e}")
            return False

    def _sync_to_memory_provider(
        self, cache_key: str, entry: SemanticCacheEntry
    ) -> bool:
        """Sync cache entry to memory provider for persistence."""
        try:
            # Prepare data for storage
            data = entry.model_dump()
            data["cache_key"] = cache_key
            data["created_at"] = datetime.fromtimestamp(entry.timestamp)
            data["agent_id"] = self.agent_id
            data["memory_id"] = self.memory_id
            data["user_id"] = getattr(entry, "user_id", None)
            data["scope"] = getattr(self.config.scope, "value", self.config.scope)
            data["similarity_threshold"] = self.config.similarity_threshold
            data["hit_count"] = entry.usage_count
            if self.config.ttl_hours > 0:
                data["expires_at"] = datetime.fromtimestamp(
                    entry.timestamp + (self.config.ttl_hours * 3600)
                )

            # Map 'query' to 'query_text' for compatibility with Oracle provider
            if "query" in data:
                data["query_text"] = data["query"]

            # Store in memory provider using the correct method
            self.memory_provider.store(
                data=data, memory_store_type=MemoryType.SEMANTIC_CACHE
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to sync cache entry to memory provider: {e}")
            return False

    def _load_from_memory_provider(self) -> int:
        """Load existing cache entries from memory provider."""
        try:
            if not self.memory_provider:
                return 0

            # Build query for cached entries
            query = {}
            if self.agent_id:
                query["agent_id"] = self.agent_id
            if self.memory_id:
                query["memory_id"] = self.memory_id

            # Retrieve cached entries using the correct method
            result = self.memory_provider.retrieve_by_query(
                query=query,
                memory_store_type=MemoryType.SEMANTIC_CACHE,
                limit=self.config.max_cache_size,
            )

            # Handle both single result and list results
            if result is None:
                entries = []
            elif hasattr(result, "__iter__") and not isinstance(result, (str, dict)):
                # Handle cursor or list-like objects
                entries = list(result)
            elif isinstance(result, dict):
                entries = [result]
            else:
                entries = [result]

            loaded_count = 0
            for entry_data in entries:
                try:
                    # Reconstruct cache entry
                    metadata = entry_data.get("metadata", {})
                    # Preserve MongoDB _id in metadata (standard pattern)
                    if "_id" in entry_data:
                        metadata["_id"] = str(entry_data["_id"])

                    # Handle both 'query' and 'query_text' field names for compatibility
                    query_text = entry_data.get("query") or entry_data.get("query_text")

                    # Handle timestamp field (Oracle uses 'created_at', cache uses 'timestamp')
                    timestamp = entry_data.get("timestamp")
                    if timestamp is None and "created_at" in entry_data:
                        # Convert datetime to timestamp if needed
                        created_at = entry_data["created_at"]
                        if hasattr(created_at, "timestamp"):
                            timestamp = created_at.timestamp()
                        else:
                            timestamp = time.time()
                    elif timestamp is None:
                        timestamp = time.time()

                    cache_entry = SemanticCacheEntry(
                        query=query_text,
                        response=entry_data["response"],
                        embedding=entry_data["embedding"],
                        timestamp=timestamp,
                        session_id=entry_data.get("session_id"),
                        memory_id=entry_data.get("memory_id"),
                        agent_id=entry_data.get("agent_id"),
                        user_id=entry_data.get("user_id"),
                        usage_count=entry_data.get("usage_count", 0),
                        last_accessed=entry_data.get("last_accessed"),
                        metadata=metadata,
                        cache_key=entry_data.get("cache_key"),
                    )

                    # Skip expired entries
                    if self._is_entry_expired(cache_entry):
                        continue

                    cache_key = entry_data.get("cache_key") or self._generate_cache_key(
                        cache_entry.query,
                        cache_entry.session_id,
                        cache_entry.user_id,
                    )

                    self.cache[cache_key] = cache_entry
                    loaded_count += 1

                except Exception as e:
                    logger.warning(f"Failed to load cache entry: {e}")
                    continue

            logger.info(f"Loaded {loaded_count} cache entries from memory provider")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load from memory provider: {e}")
            return 0

    def clear(
        self,
        session_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        clear_persistent: Optional[bool] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Clear cache entries with optional filtering.

        Parameters:
        -----------
        session_id : Optional[str]
            Clear only entries for this session ID
        memory_id : Optional[str]
            Clear only entries for this memory ID
        clear_persistent : Optional[bool]
            Whether to also clear from persistent storage (MongoDB).
            If None, defaults to True when enable_memory_provider_sync=True

        Returns:
        --------
        int
            Number of entries cleared from in-memory cache
        """
        # Determine if we should clear persistent storage
        if clear_persistent is None:
            clear_persistent = (
                self.memory_provider is not None
                and self.config.enable_memory_provider_sync
            )

        # Count entries before clearing for return value
        memory_cleared = 0
        persistent_cleared = 0

        if session_id is None and memory_id is None and user_id is None:
            # Clear all entries
            memory_cleared = len(self.cache)
            self.cache.clear()

            # Also clear from persistent storage if enabled
            if clear_persistent and self.memory_provider:
                try:
                    # Clear all entries for this agent
                    persistent_cleared = self.memory_provider.clear_semantic_cache(
                        agent_id=self.agent_id,
                        memory_id=None,  # Clear all memory IDs for this agent
                    )
                    logger.info(
                        f"Cleared {persistent_cleared} entries from persistent storage"
                    )
                except Exception as e:
                    logger.error(f"Failed to clear persistent cache: {e}")

            logger.info(f"Cleared all {memory_cleared} cache entries from memory")
            return memory_cleared

        # Clear filtered entries from memory
        keys_to_remove = []
        for key, entry in self.cache.items():
            should_remove = True
            if session_id is not None and entry.session_id != session_id:
                should_remove = False
            if memory_id is not None and entry.memory_id != memory_id:
                should_remove = False
            if user_id is not None and getattr(entry, "user_id", None) != user_id:
                should_remove = False
            if should_remove:
                keys_to_remove.append(key)

        memory_cleared = len(keys_to_remove)
        for key in keys_to_remove:
            del self.cache[key]

        # Also clear from persistent storage if enabled
        if clear_persistent and self.memory_provider and memory_cleared > 0:
            try:
                # Build filter for persistent storage
                # Note: session_id filtering not supported by memory provider currently
                # so we clear by agent_id and memory_id only
                persistent_cleared = self.memory_provider.clear_semantic_cache(
                    agent_id=self.agent_id,
                    memory_id=memory_id,  # This will be passed if specified
                )
                logger.info(
                    f"Cleared {persistent_cleared} entries from persistent storage"
                )

                if session_id is not None:
                    logger.warning(
                        "Session-specific clearing from persistent storage is not supported. "
                        "Cleared by agent_id and memory_id only."
                    )
            except Exception as e:
                logger.error(f"Failed to clear persistent cache: {e}")

        logger.info(f"Cleared {memory_cleared} filtered cache entries from memory")
        return memory_cleared

    def invalidate(
        self,
        *,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        data_version: Optional[str] = None,
    ) -> int:
        """Invalidate entries by operational domain, tag, or data version."""
        wanted_domains = {str(item) for item in (domains or [])}
        wanted_tags = {str(item) for item in (tags or [])}
        keys: List[str] = []
        for key, entry in self.cache.items():
            metadata = dict(entry.metadata or {})
            entry_domains = {
                str(item)
                for item in [metadata.get("domain"), *(metadata.get("domains") or [])]
                if item is not None
            }
            entry_tags = {str(item) for item in (metadata.get("tags") or [])}
            fingerprints = dict(metadata.get("fingerprints") or {})
            if (
                (wanted_domains and wanted_domains.intersection(entry_domains))
                or (wanted_tags and wanted_tags.intersection(entry_tags))
                or (
                    data_version is not None
                    and fingerprints.get("data_version") == str(data_version)
                )
            ):
                keys.append(key)
        for key in keys:
            self.cache.pop(key, None)
        self._stats["evictions"] += len(keys)
        provider_hook = getattr(self.memory_provider, "invalidate_semantic_cache", None)
        persistent_removed = 0
        if callable(provider_hook):
            try:
                persistent_removed = int(
                    provider_hook(
                        agent_id=self.agent_id,
                        memory_id=self.memory_id,
                        domains=sorted(wanted_domains),
                        tags=sorted(wanted_tags),
                        data_version=data_version,
                    )
                    or 0
                )
            except Exception as exc:
                logger.warning("Persistent semantic-cache invalidation failed: %s", exc)
        if persistent_removed > len(keys):
            self._stats["evictions"] += persistent_removed - len(keys)
        return max(len(keys), persistent_removed)


# Standalone semantic cache for external frameworks
class StandaloneSemanticCache(SemanticCache):
    """Standalone semantic cache that can be used with any agent framework."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
        ttl_hours: float = 24.0,
        scope: SemanticCacheScope = SemanticCacheScope.LOCAL,
        embedding_provider: str = "openai",
        embedding_config: Optional[Dict[str, Any]] = None,
        enable_persistence: bool = False,
        persistence_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize standalone semantic cache for external use.

        Parameters:
        -----------
        similarity_threshold : float
            Minimum similarity score for cache hits (0.0-1.0)
        max_cache_size : int
            Maximum number of entries to keep in memory
        ttl_hours : float
            Time-to-live in hours (0 = no expiration)
        scope : SemanticCacheScope
            Cache scope (LOCAL = agent-specific, GLOBAL = cross-agent)
        embedding_provider : str
            Embedding provider ('openai', 'voyageai', 'ollama')
        embedding_config : Optional[Dict[str, Any]]
            Configuration for embedding provider
        enable_persistence : bool
            Whether to enable MongoDB persistence
        persistence_config : Optional[Dict[str, Any]]
            MongoDB configuration for persistence
        """
        config = SemanticCacheConfig(
            similarity_threshold=similarity_threshold,
            max_cache_size=max_cache_size,
            ttl_hours=ttl_hours,
            scope=scope,
            embedding_provider=embedding_provider,
            embedding_config=embedding_config,
            enable_memory_provider_sync=enable_persistence,
        )

        memory_provider = None
        if enable_persistence and persistence_config:
            try:
                from ..memory_provider.mongodb.provider import (
                    MongoDBConfig,
                    MongoDBProvider,
                )

                mongodb_config = MongoDBConfig(**persistence_config)
                memory_provider = MongoDBProvider(mongodb_config)
            except Exception as e:
                logger.warning(f"Failed to initialize persistence: {e}")

        super().__init__(
            config=config, memory_provider=memory_provider, agent_id="standalone_cache"
        )

    def query(self, text: str, session_id: Optional[str] = None) -> Optional[str]:
        """Simple query interface for external frameworks."""
        return self.get(text, session_id=session_id)

    def cache_response(
        self, query: str, response: str, session_id: Optional[str] = None
    ) -> bool:
        """Simple caching interface for external frameworks."""
        return self.set(query, response, session_id=session_id)
