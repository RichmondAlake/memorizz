# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Cache management functionality for MemAgent."""

import logging
from typing import Any, Dict, Optional, Union

from ...short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages semantic caching for MemAgent.

    This class encapsulates cache-related functionality that was
    previously embedded in the main MemAgent class.
    """

    def __init__(
        self,
        enabled: bool = False,
        config: Optional[Union[SemanticCacheConfig, Dict[str, Any]]] = None,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        memory_provider: Optional[Any] = None,
    ):
        """
        Initialize the cache manager.

        Args:
            enabled: Whether semantic caching is enabled.
            config: Configuration for the semantic cache.
            agent_id: The agent ID for cache scoping.
            memory_id: The memory ID for cache scoping.
            memory_provider: Memory provider for cache persistence.
        """
        self.enabled = enabled
        self.cache_instance = None
        self.memory_provider = memory_provider

        if enabled:
            self._initialize_cache(config, agent_id, memory_id)

    def _initialize_cache(
        self,
        config: Optional[Union[SemanticCacheConfig, Dict[str, Any]]],
        agent_id: Optional[str],
        memory_id: Optional[str],
    ):
        """Initialize the semantic cache instance."""
        try:
            # Handle configuration
            if config is None:
                cache_config = SemanticCacheConfig()
            elif isinstance(config, dict):
                cache_config = SemanticCacheConfig(**config)
            else:
                cache_config = config

            # Create cache instance
            self.cache_instance = SemanticCache(
                config=cache_config,
                memory_provider=self.memory_provider,  # ✅ Pass memory provider for persistence
                agent_id=agent_id,
                memory_id=memory_id,
            )

            logger.info(f"Initialized semantic cache for agent {agent_id}")

        except Exception as e:
            logger.error(f"Failed to initialize cache: {e}")
            self.enabled = False
            self.cache_instance = None

    def get_cached_response(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get a cached response for a query.

        Args:
            query: The query to look up.
            session_id: Optional session/conversation ID.
            user_id: Optional end-user identifier for multi-tenant scoping.
                When set, only cache entries matching the same ``user_id`` are
                considered.

        Returns:
            Cached response if found, None otherwise.
        """
        if not self.enabled or not self.cache_instance:
            return None

        try:
            response = self.cache_instance.get(
                query=query, session_id=session_id, user_id=user_id
            )

            if response:
                logger.debug(f"Cache hit for query: {query[:50]}...")
            else:
                logger.debug(f"Cache miss for query: {query[:50]}...")

            return response

        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None

    def cache_response(
        self,
        query: str,
        response: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Cache a query-response pair.

        Args:
            query: The query.
            response: The response to cache.
            session_id: Optional session/conversation ID.
            user_id: Optional end-user identifier for multi-tenant scoping.

        Returns:
            True if successfully cached, False otherwise.
        """
        if not self.enabled or not self.cache_instance:
            return False

        try:
            self.cache_instance.set(
                query=query,
                response=response,
                session_id=session_id,
                user_id=user_id,
            )

            logger.debug(f"Cached response for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
            return False

    def clear_cache(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Clear the cache.

        Args:
            session_id: If provided, only clear cache for this session.
                       Otherwise, clear entire cache.
            user_id: If provided, only clear cache entries for this user.
        """
        if not self.cache_instance:
            return

        try:
            if session_id or user_id is not None:
                # Clear session-specific / user-specific cache
                self.cache_instance.clear(session_id=session_id, user_id=user_id)
                logger.debug(
                    "Cleared cache for session=%s user_id=%s", session_id, user_id
                )
            else:
                # Clear entire cache
                self.cache_instance.clear()
                logger.debug("Cleared entire semantic cache")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def update_scope(
        self, agent_id: Optional[str] = None, memory_id: Optional[str] = None
    ):
        """
        Update the cache scope.

        Args:
            agent_id: New agent ID for scoping.
            memory_id: New memory ID for scoping.
        """
        if not self.cache_instance:
            return

        if agent_id:
            self.cache_instance.agent_id = agent_id

        if memory_id:
            self.cache_instance.memory_id = memory_id

        logger.debug(f"Updated cache scope - agent: {agent_id}, memory: {memory_id}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache statistics.
        """
        if not self.cache_instance:
            return {"enabled": False, "hits": 0, "misses": 0, "size": 0}

        # Get stats from cache instance
        # This would need to be implemented in the SemanticCache class
        return {
            "enabled": True,
            "hits": getattr(self.cache_instance, "hits", 0),
            "misses": getattr(self.cache_instance, "misses", 0),
            "size": getattr(self.cache_instance, "size", 0),
        }
