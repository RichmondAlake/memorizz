"""Memory management functionality for MemAgent."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...enums import MemoryType, Role
from ...long_term_memory.episodic.conversational_memory_unit import (
    ConversationMemoryUnit,
)
from ...memory_provider import MemoryProvider
from ...memory_unit import MemoryUnit

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages all memory-related operations for MemAgent.

    This class encapsulates memory storage, retrieval, and management functionality
    that was previously embedded in the main MemAgent class.
    """

    def __init__(self, memory_provider: MemoryProvider):
        """
        Initialize the memory manager.

        Args:
            memory_provider: The memory provider instance to use for storage.
        """
        self.memory_provider = memory_provider
        self._conversation_memory_cache = {}

    @staticmethod
    def _history_timestamp(entry: Dict[str, Any]) -> float:
        """Build a numeric timestamp key for stable conversation ordering."""
        if not isinstance(entry, dict):
            return 0.0

        raw_value = (
            entry.get("timestamp")
            or entry.get("created_at")
            or entry.get("createdAt")
            or 0
        )

        if isinstance(raw_value, datetime):
            return raw_value.timestamp()
        if isinstance(raw_value, (int, float)):
            return float(raw_value)

        text = str(raw_value).strip()
        if not text:
            return 0.0

        try:
            return float(text)
        except (TypeError, ValueError):
            pass

        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    def load_conversation_history(
        self, memory_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Load conversation history for a given memory ID.

        Args:
            memory_id: The memory ID to load history for.
            limit: Maximum number of conversations to retrieve.

        Returns:
            List of conversation history entries.
        """
        try:
            logger.debug(f"Loading conversation history for memory_id: {memory_id}")

            # Check cache first
            if memory_id in self._conversation_memory_cache:
                cached = self._conversation_memory_cache[memory_id]
                if not limit or limit <= 0:
                    return list(cached)
                return cached[-limit:]

            # Load from memory provider
            history = (
                self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(
                    memory_id=memory_id,
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    limit=None,
                )
            )
            history = [row for row in (history or []) if isinstance(row, dict)]
            history.sort(key=self._history_timestamp)

            # Cache the results
            self._conversation_memory_cache[memory_id] = history

            logger.info(
                f"Loaded {len(history)} conversation entries for memory_id: {memory_id}"
            )
            if not limit or limit <= 0:
                return history
            return history[-limit:]

        except Exception as e:
            logger.error(f"Failed to load conversation history: {e}")
            return []

    def save_memory_unit(
        self, memory_unit: MemoryUnit, memory_id: str
    ) -> Optional[str]:
        """
        Save a memory unit to storage.

        Args:
            memory_unit: The memory unit to save.
            memory_id: The memory ID to associate with.

        Returns:
            The ID of the saved memory unit, or None if failed.
        """
        try:
            # Store the memory unit
            unit_id = self.memory_provider.store(
                memory_id=memory_id, memory_unit=memory_unit
            )

            # Invalidate cache for this memory_id
            if memory_id in self._conversation_memory_cache:
                del self._conversation_memory_cache[memory_id]

            logger.debug(f"Saved memory unit {unit_id} for memory_id: {memory_id}")
            return unit_id

        except Exception as e:
            logger.error(f"Failed to save memory unit: {e}")
            return None

    def retrieve_relevant_memories(
        self, query: str, memory_type: MemoryType, memory_id: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a query.

        Args:
            query: The query to search for.
            memory_type: Type of memory to retrieve.
            memory_id: The memory ID to search within.
            limit: Maximum number of results.

        Returns:
            List of relevant memory entries.
        """
        try:
            results = self.memory_provider.retrieve_by_query(
                query=query, memory_id=memory_id, memory_type=memory_type, limit=limit
            )

            logger.debug(
                f"Retrieved {len(results)} relevant memories for query: {query[:50]}..."
            )
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []

    def create_conversation_memory_unit(
        self,
        role: Role,
        content: str,
        conversation_id: str,
        memory_id: str,
        timestamp: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> ConversationMemoryUnit:
        """
        Create a conversation memory unit.

        Args:
            role: The role (user or assistant).
            content: The message content.
            conversation_id: The conversation ID.
            memory_id: The memory ID.
            timestamp: Optional timestamp.
            agent_id: Optional agent ID.

        Returns:
            A new ConversationMemoryUnit instance.
        """
        if timestamp is None:
            timestamp = datetime.now()

        return ConversationMemoryUnit(
            role=role.value,
            content=content,
            conversation_id=conversation_id,
            memory_id=memory_id,
            timestamp=timestamp.isoformat(),
            embedding=None,  # None instead of [], Oracle VECTOR requires NULL not empty list
            agent_id=agent_id,
        )

    def clear_conversation_cache(self, memory_id: Optional[str] = None):
        """
        Clear the conversation cache.

        Args:
            memory_id: If provided, only clear cache for this memory_id.
                       Otherwise, clear entire cache.
        """
        if memory_id:
            if memory_id in self._conversation_memory_cache:
                del self._conversation_memory_cache[memory_id]
                logger.debug(f"Cleared conversation cache for memory_id: {memory_id}")
        else:
            self._conversation_memory_cache.clear()
            logger.debug("Cleared entire conversation cache")

    def update_memory_ids(self, agent_id: str, memory_ids: List[str]) -> bool:
        """
        Update the memory IDs associated with an agent.

        Args:
            agent_id: The agent ID to update.
            memory_ids: The new list of memory IDs.

        Returns:
            True if successful, False otherwise.
        """
        try:
            success = self.memory_provider.update_memagent_memory_ids(
                agent_id=agent_id, memory_ids=memory_ids
            )

            if success:
                logger.info(f"Updated memory IDs for agent {agent_id}: {memory_ids}")
            else:
                logger.warning(f"Failed to update memory IDs for agent {agent_id}")

            return success

        except Exception as e:
            logger.error(f"Error updating memory IDs: {e}")
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete all memories associated with a memory ID.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Clear cache first
            self.clear_conversation_cache(memory_id)

            # Delete from storage
            success = self.memory_provider.delete_by_id(memory_id)

            if success:
                logger.info(f"Deleted all memories for memory_id: {memory_id}")
            else:
                logger.warning(f"Failed to delete memories for memory_id: {memory_id}")

            return success

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
