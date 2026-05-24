# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memory management functionality for MemAgent."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...enums import MemoryType, Role
from ...long_term.episodic.conversational_memory_unit import ConversationMemoryUnit
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
        # Prevent unbounded growth when conversation memory is updated in-place.
        self._conversation_memory_cache_max_entries = 5000

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

    @staticmethod
    def _is_summarized_message(entry: Any) -> bool:
        """Check if a conversation entry has been marked as summarized."""
        if not isinstance(entry, dict):
            return False
        # Check top-level summary_id
        if entry.get("summary_id"):
            return True
        # Check nested content.summary_id
        content = entry.get("content")
        if isinstance(content, dict) and content.get("summary_id"):
            return True
        return False

    def load_conversation_history(
        self,
        memory_id: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load conversation history for a given memory ID.

        Args:
            memory_id: The memory ID to load history for.
            limit: Maximum number of conversations to retrieve.
            user_id: Optional user scope. When set, history is restricted to
                entries with a matching ``user_id``. When ``None``, only
                legacy/unscoped entries are returned.

        Returns:
            List of conversation history entries.
        """
        try:
            logger.debug(f"Loading conversation history for memory_id: {memory_id}")

            cache_key = (memory_id, user_id)
            if cache_key in self._conversation_memory_cache:
                cached = self._conversation_memory_cache[cache_key]
                if limit and limit > 0 and len(cached) < limit:
                    # Cache does not satisfy requested limit; fall through to reload.
                    pass
                else:
                    if not limit or limit <= 0:
                        return list(cached)
                    return cached[-limit:]

            # Load from memory provider
            fetch_limit = None
            if limit and limit > 0:
                fetch_limit = int(limit)
            history = (
                self.memory_provider.retrieve_conversation_history_ordered_by_timestamp(
                    memory_id=memory_id,
                    memory_type=MemoryType.CONVERSATION_MEMORY,
                    limit=fetch_limit,
                    user_id=user_id,
                )
            )
            history = [row for row in (history or []) if isinstance(row, dict)]
            # Exclude messages that have been compacted into summaries
            history = [row for row in history if not self._is_summarized_message(row)]
            history.sort(key=self._history_timestamp)

            # Cache the results
            self._conversation_memory_cache[cache_key] = history

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

            # Update conversation cache in-place when we can, so hot paths don't
            # re-load and re-sort large histories on every message.
            unit_user_id = getattr(memory_unit, "user_id", None)
            cache_key = (memory_id, unit_user_id)
            cached = self._conversation_memory_cache.get(cache_key)
            if (
                cached is not None
                and hasattr(memory_unit, "role")
                and hasattr(memory_unit, "thread_id")
            ):
                try:
                    timestamp = getattr(memory_unit, "timestamp", None)
                    entry = {
                        "id": unit_id,
                        "memory_type": MemoryType.CONVERSATION_MEMORY,
                        "timestamp": timestamp,
                        "memory_id": memory_id,
                        "user_id": unit_user_id,
                        "content": {
                            "role": getattr(memory_unit, "role", None),
                            "content": getattr(memory_unit, "content", None),
                            "thread_id": getattr(memory_unit, "thread_id", None),
                            "timestamp": timestamp,
                            "user_id": unit_user_id,
                        },
                    }
                    cached.append(entry)
                    if (
                        self._conversation_memory_cache_max_entries
                        and len(cached) > self._conversation_memory_cache_max_entries
                    ):
                        self._conversation_memory_cache[cache_key] = cached[
                            -self._conversation_memory_cache_max_entries :
                        ]
                except Exception:
                    # Fall back to invalidation if cache update fails.
                    self._conversation_memory_cache.pop(cache_key, None)
            elif cache_key in self._conversation_memory_cache:
                # Unknown memory unit shape; safest to invalidate.
                self._conversation_memory_cache.pop(cache_key, None)

            logger.debug(f"Saved memory unit {unit_id} for memory_id: {memory_id}")
            return unit_id

        except Exception as e:
            logger.error(f"Failed to save memory unit: {e}")
            return None

    def retrieve_relevant_memories(
        self,
        query: str,
        memory_type: MemoryType,
        memory_id: str,
        limit: int = 5,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a query.

        Args:
            query: The query to search for.
            memory_type: Type of memory to retrieve.
            memory_id: The memory ID to search within.
            limit: Maximum number of results.
            user_id: Optional user scope. When set, results are restricted to
                entries with a matching ``user_id``.

        Returns:
            List of relevant memory entries. Always a list — providers that
            return ``None`` (legacy "no results" convention) or a PyMongo
            cursor are coerced here so callers can rely on the shape.
        """
        try:
            results = self.memory_provider.retrieve_by_query(
                query=query,
                memory_id=memory_id,
                memory_type=memory_type,
                limit=limit,
                user_id=user_id,
            )

            # Normalize provider return types:
            # - ``None`` is the legacy "no results" sentinel from some
            #   providers (e.g. MongoDB's ``retrieve_toolbox_item``).
            # - PyMongo cursors don't support ``len()`` and exhaust on
            #   iteration, so materialize them here once.
            if results is None:
                results = []
            elif not isinstance(results, list):
                try:
                    results = list(results)
                except TypeError:
                    # Single dict from "find_one"-style helpers.
                    results = [results] if results else []

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
        thread_id: str,
        memory_id: str,
        timestamp: Optional[datetime] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ConversationMemoryUnit:
        """
        Create a conversation memory unit.

        Args:
            role: The role (user or assistant).
            content: The message content.
            thread_id: The thread ID.
            memory_id: The memory ID.
            timestamp: Optional timestamp.
            agent_id: Optional agent ID.
            user_id: Optional end-user identifier for multi-tenant scoping.

        Returns:
            A new ConversationMemoryUnit instance.
        """
        if timestamp is None:
            timestamp = datetime.now()

        return ConversationMemoryUnit(
            role=role.value,
            content=content,
            thread_id=thread_id,
            memory_id=memory_id,
            timestamp=timestamp.isoformat(),
            embedding=None,  # None instead of [], Oracle VECTOR requires NULL not empty list
            agent_id=agent_id,
            user_id=user_id,
        )

    def clear_conversation_cache(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        """
        Clear the conversation cache.

        Args:
            memory_id: If provided, only clear cache for this memory_id.
                       Otherwise, clear entire cache.
            user_id: If provided alongside ``memory_id``, only clear the cache
                slot for that specific ``(memory_id, user_id)`` pair. When
                ``memory_id`` is provided without ``user_id``, every cache slot
                for that ``memory_id`` (across all users) is cleared.
        """
        if memory_id:
            if user_id is not None:
                self._conversation_memory_cache.pop((memory_id, user_id), None)
                logger.debug(
                    "Cleared conversation cache for memory_id=%s user_id=%s",
                    memory_id,
                    user_id,
                )
            else:
                # Clear every slot associated with ``memory_id`` regardless of
                # its user_id scope. Also tolerate legacy callers that stored
                # cache entries under a raw ``memory_id`` string key before
                # this provider learned about user_id.
                keys_to_remove = [
                    key
                    for key in self._conversation_memory_cache
                    if (isinstance(key, tuple) and key[0] == memory_id)
                    or key == memory_id
                ]
                for key in keys_to_remove:
                    self._conversation_memory_cache.pop(key, None)
                logger.debug(
                    "Cleared conversation cache for memory_id=%s (all users)",
                    memory_id,
                )
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

    def store_tool_log(
        self,
        tool_name: str,
        arguments: Any,
        result: Any,
        memory_id: str,
        agent_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        success: bool = True,
        error: Optional[str] = None,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a tool execution log entry to the database.

        Tool log entries are offloaded to the database instead of being kept
        in the context window, saving context space. Only a compact reference
        is kept in working memory.

        Args:
            tool_name: Name of the tool that was executed.
            arguments: The arguments passed to the tool.
            result: The tool's output/result.
            memory_id: The memory ID to associate with.
            agent_id: Optional agent ID.
            tool_call_id: Optional tool call ID from the LLM.
            success: Whether the tool executed successfully.
            error: Error message if the tool failed.
            thread_id: Optional thread ID.

        Returns:
            The ID of the stored tool log entry, or None if failed.
        """
        try:
            log_id = str(uuid.uuid4())
            timestamp = datetime.now()

            # Serialize result to string if needed
            result_str = result
            if not isinstance(result, str):
                try:
                    result_str = json.dumps(result, default=str)
                except (TypeError, ValueError):
                    result_str = str(result)

            args_str = arguments
            if not isinstance(arguments, str):
                try:
                    args_str = json.dumps(arguments, default=str)
                except (TypeError, ValueError):
                    args_str = str(arguments)

            tool_log_data = {
                "tool_log_id": log_id,
                "tool_name": tool_name,
                "arguments": args_str,
                "result": result_str,
                "success": success,
                "error": error,
                "timestamp": timestamp.isoformat(),
                "agent_id": agent_id,
                "tool_call_id": tool_call_id or "",
                "thread_id": thread_id or "",
                "memory_id": memory_id,
                "user_id": user_id,
            }

            stored_id = self.memory_provider.store(
                tool_log_data, memory_store_type=MemoryType.TOOL_LOG
            )

            logger.debug(
                "Stored tool log for '%s' (id=%s, memory_id=%s)",
                tool_name,
                stored_id or log_id,
                memory_id,
            )
            return stored_id or log_id

        except Exception as e:
            logger.error("Failed to store tool log for '%s': %s", tool_name, e)
            return None

    def retrieve_tool_log(
        self,
        tool_log_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a tool log entry by its ID.

        Args:
            tool_log_id: The ID of the tool log to retrieve.
            user_id: Optional user scope. When set, returns ``None`` if the
                stored log belongs to a different user.

        Returns:
            The tool log entry dict, or None if not found.
        """
        try:
            result = self.memory_provider.retrieve_by_id(
                tool_log_id, MemoryType.TOOL_LOG
            )
            if result and isinstance(result, dict):
                if result.get("user_id") != user_id:
                    return None
            return result
        except Exception as e:
            logger.error("Failed to retrieve tool log %s: %s", tool_log_id, e)
            return None

    def list_tool_logs(
        self,
        memory_id: str,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent tool log entries for a given memory ID.

        Args:
            memory_id: The memory ID to list logs for.
            limit: Maximum number of entries to return.
            user_id: Optional user scope.

        Returns:
            List of tool log entries, most recent first.
        """
        try:
            # Use list_all so we hit the provider's relational TOOL_LOG branch
            # (retrieve_by_query with a string query falls through to the
            # empty default for TOOL_LOG and returns nothing). The set is
            # scoped per-thread so the client-side filter is cheap even for
            # agents with thousands of lifetime tool calls.
            all_rows = self.memory_provider.list_all(MemoryType.TOOL_LOG) or []
            filtered: List[Dict[str, Any]] = []
            for row in all_rows:
                if not isinstance(row, dict):
                    continue
                if memory_id and row.get("memory_id") != memory_id:
                    continue
                if user_id is not None and row.get("user_id") != user_id:
                    continue
                filtered.append(row)

            # Sort by timestamp descending so "recent" means recent.
            def _ts(row: Dict[str, Any]) -> str:
                return str(row.get("timestamp") or "")

            filtered.sort(key=_ts, reverse=True)
            if limit and limit > 0:
                filtered = filtered[:limit]
            return filtered
        except Exception as e:
            logger.error("Failed to list tool logs for %s: %s", memory_id, e)
            return []

    def load_summaries_for_thread(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load existing summary documents for a given thread.

        Returns compact summary references (summary_id + short description)
        that can be injected into the context window without consuming
        excessive tokens. The agent can expand any summary on demand.
        """
        try:
            from ...enums.memory_type import MemoryType

            documents = self.memory_provider.list_all(MemoryType.SUMMARIES) or []
        except Exception as exc:
            logger.debug("Failed to list summaries for thread %s: %s", memory_id, exc)
            return []

        normalized_memory_id = str(memory_id or "").strip()
        normalized_agent_id = str(agent_id or "").strip()
        filtered: List[Dict[str, Any]] = []

        for doc in documents:
            if not isinstance(doc, dict):
                continue

            # Tenant isolation: only include rows whose user_id exactly matches
            # the requested scope (including None == None for legacy rows).
            doc_user_id = doc.get("user_id")
            if doc_user_id != user_id:
                continue

            doc_memory_id = str(
                doc.get("memory_id") or doc.get("memoryId") or ""
            ).strip()
            doc_agent_id = str(doc.get("agent_id") or doc.get("agentId") or "").strip()

            # Match by memory_id or agent_id
            if normalized_memory_id and doc_memory_id != normalized_memory_id:
                if normalized_agent_id and doc_agent_id != normalized_agent_id:
                    continue
                elif not normalized_agent_id:
                    continue

            content = str(doc.get("content") or "").strip()
            summary_id = str(doc.get("id") or doc.get("_id") or "").strip()
            period_start = doc.get("period_start")
            period_end = doc.get("period_end")
            units_count = doc.get("memory_units_count", 0)
            message_ids = doc.get("source_message_ids") or []

            # Build a short description (first 200 chars of content)
            short_desc = content[:200] + ("..." if len(content) > 200 else "")

            filtered.append(
                {
                    "summary_id": summary_id,
                    "short_description": short_desc,
                    "period_start": period_start,
                    "period_end": period_end,
                    "memory_units_count": units_count,
                    "source_message_ids": message_ids,
                    "memory_id": doc_memory_id,
                }
            )

        # Sort by period_end descending (most recent first)
        def _sort_key(item):
            val = item.get("period_end") or item.get("period_start") or 0
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        filtered.sort(key=_sort_key, reverse=True)
        return filtered[:limit] if limit and limit > 0 else filtered

    def get_messages_by_ids(
        self,
        message_ids: List[str],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve specific conversation messages by their IDs.

        Used to reconstruct the original conversation from a summary.
        """
        results: List[Dict[str, Any]] = []
        for msg_id in message_ids:
            try:
                doc = self.memory_provider.retrieve_by_id(
                    msg_id, MemoryType.CONVERSATION_MEMORY
                )
                if doc and isinstance(doc, dict):
                    if doc.get("user_id") != user_id:
                        # Tenant isolation: skip rows belonging to another user.
                        continue
                    results.append(doc)
            except Exception as exc:
                logger.debug("Could not retrieve message %s: %s", msg_id, exc)
        return results

    def mark_messages_as_summarized(
        self, message_ids: List[str], summary_id: str
    ) -> int:
        """
        Mark conversation messages as summarized by setting their summary_id field.

        Summarized messages will be excluded from conversation history loading,
        preventing double-counting of information.

        Returns the number of messages successfully marked.
        """
        marked = 0
        for msg_id in message_ids:
            try:
                success = self.memory_provider.update_by_id(
                    msg_id,
                    {"summary_id": summary_id},
                    MemoryType.CONVERSATION_MEMORY,
                )
                if success:
                    marked += 1
            except Exception as exc:
                logger.debug("Could not mark message %s as summarized: %s", msg_id, exc)
        if marked:
            logger.info(
                "Marked %d/%d messages as summarized (summary_id=%s)",
                marked,
                len(message_ids),
                summary_id,
            )
        return marked

    def get_unsummarized_messages(
        self,
        memory_id: str,
        limit: int = 200,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages that have NOT been summarized yet.

        Returns messages without a summary_id field, ordered by timestamp.
        """
        try:
            history = self.load_conversation_history(
                memory_id, limit=limit, user_id=user_id
            )
            unsummarized = []
            for item in history:
                content = item.get("content") or item
                if isinstance(content, dict):
                    if content.get("summary_id"):
                        continue
                elif isinstance(item, dict) and item.get("summary_id"):
                    continue
                unsummarized.append(item)
            return unsummarized
        except Exception as exc:
            logger.error("Failed to get unsummarized messages: %s", exc)
            return []

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
