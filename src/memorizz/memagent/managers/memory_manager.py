# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Memory management functionality for MemAgent."""

import inspect
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...enums import MemoryType, Role
from ...long_term.episodic.conversational_memory_unit import ConversationMemoryUnit
from ...memory_provider import MemoryProvider
from ...memory_provider.base import _UNSET, filter_tool_log_rows

logger = logging.getLogger(__name__)


def _callable_accepts(fn: Any, name: str) -> bool:
    """True if ``fn`` accepts a keyword argument ``name`` (or **kwargs).

    Lets ``list_tool_logs`` hand the ``thread_id`` filter to a provider's native
    query only when that native actually understands it — older / third-party
    providers fall back to the shared in-memory scan instead of erroring.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        p.name == name or p.kind == inspect.Parameter.VAR_KEYWORD for p in params
    )


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

    @staticmethod
    def _entry_thread_id(entry: Any) -> str:
        """Read a thread id from provider rows and cached nested rows."""
        if not isinstance(entry, dict):
            return ""
        content = entry.get("content")
        nested = content if isinstance(content, dict) else {}
        return str(
            entry.get("thread_id")
            or entry.get("conversation_id")
            or nested.get("thread_id")
            or nested.get("conversation_id")
            or ""
        )

    @staticmethod
    def _entry_user_id(entry: Any) -> Optional[str]:
        """Read user scope from provider rows and cached nested rows."""
        if not isinstance(entry, dict):
            return None
        if "user_id" in entry:
            return entry.get("user_id")
        content = entry.get("content")
        if isinstance(content, dict):
            return content.get("user_id")
        return None

    @staticmethod
    def _entry_memory_id(entry: Any) -> str:
        """Read the owning memory id from flat and nested provider rows."""
        if not isinstance(entry, dict):
            return ""
        content = entry.get("content")
        nested = content if isinstance(content, dict) else {}
        return str(entry.get("memory_id") or nested.get("memory_id") or "")

    def load_conversation_history(
        self,
        memory_id: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load conversation history for a given memory ID.

        Args:
            memory_id: The memory ID to load history for.
            limit: Maximum number of conversations to retrieve.
            user_id: Optional user scope. When set, history is restricted to
                entries with a matching ``user_id``. When ``None``, only
                legacy/unscoped entries are returned.
            thread_id: Optional exact conversation thread. When omitted,
                history from every thread in the memory scope is returned.

        Returns:
            List of conversation history entries.
        """
        try:
            logger.debug(f"Loading conversation history for memory_id: {memory_id}")

            thread_scope = str(thread_id) if thread_id is not None else None
            cache_key = (
                (memory_id, user_id, thread_scope)
                if thread_scope is not None
                else (memory_id, user_id)
            )
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
            provider_retrieve = (
                self.memory_provider.retrieve_conversation_history_ordered_by_timestamp
            )
            provider_filters_thread = thread_id is None or _callable_accepts(
                provider_retrieve, "thread_id"
            )
            provider_filters_user = _callable_accepts(provider_retrieve, "user_id")
            fetch_limit = None
            if (
                limit
                and limit > 0
                and provider_filters_thread
                and provider_filters_user
            ):
                fetch_limit = int(limit)
            retrieve_kwargs: Dict[str, Any] = {
                "memory_id": memory_id,
                "memory_type": MemoryType.CONVERSATION_MEMORY,
                "limit": fetch_limit,
            }
            if provider_filters_user:
                retrieve_kwargs["user_id"] = user_id
            if thread_id is not None and provider_filters_thread:
                retrieve_kwargs["thread_id"] = str(thread_id)
            history = provider_retrieve(**retrieve_kwargs)
            history = [row for row in (history or []) if isinstance(row, dict)]
            if not provider_filters_user:
                history = [
                    row for row in history if self._entry_user_id(row) == user_id
                ]
            # Third-party providers may not expose a native thread filter.
            # Fetch their full memory scope above, then apply the same exact
            # match here before limiting.
            if thread_id is not None:
                wanted_thread_id = str(thread_id)
                history = [
                    row
                    for row in history
                    if self._entry_thread_id(row) == wanted_thread_id
                ]
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
        self, memory_unit: ConversationMemoryUnit, memory_id: str
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
            unit_thread_id = str(getattr(memory_unit, "thread_id", None) or "")
            matching_cache_keys = []
            for key in self._conversation_memory_cache:
                if not isinstance(key, tuple) or len(key) < 2:
                    continue
                if key[0] != memory_id or key[1] != unit_user_id:
                    continue
                cached_thread_id = key[2] if len(key) >= 3 else None
                if cached_thread_id is None or str(cached_thread_id) == unit_thread_id:
                    matching_cache_keys.append(key)

            if hasattr(memory_unit, "role") and hasattr(memory_unit, "thread_id"):
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
                for cache_key in matching_cache_keys:
                    try:
                        cached = self._conversation_memory_cache.get(cache_key)
                        if cached is None:
                            continue
                        cached.append(entry)
                        if (
                            self._conversation_memory_cache_max_entries
                            and len(cached)
                            > self._conversation_memory_cache_max_entries
                        ):
                            self._conversation_memory_cache[cache_key] = cached[
                                -self._conversation_memory_cache_max_entries :
                            ]
                    except Exception:
                        self._conversation_memory_cache.pop(cache_key, None)
            else:
                # Unknown memory unit shape; safest to invalidate matching slots.
                for cache_key in matching_cache_keys:
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
        include_embedding: bool = False,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
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
            include_embedding: When True (and the provider supports it),
                results carry their stored embedding vectors so callers can
                run similarity dedup/MMR without re-embedding anything.
            thread_id: Optional exact thread boundary for episodic recall.
            namespace: Optional exact namespace boundary for knowledge recall.

        Returns:
            List of relevant memory entries. Always a list — providers that
            return ``None`` (legacy "no results" convention) or a PyMongo
            cursor are coerced here so callers can rely on the shape.
        """
        try:
            kwargs: Dict[str, Any] = {
                "query": query,
                "memory_id": memory_id,
                "memory_type": memory_type,
                "limit": limit,
                "user_id": user_id,
            }
            if thread_id is not None:
                kwargs["thread_id"] = str(thread_id)
            if namespace is not None:
                kwargs["namespace"] = str(namespace)
            # Only forward include_embedding to providers that understand it;
            # older/third-party providers keep their default projection.
            if include_embedding and _callable_accepts(
                self.memory_provider.retrieve_by_query, "include_embedding"
            ):
                kwargs["include_embedding"] = True
            results = self.memory_provider.retrieve_by_query(**kwargs)

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
                    # Any other non-iterable value violates the provider
                    # contract and must degrade to no evidence. Treating a
                    # truthy sentinel (for example a client mock) as a memory
                    # row leaks arbitrary attributes into provenance fields.
                    results = [results] if isinstance(results, dict) else []

            # ``**kwargs`` support does not prove that a third-party provider
            # applied those filters.  Semantic recall is a prompt-injection
            # boundary, so enforce the authenticated memory and user scope a
            # second time before any row can reach the model.
            wanted_memory_id = str(memory_id or "")
            results = [
                row
                for row in results
                if isinstance(row, dict)
                and self._entry_memory_id(row) == wanted_memory_id
                and self._entry_user_id(row) == user_id
            ]

            # Providers may accept **kwargs yet not push every scope into their
            # native query. Enforce exact boundaries again in the manager so a
            # third-party backend cannot silently broaden automatic recall.
            if thread_id is not None:
                wanted_thread = str(thread_id)
                results = [
                    row
                    for row in results
                    if isinstance(row, dict)
                    and self._entry_thread_id(row) == wanted_thread
                ]
            if namespace is not None:
                wanted_namespace = str(namespace)

                def _entry_namespace(row: Dict[str, Any]) -> str:
                    content = row.get("content")
                    nested = content if isinstance(content, dict) else {}
                    return str(row.get("namespace") or nested.get("namespace") or "")

                results = [
                    row
                    for row in results
                    if isinstance(row, dict)
                    and _entry_namespace(row) == wanted_namespace
                ]

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

    def is_duplicate_of_recent(
        self,
        memory_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        lookback: int = 4,
    ) -> bool:
        """True when an identical (role, content) row was just stored.

        Write-path dedup guard: repeated identical queries (semantic-cache
        hits, client retries) would otherwise accumulate duplicate
        conversation rows that later bloat the history window. Only the
        cached tail is inspected — this is a cheap same-session guard, not a
        full-history scan.
        """
        if not content or not str(content).strip():
            return False
        thread_scope = str(thread_id) if thread_id is not None else None
        cached = self._conversation_memory_cache.get((memory_id, user_id, thread_scope))
        if not cached:
            cached = self._conversation_memory_cache.get((memory_id, user_id, None))
        if not cached:
            # Backward-compatible cache shape from before thread scoping.
            cached = self._conversation_memory_cache.get((memory_id, user_id))
        if not cached:
            return False
        target_role = str(role or "").strip().lower()
        target_content = str(content).strip()
        for entry in reversed(cached[-max(lookback, 1) :]):
            if not isinstance(entry, dict):
                continue
            nested = entry.get("content")
            nested = nested if isinstance(nested, dict) else entry
            if thread_id and str(nested.get("thread_id") or "") != str(thread_id):
                continue
            entry_role = str(nested.get("role") or "").strip().lower()
            entry_content = str(nested.get("content") or "").strip()
            if entry_role == target_role and entry_content == target_content:
                return True
        return False

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
                keys_to_remove = [
                    key
                    for key in self._conversation_memory_cache
                    if (
                        isinstance(key, tuple)
                        and len(key) >= 2
                        and key[0] == memory_id
                        and key[1] == user_id
                    )
                ]
                for key in keys_to_remove:
                    self._conversation_memory_cache.pop(key, None)
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
        outcome: Optional[str] = None,
        outcome_details: Optional[Dict[str, Any]] = None,
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
            outcome: Structured terminal outcome such as ``fallback`` or
                ``provider_error``.
            outcome_details: Content-free provider, reason, and result-count
                metadata for the structured outcome.
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
                "outcome": outcome or ("success" if success else "error"),
                "outcome_details": dict(outcome_details or {}),
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
        user_id: Any = _UNSET,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a tool log entry by its ID.

        Args:
            tool_log_id: The ID of the tool log to retrieve.
            user_id: Optional user scope. Omit for an administrative/unscoped
                read; pass ``None`` for anonymous rows or a tenant ID for an
                exact scoped read.

        Returns:
            The tool log entry dict, or None if not found.
        """
        try:
            result = self.memory_provider.retrieve_by_id(
                tool_log_id, MemoryType.TOOL_LOG
            )
            if result and isinstance(result, dict) and user_id is not _UNSET:
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
        user_id: Any = _UNSET,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List recent tool log entries for a given memory ID.

        Args:
            memory_id: The memory ID to list logs for.
            limit: Maximum number of entries to return.
            user_id: User scope. Omit for an administrative/unscoped read,
                pass ``None`` for anonymous/legacy rows, or pass a tenant ID
                for an exact scoped read.
            thread_id: Optional thread scope. When set, only tool logs recorded
                in that conversation thread are returned — this is what keeps the
                system-prompt tool-log digest from leaking another thread's tool
                ids or letting one fall off a global last-N window. ``None``
                preserves the legacy all-threads behaviour.

        Returns:
            List of tool log entries, most recent first.
        """
        try:
            # Prefer the provider's native scoped query when it ships one
            # (MongoDBProvider.list_tool_logs uses an indexed find + sort +
            # limit aggregate). Falls back to the shared in-memory scan for
            # providers without a tuned implementation so MemoryManager stays
            # provider-agnostic. ``hasattr`` rather than isinstance avoids
            # pulling the MongoDB provider import path here.
            native = getattr(self.memory_provider, "list_tool_logs", None)
            # Only delegate the thread filter to the native query when that
            # native actually accepts it; an older / third-party provider that
            # predates thread scoping is routed to the fallback so the filter is
            # still applied (correctness over the indexed fast path).
            if callable(native) and (
                thread_id is None or _callable_accepts(native, "thread_id")
            ):
                kwargs: Dict[str, Any] = {
                    "memory_id": memory_id,
                    "limit": limit,
                }
                if user_id is not _UNSET:
                    kwargs["user_id"] = user_id
                if thread_id is not None:
                    kwargs["thread_id"] = thread_id
                rows = native(**kwargs) or []
                return [r for r in rows if isinstance(r, dict)]

            # Fallback: read every row + filter/sort/limit in Python via the
            # shared helper. O(n) in the provider's tool_log size but correct
            # for any backend (and identical scoping rules to the natives).
            all_rows = self.memory_provider.list_all(MemoryType.TOOL_LOG) or []
            return filter_tool_log_rows(
                all_rows,
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                limit=limit,
            )
        except Exception as e:
            logger.error("Failed to list tool logs for %s: %s", memory_id, e)
            return []

    def load_summaries_for_thread(
        self,
        memory_id: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load existing summary documents for a given thread.

        Returns compact summary references (summary_id + short description)
        that can be injected into the context window without consuming
        excessive tokens. The agent can expand any summary on demand.
        """
        try:
            from ...enums.memory_type import MemoryType

            # Prefer a provider-native scoped query (indexed find + sort +
            # limit) over reading the whole summaries collection and
            # filtering in Python. Fallback keeps third-party providers
            # working unchanged.
            native = getattr(self.memory_provider, "list_summaries", None)
            native_is_exact = (
                callable(native)
                and _callable_accepts(native, "user_id")
                and (thread_id is None or _callable_accepts(native, "thread_id"))
            )
            if native_is_exact:
                kwargs: Dict[str, Any] = {
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "limit": max(limit, 1) * 3,  # headroom for python-side OR-match
                }
                if _callable_accepts(native, "user_id"):
                    kwargs["user_id"] = user_id
                if thread_id is not None and _callable_accepts(native, "thread_id"):
                    kwargs["thread_id"] = str(thread_id)
                documents = native(**kwargs) or []
            else:
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
            doc_thread_id = self._entry_thread_id(doc)

            if thread_id is not None and doc_thread_id != str(thread_id):
                continue

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
            message_ids = (
                doc.get("source_message_ids") or doc.get("original_memory_ids") or []
            )

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
                    "thread_id": doc_thread_id or None,
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
        thread_id: Optional[str] = None,
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
                    if thread_id is not None and self._entry_thread_id(doc) != str(
                        thread_id
                    ):
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
        thread_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation messages that have NOT been summarized yet.

        Returns messages without a summary_id field, ordered by timestamp.
        """
        try:
            history = self.load_conversation_history(
                memory_id,
                limit=limit,
                user_id=user_id,
                thread_id=thread_id,
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

    def generate_summaries(
        self,
        *,
        model: Any,
        agent_id: Optional[str],
        memory_ids: Optional[List[str]],
        current_memory_id: Optional[str],
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        days_back: int = 7,
        max_memories_per_summary: int = 50,
        record_context_usage: Optional[Any] = None,
    ) -> List[str]:
        """
        Generate summaries by compressing memory units from a specified time period.

        Implementation moved verbatim from ``MemAgent.generate_summaries`` —
        the agent's collaborators (LLM model, agent/memory ids, context-usage
        recorder) are threaded through explicitly.

        Returns:
        --------
        List[str]
            List of summary IDs that were created
        """
        try:
            import time

            from ...embeddings import get_embedding

            # Calculate time range (days_back days ago to now)
            current_time = time.time()
            start_time = current_time - (days_back * 24 * 60 * 60)

            logger.info(
                f"Generating summaries for agent {agent_id} from {days_back} days back"
            )
            logger.info(f"Agent memory_ids: {memory_ids}")
            logger.info(f"Current memory_id: {current_memory_id}")
            logger.info(f"Time range: {start_time} to {current_time}")

            # Ensure we have memory IDs to search
            memory_ids_to_search = memory_ids or []
            if current_memory_id and current_memory_id not in memory_ids_to_search:
                memory_ids_to_search = [current_memory_id] + memory_ids_to_search

            if not memory_ids_to_search:
                logger.warning(
                    f"Agent {agent_id} has no memory_ids to search for summaries"
                )
                return []

            logger.info(
                f"Searching {len(memory_ids_to_search)} memory_ids: {memory_ids_to_search}"
            )

            # Collect conversation memories from all memory IDs
            all_memories = []
            for memory_id in memory_ids_to_search:
                logger.info(
                    f"Retrieving conversation history for memory_id: {memory_id}"
                )
                try:
                    # Retrieve all conversation history
                    retrieve = (
                        self.memory_provider.retrieve_conversation_history_ordered_by_timestamp
                    )
                    retrieve_kwargs: Dict[str, Any] = {"memory_id": memory_id}
                    if _callable_accepts(retrieve, "include_embedding"):
                        retrieve_kwargs["include_embedding"] = False
                    if _callable_accepts(retrieve, "user_id"):
                        # Summarization always uses an explicit tenant scope;
                        # ``None`` means the anonymous/legacy tenant.
                        retrieve_kwargs["user_id"] = user_id
                    if thread_id is not None and _callable_accepts(
                        retrieve, "thread_id"
                    ):
                        retrieve_kwargs["thread_id"] = thread_id
                    memories = retrieve(**retrieve_kwargs)
                    memories = [
                        memory
                        for memory in (memories or [])
                        if isinstance(memory, dict)
                        and memory.get("user_id") == user_id
                        and (
                            thread_id is None
                            or self._entry_thread_id(memory) == str(thread_id)
                        )
                    ]

                    if memories:
                        logger.info(
                            f"Retrieved {len(memories)} raw memories for memory_id: {memory_id}"
                        )

                        # Filter by time range
                        filtered = []
                        for idx, mem in enumerate(memories):
                            mem_timestamp = mem.get("timestamp")
                            original_timestamp = mem_timestamp

                            # Convert timestamp to float if needed
                            if isinstance(mem_timestamp, str):
                                try:
                                    from datetime import datetime

                                    # Try multiple timestamp formats
                                    if "T" in mem_timestamp:
                                        # ISO format
                                        mem_timestamp = datetime.fromisoformat(
                                            mem_timestamp.replace("Z", "+00:00")
                                        ).timestamp()
                                    else:
                                        # Try parsing as float string
                                        mem_timestamp = float(mem_timestamp)
                                except Exception as e:
                                    logger.warning(
                                        f"Could not parse timestamp '{original_timestamp}' at index {idx}: {e}"
                                    )
                                    continue
                            elif hasattr(mem_timestamp, "timestamp"):
                                # datetime object
                                mem_timestamp = mem_timestamp.timestamp()
                            elif not isinstance(mem_timestamp, (int, float)):
                                logger.warning(
                                    f"Unknown timestamp type at index {idx}: {type(mem_timestamp)} = {original_timestamp}"
                                )
                                continue

                            # Convert to float
                            mem_timestamp = float(mem_timestamp)

                            # Debug first few timestamps
                            if idx < 3:
                                logger.info(
                                    f"Memory {idx}: timestamp={mem_timestamp}, start_time={start_time}, current_time={current_time}, in_range={start_time <= mem_timestamp <= current_time}"
                                )

                            if (
                                start_time <= mem_timestamp <= current_time
                                and not self._is_summarized_message(mem)
                            ):
                                filtered.append(mem)

                        logger.info(
                            f"Found {len(filtered)} memories within time range (out of {len(memories)} total) for memory_id: {memory_id}"
                        )
                        all_memories.extend(filtered)
                    else:
                        logger.info(f"No memories returned for memory_id: {memory_id}")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve memories for memory_id {memory_id}: {e}"
                    )
                    import traceback

                    logger.debug(traceback.format_exc())

            if not all_memories:
                logger.info(
                    f"No memories found for agent {agent_id} in the specified time range"
                )
                return []

            # Sort memories by timestamp
            def get_timestamp(mem):
                ts = mem.get("timestamp", 0)
                if isinstance(ts, str):
                    try:
                        from datetime import datetime

                        return datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        ).timestamp()
                    except (ValueError, Exception):
                        return 0
                return float(ts) if isinstance(ts, (int, float)) else 0

            all_memories.sort(key=get_timestamp)

            logger.info(f"Found {len(all_memories)} memory units to summarize")

            # Keep each summary inside one memory/thread/tenant boundary.
            grouped: Dict[tuple, List[Dict[str, Any]]] = {}
            for memory in all_memories:
                key = (
                    memory.get("memory_id"),
                    self._entry_thread_id(memory),
                    memory.get("user_id"),
                )
                grouped.setdefault(key, []).append(memory)
            chunks: List[List[Dict[str, Any]]] = []
            for group in grouped.values():
                group.sort(key=get_timestamp)
                for offset in range(0, len(group), max_memories_per_summary):
                    chunks.append(group[offset : offset + max_memories_per_summary])
            chunks.sort(key=lambda chunk: get_timestamp(chunk[0]))

            # Split memories into chunks and create summaries
            summary_ids = []
            for memory_chunk in chunks:
                # Generate summary for this chunk
                summary_content = self.compress_memories_with_llm(
                    memory_chunk,
                    model=model,
                    record_context_usage=record_context_usage,
                )

                if summary_content:
                    # Get timestamps for period
                    period_start = get_timestamp(memory_chunk[0])
                    period_end = get_timestamp(memory_chunk[-1])

                    # Get the memory_id from the first memory in the chunk
                    chunk_memory_id = memory_chunk[0].get("memory_id")
                    if not chunk_memory_id:
                        # Fallback to current memory_id or first in list
                        chunk_memory_id = current_memory_id or (
                            memory_ids[0] if memory_ids else "default"
                        )

                    # Collect source message IDs for back-reference
                    source_message_ids = []
                    for mem in memory_chunk:
                        msg_id = (
                            mem.get("id") or mem.get("_id") or mem.get("memory_unit_id")
                        )
                        if msg_id:
                            source_message_ids.append(str(msg_id))

                    # Create summary document with source references
                    summary_doc = {
                        "memory_id": chunk_memory_id,
                        "agent_id": agent_id,
                        "user_id": memory_chunk[0].get("user_id"),
                        "thread_id": self._entry_thread_id(memory_chunk[0]) or None,
                        "content": summary_content,
                        "period_start": period_start,
                        "period_end": period_end,
                        "memory_units_count": len(memory_chunk),
                        "source_message_ids": source_message_ids,
                        "summary_type": "automatic",
                        "created_at": current_time,
                        "embedding": get_embedding(summary_content),
                    }

                    # Store summary
                    atomic_store = getattr(
                        self.memory_provider, "store_summary_with_links", None
                    )
                    if callable(atomic_store):
                        summary_id = atomic_store(summary_doc)
                    else:
                        summary_id = self.memory_provider.store(
                            summary_doc, MemoryType.SUMMARIES
                        )
                    summary_ids.append(summary_id)

                    # Mark original messages as summarized so they are
                    # excluded from conversation history on future loads
                    if source_message_ids and not callable(atomic_store):
                        try:
                            self.mark_messages_as_summarized(
                                source_message_ids, summary_id
                            )
                            # Clear conversation cache so next load reflects changes
                            self.clear_conversation_cache(chunk_memory_id)
                        except Exception as mark_exc:
                            logger.debug(
                                "Could not mark messages as summarized: %s",
                                mark_exc,
                            )

                    logger.info(
                        f"Created summary {summary_id} for memory_id {chunk_memory_id} covering {len(memory_chunk)} memories"
                    )

            logger.info(f"Generated {len(summary_ids)} summaries for agent {agent_id}")
            return summary_ids

        except Exception as e:
            logger.error(f"Error generating summaries: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def compress_memories_with_llm(
        self,
        memories: List[Dict],
        *,
        model: Any,
        record_context_usage: Optional[Any] = None,
    ) -> str:
        """
        Use LLM to compress memory units into an emotionally and situationally relevant summary.

        Implementation moved verbatim from ``MemAgent._compress_memories_with_llm``.

        Parameters:
        -----------
        memories : List[Dict]
            List of memory units to compress
        model : Any
            The LLM provider used to generate the summary
        record_context_usage : Optional[Any]
            Callback invoked as ``record_context_usage(stage=...)`` after a
            successful generation (the agent's context-window tracker).

        Returns:
        --------
        str
            Compressed summary content
        """
        try:
            # Extract content from memories
            memory_contents = []
            for memory in memories:
                content = memory.get("content", "")
                role = memory.get("role", "")

                if content:
                    if role:
                        memory_contents.append(f"[{role}]: {content}")
                    else:
                        memory_contents.append(content)

            if not memory_contents:
                return ""

            # Create compression prompt
            memories_text = "\n".join(memory_contents)
            compression_prompt = f"""
Analyze the following memory units and create a concise summary that captures:
1. Emotionally significant moments and interactions
2. Situationally relevant context and patterns
3. Key achievements, challenges, or learning experiences
4. Important facts and information learned

Memory Units:
{memories_text}

Provide a comprehensive but concise summary:"""

            # Use the LLM to generate the summary
            if model:
                messages = [{"role": "user", "content": compression_prompt}]
                summary = model.generate(messages)
                if record_context_usage:
                    record_context_usage(stage="memory_compression")
                return summary.strip()
            else:
                logger.warning("No LLM model available for memory compression")
                return ""

        except Exception as e:
            logger.error(f"Error compressing memories with LLM: {e}")
            return ""
