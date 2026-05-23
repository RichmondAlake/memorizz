# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Conversation flow management for MemAgent."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .prompt_handler import PromptHandler
from .response_handler import ResponseHandler

logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Handles conversation flow and management for MemAgent.

    This class orchestrates the conversation lifecycle, including
    context building, turn management, and conversation state tracking.
    """

    def __init__(self, memory_manager=None):
        """Initialize the conversation handler."""
        self.memory_manager = memory_manager
        self.prompt_handler = PromptHandler()
        self.response_handler = ResponseHandler()

        # Conversation state tracking
        self.active_conversations = {}
        self.conversation_metadata = {}

    def start_conversation(
        self,
        thread_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new conversation.

        Args:
            thread_id: Optional conversation ID
            memory_id: Optional memory ID for persistence
            initial_context: Optional initial context

        Returns:
            Conversation ID
        """
        try:
            import uuid

            conv_id = thread_id or str(uuid.uuid4())

            # Initialize conversation state
            self.active_conversations[conv_id] = {
                "memory_id": memory_id,
                "context": initial_context or {},
                "turn_count": 0,
                "start_time": datetime.now(),
                "last_activity": datetime.now(),
            }

            # Initialize metadata
            self.conversation_metadata[conv_id] = {
                "participant_count": 1,  # Just the user initially
                "topic_keywords": [],
                "conversation_type": "general",
            }

            logger.info(f"Started conversation {conv_id}")
            return conv_id

        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise

    def process_turn(
        self,
        query: str,
        thread_id: str,
        memory_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a conversation turn.

        Args:
            query: User's query
            thread_id: Conversation ID
            memory_id: Optional memory ID
            additional_context: Optional additional context
            user_id: Optional end-user identifier for multi-tenant scoping.

        Returns:
            Dictionary containing turn processing results
        """
        try:
            # Ensure conversation exists
            if thread_id not in self.active_conversations:
                self.start_conversation(thread_id, memory_id)

            conv_state = self.active_conversations[thread_id]

            # Update conversation state
            conv_state["turn_count"] += 1
            conv_state["last_activity"] = datetime.now()

            # Build context for this turn
            context = self._build_turn_context(
                query=query,
                thread_id=thread_id,
                memory_id=memory_id or conv_state.get("memory_id"),
                additional_context=additional_context,
                user_id=user_id,
            )

            # Update conversation metadata
            self._update_conversation_metadata(thread_id, query, context)

            # Return processing results
            return {
                "context": context,
                "turn_count": conv_state["turn_count"],
                "conversation_state": conv_state.copy(),
                "metadata": self.conversation_metadata.get(thread_id, {}),
            }

        except Exception as e:
            logger.error(f"Failed to process conversation turn: {e}")
            raise

    def end_conversation(self, thread_id: str) -> Dict[str, Any]:
        """
        End a conversation and return summary.

        Args:
            thread_id: Conversation ID to end

        Returns:
            Conversation summary
        """
        try:
            if thread_id not in self.active_conversations:
                logger.warning(f"Conversation {thread_id} not found")
                return {}

            conv_state = self.active_conversations[thread_id]
            metadata = self.conversation_metadata.get(thread_id, {})

            # Calculate conversation duration
            start_time = conv_state.get("start_time", datetime.now())
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Build summary
            summary = {
                "thread_id": thread_id,
                "turn_count": conv_state.get("turn_count", 0),
                "duration_seconds": duration,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metadata": metadata,
            }

            # Clean up
            del self.active_conversations[thread_id]
            if thread_id in self.conversation_metadata:
                del self.conversation_metadata[thread_id]

            logger.info(
                f"Ended conversation {thread_id} after {summary['turn_count']} turns"
            )
            return summary

        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return {}

    def get_conversation_history(
        self,
        thread_id: str,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            thread_id: Conversation ID
            limit: Maximum number of entries to return
            user_id: Optional user scope for multi-tenant apps.

        Returns:
            List of conversation history entries
        """
        try:
            if not self.memory_manager:
                logger.warning("No memory manager available for conversation history")
                return []

            conv_state = self.active_conversations.get(thread_id, {})
            memory_id = conv_state.get("memory_id")

            if memory_id:
                history = self.memory_manager.load_conversation_history(
                    memory_id, limit, user_id=user_id
                )
                return self.prompt_handler.format_conversation_history(history, limit)

            return []

        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    def _build_turn_context(
        self,
        query: str,
        thread_id: str,
        memory_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build context for the current turn."""
        try:
            context = {"query": query}

            # Add conversation history
            if memory_id and self.memory_manager:
                try:
                    history = self.memory_manager.load_conversation_history(
                        memory_id, limit=10, user_id=user_id
                    )
                    context["conversation_history"] = history

                    # Add relevant memories
                    from ...enums import MemoryType

                    relevant_memories = self.memory_manager.retrieve_relevant_memories(
                        query=query,
                        memory_type=MemoryType.CONVERSATION_MEMORY,
                        memory_id=memory_id,
                        limit=5,
                        user_id=user_id,
                    )
                    context["relevant_memories"] = relevant_memories

                except Exception as e:
                    logger.warning(f"Failed to load memory context: {e}")

            # Add conversation state
            conv_state = self.active_conversations.get(thread_id, {})
            context["turn_count"] = conv_state.get("turn_count", 0)
            context["conversation_duration"] = self._get_conversation_duration(
                thread_id
            )

            # Add conversation metadata
            metadata = self.conversation_metadata.get(thread_id, {})
            context["conversation_metadata"] = metadata

            # Merge additional context
            if additional_context:
                context.update(additional_context)

            return context

        except Exception as e:
            logger.error(f"Failed to build turn context: {e}")
            return {"query": query}  # Minimal fallback

    def _update_conversation_metadata(
        self, thread_id: str, query: str, context: Dict[str, Any]
    ):
        """Update conversation metadata based on the current turn."""
        try:
            if thread_id not in self.conversation_metadata:
                return

            metadata = self.conversation_metadata[thread_id]

            # Extract and update topic keywords
            query_words = [word.lower() for word in query.split() if len(word) > 3]
            existing_keywords = metadata.get("topic_keywords", [])

            # Add new keywords (simple approach - could be more sophisticated)
            new_keywords = [
                word for word in query_words if word not in existing_keywords
            ]
            metadata["topic_keywords"] = (
                existing_keywords + new_keywords[:5]
            )  # Limit growth

            # Update conversation type based on patterns
            if any(
                word in query.lower() for word in ["help", "how", "what", "explain"]
            ):
                metadata["conversation_type"] = "help_seeking"
            elif any(
                word in query.lower()
                for word in ["create", "make", "build", "generate"]
            ):
                metadata["conversation_type"] = "creative"
            elif any(
                word in query.lower() for word in ["analyze", "compare", "evaluate"]
            ):
                metadata["conversation_type"] = "analytical"

        except Exception as e:
            logger.warning(f"Failed to update conversation metadata: {e}")

    def _get_conversation_duration(self, thread_id: str) -> float:
        """Get conversation duration in seconds."""
        try:
            conv_state = self.active_conversations.get(thread_id, {})
            start_time = conv_state.get("start_time")

            if start_time:
                return (datetime.now() - start_time).total_seconds()

            return 0.0

        except Exception:
            return 0.0

    def get_active_conversations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active conversations.

        Returns:
            Dictionary of active conversations
        """
        return self.active_conversations.copy()

    def cleanup_inactive_conversations(self, max_inactive_hours: int = 24):
        """
        Clean up conversations that have been inactive for too long.

        Args:
            max_inactive_hours: Maximum hours of inactivity before cleanup
        """
        try:
            current_time = datetime.now()
            to_remove = []

            for conv_id, conv_state in self.active_conversations.items():
                last_activity = conv_state.get("last_activity", current_time)
                hours_inactive = (current_time - last_activity).total_seconds() / 3600

                if hours_inactive > max_inactive_hours:
                    to_remove.append(conv_id)

            # Remove inactive conversations
            for conv_id in to_remove:
                self.end_conversation(conv_id)
                logger.info(f"Cleaned up inactive conversation: {conv_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup inactive conversations: {e}")
