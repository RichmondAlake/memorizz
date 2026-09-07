# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...enums.memory_type import MemoryType
from ...memory_provider import MemoryProvider
from .messages import (
    SharedMemoryMessage,
    SharedMemoryMessageType,
    create_command_message,
    create_report_message,
    create_status_message,
)

logger = logging.getLogger(__name__)


class BlackboardEntry:
    """Individual entry in the shared memory blackboard."""

    def __init__(
        self, agent_id: str, content: Any, entry_type: str, created_at: datetime = None
    ):
        self.agent_id = agent_id
        self.content = content
        self.entry_type = (
            entry_type  # "tool_call", "conversation", "task_assignment", "result"
        )
        self.created_at = created_at or datetime.now()
        self.memory_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            # Ensure downstream JSON serialization never sees provider-specific objects
            "content": SharedMemory._sanitize_payload(self.content),
            "entry_type": self.entry_type,
            "created_at": self.created_at.isoformat(),
        }


class SharedMemory:
    """Shared memory system for multi-agent coordination."""

    def __init__(self, memory_provider: MemoryProvider):
        self.memory_provider = memory_provider

    def create_shared_session(
        self,
        root_agent_id: str,
        delegate_agent_ids: List[str] = None,
        *,
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Create a new shared memory session for multi-agent coordination.

        Parameters:
            root_agent_id (str): The ID of the root/orchestrating agent
            delegate_agent_ids (List[str]): List of delegate agent IDs

        Returns:
            str: The memory ID for the shared session
        """
        payload = {
            "version": 1,
            "root_agent_id": root_agent_id,
            "delegate_agent_ids": delegate_agent_ids or [],
            "sub_agent_ids": [],
            "blackboard": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",  # active, completed, failed
            "workflow_id": workflow_id,
            "user_id": user_id,
            "trace_id": trace_id,
        }

        shared_session = {
            "memory_id": str(uuid.uuid4()),
            "content": json.dumps(payload),
            "owner_agent_id": root_agent_id,
            # Keep this inside the portable provider vocabulary. Exact workflow
            # and tenant isolation is carried in the versioned payload below.
            "scope": "private",
            "user_id": user_id,
            "memory_type": MemoryType.SHARED_MEMORY.value,
        }

        memory_id = self.memory_provider.store(shared_session, MemoryType.SHARED_MEMORY)
        if not memory_id:
            raise RuntimeError("Shared memory provider did not return a session ID")
        return str(memory_id)

    @staticmethod
    def _decode_payload(session: Dict[str, Any]) -> Dict[str, Any]:
        """Decode JSON payload stored in the content field."""
        # Sanitize session dict first to handle any JsonId objects in session fields
        session = SharedMemory._sanitize_payload(session)
        if not isinstance(session, dict):
            session = {}

        content = session.get("content")
        payload = None

        try:
            if content is not None:
                # Handle JsonId objects in content field
                content = SharedMemory._sanitize_payload(content)

                if hasattr(content, "as_json"):
                    payload = json.loads(content.as_json())
                elif hasattr(content, "as_dict"):
                    payload = content.as_dict()
                elif hasattr(content, "value"):
                    payload = content.value
        except Exception as exc:
            logger.warning("Failed to decode shared memory JSON payload: %s", exc)

        if payload is None:
            if isinstance(content, dict):
                payload = dict(content)
            elif isinstance(content, str) and content:
                try:
                    payload = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode shared memory content string.")

        if not isinstance(payload, dict):
            # Sanitize session fields before using them in fallback payload
            payload = {
                "root_agent_id": SharedMemory._sanitize_payload(
                    session.get("root_agent_id")
                ),
                "delegate_agent_ids": SharedMemory._sanitize_payload(
                    session.get("delegate_agent_ids", [])
                ),
                "sub_agent_ids": SharedMemory._sanitize_payload(
                    session.get("sub_agent_ids", [])
                ),
                "blackboard": SharedMemory._sanitize_payload(
                    session.get("blackboard", [])
                ),
                "created_at": SharedMemory._sanitize_payload(session.get("created_at")),
                "updated_at": SharedMemory._sanitize_payload(session.get("updated_at")),
                "status": SharedMemory._sanitize_payload(
                    session.get("status", "active")
                ),
            }

        payload.setdefault("blackboard", [])
        payload.setdefault("delegate_agent_ids", [])
        payload.setdefault("sub_agent_ids", [])
        payload.setdefault("status", "active")

        payload = SharedMemory._sanitize_payload(payload)
        if not isinstance(payload, dict):
            # Sanitize session fields again in final fallback
            payload = {
                "root_agent_id": SharedMemory._sanitize_payload(
                    session.get("root_agent_id")
                ),
                "delegate_agent_ids": SharedMemory._sanitize_payload(
                    session.get("delegate_agent_ids", [])
                ),
                "sub_agent_ids": SharedMemory._sanitize_payload(
                    session.get("sub_agent_ids", [])
                ),
                "blackboard": [],
                "created_at": SharedMemory._sanitize_payload(session.get("created_at")),
                "updated_at": SharedMemory._sanitize_payload(session.get("updated_at")),
                "status": SharedMemory._sanitize_payload(
                    session.get("status", "active")
                ),
            }
        return payload

    def _mutate_payload(
        self, memory_id: str, mutate, *, max_attempts: int = 16
    ) -> bool:
        """Retry only conflicting snapshots, never an unconditional replacement.

        The mutation is reapplied to a fresh snapshot on each attempt. Entry IDs
        must be allocated outside it so retries cannot duplicate an append.
        Unsupported providers explicitly fail instead of losing concurrent data.
        """
        try:
            compare_and_swap = getattr(
                self.memory_provider, "compare_and_swap_shared_memory", None
            )
            if not callable(compare_and_swap):
                raise NotImplementedError(
                    "Provider does not support atomic shared-memory updates"
                )
            for attempt in range(max_attempts):
                session = self.memory_provider.retrieve_by_id(
                    memory_id, MemoryType.SHARED_MEMORY
                )
                if not session:
                    return False
                expected_content = session.get("content")
                payload = self._decode_payload(session)
                if mutate(payload) is False:
                    return True  # already applied, with the same idempotency key
                payload["updated_at"] = datetime.now().isoformat()
                payload["revision"] = int(payload.get("revision", 0)) + 1
                serialized = json.dumps(
                    self._sanitize_payload(self._build_storage_payload(payload)),
                    default=str,
                )
                if compare_and_swap(memory_id, expected_content, serialized):
                    return True
                time.sleep(min(0.002 * (attempt + 1), 0.025))
            raise RuntimeError("Shared-memory update conflict retry budget exhausted")
        except Exception as exc:
            logger.error(f"Failed to persist shared memory payload: {exc}")
            return False

    @staticmethod
    def _build_storage_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return the minimal payload we need to persist."""
        return {
            "version": payload.get("version", 1),
            "root_agent_id": payload.get("root_agent_id"),
            "delegate_agent_ids": payload.get("delegate_agent_ids", []),
            "sub_agent_ids": payload.get("sub_agent_ids", []),
            "blackboard": payload.get("blackboard", []),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "status": payload.get("status", "active"),
            "workflow_id": payload.get("workflow_id"),
            "user_id": payload.get("user_id"),
            "trace_id": payload.get("trace_id"),
            "revision": payload.get("revision", 0),
            "outcome": payload.get("outcome"),
            "counts": payload.get("counts", {}),
        }

    @staticmethod
    def _sanitize_payload(value: Any) -> Any:
        """Recursively convert payload objects into JSON-serializable structures."""
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Handle Oracle JsonId objects (from oracledb Duality Views)
        # Check both by isinstance and by type name for robustness
        try:
            from oracledb import JsonId

            if isinstance(value, JsonId):
                return str(value)
        except (ImportError, AttributeError):
            pass
        # Also check by type name in case isinstance fails
        # Check both the type name and class name
        type_str = str(type(value))
        class_name = getattr(value, "__class__", None)
        class_name_str = str(class_name) if class_name else ""
        if "JsonId" in type_str or "JsonId" in class_name_str:
            try:
                return str(value)
            except Exception:
                pass
        if hasattr(value, "as_json"):
            try:
                return json.loads(value.as_json())
            except Exception:
                return value.as_json()
        if hasattr(value, "as_dict"):
            return SharedMemory._sanitize_payload(value.as_dict())
        if hasattr(value, "value"):
            return SharedMemory._sanitize_payload(value.value)
        if isinstance(value, dict):
            return {k: SharedMemory._sanitize_payload(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SharedMemory._sanitize_payload(item) for item in value]
        return str(value)

    def add_blackboard_entry(
        self,
        memory_id: str,
        agent_id: str,
        content: Any,
        entry_type: str,
        *,
        entry_id: Optional[str] = None,
    ) -> bool:
        """
        Add an entry to the shared blackboard.

        Parameters:
            memory_id (str): The shared memory ID
            agent_id (str): The ID of the agent adding the entry
            content (Any): The content to add
            entry_type (str): Type of entry (tool_call, conversation, etc.)

        Returns:
            bool: Success status
        """
        entry = BlackboardEntry(agent_id, content, entry_type)
        if entry_id is not None:
            entry.memory_id = str(entry_id)
        value = entry.to_dict()

        def append(payload):
            entries = payload.setdefault("blackboard", [])
            for existing in entries:
                if existing.get("memory_id") == entry.memory_id:
                    if any(
                        existing.get(key) != value[key]
                        for key in ("agent_id", "content", "entry_type")
                    ):
                        raise ValueError(
                            "Shared-memory entry ID reused with different content"
                        )
                    return False
            entries.append(value)

        return self._mutate_payload(memory_id, append)

    # Structured message helpers -------------------------------------------------
    def post_message(
        self, memory_id: str, agent_id: str, message: SharedMemoryMessage
    ) -> bool:
        """Persist a typed shared-memory message."""
        return self.add_blackboard_entry(
            memory_id=memory_id,
            agent_id=agent_id,
            content=message.to_dict(),
            entry_type=message.message_type.value,
        )

    def post_command(
        self,
        memory_id: str,
        agent_id: str,
        command_id: str,
        target_agent_id: str,
        instructions: str,
        priority: int = 3,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish a COMMAND message."""
        message = create_command_message(
            command_id=command_id,
            target_agent_id=target_agent_id,
            instructions=instructions,
            priority=priority,
            dependencies=dependencies,
            metadata=metadata,
        )
        return self.post_message(memory_id, agent_id, message)

    def post_status(
        self,
        memory_id: str,
        agent_id: str,
        command_id: str,
        status: str,
        progress: int,
        blockers: Optional[str] = None,
        summary_ids: Optional[List[str]] = None,
    ) -> bool:
        """Publish a STATUS update."""
        message = create_status_message(
            command_id=command_id,
            agent_id=agent_id,
            status=status,
            progress=progress,
            blockers=blockers,
            summary_ids=summary_ids,
        )
        return self.post_message(memory_id, agent_id, message)

    def post_report(
        self,
        memory_id: str,
        agent_id: str,
        command_id: str,
        findings: str,
        citations: Optional[List[str]] = None,
        gaps: Optional[List[str]] = None,
        summary_ids: Optional[List[str]] = None,
    ) -> bool:
        """Publish a REPORT message."""
        message = create_report_message(
            command_id=command_id,
            agent_id=agent_id,
            findings=findings,
            citations=citations,
            gaps=gaps,
            summary_ids=summary_ids,
        )
        return self.post_message(memory_id, agent_id, message)

    def get_blackboard_entries(
        self, memory_id: str, agent_id: str = None, entry_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve blackboard entries with optional filtering.

        Parameters:
            memory_id (str): The shared memory ID
            agent_id (str, optional): Filter by agent ID
            entry_type (str, optional): Filter by entry type

        Returns:
            List[Dict[str, Any]]: List of blackboard entries
        """
        try:
            session = self.memory_provider.retrieve_by_id(
                memory_id, MemoryType.SHARED_MEMORY
            )
            if not session:
                return []

            payload = self._decode_payload(session)
            entries = payload.get("blackboard", [])

            # Apply filters
            if agent_id:
                entries = [e for e in entries if e.get("agent_id") == agent_id]
            if entry_type:
                entries = [e for e in entries if e.get("entry_type") == entry_type]

            return entries

        except Exception as e:
            logger.error(f"Error retrieving blackboard entries: {e}")
            return []

    def list_messages(
        self,
        memory_id: str,
        message_type: SharedMemoryMessageType,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return structured messages filtered by type (and optional agent)."""
        entries = self.get_blackboard_entries(
            memory_id=memory_id, agent_id=agent_id, entry_type=message_type.value
        )
        results = []
        for entry in entries:
            content = entry.get("content", {})
            if isinstance(content, dict) and content.get("message_type"):
                results.append(content)
        return results

    def update_session_status(
        self, memory_id: str, status: str, *, outcome=None, counts=None
    ) -> bool:
        """Update the status of a shared session."""

        def update(payload):
            payload["status"] = status
            if outcome is not None:
                payload["outcome"] = outcome
            if counts is not None:
                payload["counts"] = dict(counts)

        return self._mutate_payload(memory_id, update)

    def find_active_session_for_agent(
        self,
        agent_id: str,
        *,
        workflow_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find an active shared memory session where the agent is already participating.

        This enables hierarchical multi-agent coordination by allowing sub-agents
        to join existing sessions rather than creating isolated ones.

        Parameters:
            agent_id (str): The ID of the agent to search for

        Returns:
            Optional[Dict[str, Any]]: The active session if found, None otherwise
        """
        try:
            # Get all active shared memory sessions
            all_sessions = self.memory_provider.list_all(MemoryType.SHARED_MEMORY)

            # Handle case where list_all returns None or empty list
            if not all_sessions:
                return None

            for session in all_sessions:
                payload = self._decode_payload(session)
                if payload.get("status") != "active":
                    continue
                if (
                    workflow_id is not None
                    and payload.get("workflow_id") != workflow_id
                ):
                    continue
                if payload.get("user_id") != user_id:
                    continue

                if (
                    payload.get("root_agent_id") == agent_id
                    or agent_id in payload.get("delegate_agent_ids", [])
                    or agent_id in payload.get("sub_agent_ids", [])
                ):
                    session_copy = dict(session)
                    session_copy["content"] = payload
                    return session_copy

            return None

        except Exception as e:
            logger.error(f"Error finding active session for agent {agent_id}: {e}")
            return None

    def register_sub_agents(
        self, memory_id: str, parent_agent_id: str, sub_agent_ids: List[str]
    ) -> bool:
        """
        Register sub-agents in an existing shared memory session.

        This method enables hierarchical agent coordination by tracking the complete
        agent hierarchy within a single shared memory session. When a delegate agent
        has its own sub-agents, they are registered here rather than creating a new session.

        Parameters:
            memory_id (str): The shared memory session ID
            parent_agent_id (str): The ID of the agent that owns these sub-agents
            sub_agent_ids (List[str]): List of sub-agent IDs to register

        Returns:
            bool: Success status
        """
        entry = BlackboardEntry(parent_agent_id, {}, "hierarchy_update")

        def register(payload):
            participants = {
                payload.get("root_agent_id"),
                *payload.get("delegate_agent_ids", []),
                *payload.get("sub_agent_ids", []),
            }
            if parent_agent_id not in participants:
                raise ValueError("Only a session participant can register sub-agents")
            sub_agents = payload.setdefault("sub_agent_ids", [])
            new = [
                agent_id
                for agent_id in dict.fromkeys(sub_agent_ids)
                if agent_id not in sub_agents
            ]
            if not new:
                return False
            sub_agents.extend(new)
            entry.content = {
                "parent_agent": parent_agent_id,
                "registered_sub_agents": new,
            }
            payload.setdefault("blackboard", []).append(entry.to_dict())

        return self._mutate_payload(memory_id, register)

    def get_agent_hierarchy(self, memory_id: str) -> Dict[str, Any]:
        """
        Get the complete agent hierarchy for a shared memory session.

        This provides visibility into the full multi-level agent structure,
        useful for debugging, monitoring, and coordination decisions.

        Parameters:
            memory_id (str): The shared memory session ID

        Returns:
            Dict[str, Any]: Hierarchy information including all agent levels
        """
        try:
            session = self.memory_provider.retrieve_by_id(
                memory_id, MemoryType.SHARED_MEMORY
            )
            if not session:
                return {}

            payload = self._decode_payload(session)

            hierarchy = {
                "root_agent": payload.get("root_agent_id"),
                "delegate_agents": payload.get("delegate_agent_ids", []),
                "sub_agents": payload.get("sub_agent_ids", []),
                "total_agents": 1
                + len(payload.get("delegate_agent_ids", []))
                + len(payload.get("sub_agent_ids", [])),
                "session_status": payload.get("status"),
                "created_at": payload.get("created_at"),
                "updated_at": payload.get("updated_at"),
            }

            return hierarchy

        except Exception as e:
            logger.error(f"Error getting agent hierarchy: {e}")
            return {}
