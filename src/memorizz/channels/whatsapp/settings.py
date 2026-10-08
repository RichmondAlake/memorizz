# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Manages global WhatsApp settings (active agent selection).

The setting lives in conversation memory under one fixed ``memory_id``; the
worker reads the newest row's ``content`` as the active agent id. The
writers here store to that same row so the reader and writer cannot drift.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ...enums import MemoryType

logger = logging.getLogger(__name__)

SETTINGS_MEMORY_ID = "_whatsapp_global_settings"


class WhatsAppSettings:
    """Manages which agent is currently active for WhatsApp."""

    def __init__(self, memory_provider: Any):
        self.provider = memory_provider
        self.settings_memory_id = SETTINGS_MEMORY_ID

    def get_active_agent_id(self) -> Optional[str]:
        """Get the currently active WhatsApp agent ID."""
        try:
            # Retrieve stored settings using memory provider
            history = self.provider.retrieve_conversation_history_ordered_by_timestamp(
                memory_id=self.settings_memory_id,
                memory_type=MemoryType.CONVERSATION_MEMORY,
                limit=1,
            )

            if history and len(history) > 0:
                # Extract active_agent_id from the most recent entry
                latest = history[0]
                if isinstance(latest, dict):
                    return latest.get("content")
                return latest.content

            return None

        except Exception:
            return None

    def set_active_agent_id(self, agent_id: str) -> bool:
        """Make ``agent_id`` the agent that answers WhatsApp messages.

        Earlier setting rows are removed first, so the single remaining row is
        what ``get_active_agent_id`` reads back on every provider.
        """
        resolved = str(agent_id or "").strip()
        if not resolved:
            raise ValueError("agent_id must be a non-empty string")
        try:
            self._delete_setting_rows()
            now = datetime.now(timezone.utc).isoformat()
            self.provider.store(
                {
                    "memory_id": self.settings_memory_id,
                    "role": "system",
                    "content": resolved,
                    "setting": "active_agent_id",
                    "timestamp": now,
                    "created_at": now,
                    "updated_at": now,
                },
                memory_store_type=MemoryType.CONVERSATION_MEMORY,
            )
            return True
        except Exception as exc:
            logger.error("Failed to set the active WhatsApp agent: %s", exc)
            return False

    def clear_active_agent_id(self) -> bool:
        """Remove the active-agent setting; the worker then answers no one."""
        try:
            self._delete_setting_rows()
            return True
        except Exception as exc:
            logger.error("Failed to clear the active WhatsApp agent: %s", exc)
            return False

    def _setting_rows(self) -> List[Dict[str, Any]]:
        rows = self.provider.retrieve_conversation_history_ordered_by_timestamp(
            memory_id=self.settings_memory_id,
            memory_type=MemoryType.CONVERSATION_MEMORY,
        )
        return [row for row in (rows or []) if isinstance(row, dict)]

    def _delete_setting_rows(self) -> int:
        deleted = 0
        for row in self._setting_rows():
            row_id = row.get("_id") or row.get("id")
            if not row_id:
                continue
            if self.provider.delete_by_id(
                str(row_id), memory_store_type=MemoryType.CONVERSATION_MEMORY
            ):
                deleted += 1
        return deleted


def get_active_agent_id(memory_provider: Any) -> Optional[str]:
    """The active WhatsApp agent id, or None."""
    return WhatsAppSettings(memory_provider).get_active_agent_id()


def set_active_agent_id(memory_provider: Any, agent_id: str) -> bool:
    """Point WhatsApp at ``agent_id``; see :meth:`WhatsAppSettings.set_active_agent_id`."""
    return WhatsAppSettings(memory_provider).set_active_agent_id(agent_id)


def clear_active_agent_id(memory_provider: Any) -> bool:
    """Remove the active WhatsApp agent setting."""
    return WhatsAppSettings(memory_provider).clear_active_agent_id()
