# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Manages global WhatsApp settings (active agent selection)."""

from typing import Any, Optional

from ...enums import MemoryType


class WhatsAppSettings:
    """Manages which agent is currently active for WhatsApp."""

    def __init__(self, memory_provider: Any):
        self.provider = memory_provider
        self.settings_memory_id = "_whatsapp_global_settings"

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
