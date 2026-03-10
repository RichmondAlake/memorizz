# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Process incoming WhatsApp messages and generate responses."""

import logging
import re
from typing import Any, Dict, Optional

from ...memagent import MemAgent

logger = logging.getLogger(__name__)


def normalize_phone_to_memory_id(phone: str) -> str:
    """
    Convert WhatsApp phone number to memory_id.

    Args:
        phone: WhatsApp phone number (e.g., "whatsapp:+1234567890" or "+1234567890")

    Returns:
        Memory ID in format "whatsapp_+1234567890"
    """
    # Remove "whatsapp:" prefix if present
    number = phone
    if number.lower().startswith("whatsapp:"):
        number = number.split(":", 1)[1].strip()

    # Remove any whitespace, hyphens, parentheses
    number = re.sub(r"[\s\-()]", "", number)

    # Ensure it starts with + for E.164 format
    if not number.startswith("+"):
        number = f"+{number}"

    # Return formatted memory_id
    return f"whatsapp_{number}"


def process_incoming_message(
    message_data: Dict[str, Any],
    active_agent_id: str,
    memory_provider: Any,
) -> Dict[str, Any]:
    """
    Process incoming WhatsApp message.

    Args:
        message_data: Dictionary containing:
            - from: Sender's WhatsApp number
            - body: Message text
            - message_sid: Twilio message SID
        active_agent_id: ID of the active WhatsApp agent
        memory_provider: Memory provider instance

    Returns:
        Dictionary with:
            - success: bool
            - response: str (agent's response or empty if failed)
            - memory_id: str (memory ID used for this conversation)
            - error: Optional[str] (error message if failed)
    """
    try:
        logger.info(f"Processing message from {message_data['from']}")

        # 1. Load the active agent
        agent = MemAgent.load(active_agent_id, memory_provider=memory_provider)

        if not agent:
            logger.error(f"Failed to load agent {active_agent_id}")
            return {
                "success": False,
                "response": "",
                "error": "Failed to load agent",
            }

        # 2. Generate memory_id from phone number
        memory_id = normalize_phone_to_memory_id(message_data["from"])
        logger.info(f"Using memory_id: {memory_id}")

        # 3. Run agent with the message
        response = agent.run(message_data["body"], memory_id=memory_id)

        logger.info(f"Agent response generated (length: {len(response)})")

        return {
            "success": True,
            "response": response,
            "memory_id": memory_id,
        }

    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}", exc_info=True)
        return {
            "success": False,
            "response": "",
            "error": str(e),
        }
