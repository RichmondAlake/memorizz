# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""In-memory queue for incoming WhatsApp messages."""

from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any, Dict, Optional

# Global queue for incoming WhatsApp messages
_whatsapp_message_queue: Queue = Queue()


def enqueue_message(message_data: Dict[str, Any]) -> None:
    """
    Add incoming WhatsApp message to processing queue.

    Args:
        message_data: Dictionary containing message information with keys:
            - from: Sender's WhatsApp number (e.g., "whatsapp:+1234567890")
            - body: Message text
            - message_sid: Twilio message SID
    """
    _whatsapp_message_queue.put(
        {
            "from": message_data["from"],
            "body": message_data["body"],
            "message_sid": message_data["message_sid"],
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def dequeue_message(timeout: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Get next message from queue (blocking with timeout).

    Args:
        timeout: Maximum time to wait for a message (in seconds)

    Returns:
        Dictionary containing message data, or None if queue is empty
    """
    try:
        return _whatsapp_message_queue.get(timeout=timeout)
    except Empty:
        return None


def get_queue_size() -> int:
    """Get the current number of messages in the queue."""
    return _whatsapp_message_queue.qsize()
