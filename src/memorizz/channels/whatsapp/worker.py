# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Background worker for processing WhatsApp messages."""

import logging
import threading
from typing import Any, Optional

from .message_handler import process_incoming_message
from .queue import dequeue_message
from .settings import WhatsAppSettings
from .twilio import TwilioWhatsAppSender

logger = logging.getLogger(__name__)


def run_whatsapp_worker(
    *,
    memory_provider: Any,
    poll_interval_s: int = 1,
    stop_event: Optional[threading.Event] = None,
) -> None:
    """
    Background worker that processes queued WhatsApp messages.

    This worker runs in a separate thread and:
    1. Dequeues incoming WhatsApp messages
    2. Gets the active agent ID from settings
    3. Processes each message with the active agent
    4. Sends the agent's response back via Twilio
    5. Handles errors gracefully without crashing

    Args:
        memory_provider: Memory provider instance for agent loading and settings
        poll_interval_s: How often to poll the queue for messages (default: 1 second)
        stop_event: Threading event to signal worker shutdown
    """
    settings = WhatsAppSettings(memory_provider)
    sender = TwilioWhatsAppSender.from_env()

    logger.info("WhatsApp worker started")

    while True:
        # Check if we should stop
        if stop_event and stop_event.is_set():
            logger.info("WhatsApp worker stopping")
            break

        # Dequeue next message (blocks up to poll_interval_s seconds)
        message = dequeue_message(timeout=poll_interval_s)
        if not message:
            continue  # No messages, loop again

        logger.info(f"Processing WhatsApp message from {message['from']}")

        try:
            # Get the active agent ID
            active_agent_id = settings.get_active_agent_id()

            if not active_agent_id:
                logger.warning("No active WhatsApp agent configured")
                try:
                    sender.send(
                        to=message["from"],
                        body="Sorry, no agent is currently available for WhatsApp chat. Please contact the administrator.",
                    )
                except Exception as e:
                    logger.error(f"Failed to send 'no agent' message: {e}")
                continue

            # Process the message with the active agent
            result = process_incoming_message(
                message_data=message,
                active_agent_id=active_agent_id,
                memory_provider=memory_provider,
            )

            # Send response back to user
            if result["success"]:
                try:
                    sender.send(to=message["from"], body=result["response"])
                    logger.info(f"Response sent to {message['from']}")
                except Exception as e:
                    logger.error(f"Failed to send response: {e}", exc_info=True)
            else:
                # Agent processing failed, send error message
                error_msg = "Sorry, I encountered an error processing your message. Please try again."
                try:
                    sender.send(to=message["from"], body=error_msg)
                except Exception as e:
                    logger.error(f"Failed to send error message: {e}")

                logger.error(f"Processing error: {result.get('error')}")

        except Exception as e:
            # Catch-all to prevent worker from crashing
            logger.error(f"Worker error: {e}", exc_info=True)
            try:
                sender.send(
                    to=message["from"],
                    body="Sorry, something went wrong. Please try again later.",
                )
            except Exception:
                pass  # Nothing we can do if sending also fails

    logger.info("WhatsApp worker stopped")
