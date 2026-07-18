# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""WhatsApp channel webhook: POST /webhook/whatsapp/incoming (Twilio callback).

Extracted verbatim from ``ui/app.py``. Route path, response class, and
behavior are unchanged. (The activate/deactivate routes were deleted earlier.)
"""

import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["whatsapp"])

logger = logging.getLogger(__name__)


@router.post("/webhook/whatsapp/incoming")
async def whatsapp_webhook_incoming(request: Request):
    """
    Twilio WhatsApp webhook receiver.
    Validates signature and queues message for processing.
    """
    try:
        # 1. Get Twilio signature from header
        signature = request.headers.get("X-Twilio-Signature", "")

        # 2. Parse form data
        form_data = await request.form()
        params = {k: v for k, v in form_data.items()}

        # 3. Validate signature
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        url = str(request.url)

        from memorizz.channels.whatsapp.webhook_validator import (
            validate_twilio_signature,
        )

        if not validate_twilio_signature(auth_token, url, params, signature):
            logger.warning("Invalid Twilio webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")

        # 4. Extract message data
        message_data = {
            "from": params.get("From", ""),
            "body": params.get("Body", ""),
            "message_sid": params.get("MessageSid", ""),
        }

        # 5. Queue for processing
        from memorizz.channels.whatsapp.queue import enqueue_message

        enqueue_message(message_data)

        logger.info(f"Queued WhatsApp message from {message_data['from']}")

        # 6. Return 200 OK immediately (Twilio requires fast response)
        return JSONResponse({"status": "queued"}, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        # Still return 200 to avoid Twilio retries
        return JSONResponse({"status": "error"}, status_code=200)
