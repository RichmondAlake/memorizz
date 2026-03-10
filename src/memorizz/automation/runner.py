# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Job execution and delivery runner."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..channels.whatsapp.twilio import TwilioWhatsAppSender
from ..memagent import MemAgent
from .models import AutomationDelivery, AutomationJob
from .schedule import render_query_template, utcnow

AUTOMATION_DIRECTIVE = (
    "IMPORTANT: This is an automated scheduled query running without a human in the loop. "
    "Do NOT ask clarifying questions or request additional information. "
    "Make reasonable assumptions and provide a complete, actionable answer directly. "
    "If information is ambiguous, choose the most likely interpretation and proceed."
)


def _normalize_whatsapp_recipient(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("whatsapp:"):
        text = text.split(":", 1)[1].strip()
    number = re.sub(r"[\s\-()]", "", text)
    if number and not number.startswith("+") and number.isdigit():
        number = f"+{number}"
    if not (number.startswith("+") and number[1:].isdigit()):
        return ""
    return f"whatsapp:{number}"


def _get_whatsapp_recipients(job: AutomationJob) -> List[str]:
    raw = job.delivery_config.get("whatsapp_to") or job.delivery_config.get("to") or []
    if isinstance(raw, str):
        raw_list = [raw]
    elif isinstance(raw, list):
        raw_list = raw
    else:
        raw_list = []

    recipients: List[str] = []
    seen = set()
    for item in raw_list:
        normalized = _normalize_whatsapp_recipient(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        recipients.append(normalized)
    return recipients


def execute_job_action(
    job: AutomationJob, *, scheduled_for_utc: datetime, memory_provider: Any
) -> Dict[str, Any]:
    if job.action_type != "agent_query":
        raise ValueError(f"Unsupported action_type '{job.action_type}'")

    query_template = str(job.action_config.get("query_template") or "").strip()
    if not query_template:
        raise ValueError(
            "action_config.query_template is required for agent_query jobs"
        )

    memory_id = str(job.action_config.get("memory_id") or "").strip() or None
    rendered_query = render_query_template(
        query_template, scheduled_for_utc=scheduled_for_utc, tz_name=job.timezone
    )

    agent = MemAgent.load(job.agent_id, memory_provider=memory_provider)
    augmented_query = f"[{AUTOMATION_DIRECTIVE}]\n\n{rendered_query}"
    response = agent.run(augmented_query, memory_id=memory_id)

    # Capture the actual memory_id used (may be auto-generated if None was passed)
    actual_memory_id = getattr(agent, "_current_memory_id", None) or memory_id

    return {
        "rendered_query": rendered_query,
        "response": response,
        "memory_id": actual_memory_id,
    }


def deliver_job_output(
    job: AutomationJob,
    *,
    run_id: str,
    output_text: str,
    store: Any,
) -> List[AutomationDelivery]:
    if not job.delivery_type or job.delivery_type == "in_chat":
        # "in_chat" delivery is a no-op here — the response is already
        # persisted in the agent's conversation memory via agent.run().
        return []

    if job.delivery_type != "whatsapp_twilio":
        raise ValueError(f"Unsupported delivery_type '{job.delivery_type}'")

    recipients = _get_whatsapp_recipients(job)
    if not recipients:
        return []

    sender = TwilioWhatsAppSender.from_env()
    deliveries: List[AutomationDelivery] = []

    for recipient in recipients:
        delivery_id = str(uuid.uuid4())
        try:
            resp = sender.send(to=recipient, body=output_text)
            msg_id = str(resp.get("provider_message_id") or "") or None
            delivery = AutomationDelivery(
                delivery_id=delivery_id,
                run_id=run_id,
                channel="whatsapp",
                provider="twilio",
                recipient=recipient,
                status="sent",
                provider_message_id=msg_id,
            )
        except Exception as exc:
            delivery = AutomationDelivery(
                delivery_id=delivery_id,
                run_id=run_id,
                channel="whatsapp",
                provider="twilio",
                recipient=recipient,
                status="failed",
                error=str(exc),
            )

        try:
            store.record_delivery(run_id, delivery)
        except Exception:
            # Recording deliveries is best-effort; the run status still matters.
            pass
        deliveries.append(delivery)

    return deliveries


def run_job_once(
    job: AutomationJob,
    *,
    run_id: str,
    scheduled_for_utc: datetime,
    memory_provider: Any,
    store: Any,
) -> Dict[str, Any]:
    result = execute_job_action(
        job, scheduled_for_utc=scheduled_for_utc, memory_provider=memory_provider
    )
    response = str(result.get("response") or "")
    deliveries = deliver_job_output(
        job, run_id=run_id, output_text=response, store=store
    )
    result["deliveries"] = [d.model_dump() for d in deliveries]
    if deliveries:
        sent = sum(1 for d in deliveries if d.status == "sent")
        failed = sum(1 for d in deliveries if d.status == "failed")
        result["delivery_summary"] = {
            "sent": sent,
            "failed": failed,
            "total": len(deliveries),
        }
        if sent == 0 and failed > 0:
            first_error = next(
                (d.error for d in deliveries if d.status == "failed" and d.error), None
            )
            result["delivery_error"] = first_error or "All deliveries failed."
    return result
