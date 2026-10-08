# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Job execution and delivery runner."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..channels.whatsapp.numbers import normalize_whatsapp_number
from ..channels.whatsapp.twilio import TwilioWhatsAppSender
from ..memagent import MemAgent
from .models import AutomationDelivery, AutomationJob
from .schedule import render_query_template

AUTOMATION_DIRECTIVE = (
    "IMPORTANT: This is an automated scheduled query running without a human in the loop. "
    "Do NOT ask clarifying questions or request additional information. "
    "Make reasonable assumptions and provide a complete, actionable answer directly. "
    "If information is ambiguous, choose the most likely interpretation and proceed."
)


class AttemptAbandoned(RuntimeError):
    """The worker gave up on this attempt (timeout); nothing may be delivered."""


def _delivery_id(run_id: str, recipient: str) -> str:
    """One stable id per (run, recipient), so a repeat never records twice."""
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"memorizz:automation:{run_id}:{recipient}")
    )


def _sent_deliveries(store: Any, run_id: str) -> Dict[str, AutomationDelivery]:
    """Recipients this run already reached, keyed by recipient."""
    lister = getattr(store, "list_deliveries", None)
    if not callable(lister):
        return {}
    try:
        rows = lister(run_id) or []
    except Exception:
        return {}
    return {d.recipient: d for d in rows if getattr(d, "status", None) == "sent"}


def _abandoned(stop_event: Optional[threading.Event]) -> bool:
    return stop_event is not None and stop_event.is_set()


def _normalize_whatsapp_recipient(value: str) -> str:
    """``whatsapp:+E164`` for a deliverable recipient, else "" (skipped)."""
    try:
        return f"whatsapp:{normalize_whatsapp_number(value)}"
    except ValueError:
        return ""


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

    # An agent executing a scheduled job should do the task, not create/list/run
    # other automations. automations_enabled=False skips fresh registration; we
    # also strip any automation-management tools that persisted on the saved
    # agent (otherwise smaller models call them mid-task -> off-task/empty runs).
    agent = MemAgent.load(
        job.agent_id, memory_provider=memory_provider, automations_enabled=False
    )
    if getattr(agent, "tool_manager", None) is not None:
        from ..memagent.managers.automation_manager import AUTOMATION_TOOL_NAMES

        for _name in AUTOMATION_TOOL_NAMES:
            agent.tool_manager.remove_tool(_name)

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
    stop_event: Optional[threading.Event] = None,
) -> List[AutomationDelivery]:
    """Send the output once per (run, recipient).

    A recipient the store already shows as sent for this run is skipped, and
    nothing more goes out once ``stop_event`` is set (the worker abandoned
    this attempt), so an attempt that outlives its timeout cannot deliver.
    """
    if not job.delivery_type or job.delivery_type == "in_chat":
        # "in_chat" delivery is a no-op here — the response is already
        # persisted in the agent's conversation memory via agent.run().
        return []

    if job.delivery_type != "whatsapp_twilio":
        raise ValueError(f"Unsupported delivery_type '{job.delivery_type}'")

    recipients = _get_whatsapp_recipients(job)
    if not recipients:
        return []

    already_sent = _sent_deliveries(store, run_id)
    pending = [r for r in recipients if r not in already_sent]
    deliveries: List[AutomationDelivery] = [
        already_sent[r] for r in recipients if r in already_sent
    ]
    if not pending:
        return deliveries

    sender = TwilioWhatsAppSender.from_env()
    for recipient in pending:
        if _abandoned(stop_event):
            break
        delivery_id = _delivery_id(run_id, recipient)
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
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    result = execute_job_action(
        job, scheduled_for_utc=scheduled_for_utc, memory_provider=memory_provider
    )
    if _abandoned(stop_event):
        # The worker already recorded this run as timed out; delivering now
        # would send a result nobody is tracking.
        raise AttemptAbandoned(
            f"Attempt for run {run_id} exceeded its time limit before delivery"
        )
    response = str(result.get("response") or "")
    deliveries = deliver_job_output(
        job,
        run_id=run_id,
        output_text=response,
        store=store,
        stop_event=stop_event,
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
