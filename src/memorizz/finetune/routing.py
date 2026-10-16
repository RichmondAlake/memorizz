"""Route a memory write with the active memory-routing model.

A write that asks for ``memory_type="auto"`` is filed where the fine-tuned
router says it belongs. A caller can only write some stores directly (the
MCP tool writes the knowledge base and short-term memory; entities,
conversation turns and summaries have their own tools), so ``place`` turns
the router's answer into a store the caller may write, or a skip, and says
why. Without an active router, or below the confidence threshold, the write
goes to the caller's default store.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

AUTO = "auto"
MIN_CONFIDENCE_ENV = "MEMORIZZ_ROUTING_MIN_CONFIDENCE"
DEFAULT_MIN_CONFIDENCE = 0.5

# Where a memory the caller cannot write directly should go instead.
NEXT_STEP = {
    "entity_memory": "memorizz_upsert_entity (a named person, team or thing and its attributes)",
    "conversation_memory": "memorizz_record_turn (a conversation turn)",
    "summaries": "memorizz_summarize_session (a summary of a session)",
    "personas": "the agent's persona (update_persona)",
    "workflow_memory": "a workflow (recorded from completed runs)",
    "toolbox": "the toolbox (register the tool)",
    "skillbox": "the skillbox (a reusable skill)",
}

_lock = threading.Lock()
_services: Dict[int, Any] = {}


def min_confidence() -> float:
    try:
        return float(os.getenv(MIN_CONFIDENCE_ENV, "") or DEFAULT_MIN_CONFIDENCE)
    except ValueError:
        return DEFAULT_MIN_CONFIDENCE


def _service(provider):
    from .service import FineTuning

    key = id(provider)
    with _lock:
        cached = _services.get(key)
        if cached is None or cached[0] is not provider:
            cached = _services[key] = (provider, FineTuning(provider))
        return cached[1]


def route(
    provider,
    text: str,
    *,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The router's answer for ``text``, or None when there is none (or the
    fine-tuning extra is not installed). An agent (named, or owning
    ``memory_id``) with its own choice uses that version, or no router;
    every other write uses the store's."""
    try:
        from . import agent_models

        service = _service(provider)
        own = agent_models.resolve(
            provider, "memory_routing", agent_id=agent_id, memory_id=memory_id
        )
        if own and own.get("mode") == "off":
            return None
        if own and own.get("mode") == "run":
            answer = next(iter(service.decide(own["run_id"], text).values()))
            return {
                "memory_type": answer.get("choice"),
                **answer,
                "run_id": own["run_id"],
                "chosen_for_agent": True,
            }
        if not service.active().get("memory_routing"):
            return None
        return service.route(text)
    except ImportError:
        return None


def place(
    answer: Optional[Dict[str, Any]],
    allowed: Iterable[str],
    *,
    default: str = "knowledge_base",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Turn a router answer into a store this caller may write.

    Returns ``memory_type`` (None means: do not store), the router's own
    answer (``routed_to``, ``confidence``, ``run_id``), whether it was
    ``applied``, a plain ``reason`` and, when the memory belongs in a store
    the caller cannot write, the ``next_step`` that would file it there."""
    allowed = {str(a) for a in allowed}
    threshold = min_confidence() if threshold is None else threshold
    if not answer:
        return {
            "memory_type": default,
            "applied": False,
            "reason": "No memory-routing model is active, so it went to the default store.",
        }
    target = str(answer.get("memory_type") or "")
    confidence = float(answer.get("confidence") or 0.0)
    out = {
        "routed_to": target,
        "confidence": round(confidence, 4),
        "run_id": answer.get("run_id"),
    }
    if confidence < threshold:
        return {
            **out,
            "memory_type": default,
            "applied": False,
            "reason": (
                f"The router was unsure ({confidence:.0%} for {target}, below "
                f"{threshold:.0%}), so it went to the default store."
            ),
        }
    if target == "do_not_store":
        return {
            **out,
            "memory_type": None,
            "applied": True,
            "reason": "The router judged it not worth keeping, so nothing was stored.",
        }
    if target in allowed:
        return {
            **out,
            "memory_type": target,
            "applied": True,
            "reason": f"Routed to {target}.",
        }
    result = {
        **out,
        "memory_type": default,
        "applied": False,
        "reason": f"The router says {target}, which this tool cannot write; stored in {default}.",
    }
    if target in NEXT_STEP:
        result["next_step"] = NEXT_STEP[target]
    return result


__all__ = [
    "AUTO",
    "MIN_CONFIDENCE_ENV",
    "NEXT_STEP",
    "min_confidence",
    "place",
    "route",
]
