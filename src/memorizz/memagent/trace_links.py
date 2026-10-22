"""Find a finished turn's trace: by turn id, the newest one, or as a UI link.

A MemAgent remembers the identity of the turns it finished (a bounded,
process-local list). ``record_task_outcome(turn_id=...)`` reads it to join a
grader's verdict to the right trace, and ``trace_url()`` turns it into a
Traces page link. A turn finished in another process is looked up in the
provider's stored trace bundles instead.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional
from urllib.parse import urlencode

# How many finished turns an agent keeps for turn_id lookups.
RECENT_TURN_LIMIT = 256
DEFAULT_UI_URL = "http://127.0.0.1:8765"
# How many stored bundles a turn_id lookup reads before giving up.
_STORED_LOOKUP_LIMIT = 200
_CONTEXT_FIELDS = (
    "application_id",
    "agent_id",
    "run_id",
    "turn_id",
    "root_trace_id",
    "memory_id",
    "thread_id",
    "user_id",
)
_LINK_FIELDS = ("agent_id", "thread_id", "root_trace_id", "turn_id")
_LOCK = threading.Lock()


def _recent(agent: Any) -> "OrderedDict[str, Dict[str, Any]]":
    with _LOCK:
        recent = agent.__dict__.get("_recent_trace_turns")
        if recent is None:
            recent = agent.__dict__["_recent_trace_turns"] = OrderedDict()
        return recent


def remember_turn(agent: Any, context: Optional[Dict[str, Any]]) -> None:
    """Keep a finished turn's identity for lookups and links."""
    identity = {
        key: str(value)
        for key, value in dict(context or {}).items()
        if key in _CONTEXT_FIELDS and value is not None
    }
    turn_id = identity.get("turn_id")
    if not turn_id or not identity.get("root_trace_id"):
        return
    recent = _recent(agent)
    with _LOCK:
        recent[turn_id] = identity
        recent.move_to_end(turn_id)
        while len(recent) > RECENT_TURN_LIMIT:
            recent.popitem(last=False)


def latest_turn(agent: Any) -> Optional[Dict[str, Any]]:
    """The newest turn this agent object finished, in any thread or stream."""
    recent = agent.__dict__.get("_recent_trace_turns")
    if not recent:
        return None
    with _LOCK:
        return dict(next(reversed(recent.values()))) if recent else None


def turn_context(agent: Any, turn_id: Any) -> Dict[str, Any]:
    """The trace context of one of this agent's turns.

    Turns this process finished are found in memory; older ones in the
    agent's stored trace bundles. Raises ``ValueError`` when neither has it.
    """
    key = str(turn_id or "").strip()
    if not key:
        raise ValueError("turn_id must be a non-empty string")
    recent = agent.__dict__.get("_recent_trace_turns") or {}
    with _LOCK:
        found = recent.get(key)
    if found:
        return dict(found)
    stored = _stored_turn_context(agent, key)
    if stored:
        return stored
    raise ValueError(
        f"No trace was found for turn {key!r} of agent {agent.agent_id!r}; "
        "pass trace_context instead"
    )


def _stored_turn_context(agent: Any, turn_id: str) -> Optional[Dict[str, Any]]:
    provider = getattr(agent, "memory_provider", None)
    query = getattr(provider, "query_observability_records", None)
    if not callable(query):
        return None
    from ..enums.memory_type import MemoryType
    from ..observability.normalization import read_payload

    try:
        page = query(
            MemoryType.SHARED_MEMORY,
            agent_ids=[agent.agent_id],
            record_type="observability_trace_bundle",
            event_filters={"turn_id": turn_id},
            limit=_STORED_LOOKUP_LIMIT,
        )
    except Exception:
        return None
    for row in page.get("items") or []:
        if not isinstance(row, dict):
            continue
        payload = read_payload(row) or {}
        # A bundle's document memory_id is its own record id; the agent
        # memory it traced is trace_memory_id.
        identity = {
            key: payload.get(key) or row.get(key)
            for key in _CONTEXT_FIELDS
            if key != "memory_id"
        }
        identity["memory_id"] = (
            payload.get("trace_memory_id")
            or row.get("trace_memory_id")
            or payload.get("memory_id")
        )
        if str(identity.get("turn_id") or "") == turn_id and identity.get(
            "root_trace_id"
        ):
            return {key: str(value) for key, value in identity.items() if value}
    return None


def trace_url(
    context: Optional[Dict[str, Any]], base_url: str = DEFAULT_UI_URL
) -> Optional[str]:
    """The Traces page link for a turn, opened at its timeline."""
    identity = dict(context or {})
    if not identity.get("turn_id") or not identity.get("root_trace_id"):
        return None
    params = {key: identity[key] for key in _LINK_FIELDS if identity.get(key)}
    return (
        f"{str(base_url or DEFAULT_UI_URL).rstrip('/')}/traces?"
        f"{urlencode(params)}#trace-timeline"
    )


__all__ = [
    "DEFAULT_UI_URL",
    "RECENT_TURN_LIMIT",
    "latest_turn",
    "remember_turn",
    "trace_url",
    "turn_context",
]
