# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What is waiting for a person, for the bell in the top bar.

One list from the places a decision or a failure otherwise sits on its own
page: tool calls and harness runs held for approval, consolidation plans and
persona changes waiting for review, automations whose last run failed and
failure alerts that are firing. Every source is read on its own, so one that
is unavailable leaves the rest in place.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List
from urllib.parse import quote

from .helpers import _extract_agent_identifier, _list_agents
from .state import _state

logger = logging.getLogger(__name__)

CACHE_SECONDS = 20.0
_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"provider": None, "at": 0.0, "items": []}


def _field(record: Any, name: str) -> Any:
    if isinstance(record, dict):
        return record.get(name)
    return getattr(record, name, None)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _approval_items(names: Dict[str, str]) -> List[Dict[str, Any]]:
    """Pending proposals in the host approval store, grouped by where a
    person decides them."""
    from ..approval import default_approval_store

    harness = 0
    mcp: Dict[str, int] = {}
    tools: Dict[str, int] = {}
    for proposal in default_approval_store().list(status="pending", limit=200):
        tool = str(proposal.tool_name or "")
        owner = str(proposal.owner_id or "")
        if tool == "metaharness.run":
            harness += 1
        elif owner not in names:
            # A proposal for an agent in another store: not this UI's to decide.
            continue
        elif tool.startswith("mcp:"):
            mcp[owner] = mcp.get(owner, 0) + 1
        else:
            tools[owner] = tools.get(owner, 0) + 1
    items: List[Dict[str, Any]] = []
    if harness:
        items.append(
            {
                "kind": "approval",
                "title": f"{_plural(harness, 'harness run')} waiting for approval",
                "detail": "A coding-agent run starts once you approve it.",
                "href": "/harnesses",
                "action": "Open agent harnesses",
            }
        )
    for owner, count in mcp.items():
        items.append(
            {
                "kind": "approval",
                "title": f"{_plural(count, 'MCP tool call')} waiting for approval",
                "detail": f"Asked for by {names[owner]}.",
                "href": f"/mcp?agent_id={quote(owner)}",
                "action": "Review on MCP connections",
            }
        )
    for owner, count in tools.items():
        items.append(
            {
                "kind": "approval",
                "title": f"{_plural(count, 'tool call')} waiting for approval",
                "detail": f"Asked for by {names[owner]}.",
                "href": f"/agents/{quote(owner)}/playground",
                "action": "Open the playground",
            }
        )
    return items


def _review_items(agents: List[Any], names: Dict[str, str]) -> List[Dict[str, Any]]:
    """Consolidation plans and persona changes a schedule left for review."""
    from ..memagent.core import _shared_upkeep

    items: List[Dict[str, Any]] = []
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        plans = _shared_upkeep(_state["provider"], agent_id).plans
        base = f"/learning-control-plane?agent_id={quote(agent_id)}"
        if plans:
            items.append(
                {
                    "kind": "review",
                    "title": f"{_plural(len(plans), 'consolidation plan')} to review",
                    "detail": f"Proposed for {names[agent_id]}. Nothing changes "
                    "until you apply a plan.",
                    "href": base + "#memory-consolidation",
                    "action": "Review the plan",
                }
            )
        state = _field(agent, "persona_learning_state")
        if isinstance(state, dict) and state.get("pending"):
            items.append(
                {
                    "kind": "review",
                    "title": "A persona change to review",
                    "detail": f"Proposed for {names[agent_id]}.",
                    "href": base + "#persona-learning",
                    "action": "Review the change",
                }
            )
    return items


def _automation_items(store: Any) -> List[Dict[str, Any]]:
    if store is None:
        return []
    items = []
    for job in store.list_jobs(agent_id=None, enabled=True):
        runs = store.list_runs(job.job_id, limit=1)
        latest = runs[0] if runs else None
        status = str(getattr(getattr(latest, "status", None), "value", "") or "")
        if status.lower() in {"failed", "error", "timed_out", "timeout"}:
            items.append(
                {
                    "kind": "failure",
                    "title": f"Automation {job.name} failed its last run",
                    "detail": "Open the automation to see the run error.",
                    "href": f"/automations/{job.job_id}",
                    "action": "Open the automation",
                }
            )
    return items


def _alert_items() -> List[Dict[str, Any]]:
    from .failure_alerts import failure_attention

    return [
        {
            "kind": "failure",
            "title": item["title"],
            "detail": item.get("detail") or "",
            "href": item["href"],
            "action": item.get("action") or "Open failures",
        }
        for item in failure_attention()
    ]


def build_inbox(automation_store: Any = None) -> List[Dict[str, Any]]:
    """Everything waiting, approvals first. Empty for an account limited to
    trace inspection, which can act on none of it."""
    from .trace_access import current_principal

    if _state.get("provider") is None or current_principal.get().restricted:
        return []
    try:
        agents = list(_list_agents() or [])
    except Exception:
        logger.debug("Inbox: agents unavailable", exc_info=True)
        agents = []
    names = {}
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if agent_id:
            names[agent_id] = str(_field(agent, "name") or agent_id[:12])
    sources: List[Callable[[], List[Dict[str, Any]]]] = [
        lambda: _approval_items(names),
        lambda: _review_items(agents, names),
        lambda: _automation_items(automation_store),
        _alert_items,
    ]
    items: List[Dict[str, Any]] = []
    for source in sources:
        try:
            items.extend(source())
        except Exception:
            logger.debug("Inbox: a source is unavailable", exc_info=True)
    return items


def cached_inbox(automation_store: Any = None) -> List[Dict[str, Any]]:
    """``build_inbox`` at most once every ``CACHE_SECONDS`` per connection:
    every page asks for it."""
    provider = _state.get("provider")
    now = time.monotonic()
    with _cache_lock:
        fresh = _cache["provider"] is provider and now - _cache["at"] < CACHE_SECONDS
        if fresh:
            return list(_cache["items"])
    items = build_inbox(automation_store)
    with _cache_lock:
        _cache.update(provider=provider, at=now, items=items)
    return list(items)


def clear_cache() -> None:
    with _cache_lock:
        _cache.update(provider=None, at=0.0, items=[])


__all__ = ["build_inbox", "cached_inbox", "clear_cache"]
