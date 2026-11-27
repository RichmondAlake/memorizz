# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What each agent in the sidebar is doing: its current step, or its last message.

A reply streamed by this console is followed step by step, in this process.
For an agent that is not replying here, the line is the newest message in its
conversations, so a run made elsewhere (the CLI, an app) still shows up once
its turn is stored.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

RUNNING = "running"
WAITING = "waiting"
IDLE = "idle"

PREVIEW_CHARS = 160
# How long a last-message lookup is reused; a finished run drops it at once.
LAST_MESSAGE_TTL = 20.0
# How long the sidebar's agent list is reused. The page asks every few
# seconds, and each listing is a query (a request, on Notion).
AGENTS_TTL = 30.0
# The most agents one request may ask about.
MAX_AGENTS = 24

_WAITING_PREFIX = "[waiting for approval]"

_lock = threading.Lock()
# agent_id -> {run_id -> {"text", "state", "started_at"}}; newest run wins.
_live: Dict[str, Dict[str, Dict[str, Any]]] = {}
# agent_id -> (looked up at, status)
_last_message: Dict[str, Any] = {}
# (listed at, agents), or None
_agents: Optional[Any] = None


def _preview(text: Any) -> str:
    """One plain line of a message, short enough for the sidebar."""
    line = re.sub(r"\s+", " ", str(text or "")).strip()
    # Markdown marks only; an underscore is part of a tool's name.
    line = re.sub(r"[`*#>]+", "", line).strip()
    if len(line) > PREVIEW_CHARS:
        line = line[: PREVIEW_CHARS - 1].rstrip() + "…"
    return line


def _step_text(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The sidebar line for one stream event, or None when it changes nothing."""
    kind = event.get("type")
    if kind == "run.started":
        return {"state": RUNNING, "text": "Getting ready"}
    if kind == "status":
        stage = event.get("stage")
        if stage == "provider_request":
            return {"state": RUNNING, "text": "Thinking"}
        if stage in {"preparing_agent", "agent_ready"}:
            return {"state": RUNNING, "text": "Getting ready"}
        # A delegate or a harness says what it is on in its own words.
        if stage != "configuration_warning" and event.get("message"):
            return {"state": RUNNING, "text": _preview(event["message"])}
        return None
    if kind == "tool.started":
        return {
            "state": RUNNING,
            "text": "Running " + _preview(event.get("tool_name") or "a tool"),
        }
    if kind == "tool.completed":
        return {"state": RUNNING, "text": "Thinking"}
    if kind == "answer.delta":
        return {"state": RUNNING, "text": "Writing a reply"}
    if kind == "approval.required":
        proposal = event.get("proposal") or {}
        tool = proposal.get("tool_name") if isinstance(proposal, dict) else None
        return {
            "state": WAITING,
            "text": "Needs approval" + (f": {_preview(tool)}" if tool else ""),
        }
    return None


def note_event(agent_id: str, event: Dict[str, Any]) -> None:
    """Follow one event of a reply this console is streaming."""
    run_id = str(event.get("run_id") or "")
    if not agent_id or not run_id:
        return
    if event.get("type") == "run.done":
        end_run(agent_id, run_id)
        return
    step = _step_text(event)
    if step is None:
        return
    with _lock:
        runs = _live.setdefault(agent_id, {})
        current = runs.get(run_id)
        if current is None:
            current = runs[run_id] = {"started_at": time.time()}
        # A reply is written in many pieces; one line covers them all.
        current.update(step)


def end_run(agent_id: str, run_id: str) -> None:
    """Stop following a reply, and look its agent's last message up afresh."""
    with _lock:
        runs = _live.get(agent_id)
        if runs is not None:
            runs.pop(str(run_id), None)
            if not runs:
                _live.pop(agent_id, None)
        _last_message.pop(agent_id, None)
        # The run may have given the agent a new conversation to read.
        global _agents
        _agents = None


def _live_status(agent_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        runs = _live.get(agent_id)
        if not runs:
            return None
        newest = max(runs.values(), key=lambda run: run["started_at"])
        return {
            "state": newest.get("state", RUNNING),
            "text": newest.get("text", "Working"),
            "at": newest["started_at"],
        }


def _newest_message(agent: Any) -> Dict[str, Any]:
    """The newest thing said in any of an agent's conversations."""
    from ..cli.conversations import conversation_row_parts
    from .helpers import (
        _extract_agent_identifier,
        _extract_agent_memory_ids,
        _extract_message_timestamp,
        _retrieve_conversation_history,
    )

    own_id = str(_extract_agent_identifier(agent) or "")
    newest: Optional[Dict[str, Any]] = None
    newest_at = 0.0
    for memory_id in _extract_agent_memory_ids(agent):
        # A few rows, not one: the newest may be a tool's result. More in a
        # memory several agents share, where most rows are another agent's.
        for row in _retrieve_conversation_history(memory_id=memory_id, limit=24):
            writer = str(row.get("agent_id") or "") if isinstance(row, dict) else ""
            if writer and own_id and writer != own_id:
                # Another agent's turn in a shared memory is not this
                # agent's last line.
                continue
            parts = conversation_row_parts(row)
            if parts.get("role") not in {"user", "assistant"}:
                continue
            if not parts.get("content"):
                continue
            stamp = _extract_message_timestamp(row) or 0.0
            if newest is None or stamp >= newest_at:
                newest, newest_at = parts, stamp
    if newest is None:
        return {"state": IDLE, "text": "", "at": None}
    content = newest["content"]
    if content.lower().startswith(_WAITING_PREFIX):
        tool = re.search(r"asked to run (\S+)", content)
        return {
            "state": WAITING,
            "text": "Needs approval" + (f": {tool.group(1)}" if tool else ""),
            "at": newest_at or None,
        }
    text = _preview(content)
    if newest["role"] == "user":
        text = "You: " + text
    return {"state": IDLE, "text": text, "at": newest_at or None}


def _idle_status(agent_id: str, agent: Any) -> Dict[str, Any]:
    now = time.monotonic()
    with _lock:
        cached = _last_message.get(agent_id)
        if cached and now - cached[0] < LAST_MESSAGE_TTL:
            return cached[1]
    try:
        status = _newest_message(agent)
    except Exception:
        status = {"state": IDLE, "text": "", "at": None}
    with _lock:
        _last_message[agent_id] = (now, status)
    return status


def nav_status(agent_ids: Iterable[str], agents: List[Any]) -> Dict[str, Any]:
    """The status of each asked-for agent, keyed by agent ID."""
    from .helpers import _extract_agent_identifier

    by_id = {_extract_agent_identifier(agent): agent for agent in agents or []}
    result: Dict[str, Any] = {}
    for agent_id in list(dict.fromkeys(agent_ids))[:MAX_AGENTS]:
        agent = by_id.get(agent_id)
        if agent is None:
            continue
        result[agent_id] = _live_status(agent_id) or _idle_status(agent_id, agent)
    return result


def known_agents(load: Any, wanted: Iterable[str] = ()) -> List[Any]:
    """The agent list for the sidebar, listed again at most every
    ``AGENTS_TTL`` seconds, or at once when an asked-for agent is not in it
    (one just created)."""
    global _agents
    from .helpers import _extract_agent_identifier

    now = time.monotonic()
    with _lock:
        cached = _agents
    if cached and now - cached[0] < AGENTS_TTL:
        have = {_extract_agent_identifier(agent) for agent in cached[1]}
        if all(agent_id in have for agent_id in wanted):
            return cached[1]
    agents = list(load() or [])
    with _lock:
        _agents = (now, agents)
    return agents


def reset() -> None:
    """Forget everything (tests, and a change of memory provider)."""
    global _agents
    with _lock:
        _live.clear()
        _last_message.clear()
        _agents = None
