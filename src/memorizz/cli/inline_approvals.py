# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Decide a paused tool call in the chat: ask y/n, then carry on.

A turn that reaches a tool needing approval (opening an app, a click, a
browser task) stops with a durable proposal and emits ``approval.required``.
Instead of leaving it for ``/approvals``, the REPL asks right away. Yes
approves the exact call and resumes the turn, which may stop at the next
action and ask again; no rejects it. Ctrl-C leaves it pending for /approvals.
"""

from __future__ import annotations

import getpass
import json
from typing import Any, Dict, Iterable, List

from rich.markdown import Markdown
from rich.markup import escape

#: Characters of one argument value shown in the question.
_VALUE_CHARS = 120


def _pending(agent) -> Dict[str, Dict[str, Any]]:
    try:
        rows = agent.list_approval_proposals(status="pending", limit=100) or []
    except Exception:
        return {}
    return {
        str(row.get("proposal_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("proposal_id")
    }


def describe(row: Dict[str, Any]) -> str:
    """``desktop__open_url(url="https://…", app="Notion")``."""
    arguments = row.get("arguments")
    parts: List[str] = []
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            text = json.dumps(value, ensure_ascii=False, default=str)
            if len(text) > _VALUE_CHARS:
                text = text[: _VALUE_CHARS - 1] + "…"
            parts.append(f"{key}={text}")
    return f"{row.get('tool_name') or 'tool'}({', '.join(parts)})"


def _approver(session) -> str:
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    return str(getattr(session, "user_id", None) or user or "cli")


def ask(session, proposal_ids: Iterable[Any]) -> None:
    """Ask about each proposal this turn raised, and the ones resuming raises."""
    agent = session.agent
    console = session.console
    queue = list(dict.fromkeys(str(item) for item in proposal_ids if item))
    approver = _approver(session)
    while queue:
        proposal_id = queue.pop(0)
        row = _pending(agent).get(proposal_id)
        if row is None:
            continue  # decided elsewhere meanwhile
        try:
            answer = console.input(
                f"[yellow]Allow[/yellow] {escape(describe(row))}? [y/N] "
            )
        except (EOFError, KeyboardInterrupt):
            console.print(
                f"\n[dim]Left pending: /approvals approve {proposal_id} "
                f"{escape(approver)}, then /approvals resume {proposal_id}[/dim]"
            )
            return
        if answer.strip().lower() not in {"y", "yes"}:
            agent.reject(
                proposal_id, approver_id=approver, reason="Declined in the chat"
            )
            console.print("[dim]Not done.[/dim]")
            continue
        agent.approve(proposal_id, approver_id=approver, reason="Approved in the chat")
        before = set(_pending(agent))
        try:
            with console.status("[dim]running the approved step[/dim]"):
                result = agent.resume_approval(proposal_id)
        except Exception as exc:
            console.print(f"[red]Could not run it:[/red] {escape(str(exc))}")
            continue
        # A resumed turn that reaches another action pauses again: ask next.
        raised = [item for item in _pending(agent) if item not in before]
        if not getattr(result, "ok", True):
            error = result.error or result.error_code or "failed"
            console.print(f"[red]It failed:[/red] {escape(str(error))}")
        elif not raised and result.assistant_response:
            console.print(Markdown(str(result.assistant_response)))
        queue = raised + queue


__all__ = ["ask", "describe"]
