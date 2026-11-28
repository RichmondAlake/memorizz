# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What a proactive agent may do while nobody asked it to.

A step the agent carries out on its own is an ordinary ``agent.run``. What
changes is which tool calls pause for approval, decided from what the tool
**declares** it does, never from what the model says about it:

``CAREFUL`` (the ``medium`` level, and any step that came from outside
content)
    A tool with side effects pauses, even one whose own policy would let it
    run. So does a tool that declares nothing about itself. The agent's own
    memory tools are let through: they change only what it remembers.
``OPEN`` (the ``high`` level)
    A tool's own policy stands. One that needs approval still pauses.

The pause is MemoRizz's durable approval: the exact call waits as a proposal a
person approves or rejects.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional

CAREFUL = "careful"
OPEN = "open"

#: Domains of tools that only change what the agent itself remembers.
INTERNAL_DOMAINS = frozenset({"memory"})

_active: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "memorizz_proactive_guard", default=None
)


def active_guard() -> Optional[Dict[str, Any]]:
    """The guard of the proactive step running in this context, if any."""
    return _active.get()


@contextmanager
def acting(mode: str, reason: str = "") -> Iterator[None]:
    """Run the enclosed agent turn as a step the agent took on its own."""
    if mode not in (CAREFUL, OPEN):
        raise ValueError("A proactive guard is 'careful' or 'open'")
    token = _active.set({"mode": mode, "reason": str(reason or "")})
    try:
        yield
    finally:
        _active.reset(token)


def guarded_policy(
    tool_name: str, policy: Dict[str, Any], *, declared: bool = True
) -> Dict[str, Any]:
    """``policy`` as it applies to ``tool_name`` in this context.

    Outside a proactive step it is returned unchanged. ``declared`` says
    whether the tool states its own effects (``governed_tool``, or an MCP
    server's hints): one that does not is treated as if it changed something.
    """
    guard = _active.get()
    if not guard or guard.get("mode") != CAREFUL:
        return policy
    if policy.get("requires_approval"):
        return policy
    domains = {str(item) for item in policy.get("domains") or ()}
    if domains & INTERNAL_DOMAINS:
        return policy
    if policy.get("side_effects") or not declared:
        why = (
            "it changes something"
            if policy.get("side_effects")
            else "it does not say whether it changes anything"
        )
        return {
            **policy,
            "requires_approval": True,
            "approval_reason": (
                f"The agent chose this step itself, and {tool_name} waits for "
                f"you because {why}."
                + (f" {guard['reason']}" if guard.get("reason") else "")
            ),
        }
    return policy
