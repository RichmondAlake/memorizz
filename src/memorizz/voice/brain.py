"""Who answers for an agent in a live call.

A live call (Vapi's servers, a LiveKit room) asks for the agent's next reply
from outside the request that started it. The reply function is kept here by
agent id: whoever starts the call registers it, the call's handler looks it up.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from .base import Brain

_brains: Dict[str, Brain] = {}
_lock = threading.Lock()


def set_brain(agent_id: str, reply: Brain) -> None:
    with _lock:
        _brains[str(agent_id)] = reply


def clear_brain(agent_id: str, reply: Optional[Brain] = None) -> None:
    """Forget an agent's reply function (only if it is still ``reply``, when given)."""
    with _lock:
        if reply is None or _brains.get(str(agent_id)) is reply:
            _brains.pop(str(agent_id), None)


def get_brain(agent_id: str) -> Optional[Brain]:
    with _lock:
        return _brains.get(str(agent_id))
