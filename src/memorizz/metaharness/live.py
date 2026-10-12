"""Text a harness streams while it writes, for whoever watches the run.

Some harness CLIs report their answer and reasoning a few words at a time
(Claude Code, pi, Hermes). The ledger keeps each message once it is
complete; the pieces in between are never stored. A terminal following a
run in this process subscribes here to show them as they arrive.

An adapter calls ``new_block`` when the harness starts a new message or
reasoning block and ``publish`` for each piece. Listeners receive
``{"type": "partial", "data": {"kind", "id", "delta"}}``: ``kind`` is
"message" or "reasoning", ``id`` names the block (a new id starts a new
block) and ``delta`` is the next piece. Pieces are redacted line by line by
the adapter loop; a listener that shows them must redact the text it
assembles too, since a secret can span pieces. Nothing is kept for a run
nobody watches.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List

Listener = Callable[[Dict[str, Any]], None]

_lock = threading.Lock()
_listeners: Dict[str, List[Listener]] = {}
_blocks: Dict[str, int] = {}


def subscribe(run_id: str, listener: Listener) -> Callable[[], None]:
    """Receive ``run_id``'s partial text; call the result to stop."""
    run_id = str(run_id)
    with _lock:
        _listeners.setdefault(run_id, []).append(listener)

    def unsubscribe() -> None:
        with _lock:
            listeners = _listeners.get(run_id, [])
            if listener in listeners:
                listeners.remove(listener)
            if not listeners:
                _listeners.pop(run_id, None)
                _blocks.pop(run_id, None)

    return unsubscribe


def new_block(run_id: str) -> None:
    """The harness starts writing a new message or reasoning block."""
    run_id = str(run_id)
    if run_id in _listeners:
        with _lock:
            _blocks[run_id] = _blocks.get(run_id, 0) + 1


def publish(run_id: str, kind: str, delta: Any) -> None:
    """Hand one piece of ``run_id``'s text to its listeners, if any."""
    run_id = str(run_id)
    if not delta or run_id not in _listeners:
        return
    with _lock:
        listeners = list(_listeners.get(run_id, ()))
        block = _blocks.get(run_id, 0)
    event = {
        "type": "partial",
        "data": {"kind": kind, "id": str(block), "delta": str(delta)},
    }
    for listener in listeners:
        try:
            listener(event)
        except Exception:
            pass  # a terminal that went away must not break the run


__all__ = ["new_block", "publish", "subscribe"]
