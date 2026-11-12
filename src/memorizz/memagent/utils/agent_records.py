"""Scoped reads of what one agent has written: its runs, tool calls and
learned skills. A store is shared, so each read keeps this agent's records
only and orders them by when they were written."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..._time import to_epoch
from ...enums.memory_type import MemoryType

_TIME_KEYS = ("created_at", "timestamp", "updated_at")


def written_at(row: Dict[str, Any]) -> float:
    """Epoch seconds a record was written (0.0 when it carries no time)."""
    for key in _TIME_KEYS:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            seconds = to_epoch(value)
        except Exception:
            seconds = None
        if seconds is not None:
            return float(seconds)
    return 0.0


def _as_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds = to_epoch(value)
    if seconds is None:
        raise ValueError(f"not a time: {value!r}")
    return float(seconds)


def agent_records(
    provider: Any,
    memory_type: MemoryType,
    agent_id: str,
    *,
    since: Any = None,
    limit: Optional[int] = None,
    newest_first: bool = False,
    keep: Any = None,
) -> List[Dict[str, Any]]:
    """One agent's records of a type, oldest first unless asked otherwise.

    ``since`` is a datetime, an ISO string or epoch seconds: records written
    at or after it. ``limit`` keeps the most recent ones.
    """
    reader = getattr(provider, "list_for_agent", None)
    if callable(reader):
        rows = reader(memory_type, agent_id)
    else:  # a provider written before the scoped read existed
        rows = [
            row
            for row in provider.list_all(memory_type) or []
            if isinstance(row, dict) and row.get("agent_id") == agent_id
        ]
    rows = [row for row in rows or [] if isinstance(row, dict)]
    if keep is not None:
        rows = [row for row in rows if keep(row)]
    start = _as_epoch(since)
    if start is not None:
        rows = [row for row in rows if written_at(row) >= start]
    rows.sort(key=written_at)
    if limit is not None:
        rows = rows[-max(0, int(limit)) :] if int(limit) > 0 else []
    if newest_first:
        rows.reverse()
    return rows
