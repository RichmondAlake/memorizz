# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Countdown timers a person can watch: a pomodoro, a tea timer, a break.

A timer is not an automation. An automation runs the agent later, which needs a
worker, approval and a model call. A timer only has to count down where the
person can see it and tell them when it ends, so nothing runs when it does.

Timers live in one small file in the MemoRizz home, so the CLI and the local UI
on this machine show the same ones. The clock is the end time, so a timer keeps
counting with both of them closed.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from ._env_io import memorizz_home

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

MAX_SECONDS = 24 * 3600
MAX_LABEL_CHARS = 60
DEFAULT_LABEL = "Timer"
# A finished timer nobody came back to.
KEEP_FINISHED_SECONDS = 24 * 3600

_thread_lock = threading.Lock()


def _path() -> Path:
    return memorizz_home() / "timers.json"


@contextmanager
def _locked() -> Iterator[None]:
    """One writer at a time, across threads and the CLI and UI processes."""
    with _thread_lock:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a") as handle:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def _usable(row: Any) -> bool:
    """A hand edit or a later version mustn't break a chat over a countdown."""
    if not isinstance(row, dict) or not isinstance(row.get("timer_id"), str):
        return False
    numbers = (row.get(key) for key in ("seconds", "started_at", "ends_at"))
    return isinstance(row.get("label"), str) and all(
        isinstance(number, (int, float)) and not isinstance(number, bool)
        for number in numbers
    )


def _read() -> List[Dict[str, Any]]:
    try:
        rows = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [row for row in rows if _usable(row)] if isinstance(rows, list) else []


def _write(rows: List[Dict[str, Any]]) -> None:
    path = _path()
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".timers-")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _clean_label(label: Any) -> str:
    text = " ".join(str(label or "").split())[:MAX_LABEL_CHARS].strip()
    return text or DEFAULT_LABEL


def _view(row: Dict[str, Any], now: float) -> Dict[str, Any]:
    remaining = max(0.0, float(row["ends_at"]) - now)
    return {
        "timer_id": row["timer_id"],
        "label": row["label"],
        "seconds": row["seconds"],
        "started_at": row["started_at"],
        "ends_at": row["ends_at"],
        "remaining": remaining,
        "state": "running" if remaining > 0 else "finished",
        "agent_id": row.get("agent_id"),
    }


def _kept(rows: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
    return [row for row in rows if float(row["ends_at"]) + KEEP_FINISHED_SECONDS > now]


def start(
    seconds: float,
    label: str = "",
    *,
    agent_id: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Start a countdown; raises ValueError for a length it can't keep."""
    length = int(round(float(seconds or 0)))
    if length < 1:
        raise ValueError("A timer needs a length of at least one second.")
    if length > MAX_SECONDS:
        raise ValueError("A timer can run for up to 24 hours.")
    moment = time.time() if now is None else now
    row = {
        "timer_id": uuid.uuid4().hex[:8],
        "label": _clean_label(label),
        "seconds": length,
        "started_at": moment,
        "ends_at": moment + length,
        "agent_id": agent_id,
    }
    with _locked():
        rows = _kept(_read(), moment)
        rows.append(row)
        _write(rows)
    return _view(row, moment)


def listed(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Running timers, then finished ones nobody dismissed, soonest first."""
    moment = time.time() if now is None else now
    views = [
        _view(row, moment)
        for row in _kept(_read(), moment)
        if float(row["ends_at"]) > moment or not row.get("dismissed")
    ]
    # "finished" sorts before "running", and running timers belong first.
    return sorted(views, key=lambda view: (view["state"] != "running", view["ends_at"]))


def running(now: Optional[float] = None) -> List[Dict[str, Any]]:
    return [view for view in listed(now) if view["state"] == "running"]


def _remove(rows: List[Dict[str, Any]], timer_id: str) -> Optional[Dict[str, Any]]:
    found = next((row for row in rows if row["timer_id"] == timer_id), None)
    if found is not None:
        rows.remove(found)
    return found


def cancel(timer_id: str = "", now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Stop a timer, or the only one running. None when that isn't clear."""
    moment = time.time() if now is None else now
    with _locked():
        rows = _read()
        wanted = str(timer_id or "").strip()
        if not wanted:
            active = [row for row in rows if float(row["ends_at"]) > moment]
            if len(active) != 1:
                return None
            wanted = active[0]["timer_id"]
        found = _remove(rows, wanted)
        if found is not None:
            _write(rows)
    return _view(found, moment) if found is not None else None


def dismiss(timer_id: str) -> bool:
    """Put away a finished timer. True when there was one."""
    with _locked():
        rows = _read()
        found = next((row for row in rows if row["timer_id"] == timer_id), None)
        if found is None:
            return False
        found["dismissed"] = True
        _write(rows)
    return True


def take_finished(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Finished timers not announced yet, marked so nothing announces them twice."""
    moment = time.time() if now is None else now
    if not _path().exists():
        return []
    with _locked():
        rows = _read()
        due = [
            row
            for row in _kept(rows, moment)
            if float(row["ends_at"]) <= moment
            and not row.get("announced")
            and not row.get("dismissed")
        ]
        if not due:
            return []
        for row in due:
            row["announced"] = True
        _write(rows)
    return [_view(row, moment) for row in due]


def format_time(seconds: float) -> str:
    """``24:59``, or ``1:00:00`` from an hour up. Rounds up, as a clock counts."""
    whole = int(seconds) + (0 if float(seconds).is_integer() else 1)
    hours, rest = divmod(max(0, whole), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


__all__ = [
    "MAX_SECONDS",
    "cancel",
    "dismiss",
    "format_time",
    "listed",
    "running",
    "start",
    "take_finished",
]
