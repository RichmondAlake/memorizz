# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Turns another process is running, as they stream: the CLI writes them and
the local UI's playground follows them.

Each turn is one JSON-lines journal under ``$MEMORIZZ_HOME/live/<agent_id>/``:
the question when the run starts, status and tool steps, the answer's text as
it arrives and how the run ended. Only the running turn matters: the stored
conversation remains the record, and journals older than an hour are removed
when a new turn starts. The journals hold conversation text, so the folder is
private to the user, like the memory store beside it.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._env_io import memorizz_home

# A journal's last write older than this is not a running turn.
RUNNING_SECONDS = 120.0
# Journals are removed this long after their last write.
KEEP_SECONDS = 3600.0
# Answer text is written at most this often; finer deltas are joined.
DELTA_SECONDS = 0.05

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def agent_dir(agent_id: str) -> Optional[Path]:
    """The agent's journal folder, or None for an id that is not a plain name."""
    agent_id = str(agent_id or "")
    if not _SAFE_ID.match(agent_id) or ".." in agent_id:
        return None
    return memorizz_home() / "live" / agent_id


class LiveTurn:
    """Write one turn's events for other processes to follow. Never raises:
    a failed write costs only the live view, never the turn."""

    def __init__(self, agent_id: Any, query: str, *, source: str = "cli"):
        self.directory = agent_dir(str(agent_id or ""))
        self.query = query
        self.source = source
        self.turn_id = uuid.uuid4().hex
        self.memory_id: Optional[str] = None
        self.thread_id: Optional[str] = None
        self.path: Optional[Path] = None
        self.done = False
        self._pending = ""
        self._last_delta = float("-inf")

    def _write(self, kind: str, **fields: Any) -> None:
        if self.path is None:
            return
        line = {
            "type": kind,
            "turn_id": self.turn_id,
            "memory_id": self.memory_id,
            "thread_id": self.thread_id,
            "source": self.source,
            "at": time.time(),
            **fields,
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            self.path = None

    def _start(self, event: Dict[str, Any]) -> None:
        self.memory_id = event.get("memory_id") or None
        self.thread_id = event.get("thread_id") or None
        if self.directory is None or not self.memory_id:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            prune(self.directory)
        except OSError:
            return
        self.path = self.directory / f"{int(time.time() * 1000)}-{self.turn_id}.jsonl"
        self._write("turn.started", query=self.query)

    def _flush(self) -> None:
        if self._pending:
            text, self._pending = self._pending, ""
            self._write("answer.delta", delta=text)
        self._last_delta = time.monotonic()

    def tick(self) -> None:
        """Write answer text held back by the delta interval; call while idle."""
        if self._pending and time.monotonic() - self._last_delta >= DELTA_SECONDS:
            self._flush()

    def event(self, event: Dict[str, Any]) -> None:
        """Record one public stream event (``run_stream_events``)."""
        kind = event.get("type")
        if kind == "run.started" and self.path is None and not self.done:
            self._start(event)
            return
        if self.path is None or self.done:
            return
        if kind == "answer.delta":
            self._pending += str(event.get("delta") or "")
            if time.monotonic() - self._last_delta >= DELTA_SECONDS:
                self._flush()
            return
        self._flush()
        if kind == "status":
            message = event.get("message") or str(
                event.get("stage") or "working"
            ).replace("_", " ")
            # A delegate's harness progress names its step and its run.
            self._write(
                "status",
                message=message,
                **{
                    key: event[key]
                    for key in ("span_id", "harness_run_id")
                    if event.get(key)
                },
            )
        elif kind in {"tool.started", "tool.completed"}:
            self._write(
                kind,
                **{
                    key: event.get(key)
                    for key in ("tool_name", "span_id", "status", "reason_code")
                    if event.get(key) is not None
                },
            )
        elif kind == "approval.required":
            self._write("status", message="Paused for approval in the CLI")
        elif kind == "run.done":
            self._write("turn.done", status=event.get("status") or "completed")
            self.done = True

    def close(self, status: str = "interrupted") -> None:
        """End a turn the stream left unfinished (Ctrl-C, an error)."""
        if self.path is not None and not self.done:
            self._flush()
            self._write("turn.done", status=status)
        self.done = True


def prune(directory: Path, keep_seconds: float = KEEP_SECONDS) -> None:
    cutoff = time.time() - keep_seconds
    for path in directory.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


class LiveTail:
    """Follow the turns written for one agent, optionally one conversation.

    The first ``poll`` replays turns still running (so a page opened mid-turn
    catches up) and skips finished ones; later polls return only new lines.
    """

    def __init__(self, agent_id: str, memory_id: Optional[str] = None):
        self.directory = agent_dir(agent_id)
        self.memory_id = memory_id or None
        self.offsets: Dict[Path, int] = {}
        self.started = False

    def _read(self, path: Path, offset: int) -> tuple[List[Dict[str, Any]], int]:
        try:
            with open(path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:
            return [], offset
        end = chunk.rfind(b"\n")
        if end < 0:
            return [], offset
        events = []
        for raw in chunk[: end + 1].splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events, offset + end + 1

    def poll(self) -> List[Dict[str, Any]]:
        if self.directory is None or not self.directory.is_dir():
            self.started = True
            return []
        first, self.started = not self.started, True
        now = time.time()
        found = []
        for path in self.directory.glob("*.jsonl"):
            try:
                found.append((path.stat().st_mtime, path))
            except OSError:
                continue
        events: List[Dict[str, Any]] = []
        for mtime, path in sorted(found, key=lambda item: item[1].name):
            if path not in self.offsets:
                if first and now - mtime > RUNNING_SECONDS:
                    self.offsets[path] = path.stat().st_size if path.exists() else 0
                    continue
                self.offsets[path] = 0
            lines, self.offsets[path] = self._read(path, self.offsets[path])
            if first and any(line.get("type") == "turn.done" for line in lines):
                continue
            events.extend(
                line
                for line in lines
                if not self.memory_id or line.get("memory_id") == self.memory_id
            )
        for path in list(self.offsets):
            if not path.exists():
                del self.offsets[path]
        return events
