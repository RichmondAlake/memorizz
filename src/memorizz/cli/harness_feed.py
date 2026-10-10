"""Show what a harness does while it works: its messages, the commands it
runs and their output, tool calls and file edits.

The reply stream only says what kind of work a harness is doing. The terminal
is the user's own, so it also asks for the harness run's ledger events (which
MemoRizz has already redacted) through ``harness_event_listener`` and prints
them above the live status, the way a coding agent's own terminal would.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from typing import Any, Dict, List, Optional

from rich.text import Text

from ..metaharness.catalog import HARNESS_LABELS

# Enough to follow along; the whole run stays in `memorizz harness show`.
MAX_MESSAGE_LINES = 8
MAX_OUTPUT_LINES = 3
MAX_LINE = 160

_SHELL_WRAPPER = re.compile(
    r"^(?:/\S+/)?(?:ba|z)?sh -l?c (?:(['\"])(.*)\1|([^'\"].*))$", re.S
)
_MCP_PREFIX = re.compile(r"^mcp__[^_]+(?:_[^_]+)*?__")


def _clip(text: Any, limit: int = MAX_LINE) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _lines(text: Any) -> List[str]:
    return [line.rstrip() for line in str(text or "").splitlines() if line.strip()]


def command_text(command: Any) -> str:
    """The command as typed, without the ``zsh -lc "..."`` Codex wraps it in."""
    if isinstance(command, list):
        command = " ".join(str(part) for part in command)
    value = str(command or "").strip()
    match = _SHELL_WRAPPER.match(value)
    if match:
        value = (match.group(2) or match.group(3) or "").replace('\\"', '"')
    return value


def tool_name(data: Dict[str, Any]) -> str:
    name = str(data.get("name") or data.get("tool") or data.get("type") or "tool")
    return _MCP_PREFIX.sub("", name)


def _arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return _clip(value, 100)
    if isinstance(value, dict):
        parts = [
            f"{key}={_clip(item if isinstance(item, str) else json.dumps(item, default=str), 60)}"
            for key, item in value.items()
            if item not in (None, "", [], {})
        ]
        return _clip(", ".join(parts), 100)
    return _clip(value, 100) if value else ""


def _result_text(data: Dict[str, Any]) -> str:
    for key in ("content", "text", "tool_use_result", "output"):
        value = data.get(key)
        if value:
            if isinstance(value, (list, dict)):
                value = json.dumps(value, default=str)
            return str(value)
    return ""


class _Run:
    def __init__(self, harness: str) -> None:
        self.harness = harness
        self.held: Optional[str] = None  # latest message: may be the answer
        self.seen: set = set()
        self.finished = False


class HarnessFeed:
    """Collect harness events from the relay thread; render them on demand.

    ``listener`` is handed to ``harness_event_listener`` and may be called
    from any thread; ``drain`` and ``finish`` run on the terminal's thread.

    A comparison or plan wants every line named (``label_every_line``), and
    prints each harness's answer when its run ends (``show_answers``): no
    single chat answer follows it there.
    """

    def __init__(
        self, *, label_every_line: bool = False, show_answers: bool = False
    ) -> None:
        self._queue: "queue.SimpleQueue" = queue.SimpleQueue()
        self._runs: Dict[str, _Run] = {}
        self.label_every_line = label_every_line
        self.show_answers = show_answers

    def listener(self, run_id: str, harness: str, event: Dict[str, Any]) -> None:
        self._queue.put((run_id, harness, event))

    def drain(self) -> List[Text]:
        out: List[Text] = []
        while True:
            try:
                run_id, harness, event = self._queue.get_nowait()
            except queue.Empty:
                return out
            run = self._runs.get(run_id)
            if run is None:
                run = self._runs[run_id] = _Run(harness)
                out.append(self._header(run))
            if harness and run.harness != harness:
                run.harness = harness
            out.extend(self._render(run, event))

    def finish(self, answer: str = "") -> List[Text]:
        """Print the messages still held back, except the one that is the answer."""
        out = self.drain()
        for run in self._runs.values():
            if run.held and run.held.strip() != (answer or "").strip():
                out.extend(self._message(run, run.held))
            run.held = None
        return out

    # -- rendering -----------------------------------------------------------

    def _label(self, run: _Run) -> str:
        return HARNESS_LABELS.get(run.harness, run.harness or "harness")

    def _header(self, run: _Run) -> Text:
        return Text.assemble(("● ", "bold magenta"), (self._label(run), "bold"))

    def _line(self, run: _Run, *parts: Any) -> Text:
        prefix = []
        if self.label_every_line or (
            len([r for r in self._runs.values() if not r.finished]) > 1
        ):
            prefix = [(f"{self._label(run)} · ", "dim magenta")]
        return Text.assemble("  ", *prefix, *parts)

    def _message(self, run: _Run, text: str) -> List[Text]:
        lines = _lines(text)
        shown = [
            self._line(run, (_clip(line, 400), ""))
            for line in lines[:MAX_MESSAGE_LINES]
        ]
        if len(lines) > MAX_MESSAGE_LINES:
            more = len(lines) - MAX_MESSAGE_LINES
            shown.append(self._line(run, (f"… {more} more lines", "dim")))
        return shown

    def _flush_held(self, run: _Run) -> List[Text]:
        if not run.held:
            return []
        text, run.held = run.held, None
        return self._message(run, text)

    def _render(self, run: _Run, event: Dict[str, Any]) -> List[Text]:
        kind = event.get("type")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if kind == "run.started":
            return []
        if kind == "run.finished":
            out = self._flush_held(run) if self.show_answers else []
            run.finished = True
            return out
        if kind == "message":
            if data.get("role") not in (None, "", "assistant"):
                return []
            # Hold the latest message back: if nothing follows it, it is the
            # harness's answer, which the chat prints in full anyway.
            out = self._flush_held(run)
            run.held = str(data.get("text") or "") or None
            return out
        rendered = self._activity(run, kind, data)
        if not rendered:
            return []
        return self._flush_held(run) + rendered

    def _activity(self, run: _Run, kind: Any, data: Dict[str, Any]) -> List[Text]:
        key = str(data.get("id") or data.get("call_id") or "")
        if kind == "command":
            command = command_text(data.get("command"))
            done = data.get("status") not in (None, "", "in_progress", "started") or (
                data.get("exit_code") is not None
            )
            out: List[Text] = []
            if not key or key not in run.seen:
                if key:
                    run.seen.add(key)
                out.append(
                    self._line(run, ("$ ", "bold cyan"), (_clip(command), "cyan"))
                )
            if done:
                output = _lines(
                    data.get("aggregated_output")
                    or data.get("output")
                    or data.get("stdout")
                    or ""
                )
                for line in output[:MAX_OUTPUT_LINES]:
                    out.append(self._line(run, ("  " + _clip(line), "dim")))
                if len(output) > MAX_OUTPUT_LINES:
                    more = len(output) - MAX_OUTPUT_LINES
                    out.append(self._line(run, (f"  … {more} more lines", "dim")))
                code = data.get("exit_code")
                if code not in (None, 0, "0"):
                    out.append(self._line(run, (f"  exit {code}", "red")))
            return out
        if kind == "tool_call":
            if data.get("harness_run_id") and not data.get("name"):
                return []  # a delegate's progress marker; its run reports itself
            if key and key in run.seen:
                return []
            if key:
                run.seen.add(key)
            if data.get("type") == "web_search" or tool_name(data) == "web_search":
                query = data.get("query") or (data.get("input") or {})
                if isinstance(query, dict):
                    query = query.get("query")
                return [
                    self._line(
                        run,
                        ("⌕ web search ", "bold blue"),
                        (_clip(query or "", 100), ""),
                    )
                ]
            marker = "⇢ " if data.get("subagent") else "⚙ "
            arguments = _arguments(data.get("input") or data.get("arguments"))
            return [
                self._line(
                    run,
                    (marker, "bold yellow"),
                    (tool_name(data), "yellow"),
                    (f"  {arguments}" if arguments else "", "dim"),
                )
            ]
        if kind == "tool_result":
            text = _lines(_result_text(data))
            if not text:
                return []
            error = data.get("is_error") in (True, "true", "True")
            return [
                self._line(
                    run, ("  ↳ " + _clip(text[0], 120), "red" if error else "dim")
                )
            ]
        if kind == "file_change":
            changes = data.get("changes")
            if not isinstance(changes, list):
                changes = [data] if data.get("path") else []
            return [
                self._line(
                    run,
                    ("✎ ", "bold green"),
                    (f"{change.get('kind') or 'edit'} ", "green"),
                    (_clip(change.get("path"), 120), ""),
                )
                for change in changes
                if isinstance(change, dict)
            ]
        if kind == "reasoning":
            text = _lines(data.get("text"))
            if not text:
                return []
            return [self._line(run, ("∴ " + _clip(text[0], 140), "dim italic"))]
        if kind == "error":
            return [self._line(run, ("✗ " + _clip(data.get("error")), "red"))]
        return []


def plain(lines: List[Text]) -> str:
    """The same lines without styling, for stderr in one-shot runs."""
    return "".join(line.plain + "\n" for line in lines)


class _NoStream:
    """Stands in for a reply stream when only the feed is wanted."""

    agent = None

    def emit(self, *args: Any, **kwargs: Any) -> None:
        pass


class FeedFollower:
    """Print the feed of one or more harness runs while they run.

    For code that waits on runs itself: ``follow`` each run once its id is
    known (a comparison starts its runs together, a plan one stage at a
    time), ``done`` when it ends, ``close`` at the end. ``write`` gets each
    batch of lines from the relay threads, one batch at a time.
    """

    def __init__(self, service: Any, write: Any, **feed_options: Any) -> None:
        self.service, self._write = service, write
        self.feed = HarnessFeed(**feed_options)
        self._lock = threading.Lock()
        self._following: Dict[str, Any] = {}
        self._done: set = set()

    def follow(self, run_id: Any, harness: Any) -> None:
        from ..streaming import HarnessProgress, harness_event_listener

        if not run_id or run_id in self._following:
            return
        token = harness_event_listener.set(self._listen)
        try:
            progress = HarnessProgress(_NoStream(), self.service, harness)
        finally:
            harness_event_listener.reset(token)
        self._following[run_id] = progress
        progress.start(run_id)

    def _emit(self, lines: List[Text]) -> None:
        if lines:
            self._write(lines)

    def _listen(self, run_id: str, harness: str, event: Dict[str, Any]) -> None:
        with self._lock:
            self.feed.listener(run_id, harness, event)
            self._emit(self.feed.drain())

    def done(self, run_id: Any) -> None:
        progress = self._following.get(run_id)
        if progress is not None and run_id not in self._done:
            self._done.add(run_id)
            progress.stop()

    def close(self, answer: str = "") -> None:
        for run_id in list(self._following):
            self.done(run_id)
        with self._lock:
            self._emit(self.feed.finish(answer))


class RunFollower(FeedFollower):
    """Print one harness run's feed to ``stream`` while it runs.

    For commands that wait on a run themselves (``memorizz harness run``):
    start it with the run's id before the blocking call, then ``close`` with
    the final answer, which the command prints itself.
    """

    def __init__(self, service: Any, run_id: str, harness: Any, stream: Any) -> None:
        super().__init__(service, lambda lines: write_plain(stream, lines))
        self.stream = stream
        self.follow(run_id, harness)


def write_plain(stream: Any, lines: List[Text]) -> None:
    stream.write(plain(lines))
    stream.flush()
