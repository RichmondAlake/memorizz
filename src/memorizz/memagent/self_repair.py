# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""How a self-aware agent sees what went wrong with it, and fixes its source.

Three parts:

- An error log per agent (``<MEMORIZZ_HOME>/self_aware/<agent>/errors.jsonl``).
  Traces never keep an exception's message; this local file does, with the
  frames inside MemoRizz, because a fix needs the file and the line.
- ``recent_failures``: that log together with the failed steps in the agent's
  own traces.
- ``SelfRepair``: a coding model reads the failures and the agent's copy of
  the source, recommends a fix, and on request makes it there, checks that
  the copy still imports, and keeps it as a version.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .self_source import PACKAGE, SelfSource, SelfSourceError, installed_package_dir

logger = logging.getLogger(__name__)

#: The agent whose turn is running in this context, for the error log.
current_agent_id: ContextVar[Optional[str]] = ContextVar(
    "memorizz_self_aware_agent", default=None
)

MAX_LOG_ENTRIES = 300
_MAX_TEXT = 4000
_lock = threading.Lock()
_watched: Dict[str, "ErrorLog"] = {}
_handler: Optional[logging.Handler] = None


def _is_openai_host(base_url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(str(base_url)).hostname or "").lower()
    return host == "openai.com" or host.endswith(".openai.com")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_frames(tb: Any) -> List[Dict[str, Any]]:
    """The frames of a traceback that are inside MemoRizz, innermost last,
    with paths as they are in an agent's copy (``memorizz/...``)."""
    package = installed_package_dir()
    frames: List[Dict[str, Any]] = []
    for frame in traceback.extract_tb(tb) if tb is not None else []:
        try:
            relative = Path(frame.filename).resolve().relative_to(package)
        except (ValueError, OSError):
            continue
        frames.append(
            {
                "file": f"{PACKAGE}/{relative.as_posix()}",
                "line": frame.lineno,
                "function": frame.name,
                "code": (frame.line or "").strip()[:200],
            }
        )
    return frames[-12:]


class ErrorLog:
    """One agent's errors, newest last, kept on this machine only."""

    def __init__(self, agent_id: str, *, home: Optional[Path] = None):
        self.path = SelfSource(agent_id, home=home).folder / "errors.jsonl"

    def record(self, entry: Dict[str, Any]) -> None:
        line = json.dumps({"at": _now(), **entry}, ensure_ascii=False, default=str)
        with _lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                lines = (
                    self.path.read_text(encoding="utf-8").splitlines()
                    if self.path.exists()
                    else []
                )
                lines = (lines + [line])[-MAX_LOG_ENTRIES:]
                self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except OSError:
                pass

    def record_exception(self, exc: BaseException, where: str = "") -> None:
        self.record(
            {
                "source": "exception",
                "where": where,
                "error": f"{type(exc).__name__}: {exc}"[:_MAX_TEXT],
                "frames": source_frames(exc.__traceback__),
            }
        )

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        entries: List[Dict[str, Any]] = []
        for line in lines[-max(1, int(limit)) :]:
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries

    def clear(self) -> None:
        with _lock:
            try:
                self.path.unlink()
            except OSError:
                pass


class _Capture(logging.Handler):
    """Copy MemoRizz's own warnings and errors into the running agent's log."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            agent_id = current_agent_id.get()
            log = _watched.get(agent_id) if agent_id else None
            if log is None or record.name == __name__:
                return
            entry: Dict[str, Any] = {
                "source": "log",
                "level": record.levelname,
                "where": record.name,
                "error": record.getMessage()[:_MAX_TEXT],
            }
            # A message logged while handling an exception rarely carries it.
            exc_info = record.exc_info if record.exc_info else sys.exc_info()
            if exc_info and exc_info[1] is not None:
                exc = exc_info[1]
                named = f"{type(exc).__name__}: {exc}"
                if named not in entry["error"]:
                    entry["error"] = f"{entry['error']} | {named}"[:_MAX_TEXT]
                entry["frames"] = source_frames(exc_info[2])
                last = traceback.extract_tb(exc_info[2])[-1:] if exc_info[2] else []
                if last and not _relative(last[0].filename).startswith(PACKAGE + "/"):
                    # Raised outside MemoRizz: a tool of the user's, or a library.
                    entry["raised_in"] = f"{last[0].filename}:{last[0].lineno}"
            else:
                entry["frames"] = [
                    {
                        "file": _relative(record.pathname),
                        "line": record.lineno,
                        "function": record.funcName,
                    }
                ]
            log.record(entry)
        except Exception:
            pass


def _relative(path: str) -> str:
    try:
        relative = Path(path).resolve().relative_to(installed_package_dir())
        return f"{PACKAGE}/{relative.as_posix()}"
    except (ValueError, OSError):
        return str(path)


def watch(agent_id: str, *, home: Optional[Path] = None) -> ErrorLog:
    """Start keeping this agent's errors. Safe to call again."""
    global _handler
    log = ErrorLog(agent_id, home=home)
    with _lock:
        _watched[str(agent_id)] = log
        if _handler is None:
            _handler = _Capture(level=logging.WARNING)
            logging.getLogger(PACKAGE).addHandler(_handler)
    return log


def unwatch(agent_id: str) -> None:
    with _lock:
        _watched.pop(str(agent_id), None)


def watched(agent_id: str) -> Optional[ErrorLog]:
    return _watched.get(str(agent_id))


# --- the agent's own traces ---


def _trace_events(agent: Any, limit: int, **filters: Any) -> List[Dict[str, Any]]:
    provider = getattr(agent, "memory_provider", None)
    query = getattr(provider, "query_trace_events", None)
    if not callable(query):
        return []
    try:
        page = query(limit=limit, agent_ids=[str(agent.agent_id)], **filters)
    except Exception as exc:
        # Worth hearing about: without its traces the agent sees only its log.
        logger.warning("Could not read the agent's own traces: %s", exc)
        return []
    items = page.get("items") if isinstance(page, dict) else page
    return [item for item in (items or []) if isinstance(item, dict)]


# The coding model's own turns are traced under the agent, in memories with
# this prefix, so its work can be read in Traces.
WORK_MEMORY_PREFIX = "self-repair-"
WORK_TAGS = {"work": "self_repair"}


def _is_repair_work(event: Dict[str, Any]) -> bool:
    if str(event.get("memory_id") or "").startswith(WORK_MEMORY_PREFIX):
        return True
    tags = event.get("trace_tags")
    return isinstance(tags, dict) and tags.get("work") == WORK_TAGS["work"]


def _trace_failed(event: Dict[str, Any]) -> bool:
    status = str(event.get("status") or "").lower()
    if "approval_required" in (event.get("error_code"), status):
        return False
    if status == "cancelled":
        return False
    return event.get("success") is False or status in {"error", "failed", "failure"}


def _compact(event: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "timestamp", "trace_kind", "title", "status", "tool_name", "error_code",
        "outcome_reason_code", "duration_ms", "turn_id", "model", "iteration",
    )  # fmt: skip
    row = {key: event[key] for key in keep if event.get(key) not in (None, "")}
    content = event.get("content")
    if content:
        row["content"] = str(content)[:1500]
    return row


def recent_failures(agent: Any, limit: int = 20) -> List[Dict[str, Any]]:
    """What went wrong lately: the error log, then failed trace steps."""
    limit = max(1, min(int(limit or 20), 100))
    failures: List[Dict[str, Any]] = []
    log = watched(agent.agent_id) or ErrorLog(agent.agent_id)
    for entry in log.recent(limit):
        failures.append({"from": "error log", **entry})
    for event in _trace_events(agent, 400):
        # A slip the coding model made while fixing is not the agent's bug.
        if _is_repair_work(event):
            continue
        if _trace_failed(event):
            failures.append({"from": "trace", **_compact(event)})
    failures.sort(key=lambda item: str(item.get("at") or item.get("timestamp") or ""))
    return failures[-limit:]


def read_trace(agent: Any, turn_id: str, limit: int = 120) -> List[Dict[str, Any]]:
    """Every recorded step of one of the agent's turns, in order."""
    events = _trace_events(agent, max(1, min(int(limit), 400)), turn_id=str(turn_id))
    events.sort(key=lambda item: str(item.get("timestamp") or ""))
    return [_compact(event) for event in events]


# --- recommending and making a fix ---

# Factory's coding agent, run without a person at the keyboard.
DROID_COMMAND = "droid"
DROID_TIMEOUT_SECONDS = 20 * 60
# The tools Droid is left with (`droid exec --only-tools`): reading for a
# diagnosis, and editing files as well for a fix.
DROID_READ_TOOLS = ("Read", "LS", "Grep", "Glob")
DROID_EDIT_TOOLS = ("Edit", "ApplyPatch", "Create")

_DROID_RULES = (
    "This folder holds an AI agent's own copy of the MemoRizz source code "
    "(Python, under memorizz/). The agent ran on this code and some of its "
    "turns failed; its recent errors are listed at the end, each with the "
    "file and line inside memorizz/ it came from. Work from evidence: open "
    "the files and lines the errors name and find the callers before you "
    "decide. Not every failure is a bug in MemoRizz: a missing key, an "
    "unreachable model, a wrong setting or a fault in a tool the user wrote "
    "is fixed outside this source; say so plainly and change no code for it."
)

_DROID_RECOMMEND = (
    _DROID_RULES + " Do not change any file and do not run any command or "
    "script (running one is not permitted here and ends the run). Answer "
    "with these four headings "
    "and nothing else:\n"
    "Problem: what fails, in one or two sentences.\n"
    "Cause: the file and line, and why it fails. Or 'Not in the source' and why.\n"
    "Fix: the exact change you would make, or what to change outside the code.\n"
    "Risk: what else the change could affect.\n"
    "If there are no errors, say that under Problem and stop."
)

_EXPLAIN = (
    "You changed an AI agent's copy of the MemoRizz source to fix an error. "
    "Explain the change to the person who owns the agent. Use only the "
    "error and the diff you are given; do not invent anything. Answer with "
    "one JSON object and nothing else:\n"
    '{"summary": "what was wrong and what the change does, in two sentences",\n'
    ' "root_cause": "why the old code failed, naming the file and line",\n'
    ' "changes": [{"file": "memorizz/...", "line": 0, '
    '"what": "what this edit does", "why": "why it is needed"}],\n'
    ' "why_this_way": "why this fix and not a larger or different one",\n'
    ' "risk": "what else could be affected, or \'None seen\'",\n'
    ' "check_it": "one thing the owner can do to confirm the fix"}\n'
    "One entry in changes for each separate edit in the diff. line is the "
    "line number of the edit in the new file. Plain words, no Markdown."
)

_DROID_FIX = (
    _DROID_RULES + " Make the smallest change that fixes the cause, in files "
    "under memorizz/ only. Do not reformat, rename or tidy anything else, do "
    "not create commits and do not touch anything outside this folder. "
    "You may read and edit files only: do not run any command, script or "
    "test (running one is not permitted here and ends the whole run; the "
    "source is checked for you afterwards). "
    "Finish with one line starting 'VERSION:' that says what you changed in "
    "under 15 words, then a short account of the fix. If no code change is "
    "right, change nothing and explain why."
)


def _droid_result(output: str) -> Optional[Dict[str, Any]]:
    """The result Droid prints with ``-o json``: the last line that is a JSON
    object of type "result"."""
    for line in reversed(str(output or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            found = json.loads(line)
        except ValueError:
            continue
        if isinstance(found, dict) and found.get("type") == "result":
            return found
    return None


# The most errors kept in the account of one fix.
_STORY_ERRORS = 8

_RULES = (
    "You maintain an AI agent's own copy of the MemoRizz source code "
    "(Python, under memorizz/). The agent ran on this code and some of its "
    "turns failed. Work from evidence: call recent_errors first, open the "
    "files and lines the errors name with read_source, and use search_source "
    "to find callers. Never guess at code you have not read.\n"
    "Not every failure is a bug in MemoRizz. A missing key, an unreachable "
    "model, a wrong setting, or a fault in a tool the user wrote is fixed "
    "outside this source: say so plainly and do not change code for it.\n"
)

_RECOMMEND = (
    _RULES + "You may only read. Answer with these four headings and nothing else:\n"
    "Problem: what fails, in one or two sentences.\n"
    "Cause: the file and line, and why it fails. Or 'Not in the source' and why.\n"
    "Fix: the exact change you would make, or what to change outside the code.\n"
    "Risk: what else the change could affect.\n"
    "If there are no errors, say that under Problem and stop."
)

_FIX = (
    _RULES + "Make the smallest change that fixes the cause. Use edit_source "
    "with old_text copied exactly from read_source (without the line numbers). "
    "Do not reformat, rename or tidy anything else. After editing, call "
    "check_source; if it fails, read its output and repair your edit. "
    "When the check passes, finish with one line starting 'VERSION:' that "
    "says what you changed in under 15 words, then a short account of the "
    "fix. If no code change is right, change nothing and explain why."
)


def _explanation(answer: str) -> Optional[Dict[str, Any]]:
    """The explainer's JSON, however the model wrapped it. None if unreadable."""
    text = str(answer or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out: Dict[str, Any] = {}
    for key in ("summary", "root_cause", "why_this_way", "risk", "check_it"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()[:1200]
    changes = []
    for item in data.get("changes") if isinstance(data.get("changes"), list) else []:
        if not isinstance(item, dict) or not item.get("file"):
            continue
        try:
            line = int(item.get("line") or 0)
        except (TypeError, ValueError):
            line = 0
        changes.append(
            {
                "file": str(item["file"])[:300],
                "line": max(0, line),
                "what": str(item.get("what") or "").strip()[:600],
                "why": str(item.get("why") or "").strip()[:800],
            }
        )
    if changes:
        out["changes"] = changes[:20]
    return out or None


class SelfRepair:
    """Recommend and make fixes to one agent's copy of the source."""

    def __init__(self, agent: Any, *, on_step: Optional[Callable[[str], None]] = None):
        self.agent = agent
        self.source: SelfSource = agent.self_source()
        self._report = on_step or (lambda text: None)
        # What one run did and was shown, kept with the version it makes.
        self._steps: List[Dict[str, Any]] = []
        self._seen: List[Dict[str, Any]] = []
        self._traces: List[Dict[str, Any]] = []
        # Where the coding model's own turns are traced during one run.
        self._work_memory: Optional[str] = None
        self._work_trace: Optional[Dict[str, Any]] = None

    def on_step(self, text: str) -> None:
        self._steps.append({"at": _now(), "text": str(text)})
        self._report(text)

    def _begin(self) -> None:
        self._steps, self._seen, self._traces = [], [], []
        self._work_memory, self._work_trace = None, None

    def _saw(self, failures: Any) -> None:
        known = {json.dumps(item, sort_keys=True, default=str) for item in self._seen}
        for item in failures if isinstance(failures, list) else []:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in known:
                known.add(key)
                self._seen.append(item)

    @property
    def _diagnosis_path(self) -> Path:
        return self.source.folder / "last_diagnosis.json"

    # The model that reads and writes code; the agent's own unless one is set.
    def coding_model(self) -> Any:
        config = (self.agent.get_self_aware_config() or {}).get("coding_model")
        if isinstance(config, dict) and config.get("model"):
            from ..llms.llm_factory import create_llm_provider

            settings = dict(config)
            key_env = settings.pop("api_key_env", None)
            if key_env:
                key = os.environ.get(str(key_env), "").strip()
                if not key:
                    raise RuntimeError(
                        f"The coding model's key is not set: add {key_env} to "
                        "the environment MemoRizz runs in"
                    )
                settings["api_key"] = key
            elif settings.get("base_url") and not _is_openai_host(settings["base_url"]):
                # Someone else's endpoint must never be sent the OpenAI key
                # that happens to be in the environment.
                settings["api_key"] = "sk-local"
            return create_llm_provider(settings)
        return self.agent.model

    def coding_agent(self) -> Optional[Dict[str, Any]]:
        """The outside coding agent that does this agent's fixes, if one is set."""
        config = (self.agent.get_self_aware_config() or {}).get("coding_agent")
        return dict(config) if isinstance(config, dict) and config.get("kind") else None

    def coding_model_label(self) -> str:
        outside = self.coding_agent()
        if outside:
            return "droid" + (f" · {outside['model']}" if outside.get("model") else "")
        config = (self.agent.get_self_aware_config() or {}).get("coding_model")
        if isinstance(config, dict) and config.get("model"):
            return f"{config.get('provider') or 'openai'} · {config['model']}"
        model = getattr(self.agent.model, "model", None)
        return f"{model} (the agent's own model)" if model else "the agent's own model"

    def _tools(self, *, write: bool) -> List[Callable[..., Any]]:
        source, agent, step = self.source, self.agent, self.on_step

        def guarded(call: Callable[[], Any]) -> Any:
            try:
                return call()
            except SelfSourceError as exc:
                return {"ok": False, "error": str(exc)}

        def recent_errors(limit: int = 15) -> Any:
            """The agent's most recent errors: messages, and the source files and lines they came from."""
            step("Reading its errors")
            found = recent_failures(agent, limit)
            self._saw(found)
            return found or "No errors are recorded."

        def read_turn_trace(turn_id: str) -> Any:
            """Every recorded step of one turn, by the turn_id an error names."""
            step("Reading a trace")
            found = read_trace(agent, turn_id)
            self._traces.append({"turn_id": str(turn_id), "steps": len(found)})
            return found or "No steps are recorded for that turn."

        def list_source(path: str = "") -> Any:
            """List a folder of the source, for example 'memorizz/llms'."""
            return guarded(lambda: source.list_files(path))

        def read_source(path: str, start_line: int = 1, end_line: int = 0) -> Any:
            """Read lines of a source file, numbered. Give a line range for a long file."""
            step(f"Reading {str(path)[-48:]}")
            return guarded(lambda: source.read_file(path, start_line, end_line))

        def search_source(text: str, path: str = "") -> Any:
            """Find lines containing an exact piece of text, in one file or folder."""
            step(f"Searching for {str(text)[:32]}")
            return guarded(lambda: source.search(text, path))

        tools: List[Callable[..., Any]] = [
            recent_errors, read_turn_trace, list_source, read_source, search_source,
        ]  # fmt: skip
        if not write:
            return tools

        def edit_source(path: str, old_text: str, new_text: str) -> Any:
            """Replace one exact piece of a source file. old_text must occur exactly once."""
            step(f"Editing {str(path)[-48:]}")
            return guarded(lambda: source.edit_file(path, old_text, new_text))

        def create_source(path: str, content: str) -> Any:
            """Create a new source file."""
            step(f"Creating {str(path)[-48:]}")
            return guarded(lambda: source.create_file(path, content))

        def check_source() -> Any:
            """Check that the edited source still compiles and imports."""
            step("Checking the source")
            return guarded(source.check)

        return tools + [edit_source, create_source, check_source]

    def _run(self, instruction: str, task: str, *, write: bool) -> str:
        outside = self.coding_agent()
        if outside:
            return self._run_droid(outside, task, write=write)
        return self._ask(
            instruction,
            task,
            tools=self._tools(write=write),
            max_steps=40 if write else 25,
        )

    def _ask(
        self,
        instruction: str,
        task: str,
        *,
        tools: Optional[List[Callable[..., Any]]] = None,
        max_steps: int = 4,
    ) -> str:
        """One piece of work by the coding model. It is traced under the
        agent, in a memory of its own, so the work can be read in Traces
        and never mixes with the agent's conversations."""
        from .core import MemAgent

        store = getattr(self.agent, "memory_provider", None)
        traced = store is not None and getattr(self.agent, "observability", True)
        worker = MemAgent(
            model=self.coding_model(),
            instruction=instruction,
            tools=tools or [],
            max_steps=max_steps,
            auto_register=False,
            streaming=False,
            **(
                {"memory_provider": store, "agent_id": self.agent.agent_id}
                if traced
                else {}
            ),
        )
        if not traced:
            return str(worker.run(task) or "").strip()
        if not self._work_memory:
            # One memory and one thread for the whole run, so the fix and
            # its explanation read as consecutive turns in Traces.
            self._work_memory = f"{WORK_MEMORY_PREFIX}{uuid.uuid4().hex[:12]}"
            self._work_trace = {
                "agent_id": str(self.agent.agent_id),
                "memory_id": self._work_memory,
                "thread_id": str(uuid.uuid4()),
            }
        return str(
            worker.run(
                task,
                memory_id=self._work_memory,
                thread_id=self._work_trace["thread_id"],
                trace_tags=dict(WORK_TAGS),
            )
            or ""
        ).strip()

    def explain(
        self, diff: str, errors: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Why the change was made, from the coding model, as plain fields.

        None when an outside coding agent did the work or the model's
        answer could not be read; the version is kept either way."""
        if self.coding_agent() or not diff.strip():
            return None
        seen = json.dumps(errors[-3:], ensure_ascii=False, default=str)[:4000]
        try:
            answer = self._ask(
                _EXPLAIN,
                f"The errors that led to the fix:\n{seen}\n\nThe diff you made:\n"
                + diff[:12000],
            )
        except Exception as exc:
            logger.warning("The fix was kept without an explanation: %s", exc)
            return None
        return _explanation(answer)

    def _run_droid(self, config: Dict[str, Any], task: str, *, write: bool) -> str:
        """Hand the task to Factory's Droid, working in the agent's copy.

        Droid has none of MemoRizz's tools, so what the coding model would
        have looked up is written into its prompt: the recent errors and
        where the source is. Without ``--auto`` Droid can only read; a fix
        runs with ``--auto low`` (file edits, no commands)."""
        import shutil
        import subprocess
        import tempfile

        command = str(config.get("command") or DROID_COMMAND)
        found = shutil.which(command)
        if not found:
            raise RuntimeError(
                "Droid is not installed here: install Factory's CLI "
                "(https://docs.factory.com/cli) and make `droid` available on PATH"
            )
        if not os.environ.get("FACTORY_API_KEY", "").strip():
            raise RuntimeError(
                "Droid needs a Factory key: set FACTORY_API_KEY where MemoRizz runs"
            )
        failures = recent_failures(self.agent, 15)
        # Droid is shown these, so they count as read in the fix's account.
        self._saw(failures)
        self.on_step("Reading its errors")
        prompt = (
            (_DROID_FIX if write else _DROID_RECOMMEND)
            + "\n\nTask: "
            + task
            + "\n\nThe agent's recent errors, oldest first (JSON):\n"
            + json.dumps(
                failures[-_STORY_ERRORS:], ensure_ascii=False, default=str, indent=1
            )
        )
        arguments = [found, "exec", "--cwd", str(self.source.root), "-o", "json"]
        autonomy = str(config.get("autonomy") or "low")
        if write:
            arguments += ["--auto", autonomy]
        if not (write and autonomy != "low"):
            # Files only: no commands, web, messages or sub-agents. At
            # "medium" Droid is trusted to run tests, so its tools are its own.
            tools = DROID_READ_TOOLS + (DROID_EDIT_TOOLS if write else ())
            arguments += ["--only-tools", ",".join(tools)]
        if config.get("model"):
            arguments += ["-m", str(config["model"])]
        # Only what Droid needs: not the keys of other services.
        environment = {
            name: os.environ[name]
            for name in (
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "TERM",
                "TMPDIR",
                "FACTORY_API_KEY",
            )
            if os.environ.get(name)
        }
        self.on_step(
            "Droid is fixing the source" if write else "Droid is reading the source"
        )
        # The prompt is a file outside the copy, so it is never kept as a version.
        with tempfile.TemporaryDirectory(prefix="memorizz-droid-") as folder:
            prompt_path = Path(folder) / "task.md"
            prompt_path.write_text(prompt, encoding="utf-8")
            try:
                done = subprocess.run(
                    arguments + ["-f", str(prompt_path)],
                    cwd=str(self.source.root),
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=DROID_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Droid did not finish within {DROID_TIMEOUT_SECONDS // 60} minutes"
                ) from exc
        result = _droid_result(done.stdout)
        if done.returncode != 0 or result is None or result.get("is_error"):
            detail = (
                str((result or {}).get("result") or "").strip()
                or (done.stderr or done.stdout or "").strip()[-600:]
                or f"exit code {done.returncode}"
            )
            if "insufficient permission" in detail.lower():
                # Droid tried to run something its level does not allow.
                detail = (
                    "it tried to run a command, which it may not do here; "
                    "nothing it changed was kept. Try again, or raise "
                    "coding_agent's autonomy to 'medium' to let it run tests"
                )
            raise RuntimeError(f"Droid stopped early: {detail}")
        turns = result.get("num_turns")
        self.on_step(
            f"Droid finished in {turns} turn{'s' if turns != 1 else ''}"
            if isinstance(turns, int)
            else "Droid finished"
        )
        return str(result.get("result") or "").strip()

    def recommend(self, focus: Optional[str] = None) -> Dict[str, Any]:
        """Read the failures and the source; say what is wrong and how to fix it."""
        self.source.ensure()
        self._begin()
        started = _now()
        failures = recent_failures(self.agent, 15)
        task = "Find the cause of this agent's recent errors and recommend a fix."
        if focus:
            task += f" Look at this in particular: {focus}"
        self.on_step("Looking for the cause")
        text = self._run(_RECOMMEND, task, write=False)
        diagnosis = {
            "text": text,
            "coding_model": self.coding_model_label(),
            "started_at": started,
            "finished_at": _now(),
            "prompt": {"instruction": _RECOMMEND, "task": task},
            "errors_read": self._seen[-_STORY_ERRORS:],
            "traces_read": list(self._traces),
            "steps": list(self._steps),
            "work_trace": self._work_trace,
        }
        try:
            # A fix made from this diagnosis carries it into the version's story.
            self._diagnosis_path.write_text(
                json.dumps(diagnosis, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass
        return {
            "recommendation": text,
            "failures": len(failures),
            "coding_model": self.coding_model_label(),
        }

    def _diagnosis_for(self, recommendation: Optional[str]) -> Optional[Dict[str, Any]]:
        """The kept diagnosis, when this fix is being made from it."""
        if not recommendation:
            return None
        try:
            kept = json.loads(self._diagnosis_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            kept = None
        if (
            isinstance(kept, dict)
            and kept.get("text", "").strip() == recommendation.strip()
        ):
            return kept
        return {"text": recommendation}

    def fix(
        self, recommendation: Optional[str] = None, focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make the fix in the agent's copy, check it, and keep it as a version.

        Edits that do not pass the check are dropped, so the copy never holds
        a version that cannot be imported."""
        self.source.ensure()
        self._begin()
        started = _now()
        on_record = recent_failures(self.agent, 15)
        left_over = self.source.changed_files()
        if left_over:
            # A fix starts from the last kept version, not a half-made edit.
            self.source.discard_changes()
        task = "Fix the cause of this agent's recent errors in the source."
        if recommendation:
            task += f"\nA review already recommended this:\n{recommendation}"
        if focus:
            task += f"\nLook at this in particular: {focus}"
        self.on_step("Working on a fix")
        try:
            answer = self._run(_FIX, task, write=True)
        except Exception:
            # A run that broke off may have left half an edit behind.
            if self.source.changed_files():
                self.source.discard_changes()
            raise
        changed = self.source.changed_files()
        result: Dict[str, Any] = {
            "answer": answer,
            "changed_files": changed,
            "version": None,
            "check": None,
            "coding_model": self.coding_model_label(),
        }
        if not changed:
            result["outcome"] = "no_change"
            return result
        self.on_step("Checking the source")
        check = self.source.check()
        if not check["ok"]:
            failed = "\n".join(
                f"{step['name']}: {step['output']}"
                for step in check["steps"]
                if not step["ok"]
            )
            self.on_step("Repairing the fix")
            answer = self._run(
                _FIX,
                "Your edit to the source does not pass the check. Repair it.\n"
                + failed,
                write=True,
            )
            result["answer"] = answer
            check = self.source.check()
        result["check"] = check
        repaired = any(step["text"] == "Repairing the fix" for step in self._steps)
        if not check["ok"]:
            result["diff"] = self.source.diff()
            result["dropped"] = self.source.discard_changes()
            result["outcome"] = "check_failed"
            return result
        # Small models write the marker loosely ("-Version:", "**VERSION**:").
        match = re.search(
            r"^\W*version\W*:\s*(.+)$", answer, flags=re.MULTILINE | re.IGNORECASE
        )
        first = answer.splitlines()[0] if answer else ""
        message = (match.group(1) if match else first).strip(" *`")[:120]
        result["changed_files"] = self.source.changed_files()
        result["version"] = self.source.save_version(message or "Fix")
        diagnosis = self._diagnosis_for(recommendation)
        read = self._seen or (diagnosis or {}).get("errors_read") or []
        self.on_step("Explaining the change")
        explanation = self.explain(
            self.source.diff(result["version"]["id"]),
            (read or on_record)[-_STORY_ERRORS:],
        )
        result["explanation"] = explanation
        story = {
            "coding_model": result["coding_model"],
            "started_at": started,
            "finished_at": _now(),
            # What the model was shown; else what was on record when it began.
            "errors": (read or on_record)[-_STORY_ERRORS:],
            "errors_read_by_model": bool(read),
            "traces_read": list(self._traces)
            + list((diagnosis or {}).get("traces_read") or []),
            "diagnosis": diagnosis,
            "prompt": {
                "instruction": _DROID_FIX if self.coding_agent() else _FIX,
                "task": task,
            },
            "steps": list(self._steps),
            "repaired": repaired,
            "check": check,
            "answer": answer,
            # Why the change was made, in the coding model's words.
            "explanation": explanation,
            # Where the coding model's own work can be read in Traces.
            "work_trace": self._work_trace,
        }
        try:
            self.source.save_story(result["version"]["id"], story)
        except (SelfSourceError, OSError) as exc:
            logger.warning("The account of this fix was not kept: %s", exc)
        result["diff"] = self.source.diff(result["version"]["id"])
        result["outcome"] = "fixed"
        return result
