"""Codex and Claude Code sessions as harness runs.

The MemoRizz plugin runs inside Codex and Claude Code rather than under
MetaHarness, so those sessions have no run record of their own. Each agent
does write a session log: Codex a rollout file under ``$CODEX_HOME/sessions``,
Claude Code a transcript under ``~/.claude/projects``. This reads that log and
records the session as a finished harness run with its steps, so the
Harnesses page shows its trajectory and can compare it with other runs.

A session is one run. It is rebuilt whole each time the log grows, so the
run keeps up with an interactive session turn by turn.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .models import (
    HarnessEvent,
    HarnessEventType,
    HarnessPermissions,
    HarnessResult,
    HarnessRun,
    HarnessStatus,
    HarnessTask,
)
from .security import redact

AGENTS = ("codex", "claude-code")
SOURCE = "plugin"
# Bounds for one recorded session: long command output and file contents are
# cut, and a very long session keeps its first steps.
MAX_TEXT = 8_000
MAX_EVENTS = 4_000
_NAMESPACE = uuid.UUID("0f5e7c1a-6d2b-4c8e-9a41-3b7d2e8f9c10")
# The text the plugin's hooks add to a session, as it appears in the log.
_PLUGIN_CONTEXT = (
    ("MemoRizz memory is connected", "SessionStart"),
    ("MemoRizz project memories that may be relevant", "UserPromptSubmit"),
)


def session_run_id(agent: str, session_id: str) -> str:
    """The run ID a session is recorded under: the same for every update."""
    return str(uuid.uuid5(_NAMESPACE, f"{agent}:{session_id}"))


def detect_agent(path: str | Path) -> Optional[str]:
    """Which agent wrote a session log, from its first records."""
    for row in _rows(Path(path), limit=20):
        if row.get("type") == "session_meta":
            return "codex"
        if row.get("sessionId") or row.get("type") in {"user", "assistant"}:
            return "claude-code"
    return None


def find_session_log(
    agent: str, session_id: str, *, home: Optional[str | Path] = None
) -> Optional[Path]:
    """A session's log when the hook didn't name it."""
    if not session_id:
        return None
    if agent == "codex":
        root = Path(home or os.environ.get("CODEX_HOME") or "~/.codex").expanduser()
        found = sorted((root / "sessions").rglob(f"rollout-*{session_id}.jsonl"))
    else:
        root = Path(
            home or os.environ.get("CLAUDE_CONFIG_DIR") or "~/.claude"
        ).expanduser()
        found = sorted((root / "projects").glob(f"*/{session_id}.jsonl"))
    return found[-1] if found else None


def read_session(
    path: str | Path,
    *,
    agent: Optional[str] = None,
    memory_id: Optional[str] = None,
) -> Optional[Tuple[HarnessRun, List[HarnessEvent]]]:
    """A session log as a run and its events, or None when it holds no turn."""
    path = Path(path).expanduser()
    agent = agent or detect_agent(path)
    if agent == "codex":
        session = _read_codex(path)
    elif agent == "claude-code":
        session = _read_claude(path)
    else:
        return None
    if session is None or not session.get("prompts"):
        return None
    return _build(agent, path, session, memory_id)


def record_session(
    path: str | Path,
    *,
    agent: Optional[str] = None,
    memory_id: Optional[str] = None,
    store: Any = None,
) -> Optional[HarnessRun]:
    """Record (or bring up to date) a session's run. Returns it, or None."""
    built = read_session(path, agent=agent, memory_id=memory_id)
    if built is None:
        return None
    owned = store is None
    if owned:
        from .store import SQLiteHarnessRunStore

        store = SQLiteHarnessRunStore()
    try:
        return store.replace(*built)
    finally:
        if owned:
            store.close()


# --------------------------------------------------------------------------
# Reading the logs


def _rows(path: Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if limit is not None and index >= limit:
                    return
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a line still being written
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def _iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            seconds = value / 1000 if value > 1e11 else value
            moment = datetime.fromtimestamp(seconds, timezone.utc)
        else:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None
    return moment.astimezone(timezone.utc).isoformat()


def _moment(value: Optional[str]) -> datetime:
    if not value:
        return datetime.max.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value)


def _clip(value: Any) -> Any:
    """Cut long strings anywhere in a value (file contents, command output)."""
    if isinstance(value, str):
        if len(value) <= MAX_TEXT:
            return value
        return value[:MAX_TEXT] + f"\n… ({len(value) - MAX_TEXT:,} more characters)"
    if isinstance(value, dict):
        return {key: _clip(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clip(item) for item in value]
    return value


def _plugin_hook(text: str) -> Optional[str]:
    for marker, hook in _PLUGIN_CONTEXT:
        if marker in text:
            return hook
    return None


class _Session:
    """What a reader collects: steps, prompts and the session's totals."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {
            "events": [],
            "prompts": [],
            "turn_times": [],
            "final": "",
            "model": None,
            "session_id": None,
            "cwd": None,
            "changed_files": False,
            "usage": {},
            "cost_usd": None,
        }

    def add(self, kind: HarnessEventType, data: Dict[str, Any], at: Any) -> None:
        self.data["events"].append((kind, data, _iso(at)))

    def prompt(self, text: str, at: Any) -> None:
        self.data["prompts"].append(text)
        self.data["turn_times"].append([_iso(at), _iso(at)])
        self.add(HarnessEventType.MESSAGE, {"role": "user", "text": text}, at)

    def touch(self, at: Any) -> None:
        """The current turn was still working at ``at``."""
        if self.data["turn_times"] and _iso(at):
            self.data["turn_times"][-1][1] = _iso(at)

    def context(self, text: str, hook: str, at: Any) -> None:
        self.add(
            HarnessEventType.STATUS,
            {
                "memory_context": {
                    "source": "memorizz_plugin",
                    "hook": hook,
                    "text": text,
                    "token_estimate": max(1, len(text) // 4),
                }
            },
            at,
        )


def _read_codex(path: Path) -> Optional[Dict[str, Any]]:
    from .adapters import codex_list_cost

    session = _Session()
    data = session.data
    started_at = None
    service_tier = None
    request_costs: List[Optional[float]] = []
    for row in _rows(path):
        kind = row.get("type")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        at = row.get("timestamp")
        if kind == "session_meta":
            data["session_id"] = payload.get("id") or payload.get("session_id")
            data["cwd"] = payload.get("cwd")
            data["version"] = payload.get("cli_version")
            started_at = at
        elif kind == "turn_context":
            data["model"] = payload.get("model") or data["model"]
            data["cwd"] = payload.get("cwd") or data["cwd"]
            service_tier = payload.get("service_tier")
        elif kind == "response_item" and payload.get("role") == "developer":
            text = _codex_text(payload.get("content"))
            hook = _plugin_hook(text)
            if hook:
                session.context(text, hook, at)
        elif kind == "event_msg":
            event = payload.get("type")
            if event == "item_completed":
                _codex_item(
                    session,
                    payload.get("item") or {},
                    at,
                    started=payload.get("started_at_ms"),
                )
            elif event == "token_count":
                info = payload.get("info") or {}
                total = info.get("total_token_usage")
                if isinstance(total, dict) and total != data["usage"]:
                    # The same totals can be logged again for rate-limit updates.
                    # Price each request: cumulative session input must not
                    # trigger a single request's long-context surcharge.
                    request_usage = info.get("last_token_usage")
                    if not isinstance(request_usage, dict):
                        request_usage = {
                            key: value - data["usage"].get(key, 0)
                            for key, value in total.items()
                            if type(value) is int
                        }
                    request_costs.append(
                        codex_list_cost(
                            data["model"],
                            {**request_usage, "service_tier": service_tier},
                        )
                    )
                    data["usage"] = dict(total)
            elif event == "task_complete" and payload.get("last_agent_message"):
                data["final"] = str(payload["last_agent_message"])
                session.touch(at)
            elif event in {"error", "turn_aborted"}:
                message = payload.get("message") or payload.get("reason") or event
                session.add(HarnessEventType.ERROR, {"error": str(message)}, at)
    if data["session_id"] is None:
        return None
    data["started_at"] = started_at
    if data["usage"]:
        usage = {
            key: data["usage"].get(key)
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
            )
            if data["usage"].get(key) is not None
        }
        if request_costs and all(cost is not None for cost in request_costs):
            data["cost_usd"] = round(sum(request_costs), 8)
        if data["cost_usd"] is not None:
            usage["cost_basis"] = "list_rate_estimate"
            usage["cost_requests"] = len(request_costs)
        data["usage"] = usage
    return data


def _codex_text(content: Any) -> str:
    return "\n".join(
        str(part.get("text") or "")
        for part in content or []
        if isinstance(part, dict) and part.get("text")
    ).strip()


def _codex_item(
    session: _Session, item: Dict[str, Any], at: Any, *, started: Any = None
) -> None:
    """One finished step. ``at`` is when it finished; ``started`` (epoch ms,
    when Codex logs it) when it began, so commands and tools show how long
    they took."""
    kind = str(item.get("type") or "")
    began = started if _iso(started) else at
    identifier = item.get("id")
    if kind == "UserMessage":
        text = _codex_text(item.get("content"))
        if text:
            session.prompt(text, at)
        return
    session.touch(at)
    if kind == "AgentMessage":
        text = _codex_text(item.get("content"))
        if text:
            session.data["final"] = text
            session.add(
                HarnessEventType.MESSAGE, {"role": "assistant", "text": text}, at
            )
    elif kind == "Reasoning":
        text = "\n".join(str(part) for part in item.get("summary_text") or []).strip()
        session.add(
            HarnessEventType.REASONING,
            {"id": identifier, "text": text, "hidden": not text},
            at,
        )
    elif kind == "CommandExecution":
        command = item.get("command")
        if isinstance(command, list):
            # ["/bin/zsh", "-lc", "<script>"]: show the script.
            command = (
                command[-1]
                if len(command) == 3 and command[1] in {"-lc", "-c"}
                else " ".join(str(part) for part in command)
            )
        if _iso(began) != _iso(at):
            session.add(
                HarnessEventType.COMMAND,
                {
                    "id": identifier,
                    "type": "command_execution",
                    "command": str(command or ""),
                    "status": "in_progress",
                },
                began,
            )
        session.add(
            HarnessEventType.COMMAND,
            {
                "id": identifier,
                "type": "command_execution",
                "command": str(command or ""),
                "aggregated_output": str(
                    item.get("aggregated_output") or item.get("stdout") or ""
                ),
                "exit_code": item.get("exit_code"),
                "status": item.get("status") or "completed",
            },
            at,
        )
    elif kind == "McpToolCall":
        server, tool = item.get("server") or "mcp", item.get("tool") or "tool"
        session.add(
            HarnessEventType.TOOL_CALL,
            {
                "id": identifier,
                "type": "mcp_tool_call",
                "name": f"mcp__{server}__{tool}",
                "server": server,
                "tool": tool,
                "arguments": item.get("arguments") or {},
                "status": item.get("status") or "completed",
            },
            began,
        )
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        output = _codex_text(result.get("content")) or str(item.get("error") or "")
        session.add(
            HarnessEventType.TOOL_RESULT,
            {
                "tool_use_id": identifier,
                "content": output,
                "is_error": bool(result.get("isError") or item.get("error")),
            },
            at,
        )
    elif kind == "FileChange":
        cwd = str(session.data.get("cwd") or "")
        changes = []
        for name, change in (item.get("changes") or {}).items():
            change = change if isinstance(change, dict) else {}
            relative = (
                name[len(cwd) :].lstrip("/") if cwd and name.startswith(cwd) else name
            )
            changes.append(
                {
                    "path": relative,
                    "kind": change.get("type") or "update",
                    "diff": change.get("unified_diff") or "",
                }
            )
        session.data["changed_files"] = session.data["changed_files"] or bool(changes)
        session.add(
            HarnessEventType.FILE_CHANGE,
            {
                "id": identifier,
                "type": "file_change",
                "changes": changes,
                "status": item.get("status"),
            },
            at,
        )
    elif kind == "WebSearch":
        session.add(
            HarnessEventType.TOOL_CALL,
            {
                "id": identifier,
                "type": "web_search",
                "name": "web_search",
                "query": item.get("query"),
                "status": "completed",
            },
            at,
        )


def _read_claude(path: Path) -> Optional[Dict[str, Any]]:
    session = _Session()
    data = session.data
    started_at = None
    usage_by_message: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for row in _rows(path):
        if row.get("isSidechain"):
            continue  # a subagent's own steps; its call and answer are shown
        kind = row.get("type")
        at = row.get("timestamp")
        data["session_id"] = data["session_id"] or row.get("sessionId")
        data["cwd"] = data["cwd"] or row.get("cwd")
        data["version"] = data.get("version") or row.get("version")
        started_at = started_at or at
        if kind == "attachment":
            attachment = row.get("attachment") or {}
            text = str(attachment.get("content") or "")
            hook = _plugin_hook(text)
            if hook:
                session.context(text, attachment.get("hookEvent") or hook, at)
        elif kind == "user":
            content = (row.get("message") or {}).get("content")
            results = (
                [
                    block
                    for block in content or []
                    if isinstance(block, dict) and block.get("type") == "tool_result"
                ]
                if isinstance(content, list)
                else []
            )
            if results:
                session.touch(at)
                session.add(
                    HarnessEventType.TOOL_RESULT,
                    {
                        "message": {
                            "content": [
                                {
                                    "tool_use_id": block.get("tool_use_id"),
                                    "content": _claude_result_text(
                                        block.get("content")
                                    ),
                                    "is_error": bool(block.get("is_error")),
                                }
                                for block in results
                            ]
                        }
                    },
                    at,
                )
            elif not row.get("isMeta"):
                from .adapters import _text_blocks

                text = _text_blocks({"content": content}).strip()
                if text and not text.startswith("<local-command"):
                    session.prompt(text, at)
        elif kind == "assistant":
            message = row.get("message") or {}
            model = str(message.get("model") or "")
            if model and not model.startswith("<"):
                data["model"] = model
                if isinstance(message.get("usage"), dict) and message.get("id"):
                    usage_by_message[str(message["id"])] = (model, message["usage"])
            session.touch(at)
            _claude_blocks(session, message, at)
        elif kind == "cost-state" and row.get("totalCostUSD") is not None:
            data["cost_usd"] = float(row["totalCostUSD"])
    if data["session_id"] is None:
        return None
    data["started_at"] = started_at
    _claude_usage(data, usage_by_message)
    return data


def _claude_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_reference":
            parts.append(f"(loaded {block.get('tool_name')})")
        else:
            parts.append(f"({block.get('type') or 'content'})")
    return "\n".join(parts)


def _claude_tool_name(name: str) -> str:
    """mcp__plugin_memorizz_memorizz__x (a plugin's server) → mcp__memorizz__x."""
    parts = name.split("__")
    if len(parts) >= 3 and parts[0] == "mcp" and parts[1].startswith("plugin_"):
        server = parts[1][len("plugin_") :]
        plugin, _, rest = server.partition("_")
        parts[1] = rest or plugin
    return "__".join(parts)


def _claude_blocks(session: _Session, message: Dict[str, Any], at: Any) -> None:
    for block in message.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in {"thinking", "redacted_thinking"}:
            text = str(block.get("thinking") or "").strip()
            session.add(
                HarnessEventType.REASONING, {"text": text, "hidden": not text}, at
            )
        elif kind == "text" and str(block.get("text") or "").strip():
            text = str(block["text"]).strip()
            session.data["final"] = text
            session.add(
                HarnessEventType.MESSAGE, {"role": "assistant", "text": text}, at
            )
        elif kind == "tool_use":
            name = str(block.get("name") or "tool")
            data = {
                "type": "tool_use",
                "id": block.get("id"),
                "name": _claude_tool_name(name),
                "input": block.get("input") or {},
            }
            if name in {"Task", "Agent"}:
                data["subagent"] = True
            if name in {"Edit", "Write", "MultiEdit", "NotebookEdit"}:
                session.data["changed_files"] = True
            session.add(HarnessEventType.TOOL_CALL, data, at)


def _claude_usage(
    data: Dict[str, Any], by_message: Dict[str, Tuple[str, Dict[str, Any]]]
) -> None:
    """Each model call is logged once per content block with the same usage;
    count it once."""
    if not by_message:
        return
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    per_model: Dict[str, Dict[str, int]] = {}
    for model, usage in by_message.values():
        counts = per_model.setdefault(model, dict.fromkeys(totals, 0))
        for key in totals:
            value = int(usage.get(key) or 0)
            totals[key] += value
            counts[key] += value
    # Per model, as the Claude Code adapter reports a whole run.
    data["usage"] = {**totals, "model": data.get("model"), "models": per_model}
    if data.get("cost_usd") is not None:
        return  # Claude Code's own total, written when the session ends
    from ..observability.pricing import DEFAULT_PRICING

    cost = 0.0
    for model, counts in per_model.items():
        cached = counts["cache_read_input_tokens"]
        writes = counts["cache_creation_input_tokens"]
        quote = DEFAULT_PRICING.quote(
            {
                "provider": "anthropic",
                "model": model,
                "input_tokens": counts["input_tokens"] + cached + writes,
                "cached_tokens": cached,
                "cache_write_tokens": writes,
                "output_tokens": counts["output_tokens"],
            }
        )
        if quote.get("cost_status") != "calculated":
            return
        cost += float(quote["cost_usd"])
    data["cost_usd"] = round(cost, 6)
    data["usage"]["cost_basis"] = "list_rate_estimate"


# --------------------------------------------------------------------------
# The run


def _build(
    agent: str, path: Path, session: Dict[str, Any], memory_id: Optional[str]
) -> Tuple[HarnessRun, List[HarnessEvent]]:
    run_id = session_run_id(agent, str(session["session_id"]))
    # In time order: a step is logged when it ends, and log lines can be out of
    # order by a few milliseconds.
    session["events"].sort(key=lambda event: _moment(event[2]))
    times = [at for _kind, _data, at in session["events"] if at]
    # Log lines aren't strictly in time order: start at the earliest.
    stamps = sorted(
        at for at in [_iso(session.get("started_at")), *times] if at is not None
    )
    started = stamps[0] if stamps else None
    finished = stamps[-1] if stamps else None
    # Time spent working, without the gaps while the user wrote the next prompt.
    active_ms = 0
    for begin, end in session["turn_times"]:
        if begin and end:
            active_ms += max(
                0,
                int(
                    (
                        datetime.fromisoformat(end) - datetime.fromisoformat(begin)
                    ).total_seconds()
                    * 1000
                ),
            )
    usage = dict(session.get("usage") or {})
    if session.get("model"):
        usage.setdefault("model", session["model"])
    turns = len(session["prompts"])

    events: List[HarnessEvent] = [
        HarnessEvent(
            run_id,
            HarnessEventType.STATUS,
            {
                "session_id": session["session_id"],
                "model": session.get("model"),
                "source": SOURCE,
            },
            timestamp=started or finished or _iso(datetime.now(timezone.utc)),
        )
    ]
    for kind, data, at in session["events"][:MAX_EVENTS]:
        events.append(
            HarnessEvent(
                run_id,
                kind,
                _clip(redact(data)),
                timestamp=at or events[-1].timestamp,
            )
        )
    if len(session["events"]) > MAX_EVENTS:
        events.append(
            HarnessEvent(
                run_id,
                HarnessEventType.STATUS,
                {
                    "status": "truncated",
                    "notice": f"Only the first {MAX_EVENTS:,} steps are shown.",
                },
                timestamp=events[-1].timestamp,
            )
        )
    if usage:
        events.append(
            HarnessEvent(
                run_id, HarnessEventType.USAGE, usage, timestamp=events[-1].timestamp
            )
        )
    events.append(
        HarnessEvent(
            run_id,
            HarnessEventType.COMPLETE,
            {
                "status": HarnessStatus.SUCCEEDED.value,
                "ok": True,
                "verified": False,
                "cost_usd": session.get("cost_usd"),
                "latency_ms": active_ms,
                "error_code": None,
            },
            timestamp=events[-1].timestamp,
        )
    )
    for sequence, event in enumerate(events, start=1):
        event.sequence = sequence

    task = HarnessTask(
        task=session["prompts"][0],
        workspace=str(session.get("cwd") or path.parent),
        harness=agent,
        memory_id=memory_id,
        thread_id=f"{agent}-{session['session_id']}",
        model=session.get("model"),
        permissions=HarnessPermissions(
            workspace_mode="direct" if session.get("changed_files") else "read_only",
            require_approval=False,
        ),
        metadata={
            "source": SOURCE,
            "session_id": session["session_id"],
            "session_log": str(path),
            "turns": turns,
            "agent_version": session.get("version"),
        },
        run_id=run_id,
    )
    result = HarnessResult(
        run_id=run_id,
        harness=agent,
        status=HarnessStatus.SUCCEEDED,
        final_response=str(redact(session.get("final") or "")),
        usage=usage,
        cost_usd=session.get("cost_usd"),
        latency_ms=active_ms,
        checkpoint={"session_id": session["session_id"]},
    )
    run = HarnessRun(
        run_id=run_id,
        task=task.to_dict(),
        status=HarnessStatus.SUCCEEDED,
        harness=agent,
        created_at=started or finished,
        updated_at=finished or started,
        started_at=started,
        finished_at=finished,
        result=result.to_dict(),
    )
    return run, events
