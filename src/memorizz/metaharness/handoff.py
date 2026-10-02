"""Stage handoffs: what one harness passes to the next, and to memory.

A harness run's normalized events are reduced to one structured handoff (its
answer, the files it changed, the commands it ran, verification and errors).
Later stages receive the handoffs of every earlier stage fitted to a character
budget: the most recent stage keeps the most room and older answers are
shortened from the middle, so neither the first plan nor the latest result is
silently cut. Each handoff can also be written to conversation memory, where
retrieval, compaction and summaries treat it like any other memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .security import redact

# Per-handoff caps before any budget fitting.
MAX_RESPONSE_CHARS = 16_000
MAX_FILES = 25
MAX_COMMANDS = 8
MAX_COMMAND_CHARS = 240
# Answers are never shortened below this, so a squeezed handoff stays useful.
MIN_RESPONSE_CHARS = 400
# Budget for rendered prior-stage evidence; the prompt renderer allows 12,000
# characters of stage context by default and this leaves room for its wrapper.
DEFAULT_HANDOFF_BUDGET = 11_000


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def shorten(text: str, limit: int) -> str:
    """Keep the head and tail of ``text`` within ``limit`` characters."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n[… {len(text) - limit} characters omitted …]\n"
    keep = max(0, limit - len(marker))
    head = keep * 2 // 3
    return text[:head] + marker + text[len(text) - (keep - head) :]


def _paths(item: Mapping[str, Any]) -> List[str]:
    found: List[str] = []
    for key in ("path", "file_path", "file", "filename"):
        value = item.get(key)
        if isinstance(value, str) and value:
            found.append(value)
    for key in ("changes", "files"):
        for change in item.get(key) or []:
            if isinstance(change, Mapping):
                found.extend(_paths(change))
            elif isinstance(change, str):
                found.append(change)
    tool_input = item.get("input")
    if isinstance(tool_input, Mapping):
        found.extend(_paths(tool_input))
    return found


def _status_paths(diff: Any) -> List[str]:
    """Paths from the ``git status --porcelain`` part of a workspace diff."""
    if not isinstance(diff, str) or "# git status --porcelain" not in diff:
        return []
    block = diff.split("# git status --porcelain", 1)[1].split("# git diff", 1)[0]
    paths = []
    for line in block.splitlines():
        if len(line) > 3 and line[2] == " ":
            paths.append(line[3:].split(" -> ")[-1].strip().strip('"'))
    return paths


def _workspace_paths(paths: Iterable[str], workspace: str) -> List[str]:
    """Workspace-relative, de-duplicated paths without bytecode caches (which
    host verification, not the harness, usually creates)."""
    root = workspace.rstrip("/") + "/" if workspace else ""
    seen: List[str] = []
    for path in paths:
        path = str(path).strip()
        if root and path.startswith(root):
            path = path[len(root) :]
        if not path or "__pycache__/" in path or path.endswith(".pyc"):
            continue
        if path not in seen:
            seen.append(path)
    return seen


def _command_text(item: Mapping[str, Any]) -> str:
    for key in ("command", "cmd", "text"):
        value = item.get(key)
        if isinstance(value, list):
            value = " ".join(str(part) for part in value)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_COMMAND_CHARS]
    return ""


def stage_handoff(
    run: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Reduce one finished harness run to the evidence the next stage needs."""
    result = _mapping(run.get("result"))
    task = _mapping(run.get("task"))
    files: List[str] = []
    commands: List[str] = []
    last_message = ""
    edit_tools = {"edit", "write", "multiedit", "str_replace_editor", "apply_patch"}
    for event in events:
        kind = str(event.get("type") or "")
        data = _mapping(event.get("data"))
        if kind == "file_change":
            files.extend(_paths(data))
        elif kind == "tool_call" and str(data.get("name") or "").lower() in edit_tools:
            files.extend(_paths(data))
        elif kind == "command":
            text = _command_text(data)
            if text and (not commands or commands[-1] != text):
                commands.append(text)
        elif kind == "message" and data.get("text"):
            last_message = str(data["text"])
    files.extend(_status_paths(result.get("workspace_diff")))
    verification = _mapping(result.get("verification"))
    response = str(result.get("final_response") or last_message or "")
    handoff: Dict[str, Any] = {
        "stage": name or _mapping(task.get("context")).get("harness_stage"),
        "harness": run.get("harness") or task.get("harness"),
        "status": run.get("status"),
        "run_id": run.get("run_id"),
        "response": shorten(response, MAX_RESPONSE_CHARS),
        "response_chars": len(response),
        "files_changed": _workspace_paths(files, str(task.get("workspace") or ""))[
            :MAX_FILES
        ],
        "commands": commands[-MAX_COMMANDS:],
    }
    if verification.get("required") or verification.get("command"):
        handoff["verification"] = {
            "command": verification.get("command"),
            "verified": bool(result.get("verified") or verification.get("verified")),
            "return_code": verification.get("return_code"),
        }
    if result.get("error"):
        handoff["error"] = str(result.get("error"))[:1_000]
    return redact(handoff)


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def fit_handoffs(
    handoffs: List[Dict[str, Any]], budget: int = DEFAULT_HANDOFF_BUDGET
) -> List[Dict[str, Any]]:
    """Fit earlier stages' handoffs into ``budget`` rendered characters.

    Answers are shortened oldest-first and from the middle; the latest stage
    keeps the most room. Every stage stays listed with its status, run ID and
    changed files, so a later harness can fetch any full transcript by run ID.
    """
    fitted = [dict(item) for item in handoffs]
    if not fitted or _size(fitted) <= budget:
        return fitted
    for item in fitted:
        item["files_changed"] = list(item.get("files_changed") or [])[:10]
        item["commands"] = list(item.get("commands") or [])[-3:]
    # Give each answer a share of what the fixed fields leave, weighting the
    # most recent stage twice as heavily as each earlier one.
    fixed = _size([{**item, "response": ""} for item in fitted])
    room = max(0, budget - fixed)
    weights = [1] * (len(fitted) - 1) + [2]
    shares = [room * weight // sum(weights) for weight in weights]
    for item, share in zip(fitted, shares):
        limit = max(MIN_RESPONSE_CHARS, share - 16)
        if len(item.get("response") or "") > limit:
            item["response"] = shorten(item["response"], limit)
            item["response_truncated"] = True
    # Escaping can still overshoot; trim the oldest answers further.
    for item in fitted:
        if _size(fitted) <= budget:
            break
        item["response"] = shorten(item.get("response") or "", MIN_RESPONSE_CHARS)
        item["response_truncated"] = True
    return fitted


def render_conversation(turns: Iterable[Mapping[str, Any]]) -> str:
    """Earlier turns of a harness conversation as plain text for the prompt."""
    blocks = []
    for index, turn in enumerate(turns, start=1):
        header = " · ".join(
            str(part) for part in (turn.get("harness"), turn.get("status")) if part
        )
        lines = [f"Turn {index}" + (f" ({header})" if header else "")]
        if turn.get("request"):
            lines.append("User: " + str(turn["request"]))
        lines.append("Answer: " + str(turn.get("response") or "(no answer text)"))
        if turn.get("files_changed"):
            lines.append("Files changed: " + ", ".join(turn["files_changed"]))
        if turn.get("commands"):
            lines.append("Commands run: " + " | ".join(turn["commands"]))
        if turn.get("error"):
            lines.append("Error: " + str(turn["error"]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_handoff(handoff: Mapping[str, Any], *, workflow: Mapping[str, Any]) -> str:
    """Plain text for conversation memory: a header, the answer, the facts."""
    step = workflow.get("step")
    total = workflow.get("steps")
    position = (
        f" {int(step) + 1}/{total}" if step is not None and total is not None else ""
    )
    kind = {"plan": "plan stage", "compare": "comparison run"}.get(
        str(workflow.get("kind") or ""), "run"
    )
    harness = handoff.get("harness")
    label = " · ".join(
        str(part)
        for part in dict.fromkeys([handoff.get("stage") or harness, harness])
        if part
    )
    lines = [
        f"[Harness {kind}{position}: {label} · {handoff.get('status')}]",
        str(handoff.get("response") or "(no answer text)"),
    ]
    if handoff.get("files_changed"):
        lines.append("Files changed: " + ", ".join(handoff["files_changed"]))
    verification = _mapping(handoff.get("verification"))
    if verification:
        state = "passed" if verification.get("verified") else "failed"
        lines.append(f"Verification {state}: {verification.get('command')}")
    if handoff.get("error"):
        lines.append(f"Error: {handoff['error']}")
    lines.append(f"Harness run: {handoff.get('run_id')}")
    return "\n".join(lines)


def handoff_memory_record(
    handoff: Mapping[str, Any],
    *,
    workflow: Mapping[str, Any],
    memory_id: str,
    thread_id: str,
    agent_id: Optional[str],
    user_id: Optional[str],
) -> Dict[str, Any]:
    """A conversation-memory row for one handoff (see ConversationMemoryUnit)."""
    from ..long_term.episodic.conversational_memory_unit import ConversationMemoryUnit

    unit = ConversationMemoryUnit(
        role="assistant",
        content=render_handoff(handoff, workflow=workflow),
        timestamp=datetime.now(timezone.utc).isoformat(),
        memory_id=memory_id,
        thread_id=thread_id,
        agent_id=agent_id,
        user_id=user_id,
    )
    return unit.model_dump()


__all__ = [
    "DEFAULT_HANDOFF_BUDGET",
    "fit_handoffs",
    "handoff_memory_record",
    "render_handoff",
    "shorten",
    "stage_handoff",
]
