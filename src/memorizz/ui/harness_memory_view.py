"""What a harness run did with memory, and what a shared memory holds.

Shapes the run ledger for two views: the "Memory" tab of a run (what it was
given at the start, what it looked up itself, what it was refused) and the
page of one memory shared between harnesses (who wrote to it, in order).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from ..metaharness.memory_refine import parse_saved

# "mcp__memorizz__memorizz_search_memories", "memorizz.memorizz_remember",
# "memorizz_list_memories": the operation is what follows the last prefix.
MEMORIZZ_TOOL = re.compile(r"^(?:.*[_./])?memorizz_([a-z][a-z_]*)$")
REFUSED = "memory_out_of_scope"
_WRITES = {"remember", "record_turn", "update_memory", "forget", "upsert_entity", "ingest", "summarize_session"}  # fmt: skip


def _short(text: Any, limit: int = 400) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _permissions(run: Dict[str, Any]) -> Dict[str, Any]:
    return dict(((run.get("task") or {}).get("permissions")) or {})


def _saved(run: Dict[str, Any]) -> bool:
    """Whether the run's answer was kept in its memory for later runs."""
    from ..metaharness.service import UNSAVED_STATUSES

    result = run.get("result") or {}
    return bool(
        (run.get("task") or {}).get("memory_id")
        and result.get("final_response")
        and str(run.get("status") or "") not in UNSAVED_STATUSES
    )


def scope_of(run: Dict[str, Any]) -> Dict[str, Any]:
    """What memory the run could reach through MemoRizz."""
    task = run.get("task") or {}
    permissions = _permissions(run)
    memory_id = str(task.get("memory_id") or "")
    access = str(permissions.get("mcp_access") or "read_only")
    opened = str(permissions.get("mcp_memory_scope") or "run") == "all"
    return {
        "memory_id": memory_id,
        "private": not memory_id and not opened,
        "open": opened,
        "read_only": [
            value
            for value in permissions.get("mcp_read_memory_ids") or []
            if value != memory_id
        ],
        "access": access,
        "can_write": access == "governed_write",
        "tools": access != "none",
    }


def given_records(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The records put in the run's prompt before it started."""
    pack = ((run.get("result") or {}).get("context_pack")) or {}
    own = str((run.get("task") or {}).get("memory_id") or "")
    rows = []
    for record in pack.get("records") or []:
        if not isinstance(record, dict):
            continue
        saved = parse_saved(record.get("content") or record.get("text"))
        memory_id = str(record.get("memory_id") or "")
        rows.append(
            {
                "id": str(record.get("source_id") or record.get("id") or ""),
                "memory_type": str(record.get("memory_type") or ""),
                "memory_id": memory_id,
                "read_only": bool(memory_id and memory_id != own),
                "text": _short(saved["text"], 600),
                "written_by": saved["harness"],
                "stage": saved["stage"],
                "run_id": saved["run_id"],
                "at": str(record.get("timestamp") or record.get("created_at") or ""),
                "score": record.get("score"),
            }
        )
    return rows


def _result_blocks(data: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    content = (data.get("message") or {}).get("content")
    for block in content if isinstance(content, list) else [data]:
        if isinstance(block, dict):
            yield block


def _result_text(block: Dict[str, Any]) -> str:
    value = block.get("content")
    if value is None:
        value = block.get("output", block.get("result", block.get("value")))
    if isinstance(value, list):
        return "\n".join(
            str(part.get("text") if isinstance(part, dict) else part) for part in value
        )
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _found(text: str) -> Optional[int]:
    """How many records a MemoRizz tool returned, when its result says."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    value = value.get("result", value) if isinstance(value, dict) else value
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("count"), int):
        return value["count"]
    for key in ("memories", "results", "entities", "conversations", "messages"):
        if isinstance(value.get(key), list):
            return len(value[key])
    return None


def lookups(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each MemoRizz tool the harness called itself, with how it went."""
    calls: Dict[str, Dict[str, Any]] = {}
    order: List[Dict[str, Any]] = []
    for event in events:
        data = event.get("data") or {}
        if event.get("type") == "tool_call":
            name = str(data.get("name") or data.get("tool") or "")
            if data.get("server") == "memorizz" and not name.startswith("memorizz_"):
                name = "memorizz_" + name
            match = MEMORIZZ_TOOL.search(name)
            if not match:
                continue
            key = str(data.get("id") or f"seq-{event.get('sequence')}")
            arguments = data.get("input", data.get("arguments", data.get("args")))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"input": arguments}
            call = calls.get(key)
            if call is None:
                call = {
                    "tool": match.group(1),
                    "writes": match.group(1) in _WRITES,
                    "arguments": {},
                    "sequence": event.get("sequence"),
                    "at": event.get("timestamp") or event.get("created_at"),
                    "outcome": "",
                    "found": None,
                    "message": "",
                }
                calls[key] = call
                order.append(call)
            if isinstance(arguments, dict) and arguments:
                call["arguments"] = {
                    str(k): _short(v, 160) for k, v in arguments.items()
                }
            # Codex reports the result on the call itself as it finishes.
            for field in ("result", "output", "error"):
                if data.get(field) not in (None, "", {}):
                    _settle(
                        call, json.dumps(data[field], default=str), field == "error"
                    )
        elif event.get("type") == "tool_result":
            for block in _result_blocks(data):
                call = calls.get(str(block.get("tool_use_id") or block.get("id") or ""))
                if call is not None:
                    _settle(call, _result_text(block), bool(block.get("is_error")))
    return order


def _settle(call: Dict[str, Any], text: str, failed: bool) -> None:
    if REFUSED in text:
        message = text.split(REFUSED, 1)[1].lstrip(": ").split("\\n")[0]
        call.update(outcome="refused", message=_short(message.strip('"}] '), 240))
        return
    if failed:
        call.update(outcome="failed", message=_short(text, 240))
        return
    found = _found(text)
    call.update(found=found, outcome="nothing" if found == 0 else "ok")


def run_memory(run: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Everything the run's Memory tab shows."""
    scope = scope_of(run)
    asked = lookups(events)
    result = run.get("result") or {}
    return {
        "scope": scope,
        "given": given_records(run),
        "given_known": bool(result.get("context_pack")),
        "lookups": asked,
        "refused": sum(1 for call in asked if call["outcome"] == "refused"),
        # A finished run's answer is kept in its memory for later runs.
        "saved": _saved(run),
    }


def known_memories(runs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The memories harness runs have used, most recently used first."""
    seen: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        memory_id = str((run.get("task") or {}).get("memory_id") or "")
        if not memory_id:
            continue
        entry = seen.setdefault(
            memory_id, {"memory_id": memory_id, "runs": 0, "harnesses": [], "last": ""}
        )
        entry["runs"] += 1
        harness = str(run.get("harness") or "")
        if harness and harness not in entry["harnesses"]:
            entry["harnesses"].append(harness)
        entry["last"] = max(entry["last"], str(run.get("created_at") or ""))
    return sorted(seen.values(), key=lambda entry: entry["last"], reverse=True)


def memory_overview(memory_id: str, runs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """One memory as harnesses used it: who wrote, who only read, in order."""
    writers: List[Dict[str, Any]] = []
    readers: List[Dict[str, Any]] = []
    for run in runs:
        task = run.get("task") or {}
        result = run.get("result") or {}
        own = str(task.get("memory_id") or "") == memory_id
        reads = memory_id in (_permissions(run).get("mcp_read_memory_ids") or [])
        if not own and not reads:
            continue
        pack = result.get("context_pack") or {}
        row = {
            "run_id": str(run.get("run_id") or ""),
            "harness": str(run.get("harness") or ""),
            "status": str(run.get("status") or ""),
            "at": str(run.get("created_at") or ""),
            "asked": _short(task.get("task"), 240),
            "answer": _short(result.get("final_response"), 420),
            "given": len(pack.get("records") or []),
            "saved": bool(own and _saved(run)),
        }
        (writers if own else readers).append(row)
    writers.sort(key=lambda row: row["at"])
    readers.sort(key=lambda row: row["at"])
    tally: Dict[str, Dict[str, int]] = {}
    for row in writers:
        count = tally.setdefault(row["harness"], {"runs": 0, "saved": 0})
        count["runs"] += 1
        count["saved"] += int(row["saved"])
    return {
        "memory_id": memory_id,
        "runs": writers,
        "readers": readers,
        "harnesses": [
            {"harness": name, **count} for name, count in sorted(tally.items())
        ],
        "saved": sum(int(row["saved"]) for row in writers),
    }
