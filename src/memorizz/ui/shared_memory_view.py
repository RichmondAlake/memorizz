# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Who shares a memory: the agents, harnesses and connected apps in each memory ID.

A memory is shared when more than one agent (or harness) reads and writes the
same memory ID in the connected store. This module reads the store's records
and shapes them for two pages: every memory with who uses it, and one memory
with its agents, the decisions it holds, the tasks coordinated in it, and who
wrote what.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..enums.memory_type import MemoryType
from ..metaharness.memory_refine import parse_saved

#: Records read per memory type: enough for a team's memory, bounded for a
#: large store.
ROW_LIMIT = 4000
TURN_LIMIT = 14
BOARD_LIMIT = 6
_INTERNAL = ("memory_history_", "observability_")


def _text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _field(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _when(value: Any) -> str:
    """An ISO timestamp, whatever the record stored (text or a datetime)."""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def rows(provider: Any, memory_type: MemoryType, limit: int = ROW_LIMIT) -> List[Dict]:
    """The newest records of a type, without failing the page on one store."""
    try:
        recent = getattr(provider, "list_recent", None)
        found = (
            recent(memory_type, limit)
            if callable(recent)
            else provider.list_all(memory_type)
        )
    except Exception:
        return []
    return [row for row in found or [] if isinstance(row, dict)][:limit]


def agent_card(agent: Any) -> Dict[str, Any]:
    """What the pages need to know about a saved agent."""
    persona = _field(agent, "persona")
    role = _field(persona, "role") if persona is not None else None
    role = getattr(role, "value", role)
    types = [str(getattr(t, "value", t)) for t in _field(agent, "memory_types") or []]
    policy = _field(agent, "entity_policy") or {}
    has_entities = "entity_memory" in types
    writes = bool(policy.get("writes", True)) if isinstance(policy, dict) else True
    delegates = [
        str(_field(item, "agent_id") or item)
        for item in _field(agent, "delegates") or []
        if item
    ]
    llm = _field(agent, "llm_config") or {}
    return {
        "agent_id": str(_field(agent, "agent_id") or ""),
        "name": str(_field(agent, "name") or "") or "Unnamed agent",
        "role": _text(role, 60) if role and str(role).lower() != "general" else "",
        "model": str(llm.get("model") or "") if isinstance(llm, dict) else "",
        # What the agent may do with the memory's records (entity memory).
        "access": "none" if not has_entities else "writes" if writes else "reads",
        "delegates": delegates,
        "memory_ids": [str(m) for m in _field(agent, "memory_ids") or [] if m],
        "knowledge_base_ids": [
            str(k) for k in _field(agent, "knowledge_base_ids") or [] if k
        ],
    }


def _harness_runs(runs: Iterable[Dict[str, Any]]) -> Dict[str, Counter]:
    by_memory: Dict[str, Counter] = defaultdict(Counter)
    for run in runs or []:
        task = run.get("task") or {}
        memory_id = str(task.get("memory_id") or "")
        if memory_id:
            by_memory[memory_id][str(run.get("harness") or "harness")] += 1
    return by_memory


def _app_saves(provider: Any) -> Dict[str, Counter]:
    """What connected apps (Muse, Grok Bot, a dot) saved, by memory and app.
    A record an app saves through the MCP server names the app."""
    by_memory: Dict[str, Counter] = defaultdict(Counter)
    for memory_type in (MemoryType.KNOWLEDGE_BASE, MemoryType.ENTITY_MEMORY):
        for row in rows(provider, memory_type):
            metadata = (
                row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            )
            app = str(row.get("created_via") or metadata.get("created_via") or "")
            memory_id = str(row.get("memory_id") or "")
            if app and memory_id:
                by_memory[memory_id][app] += 1
    return by_memory


def _members(
    memory_id: str, cards: List[Dict[str, Any]], turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """The agents that run in a memory: those set up with it and those that
    wrote to it, each with how much it said and when it last did."""
    written = Counter(str(row.get("agent_id") or "") for row in turns)
    last: Dict[str, str] = {}
    for row in turns:
        agent_id, at = str(row.get("agent_id") or ""), _when(row.get("timestamp"))
        if at > last.get(agent_id, ""):
            last[agent_id] = at
    members = []
    for card in cards:
        if memory_id in card["memory_ids"] or written.get(card["agent_id"]):
            members.append(
                {
                    **card,
                    "turns": written.get(card["agent_id"], 0),
                    "last_at": last.get(card["agent_id"], ""),
                }
            )
    order = {"writes": 0, "reads": 1, "none": 2}
    members.sort(key=lambda m: (order[m["access"]], -m["turns"], m["name"].lower()))
    return members


def _by_memory(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        memory_id = str(row.get("memory_id") or "")
        if memory_id:
            grouped[memory_id].append(row)
    return grouped


def overview(
    provider: Any, agents: Iterable[Any], harness_runs: Iterable[Dict[str, Any]] = ()
) -> List[Dict[str, Any]]:
    """Every memory ID in use, with who runs in it, most shared first."""
    cards = [agent_card(agent) for agent in agents or []]
    turns = _by_memory(rows(provider, MemoryType.CONVERSATION_MEMORY))
    entities = _by_memory(rows(provider, MemoryType.ENTITY_MEMORY))
    harnesses = _harness_runs(harness_runs)
    apps = _app_saves(provider)
    known = set(turns) | set(entities) | set(harnesses) | set(apps)
    for card in cards:
        known.update(card["memory_ids"])
    memories = []
    for memory_id in known:
        members = _members(memory_id, cards, turns.get(memory_id, []))
        stamps = [_when(row.get("timestamp")) for row in turns.get(memory_id, [])]
        used_by = (
            len(members)
            + len(harnesses.get(memory_id, {}))
            + len(apps.get(memory_id, {}))
        )
        if not used_by:
            continue  # another process's memory: nothing here runs in it
        memories.append(
            {
                "memory_id": memory_id,
                "agents": members,
                "harnesses": dict(harnesses.get(memory_id, {})),
                "apps": dict(apps.get(memory_id, {})),
                "records": len(entities.get(memory_id, [])),
                "turns": len(turns.get(memory_id, [])),
                "last_at": max(stamps) if stamps else "",
                "shared": used_by > 1,
                "used_by": used_by,
            }
        )
    memories.sort(key=lambda m: (-m["used_by"], -m["turns"], m["memory_id"]))
    return memories


def _boards(
    provider: Any, names: Dict[str, str], member_ids: set
) -> List[Dict[str, Any]]:
    """Tasks a member coordinated: the request, the plan and how each part
    ended, read from the coordination board the coordinator kept."""
    boards = []
    for row in rows(provider, MemoryType.SHARED_MEMORY):
        content = row.get("content")
        if not isinstance(content, str) or '"blackboard"' not in content:
            continue
        try:
            payload = json.loads(content)
        except ValueError:
            continue
        root = str(payload.get("root_agent_id") or "")
        if root not in member_ids:
            continue
        request, tasks = "", {}
        for entry in payload.get("blackboard") or []:
            detail = entry.get("content") if isinstance(entry, dict) else None
            if not isinstance(detail, dict):
                continue
            kind = entry.get("entry_type")
            if kind == "workflow_start":
                request = _text(detail.get("original_query"), 320)
            elif kind == "task_decomposition":
                for task in detail.get("sub_tasks") or []:
                    tasks[str(task.get("task_id"))] = {
                        "task_id": str(task.get("task_id")),
                        "description": _text(task.get("description"), 220),
                        "agent": names.get(str(task.get("assigned_agent_id")), "?"),
                        "after": [str(d) for d in task.get("dependencies") or []],
                        "status": "planned",
                        "result": "",
                        "seconds": None,
                    }
            elif kind == "task_start" and str(detail.get("task_id")) in tasks:
                tasks[str(detail.get("task_id"))]["status"] = "running"
            elif kind == "task_completion" and str(detail.get("task_id")) in tasks:
                task = tasks[str(detail.get("task_id"))]
                task["status"] = str(detail.get("status") or "completed")
                task["result"] = _text(detail.get("result"), 320)
                if isinstance(detail.get("duration_ms"), (int, float)):
                    task["seconds"] = round(detail["duration_ms"] / 1000, 1)
        boards.append(
            {
                "coordinator": names.get(root, "?"),
                "request": request,
                "status": str(payload.get("status") or ""),
                "outcome": str(payload.get("outcome") or ""),
                "started_at": _when(payload.get("created_at")),
                "tasks": list(tasks.values()),
            }
        )
    boards.sort(key=lambda board: board["started_at"], reverse=True)
    return boards[:BOARD_LIMIT]


def _authors(provider: Any, memory_id: str, names: Dict[str, str]) -> Dict[str, Any]:
    """Who wrote what, from the store's change history (when it is kept)."""
    try:
        from ..memory_history import MemoryHistory

        events = MemoryHistory(provider).timeline(memory_id=memory_id, limit=800)[
            "events"
        ]
    except Exception:
        events = []
    tally: Dict[str, Counter] = defaultdict(Counter)
    last_change: Dict[str, Dict[str, str]] = {}
    for event in events:
        who = names.get(str(event.get("agent_id") or ""), "")
        kind, action = str(event.get("memory_type")), str(event.get("action"))
        if not who or kind.startswith(_INTERNAL):
            continue
        # A conversation row's "update" is its importance rating being filed.
        if kind == "conversation_memory" and action != "created":
            continue
        tally[who][f"{kind}:{action}"] += 1
        target = str(event.get("target_record_id") or "")
        stamp = str(event.get("timestamp") or "")
        if kind == "entity_memory" and stamp > last_change.get(target, {}).get(
            "at", ""
        ):
            last_change[target] = {"by": who, "at": stamp, "action": action}
    return {
        "kept": bool(events),
        "rows": [
            {
                "agent": who,
                "turns": counts.get("conversation_memory:created", 0),
                "records_created": counts.get("entity_memory:created", 0),
                "records_updated": counts.get("entity_memory:updated", 0),
            }
            for who, counts in sorted(tally.items())
        ],
        "last_change": last_change,
    }


def detail(
    provider: Any,
    agents: Iterable[Any],
    memory_id: str,
    harness_runs: Iterable[Dict[str, Any]] = (),
) -> Optional[Dict[str, Any]]:
    """One memory: its agents, decisions, handbooks, coordinated tasks, the
    newest turns and who wrote what. None when nothing runs in it."""
    cards = [agent_card(agent) for agent in agents or []]
    names = {card["agent_id"]: card["name"] for card in cards}
    turns = [
        row
        for row in rows(provider, MemoryType.CONVERSATION_MEMORY)
        if str(row.get("memory_id") or "") == memory_id
    ]
    members = _members(memory_id, cards, turns)
    harnesses = dict(_harness_runs(harness_runs).get(memory_id, {}))
    apps = dict(_app_saves(provider).get(memory_id, {}))
    if not members and not harnesses and not turns and not apps:
        return None
    member_ids = {member["agent_id"] for member in members}
    authors = _authors(provider, memory_id, names)

    records = []
    for row in rows(provider, MemoryType.ENTITY_MEMORY):
        if str(row.get("memory_id") or "") != memory_id:
            continue
        key = str(row.get("entity_id") or row.get("_id") or row.get("id") or "")
        change = authors["last_change"].get(key) or authors["last_change"].get(
            str(row.get("_id") or "")
        )
        records.append(
            {
                "name": _text(row.get("name"), 120) or "Unnamed record",
                "type": _text(row.get("entity_type"), 40),
                "attributes": [
                    {"name": _text(a.get("name"), 60), "value": _text(a.get("value"))}
                    for a in row.get("attributes") or []
                    if isinstance(a, dict)
                ][:20],
                "changed_by": (change or {}).get("by", ""),
                "changed_at": (change or {}).get("at", "")
                or _when(row.get("updated_at") or row.get("timestamp")),
            }
        )
    records.sort(key=lambda record: record["changed_at"], reverse=True)

    attached: Dict[str, List[str]] = defaultdict(list)
    for member in members:
        for knowledge_base_id in member["knowledge_base_ids"]:
            attached[knowledge_base_id].append(member["name"])
    handbooks = []
    if attached:
        chunks: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows(provider, MemoryType.KNOWLEDGE_BASE):
            chunks[str(row.get("knowledge_base_id") or "")].append(row)
        for knowledge_base_id, readers in attached.items():
            found = chunks.get(knowledge_base_id, [])
            handbooks.append(
                {
                    "knowledge_base_id": knowledge_base_id,
                    "name": _text((found[0].get("namespace") if found else ""), 80)
                    or knowledge_base_id[:8],
                    "chunks": len(found),
                    "agents": sorted(readers),
                    "everyone": len(readers) == len(members),
                }
            )

    turns.sort(key=lambda row: _when(row.get("timestamp")), reverse=True)
    # An answer a harness run saved here is a turn too, written by the harness.
    saved = {id(row): parse_saved(row.get("content")) for row in turns}
    answered = Counter(s["harness"] for s in saved.values() if s["harness"])
    newest = [
        {
            "at": _when(row.get("timestamp")),
            "agent": saved[id(row)]["harness"]
            or names.get(str(row.get("agent_id") or ""), "?"),
            "harness": bool(saved[id(row)]["harness"]),
            "role": str(row.get("role") or ""),
            "thread_id": _text(row.get("thread_id"), 40),
            "text": _text(
                saved[id(row)]["text"]
                if saved[id(row)]["harness"]
                else row.get("content"),
                260,
            ),
        }
        for row in turns[:TURN_LIMIT]
    ]
    authors["rows"] += [
        {
            "agent": harness,
            "harness": True,
            "turns": count,
            "records_created": 0,
            "records_updated": 0,
        }
        for harness, count in sorted(answered.items())
    ]
    coordinators = {
        member["agent_id"]: [names.get(d, d[:8]) for d in member["delegates"]]
        for member in members
        if member["delegates"]
    }
    for member in members:
        member["coordinates"] = coordinators.get(member["agent_id"], [])
    return {
        "memory_id": memory_id,
        "agents": members,
        "writers": [m for m in members if m["access"] == "writes"],
        "readers": [m for m in members if m["access"] != "writes"],
        "shared": len(members) + len(harnesses) + len(apps) > 1,
        "harnesses": harnesses,
        "apps": apps,
        "harness_runs": sum(harnesses.values()),
        "records": records,
        "handbooks": handbooks,
        "boards": _boards(provider, names, member_ids),
        "authors": authors["rows"],
        "history_kept": authors["kept"],
        "turns": newest,
        "turn_count": len(turns),
    }


__all__ = ["agent_card", "detail", "overview", "rows"]
