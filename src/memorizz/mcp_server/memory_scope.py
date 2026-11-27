"""Keep an MCP server inside the memories it was started for.

A harness run gets a MemoRizz MCP server of its own. With
``MEMORIZZ_MCP_SERVER_MEMORY_IDS`` set (the meta-harness sets it to the
run's ``memory_id``) that server reads and writes only those memories:

* an operation that takes a ``memory_id`` uses the scoped one when none is
  given, and is refused for any other;
* an operation that names a record, a conversation or a harness run is
  refused unless what it names belongs to a scoped memory;
* listings are cut down to what belongs to a scoped memory;
* the ``search`` and ``fetch`` a connected app uses look only in the scoped
  memories;
* memories named in ``MEMORIZZ_MCP_SERVER_READ_MEMORY_IDS`` can be read and
  never written;
* anything that could read across memories (traces, exports, agent
  inspection, starting runs) is refused.

Operations are refused unless they are listed here, so a new one is not
reachable from a scoped server until it has been placed.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Optional, Set, Tuple

# Take a ``memory_id``; it is filled in or checked.
_BY_MEMORY = frozenset(
    {
        "list_memories", "search_memories", "remember", "record_turn",
        "summarize_session", "ingest", "lookup_entities", "upsert_entity",
        "memory_status", "memory_timeline", "get_conversation",
    }
)  # fmt: skip
# Of those, the ones that only read; a read-only memory may be named.
_READS = frozenset(
    {
        "list_memories", "search_memories", "lookup_entities",
        "memory_status", "memory_timeline", "get_conversation", "get_memory",
    }
)  # fmt: skip
# Name one record; the record must belong to a scoped memory.
_BY_RECORD = frozenset({"get_memory", "forget", "update_memory"})
# Name one harness run; the run must have been given a scoped memory.
_BY_RUN = frozenset({"get_harness_run", "get_harness_events"})
# Listings whose rows carry the memory they belong to.
_FILTERED = {
    "list_harness_runs": "runs",
    "list_conversations": "conversations",
    "get_harness_conversation": None,
}
_AGENTS = frozenset({"list_agents", "get_agent"})
# Nothing here reads a memory's content.
_OPEN = frozenset(
    {
        "server_info", "list_harnesses", "list_connected_tools",
        "call_connected_tool", "resume_connected_tool_call", "list_approvals",
        "approve_proposal", "reject_proposal", "cancel_proposal",
        "resume_proposal", "start_timer", "list_timers", "cancel_timer",
    }
)  # fmt: skip


def _refuse(message: str):
    from .runtime import MemorizzServerError

    return MemorizzServerError("memory_out_of_scope", message)


def _named(scope: Set[str]) -> str:
    return ", ".join(sorted(scope))


def _row_memory(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    for holder in (row, row.get("task"), row.get("metadata")):
        if isinstance(holder, dict) and holder.get("memory_id"):
            return str(holder["memory_id"])
    return None


def scoped_call(
    runtime: Any,
    function: Callable[..., Any],
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    """Run ``function`` under the runtime's memory scope, if it has one."""
    scope: Optional[Set[str]] = getattr(runtime.config, "memory_ids", None)
    if not scope:
        return function(*args, **kwargs)
    name = getattr(function, "__name__", "")
    readable = set(getattr(runtime.config, "read_memory_ids", None) or ())
    # What this operation may touch: writes stay in the server's own
    # memories; reads also reach the read-only ones.
    reach = scope | readable if name in _READS else scope
    if getattr(function, "__self__", None) is not runtime:
        # A tool group beside the runtime; none of them is placed here.
        raise _refuse(
            f"{name} is not available here: this server is limited to the "
            f"memory {_named(scope)}"
        )
    if name in _OPEN:
        return function(*args, **kwargs)
    if name in _AGENTS:
        # An agent's settings name its memories, so they are shown only
        # when the server was also limited to chosen agents.
        if not getattr(runtime.config, "exposed_agent_ids", None):
            raise _refuse(
                f"{name} is not available here: this server is limited to "
                f"the memory {_named(scope)}"
            )
        return function(*args, **kwargs)

    bound = inspect.signature(function).bind(*args, **kwargs)
    given = bound.arguments

    if name in _BY_MEMORY:
        wanted = str(given.get("memory_id") or "").strip()
        if not wanted:
            if len(scope) != 1:
                raise _refuse(f"Say which memory: one of {_named(scope)}")
            bound.arguments["memory_id"] = next(iter(scope))
        elif wanted not in reach:
            if wanted in readable:
                raise _refuse(f"The memory {wanted!r} is read-only here")
            raise _refuse(
                f"This server is limited to the memory {_named(scope)}; "
                f"it cannot reach {wanted!r}"
            )
        return function(*bound.args, **bound.kwargs)

    if name in _BY_RECORD:
        identity = given["identity"]
        memory_type = given.get("memory_type") or "knowledge_base"
        found = runtime.get_memory(given["record_id"], memory_type, identity)
        owner = _row_memory(found.get("memory"))
        if owner in readable and owner not in reach:
            raise _refuse(f"The memory {owner!r} is read-only here")
        if owner not in reach:
            # The same answer as for a record that does not exist.
            from .runtime import MemorizzServerError

            raise MemorizzServerError("memory_not_found", "Memory was not found")
        return found if name == "get_memory" else function(*args, **kwargs)

    if name == "connector_search":
        # The search a connected app makes: every memory it may read.
        bound.arguments["memory_ids"] = sorted(scope | readable)
        return function(*bound.args, **bound.kwargs)

    if name == "connector_fetch":
        kind, _, record_id = str(given.get("record") or "").partition(":")
        if kind and record_id:
            from .runtime import MemorizzServerError

            try:
                found = runtime.get_memory(record_id, kind, given["identity"])
            except MemorizzServerError:
                found = {}
            if _row_memory(found.get("memory")) not in scope | readable:
                raise MemorizzServerError("memory_not_found", "Memory was not found")
        return function(*args, **kwargs)

    if name in _BY_RUN:
        run = runtime.get_harness_run(given["run_id"], given["identity"])
        if _row_memory(run.get("run")) not in scope | readable:
            raise _refuse("That run belongs to another memory")
        return function(*args, **kwargs)

    if name in _FILTERED:
        result = function(*args, **kwargs)
        key = _FILTERED[name]
        if key is None:
            # A harness conversation: every turn is a run of one memory.
            runs = runtime._conversation_for_identity(
                given["conversation"], given["identity"]
            )[1]
            if any(_row_memory(run) not in scope | readable for run in runs):
                raise _refuse("That conversation belongs to another memory")
            return result
        rows = [
            row for row in result.get(key) or [] if _row_memory(row) in scope | readable
        ]
        return {**result, key: rows, "count": len(rows)}

    raise _refuse(
        f"{name} is not available here: this server is limited to the memory "
        f"{_named(scope)}"
    )
