# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Session-scoped harness routing for the interactive CLI.

``/harness codex`` makes the current agent execute every following turn on
that external harness (Codex, Claude Code, ...) while MemoRizz keeps
supplying memory, tracing and approvals. ``/harness delegate`` keeps the
agent's own model in charge and lets it hand parts of a request to its
harness delegates and call harnesses as tools. ``/compare`` runs one task on
several harnesses side by side. The choice lives on the CLI session only: it
attaches the agent's meta-harness for this process and never saves the agent
record, so ``/harness off`` restores exactly what the agent had before.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

FINAL_STATES = {
    "succeeded",
    "failed",
    "canceled",
    "interrupted",
    "budget_exceeded",
    "verification_failed",
}
META_HARNESS_TOOL_NAMES = (
    "run_harness_task",
    "get_harness_run",
    "list_agent_harnesses",
)
DELEGATE = "delegate"

HARNESS_ALIASES = {
    "claude": "claude-code",
    "claudecode": "claude-code",
    "claude_code": "claude-code",
    "cc": "claude-code",
}


def canonical_harness(name: Any) -> str:
    value = str(name or "").strip().lower().replace("_", "-")
    return HARNESS_ALIASES.get(value, value)


def _service(session) -> Any:
    """The agent's meta-harness service, created on demand for listing/probing."""
    agent = session.agent
    service = getattr(agent, "meta_harness", None)
    if service is not None and hasattr(service, "list_harnesses"):
        return service
    from ..metaharness import MetaHarness

    return MetaHarness.from_env(
        memory_provider=getattr(agent, "memory_provider", None) or session.provider,
        agent=agent,
        approval_store=getattr(agent, "approval_store", None),
    )


def list_harness_status(session, *, probe: bool = True) -> List[Dict[str, Any]]:
    """Each configured harness with its readiness, as ``memorizz harness list`` shows.

    ``probe=False`` only names them (no version or login checks).
    """
    rows = _service(session).list_harnesses(probe=probe) or []
    status: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = canonical_harness(row.get("name"))
        if not name:
            continue
        ready = row.get("ready", row.get("available"))
        status.append(
            {
                "name": name,
                "ready": None if ready is None else bool(ready),
                "reason": row.get("reason") or row.get("error") or row.get("detail"),
                "version": row.get("version"),
            }
        )
    return status


def harness_delegates(session) -> List[Dict[str, Any]]:
    """The agent's delegates that run whole turns on a harness, as the playground lists them."""
    from ..metaharness.catalog import agent_id_of, agent_name_of, harness_backing

    agent = session.agent
    provider = getattr(agent, "memory_provider", None) or session.provider
    rows: List[Dict[str, Any]] = []
    for entry in getattr(agent, "delegates", None) or []:
        delegate = entry
        if isinstance(entry, str):
            try:
                delegate = provider.retrieve_memagent(entry)
            except Exception:
                delegate = None
        backing = harness_backing(delegate) if delegate is not None else None
        if backing:
            rows.append(
                {
                    "id": agent_id_of(delegate),
                    "name": agent_name_of(delegate),
                    "harness": backing["harness"],
                    "model": backing.get("model") or "",
                }
            )
    return rows


def delegation_enabled(session) -> bool:
    from ..metaharness.catalog import delegates_work

    return bool(delegates_work(session.agent))


def active_harness(session) -> Optional[str]:
    return getattr(session, "harness", None) or None


def prompt_label(session) -> str:
    """The REPL prompt, carrying the active harness next to the mode."""
    base = "code" if getattr(session, "code_mode", False) else "memorizz"
    harness = active_harness(session)
    return f"{base}·{harness}> " if harness else f"{base}> "


def apply_harness(session, name: Any) -> Dict[str, Any]:
    """Route this session's turns through ``name`` (``auto`` lets MemoRizz pick).

    Returns a summary dict. Raises ValueError for an unknown harness or one
    that is not ready; nothing is persisted on the agent record.
    """
    agent = session.agent
    harness = canonical_harness(name)
    if not harness:
        raise ValueError("Usage: /harness <codex|claude-code|auto|delegate|off>")
    if harness == DELEGATE:
        return _apply_delegate_mode(session)
    status = list_harness_status(session)
    known = {row["name"]: row for row in status}
    if harness != "auto":
        if harness not in known:
            names = ", ".join(sorted(known)) or "none configured"
            raise ValueError(
                f"Unknown harness '{harness}'. Configured: {names}. "
                "Run `memorizz harness doctor` for setup help."
            )
        row = known[harness]
        if row.get("ready") is False:
            reason = row.get("reason") or "not ready"
            raise ValueError(
                f"Harness '{harness}' is not ready: {reason}. "
                f"Run `memorizz harness doctor {harness}`."
            )
    _remember_agent_state(session)
    agent.with_meta_harness(
        getattr(agent, "meta_harness", None),
        mode="runtime",
        default_harness=harness,
        config=session.harness_backup["harness_config"] or None,
    )
    session.harness = harness
    return {
        "harness": harness,
        "mode": "runtime",
        "ready": True if harness == "auto" else known[harness].get("ready"),
        "memory_id": getattr(session, "memory_id", None),
    }


def _remember_agent_state(session) -> None:
    """Keep what the saved agent had before the first session switch."""
    if getattr(session, "harness_backup", None) is not None:
        return
    agent = session.agent
    session.harness_backup = {
        "meta_harness": getattr(agent, "meta_harness", None),
        "meta_harness_mode": getattr(agent, "meta_harness_mode", None),
        "default_harness": getattr(agent, "default_harness", "auto"),
        "harness_config": dict(getattr(agent, "harness_config", None) or {}),
        "owns_meta_harness": bool(getattr(agent, "_owns_meta_harness", False)),
        "tools_registered": bool(
            getattr(agent, "_meta_harness_tools_registered", False)
        ),
    }


def _apply_delegate_mode(session) -> Dict[str, Any]:
    """The agent's model stays in charge: it can hand work to its harness
    delegates and call harnesses as tools (``run_harness_task``)."""
    agent = session.agent
    _remember_agent_state(session)
    backup = session.harness_backup
    agent.with_meta_harness(
        getattr(agent, "meta_harness", None),
        mode=DELEGATE,
        default_harness=backup["default_harness"] or "auto",
        config=backup["harness_config"] or None,
    )
    session.harness = DELEGATE
    return {
        "harness": DELEGATE,
        "mode": DELEGATE,
        "ready": True,
        "memory_id": getattr(session, "memory_id", None),
        "delegates": harness_delegates(session),
        "delegation_enabled": delegation_enabled(session),
        "tools": list(META_HARNESS_TOOL_NAMES),
    }


def clear_harness(session) -> Dict[str, Any]:
    """Return the agent to native execution, restoring its saved configuration."""
    agent = session.agent
    backup = getattr(session, "harness_backup", None)
    previous = active_harness(session)
    if backup is not None:
        if not backup.get("tools_registered") and getattr(
            agent, "_meta_harness_tools_registered", False
        ):
            # Delegate mode registered the harness tools for this session only.
            manager = getattr(agent, "tool_manager", None)
            for tool_name in META_HARNESS_TOOL_NAMES:
                try:
                    if manager is not None:
                        manager.remove_tool(tool_name)
                except Exception:
                    pass
            agent._meta_harness_tools_registered = False
        agent.meta_harness = backup["meta_harness"]
        agent.meta_harness_mode = backup["meta_harness_mode"]
        agent.default_harness = backup["default_harness"]
        agent.harness_config = dict(backup["harness_config"])
        agent._owns_meta_harness = backup["owns_meta_harness"]
        session.harness_backup = None
    session.harness = None
    return {
        "harness": None,
        "previous": previous,
        "mode": getattr(agent, "meta_harness_mode", None),
    }


def pending_harness_approvals(session, *, limit: int = 5) -> List[Dict[str, Any]]:
    """Approval proposals a harness turn left waiting for a decision."""
    agent = session.agent
    lister = getattr(agent, "list_approval_proposals", None)
    if not callable(lister):
        return []
    try:
        rows = lister(status="pending", limit=limit) or []
    except Exception:
        return []
    return [row for row in rows if isinstance(row, dict)]


def report_pending_approvals(session, console) -> int:
    """After a harness turn, surface waiting approvals inline (the UI shows a panel)."""
    if not active_harness(session):
        return 0
    rows = pending_harness_approvals(session)
    if not rows:
        return 0
    console.print(
        f"[yellow]{len(rows)} approval(s) waiting[/yellow] from the "
        f"{active_harness(session)} harness:"
    )
    for row in rows:
        proposal_id = row.get("proposal_id") or row.get("id") or "?"
        tool = row.get("tool_name") or row.get("operation") or ""
        reason = row.get("policy_reason") or row.get("reason") or ""
        console.print(f"  [cyan]{proposal_id}[/cyan]  {tool}  [dim]{reason}[/dim]")
    console.print(
        "  Approve with [cyan]/approvals approve <proposal-id> <your-name>[/cyan], "
        "reject with [cyan]/approvals reject <proposal-id> <your-name>[/cyan], "
        "then [cyan]/approvals resume <proposal-id>[/cyan]."
    )
    return len(rows)


def parse_compare_args(session, args: str) -> tuple:
    """Split ``/compare codex claude-code <task>`` into harness names, the task
    and options.

    Leading tokens that name a configured harness are the harnesses; the rest
    is the task. ``memagent=<agent-id>`` names the saved agent a compared
    memagent step runs. Raises ValueError with a usage message when fewer than
    two harnesses or no task are given.
    """
    usage = (
        "Usage: /compare <harness> <harness> [more...] <task>   "
        "(see /harnesses; memagent=<agent-id> picks the saved agent it runs)"
    )
    tokens = args.strip().split()
    if not tokens:
        raise ValueError(usage)
    known = {row["name"] for row in list_harness_status(session, probe=False)}
    names: List[str] = []
    options: Dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        candidate, _, value = token.partition("=")
        candidate = canonical_harness(candidate)
        if candidate not in known or candidate == "auto":
            break
        if value:
            if candidate != "memagent":
                raise ValueError(f"'{token}': only memagent takes =<agent-id>. {usage}")
            options["memagent_id"] = value
        if candidate not in names:
            names.append(candidate)
        index += 1
    task = " ".join(tokens[index:]).strip()
    if len(names) < 2 or not task:
        first = canonical_harness(tokens[0].partition("=")[0])
        if first not in known and not names:
            raise ValueError(f"'{tokens[0]}' is not a configured harness. {usage}")
        raise ValueError(usage)
    return names, task, options


def memagent_for_compare(
    session, service, memagent_id: Optional[str] = None
) -> Dict[str, str]:
    """The saved MemAgent a compared ``memagent`` step runs: the one named by
    ``memagent_id`` (an id or unique id prefix), else chosen as the CLI does
    (last used with a harness, else newest); never the chat's own agent, which
    cannot hand a task to itself."""
    from ..metaharness import catalog

    provider = (
        getattr(service, "memory_provider", None)
        or getattr(session.agent, "memory_provider", None)
        or session.provider
    )
    lister = getattr(provider, "list_memagents", None)
    agents = list(lister() or []) if callable(lister) else []
    own = str(getattr(session.agent, "agent_id", "") or "")
    others = [
        a for a in agents if catalog.agent_id_of(a) and catalog.agent_id_of(a) != own
    ]
    if memagent_id:
        wanted = str(memagent_id).strip()
        matches = [a for a in others if catalog.agent_id_of(a).startswith(wanted)]
        if len(matches) != 1:
            if any(catalog.agent_id_of(a) == wanted for a in agents):
                raise ValueError(
                    "memagent can't run this chat's own agent (see /agents)."
                )
            raise ValueError(
                f"memagent={wanted}: {'no' if not matches else 'more than one'} saved "
                "agent matches that id; see /agents."
            )
        chosen_agent = matches[0]
        return {
            "id": catalog.agent_id_of(chosen_agent),
            "name": catalog.agent_name_of(chosen_agent),
        }
    chosen = catalog.default_memagent(service, others)
    if chosen is None:
        raise ValueError(
            "memagent runs another saved MemAgent (this chat's agent can't hand a "
            "task to itself), and there is none yet. Create one with "
            "`memorizz agents create`, or leave memagent out of the comparison."
        )
    return chosen


def run_compare(
    session,
    harnesses: List[str],
    task: str,
    *,
    workspace: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    poll_seconds: float = 1.0,
    memagent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one read-only task on several harnesses side by side and wait.

    The comparison is a durable workflow in the harness ledger (the Agent
    Harnesses page shows it). Ctrl-C cancels it. Returns the workflow with one
    verdict row per harness.
    """
    from ..metaharness.requests import harness_task

    names = [canonical_harness(name) for name in harnesses]
    status = {row["name"]: row for row in list_harness_status(session)}
    for name in names:
        row = status.get(name)
        if row is None:
            raise ValueError(
                f"Unknown harness '{name}'. Configured: {', '.join(sorted(status))}."
            )
        if row.get("ready") is False:
            raise ValueError(
                f"Harness '{name}' is not ready: {row.get('reason') or 'not ready'}. "
                f"Run `memorizz harness doctor {name}`."
            )
    service = _service(session)
    agent = session.agent
    report = on_progress or (lambda line: None)
    agent_id = getattr(agent, "agent_id", None)
    if "memagent" in names:
        chosen = memagent_for_compare(session, service, memagent_id)
        agent_id = chosen["id"]
        report(f"memagent will run {chosen['name']} ({chosen['id'][:8]}).")
    request = harness_task(
        task,
        str(workspace or Path.cwd()),
        agent_id=agent_id,
        memory_id=getattr(session, "memory_id", None),
        user_id=getattr(session, "user_id", None),
        thread_id=getattr(session, "thread_id", None),
        metadata={"execution_backend": "local", "source": "cli-chat"},
    )
    started = service.start_compare(request, names)
    workflow_id = started["orchestration_id"]
    report(f"Comparison {workflow_id} started.")
    seen: Dict[int, str] = {}
    try:
        while True:
            value = service.get_orchestration(workflow_id) or {}
            steps = value.get("steps") or []
            for index, step in enumerate(steps):
                run = service.get_run(step["run_id"]) if step.get("run_id") else None
                state = (run or {}).get("status")
                if state and seen.get(index) != state:
                    seen[index] = state
                    report(
                        f"[{index + 1}/{len(steps)}] "
                        f"{(run or {}).get('harness') or step.get('harness')}: {state}"
                    )
            if value.get("status") in FINAL_STATES:
                break
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        report("Canceling the comparison…")
        service.cancel_orchestration(workflow_id)
        while (service.get_orchestration(workflow_id) or {}).get(
            "status"
        ) not in FINAL_STATES:
            time.sleep(0.2)
        value = service.get_orchestration(workflow_id) or {}
    rows: List[Dict[str, Any]] = []
    for step in value.get("steps") or []:
        run = service.get_run(step["run_id"]) if step.get("run_id") else None
        result = (run or {}).get("result") or {}
        rows.append(
            {
                "harness": (run or {}).get("harness") or step.get("harness"),
                "status": (run or {}).get("status") or "not started",
                "verified": bool(result.get("verified")),
                "cost_usd": result.get("cost_usd"),
                "latency_ms": result.get("latency_ms"),
                "answer": str(result.get("final_response") or "").strip(),
                "error": result.get("error") or (run or {}).get("error"),
                "run_id": step.get("run_id"),
            }
        )
    return {
        "workflow_id": workflow_id,
        "status": value.get("status"),
        "summary": value.get("summary") or {},
        "rows": rows,
    }


__all__ = [
    "HARNESS_ALIASES",
    "DELEGATE",
    "META_HARNESS_TOOL_NAMES",
    "active_harness",
    "apply_harness",
    "delegation_enabled",
    "harness_delegates",
    "memagent_for_compare",
    "parse_compare_args",
    "run_compare",
    "canonical_harness",
    "clear_harness",
    "list_harness_status",
    "pending_harness_approvals",
    "prompt_label",
    "report_pending_approvals",
]
