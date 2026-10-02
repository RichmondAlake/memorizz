"""Shape execution records for the monitor pages: harness runs, automations
and the playground launcher.

Pure functions over the plain dicts and models the stores already return, so
each page gets one summary tape, one row per record and the detail its pane
shows, without extra reads. Times come back as timezone-aware datetimes; the
templates format them with the shared overview helpers.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .helpers import _agent_field

# Harness run status -> (filter group, signal). The signal picks the status dot:
# good (green), warn (amber), bad (red), active (accent) or idle (muted).
HARNESS_STATUS = {
    "pending_approval": ("approval", "warn"),
    "queued": ("active", "active"),
    "running": ("active", "active"),
    "succeeded": ("succeeded", "good"),
    "failed": ("failed", "bad"),
    "verification_failed": ("failed", "bad"),
    "budget_exceeded": ("failed", "bad"),
    "interrupted": ("failed", "bad"),
    "canceled": ("canceled", "idle"),
}
CANCELLABLE = {"queued", "running", "pending_approval"}
AUTOMATION_SIGNAL = {
    "succeeded": "good",
    "failed": "bad",
    "running": "active",
    "canceled": "idle",
}
LINE_CHARS = 200
# Runs read per automation job: enough for a recent-results strip.
AUTOMATION_RECENT_RUNS = 10


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def parse_time(value: Any) -> Optional[datetime]:
    """A timezone-aware datetime from a datetime or ISO string, else None."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def first_line(text: Any, limit: int = LINE_CHARS) -> str:
    """The first non-empty line, whitespace collapsed and bounded."""
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line if len(line) <= limit else line[: limit - 1] + "…"
    return ""


def humanize(code: Any) -> str:
    """``verification_failed`` -> ``Verification failed``."""
    text = str(code or "").replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def _number(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _latest(values: Iterable[Optional[datetime]]) -> Optional[datetime]:
    present = [value for value in values if value is not None]
    return max(present) if present else None


# ---------------------------------------------------------------- harnesses


def harness_availability(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per adapter: readiness, the fix when it is not ready."""
    rows = []
    for item in items or []:
        if not isinstance(item, Mapping):
            continue
        ready = bool(item.get("ready"))
        if ready:
            state, label, signal = "ready", "Ready", "good"
        elif item.get("error_code") == "authentication_required":
            state, label, signal = "setup", "Setup required", "warn"
        else:
            state, label, signal = "unavailable", "Unavailable", "idle"
        features = [
            name
            for key, name in (
                ("structured_events", "Structured events"),
                ("mcp", "MCP"),
                ("resume", "Resume"),
                ("usage_reporting", "Usage reporting"),
                ("per_action_approvals", "Per-action approvals"),
                ("requires_external_isolation", "Needs external isolation"),
            )
            if item.get(key)
        ]
        rows.append(
            {
                "name": str(item.get("name") or ""),
                "ready": ready,
                "state": state,
                "label": label,
                "signal": signal,
                "version": str(item.get("version") or ""),
                "command": str(item.get("command") or ""),
                "error": str(item.get("error") or ""),
                "remediation": str(item.get("remediation") or ""),
                "features": features,
                "models": [str(m) for m in (item.get("models") or []) if m],
            }
        )
    return rows


def _verification_state(spec: Mapping[str, Any], result: Mapping[str, Any]):
    evidence = _mapping(result.get("verification"))
    if result and evidence.get("required"):
        if result.get("verified") or evidence.get("verified"):
            return "verified", "Verified", evidence
        return "failed", "Failed", evidence
    if spec.get("command"):
        return "pending", "Pending", evidence
    return "none", "Not required", evidence


def shape_harness_run(
    run: Mapping[str, Any],
    *,
    approval: Optional[Mapping[str, Any]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """One ledger row plus everything the detail pane shows for it."""
    now = now or datetime.now(timezone.utc)
    task = _mapping(run.get("task"))
    permissions = _mapping(task.get("permissions"))
    budget = _mapping(task.get("budget"))
    spec = _mapping(task.get("verification"))
    metadata = _mapping(task.get("metadata"))
    result = _mapping(run.get("result"))
    usage = _mapping(result.get("usage"))
    status = str(run.get("status") or "queued")
    group, signal = HARNESS_STATUS.get(status, ("other", "idle"))

    requested = str(task.get("harness") or "auto")
    harness = str(run.get("harness") or (requested if requested != "auto" else ""))
    created = parse_time(run.get("created_at"))
    started = parse_time(run.get("started_at"))
    finished = parse_time(run.get("finished_at"))
    updated = parse_time(run.get("updated_at")) or finished or started or created

    duration_ms = _number(result.get("latency_ms"))
    if duration_ms is None and started is not None:
        end = finished or (now if status == "running" else None)
        if end is not None:
            duration_ms = max(0.0, (end - started).total_seconds() * 1000)
    cost = _number(result.get("cost_usd"))
    input_tokens = _number(usage.get("input_tokens"))
    output_tokens = _number(usage.get("output_tokens"))
    tokens = (
        None
        if input_tokens is None and output_tokens is None
        else (input_tokens or 0) + (output_tokens or 0)
    )
    verify_state, verify_label, evidence = _verification_state(spec, result)

    if result.get("error_code"):
        outcome = humanize(result.get("error_code"))
    elif status == "succeeded":
        outcome = "Verified" if verify_state == "verified" else "Complete"
    elif result:
        outcome = humanize(status)
    else:
        outcome = ""

    workspace = str(task.get("workspace") or "")
    workspace_name = workspace.rstrip("/").rsplit("/", 1)[-1] or workspace
    text = str(task.get("task") or "")
    workflow = _mapping(metadata.get("orchestration"))
    return {
        "key": str(run.get("run_id") or ""),
        "run_id": str(run.get("run_id") or ""),
        "status": status,
        "status_label": humanize(status),
        "group": group,
        "signal": signal,
        "cancellable": status in CANCELLABLE,
        "harness": harness,
        "harness_label": harness or "Auto route",
        "requested_harness": requested,
        "task": text,
        "task_line": first_line(text),
        "workspace": workspace,
        "workspace_name": workspace_name,
        "writes": permissions.get("workspace_mode") == "direct",
        "mode_label": (
            "Direct edits"
            if permissions.get("workspace_mode") == "direct"
            else "Read only"
        ),
        "network": humanize(permissions.get("network") or "none"),
        "mcp_access": humanize(permissions.get("mcp_access") or "read_only"),
        "allow_subagents": bool(permissions.get("allow_subagents")),
        "allow_dirty": bool(permissions.get("allow_dirty_workspace")),
        "backend": humanize(metadata.get("execution_backend") or "local"),
        "source": str(metadata.get("source") or ""),
        # A Codex or Claude Code session the MemoRizz plugin recorded.
        "plugin_session": metadata.get("source") == "plugin",
        "turns": int(metadata.get("turns") or 0),
        "session_log": str(metadata.get("session_log") or ""),
        "mode": str(task.get("mode") or "runtime"),
        # The model the harness actually used: the override, else the one the
        # adapter reported (e.g. Codex's default from its catalog).
        # What the harness reported running wins over what was asked for.
        "model": str(usage.get("model") or task.get("model") or ""),
        "model_is_default": not task.get("model") and bool(usage.get("model")),
        "agent_id": str(task.get("agent_id") or ""),
        # A harness delegate's share of a MemAgent harness run.
        "parent_run_id": str(_mapping(task.get("metadata")).get("parent_run_id") or ""),
        "memory_id": str(task.get("memory_id") or ""),
        "user_id": str(task.get("user_id") or ""),
        "limits": {
            "wall_seconds": _number(budget.get("max_wall_time_seconds")),
            "steps": _number(budget.get("max_steps")),
            "cost_usd": _number(budget.get("max_cost_usd")),
            "input_tokens": _number(budget.get("max_input_tokens")),
            "output_tokens": _number(budget.get("max_output_tokens")),
        },
        "verify_state": verify_state,
        "verify_label": verify_label,
        "verify_command": str(spec.get("command") or evidence.get("command") or ""),
        "verify_return_code": evidence.get("return_code"),
        "verify_output": str(
            evidence.get("stderr") or evidence.get("stdout") or ""
        ).strip()[-4000:],
        "outcome": outcome,
        "final_response": str(result.get("final_response") or ""),
        "error_code": str(result.get("error_code") or ""),
        "error": str(result.get("error") or ""),
        "remediation": str(result.get("remediation") or ""),
        "cost_usd": cost,
        "cost_estimated": usage.get("cost_basis") == "list_rate_estimate",
        "tokens": tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_ms": duration_ms,
        "created_at": created,
        "started_at": started,
        "finished_at": finished,
        "updated_at": updated,
        "approval_id": str(
            run.get("approval_proposal_id") or (approval or {}).get("proposal_id") or ""
        ),
        "approval": dict(approval) if approval else None,
        "workflow": (
            {
                "id": str(workflow.get("id") or ""),
                "kind": str(workflow.get("kind") or ""),
                "step": int(workflow.get("step") or 0),
                "steps": int(workflow.get("steps") or 0),
                "name": str(workflow.get("name") or ""),
                "label": (
                    f"Stage {int(workflow.get('step') or 0) + 1}/"
                    f"{int(workflow.get('steps') or 0)}"
                    if workflow.get("kind") == "plan"
                    else "Compare"
                ),
            }
            if workflow.get("id")
            else None
        ),
    }


WORKFLOW_KIND = {"plan": "Staged plan", "compare": "Comparison"}
EXCERPT_CHARS = 280


def delegate_cost_totals(
    runs: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """For each run that MemAgent delegates worked for: how many delegate
    runs it led to (nested ones too) and what they cost together."""
    children: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        parent = _mapping(_mapping(run.get("task")).get("metadata")).get(
            "parent_run_id"
        )
        if parent:
            children.setdefault(str(parent), []).append(run)
    totals: Dict[str, Dict[str, Any]] = {}

    def walk(run_id: str, seen: set) -> Dict[str, Any]:
        found = {"runs": 0, "cost_usd": None, "estimated": False, "unpriced": 0}
        for child in children.get(run_id, []):
            child_id = str(child.get("run_id") or "")
            if not child_id or child_id in seen:
                continue
            seen.add(child_id)
            result = _mapping(child.get("result"))
            cost = _number(result.get("cost_usd"))
            nested = walk(child_id, seen)
            found["runs"] += 1 + nested["runs"]
            found["unpriced"] += nested["unpriced"] + (cost is None)
            for part in (cost, nested["cost_usd"]):
                if part is not None:
                    found["cost_usd"] = (found["cost_usd"] or 0.0) + part
            found["estimated"] = bool(
                found["estimated"]
                or nested["estimated"]
                or _mapping(result.get("usage")).get("cost_basis")
                == "list_rate_estimate"
            )
        return found

    for run_id in children:
        totals[run_id] = walk(run_id, {run_id})
    return totals


def with_delegate_costs(
    row: Dict[str, Any], totals: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Add a run's delegates' cost and the total to its shaped row."""
    extra = totals.get(row["run_id"]) or {}
    row["delegate_runs"] = int(extra.get("runs") or 0)
    row["delegate_cost_usd"] = extra.get("cost_usd")
    row["delegate_unpriced"] = int(extra.get("unpriced") or 0)
    if row["delegate_cost_usd"] is None:
        row["total_cost_usd"] = row["cost_usd"]
        row["total_cost_estimated"] = row["cost_estimated"]
    else:
        row["total_cost_usd"] = (row["cost_usd"] or 0.0) + row["delegate_cost_usd"]
        row["total_cost_estimated"] = bool(
            row["cost_estimated"] or extra.get("estimated")
        )
    return row


def shape_workflow(
    workflow: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    *,
    approvals: Optional[Mapping[str, Mapping[str, Any]]] = None,
    now: Optional[datetime] = None,
    delegate_totals: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """One plan or comparison: its status, and each stage's run at a glance."""
    now = now or datetime.now(timezone.utc)
    status = str(workflow.get("status") or "queued")
    group, signal = HARNESS_STATUS.get(status, ("other", "idle"))
    kind = str(workflow.get("kind") or "plan")
    task = _mapping(workflow.get("task"))
    steps = []
    costs = []
    for index, step in enumerate(workflow.get("steps") or []):
        step = _mapping(step)
        run_id = str(step.get("run_id") or "")
        run = runs.get(run_id) if run_id else None
        row = (
            with_delegate_costs(
                shape_harness_run(run, approval=(approvals or {}).get(run_id), now=now),
                delegate_totals or {},
            )
            if run
            else None
        )
        if row and row["total_cost_usd"] is not None:
            costs.append(row["total_cost_usd"])
        answer = (row or {}).get("final_response") or ""
        error = (row or {}).get("error") or ""
        handoff = _mapping(step.get("handoff"))
        steps.append(
            {
                "index": index,
                "number": index + 1,
                "name": str(step.get("name") or step.get("harness") or ""),
                "harness": (row or {}).get("harness")
                or str(step.get("harness") or "auto"),
                "writes": step.get("workspace_mode") == "direct",
                "run_id": run_id,
                "run": row,
                "status": row["status"] if row else "not_started",
                "status_label": row["status_label"] if row else "Not started",
                "signal": row["signal"] if row else "idle",
                # The whole answer, flattened: many start with a lead-in line.
                "excerpt": first_line(
                    " ".join(str(error or answer).split()), EXCERPT_CHARS
                ),
                "is_error": bool(error),
                "files_changed": list(handoff.get("files_changed") or []),
                "remembered": bool(handoff.get("memory_record_id")),
            }
        )
    started = parse_time(workflow.get("started_at")) or parse_time(
        workflow.get("created_at")
    )
    finished = parse_time(workflow.get("finished_at"))
    end = finished or (now if group in {"active", "approval"} else None)
    done = sum(1 for step in steps if step["status"] == "succeeded")
    if kind == "plan":
        current = workflow.get("current_step")
        progress = f"{done} of {len(steps)} stages done"
        if group in {"active", "approval"} and current is not None:
            progress = f"Stage {int(current) + 1} of {len(steps)}"
    else:
        progress = f"{done} of {len(steps)} succeeded"
    return {
        "id": str(workflow.get("orchestration_id") or ""),
        "kind": kind,
        "kind_label": WORKFLOW_KIND.get(kind, humanize(kind)),
        "status": status,
        "status_label": humanize(status),
        "group": group,
        "signal": signal,
        "cancellable": status in CANCELLABLE and not workflow.get("cancel_requested"),
        "canceling": bool(workflow.get("cancel_requested")) and status in CANCELLABLE,
        "stale": bool(workflow.get("stale")),
        "goal": str(task.get("task") or ""),
        "goal_line": first_line(task.get("task")),
        "workspace": str(task.get("workspace") or ""),
        "memory_id": str(task.get("memory_id") or ""),
        "error": str(workflow.get("error") or ""),
        "error_code": str(workflow.get("error_code") or ""),
        "steps": steps,
        "progress": progress,
        "cost_usd": sum(costs) if costs else None,
        "duration_ms": (
            max(0.0, (end - started).total_seconds() * 1000)
            if started is not None and end is not None
            else None
        ),
        "created_at": parse_time(workflow.get("created_at")),
        "updated_at": parse_time(workflow.get("updated_at")),
        "remembered": sum(1 for step in steps if step["remembered"]),
        "rerunnable": group not in {"active", "approval"}
        and kind in {"plan", "compare"},
        "rerun_of": str(_mapping(task.get("metadata")).get("rerun_of") or ""),
        "setup": workflow_setup(kind, task, workflow.get("steps") or []),
    }


def _stage_role(instruction: Any, goal: str) -> str:
    """A stage's own instruction, without the goal a launch appended to it."""
    text = str(instruction or "")
    suffix = f"\n\nGoal:\n{goal}"
    return text[: -len(suffix)] if goal and text.endswith(suffix) else text


def workflow_setup(
    kind: str, task: Mapping[str, Any], steps: Iterable[Any]
) -> Dict[str, Any]:
    """A workflow's settings in the launch form's terms, to launch it again."""
    task = _mapping(task)
    goal = str(task.get("task") or "")
    permissions = _mapping(task.get("permissions"))
    budget = _mapping(task.get("budget"))
    verification = _mapping(task.get("verification"))
    metadata = _mapping(task.get("metadata"))
    setup: Dict[str, Any] = {
        "mode": kind,
        "task": goal,
        "workspace": str(task.get("workspace") or ""),
        "agent_id": str(task.get("agent_id") or ""),
        "model": str(task.get("model") or ""),
        "verification_command": str(verification.get("command") or ""),
        "memory_id": str(task.get("memory_id") or ""),
        "user_id": str(task.get("user_id") or ""),
        "thread_id": str(task.get("thread_id") or ""),
        "timeout_seconds": budget.get("max_wall_time_seconds"),
        "max_cost_usd": budget.get("max_cost_usd"),
        "max_steps": budget.get("max_steps"),
        "max_input_tokens": budget.get("max_input_tokens"),
        "max_output_tokens": budget.get("max_output_tokens"),
        "execution_backend": str(metadata.get("execution_backend") or "local"),
        "network": str(permissions.get("network") or "none"),
        "mcp_access": str(permissions.get("mcp_access") or "read_only"),
        "allowed_env": ", ".join(permissions.get("allowed_env") or []),
        "allowed_tools": ", ".join(permissions.get("allowed_tools") or []),
        "denied_tools": ", ".join(permissions.get("denied_tools") or []),
        "allow_dirty_workspace": bool(permissions.get("allow_dirty_workspace")),
        "allow_subagents": bool(permissions.get("allow_subagents")),
    }
    rows = [_mapping(step) for step in steps]
    if kind == "compare":
        setup["harnesses"] = [str(row.get("harness") or "") for row in rows]
        setup["harness_models"] = {
            str(row.get("harness") or ""): str(row.get("model") or "")
            for row in rows
            if row.get("model")
        }
    else:
        setup["stages"] = [
            {
                "name": str(row.get("name") or ""),
                "harness": str(row.get("harness") or "auto"),
                "instruction": _stage_role(row.get("instruction"), goal),
                "write": row.get("workspace_mode") == "direct",
                "model": str(row.get("model") or ""),
                "agent_id": str(row.get("agent_id") or ""),
                "verification_command": str(
                    _mapping(row.get("verification")).get("command") or ""
                ),
            }
            for row in rows
        ]
    return setup


def build_harness_monitor(
    harnesses: Iterable[Mapping[str, Any]],
    runs: Iterable[Mapping[str, Any]],
    approvals: Iterable[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
    run_limit: int = 100,
    workflows: Iterable[Mapping[str, Any]] = (),
    workflow_runs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Tape figures, availability, workflows and ledger rows for the harnesses
    page."""
    now = now or datetime.now(timezone.utc)
    availability = harness_availability(harnesses)
    pending = [
        {**a, "expires": parse_time(a.get("expires_at"))}
        for a in approvals or []
        if isinstance(a, Mapping)
    ]
    by_run = {}
    for proposal in pending:
        run_id = _mapping(proposal.get("arguments")).get("run_id")
        if run_id:
            by_run[str(run_id)] = proposal
    rows = [
        shape_harness_run(run, approval=by_run.get(str(run.get("run_id"))), now=now)
        for run in runs or []
        if isinstance(run, Mapping)
    ]
    run_index = dict(workflow_runs or {})
    for run in runs or []:
        if isinstance(run, Mapping):
            run_index.setdefault(str(run.get("run_id")), run)
    delegate_totals = delegate_cost_totals(run_index.values())
    for row in rows:
        with_delegate_costs(row, delegate_totals)
    groups = Counter(row["group"] for row in rows)
    costs = [row["cost_usd"] for row in rows if row["cost_usd"] is not None]
    checked = [row for row in rows if row["verify_state"] in ("verified", "failed")]
    ready = sum(1 for item in availability if item["ready"])
    ledger_ids = {row["run_id"] for row in rows}
    shaped_workflows = [
        shape_workflow(
            item, run_index, approvals=by_run, now=now, delegate_totals=delegate_totals
        )
        for item in workflows or []
        if isinstance(item, Mapping)
    ]
    return {
        "workflows": shaped_workflows,
        "workflows_active": sum(
            1 for item in shaped_workflows if item["group"] in {"active", "approval"}
        ),
        "availability": availability,
        "ready": ready,
        "total_harnesses": len(availability),
        "needs_setup": [item for item in availability if not item["ready"]],
        "runs": rows,
        "run_count": len(rows),
        "has_more": len(rows) >= run_limit,
        "counts": {
            "active": groups.get("active", 0),
            "approval": groups.get("approval", 0),
            "succeeded": groups.get("succeeded", 0),
            "failed": groups.get("failed", 0),
            "canceled": groups.get("canceled", 0),
        },
        "awaiting_approval": len(pending),
        "approvals": pending,
        # Proposals whose run is not in the ledger page still need a decision.
        "unlisted_approvals": [
            p
            for p in pending
            if str(_mapping(p.get("arguments")).get("run_id") or "") not in ledger_ids
        ],
        "verified": sum(1 for row in checked if row["verify_state"] == "verified"),
        "verification_checked": len(checked),
        "spend": sum(costs) if costs else None,
        "last_activity": _latest(row["updated_at"] for row in rows),
        "harness_names": sorted({row["harness"] for row in rows if row["harness"]}),
    }


# -------------------------------------------------------------- automations


def describe_schedule(job: Any) -> str:
    """``Cron 0 8 * * 1-5``, ``Every 1h`` or ``One-shot``."""
    kind = str(_agent_field(job, "schedule_type", "") or "")
    if kind == "cron":
        expr = str(_agent_field(job, "cron_expr", "") or "")
        return f"Cron {expr}".strip()
    if kind == "interval":
        seconds = int(_number(_agent_field(job, "interval_seconds")) or 0)
        if not seconds:
            return "Interval"
        for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
            if seconds >= size and seconds % size == 0:
                return f"Every {seconds // size}{unit}"
        return f"Every {seconds}s"
    if kind == "one_shot":
        return "One-shot"
    return humanize(kind) or "—"


def _run_duration_ms(run: Mapping[str, Any]) -> Optional[float]:
    started = parse_time(run.get("started_at"))
    finished = parse_time(run.get("finished_at"))
    if started is None or finished is None:
        return None
    return max(0.0, (finished - started).total_seconds() * 1000)


def shape_automation_job(
    job: Mapping[str, Any],
    *,
    agent_names: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """One grid row per job: schedule, next run, latest result, recent history."""
    now = now or datetime.now(timezone.utc)
    agent_id = str(job.get("agent_id") or "")
    latest = _mapping(job.get("latest_run"))
    history = [
        _mapping(run) for run in (job.get("recent_runs") or []) if run is not None
    ]
    if not history and latest:
        history = [latest]
    runs = []
    for run in history:
        status = str(run.get("status") or "unknown")
        runs.append(
            {
                "status": status,
                "label": humanize(status),
                "signal": AUTOMATION_SIGNAL.get(status, "idle"),
                "at": parse_time(
                    run.get("finished_at")
                    or run.get("started_at")
                    or run.get("scheduled_for")
                ),
                "duration_ms": _run_duration_ms(run),
                "error": first_line(run.get("error")),
            }
        )
    last_status = str(latest.get("status") or "") if latest else ""
    payload = _mapping(latest.get("result_payload"))
    output = str(latest.get("result_summary") or payload.get("response") or "")
    enabled = bool(job.get("enabled"))
    next_run = parse_time(job.get("next_run_at"))
    tags = ["active" if enabled else "paused"]
    if last_status == "failed":
        tags.append("failed")
    if not latest:
        tags.append("never")
    action = _mapping(job.get("action_config"))
    delivery = _mapping(job.get("delivery_config"))
    return {
        "key": str(job.get("job_id") or ""),
        "job_id": str(job.get("job_id") or ""),
        "name": str(job.get("name") or "Automation"),
        "agent_id": agent_id,
        "agent_name": (agent_names or {}).get(agent_id) or "",
        "enabled": enabled,
        "tags": tags,
        "schedule": describe_schedule(job),
        "schedule_type": str(job.get("schedule_type") or ""),
        "cron_expr": str(job.get("cron_expr") or ""),
        "interval_seconds": job.get("interval_seconds"),
        "timezone": str(job.get("timezone") or "UTC"),
        "next_run_at": next_run,
        "next_run_overdue": bool(enabled and next_run and next_run < now),
        "last_run_at": parse_time(job.get("last_run_at"))
        or (runs[0]["at"] if runs else None),
        "last_status": last_status,
        "last_label": humanize(last_status) if last_status else "Never run",
        "last_signal": AUTOMATION_SIGNAL.get(last_status, "idle"),
        "last_at": runs[0]["at"] if runs else None,
        "output": output,
        "output_line": first_line(output),
        "error": str(latest.get("error") or ""),
        "trace_memory_id": str(payload.get("memory_id") or ""),
        # Oldest first, so the strip reads left to right like a chart.
        "history": list(reversed(runs)),
        "recent": runs,
        "failures": sum(1 for run in runs if run["status"] == "failed"),
        "query": str(action.get("query_template") or ""),
        "action_type": humanize(job.get("action_type") or ""),
        "delivery": humanize(job.get("delivery_type") or "") or "None",
        "recipients": len(delivery.get("whatsapp_to") or []),
        "misfire_policy": humanize(job.get("misfire_policy") or ""),
        "max_run_seconds": job.get("max_run_seconds"),
        "retry_max_attempts": job.get("retry_max_attempts"),
    }


def build_automation_monitor(
    jobs: Iterable[Mapping[str, Any]],
    *,
    agent_names: Optional[Mapping[str, str]] = None,
    now: Optional[datetime] = None,
    runs_per_job: Optional[int] = None,
) -> Dict[str, Any]:
    """Tape figures and grid rows for the automations page."""
    now = now or datetime.now(timezone.utc)
    rows = [
        shape_automation_job(job, agent_names=agent_names, now=now)
        for job in jobs or []
        if isinstance(job, Mapping)
    ]
    upcoming = [
        row
        for row in rows
        if row["enabled"] and row["next_run_at"] and not row["next_run_overdue"]
    ]
    upcoming.sort(key=lambda row: row["next_run_at"])
    runs_recorded = sum(len(row["recent"]) for row in rows)
    results = [row for row in rows if row["last_at"] is not None]
    results.sort(key=lambda row: row["last_at"], reverse=True)
    return {
        "jobs": rows,
        "total": len(rows),
        "active": sum(1 for row in rows if row["enabled"]),
        "paused": sum(1 for row in rows if not row["enabled"]),
        "failed_last": sum(1 for row in rows if row["last_status"] == "failed"),
        "never_run": sum(1 for row in rows if "never" in row["tags"]),
        # Enabled jobs past their next run time: no worker is picking them up.
        "overdue": sum(1 for row in rows if row["next_run_overdue"]),
        "runs_recorded": runs_recorded,
        # Only the newest runs per job are read; say so when a job hit the cap.
        "runs_capped": bool(
            runs_per_job and any(len(row["recent"]) >= runs_per_job for row in rows)
        ),
        "next": upcoming[0] if upcoming else None,
        "last_activity": results[0]["last_at"] if results else None,
        "latest_results": results,
    }


# ---------------------------------------------------------------- launcher


def build_playground_launcher(
    agents: Iterable[Any],
    *,
    last_run_by_agent: Optional[Mapping[str, float]] = None,
    now: Optional[datetime] = None,
    default_model: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """One row per agent, most recently used first, from data already loaded."""
    now = now or datetime.now(timezone.utc)
    last_runs = last_run_by_agent or {}
    rows = []
    for index, agent in enumerate(agents or []):
        agent_id = str(
            _agent_field(agent, "agent_id", "") or _agent_field(agent, "_id", "")
        ).strip()
        if not agent_id:
            continue
        persona = _agent_field(agent, "persona")
        name = str(_agent_field(agent, "name", "") or "").strip() or (
            str(_agent_field(persona, "name", "") or "").strip() if persona else ""
        )
        llm = _mapping(_agent_field(agent, "llm_config"))
        provider = str(llm.get("provider") or "").strip().lower()
        model = str(llm.get("model") or llm.get("deployment_name") or "")
        if not model and default_model is not None:
            model = default_model(provider)
        stamp = _number(last_runs.get(agent_id))
        last_run = datetime.fromtimestamp(stamp, tz=timezone.utc) if stamp else None
        memory_ids = [m for m in (_agent_field(agent, "memory_ids", []) or []) if m]
        rows.append(
            {
                "key": agent_id,
                "agent_id": agent_id,
                "name": name or "Agent",
                "model": model,
                "provider": provider,
                "mode": humanize(
                    _agent_field(agent, "application_mode", "") or "assistant"
                ),
                "role": str(_agent_field(persona, "role", "") or "") if persona else "",
                "instruction": first_line(_agent_field(agent, "instruction", ""), 160),
                "threads": len(memory_ids),
                "last_run": last_run,
                "favorite": bool(_agent_field(agent, "is_favorite", False)),
                "order": index,
            }
        )
    # Most recently used first; never-used agents keep their stored order.
    rows.sort(
        key=lambda row: (
            row["last_run"] is None,
            -row["last_run"].timestamp() if row["last_run"] else 0,
            row["order"],
        )
    )
    day, week = now - timedelta(days=1), now - timedelta(days=7)
    return {
        "agents": rows,
        "total": len(rows),
        "used_day": sum(
            1 for row in rows if row["last_run"] and row["last_run"] >= day
        ),
        "used_week": sum(
            1 for row in rows if row["last_run"] and row["last_run"] >= week
        ),
        "last_activity": _latest(row["last_run"] for row in rows),
        "threads": sum(row["threads"] for row in rows),
        "models": len({row["model"] for row in rows if row["model"]}),
    }
