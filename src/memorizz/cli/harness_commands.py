"""CLI for the MemoRizz meta-harness: runs, plans, comparisons, conversations,
approvals, deletion, model choices and harness delegates."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import typer

from . import config as cli_config

harness_app = typer.Typer(
    help=(
        "Run, orchestrate and inspect Codex, Claude Code, OpenHands, DeepSeek, "
        "pi, Hermes and native MemAgent harnesses."
    ),
    no_args_is_help=True,
)
delegate_app = typer.Typer(
    help=(
        "Agents that run their share of a coordinator's work on a harness "
        "(harness delegates)."
    ),
    no_args_is_help=True,
)
harness_app.add_typer(delegate_app, name="delegate")

# How often a foreground plan, comparison or conversation turn reports progress.
FOLLOW_POLL_SECONDS = 1.0
FINAL_STATES = {
    "succeeded",
    "failed",
    "canceled",
    "interrupted",
    "budget_exceeded",
    "verification_failed",
}


def _print(value: Any, raw_json: bool = False) -> None:
    text = json.dumps(
        value, indent=None if raw_json else 2, sort_keys=True, default=str
    )
    typer.echo(text)


def _service(*, require_memory: bool = True):
    from ..metaharness import MetaHarness
    from .agent_commands import _provider

    cli_config.load_layered_env()
    try:
        provider, warnings = _provider()
    except Exception as exc:
        if require_memory:
            raise typer.BadParameter(
                "The configured memory provider could not be initialized. "
                "Check `memorizz config` and the provider credentials."
            ) from exc
        provider = None
        warnings = [
            "The configured memory provider is unavailable; this diagnostic "
            f"omits saved MemAgent readiness ({type(exc).__name__})."
        ]
    return MetaHarness.from_env(memory_provider=provider), provider, warnings


def _clean_errors() -> tuple:
    from ..approval import ApprovalError
    from ..metaharness.router import HarnessReadinessError

    # HarnessSecurityError is a ValueError.
    return (KeyError, ValueError, TypeError, ApprovalError, HarnessReadinessError)


@contextmanager
def _session(*, require_memory: bool = True) -> Iterator[tuple[Any, List[str]]]:
    """The harness service for one command, closed with its provider.

    Unknown IDs and refused requests end the command with their message
    (exit code 1), not a traceback.
    """
    service, provider, warnings = (
        _service() if require_memory else _service(require_memory=False)
    )
    try:
        yield service, warnings
    except typer.Exit:
        raise
    except _clean_errors() as exc:
        typer.echo(str(exc).strip("'\""), err=True)
        raise typer.Exit(1) from None
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


def _run_id(service: Any, value: str) -> str:
    """Accept a full run id or a unique prefix (the ids the chat and the UI
    show are shortened). Raises KeyError with a clear message otherwise."""
    value = str(value or "").strip()
    if not value:
        raise KeyError("Harness run not found: (empty id)")
    if service.get_run(value) is not None:
        return value
    matches = sorted(
        {
            str(run.get("run_id"))
            for run in service.list_runs(limit=1000) or []
            if str(run.get("run_id") or "").startswith(value)
        }
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Harness run not found: {value}")
    raise KeyError(
        f"Run id prefix {value} matches {len(matches)} runs; give more characters: "
        + ", ".join(m[:12] for m in matches[:6])
    )


def _canonical(name: Optional[str]) -> str:
    from ..metaharness.service import HARNESS_ALIASES

    value = str(name or "").strip().lower()
    return HARNESS_ALIASES.get(value, value)


def _saved_agent(
    service: Any, agent_id: Optional[str], harnesses: List[Any]
) -> Optional[str]:
    """The saved MemAgent a memagent run uses: the one named, else the one
    last used with a harness, else the newest (as the UI does)."""
    from ..metaharness import catalog

    if agent_id or "memagent" not in {_canonical(name) for name in harnesses}:
        return agent_id
    provider = getattr(service, "memory_provider", None)
    lister = getattr(provider, "list_memagents", None)
    agents = list(lister() or []) if callable(lister) else []
    payload, chosen = catalog.with_memagent(
        {},
        ["memagent"],
        lambda: catalog.default_memagent(service, agents),
        where_to_create="with `memorizz agents create`",
    )
    if chosen:
        typer.echo(
            f"memagent will run {chosen['name']} ({chosen['id'][:8]}).", err=True
        )
    return payload.get("agent_id")


def _pairs(items: Optional[List[str]], option: str) -> Dict[str, str]:
    """KEY=VALUE options into a dict."""
    found: Dict[str, str] = {}
    for item in items or []:
        key, _, value = str(item).partition("=")
        if not key.strip() or not value.strip():
            raise typer.BadParameter(f"{item!r} must be KEY=VALUE", param_hint=option)
        found[key.strip()] = value.strip()
    return found


def _task_options(
    task: str,
    workspace: Optional[Path],
    scratch: bool,
    service: Any,
    **options: Any,
):
    """One HarnessTask from the options run, plan, compare and continue share."""
    from ..metaharness.requests import harness_task

    if scratch:
        folder = service.scratch_workspace()
    else:
        folder = str(workspace or Path.cwd())
    schema_path = options.pop("output_schema_path", None)
    schema = None
    if schema_path:
        try:
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"--output-schema: {exc}") from exc
    backend = options.pop("execution_backend", None) or "local"
    return harness_task(
        task,
        folder,
        output_schema=schema,
        metadata={"execution_backend": backend, "source": "cli"},
        **options,
    )


# Options shared by run, plan, compare and continue.
JSON_OUT = typer.Option(False, "--json", help="Emit compact JSON.")
WORKSPACE = typer.Option(
    None, "--workspace", "-C", help="Project folder (default: current folder)."
)
SCRATCH = typer.Option(
    False, "--scratch", help="Run in a fresh empty folder instead of a project."
)
NETWORK = typer.Option(
    "none", "--network", help="none, restricted or full (full asks for approval)."
)
MCP_ACCESS = typer.Option(
    "read_only", "--mcp-access", help="MemoRizz MCP: none, read_only or governed_write."
)
ALLOW_ENV = typer.Option(
    None, "--allow-env", help="Environment variable name; repeat as needed."
)
ALLOW_TOOL = typer.Option(None, "--allow-tool", help="Only these tools; repeatable.")
DENY_TOOL = typer.Option(None, "--deny-tool", help="Never these tools; repeatable.")
ALLOW_SUBAGENTS = typer.Option(
    False,
    "--allow-subagents",
    help=(
        "Let Claude Code use its Task tool and Codex start sub-agents (and a "
        "memagent agent's harness delegates do the same)."
    ),
)
AGENT_ID = typer.Option(
    None,
    "--agent-id",
    help="Saved MemAgent for --harness memagent (default: the last one used, else the newest).",
)
EXECUTION_BACKEND = typer.Option(
    "local",
    "--execution-backend",
    help="local, docker, sandbox or remote (OpenHands needs one other than local).",
)
ALLOW_DIRTY = typer.Option(
    False,
    "--allow-dirty",
    help="Let an edit run start on a Git folder with uncommitted changes.",
)
MEMORY_ID = typer.Option(
    None, "--memory-id", help="Save answers to this memory, and recall from it."
)
USER_ID = typer.Option(None, "--user-id", help="End user the run is for.")
THREAD_ID = typer.Option(None, "--thread-id", help="Conversation thread in the memory.")
MAX_COST = typer.Option(
    None, "--max-cost-usd", min=0.0, help="Stop when a run's cost passes this."
)
MAX_INPUT = typer.Option(
    None, "--max-input-tokens", min=1, help="Input-token limit per run."
)
MAX_OUTPUT = typer.Option(
    None, "--max-output-tokens", min=1, help="Output-token limit per run."
)
MAX_STEPS = typer.Option(80, "--max-steps", min=1, help="Action limit per run.")
TIMEOUT = typer.Option(
    900, "--timeout", min=1, help="Wall-time limit per run, in seconds."
)
KEEP_SCRATCH = typer.Option(
    False,
    "--keep-scratch",
    help="Keep scratch folders MemoRizz made for these runs (removed by default once unused).",
)


@harness_app.command("list")
def list_harnesses(raw_json: bool = JSON_OUT):
    """List the harnesses and whether each is ready to run."""
    with _session(require_memory=False) as (service, warnings):
        values = service.list_harnesses()
        _print(
            {
                "ok": True,
                "harnesses": values,
                "count": len(values),
                "warnings": warnings,
            },
            raw_json,
        )


@harness_app.command("doctor")
def doctor(
    name: Optional[str] = typer.Argument(None, help="Optional harness name."),
    raw_json: bool = JSON_OUT,
):
    """Check one harness (or all) and say how to fix what isn't ready."""
    with _session(require_memory=False) as (service, warnings):
        values = [service.probe(_canonical(name))] if name else service.list_harnesses()
        ok = all(bool(item.get("ready", item.get("available"))) for item in values)
        _print({"ok": ok, "harnesses": values, "warnings": warnings}, raw_json)
        if name and not ok:
            raise typer.Exit(1)


@harness_app.command("init")
def init_config(
    force: bool = typer.Option(
        False, "--force", help="Overwrite an existing harnesses.json."
    ),
):
    """Write a starter harnesses.json."""
    from ..metaharness.config import (
        default_harness_config,
        harness_config_path,
        save_harness_config,
    )

    target = harness_config_path()
    if target.exists() and not force:
        typer.echo(f"Harness config already exists: {target}", err=True)
        raise typer.Exit(1)
    save_harness_config(default_harness_config(), target)
    typer.echo(str(target))


@harness_app.command("config")
def show_config(raw_json: bool = JSON_OUT):
    """Show harnesses.json and where it is."""
    from ..metaharness.config import harness_config_path, load_harness_config

    _print(
        {
            "ok": True,
            "path": str(harness_config_path()),
            "config": load_harness_config(),
        },
        raw_json,
    )


@harness_app.command("models")
def models(
    name: Optional[str] = typer.Argument(None, help="Optional harness name."),
    raw_json: bool = JSON_OUT,
):
    """Models each harness can run, as the UI's model pickers offer them.

    Each harness's default and the newest models per provider you have a key
    for.
    """
    from ..metaharness import catalog

    with _session(require_memory=False) as (service, _warnings):
        harnesses = service.list_harnesses()
        if name:
            wanted = _canonical(name)
            harnesses = [item for item in harnesses if item.get("name") == wanted]
            if not harnesses:
                raise KeyError(f"Unknown harness: {name}")
        _print(
            {"ok": True, "models": catalog.model_choices(service, harnesses)}, raw_json
        )


@harness_app.command("run")
def run_harness(
    task: str = typer.Argument(..., help="Workspace task for the harness."),
    workspace: Optional[Path] = WORKSPACE,
    scratch: bool = SCRATCH,
    harness: str = typer.Option(
        "auto", "--harness", help="A harness name, or auto to pick from past results."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model for the harness (see `harness models`)."
    ),
    agent_id: Optional[str] = AGENT_ID,
    write: bool = typer.Option(
        False, "--write/--read-only", help="Allow direct edits (asks for approval)."
    ),
    allow_dirty: bool = ALLOW_DIRTY,
    execution_backend: str = EXECUTION_BACKEND,
    network: str = NETWORK,
    mcp_access: str = MCP_ACCESS,
    allowed_env: Optional[List[str]] = ALLOW_ENV,
    allowed_tools: Optional[List[str]] = ALLOW_TOOL,
    denied_tools: Optional[List[str]] = DENY_TOOL,
    allow_subagents: bool = ALLOW_SUBAGENTS,
    output_schema: Optional[Path] = typer.Option(
        None, "--output-schema", help="JSON Schema file the answer must match."
    ),
    verify: Optional[str] = typer.Option(
        None, "--verify", help="Host command that checks the result, e.g. 'pytest -q'."
    ),
    memory_id: Optional[str] = MEMORY_ID,
    user_id: Optional[str] = USER_ID,
    thread_id: Optional[str] = THREAD_ID,
    max_cost_usd: Optional[float] = MAX_COST,
    max_input_tokens: Optional[int] = MAX_INPUT,
    max_output_tokens: Optional[int] = MAX_OUTPUT,
    max_steps: int = MAX_STEPS,
    timeout: int = TIMEOUT,
    raw_json: bool = JSON_OUT,
):
    """Run one bounded task and wait for its result.

    With --harness memagent, a saved MemAgent runs it; if that agent has
    harness delegates, they work in this run's folder with what it was
    approved for.
    """
    with _session() as (service, warnings):
        result = service.run(
            _task_options(
                task,
                workspace,
                scratch,
                service,
                harness=harness,
                model=model,
                agent_id=_saved_agent(service, agent_id, [harness]),
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                write=write,
                allow_dirty_workspace=allow_dirty,
                network=network,
                mcp_access=mcp_access,
                allowed_env=allowed_env,
                allowed_tools=allowed_tools,
                denied_tools=denied_tools,
                allow_subagents=allow_subagents,
                timeout_seconds=timeout,
                max_steps=max_steps,
                max_cost_usd=max_cost_usd,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                verification_command=verify,
                output_schema_path=output_schema,
                execution_backend=execution_backend,
            )
        )
        payload = result.to_dict()
        payload["warnings"] = warnings
        _print(payload, raw_json)
        if result.status.value not in {"succeeded", "pending_approval"}:
            raise typer.Exit(1)


def _follow(service: Any, workflow_id: str) -> dict:
    """Drive a plan or comparison in this process until it finishes.

    Progress goes to stderr; approve edit stages from another terminal or the
    UI. Ctrl+C cancels the workflow.
    """
    seen: dict = {}
    try:
        while True:
            value = service.get_orchestration(workflow_id) or {}
            for index, step in enumerate(value.get("steps") or []):
                run = service.get_run(step["run_id"]) if step.get("run_id") else None
                status = (run or {}).get("status")
                if status and seen.get(index) != status:
                    seen[index] = status
                    line = f"[{index + 1}/{len(value['steps'])}] {step.get('name')}"
                    line += f" ({(run or {}).get('harness') or step.get('harness')})"
                    line += f": {status}"
                    if status == "pending_approval":
                        line += (
                            f". Approve with: memorizz harness approve "
                            f"{run.get('approval_proposal_id')} --approver <you>"
                        )
                    typer.echo(line, err=True)
            if value.get("status") in FINAL_STATES:
                return value
            time.sleep(FOLLOW_POLL_SECONDS)
    except KeyboardInterrupt:
        typer.echo("Canceling the workflow…", err=True)
        service.cancel_orchestration(workflow_id)
        while (service.get_orchestration(workflow_id) or {}).get(
            "status"
        ) not in FINAL_STATES:
            time.sleep(0.2)
        return service.get_orchestration(workflow_id) or {}


def _finish_workflow(service: Any, value: dict, raw_json: bool) -> None:
    runs = [
        service.get_run(step["run_id"])
        for step in value.get("steps") or []
        if step.get("run_id")
    ]
    _print(
        {"ok": value.get("status") == "succeeded", "workflow": value, "runs": runs},
        raw_json,
    )
    if value.get("status") != "succeeded":
        raise typer.Exit(1)


def _wait_for_run(service: Any, run_id: str) -> dict:
    """Wait for a background run to finish or stop for approval. Ctrl+C
    cancels it."""
    try:
        while True:
            run = service.get_run(run_id) or {}
            status = run.get("status")
            if status in FINAL_STATES:
                return run
            if status == "pending_approval":
                typer.echo(
                    "Waiting for approval. Approve with: memorizz harness approve "
                    f"{run.get('approval_proposal_id')} --approver <you>",
                    err=True,
                )
                return run
            time.sleep(FOLLOW_POLL_SECONDS)
    except KeyboardInterrupt:
        typer.echo("Canceling the run…", err=True)
        service.cancel(run_id)
        while (service.get_run(run_id) or {}).get("status") not in FINAL_STATES:
            time.sleep(0.2)
        return service.get_run(run_id) or {}


@harness_app.command("plan")
def plan(
    task: str = typer.Argument(..., help="The goal every stage works toward."),
    stages: List[str] = typer.Option(
        ...,
        "--stage",
        help="NAME:HARNESS or NAME:HARNESS:edit, in order; repeat per stage.",
    ),
    instructions: Optional[List[str]] = typer.Option(
        None, "--instruction", help="NAME=what that stage does; repeatable."
    ),
    stage_verify: Optional[List[str]] = typer.Option(
        None, "--stage-verify", help="NAME=host verification command; repeatable."
    ),
    stage_models: Optional[List[str]] = typer.Option(
        None, "--stage-model", help="NAME=model for that stage; repeatable."
    ),
    stage_agents: Optional[List[str]] = typer.Option(
        None,
        "--stage-agent",
        help="NAME=saved MemAgent ID for a memagent stage; repeatable.",
    ),
    workspace: Optional[Path] = WORKSPACE,
    scratch: bool = SCRATCH,
    allow_dirty: bool = ALLOW_DIRTY,
    execution_backend: str = EXECUTION_BACKEND,
    network: str = NETWORK,
    mcp_access: str = MCP_ACCESS,
    allowed_env: Optional[List[str]] = ALLOW_ENV,
    allowed_tools: Optional[List[str]] = ALLOW_TOOL,
    denied_tools: Optional[List[str]] = DENY_TOOL,
    allow_subagents: bool = ALLOW_SUBAGENTS,
    memory_id: Optional[str] = MEMORY_ID,
    user_id: Optional[str] = USER_ID,
    thread_id: Optional[str] = THREAD_ID,
    agent_id: Optional[str] = AGENT_ID,
    max_cost_usd: Optional[float] = MAX_COST,
    max_input_tokens: Optional[int] = MAX_INPUT,
    max_output_tokens: Optional[int] = MAX_OUTPUT,
    max_steps: int = MAX_STEPS,
    timeout: int = TIMEOUT,
    raw_json: bool = JSON_OUT,
):
    """Run stages in order, each on its own harness, and wait for the plan.

    Example: memorizz harness plan "Fix the rounding bug" --stage plan:pi
    --stage implement:codex:edit --stage review:claude-code
    --stage-model review=claude-sonnet-5-5
    """
    from ..metaharness.requests import parse_stage, plan_stages

    try:
        rows = [parse_stage(spec) for spec in stages]
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    by_name = {row["name"]: row for row in rows}
    for option, key, hint in (
        (instructions, "instruction", "--instruction"),
        (stage_verify, "verification_command", "--stage-verify"),
        (stage_models, "model", "--stage-model"),
        (stage_agents, "agent_id", "--stage-agent"),
    ):
        for name, text in _pairs(option, hint).items():
            if name not in by_name:
                raise typer.BadParameter(f"{name!r} is not a --stage", param_hint=hint)
            by_name[name][key] = text
    with _session() as (service, _warnings):
        harnesses = [row["harness"] for row in rows if not row.get("agent_id")]
        base = _task_options(
            task,
            workspace,
            scratch,
            service,
            agent_id=_saved_agent(service, agent_id, harnesses),
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            allow_dirty_workspace=allow_dirty,
            network=network,
            mcp_access=mcp_access,
            allowed_env=allowed_env,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            allow_subagents=allow_subagents,
            max_cost_usd=max_cost_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout,
            max_steps=max_steps,
            execution_backend=execution_backend,
        )
        started = service.start_plan(base, plan_stages(rows, base.task))
        typer.echo(f"Plan {started['orchestration_id']} started.", err=True)
        _finish_workflow(
            service, _follow(service, started["orchestration_id"]), raw_json
        )


@harness_app.command("compare")
def compare(
    task: str = typer.Argument(..., help="The read-only task every harness gets."),
    harnesses: List[str] = typer.Option(
        ..., "--harness", help="A harness to compare; give two or more."
    ),
    harness_models: Optional[List[str]] = typer.Option(
        None,
        "--harness-model",
        help="HARNESS=model for one compared harness; repeatable.",
    ),
    workspace: Optional[Path] = WORKSPACE,
    scratch: bool = SCRATCH,
    model: Optional[str] = typer.Option(
        None, "--model", help="Model for harnesses without --harness-model."
    ),
    agent_id: Optional[str] = AGENT_ID,
    allow_dirty: bool = ALLOW_DIRTY,
    execution_backend: str = EXECUTION_BACKEND,
    network: str = NETWORK,
    mcp_access: str = MCP_ACCESS,
    allowed_env: Optional[List[str]] = ALLOW_ENV,
    allowed_tools: Optional[List[str]] = ALLOW_TOOL,
    denied_tools: Optional[List[str]] = DENY_TOOL,
    allow_subagents: bool = ALLOW_SUBAGENTS,
    memory_id: Optional[str] = MEMORY_ID,
    user_id: Optional[str] = USER_ID,
    thread_id: Optional[str] = THREAD_ID,
    verify: Optional[str] = typer.Option(
        None, "--verify", help="Host command that checks each result."
    ),
    max_cost_usd: Optional[float] = MAX_COST,
    max_input_tokens: Optional[int] = MAX_INPUT,
    max_output_tokens: Optional[int] = MAX_OUTPUT,
    max_steps: int = MAX_STEPS,
    timeout: int = TIMEOUT,
    raw_json: bool = JSON_OUT,
):
    """Run one read-only task on several harnesses at once and wait.

    Example: memorizz harness compare "Where can totals lose precision?"
    --harness codex --harness claude-code --harness-model claude-code=claude-sonnet-5-5
    """
    models = {
        _canonical(key): value
        for key, value in _pairs(harness_models, "--harness-model").items()
    }
    unknown = set(models) - {_canonical(name) for name in harnesses}
    if unknown:
        raise typer.BadParameter(
            f"{', '.join(sorted(unknown))} is not a compared --harness",
            param_hint="--harness-model",
        )
    with _session() as (service, _warnings):
        base = _task_options(
            task,
            workspace,
            scratch,
            service,
            model=model,
            agent_id=_saved_agent(service, agent_id, harnesses),
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            allow_dirty_workspace=allow_dirty,
            network=network,
            mcp_access=mcp_access,
            allowed_env=allowed_env,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            allow_subagents=allow_subagents,
            verification_command=verify,
            max_cost_usd=max_cost_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout,
            max_steps=max_steps,
            execution_backend=execution_backend,
        )
        started = service.start_compare(base, harnesses, models=models or None)
        typer.echo(f"Comparison {started['orchestration_id']} started.", err=True)
        _finish_workflow(
            service, _follow(service, started["orchestration_id"]), raw_json
        )


@harness_app.command("workflows")
def list_workflows(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        help="queued, running, succeeded, failed, canceled or interrupted.",
    ),
    limit: int = typer.Option(50, "--limit", min=1, max=500),
    raw_json: bool = JSON_OUT,
):
    """List recent plans and comparisons."""
    with _session(require_memory=False) as (service, _warnings):
        rows = service.list_orchestrations(limit=limit, status=status)
        _print({"ok": True, "workflows": rows, "count": len(rows)}, raw_json)


@harness_app.command("show-workflow")
def show_workflow(
    workflow_id: str = typer.Argument(...),
    raw_json: bool = JSON_OUT,
):
    """Show one plan or comparison with its runs."""
    with _session(require_memory=False) as (service, _warnings):
        value = service.get_orchestration(workflow_id)
        if value is None:
            raise KeyError(f"Harness workflow not found: {workflow_id}")
        runs = [
            service.get_run(step["run_id"])
            for step in value.get("steps") or []
            if step.get("run_id")
        ]
        _print({"ok": True, "workflow": value, "runs": runs}, raw_json)


@harness_app.command("cancel-workflow")
def cancel_workflow(
    workflow_id: str = typer.Argument(...),
    raw_json: bool = JSON_OUT,
):
    """Cancel a plan or comparison: stop its active runs, start no more."""
    with _session(require_memory=False) as (service, _warnings):
        _print(service.cancel_orchestration(workflow_id), raw_json)


@harness_app.command("rerun-workflow")
def rerun_workflow(
    workflow_id: str = typer.Argument(...),
    agent_id: Optional[str] = AGENT_ID,
    raw_json: bool = JSON_OUT,
):
    """Run a finished plan or comparison again with the same settings.

    A fresh scratch folder if it had one. Waits for it to finish.
    """
    with _session() as (service, _warnings):
        record = service.get_orchestration(workflow_id)
        if record is None:
            raise KeyError(f"Harness workflow not found: {workflow_id}")
        involved = [
            step.get("harness")
            for step in record.get("steps") or []
            if not step.get("agent_id")
        ]
        if not (record.get("task") or {}).get("agent_id"):
            agent_id = _saved_agent(service, agent_id, involved)
        started = service.rerun_orchestration(workflow_id, agent_id=agent_id)
        typer.echo(f"Workflow {started['orchestration_id']} started.", err=True)
        _finish_workflow(
            service, _follow(service, started["orchestration_id"]), raw_json
        )


@harness_app.command("delete-workflow")
def delete_workflow(
    workflow_id: str = typer.Argument(...),
    keep_scratch: bool = KEEP_SCRATCH,
    raw_json: bool = JSON_OUT,
):
    """Delete a finished plan or comparison with all its runs.

    Files in your own folders and answers saved to memory stay.
    """
    with _session(require_memory=False) as (service, _warnings):
        _print(
            service.delete_orchestration(workflow_id, remove_scratch=not keep_scratch),
            raw_json,
        )


@harness_app.command("show")
def show_run(
    run_id: str = typer.Argument(...),
    include_events: bool = typer.Option(
        False, "--events", help="Include the run's events."
    ),
    raw_json: bool = JSON_OUT,
):
    """Show one run: its task, status, result, cost and approval."""
    with _session(require_memory=False) as (service, _warnings):
        run_id = _run_id(service, run_id)
        run = service.get_run(run_id)
        if include_events:
            run["events"] = service.events(run_id)
        _print({"ok": True, "run": run}, raw_json)


@harness_app.command("runs")
def list_runs(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        help="pending_approval, queued, running, succeeded, failed, canceled, interrupted, budget_exceeded or verification_failed.",
    ),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
    raw_json: bool = JSON_OUT,
):
    """List recent runs, newest first, with the model each one ran on."""
    from ..metaharness.catalog import fill_reported_models

    with _session(require_memory=False) as (service, _warnings):
        rows = service.list_runs(limit=limit, status=status)
        reported = [
            {"run_id": row["run_id"], "model": (row.get("task") or {}).get("model")}
            for row in rows
        ]
        fill_reported_models(service, reported)
        for row, found in zip(rows, reported):
            row["model"] = found.get("model")
        _print({"ok": True, "runs": rows, "count": len(rows)}, raw_json)


@harness_app.command("events")
def events(
    run_id: str = typer.Argument(...),
    after: int = typer.Option(
        0, "--after", min=0, help="Only events after this sequence number."
    ),
    limit: int = typer.Option(1000, "--limit", min=1, max=10_000),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Keep printing events (one JSON line each) until the run ends.",
    ),
    raw_json: bool = JSON_OUT,
):
    """Print a run's events: messages, commands, tool calls, usage, errors."""
    with _session(require_memory=False) as (service, _warnings):
        run_id = _run_id(service, run_id)
        if follow:
            try:
                for event in service.stream(run_id, after=after, poll_seconds=0.5):
                    typer.echo(json.dumps(event, sort_keys=True, default=str))
            except KeyboardInterrupt:
                pass
            return
        values = service.events(run_id, after=after, limit=limit)
        _print({"ok": True, "events": values, "count": len(values)}, raw_json)


@harness_app.command("delete")
def delete_runs(
    run_ids: List[str] = typer.Argument(..., help="Run IDs to delete."),
    keep_scratch: bool = KEEP_SCRATCH,
    raw_json: bool = JSON_OUT,
):
    """Delete finished runs, with their traces and their delegates' runs.

    Runs still working and workflow steps are kept, with the reason; files in
    your own folders and answers saved to memory stay.
    """
    with _session(require_memory=False) as (service, _warnings):
        result = service.delete_runs(run_ids, remove_scratch=not keep_scratch)
        _print(result, raw_json)
        for item in result.get("kept") or []:
            why = {
                "active": "still working; cancel it first",
                "workflow": "a workflow step; delete the workflow "
                f"({str(item.get('orchestration_id') or '')[:8]}) instead",
                "not_found": "not found",
            }.get(item.get("reason"), item.get("reason"))
            typer.echo(f"Kept {item['run_id'][:8]}: {why}.", err=True)
        if not result.get("deleted"):
            raise typer.Exit(1)


@harness_app.command("conversation")
def conversation(
    conversation: str = typer.Argument(
        ..., help="A conversation ID (hxc-…) or any of its runs."
    ),
    raw_json: bool = JSON_OUT,
):
    """Show a harness conversation: each turn's request, status and answer."""
    with _session(require_memory=False) as (service, _warnings):
        conversation_id, runs = _conversation(service, conversation)
        turns = [
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "harness": run.get("harness"),
                "request": (run.get("task") or {}).get("task"),
                "answer": (run.get("result") or {}).get("final_response"),
            }
            for run in runs
        ]
        _print(
            {"ok": True, "conversation_id": conversation_id, "turns": turns}, raw_json
        )


def _conversation(service: Any, value: str) -> tuple:
    conversation_id = value
    if not value.startswith("hxc-"):
        run = service.get_run(value)
        if run is None:
            raise KeyError(f"Harness run or conversation not found: {value}")
        conversation_id = service.conversation_id_for(run)
    runs = service.conversation(conversation_id)
    if not runs:
        raise KeyError(f"Harness conversation not found: {value}")
    return conversation_id, runs


@harness_app.command("continue")
def continue_conversation(
    conversation: str = typer.Argument(
        ..., help="A conversation ID (hxc-…) or any of its runs."
    ),
    message: str = typer.Argument(..., help="The next message."),
    harness: Optional[str] = typer.Option(
        None, "--harness", help="Switch harness (default: the one that answered last)."
    ),
    model: Optional[str] = typer.Option(None, "--model"),
    agent_id: Optional[str] = AGENT_ID,
    workspace: Optional[Path] = typer.Option(
        None, "--workspace", "-C", help="Default: the conversation's folder."
    ),
    write: Optional[bool] = typer.Option(
        None, "--write/--read-only", help="Default: as the last turn."
    ),
    network: Optional[str] = typer.Option(
        None, "--network", help="Default: as the last turn."
    ),
    mcp_access: Optional[str] = typer.Option(
        None, "--mcp-access", help="Default: as the last turn."
    ),
    allow_subagents: Optional[bool] = typer.Option(
        None, "--allow-subagents/--no-subagents", help="Default: as the last turn."
    ),
    execution_backend: Optional[str] = typer.Option(
        None, "--execution-backend", help="Default: as the last turn."
    ),
    raw_json: bool = JSON_OUT,
):
    """Send the next message in a harness conversation and wait for the answer.

    The setup carries over from the latest turn; earlier turns go with it as
    context.
    """
    from ..metaharness import catalog

    with _session() as (service, warnings):
        conversation_id, runs = _conversation(service, conversation)
        setup = catalog.chat_setup(runs[-1])
        chosen = _canonical(harness) if harness else setup["harness"]
        task = _task_options(
            message,
            workspace or Path(setup["workspace"]),
            False,
            service,
            harness=chosen,
            model=model if model is not None else (setup["model"] or None),
            agent_id=_saved_agent(
                service, agent_id or setup["agent_id"] or None, [chosen]
            ),
            memory_id=setup["memory_id"] or None,
            user_id=setup["user_id"] or None,
            write=setup["write"] if write is None else write,
            allow_dirty_workspace=setup["allow_dirty_workspace"],
            network=network or setup["network"],
            mcp_access=mcp_access or setup["mcp_access"],
            allow_subagents=setup["allow_subagents"]
            if allow_subagents is None
            else allow_subagents,
            timeout_seconds=setup["timeout_seconds"],
            max_steps=setup["max_steps"],
            max_cost_usd=setup["max_cost_usd"],
            verification_command=setup["verification_command"] or None,
            execution_backend=execution_backend or setup["execution_backend"],
        )
        started = service.continue_conversation(conversation_id, task)
        typer.echo(f"Turn {started.run_id[:8]} started.", err=True)
        run = _wait_for_run(service, started.run_id)
        _print(
            {
                "ok": run.get("status") == "succeeded",
                "conversation_id": conversation_id,
                "run": run,
                "warnings": warnings,
            },
            raw_json,
        )
        if run.get("status") not in {"succeeded", "pending_approval"}:
            raise typer.Exit(1)


@harness_app.command("delete-conversation")
def delete_conversation(
    conversation: str = typer.Argument(
        ..., help="A conversation ID (hxc-…) or any of its runs."
    ),
    keep_scratch: bool = KEEP_SCRATCH,
    raw_json: bool = JSON_OUT,
):
    """Delete every finished turn of a harness conversation."""
    with _session(require_memory=False) as (service, _warnings):
        conversation_id, _runs = _conversation(service, conversation)
        result = service.delete_conversation(
            conversation_id, remove_scratch=not keep_scratch
        )
        _print(result, raw_json)
        if not result.get("deleted"):
            raise typer.Exit(1)


@harness_app.command("approvals")
def approvals(
    status: Optional[str] = typer.Option(
        None, "--status", help="pending, approved, rejected, expired or consumed."
    ),
    owner_id: Optional[str] = typer.Option(None, "--owner-id"),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
    raw_json: bool = JSON_OUT,
):
    """List exact run-envelope proposals awaiting a host decision."""
    with _session(require_memory=False) as (service, _warnings):
        values = service.list_approvals(status=status, owner_id=owner_id, limit=limit)
        _print({"ok": True, "approvals": values, "count": len(values)}, raw_json)


@harness_app.command("cancel")
def cancel(run_id: str = typer.Argument(...), raw_json: bool = JSON_OUT):
    """Cancel a run. A memagent run also stops the runs its delegates started."""
    with _session(require_memory=False) as (service, _warnings):
        _print(service.cancel(run_id), raw_json)


@harness_app.command("approve")
def approve(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver", help="Who approves (recorded)."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Run the approved task now and wait (default), or approve only.",
    ),
    raw_json: bool = JSON_OUT,
):
    """Approve a proposed run envelope, then run it."""
    with _session(require_memory=resume) as (service, _warnings):
        proposal = service.approve(proposal_id, approver_id=approver_id, reason=reason)
        result = service.resume_approval(proposal_id).to_dict() if resume else None
        _print({"ok": True, "proposal": proposal, "result": result}, raw_json)


@harness_app.command("reject")
def reject(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver", help="Who rejects (recorded)."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = JSON_OUT,
):
    """Reject a proposed run envelope; the run never starts."""
    with _session(require_memory=False) as (service, _warnings):
        proposal = service.reject(proposal_id, approver_id=approver_id, reason=reason)
        _print({"ok": True, "proposal": proposal}, raw_json)


@harness_app.command("resume")
def resume(
    proposal_id: str = typer.Argument(...),
    raw_json: bool = JSON_OUT,
):
    """Consume an already-approved envelope and run its exact checkpoint."""
    with _session() as (service, _warnings):
        result = service.resume_approval(proposal_id)
        _print(result.to_dict(), raw_json)
        if not result.ok:
            raise typer.Exit(1)


@harness_app.command("retry")
def retry(
    run_id: str = typer.Argument(...),
    raw_json: bool = JSON_OUT,
):
    """Retry a terminal run as a new run with the same bounded task."""
    with _session() as (service, _warnings):
        result = service.retry(run_id)
        _print(result.to_dict(), raw_json)
        if result.status.value not in {"succeeded", "pending_approval"}:
            raise typer.Exit(1)


@delegate_app.command("options")
def delegate_options(raw_json: bool = JSON_OUT):
    """Harnesses a delegate can run on, whether each is ready, and its models."""
    from ..metaharness import catalog

    with _session(require_memory=False) as (service, _warnings):
        harnesses = [
            item
            for item in service.list_harnesses()
            if item.get("name") in catalog.HARNESS_LABELS
        ]
        _print(
            {
                "ok": True,
                "harnesses": [
                    {
                        "name": item["name"],
                        "label": catalog.HARNESS_LABELS[item["name"]],
                        "ready": bool(item.get("ready")),
                    }
                    for item in harnesses
                ],
                "models": catalog.model_choices(service, harnesses),
            },
            raw_json,
        )


@delegate_app.command("create")
def delegate_create(
    harness: str = typer.Option(
        ..., "--harness", help="codex, claude-code, openhands, deepseek, pi or hermes."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Model the harness runs (default: the harness's own)."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Default: '<Harness> delegate'."
    ),
    coordinator: Optional[str] = typer.Option(
        None,
        "--coordinator",
        help="Agent whose model the delegate loads with (and joins, with --attach).",
    ),
    attach: bool = typer.Option(
        False, "--attach", help="Add the new delegate to --coordinator's delegates."
    ),
    raw_json: bool = JSON_OUT,
):
    """Create an agent that runs each task it is given on a harness.

    Ready to be a coordinator's delegate (--attach adds it to one).
    """
    from ..metaharness import catalog
    from .agent_commands import _llm_config

    if attach and not coordinator:
        raise typer.BadParameter("--attach needs --coordinator")
    with _session() as (service, _warnings):
        provider = service.memory_provider
        agent = catalog.create_harness_delegate(
            provider,
            harness=harness,
            model=model or "",
            name=name or "",
            coordinator_id=coordinator or "",
            default_llm_config=_llm_config(None, None, disabled=False),
        )
        attached = False
        if attach:
            saved = provider.retrieve_memagent(coordinator)
            delegates = list(getattr(saved, "delegates", None) or [])
            if agent["id"] not in delegates:
                provider.store_memagent(
                    saved.model_copy(update={"delegates": [*delegates, agent["id"]]})
                )
            attached = True
        _print(
            {
                "ok": True,
                "agent": agent,
                "attached_to": coordinator if attached else None,
            },
            raw_json,
        )


__all__ = ["harness_app"]
