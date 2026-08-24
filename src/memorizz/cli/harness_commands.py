"""CLI parity for the MemoRizz meta-harness control plane."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import typer

from . import config as cli_config

harness_app = typer.Typer(
    help="Run and inspect Codex, Claude Code, OpenHands, and native harnesses.",
    no_args_is_help=True,
)


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


@harness_app.command("list")
def list_harnesses(
    raw_json: bool = typer.Option(False, "--json", help="Emit compact JSON."),
):
    service, provider, warnings = _service(require_memory=False)
    try:
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
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("doctor")
def doctor(
    name: Optional[str] = typer.Argument(None, help="Optional harness name."),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, warnings = _service(require_memory=False)
    try:
        values = [service.probe(name)] if name else service.list_harnesses()
        ok = all(bool(item.get("ready", item.get("available"))) for item in values)
        _print({"ok": ok, "harnesses": values, "warnings": warnings}, raw_json)
        if name and not ok:
            raise typer.Exit(1)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("init")
def init_config(
    force: bool = typer.Option(False, "--force"),
):
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
def show_config(raw_json: bool = typer.Option(False, "--json")):
    from ..metaharness.config import harness_config_path, load_harness_config

    _print(
        {
            "ok": True,
            "path": str(harness_config_path()),
            "config": load_harness_config(),
        },
        raw_json,
    )


@harness_app.command("run")
def run_harness(
    task: str = typer.Argument(..., help="Workspace task for the harness."),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-C"),
    harness: str = typer.Option("auto", "--harness"),
    model: Optional[str] = typer.Option(None, "--model"),
    agent_id: Optional[str] = typer.Option(
        None, "--agent-id", help="Saved MemAgent ID; required by --harness native."
    ),
    write: bool = typer.Option(False, "--write/--read-only"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty"),
    execution_backend: str = typer.Option("local", "--execution-backend"),
    network: str = typer.Option("none", "--network"),
    mcp_access: str = typer.Option("read_only", "--mcp-access"),
    allowed_env: Optional[List[str]] = typer.Option(
        None, "--allow-env", help="Environment variable name; repeat as needed."
    ),
    verify: Optional[str] = typer.Option(None, "--verify"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    max_cost_usd: Optional[float] = typer.Option(None, "--max-cost-usd", min=0.0),
    max_input_tokens: Optional[int] = typer.Option(None, "--max-input-tokens", min=1),
    max_output_tokens: Optional[int] = typer.Option(None, "--max-output-tokens", min=1),
    max_steps: int = typer.Option(80, "--max-steps", min=1),
    timeout: int = typer.Option(900, "--timeout", min=1),
    raw_json: bool = typer.Option(False, "--json"),
):
    from ..metaharness import (
        HarnessBudget,
        HarnessPermissions,
        HarnessTask,
        VerificationSpec,
    )

    service, provider, warnings = _service()
    try:
        result = service.run(
            HarnessTask(
                task=task,
                workspace=str(workspace),
                harness=harness,
                model=model,
                agent_id=agent_id,
                memory_id=memory_id,
                user_id=user_id,
                thread_id=thread_id,
                permissions=HarnessPermissions(
                    workspace_mode="direct" if write else "read_only",
                    allow_dirty_workspace=allow_dirty,
                    network=network,
                    mcp_access=mcp_access,
                    allowed_env=list(allowed_env or []),
                ),
                budget=HarnessBudget(
                    max_wall_time_seconds=timeout,
                    max_cost_usd=max_cost_usd,
                    max_input_tokens=max_input_tokens,
                    max_output_tokens=max_output_tokens,
                    max_steps=max_steps,
                ),
                verification=VerificationSpec(command=verify),
                metadata={"execution_backend": execution_backend},
            )
        )
        payload = result.to_dict()
        payload["warnings"] = warnings
        _print(payload, raw_json)
        if result.status.value not in {"succeeded", "pending_approval"}:
            raise typer.Exit(1)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("show")
def show_run(
    run_id: str = typer.Argument(...),
    include_events: bool = typer.Option(False, "--events"),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, _ = _service(require_memory=False)
    try:
        run = service.get_run(run_id)
        if run is None:
            typer.echo(f"Harness run not found: {run_id}", err=True)
            raise typer.Exit(1)
        if include_events:
            run["events"] = service.events(run_id)
        _print({"ok": True, "run": run}, raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("runs")
def list_runs(
    status: Optional[str] = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit", min=1, max=1000),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, _ = _service(require_memory=False)
    try:
        rows = service.list_runs(limit=limit, status=status)
        _print({"ok": True, "runs": rows, "count": len(rows)}, raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("events")
def events(
    run_id: str = typer.Argument(...),
    after: int = typer.Option(0, "--after", min=0),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, _ = _service(require_memory=False)
    try:
        values = service.events(run_id, after=after)
        _print({"ok": True, "events": values, "count": len(values)}, raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("approvals")
def approvals(
    status: Optional[str] = typer.Option(None, "--status"),
    owner_id: Optional[str] = typer.Option(None, "--owner-id"),
    limit: int = typer.Option(100, "--limit", min=1, max=1000),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List exact run-envelope proposals awaiting a host decision."""
    service, provider, _ = _service(require_memory=False)
    try:
        values = service.list_approvals(status=status, owner_id=owner_id, limit=limit)
        _print({"ok": True, "approvals": values, "count": len(values)}, raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("cancel")
def cancel(
    run_id: str = typer.Argument(...), raw_json: bool = typer.Option(False, "--json")
):
    service, provider, _ = _service(require_memory=False)
    try:
        _print(service.cancel(run_id), raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("approve")
def approve(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, _ = _service(require_memory=resume)
    try:
        proposal = service.approve(proposal_id, approver_id=approver_id, reason=reason)
        result = service.resume_approval(proposal_id).to_dict() if resume else None
        _print({"ok": True, "proposal": proposal, "result": result}, raw_json)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("reject")
def reject(
    proposal_id: str = typer.Argument(...),
    approver_id: str = typer.Option(..., "--approver"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    raw_json: bool = typer.Option(False, "--json"),
):
    service, provider, _ = _service(require_memory=False)
    try:
        _print(
            {
                "ok": True,
                "proposal": service.reject(
                    proposal_id, approver_id=approver_id, reason=reason
                ),
            },
            raw_json,
        )
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("resume")
def resume(
    proposal_id: str = typer.Argument(...),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Consume an already-approved envelope and run its exact checkpoint."""
    service, provider, _ = _service()
    try:
        result = service.resume_approval(proposal_id)
        _print(result.to_dict(), raw_json)
        if not result.ok:
            raise typer.Exit(1)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@harness_app.command("retry")
def retry(
    run_id: str = typer.Argument(...),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Retry a terminal run as a new run with the same bounded task."""
    service, provider, _ = _service()
    try:
        result = service.retry(run_id)
        _print(result.to_dict(), raw_json)
        if result.status.value not in {"succeeded", "pending_approval"}:
            raise typer.Exit(1)
    finally:
        service.close()
        close = getattr(provider, "close", None)
        if callable(close):
            close()


__all__ = ["harness_app"]
