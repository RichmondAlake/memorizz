"""One way to turn launch options into harness tasks and plan stages.

The UI, CLI and MCP server accept the same options for a run, a staged plan
and a comparison; these helpers keep their meaning identical everywhere.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from .models import HarnessBudget, HarnessPermissions, HarnessTask, VerificationSpec


def _text(value: Any) -> Optional[str]:
    return str(value or "").strip() or None


def _names(values: Optional[Iterable[Any]]) -> List[str]:
    """Distinct non-empty names from a list or a comma-separated string."""
    if isinstance(values, str):
        values = values.split(",")
    names: List[str] = []
    for value in values or []:
        name = str(value or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def harness_task(
    task: str,
    workspace: str,
    *,
    harness: str = "auto",
    model: Optional[str] = None,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    write: bool = False,
    allow_dirty_workspace: bool = False,
    network: str = "none",
    mcp_access: str = "read_only",
    allowed_env: Optional[Iterable[str]] = None,
    allowed_tools: Optional[Iterable[str]] = None,
    denied_tools: Optional[Iterable[str]] = None,
    allowed_roots: Optional[Iterable[str]] = None,
    allow_subagents: bool = False,
    timeout_seconds: float = 900,
    max_steps: int = 80,
    max_cost_usd: Optional[float] = None,
    max_input_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    verification_command: Optional[str] = None,
    output_schema: Optional[Mapping[str, Any]] = None,
    mode: str = "runtime",
    metadata: Optional[Mapping[str, Any]] = None,
) -> HarnessTask:
    """A HarnessTask from the launch options every surface shares."""
    return HarnessTask(
        task=str(task or ""),
        workspace=str(workspace or ""),
        harness=str(harness or "auto"),
        model=_text(model),
        agent_id=_text(agent_id),
        memory_id=_text(memory_id),
        user_id=_text(user_id),
        thread_id=_text(thread_id),
        mode=mode,
        permissions=HarnessPermissions(
            workspace_mode="direct" if write else "read_only",
            allowed_roots=_names(allowed_roots),
            allow_dirty_workspace=bool(allow_dirty_workspace),
            network=str(network or "none"),
            mcp_access=str(mcp_access or "read_only"),
            allowed_env=_names(allowed_env),
            allowed_tools=_names(allowed_tools),
            denied_tools=_names(denied_tools),
            allow_subagents=bool(allow_subagents),
        ),
        budget=HarnessBudget(
            max_wall_time_seconds=timeout_seconds,
            max_steps=max_steps,
            max_cost_usd=max_cost_usd,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        ),
        verification=VerificationSpec(command=_text(verification_command)),
        output_schema=dict(output_schema) if output_schema else None,
        metadata=dict(metadata or {}),
    )


def plan_stages(rows: Iterable[Mapping[str, Any]], goal: str) -> List[Dict[str, Any]]:
    """Plan stages from launch rows. A stage's instruction is its role; the
    goal follows it, so every harness knows what the whole plan is for."""
    rows = list(rows or [])
    if not rows:
        raise ValueError("Add at least one stage to the plan")
    stages = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("Each stage must be an object")
        role = _text(row.get("instruction"))
        verify = _text(row.get("verification_command"))
        stages.append(
            {
                "name": _text(row.get("name")) or f"Stage {index + 1}",
                "harness": _text(row.get("harness")) or "auto",
                "instruction": f"{role}\n\nGoal:\n{goal}" if role else None,
                "workspace_mode": "direct" if row.get("write") else "read_only",
                "verification": {"command": verify} if verify else None,
                "model": _text(row.get("model")),
                "agent_id": _text(row.get("agent_id")),
            }
        )
    return stages


def parse_stage(spec: str) -> Dict[str, Any]:
    """``NAME:HARNESS[:edit]`` from the command line, e.g. ``implement:codex:edit``."""
    parts = [part.strip() for part in str(spec or "").split(":")]
    if len(parts) not in {2, 3} or not all(parts[:2]):
        raise ValueError(
            f"Stage {spec!r} must look like NAME:HARNESS or NAME:HARNESS:edit"
        )
    if len(parts) == 3 and parts[2].lower() not in {"edit", "edits", "write"}:
        raise ValueError(f"Stage {spec!r}: the third part may only be 'edit'")
    return {"name": parts[0], "harness": parts[1], "write": len(parts) == 3}


__all__ = ["harness_task", "parse_stage", "plan_stages"]
