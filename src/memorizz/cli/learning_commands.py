"""Operator commands for the MemoRizz learning control plane."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import typer

from . import config as cfg

learning_app = typer.Typer(
    help="Inspect, compile, and govern agent learning memory.",
    no_args_is_help=True,
)


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


@contextmanager
def _runtime(agent_id: str, backend: Optional[str] = None) -> Iterator[Any]:
    cfg.load_layered_env()
    resolved = str(backend or _env("MEMORIZZ_BACKEND") or "filesystem").lower()
    if resolved == "oracle":
        from ..memory_provider.oracle import OracleProvider

        provider = OracleProvider.from_env(index_policy="lazy")
    elif resolved == "mongodb":
        from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

        uri = _env("MONGODB_URI")
        if not uri:
            raise typer.BadParameter("MONGODB_URI is required for --backend mongodb")
        provider = MongoDBProvider(
            MongoDBConfig(uri=uri, db_name=_env("MONGODB_DB_NAME") or "memorizz")
        )
    elif resolved == "filesystem":
        from ..memory_provider import FileSystemConfig, FileSystemProvider

        provider = FileSystemProvider(FileSystemConfig(root_path=cfg.memory_root()))
    else:
        raise typer.BadParameter("backend must be filesystem, mongodb, or oracle")

    from ..learning import LearningControlPlane

    plane = LearningControlPlane(
        provider,
        agent_id=agent_id,
        config={"enabled": True, "compile_async": False},
    )
    try:
        yield plane
    finally:
        plane.close()
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()


def _print(value: Any, raw_json: bool) -> None:
    print(
        json.dumps(value, indent=None if raw_json else 2, sort_keys=True, default=str)
    )


@learning_app.command("status")
def status(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Show scoped event, artifact, and control-plane state."""
    with _runtime(agent_id, backend) as plane:
        report = plane.report(memory_id=memory_id, user_id=user_id, thread_id=thread_id)
    _print(report, raw_json)


@learning_app.command("events")
def events(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    limit: int = typer.Option(50, "--limit", min=1, max=1000),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List immutable learning events for one scope."""
    with _runtime(agent_id, backend) as plane:
        rows = plane.store.list_events(
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
            limit=limit,
        )
    _print({"count": len(rows), "events": [row.to_dict() for row in rows]}, raw_json)


@learning_app.command("compile")
def compile_memory(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Incrementally compile pending events into retrieval artifacts."""
    with _runtime(agent_id, backend) as plane:
        report = plane.compile(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        ).to_dict()
    _print(report, raw_json)


@learning_app.command("forget-plan")
def forget_plan(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Create a durable dry-run plan; this never deletes memory."""
    with _runtime(agent_id, backend) as plane:
        report = plane.plan_forgetting(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        ).to_dict()
    _print(report, raw_json)


@learning_app.command("forget-apply")
def forget_apply(
    plan_id: str = typer.Argument(...),
    agent_id: str = typer.Option(..., "--agent-id"),
    approved_by: str = typer.Option(..., "--approved-by"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Apply a stored plan as reversible tombstones with operator identity."""
    with _runtime(agent_id, backend) as plane:
        report = plane.get_forgetting_plan(
            plan_id,
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        applied = plane.apply_forgetting(
            report,
            approved_by=approved_by,
            reason=reason,
            scope={
                "memory_id": memory_id,
                "user_id": user_id,
                "thread_id": thread_id,
            },
        ).to_dict()
    _print(applied, raw_json)


__all__ = ["learning_app"]
