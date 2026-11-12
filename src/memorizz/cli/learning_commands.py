"""Operator commands for the MemoRizz learning control plane."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

import typer

from .._env_io import env_text as _env
from . import config as cfg

learning_app = typer.Typer(
    help="Inspect, compile, and govern agent learning memory.",
    no_args_is_help=True,
)


def _open_provider(backend: Optional[str] = None) -> Any:
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
    return provider


def _close_provider(provider: Any) -> None:
    closer = getattr(provider, "close", None)
    if callable(closer):
        closer()


@contextmanager
def _runtime(agent_id: str, backend: Optional[str] = None) -> Iterator[Any]:
    provider = _open_provider(backend)
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
        _close_provider(provider)


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


@learning_app.command("retention-plan")
def retention_plan(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_id: Optional[str] = typer.Option(None, "--thread-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Dry-run plan over primary memories whose retention score decayed (nothing is changed)."""
    with _runtime(agent_id, backend) as plane:
        report = plane.plan_retention(
            memory_id=memory_id,
            user_id=user_id if user_id is not None else ...,
            thread_id=thread_id,
        ).to_dict()
    report["plan_kind"] = "retention"
    _print(report, raw_json)


@learning_app.command("retention-apply")
def retention_apply(
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
    """Suppress (never delete) the memories of a stored retention plan; reversible."""
    with _runtime(agent_id, backend) as plane:
        report = plane.get_forgetting_plan(
            plan_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
        )
        if not plane.is_retention_plan(report):
            raise typer.BadParameter(
                "plan_id refers to an artifact forgetting plan; use forget-apply"
            )
        applied = plane.apply_retention(
            report,
            approved_by=approved_by,
            reason=reason,
            scope={"memory_id": memory_id, "user_id": user_id, "thread_id": thread_id},
        ).to_dict()
    applied["plan_kind"] = "retention"
    _print(applied, raw_json)


@learning_app.command("suppressed")
def suppressed_memories(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List memories currently hidden from retrieval by a retention plan."""
    with _runtime(agent_id, backend) as plane:
        rows = plane.suppressed_memories(
            memory_id=memory_id, user_id=user_id if user_id is not None else ...
        )
    _print({"agent_id": agent_id, "count": len(rows), "suppressed": rows}, raw_json)


@learning_app.command("unsuppress")
def unsuppress_memory(
    record_id: str = typer.Argument(...),
    memory_type: str = typer.Option(..., "--memory-type"),
    agent_id: str = typer.Option(..., "--agent-id"),
    approved_by: str = typer.Option(..., "--approved-by"),
    reason: Optional[str] = typer.Option(None, "--reason"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Restore a suppressed memory to retrieval."""
    with _runtime(agent_id, backend) as plane:
        restored = plane.unsuppress_memory(
            record_id, memory_type, approved_by=approved_by, reason=reason
        )
    _print(
        {"record_id": record_id, "memory_type": memory_type, "restored": restored},
        raw_json,
    )


def _saved_agent(provider: Any, agent_id: str) -> dict:
    record = provider.retrieve_memagent(agent_id)
    if record is None:
        raise typer.BadParameter(f"No saved agent {agent_id}")
    return record if isinstance(record, dict) else record.model_dump()


def _consolidation_model(record: dict, spec: Optional[str]) -> Any:
    """``--model provider:model``, else the agent's background or chat model."""
    from ..learning.consolidation import model_for_agent
    from ..llms.llm_factory import create_llm_provider

    if spec:
        name, _, model = spec.partition(":")
        return create_llm_provider({"provider": name, "model": model or None})
    return model_for_agent(record)


@learning_app.command("consolidate")
def consolidate(
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: str = typer.Option(..., "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    thread_ids: List[str] = typer.Option(
        [], "--thread-id", help="A conversation to mine for missed facts (repeatable)."
    ),
    knowledge_base: bool = typer.Option(
        False, "--knowledge-base", help="Also hide duplicate knowledge-base chunks."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="provider:model (default: the agent's background model)."
    ),
    no_model: bool = typer.Option(
        False, "--no-model", help="Only the duplicate checks that need no model."
    ),
    apply: bool = typer.Option(False, "--apply"),
    approved_by: Optional[str] = typer.Option(None, "--approved-by"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Plan (and with --apply, make) a memory consolidation; every change is reversible."""
    from ..learning.consolidation import ConsolidationConfig, MemoryConsolidator

    if apply and not (approved_by or "").strip():
        raise typer.BadParameter("--apply needs --approved-by")
    provider = _open_provider(backend)
    try:
        record = _saved_agent(provider, agent_id)
        consolidator = MemoryConsolidator(
            provider,
            agent_id=agent_id,
            model=None if no_model else _consolidation_model(record, model),
            knowledge_base_ids=record.get("knowledge_base_ids") or (),
            config=ConsolidationConfig.from_value(
                record.get("memory_consolidation") or None
            ),
            decision_model=record.get("decision_model") or None,
        )
        plan = consolidator.plan(
            memory_id=memory_id,
            user_id=user_id,
            thread_ids=thread_ids,
            knowledge_base=knowledge_base,
            # An agent that keeps entity facts per thread is left alone there.
            entities=(record.get("retrieval_policy") or {}).get(
                "entity_scope", "memory"
            )
            == "memory",
        )
        report: dict = {"plan": plan.to_dict()}
        if apply and plan.ops:
            report["result"] = consolidator.apply(
                plan, approved_by=approved_by
            ).to_dict()
    finally:
        _close_provider(provider)
    _print(report, raw_json)


@learning_app.command("consolidation-restore")
def consolidation_restore(
    plan_id: str = typer.Argument(...),
    agent_id: str = typer.Option(..., "--agent-id"),
    memory_id: str = typer.Option(..., "--memory-id"),
    approved_by: str = typer.Option(..., "--approved-by"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Undo an applied consolidation plan."""
    from ..learning.consolidation import MemoryConsolidator

    provider = _open_provider(backend)
    try:
        counts = MemoryConsolidator(provider, agent_id=agent_id).restore(
            plan_id, memory_id=memory_id, user_id=user_id, approved_by=approved_by
        )
    finally:
        _close_provider(provider)
    _print({"plan_id": plan_id, "restored": counts}, raw_json)


__all__ = ["learning_app"]


@contextmanager
def _saved_memagent(agent_id: str, backend: Optional[str] = None) -> Iterator[Any]:
    """The saved agent, loaded with its settings, on the chosen backend."""
    provider = _open_provider(backend)
    from ..memagent import MemAgent

    try:
        agent = MemAgent.load(agent_id, memory_provider=provider)
    except Exception as exc:
        _close_provider(provider)
        raise typer.BadParameter(f"Could not load agent {agent_id}: {exc}")
    try:
        yield agent
    finally:
        _close_provider(provider)


def _refine_lines(report: dict) -> List[str]:
    lines = [
        f"Refine cycle {report['run_id']}: {report['status']}"
        + (f" ({report['reason']})" if report.get("reason") else ""),
        f"  started by {report['trigger']} at {report['started_at']}",
    ]
    for job in report.get("jobs") or []:
        changes = ", ".join(
            f"{count} {kind.replace('_', ' ')}"
            for kind, count in sorted((job.get("changes") or {}).items())
        )
        tokens = (
            f", {'about ' if job.get('tokens_estimated') else ''}{job['tokens']} tokens"
            if job.get("tokens")
            else ""
        )
        lines.append(
            f"  {job['label']:<21} {job['status']:<8} {changes or 'no changes'}"
            f"{', ' + str(job['waiting']) + ' waiting' if job.get('waiting') else ''}"
            f"{tokens}"
        )
        for note in (job.get("notes") or [])[:3]:
            lines.append(f"      {note}")
        for item in job.get("items") or []:
            mark = "+" if item.get("applied") else "·"
            lines.append(f"      {mark} {item.get('text')}")
    waiting = report.get("pending") or []
    if waiting:
        lines.append(f"  {len(waiting)} plan(s) waiting for review:")
        for item in waiting:
            lines.append(
                f"    {item.get('kind')}: {item.get('summary')} [{item.get('plan_id')}]"
            )
    return lines


@learning_app.command("refine")
def refine(
    agent_id: str = typer.Option(..., "--agent-id"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan every job and apply nothing."
    ),
    memory_id: Optional[str] = typer.Option(None, "--memory-id"),
    user_id: Optional[str] = typer.Option(None, "--user-id"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Run one refine cycle: the agent's upkeep jobs once, in a fixed order.

    Entities, stated facts, skill staleness, promotion, skill review, persona
    learning and retention. Each job keeps its own apply rule; with --dry-run
    nothing is applied. The report is stored with the agent.
    """
    with _saved_memagent(agent_id, backend) as agent:
        report = agent.refine(dry_run=dry_run, memory_id=memory_id, user_id=user_id)
    if raw_json:
        _print(report, True)
    else:
        print("\n".join(_refine_lines(report)))


@learning_app.command("refine-runs")
def refine_runs(
    agent_id: str = typer.Option(..., "--agent-id"),
    limit: int = typer.Option(10, "--limit", min=1, max=200),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """List stored refine cycles, newest first."""
    with _saved_memagent(agent_id, backend) as agent:
        runs = agent.refine_runs(limit=limit)
    if raw_json:
        _print(runs, True)
        return
    if not runs:
        print("No refine cycle has run for this agent yet.")
    for run in runs:
        print(
            f"{run['started_at'][:19]}  {run['run_id']}  {run['status']:<8} "
            f"{run['trigger']:<9} {run.get('changed', 0)} change(s), "
            f"{run.get('waiting', 0)} waiting, {run.get('tokens', 0)} tokens"
        )


@learning_app.command("holdout")
def holdout(
    agent_id: str = typer.Option(..., "--agent-id"),
    rate: Optional[float] = typer.Option(
        None,
        "--rate",
        min=0.0,
        max=1.0,
        help="Share of matching threads that run without an active skill. "
        "0 turns the holdout off.",
    ),
    min_runs: Optional[int] = typer.Option(
        None, "--min-runs", min=1, help="Runs on each side before a skill is flagged."
    ),
    flag_z: Optional[float] = typer.Option(
        None,
        "--flag-z",
        min=0.0,
        help="Standard errors below the 'without' figure that raise the flag. "
        "0 flags any lower figure.",
    ),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Show or change an agent's skill holdout.

    The holdout runs a share of matching threads without an active skill, so
    each skill has a with-and-without figure. With no option it prints the
    settings in force.
    """
    changing = any(value is not None for value in (rate, min_runs, flag_z))
    with _saved_memagent(agent_id, backend) as agent:
        try:
            value = agent.set_holdout(
                rate, min_runs=min_runs, flag_z=flag_z, save=changing
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
    if raw_json:
        _print({**value, "updated": changing}, True)
        return
    if changing:
        print("Holdout saved.")
    if value["holdout_rate"] > 0:
        print(f"{value['holdout_rate']:.0%} of matching threads run without the skill.")
    else:
        print("The holdout is off.")
    print(
        f"A skill is flagged after {value['holdout_min_runs']} runs on each side, "
        f"when its side is lower by more than {value['holdout_flag_z']:g} "
        "standard errors."
    )


@learning_app.command("guardrails")
def guardrails(
    agent_id: str = typer.Option(..., "--agent-id"),
    set_: Optional[List[str]] = typer.Option(
        None,
        "--set",
        help="A guardrail; repeat for each one. Replaces the whole list.",
    ),
    clear: bool = typer.Option(False, "--clear", help="Remove every guardrail."),
    reason: str = typer.Option("Set from the command line", "--reason"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    raw_json: bool = typer.Option(False, "--json"),
):
    """Show or replace a persona's guardrails: rules learning cannot write.

    At most 12, each at most 240 characters. Only an operator (this command,
    the UI) or host code can change them.
    """
    with _saved_memagent(agent_id, backend) as agent:
        manager = getattr(agent, "persona_manager", None)
        persona = manager.current_persona if manager else None
        if persona is None:
            raise typer.BadParameter("This agent has no persona.")
        if set_ or clear:
            try:
                result = agent.set_persona_guardrails(
                    [] if clear else list(set_ or []), reason, source_type="manual"
                )
            except (ValueError, PermissionError) as exc:
                raise typer.BadParameter(str(exc))
            persona = agent.persona_manager.current_persona
        else:
            result = {"updated": False}
        value = {
            "guardrails": list(persona.guardrails),
            "version": persona.version,
            "updated": bool(result.get("updated")),
        }
    if raw_json:
        _print(value, True)
        return
    if value["updated"]:
        print(f"Guardrails saved (persona version {value['version']}).")
    if not value["guardrails"]:
        print("No guardrails.")
    for number, rule in enumerate(value["guardrails"], 1):
        print(f"{number}. {rule}")
