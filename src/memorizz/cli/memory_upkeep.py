"""``/background-model`` and ``/consolidate``: memory upkeep in the CLI."""

from __future__ import annotations

from typing import Any, Dict

from rich.markup import escape

from . import agent_factory
from .commands import _con

_PROVIDERS = ("openai", "anthropic", "ollama", "azure", "huggingface", "mlx")

_OP_LABELS = {
    "merge_entities": "merge",
    "suppress_duplicate": "hide duplicate",
    "retire_attribute": "retire",
    "add_fact": "add",
}


def _label(config: Dict[str, Any]) -> str:
    model = config.get("model") or config.get("deployment_name") or "?"
    return f"{config.get('provider') or '?'}/{model}"


def _background_config(session, spec: str) -> Dict[str, Any]:
    """``provider:model``, a provider alone, or a model of the chat provider."""
    head, _, rest = spec.partition(":")
    if head.lower() in _PROVIDERS:
        config = agent_factory.config_for_provider(head.lower())
        if rest.strip():
            config["model"] = rest.strip()
        return config
    config = dict(session.llm_config or {})
    config["model"] = spec
    return config


def cmd_background_model(session, args: str) -> None:
    """Show or set the model memory upkeep runs on."""
    from .commands import save_agent_fields

    console = _con(session)
    agent = session.agent
    spec = args.strip()
    if not spec:
        own = agent.__dict__.get("_background_model")
        if own is None:
            console.print(
                "Background model: [cyan]the chat model[/cyan] "
                f"({escape(_label(session.llm_config or {}))})"
            )
        else:
            console.print(
                f"Background model: [cyan]{escape(_label(agent.background_llm_config))}[/cyan]"
            )
        console.print(
            "[dim]Summaries, compaction, reflection, skill distillation and "
            "/consolidate run on it; your turns keep the chat model.\n"
            "Usage: /background-model <model> · <provider>:<model> · off[/dim]"
        )
        return
    if spec.lower() in {"off", "none", "default"}:
        agent.with_background_model(None)
        save_agent_fields(session, background_llm_config=None)
        console.print("[green]Background model →[/green] the chat model")
        return
    config = _background_config(session, spec)
    agent.with_background_model(config)
    error = getattr(agent, "_background_model_error", None)
    if error:
        console.print(
            f"[red]Could not use {escape(_label(config))}:[/red] {escape(error)}"
        )
        agent.with_background_model(None)
        return
    from ..memagent.persistence import _background_llm_config_to_save

    save_agent_fields(
        session, background_llm_config=_background_llm_config_to_save(agent)
    )
    console.print(f"[green]Background model →[/green] {escape(_label(config))}")


def _describe(op) -> str:
    details = op.details
    if op.kind == "merge_entities":
        count = len(details.get("duplicate_entity_ids") or []) + 1
        return f"{count} records of one entity into one"
    if op.kind == "suppress_duplicate":
        return (
            f"{op.memory_type} {op.target_id} (same as {details.get('duplicate_of')})"
        )
    if op.kind == "retire_attribute":
        return (
            f"{details.get('entity')}.{details.get('attribute')} = "
            f"{details.get('value')}: {op.reason}"
        )
    if op.kind == "add_fact":
        replaced = (
            f" (was {details['replaces']})"
            if details.get("replaces") is not None
            else ""
        )
        return (
            f"{details.get('entity')}.{details.get('attribute')} = "
            f"{details.get('value')}{replaced}, from “{details.get('evidence')}”"
        )
    return op.reason


def _confirm(question: str) -> bool:
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _approver(session) -> str:
    return f"cli:{session.user_id}" if session.user_id else "cli"


def cmd_consolidate(session, args: str) -> None:
    """Plan a consolidation of this memory, show it, apply it on a yes."""
    console = _con(session)
    agent = session.agent
    parts = args.split()
    if parts and parts[0].lower() == "undo":
        if len(parts) < 2:
            console.print("Usage: /consolidate undo <plan id>")
            return
        counts = agent.restore_memory_consolidation(
            parts[1],
            approved_by=_approver(session),
            memory_id=session.memory_id,
            user_id=session.user_id,
        )
        console.print(f"[green]Undone[/green] {escape(parts[1])}: {counts}")
        return
    if not session.memory_id:
        console.print("[dim]Nothing to consolidate yet: no memory in use.[/dim]")
        return
    threads = {
        str(item)
        for item in (
            agent._consolidation_state.get((session.memory_id, session.user_id)) or {}
        ).get("threads")
        or ()
    }
    if session.thread_id:
        threads.add(str(session.thread_id))
    console.print(
        f"[dim]Reviewing memory {escape(session.memory_id)} on "
        f"{escape(_label(agent.background_llm_config or session.llm_config or {}))} …[/dim]"
    )
    plan = agent.plan_memory_consolidation(
        session.memory_id,
        user_id=session.user_id,
        thread_ids=sorted(threads),
        knowledge_base="kb" in [part.lower() for part in parts],
    )
    for warning in plan.warnings:
        console.print(f"[yellow]{escape(warning)}[/yellow]")
    if not plan.ops:
        console.print("[green]Memory is tidy:[/green] nothing to change.")
        return
    for op in plan.ops:
        console.print(
            f"  • [bold]{_OP_LABELS.get(op.kind, op.kind)}[/bold] {escape(_describe(op))}"
        )
    if not _confirm(f"Apply these {len(plan.ops)} changes?"):
        console.print(
            f"[dim]Nothing changed. Plan {plan.plan_id} stays available this session.[/dim]"
        )
        return
    result = agent.apply_memory_consolidation(plan, approved_by=_approver(session))
    for error in result.errors:
        console.print(f"[red]{escape(error)}[/red]")
    applied = sum(result.applied.values())
    console.print(
        f"[green]Applied {applied} changes.[/green] "
        f"[dim]Undo with /consolidate undo {plan.plan_id}[/dim]"
    )


__all__ = ["cmd_background_model", "cmd_consolidate"]
