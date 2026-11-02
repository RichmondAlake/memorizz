"""``/background-model`` and ``/consolidate``: memory upkeep in the CLI."""

from __future__ import annotations

from typing import Any, Dict

from rich.markup import escape

from . import agent_factory
from .commands import _con

LLM_PROVIDERS = ("openai", "anthropic", "ollama", "azure", "huggingface", "mlx")

_OP_LABELS = {
    "merge_entities": "merge",
    "suppress_duplicate": "hide duplicate",
    "retire_attribute": "retire",
    "add_fact": "add",
}


def _label(config: Dict[str, Any]) -> str:
    model = config.get("model") or config.get("deployment_name") or "?"
    return f"{config.get('provider') or '?'}/{model}"


def llm_config_for(session, spec: str) -> Dict[str, Any]:
    """``provider:model``, a provider alone, or a model of the chat provider."""
    head, _, rest = spec.partition(":")
    if head.lower() in LLM_PROVIDERS:
        config = agent_factory.config_for_provider(head.lower())
        if rest.strip():
            key = "deployment_name" if head.lower() == "azure" else "model"
            config[key] = rest.strip()
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
    config = llm_config_for(session, spec)
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


_LEARNING_USAGE = (
    "Usage: /persona-learning on [review] · off · now · approve · dismiss · "
    "undo [version]"
)


def _show_proposal(console, proposal: Dict[str, Any]) -> None:
    console.print(
        f"  Add to [bold]{escape(str(proposal.get('field')))}[/bold]: "
        f"{escape(str(proposal.get('addition')))}"
    )
    for item in proposal.get("evidence") or []:
        console.print(f"    [dim]“{escape(str(item.get('quote')))}”[/dim]")


def _ask_about_proposal(session, proposal: Dict[str, Any]) -> None:
    console = _con(session)
    agent = session.agent
    _show_proposal(console, proposal)
    if _confirm("Add this to the persona?"):
        result = agent.apply_persona_proposal(_approver(session))
        console.print(
            f"[green]Persona is now version {result.get('version')}.[/green] "
            "[dim]Undo with /persona-learning undo[/dim]"
        )
    elif _confirm("Turn it down so it isn't proposed again?"):
        agent.dismiss_persona_proposal(_approver(session))
        console.print("[dim]Turned down.[/dim]")
    else:
        console.print("[dim]Left waiting.[/dim]")


def _show_persona_learning(session) -> None:
    console = _con(session)
    agent = session.agent
    config = agent.persona_learning
    if config.enabled:
        how = "applied automatically" if config.auto_apply else "waits for review"
        console.print(
            f"Persona learning: [cyan]on[/cyan], every {config.every_n_turns} turns "
            f"or {config.every_hours:g} hours, {how}"
        )
    else:
        console.print("Persona learning: [cyan]off[/cyan]")
    status = agent.persona_learning_status()
    review = status.get("last_review")
    if review:
        console.print(f"Last review: {escape(str(review.get('reason')))}")
    pending = status.get("pending")
    if pending:
        console.print("Waiting for review:")
        _show_proposal(console, pending)
    changes = [item for item in status.get("changes") or [] if not item["undone"]]
    if changes:
        console.print("Changes it made:")
        for change in changes[:5]:
            console.print(
                f"  v{change.get('version')} {escape(str(change.get('addition')))}"
            )
    console.print(
        f"[dim]The name and role never change this way. {escape(_LEARNING_USAGE)}[/dim]"
    )


def cmd_persona_learning(session, args: str) -> None:
    """Let the persona grow from conversations, and review what that did."""
    from .commands import save_agent_fields

    console = _con(session)
    agent = session.agent
    parts = args.split()
    action = parts[0].lower() if parts else ""
    if not action:
        _show_persona_learning(session)
        return
    if action in {"on", "off"}:
        manager = getattr(agent, "persona_manager", None)
        if action == "on" and not getattr(manager, "configuration_persona", None):
            console.print(
                "[yellow]This agent has no persona to grow.[/yellow] "
                "Set one first with /persona <name>."
            )
            return
        review = [part.lower() for part in parts[1:]] == ["review"]
        agent.with_persona_learning(action == "on", auto_apply=not review)
        save_agent_fields(
            session,
            persona_learning=(
                agent.persona_learning.to_dict() if action == "on" else None
            ),
        )
        _show_persona_learning(session)
        return
    if action == "now":
        if not session.memory_id:
            console.print("[dim]Nothing to review yet: no memory in use.[/dim]")
            return
        console.print(
            f"[dim]Reading recent conversations on "
            f"{escape(_label(agent.background_llm_config or session.llm_config or {}))} …[/dim]"
        )
        review = agent.review_persona(
            session.memory_id, user_id=session.user_id, force=True
        )
        for warning in review.warnings:
            console.print(f"[yellow]{escape(warning)}[/yellow]")
        if review.proposal is None:
            console.print(escape(review.reason))
            return
        _ask_about_proposal(session, review.proposal.to_dict())
        return
    try:
        if action == "approve":
            result = agent.apply_persona_proposal(_approver(session))
            console.print(
                f"[green]Persona is now version {result.get('version')}.[/green]"
            )
        elif action == "dismiss":
            agent.dismiss_persona_proposal(_approver(session))
            console.print("[dim]Turned down.[/dim]")
        elif action == "undo":
            version = int(parts[1]) if len(parts) > 1 else None
            result = agent.undo_persona_change(_approver(session), version)
            console.print(
                f"[green]Undone.[/green] Persona is now version {result.get('version')}."
            )
        else:
            console.print(escape(_LEARNING_USAGE))
    except ValueError as exc:
        console.print(f"[yellow]{escape(str(exc))}[/yellow]")


__all__ = ["cmd_background_model", "cmd_consolidate", "cmd_persona_learning"]
