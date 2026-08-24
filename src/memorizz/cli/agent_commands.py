# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Create and inspect persisted MemAgents from the command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer

from . import config as cfg

agents_app = typer.Typer(
    help="Create and inspect persisted MemAgents.",
    no_args_is_help=True,
)

_LLM_PROVIDERS = {"anthropic", "azure", "huggingface", "mlx", "ollama", "openai"}


def _provider() -> Tuple[Any, List[str]]:
    """Resolve the configured provider, including the filesystem default."""
    from . import agent_factory

    cfg.load_layered_env()
    warnings: List[str] = []
    provider = agent_factory.detect_memory_provider({}, warnings)
    return provider, warnings


def _llm_config(
    provider: Optional[str], model: Optional[str], *, disabled: bool
) -> Optional[Dict[str, Any]]:
    if disabled:
        if provider or model:
            raise typer.BadParameter("--no-llm cannot be combined with LLM options")
        return None

    from . import agent_factory

    normalized = str(provider or "").strip().lower()
    if normalized:
        if normalized not in _LLM_PROVIDERS:
            choices = ", ".join(sorted(_LLM_PROVIDERS))
            raise typer.BadParameter(f"Unsupported LLM provider. Choose: {choices}")
        config = agent_factory.config_for_provider(normalized)
    elif model:
        raise typer.BadParameter("--model requires --llm-provider")
    else:
        try:
            return agent_factory.detect_llm_config()
        except (agent_factory.NoProviderConfigured, agent_factory.NeedsOllamaPull):
            return None

    if model:
        if normalized == "azure":
            config.pop("model", None)
            config["deployment_name"] = str(model).strip()
        else:
            config["model"] = str(model).strip()
    return config


def _agent_summary(agent: Any) -> Dict[str, Any]:
    application_mode = getattr(agent, "application_mode", None)
    if hasattr(application_mode, "value"):
        application_mode = application_mode.value
    cache_manager = getattr(agent, "cache_manager", None)
    semantic_cache = (
        bool(getattr(cache_manager, "enabled", False))
        if cache_manager is not None
        else bool(getattr(agent, "semantic_cache", False))
    )
    return {
        "agent_id": str(getattr(agent, "agent_id", "") or ""),
        "name": getattr(agent, "name", None),
        "instruction": getattr(agent, "instruction", None),
        "application_mode": application_mode,
        "max_steps": getattr(agent, "max_steps", None),
        "semantic_cache": semantic_cache,
        "continual_learning": bool(getattr(agent, "continual_learning", False)),
        "learning_control_plane": bool(
            getattr(
                agent,
                "learning_control_plane_enabled",
                getattr(agent, "learning_control_plane", False),
            )
        ),
        "meta_harness": bool(getattr(agent, "meta_harness", False)),
        "meta_harness_mode": getattr(agent, "meta_harness_mode", None),
        "default_harness": getattr(agent, "default_harness", "auto"),
        "memory_ids": list(getattr(agent, "memory_ids", None) or []),
        "llm_provider": (
            (getattr(agent, "llm_config", None) or {}).get("provider")
            if isinstance(getattr(agent, "llm_config", None), dict)
            else None
        ),
    }


def _print_payload(payload: Dict[str, Any], *, raw_json: bool) -> None:
    if raw_json:
        typer.echo(json.dumps(payload, sort_keys=True))
        return
    from rich.console import Console

    console = Console()
    if payload.get("ok") and payload.get("agent"):
        agent = payload["agent"]
        console.print(
            f"[green]Created MemAgent[/green] {agent['agent_id']} "
            f"[cyan]{agent.get('name') or ''}[/cyan]"
        )
        if not agent.get("llm_provider"):
            console.print(
                "[yellow]No LLM configured.[/yellow] Configure one before running the agent."
            )
        for warning in payload.get("warnings") or []:
            console.print(f"[yellow]{warning}[/yellow]")
        return
    console.print_json(data=payload)


@agents_app.command("create")
def create_agent(
    name: str = typer.Option(..., "--name", "-n", help="Agent display name."),
    instruction: Optional[str] = typer.Option(
        None, "--instruction", "-i", help="System instruction for the agent."
    ),
    application_mode: str = typer.Option(
        "assistant",
        "--application-mode",
        help="assistant, workflow, or deep_research.",
    ),
    max_steps: int = typer.Option(20, "--max-steps", min=1, max=1000),
    memory_id: Optional[List[str]] = typer.Option(
        None, "--memory-id", help="Attach a memory ID; repeat for multiple IDs."
    ),
    semantic_cache: bool = typer.Option(False, "--semantic-cache/--no-semantic-cache"),
    continual_learning: bool = typer.Option(
        False, "--continual-learning/--no-continual-learning"
    ),
    learning_control_plane: bool = typer.Option(
        False, "--learning-control-plane/--no-learning-control-plane"
    ),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    no_llm: bool = typer.Option(
        False, "--no-llm", help="Persist the agent without an LLM configuration."
    ),
    harness_mode: Optional[str] = typer.Option(
        None,
        "--harness-mode",
        help="Enable the meta-harness in delegate or runtime mode.",
    ),
    default_harness: str = typer.Option(
        "auto",
        "--default-harness",
        help="auto, codex, claude-code, or openhands.",
    ),
    harness_workspace: Optional[str] = typer.Option(
        None,
        "--harness-workspace",
        help="Default workspace for runtime-mode harness turns.",
    ),
    set_default: bool = typer.Option(
        False,
        "--set-default/--no-set-default",
        help="Make this the default agent used by `memorizz chat`.",
    ),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Create, validate, and persist a MemAgent."""
    from ..enums import ApplicationModeConfig
    from ..memagent.builders import MemAgentBuilder

    normalized_name = str(name or "").strip()
    if not normalized_name:
        raise typer.BadParameter("--name cannot be empty")
    try:
        mode = ApplicationModeConfig.validate_mode(application_mode).value
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--application-mode") from exc
    resolved_llm = _llm_config(llm_provider, model, disabled=no_llm)
    provider, warnings = _provider()
    agent = None
    try:
        builder = (
            MemAgentBuilder()
            .with_name(normalized_name)
            .with_application_mode(mode)
            .with_max_steps(max_steps)
            .with_memory_provider(provider)
            .with_semantic_cache(semantic_cache)
            .with_continual_learning(continual_learning)
            .with_learning_control_plane(learning_control_plane)
        )
        if instruction and instruction.strip():
            builder.with_instruction(instruction.strip())
        if memory_id:
            builder.with_memory_ids(
                [value.strip() for value in memory_id if value.strip()]
            )
        if resolved_llm:
            builder.with_llm_config(resolved_llm)
        if harness_mode is not None:
            normalized_harness_mode = str(harness_mode).strip().lower()
            if normalized_harness_mode not in {"delegate", "runtime"}:
                raise typer.BadParameter("--harness-mode must be delegate or runtime")
            normalized_default_harness = (
                str(default_harness or "auto").strip().lower().replace("_", "-")
            )
            if normalized_default_harness not in {
                "auto",
                "codex",
                "claude-code",
                "openhands",
                "native",
            }:
                raise typer.BadParameter(
                    "--default-harness must be auto, codex, claude-code, "
                    "openhands, or native"
                )
            harness_config: Dict[str, Any] = {}
            if harness_workspace:
                try:
                    workspace_path = (
                        Path(harness_workspace).expanduser().resolve(strict=True)
                    )
                except OSError as exc:
                    raise typer.BadParameter(
                        "--harness-workspace must be an existing directory"
                    ) from exc
                if not workspace_path.is_dir():
                    raise typer.BadParameter(
                        "--harness-workspace must be an existing directory"
                    )
                harness_config["workspace"] = str(workspace_path)
                harness_config["permissions"] = {"allowed_roots": [str(workspace_path)]}
            builder.with_meta_harness(
                mode=normalized_harness_mode,
                default_harness=normalized_default_harness,
                config=harness_config,
            )
        agent = builder.build()
        if resolved_llm and getattr(agent, "_llm_init_error", None):
            raise typer.BadParameter(
                "The requested LLM provider could not be initialized. Check its "
                "credentials and configuration."
            )
        agent.save()
        if set_default:
            cfg.save_state({"agent_id": agent.agent_id})
        _print_payload(
            {
                "ok": True,
                "agent": _agent_summary(agent),
                "default_agent": bool(set_default),
                "warnings": warnings,
            },
            raw_json=raw_json,
        )
    finally:
        if agent is not None:
            agent.close(close_memory_provider=True)
        else:
            close = getattr(provider, "close", None)
            if callable(close):
                close()


@agents_app.command("list")
def list_agents(
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """List persisted MemAgents in the configured provider."""
    provider, warnings = _provider()
    try:
        agents = [_agent_summary(agent) for agent in provider.list_memagents() or []]
        payload = {
            "ok": True,
            "agents": agents,
            "count": len(agents),
            "warnings": warnings,
        }
        if raw_json:
            typer.echo(json.dumps(payload, sort_keys=True))
            return
        from rich.console import Console

        console = Console()
        if not agents:
            console.print("[dim]No saved agents.[/dim]")
            return
        for agent in agents:
            console.print(
                f"{agent['agent_id']}  [cyan]{agent.get('name') or ''}[/cyan] "
                f"[dim]{agent.get('application_mode') or 'assistant'}[/dim]"
            )
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()


@agents_app.command("show")
def show_agent(
    agent_id: str = typer.Argument(..., help="Persisted agent ID."),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Show secret-free metadata for one persisted MemAgent."""
    provider, warnings = _provider()
    try:
        agent = provider.retrieve_memagent(str(agent_id).strip())
        if not agent:
            typer.echo(f"Agent not found: {agent_id}", err=True)
            raise typer.Exit(1)
        payload = {"ok": True, "agent": _agent_summary(agent), "warnings": warnings}
        if raw_json:
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            from rich.console import Console

            Console().print_json(data=payload)
    finally:
        close = getattr(provider, "close", None)
        if callable(close):
            close()
