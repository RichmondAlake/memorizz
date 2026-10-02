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
    harness_config = getattr(agent, "harness_config", None) or {}
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
        "tool_cache": bool(
            getattr(agent, "tool_cache", None)
            or getattr(agent, "tool_cache_config", None)
        ),
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
        "harness_model": (
            harness_config.get("model") if isinstance(harness_config, dict) else None
        ),
        "harness_workspace": (
            harness_config.get("workspace")
            if isinstance(harness_config, dict)
            else None
        ),
        "delegates": [
            str(getattr(item, "agent_id", item) or "")
            for item in (getattr(agent, "delegates", None) or [])
        ],
        "delegation": dict(
            getattr(agent, "delegation_config", None)
            or getattr(agent, "delegation", None)
            or {}
        ),
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


DELEGATE = typer.Option(
    None,
    "--delegate",
    help="A saved agent this one hands parts of a request to; repeat for more.",
)
DELEGATION = typer.Option(
    True,
    "--delegation/--no-delegation",
    help="Whether the agent hands work to its delegates.",
)
DELEGATION_WORKERS = typer.Option(
    None,
    "--delegation-max-workers",
    min=1,
    max=8,
    help="Delegates working at once (1-8).",
)
DELEGATION_CONSOLIDATION = typer.Option(
    "model",
    "--delegation-consolidation",
    help="model (the agent writes one answer) or deterministic (list each delegate's result).",
)
ROOT_FALLBACK = typer.Option(
    True,
    "--root-fallback/--no-root-fallback",
    help="Answer itself when delegating fails.",
)


def _harness_settings(
    harness_mode: Optional[str],
    default_harness: Optional[str],
    harness_workspace: Optional[str],
    harness_model: Optional[str],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """(mode, default harness, harness config) from the harness options."""
    if harness_mode is None:
        given = [
            flag
            for flag, value in (
                ("--default-harness", default_harness),
                ("--harness-workspace", harness_workspace),
                ("--harness-model", harness_model),
            )
            if value
        ]
        if given:
            raise typer.BadParameter(
                f"{', '.join(given)} needs --harness-mode delegate or runtime"
            )
        return None
    mode = str(harness_mode).strip().lower()
    if mode not in {"delegate", "runtime"}:
        raise typer.BadParameter("--harness-mode must be delegate or runtime")
    harness = str(default_harness or "auto").strip().lower().replace("_", "-")
    from ..metaharness.config import DEFAULT_HARNESS_CHOICES

    if harness not in DEFAULT_HARNESS_CHOICES:
        raise typer.BadParameter(
            "--default-harness must be auto, codex, claude-code, "
            "openhands, deepseek, pi, hermes, or native"
        )
    config: Dict[str, Any] = {}
    if harness_workspace:
        try:
            workspace_path = Path(harness_workspace).expanduser().resolve(strict=True)
        except OSError as exc:
            raise typer.BadParameter(
                "--harness-workspace must be an existing directory"
            ) from exc
        if not workspace_path.is_dir():
            raise typer.BadParameter(
                "--harness-workspace must be an existing directory"
            )
        config["workspace"] = str(workspace_path)
        config["permissions"] = {"allowed_roots": [str(workspace_path)]}
    if harness_model and harness_model.strip():
        config["model"] = harness_model.strip()
    return mode, harness, config


def _with_delegates(
    provider: Any,
    agent_id: str,
    delegates: List[str],
    *,
    enabled: bool,
    max_workers: Optional[int],
    consolidation: str,
    root_fallback: bool,
) -> None:
    """Save validated delegates and delegation settings on a saved agent."""
    from ..memagent.delegation_settings import delegation_settings

    record = provider.retrieve_memagent(agent_id)
    try:
        ids, config = delegation_settings(
            provider,
            agent_id,
            delegates,
            enabled=enabled,
            max_workers=max_workers,
            consolidation=consolidation,
            root_fallback=root_fallback,
            existing_config=getattr(record, "delegation_config", None),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--delegate") from exc
    provider.store_memagent(
        record.model_copy(update={"delegates": ids, "delegation_config": config})
    )


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
    tool_cache: bool = typer.Option(
        False,
        "--tool-cache/--no-tool-cache",
        help="Reuse results of repeated calls to cacheable tools (and read-only, idempotent MCP tools).",
    ),
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
    default_harness: Optional[str] = typer.Option(
        None,
        "--default-harness",
        help="auto (default), codex, claude-code, openhands, deepseek, pi, hermes or native; needs --harness-mode.",
    ),
    harness_workspace: Optional[str] = typer.Option(
        None,
        "--harness-workspace",
        help="Default workspace for runtime-mode harness turns; needs --harness-mode.",
    ),
    harness_model: Optional[str] = typer.Option(
        None,
        "--harness-model",
        help="Model the harness runs (see `memorizz harness models`); needs --harness-mode.",
    ),
    delegate: Optional[List[str]] = DELEGATE,
    delegation: bool = DELEGATION,
    delegation_max_workers: Optional[int] = DELEGATION_WORKERS,
    delegation_consolidation: str = DELEGATION_CONSOLIDATION,
    root_fallback: bool = ROOT_FALLBACK,
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
            .with_tool_cache(tool_cache)
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
        harness_settings = _harness_settings(
            harness_mode, default_harness, harness_workspace, harness_model
        )
        if harness_settings is not None:
            mode_value, harness_value, harness_config = harness_settings
            builder.with_meta_harness(
                mode=mode_value,
                default_harness=harness_value,
                config=harness_config,
            )
        agent = builder.build()
        if resolved_llm and getattr(agent, "_llm_init_error", None):
            raise typer.BadParameter(
                "The requested LLM provider could not be initialized. Check its "
                "credentials and configuration."
            )
        agent.save()
        if delegate:
            _with_delegates(
                provider,
                agent.agent_id,
                delegate,
                enabled=delegation,
                max_workers=delegation_max_workers,
                consolidation=delegation_consolidation,
                root_fallback=root_fallback,
            )
        if set_default:
            cfg.save_state({"agent_id": agent.agent_id})
        _print_payload(
            {
                "ok": True,
                "agent": _agent_summary(
                    provider.retrieve_memagent(agent.agent_id) or agent
                )
                if delegate
                else _agent_summary(agent),
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


def _close(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        close()


@agents_app.command("update")
def update_agent(
    agent_id: str = typer.Argument(..., help="Persisted agent ID."),
    name: Optional[str] = typer.Option(None, "--name", "-n"),
    instruction: Optional[str] = typer.Option(None, "--instruction", "-i"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", min=1, max=1000),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider"),
    model: Optional[str] = typer.Option(None, "--model", help="With --llm-provider."),
    harness_mode: Optional[str] = typer.Option(
        None,
        "--harness-mode",
        help="delegate, runtime, or off to turn the meta-harness off.",
    ),
    default_harness: Optional[str] = typer.Option(None, "--default-harness"),
    harness_workspace: Optional[str] = typer.Option(None, "--harness-workspace"),
    harness_model: Optional[str] = typer.Option(None, "--harness-model"),
    delegate: Optional[List[str]] = typer.Option(
        None,
        "--delegate",
        help="Replace the delegates with these saved agents; repeatable.",
    ),
    add_delegate: Optional[List[str]] = typer.Option(
        None, "--add-delegate", help="Add a delegate; repeatable."
    ),
    remove_delegate: Optional[List[str]] = typer.Option(
        None, "--remove-delegate", help="Remove a delegate; repeatable."
    ),
    delegation: Optional[bool] = typer.Option(None, "--delegation/--no-delegation"),
    delegation_max_workers: Optional[int] = typer.Option(
        None, "--delegation-max-workers", min=1, max=8
    ),
    delegation_consolidation: Optional[str] = typer.Option(
        None, "--delegation-consolidation"
    ),
    root_fallback: Optional[bool] = typer.Option(
        None, "--root-fallback/--no-root-fallback"
    ),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Change a saved MemAgent's name, instruction, model, harness or delegates.

    Options left out keep their current values.
    """
    provider, warnings = _provider()
    try:
        record = provider.retrieve_memagent(str(agent_id).strip())
        if not record:
            typer.echo(f"Agent not found: {agent_id}", err=True)
            raise typer.Exit(1)
        changes: Dict[str, Any] = {}
        if name is not None:
            if not name.strip():
                raise typer.BadParameter("--name cannot be empty")
            changes["name"] = name.strip()
        if instruction is not None:
            changes["instruction"] = instruction.strip()
        if max_steps is not None:
            changes["max_steps"] = max_steps
        if llm_provider or model:
            changes["llm_config"] = _llm_config(llm_provider, model, disabled=False)
        if harness_mode is not None and harness_mode.strip().lower() == "off":
            changes.update(meta_harness=False, meta_harness_mode=None)
        elif (
            harness_mode is not None
            or default_harness
            or harness_workspace
            or harness_model
        ):
            settings = _harness_settings(
                harness_mode or record.meta_harness_mode,
                default_harness or record.default_harness,
                harness_workspace,
                harness_model,
            )
            if settings is None:
                raise typer.BadParameter(
                    "--default-harness, --harness-workspace and --harness-model need --harness-mode"
                )
            mode_value, harness_value, harness_config = settings
            merged = dict(record.harness_config or {})
            merged.update(harness_config)
            changes.update(
                meta_harness=True,
                meta_harness_mode=mode_value,
                default_harness=harness_value,
                harness_config=merged,
            )
        if changes:
            record = record.model_copy(update=changes)
            provider.store_memagent(record)
        current = list(record.delegates or [])
        wanted = list(delegate) if delegate else list(current)
        wanted += [item for item in add_delegate or [] if item not in wanted]
        wanted = [item for item in wanted if item not in set(remove_delegate or [])]
        config = dict(record.delegation_config or {})
        if (
            wanted != current
            or delegation is not None
            or delegation_max_workers is not None
            or delegation_consolidation is not None
            or root_fallback is not None
        ):
            _with_delegates(
                provider,
                record.agent_id,
                wanted,
                enabled=config.get("enabled", True)
                if delegation is None
                else delegation,
                max_workers=delegation_max_workers
                if delegation_max_workers is not None
                else config.get("max_workers"),
                consolidation=delegation_consolidation
                or config.get("consolidation_strategy")
                or "model",
                root_fallback=config.get("allow_root_fallback", True)
                if root_fallback is None
                else root_fallback,
            )
        payload = {
            "ok": True,
            "agent": _agent_summary(provider.retrieve_memagent(record.agent_id)),
            "warnings": warnings,
        }
        if raw_json:
            typer.echo(json.dumps(payload, sort_keys=True))
        else:
            from rich.console import Console

            Console().print_json(data=payload)
    finally:
        _close(provider)


@agents_app.command("delete")
def delete_agent(
    agent_id: str = typer.Argument(..., help="Persisted agent ID."),
    cascade: bool = typer.Option(
        False, "--cascade", help="Also delete the agent's memories."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation."),
    raw_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Delete a saved MemAgent. Agents that had it as a delegate stop using it."""
    provider, warnings = _provider()
    try:
        normalized = str(agent_id).strip()
        record = provider.retrieve_memagent(normalized)
        if not record:
            typer.echo(f"Agent not found: {agent_id}", err=True)
            raise typer.Exit(1)
        if not yes and not typer.confirm(
            f"Delete agent {record.name or normalized}"
            + (" and its memories" if cascade else "")
            + "?",
            default=False,
        ):
            raise typer.Exit(1)
        coordinators = []
        for other in provider.list_memagents() or []:
            read = (
                other.get
                if isinstance(other, dict)
                else lambda key, o=other: getattr(o, key, None)
            )
            other_id = str(read("agent_id") or "")
            if other_id and normalized in (read("delegates") or []):
                saved = provider.retrieve_memagent(other_id)
                provider.store_memagent(
                    saved.model_copy(
                        update={
                            "delegates": [
                                item
                                for item in saved.delegates or []
                                if item != normalized
                            ]
                        }
                    )
                )
                coordinators.append(other_id)
        deleted = bool(provider.delete_memagent(normalized, cascade=cascade))
        payload = {
            "ok": deleted,
            "agent_id": normalized,
            "deleted": deleted,
            "cascade": cascade,
            "removed_as_delegate_from": coordinators,
            "warnings": warnings,
        }
        typer.echo(json.dumps(payload, sort_keys=True, indent=None if raw_json else 2))
        if not deleted:
            raise typer.Exit(1)
    finally:
        _close(provider)
