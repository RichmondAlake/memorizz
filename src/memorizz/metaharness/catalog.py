"""Lookups shared by the harness front ends (UI, CLI and MCP server).

Model choices per harness, saved agents and the harness delegates they hand
work to, the saved MemAgent a memagent run uses when none is named, the
setup a harness conversation carries forward, and creating a delegate that
runs on a harness. Nothing here imports the web UI.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Harnesses a delegate can run its share of a request on.
HARNESS_LABELS = {
    "codex": "Codex",
    "claude-code": "Claude Code",
    "openhands": "OpenHands",
    "deepseek": "DeepSeek",
    "pi": "pi",
    "hermes": "Hermes",
}

PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "deepseek": "DeepSeek",
    "ollama": "Ollama (local)",
}


def model_choices(
    service: Any, harnesses: Iterable[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """For each harness: model groups for its picker (newest first, up to
    ten per provider) and the default it runs without a pick."""
    from ..llms.model_lists import available_providers, latest_models_for

    adapters = getattr(service, "adapters", {}) or {}
    providers = available_providers()
    lists = latest_models_for(providers)

    def group(provider: str, prefix: bool = False, only: Optional[set] = None):
        models = lists.get(provider) or []
        if only is not None:
            models = [m for m in models if f"{provider}/{m}" in only or m in only]
        values = [f"{provider}/{m}" if prefix else m for m in models]
        return {"label": PROVIDER_LABELS.get(provider, provider), "models": values}

    result: Dict[str, Dict[str, Any]] = {}
    for item in harnesses:
        name = item.get("name")
        metadata = item.get("metadata") or {}
        adapter = adapters.get(name)
        default = (
            getattr(adapter, "default_model", None)
            or metadata.get("default_model")
            or ((item.get("models") or [None])[0] if name == "codex" else None)
        )
        groups: List[Dict[str, Any]] = []
        if name == "codex":
            groups = [{"label": "Codex", "models": list(item.get("models") or [])[:10]}]
        elif name == "claude-code":
            groups = [group("anthropic")]
        elif name == "deepseek":
            groups = [group("deepseek")]
        elif name == "pi":
            catalog = set(item.get("models") or [])
            groups = [
                group(provider, prefix=True, only=catalog or None)
                for provider in providers
                if provider != "ollama"
            ]
        elif name == "hermes":
            configured = str(
                getattr(adapter, "provider", None) or metadata.get("provider") or ""
            )
            groups = [group(configured)] if configured in lists else [group("ollama")]
        elif name == "openhands":
            groups = [group(p, prefix=True) for p in providers if p != "ollama"]
        elif name == "memagent":
            # A saved agent runs on any provider MemoRizz can reach.
            groups = [group(provider, prefix=True) for provider in providers]
        result[name] = {
            "default": default or "",
            "groups": [g for g in groups if g["models"]],
        }
    return result


def fill_reported_models(service: Any, rows: List[Dict[str, Any]]) -> None:
    """Name the model each run used when its record doesn't: harnesses report
    it in their events (live, while a run is still going)."""
    finder = getattr(getattr(service, "run_store", None), "event_models", None)
    if not rows or not callable(finder):
        return
    try:
        models = finder([row["run_id"] for row in rows])
    except Exception:
        return
    for row in rows:
        reported = models.get(row["run_id"])
        if reported and reported != row.get("model"):
            row["model_is_default"] = not row.get("model")
            row["model"] = reported


def _value(agent: Any, key: str) -> Any:
    return agent.get(key) if isinstance(agent, dict) else getattr(agent, key, None)


def agent_id_of(agent: Any) -> str:
    return str(_value(agent, "agent_id") or "")


def agent_name_of(agent: Any) -> str:
    persona = _value(agent, "persona")
    name = _value(agent, "name") or (
        persona.get("name")
        if isinstance(persona, dict)
        else getattr(persona, "name", None)
    )
    return str(name or f"Untitled agent · {agent_id_of(agent)[:8]}")


def harness_backing(agent: Any) -> Optional[Dict[str, str]]:
    """The harness an agent runs whole turns on (a harness delegate), if any."""
    if str(_value(agent, "meta_harness_mode") or "").strip().lower() != "runtime":
        return None
    harness = str(_value(agent, "default_harness") or "").strip() or "auto"
    config = _value(agent, "harness_config") or {}
    model = str(config.get("model") or "").strip() if isinstance(config, dict) else ""
    return {
        "harness": harness,
        "label": HARNESS_LABELS.get(harness, harness),
        "model": model,
    }


def delegates_work(agent: Any) -> bool:
    """Whether an agent hands parts of a request to its delegates."""
    delegation = _value(agent, "delegation_config") or {}
    if not isinstance(delegation, dict):
        delegation = {}
    return (
        bool(_value(agent, "delegates"))
        and delegation.get("enabled", True) is not False
        and (delegation.get("mode", "auto") in {"auto", "deterministic"})
    )


def agent_teams(agents: Optional[Iterable[Any]]) -> Dict[str, List[str]]:
    """Each coordinator's delegates, labelled with the harness each runs on,
    for agents that hand parts of a request to delegates."""
    by_id = {agent_id_of(agent): agent for agent in agents or [] if agent_id_of(agent)}
    teams: Dict[str, List[str]] = {}
    for agent_id, agent in by_id.items():
        if not delegates_work(agent):
            continue
        labels = []
        for delegate_id in _value(agent, "delegates") or []:
            delegate = by_id.get(str(delegate_id))
            if delegate is None:
                continue
            backing = harness_backing(delegate)
            labels.append(
                agent_name_of(delegate)
                + (f" ({backing['harness']})" if backing else "")
            )
        if labels:
            teams[agent_id] = labels
    return teams


def default_memagent(service: Any, agents: Iterable[Any]) -> Optional[Dict[str, str]]:
    """The saved MemAgent to use when a memagent run names none: the one last
    used with a harness, else the newest saved agent."""
    by_id = {agent_id_of(agent): agent for agent in agents or [] if agent_id_of(agent)}
    if not by_id:
        return None
    try:
        recent = service.list_runs(limit=50) if service is not None else []
    except Exception:
        recent = []
    for run in recent:
        agent_id = str((run.get("task") or {}).get("agent_id") or "")
        if agent_id in by_id:
            return {"id": agent_id, "name": agent_name_of(by_id[agent_id])}
    newest = sorted(
        by_id.values(),
        key=lambda agent: str(_value(agent, "created_at") or ""),
        reverse=True,
    )[0]
    return {"id": agent_id_of(newest), "name": agent_name_of(newest)}


def with_memagent(
    payload: Dict[str, Any],
    harnesses: Iterable[Any],
    choose: Callable[[], Optional[Dict[str, str]]],
    *,
    where_to_create: str = "on the Agents page",
) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """Fill in a saved MemAgent when memagent would otherwise fail to start.

    Returns the payload and the agent chosen for it, or None when the payload
    already names one or memagent is not involved.
    """
    wants = "memagent" in {str(name or "").strip().lower() for name in harnesses}
    if not wants or str(payload.get("agent_id") or "").strip():
        return payload, None
    chosen = choose()
    if chosen is None:
        raise ValueError(
            "memagent runs a saved MemAgent, and there is none yet. Create one "
            f"{where_to_create}, or leave memagent out of this run."
        )
    return {**payload, "agent_id": chosen["id"]}, chosen


def chat_setup(run: Dict[str, Any]) -> Dict[str, Any]:
    """The launch settings a conversation carries forward from its latest run."""
    task = dict(run.get("task") or {})
    permissions = dict(task.get("permissions") or {})
    budget = dict(task.get("budget") or {})
    usage = dict((run.get("result") or {}).get("usage") or {})
    requested = str(task.get("harness") or "auto")
    return {
        # Stay on the harness that answered, even if auto route picked it.
        "harness": str(run.get("harness") or requested),
        "model": str(task.get("model") or ""),
        "model_used": str(task.get("model") or usage.get("model") or ""),
        "workspace": str(task.get("workspace") or ""),
        "network": str(permissions.get("network") or "none"),
        "write": permissions.get("workspace_mode") == "direct",
        "mcp_access": str(permissions.get("mcp_access") or "read_only"),
        "allow_dirty_workspace": bool(permissions.get("allow_dirty_workspace")),
        "allow_subagents": bool(permissions.get("allow_subagents")),
        "agent_id": str(task.get("agent_id") or ""),
        "memory_id": str(task.get("memory_id") or ""),
        "user_id": str(task.get("user_id") or ""),
        "timeout_seconds": int(budget.get("max_wall_time_seconds") or 900),
        "max_steps": int(budget.get("max_steps") or 80),
        "max_cost_usd": budget.get("max_cost_usd"),
        "verification_command": str(
            (task.get("verification") or {}).get("command") or ""
        ),
        "execution_backend": str(
            (task.get("metadata") or {}).get("execution_backend") or "local"
        ),
    }


def create_harness_delegate(
    provider: Any,
    *,
    harness: str,
    model: str = "",
    name: str = "",
    coordinator_id: str = "",
    default_llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create and save an agent that runs each task it is given on a harness.

    The delegate loads with a model like any saved agent (the coordinator's,
    else ``default_llm_config``); its own turns run on the harness, in the
    workspace of the run it is part of. Returns its id, name and harness.
    """
    from ..memagent.models import MemAgentModel

    harness = str(harness or "").strip().lower().replace("_", "-")
    if harness not in HARNESS_LABELS:
        raise ValueError("Choose a harness: " + ", ".join(HARNESS_LABELS.values()))
    model = str(model or "").strip()
    if len(model) > 200 or any(ch in model for ch in "\n\r\t"):
        raise ValueError("Model name is not valid")
    label = HARNESS_LABELS[harness]
    name = str(name or "").strip()[:80] or (
        f"{label} delegate" + (f" ({model})" if model else "")
    )
    llm_config: Optional[Dict[str, Any]] = None
    if coordinator_id:
        coordinator = provider.retrieve_memagent(str(coordinator_id))
        if coordinator is None:
            raise KeyError(f"Unknown agent: {coordinator_id}")
        saved = _value(coordinator, "llm_config")
        if isinstance(saved, dict) and saved:
            llm_config = dict(saved)
    if llm_config is None:
        llm_config = dict(default_llm_config or {})
    memagent = MemAgentModel(
        name=name,
        instruction=f"Runs each task it is given on {label}.",
        application_mode="assistant",
        llm_config=llm_config,
        meta_harness=True,
        meta_harness_mode="runtime",
        default_harness=harness,
        harness_config={"model": model} if model else {},
    )
    result = provider.store_memagent(memagent)
    agent_id = memagent.agent_id
    if isinstance(result, str) and result:
        agent_id = result
    elif isinstance(result, dict):
        agent_id = (
            result.get("agent_id")
            or result.get("_id")
            or result.get("agentId")
            or agent_id
        )
    elif getattr(result, "agent_id", None):
        agent_id = result.agent_id
    agent_id = str(agent_id)
    return {
        "id": agent_id,
        "name": name,
        "model": str(llm_config.get("model") or "").strip(),
        "runs_on": {"harness": harness, "label": label, "model": model},
    }


__all__ = [
    "HARNESS_LABELS",
    "PROVIDER_LABELS",
    "agent_id_of",
    "agent_name_of",
    "agent_teams",
    "chat_setup",
    "create_harness_delegate",
    "default_memagent",
    "delegates_work",
    "fill_reported_models",
    "harness_backing",
    "model_choices",
    "with_memagent",
]
