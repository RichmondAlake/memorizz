"""Validated delegate settings for a saved agent, shared by the UI and CLI.

Which saved agents an agent hands parts of a request to, whether it does,
how many work at once, how their results are combined, and whether the agent
answers itself when delegating fails. Choices that would make agents
delegate to each other in a loop are refused.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

DELEGATION_CONSOLIDATIONS = {
    "model": "Agent writes one answer",
    "deterministic": "List each delegate's result",
}
MAX_DELEGATION_WORKERS = 8


def _value(agent: Any, key: str) -> Any:
    return agent.get(key) if isinstance(agent, dict) else getattr(agent, key, None)


def _display_name(agent: Any) -> str:
    name = str(_value(agent, "name") or "").strip()
    if not name:
        persona = _value(agent, "persona")
        name = str(
            (
                persona.get("name")
                if isinstance(persona, dict)
                else getattr(persona, "name", "")
            )
            or ""
        ).strip()
    return name or "Untitled agent"


def delegate_ids(agent: Any) -> List[str]:
    return [
        str(item).strip()
        for item in (_value(agent, "delegates") or [])
        if str(item).strip()
    ]


def delegation_graph(provider: Any) -> Dict[str, List[str]]:
    """Each saved agent's delegates, by agent ID."""
    try:
        agents = list(provider.list_memagents() or [])
    except Exception:
        return {}
    graph: Dict[str, List[str]] = {}
    for agent in agents:
        agent_id = (
            _value(agent, "agent_id")
            or _value(agent, "agentId")
            or _value(agent, "_id")
        )
        if agent_id:
            graph[str(agent_id).strip()] = delegate_ids(agent)
    return graph


def delegation_path(
    graph: Dict[str, List[str]], start: str, target: str
) -> Optional[List[str]]:
    """The delegate chain from ``start`` that reaches ``target``, if any."""
    stack = [(start, [start])]
    seen = set()
    while stack:
        node, path = stack.pop()
        if node == target:
            return path
        if node in seen:
            continue
        seen.add(node)
        for child in graph.get(node, []):
            stack.append((child, path + [child]))
    return None


def delegation_settings(
    provider: Any,
    agent_id: Optional[str],
    delegates: Iterable[Any],
    *,
    enabled: bool,
    max_workers: Any = None,
    consolidation: Optional[str] = None,
    root_fallback: bool = True,
    existing_config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Validate delegate choices: (delegate IDs, delegation config).

    Raises ValueError with a message for the person choosing.
    """
    from ..task_decomposition import normalize_delegation_config

    ids = list(
        dict.fromkeys(
            str(item).strip() for item in delegates if str(item or "").strip()
        )
    )
    config = dict(existing_config) if isinstance(existing_config, dict) else {}
    config["enabled"] = bool(enabled)
    if config.get("mode") not in {"auto", "deterministic"} or (
        config.get("mode") == "deterministic" and not config.get("plan")
    ):
        config["mode"] = "auto"
    workers_text = str(max_workers if max_workers is not None else "").strip()
    if workers_text:
        try:
            workers = int(workers_text)
        except ValueError:
            raise ValueError(
                "Delegates working at once must be a whole number"
            ) from None
        if not 1 <= workers <= MAX_DELEGATION_WORKERS:
            raise ValueError(
                f"Delegates working at once must be between 1 and {MAX_DELEGATION_WORKERS}"
            )
        config["max_workers"] = workers
    else:
        config.pop("max_workers", None)
    strategy = str(consolidation or "").strip().lower() or "model"
    if strategy not in DELEGATION_CONSOLIDATIONS:
        raise ValueError("Choose how the agent combines its delegates' results")
    config["consolidation_strategy"] = strategy
    config["allow_root_fallback"] = bool(root_fallback)

    if agent_id and agent_id in ids:
        raise ValueError("An agent can't be its own delegate")
    graph = delegation_graph(provider)
    names: Dict[str, str] = {}
    for delegate_id in ids:
        record = provider.retrieve_memagent(delegate_id)
        if not record:
            raise ValueError(f"Delegate agent {delegate_id} was not found")
        names[delegate_id] = _display_name(record)
        if agent_id:
            path = delegation_path(graph, delegate_id, agent_id)
            if path:
                chain = " → ".join(
                    names.get(item)
                    or _display_name(provider.retrieve_memagent(item) or {})
                    for item in path[:-1]
                )
                raise ValueError(
                    f"{names[delegate_id]} can't be a delegate: its delegates lead "
                    f"back to this agent ({chain} → this agent), which would loop."
                )
    try:
        config = normalize_delegation_config(config, for_persistence=True)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc
    return ids, config


__all__ = [
    "DELEGATION_CONSOLIDATIONS",
    "MAX_DELEGATION_WORKERS",
    "delegate_ids",
    "delegation_graph",
    "delegation_path",
    "delegation_settings",
]
