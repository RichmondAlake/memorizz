"""A saved agent's configuration for an isolated evaluation, without secrets.

Evalground and ``memorizz eval run --agent-id`` write this as the agent
template a memory-suite run rebuilds the agent from: keys, tokens and other
credentials are dropped, and tools, delegates, MCP servers, sandbox, browser,
internet, harness, learning and automations are switched off so the run
measures the agent's memory alone.
"""

from __future__ import annotations

from typing import Any, Dict

_AGENT_TEMPLATE_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "credential",
    "cookie",
    "private_key",
)


def secret_free_agent_template(agent: Any) -> Dict[str, Any]:
    """Return the persisted agent configuration needed by an isolated eval."""

    if hasattr(agent, "model_dump"):
        payload = agent.model_dump(
            mode="python",
            exclude={"model", "tools"},
            exclude_none=True,
        )
    elif isinstance(agent, dict):
        payload = {
            key: value for key, value in agent.items() if key not in {"model", "tools"}
        }
    else:
        raise TypeError("Selected agent does not expose a serializable configuration")

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): clean(child)
                for key, child in value.items()
                if not any(
                    marker in str(key).casefold()
                    for marker in _AGENT_TEMPLATE_SECRET_MARKERS
                )
            }
        if isinstance(value, (list, tuple, set)):
            return [clean(item) for item in value]
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, (str, int, float, bool)):
            return enum_value
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    snapshot = clean(payload)
    snapshot.update(
        {
            "tools": [],
            "delegates": [],
            "mcp_servers": [],
            "internet_access_provider": None,
            "internet_access_config": None,
            "sandbox_provider": None,
            "browser_control": None,
            "meta_harness": False,
            "continual_learning": False,
            "learning_control_plane": False,
            "learning_control_plane_config": None,
            "automations_enabled": False,
        }
    )
    return snapshot


__all__ = ["secret_free_agent_template"]
