"""Per-agent choices for the models that do not touch stored data.

The memory router (``memory_type="auto"`` writes) and the reranker (recall
and search) can differ by agent: each agent follows the store's choice
unless it has its own, a fine-tuned version or none at all. Embeddings
cannot: every memory in a store is a vector from one model, and agents that
share a store search the same vectors.

Choices live in the memory provider with the other fine-tuning records.
An agent is found from its id (its own recall) or from a memory id it owns
(MCP writes and searches name a memory id). Lookups are cached briefly, so
recall does not read the store each time.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

JOBS = ("memory_routing", "reranking")
TTL = 30.0

_lock = threading.Lock()
_cache: Dict[tuple, tuple] = {}


def setting_key(agent_id: str, job: str) -> str:
    return f"agent:{agent_id}:{job}"


def _cached(key: tuple, load):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < TTL:
            return hit[1]
    value = load()
    with _lock:
        _cache[key] = (now, value)
    return value


def forget() -> None:
    """Drop cached choices (after one changes in this process)."""
    with _lock:
        _cache.clear()


def _store(provider: Any):
    from .store import FineTuneStore

    return FineTuneStore(provider)


def agent_for_memory(provider: Any, memory_id: Optional[str]) -> Optional[str]:
    """The agent that owns ``memory_id``, if one does."""
    if not memory_id or provider is None:
        return None

    def load():
        owners: Dict[str, str] = {}
        try:
            for agent in provider.list_memagents() or []:
                data = (
                    agent if isinstance(agent, dict) else getattr(agent, "__dict__", {})
                )
                agent_id = data.get("agent_id") or getattr(agent, "agent_id", None)
                for mid in (
                    data.get("memory_ids") or getattr(agent, "memory_ids", None) or []
                ):
                    owners.setdefault(str(mid), str(agent_id))
        except Exception:
            pass
        return owners

    return _cached(("owners", id(provider)), load).get(str(memory_id))


def choice(
    provider: Any, agent_id: Optional[str], job: str
) -> Optional[Dict[str, Any]]:
    """An agent's own choice for ``job`` ({"mode": "off"} or {"mode": "run",
    "run_id", "model_dir"}), or None when it follows the store."""
    if not agent_id or provider is None:
        return None

    def load():
        try:
            return _store(provider).get_setting(setting_key(agent_id, job))
        except Exception:
            return None

    return _cached(("choice", id(provider), agent_id, job), load)


def record_setting(provider: Any, agent_id: Optional[str], name: str) -> Any:
    """A setting saved on the agent's own record (``reranker``,
    ``memory_routing``), or None."""
    if not agent_id or provider is None:
        return None

    def load():
        try:
            record = provider.retrieve_memagent(agent_id)
        except Exception:
            return None
        if isinstance(record, dict):
            return record.get(name)
        return getattr(record, name, None)

    return _cached(("record", id(provider), agent_id, name), load)


def resolve(
    provider: Any,
    job: str,
    *,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The agent's own choice for ``job`` (found from ``agent_id`` or the
    owner of ``memory_id``), or None to use the store's."""
    agent = agent_id or agent_for_memory(provider, memory_id)
    return choice(provider, agent, job) if agent else None


__all__ = [
    "JOBS",
    "agent_for_memory",
    "choice",
    "forget",
    "record_setting",
    "resolve",
    "setting_key",
]
