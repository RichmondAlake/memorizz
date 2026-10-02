"""Reuse the results of repeated tool calls.

A MemAgent with a tool cache answers a repeated call (the same tool, the same
arguments, for the same user) from the cache instead of running the tool
again, as long as the stored result is still fresh. Only tools that are safe
to reuse are cached:

- Python tools that opt in with ``@governed_tool(cacheable=True)``. Such a
  tool must be deterministic, free of side effects and not need approval.
- MCP tools whose server marks them both read-only (``readOnlyHint``) and
  idempotent (``idempotentHint``), when ``mcp="annotated"`` (the default).

Everything else runs every time. Failed calls are never stored. A result is
kept for the tool's own ``cache_ttl_seconds``, else the config's
``ttl_seconds``, and never longer than the freshness of any domain the tool
declares (MCP results: 5 minutes by default).

Entries live in a process-wide, thread-safe LRU store, so runs of the same
agent in one process (the local UI, a REPL session, a worker) share them. A
key covers the tool's name and arguments, a fingerprint of the tool itself
(its code and schema, or its MCP server), the user, and, with
``scope="agent"`` (the default), the agent; a changed tool or another user
never sees a stale or foreign result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

# How long results from a data domain stay fresh, in seconds; shared with the
# semantic cache.
DEFAULT_FRESHNESS_BY_DOMAIN = {"mcp": 300.0, "inventory": 60.0, "calendar": 60.0}
_SCOPES = {"agent", "user"}
_MCP_MODES = {"annotated", "off"}


@dataclass
class ToolCacheConfig:
    """How a MemAgent reuses tool results."""

    enabled: bool = True
    # Default freshness for a cacheable tool without its own TTL.
    ttl_seconds: float = 300.0
    # "agent": entries belong to one agent and one user. "user": agents
    # serving the same user share results of the same (fingerprinted) tool.
    scope: str = "agent"
    # "annotated": cache MCP tools marked read-only and idempotent; "off".
    mcp: str = "annotated"
    # Results larger than this (serialized) are not stored.
    max_result_chars: int = 200_000
    freshness_by_domain: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FRESHNESS_BY_DOMAIN)
    )

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.ttl_seconds = max(0.0, float(self.ttl_seconds))
        self.scope = str(self.scope or "agent").strip().lower()
        if self.scope not in _SCOPES:
            raise ValueError("tool cache scope must be 'agent' or 'user'")
        self.mcp = str(self.mcp or "annotated").strip().lower()
        if self.mcp not in _MCP_MODES:
            raise ValueError("tool cache mcp must be 'annotated' or 'off'")
        self.max_result_chars = max(1, int(self.max_result_chars))
        self.freshness_by_domain = {
            str(domain): max(0.0, float(seconds))
            for domain, seconds in dict(self.freshness_by_domain or {}).items()
        }

    @classmethod
    def from_value(
        cls, value: Union[None, bool, Dict[str, Any], "ToolCacheConfig"]
    ) -> Optional["ToolCacheConfig"]:
        """``True``/a dict/a config -> config; ``None``/``False``/disabled -> None."""
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if isinstance(value, cls):
            config = value
        elif isinstance(value, dict):
            known = {
                key: value[key] for key in cls.__dataclass_fields__ if key in value
            }
            config = cls(**known)
        else:
            raise TypeError("tool_cache must be a bool, a dict or a ToolCacheConfig")
        return config if config.enabled else None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCacheEntry:
    result: Any
    tool_name: str
    agent_id: Optional[str]
    user_id: Optional[str]
    stored_at: float  # time.time()
    expires_at: float  # time.time()
    duration_ms: float  # how long the call that produced it took


@dataclass(frozen=True)
class ToolCacheDecision:
    """Whether one call may use the cache, and for how long a result lives."""

    cacheable: bool
    reason: str
    ttl_seconds: float = 0.0


@dataclass(frozen=True)
class ToolCacheHit:
    result: Any
    age_seconds: float
    saved_ms: float
    expires_in_seconds: float


class ToolCacheStore:
    """Thread-safe LRU of tool results with per-entry expiry."""

    def __init__(self, max_entries: int = 2048):
        self.max_entries = max(1, int(max_entries))
        self._entries: "OrderedDict[str, ToolCacheEntry]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str, now: Optional[float] = None) -> Optional[ToolCacheEntry]:
        now = time.time() if now is None else now
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry

    def put(self, key: str, entry: ToolCacheEntry) -> None:
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def remove_where(self, predicate: Callable[[ToolCacheEntry], bool]) -> int:
        with self._lock:
            doomed = [key for key, entry in self._entries.items() if predicate(entry)]
            for key in doomed:
                del self._entries[key]
            return len(doomed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_SHARED_STORE = ToolCacheStore()


def shared_store() -> ToolCacheStore:
    """The process-wide store every MemAgent tool cache uses by default."""
    return _SHARED_STORE


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)


def callable_fingerprint(function: Any, metadata: Any = None) -> str:
    """A tool's identity: where it lives, its code and its schema. A changed
    function body or schema gives a new fingerprint, so old results lapse."""
    parts: List[Any] = [
        getattr(function, "__module__", None),
        getattr(function, "__qualname__", None),
    ]
    code = getattr(function, "__code__", None)
    if code is not None:
        parts.append(hashlib.sha256(code.co_code).hexdigest())
        parts.append(repr(code.co_consts))
        parts.append(list(code.co_names))
    if isinstance(metadata, dict):
        parts.append(metadata.get("input_schema") or metadata.get("parameters"))
    return hashlib.sha256(_canonical(parts).encode()).hexdigest()


class ToolCache:
    """One agent's view of the shared tool-result store."""

    def __init__(
        self,
        config: ToolCacheConfig,
        *,
        agent_id: Optional[str] = None,
        store: Optional[ToolCacheStore] = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.agent_id = agent_id
        self.store = store if store is not None else shared_store()
        self._clock = clock
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stored": 0,
            "bypasses": 0,
            "saved_ms": 0.0,
        }
        self._bypass_reasons: Dict[str, int] = {}

    # Admission -------------------------------------------------------

    def decide(
        self, policy: Dict[str, Any], *, is_mcp: bool = False
    ) -> ToolCacheDecision:
        """Decide from a tool's effective policy whether its call may be cached."""
        policy = dict(policy or {})
        if policy.get("side_effects"):
            return ToolCacheDecision(False, "side_effects")
        if policy.get("requires_approval"):
            return ToolCacheDecision(False, "requires_approval")
        if is_mcp:
            if self.config.mcp == "off":
                return ToolCacheDecision(False, "mcp_caching_off")
            if not policy.get("cacheable"):
                return ToolCacheDecision(False, "mcp_not_read_only_and_idempotent")
        else:
            if not policy.get("cacheable"):
                return ToolCacheDecision(False, "not_cacheable")
            if policy.get("deterministic") is False:
                return ToolCacheDecision(False, "not_deterministic")
        ttl = policy.get("cache_ttl_seconds")
        ttl = self.config.ttl_seconds if ttl is None else max(0.0, float(ttl))
        for domain in policy.get("domains") or []:
            limit = self.config.freshness_by_domain.get(str(domain))
            if limit is not None:
                ttl = min(ttl, limit)
        if ttl <= 0:
            return ToolCacheDecision(False, "no_freshness")
        return ToolCacheDecision(True, "cacheable", ttl)

    # Keys ------------------------------------------------------------

    def key(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        user_id: Optional[str],
        fingerprint: str,
    ) -> str:
        scope = self.agent_id if self.config.scope == "agent" else "<shared>"
        return hashlib.sha256(
            _canonical(
                {
                    "tool": str(tool_name),
                    "arguments": arguments,
                    "fingerprint": fingerprint,
                    "user": user_id if user_id is not None else "<anonymous>",
                    "agent": scope,
                }
            ).encode()
        ).hexdigest()

    # Lookup and storage ----------------------------------------------

    def lookup(self, key: str) -> Optional[ToolCacheHit]:
        now = self._clock()
        entry = self.store.get(key, now)
        if entry is None:
            with self._lock:
                self._stats["misses"] += 1
            return None
        try:
            result = copy.deepcopy(entry.result)
        except Exception:
            # Should not happen (results are copied on the way in), but a hit
            # must never hand out the stored object itself.
            with self._lock:
                self._stats["misses"] += 1
            return None
        with self._lock:
            self._stats["hits"] += 1
            self._stats["saved_ms"] += entry.duration_ms
        return ToolCacheHit(
            result=result,
            age_seconds=max(0.0, now - entry.stored_at),
            saved_ms=entry.duration_ms,
            expires_in_seconds=max(0.0, entry.expires_at - now),
        )

    def store_result(
        self,
        key: str,
        result: Any,
        *,
        tool_name: str,
        user_id: Optional[str],
        ttl_seconds: float,
        duration_ms: float,
        serialized: Optional[str] = None,
    ) -> bool:
        """Keep a successful result; returns whether it was stored."""
        size = len(serialized) if serialized is not None else len(_canonical(result))
        if size > self.config.max_result_chars:
            self.bypass("result_too_large")
            return False
        try:
            value = copy.deepcopy(result)
        except Exception:
            self.bypass("result_not_copyable")
            return False
        now = self._clock()
        self.store.put(
            key,
            ToolCacheEntry(
                result=value,
                tool_name=str(tool_name),
                agent_id=self.agent_id,
                user_id=user_id,
                stored_at=now,
                expires_at=now + float(ttl_seconds),
                duration_ms=max(0.0, float(duration_ms)),
            ),
        )
        with self._lock:
            self._stats["stored"] += 1
        return True

    def bypass(self, reason: str) -> None:
        with self._lock:
            self._stats["bypasses"] += 1
            self._bypass_reasons[reason] = self._bypass_reasons.get(reason, 0) + 1

    # Maintenance -----------------------------------------------------

    def invalidate(
        self, tool_name: Optional[str] = None, *, user_id: Optional[str] = None
    ) -> int:
        """Drop this agent's entries, optionally for one tool and/or user."""

        def matches(entry: ToolCacheEntry) -> bool:
            if self.config.scope == "agent" and entry.agent_id != self.agent_id:
                return False
            if tool_name is not None and entry.tool_name != str(tool_name):
                return False
            if user_id is not None and entry.user_id != user_id:
                return False
            return True

        return self.store.remove_where(matches)

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            stats = dict(self._stats)
            reasons = dict(self._bypass_reasons)
        lookups = stats["hits"] + stats["misses"]
        stats["saved_ms"] = round(stats["saved_ms"], 3)
        stats["hit_rate"] = round(stats["hits"] / lookups, 4) if lookups else 0.0
        stats["bypass_reasons"] = reasons
        stats["config"] = self.config.to_dict()
        return stats


def mcp_tool_is_cacheable(annotations: Optional[Dict[str, Any]]) -> bool:
    """MCP's own hints: a read-only, idempotent tool is safe to reuse."""
    values = dict(annotations or {})

    def hint(camel: str, snake: str) -> Any:
        return values.get(camel, values.get(snake))

    return (
        hint("readOnlyHint", "read_only_hint") is True
        and hint("idempotentHint", "idempotent_hint") is True
    )


__all__ = [
    "DEFAULT_FRESHNESS_BY_DOMAIN",
    "ToolCache",
    "ToolCacheConfig",
    "ToolCacheDecision",
    "ToolCacheEntry",
    "ToolCacheHit",
    "ToolCacheStore",
    "callable_fingerprint",
    "mcp_tool_is_cacheable",
    "shared_store",
]
