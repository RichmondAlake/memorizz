"""Provider-neutral memory changes, attribution and explicit lineage.

Recording is opt-in for SDK providers and enabled by the UI/MCP host. Existing
records are observations, never invented historical creations. Journal entries
contain hashes and field names, not copies of memory content (including deleted
content). Request-local attribution propagates through asyncio/to_thread.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps

from .enums.memory_type import MemoryType
from .memory_provider.base import _UNSET

logger = logging.getLogger(__name__)
CHANGE_TYPE = "memory_history_change"
HEAD_TYPE = "memory_history_head"
_context: ContextVar[dict] = ContextVar("memorizz_memory_change_context", default={})
_writing: ContextVar[frozenset] = ContextVar(
    "memorizz_memory_history_writing", default=frozenset()
)
_install_lock = threading.RLock()
_IGNORE = {"embedding", "vector", "_id", "id", "updated_at", "timestamp"}
_IDENTITY = (
    "agent_id",
    "memory_id",
    "thread_id",
    "user_id",
    "application_id",
    "run_id",
    "turn_id",
    "root_trace_id",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json(value):
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _hash(value):
    if value is None:
        return None
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _fields(row):
    return {str(k): _json(v) for k, v in (row or {}).items() if k not in _IGNORE}


def _internal(row):
    kind = str((row or {}).get("record_type") or "")
    content = (row or {}).get("content")
    if not kind and isinstance(content, str):
        try:
            content = json.loads(content)
        except (ValueError, TypeError):
            content = None
    if not kind and isinstance(content, dict):
        kind = str(content.get("record_type") or "")
    return kind.startswith(("memory_history_", "observability_"))


def _record_id(row):
    return str(
        (row or {}).get("_id")
        or (row or {}).get("id")
        or (row or {}).get("record_id")
        or ""
    )


def _parents(row):
    """Only explicit identifiers constitute lineage; no semantic guessing."""
    links = []
    metadata = (row or {}).get("metadata")
    fields = {**(metadata if isinstance(metadata, dict) else {}), **(row or {})}
    for field, relation in (
        ("supersedes", "supersedes"),
        ("supersedes_id", "supersedes"),
        ("duplicate_of", "duplicate_of"),
        ("parent_id", "derived_from"),
        ("source_id", "derived_from"),
        ("source_ids", "derived_from"),
        ("source_record_ids", "derived_from"),
        ("covered_message_ids", "summarizes"),
        ("message_ids", "summarizes"),
        ("source_message_ids", "summarizes"),
        ("derived_from", "derived_from"),
    ):
        value = fields.get(field)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, (str, int)) and str(item):
                links.append({"record_id": str(item), "relation": relation})
    return list({(p["record_id"], p["relation"]): p for p in links}.values())


@contextmanager
def memory_change_context(*, actor=None, source=None, **identity):
    """Attribute writes in this block; IDs are assertions by the host, not authentication.

    For example ``with memory_change_context(actor='operator-7', source='sdk')``.
    Provider authorization still governs the underlying write.
    """
    value = dict(_context.get())
    if actor is not None and value.get("actor") and actor != value["actor"]:
        value.setdefault("initiator", value["actor"])
    value.update(
        {
            k: v
            for k, v in {"actor": actor, "source": source, **identity}.items()
            if v is not None
        }
    )
    token = _context.set(value)
    try:
        yield
    finally:
        _context.reset(token)


def _lookup(provider, record_id, kind):
    if not record_id:
        return None
    row = provider.retrieve_by_id(str(record_id), kind)
    if not isinstance(row, dict) and kind == MemoryType.MEMAGENT:
        model = provider.retrieve_memagent(str(record_id))
        row = model.model_dump(mode="json") if hasattr(model, "model_dump") else None
    if not isinstance(row, dict) and kind == MemoryType.SHARED_MEMORY:
        row = provider.retrieve_by_name(str(record_id), kind)
    return _json(row) if isinstance(row, dict) else None


def _emit_change(provider, kind, record_id, before, after, operation):
    from .observability.store import ObservabilityStore

    row = after or before or {}
    if _internal(row):
        return
    metadata = row.get("metadata")
    description = {**(metadata if isinstance(metadata, dict) else {}), **row}
    old, new = _fields(before), _fields(after)
    if before is not None and after is not None and old == new:
        return
    context = dict(_context.get())
    scope = {k: row.get(k, context.get(k)) for k in _IDENTITY}
    scope["agent_id"] = (
        row.get("agent_id") or row.get("owner_agent_id") or context.get("agent_id")
    )
    scope["run_id"] = (
        context.get("harness_run_id")
        or row.get("run_id")
        or context.get("run_id")
        or os.getenv("MEMORIZZ_HARNESS_RUN_ID")
    )
    store = ObservabilityStore(provider)
    head_id = "mem-head-" + _hash([str(kind), record_id])[:32]
    head_row = _lookup(provider, head_id, MemoryType.SHARED_MEMORY)
    head = ObservabilityStore._payload(head_row) if head_row else None
    previous = (
        head.get("event_id")
        if head and head.get("version_hash") == _hash(old)
        else None
    )
    event = {
        "record_id": "mem-change-" + str(uuid.uuid4()),
        "record_type": CHANGE_TYPE,
        "timestamp": _now(),
        **scope,
        "target_record_id": str(record_id),
        "memory_type": str(getattr(kind, "value", kind)),
        "action": "deleted"
        if after is None
        else "updated"
        if before is not None
        else "created",
        "operation": operation,
        "actor": str(
            context.get("actor")
            or (
                description.get("updated_by")
                if before is not None
                else description.get("created_by")
            )
            or "unknown"
        ),
        "initiator": context.get("initiator"),
        "source": str(context.get("source") or description.get("source") or "sdk"),
        "source_ref": description.get("source_path")
        or description.get("source_uri")
        or description.get("source_url"),
        "attribution": "host_asserted"
        if context.get("actor")
        else "record_metadata"
        if description.get("created_by") or description.get("updated_by")
        else "unknown",
        "previous_event_id": previous,
        "before_hash": _hash(old) if before is not None else None,
        "after_hash": _hash(new) if after is not None else None,
        "changed_fields": sorted(
            k
            for k in set(old) | set(new)
            if old.get(k) != new.get(k) or (k in old) != (k in new)
        ),
        "field_changes": {
            k: "added" if k not in old else "removed" if k not in new else "modified"
            for k in set(old) | set(new)
            if old.get(k) != new.get(k) or (k in old) != (k in new)
        },
        "parents": _parents(row),
        "history_gap": before is not None and not previous,
        "label": str(
            context.get("label")
            or description.get("name")
            or description.get("title")
            or description.get("tool_name")
            or description.get("role")
            or str(getattr(kind, "value", kind)).replace("_", " ")
        )[:160],
        "content_stored": False,
    }
    from .metaharness.security import redact

    event = redact(event)
    # The journal has its own namespace: _put uses memory_id only for filtering.
    store._put(event)
    store._put(
        {
            "record_id": head_id,
            "record_type": HEAD_TYPE,
            "event_id": event["record_id"],
            "version_hash": _hash(new) if after is not None else None,
            "target_record_id": str(record_id),
            **scope,
        }
    )


@contextmanager
def _suspend_recording(provider):
    """Let a compound operation record its final changes once, without subwrites."""
    token = _writing.set(_writing.get() | {id(provider)})
    try:
        yield
    finally:
        _writing.reset(token)


def enable_memory_history(provider):
    """Install reversible-method instrumentation once on a provider instance.

    Failures in the auxiliary journal do not change a successful memory write.
    They log a warning; this journal is not a transactional compliance log.
    """
    if provider is None or not hasattr(provider, "__dict__"):
        return provider
    with _install_lock:
        if getattr(provider, "_memory_history_enabled", None) is True:
            return provider
        lock = threading.RLock()
        for name in (
            "store",
            "store_many",
            "update_by_id",
            "delete_by_id",
            "delete_by_name",
            "delete_all",
            "store_memagent",
            "update_memagent",
            "update_memagent_memory_ids",
            "delete_memagent_memory_ids",
            "store_summary_with_links",
            "compare_and_swap_shared_memory",
        ):
            original = getattr(provider, name, None)
            if not callable(original):
                continue
            signature = inspect.signature(original)

            def decorate(method, method_name, sig):
                @wraps(method)
                def tracked(*args, **kwargs):
                    if id(provider) in _writing.get():
                        return method(*args, **kwargs)
                    with lock:
                        token = _writing.set(_writing.get() | {id(provider)})
                        try:
                            try:
                                bound = sig.bind(*args, **kwargs)
                                bound.apply_defaults()
                                values = bound.arguments
                                kind = values.get("memory_store_type")
                                if method_name in {
                                    "store_memagent",
                                    "update_memagent",
                                    "update_memagent_memory_ids",
                                    "delete_memagent_memory_ids",
                                }:
                                    kind = MemoryType.MEMAGENT
                                if method_name == "store_summary_with_links":
                                    kind = MemoryType.SUMMARIES
                                if method_name == "compare_and_swap_shared_memory":
                                    kind = MemoryType.SHARED_MEMORY
                                unit = values.get("memory_unit")
                                before = {}
                                targets = []
                                data = values.get("data") or values.get("memagent")
                                if unit is not None:
                                    if isinstance(unit, dict):
                                        data = unit
                                    elif hasattr(unit, "model_dump"):
                                        data = unit.model_dump(mode="json")
                                    elif callable(getattr(unit, "dict", None)):
                                        data = unit.dict()
                                    elif hasattr(unit, "__dict__"):
                                        data = vars(unit)
                                    else:
                                        data = dict(unit)
                                    kind = (
                                        getattr(unit, "memory_type", None)
                                        or data.get("memory_type")
                                        or kind
                                        or MemoryType.CONVERSATION_MEMORY
                                    )
                                if method_name == "store_many":
                                    targets = [
                                        (_record_id(r), r)
                                        for r in values.get("rows", [])
                                    ]
                                elif method_name in {
                                    "store",
                                    "store_memagent",
                                    "update_memagent",
                                    "store_summary_with_links",
                                }:
                                    data = _json(data) if data is not None else {}
                                    target = _record_id(data) or str(
                                        data.get("agent_id") or ""
                                    )
                                    targets = [(target, data)]
                                elif method_name == "delete_all":
                                    try:
                                        targets = [
                                            (_record_id(r), r)
                                            for r in provider.list_all(kind) or []
                                            if isinstance(r, dict) and not _internal(r)
                                        ]
                                    except Exception:
                                        logger.warning(
                                            "Memory history could not read deletion targets"
                                        )
                                else:
                                    target = values.get("id") or values.get("agent_id")
                                    if method_name == "compare_and_swap_shared_memory":
                                        target = values.get("memory_id")
                                    if method_name == "delete_by_name":
                                        try:
                                            row = provider.retrieve_by_name(
                                                values.get("name"), kind
                                            )
                                            target = _record_id(row)
                                        except Exception:
                                            target = None
                                    targets = [(str(target or ""), None)]
                                for target, row in targets:
                                    try:
                                        before[target] = (
                                            _lookup(provider, target, kind)
                                            if target
                                            else None
                                        )
                                    except Exception:
                                        before[target] = None
                                linked = {}
                                if method_name == "store_summary_with_links":
                                    for message_id in (data or {}).get(
                                        "source_message_ids", []
                                    ):
                                        linked[message_id] = _lookup(
                                            provider,
                                            message_id,
                                            MemoryType.CONVERSATION_MEMORY,
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "Memory history preparation failed; proceeding with "
                                    "the write (%s: %s)",
                                    type(exc).__name__,
                                    exc,
                                    exc_info=logger.isEnabledFor(logging.DEBUG),
                                )
                                return method(*args, **kwargs)
                            result = method(*args, **kwargs)
                            if result is False or result is None:
                                return result
                            if method_name == "store_many":
                                ids = (
                                    list(result)
                                    if isinstance(result, (list, tuple))
                                    else []
                                )
                            elif method_name in {
                                "store",
                                "store_memagent",
                                "update_memagent",
                                "store_summary_with_links",
                            }:
                                ids = [
                                    str(result)
                                    if isinstance(result, (str, int))
                                    else targets[0][0]
                                ]
                            else:
                                ids = [target for target, _ in targets]
                            for index, target in enumerate(ids):
                                try:
                                    input_id, input_row = targets[index]
                                    after = (
                                        None
                                        if method_name.startswith("delete")
                                        and method_name != "delete_memagent_memory_ids"
                                        else _lookup(provider, target, kind)
                                    )
                                    # Only journal a confirmed provider result; never fabricate a row.
                                    old = before.get(input_id)
                                    if after is not None or old is not None:
                                        _emit_change(
                                            provider,
                                            kind,
                                            target,
                                            old,
                                            after,
                                            method_name,
                                        )
                                except Exception:
                                    logger.warning(
                                        "Memory write succeeded but history recording failed",
                                        exc_info=False,
                                    )
                            for message_id, old in linked.items():
                                try:
                                    after = _lookup(
                                        provider,
                                        message_id,
                                        MemoryType.CONVERSATION_MEMORY,
                                    )
                                    if old is not None and after is not None:
                                        _emit_change(
                                            provider,
                                            MemoryType.CONVERSATION_MEMORY,
                                            message_id,
                                            old,
                                            after,
                                            method_name,
                                        )
                                except Exception:
                                    logger.warning(
                                        "Summary committed but source-link history recording failed",
                                        exc_info=False,
                                    )
                            return result
                        finally:
                            _writing.reset(token)

                return tracked

            setattr(provider, name, decorate(original, name, signature))
        provider._memory_history_enabled = True
    return provider


class MemoryHistory:
    """Read a bounded, scoped journal and optionally record subsequent writes."""

    def __init__(self, provider, *, record_changes=False):
        self.provider = provider
        if record_changes:
            enable_memory_history(provider)

    def recording(self, *, actor=None, source="sdk", **identity):
        enable_memory_history(self.provider)
        return memory_change_context(actor=actor, source=source, **identity)

    def timeline(
        self,
        *,
        agent_id=None,
        memory_id=None,
        run_id=None,
        user_id=_UNSET,
        application_id=None,
        memory_type=None,
        action=None,
        actor=None,
        limit=200,
        cursor=None,
    ):
        from .observability.store import ObservabilityStore

        memory_type = getattr(memory_type, "value", memory_type)
        safe_limit = max(1, min(int(limit), 1000))
        # Agent and namespace filters must intersect. The portable provider query
        # combines them with OR, so request one indexed scope and enforce both.
        filters = {
            key: value
            for key, value in (
                ("agent_id", agent_id),
                ("memory_id", memory_id),
                ("run_id", run_id),
                ("memory_type", memory_type),
                ("action", action),
                ("actor", actor),
                ("application_id", application_id),
            )
            if value is not None and (key == "application_id" or value != "")
        }
        if user_id is not _UNSET:
            filters["user_id"] = user_id
        arguments = {
            "agent_ids": [agent_id] if agent_id else None,
            "memory_ids": [memory_id] if memory_id and not agent_id else None,
            "application_id": application_id,
            "record_type": CHANGE_TYPE,
        }
        # Providers historically used different omitted-user sentinels.
        if user_id is not _UNSET:
            arguments["user_id"] = user_id
        method = self.provider.query_observability_records
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            parameters = {}  # Native/third-party callables can lack a signature.
        if "event_filters" in parameters:
            arguments["event_filters"] = filters
        events = []
        next_cursor, seen = cursor, set()
        while len(events) < safe_limit:
            query = method(
                MemoryType.SHARED_MEMORY,
                **arguments,
                limit=safe_limit - len(events),
                cursor=next_cursor,
            )
            for row in query.get("items", []):
                event = ObservabilityStore._payload(row)
                if not event or event.get("record_type") != CHANGE_TYPE:
                    continue
                if all(event.get(key) == value for key, value in filters.items()):
                    events.append(event)
            continuation = query.get("next_cursor")
            if continuation and (continuation == next_cursor or continuation in seen):
                raise ValueError("Memory history pagination did not advance")
            next_cursor = continuation
            if not next_cursor:
                break
            seen.add(next_cursor)
        events.sort(key=lambda e: (str(e.get("timestamp") or ""), e["record_id"]))
        return {
            "events": events,
            "next_cursor": next_cursor,
            "limit": safe_limit,
            "content_stored": False,
            "coverage": "Recorded writes after history was enabled; older history and direct backend writes may be unavailable.",
        }

    def get_change(
        self, record_id, *, agent_id=None, user_id=_UNSET, application_id=None
    ):
        """Read one journal entry within the caller's explicit scope."""
        from .observability.store import ObservabilityStore

        event = ObservabilityStore(self.provider)._get(record_id)
        if not event or event.get("record_type") != CHANGE_TYPE:
            return None
        if agent_id is not None and event.get("agent_id") != agent_id:
            return None
        if user_id is not _UNSET and event.get("user_id") != user_id:
            return None
        if application_id is not None and event.get("application_id") != application_id:
            return None
        return event

    def current_record(self, event):
        """Read the current target, never reconstruct a historical value.

        The event must already have been authorized by its caller. A target
        reassigned to another tenant/application/owner must not be exposed.
        """
        kind = MemoryType(event["memory_type"])
        row = _lookup(self.provider, event["target_record_id"], kind)
        if not row:
            return {
                "state": "deleted" if event["action"] == "deleted" else "unavailable"
            }
        if any(row.get(key) != event.get(key) for key in ("user_id", "application_id")):
            return {"state": "unavailable"}
        owner = row.get("agent_id") or row.get("owner_agent_id")
        if owner and owner != event.get("agent_id"):
            return {"state": "unavailable"}
        if row.get("memory_id") and row["memory_id"] != event.get("memory_id"):
            return {"state": "unavailable"}
        return {
            "state": "current",
            "matches_selected_version": bool(event.get("after_hash"))
            and _hash(_fields(row)) == event["after_hash"],
            "record": row,
        }

    def observations(
        self,
        *,
        agent_id=None,
        memory_ids=None,
        user_id=_UNSET,
        application_id=None,
        limit=200,
        exclude_records=(),
    ):
        """Current records with unknown historical actors, not backfilled changes."""
        from .memory_provider.base import MemoryProvider, _observation_matches

        wanted = set(memory_ids or [])
        safe_limit = max(1, min(int(limit), 1000))
        rows = []
        query = getattr(self.provider, "query_memory_observations", None)
        for kind in MemoryType:
            if kind in {MemoryType.SHARED_MEMORY, MemoryType.MEMAGENT}:
                continue
            excluded = {
                str(identifier)
                for memory_type, identifier in exclude_records
                if str(getattr(memory_type, "value", memory_type)) == kind.value
            }
            scope = dict(
                agent_id=agent_id,
                memory_ids=wanted,
                user_id=user_id,
                application_id=application_id,
                exclude_ids=excluded,
            )
            candidates = (
                query(kind, **scope, limit=safe_limit - len(rows))
                if callable(query)
                else MemoryProvider.query_memory_observations(
                    self.provider, kind, **scope, limit=safe_limit - len(rows)
                )
            )
            for row in candidates or []:
                if not isinstance(row, dict):
                    continue
                owner = str(row.get("agent_id") or row.get("owner_agent_id") or "")
                if not _observation_matches(row, **scope):
                    continue
                rows.append(
                    {
                        "record_id": "observed-"
                        + _hash([kind.value, _record_id(row)])[:24],
                        "target_record_id": _record_id(row),
                        "memory_type": kind.value,
                        "agent_id": owner or None,
                        "memory_id": row.get("memory_id"),
                        "action": "observed",
                        "timestamp": row.get("timestamp") or row.get("created_at"),
                        "actor": "unknown",
                        "source": "current_record",
                        "parents": _parents(row),
                        "label": str(
                            row.get("name")
                            or row.get("title")
                            or row.get("tool_name")
                            or row.get("role")
                            or kind.value.replace("_", " ")
                        )[:160],
                        "history_gap": True,
                        "content_stored": False,
                    }
                )
                if len(rows) >= safe_limit:
                    return rows
        return rows
