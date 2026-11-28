# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Where a proactive agent keeps its suggestions and what became of them.

One record per agent and user, in the agent's own store, read and written by
its ID. It holds the admission state (when the agent may next be asked, how
often it has been today) and the suggestions with their outcomes, which the
agent reads back so it does not repeat itself and learns what gets accepted.
Like the learning and fine-tune records, it is kept under
``MemoryType.SHARED_MEMORY`` with an ID of its own, so it never appears in
recall.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, TypeVar

from ..enums import MemoryType

logger = logging.getLogger(__name__)

T = TypeVar("T")

RECORD_TYPE = "proactive_state"
MAX_EVENTS = 200

_locks: Dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()
#: Agents with no store keep their record here, for the life of the process.
_volatile: Dict[str, Dict[str, Any]] = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty() -> Dict[str, Any]:
    return {"state": {}, "events": []}


class ProactiveStore:
    def __init__(self, provider: Any, agent_id: str, user_id: Optional[str] = None):
        self.provider = provider
        self.agent_id = str(agent_id)
        self.user_id = None if user_id is None else str(user_id)
        digest = hashlib.sha256(
            (self.agent_id + "\x00" + (self.user_id or "")).encode()
        ).hexdigest()
        self.record_id = "proactive-" + digest[:40]
        #: A memory ID of its own for the deciding run, never one of the agent's.
        self.memory_id = "proactive-" + digest[:24]
        with _locks_guard:
            self._lock = _locks.setdefault(self.record_id, threading.RLock())

    def _load_with_raw(self) -> tuple[Dict[str, Any], Optional[str]]:
        if self.provider is None:
            return copy.deepcopy(_volatile.get(self.record_id) or _empty()), None
        try:
            row = self.provider.retrieve_by_id(self.record_id, MemoryType.SHARED_MEMORY)
        except Exception as exc:
            logger.debug("Proactive record could not be read: %s", exc)
            row = None
        if not isinstance(row, dict):
            return _empty(), None
        raw_content = row.get("content")
        raw_str = raw_content if isinstance(raw_content, str) else None
        content = raw_content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                content = None
        if (
            not isinstance(content, dict)
            or content.get("record_type") != RECORD_TYPE
            or content.get("agent_id") != self.agent_id
            or (content.get("user_id") or None) != self.user_id
        ):
            return _empty(), None
        return (
            {
                "state": dict(content.get("state") or {}),
                "events": [
                    e for e in content.get("events") or [] if isinstance(e, dict)
                ],
            },
            raw_str,
        )

    def load(self) -> Dict[str, Any]:
        data, _ = self._load_with_raw()
        return data

    def _payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        events = list(data.get("events") or [])[-MAX_EVENTS:]
        return {
            "record_type": RECORD_TYPE,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "state": dict(data.get("state") or {}),
            "events": events,
            "timestamp": utcnow(),
        }

    def save(self, data: Dict[str, Any]) -> None:
        payload = self._payload(data)
        if self.provider is None:
            _volatile[self.record_id] = copy.deepcopy(payload)
            return
        document = {
            "_id": self.record_id,
            "memory_id": self.record_id,
            "content": json.dumps(payload, ensure_ascii=False, default=str),
            "memory_type": MemoryType.SHARED_MEMORY.value,
            "scope": "private",
            "owner_agent_id": self.agent_id,
            "record_type": RECORD_TYPE,
            "agent_id": self.agent_id,
            "timestamp": payload["timestamp"],
        }
        if self.user_id is not None:
            document["user_id"] = self.user_id
        self.provider.store(document, MemoryType.SHARED_MEMORY)

    def update(self, change: Callable[[Dict[str, Any]], T]) -> T:
        """Read the record, let ``change`` alter it, save it, return what
        ``change`` returned. Uses atomic compare-and-swap when supported."""
        cas = (
            getattr(self.provider, "compare_and_swap_shared_memory", None)
            if self.provider is not None
            else None
        )
        with self._lock:
            for _attempt in range(6):
                data, raw_content = self._load_with_raw()
                result = change(data)
                if raw_content is not None and callable(cas):
                    serialized = json.dumps(
                        self._payload(data), ensure_ascii=False, default=str
                    )
                    try:
                        if cas(self.record_id, raw_content, serialized):
                            return result
                        continue
                    except NotImplementedError:
                        pass
                self.save(data)
                return result
            data, _ = self._load_with_raw()
            result = change(data)
            self.save(data)
            return result

    def forget(self) -> bool:
        with self._lock:
            if self.provider is None:
                return _volatile.pop(self.record_id, None) is not None
            try:
                return bool(
                    self.provider.delete_by_id(self.record_id, MemoryType.SHARED_MEMORY)
                )
            except Exception as exc:
                logger.debug("Proactive record could not be deleted: %s", exc)
                return False
