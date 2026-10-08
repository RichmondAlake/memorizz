"""Immutable inputs captured at the MemAgent model boundary after fitting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..enums.memory_type import MemoryType
from ..memory_provider.base import _UNSET
from .store import ObservabilityStore

RECORD_TYPE = "observability_context_snapshot"


class ContextSnapshots:
    def __init__(self, provider):
        self.provider = provider
        self.store = ObservabilityStore(provider)

    def capture(
        self,
        *,
        identity,
        span_id,
        messages,
        tools,
        model,
        window_tokens,
        iteration,
        stage,
        memory_evidence=None,
    ):
        from ..memagent.utils.prompt_budget import estimate_tokens

        record_id = "ctx-" + str(span_id)
        existing = None
        for method_name in ("retrieve_by_id", "retrieve_by_name"):
            try:
                row = getattr(self.provider, method_name)(
                    record_id, MemoryType.SHARED_MEMORY
                )
                if isinstance(row, dict):
                    existing = ObservabilityStore._payload(row)
                    break
            except (NotImplementedError, TypeError, ValueError):
                continue
        if existing:
            return existing
        # JSON copying freezes mutable message/tool lists before providers touch them.
        body = json.loads(
            json.dumps(
                {
                    "messages": messages,
                    "tools": tools or [],
                    "memory_evidence": memory_evidence or {},
                },
                ensure_ascii=False,
                default=str,
            )
        )
        payload = {
            **identity,
            "timestamp": identity.get("timestamp")
            or datetime.now(timezone.utc).isoformat(),
            "record_id": record_id,
            "record_type": RECORD_TYPE,
            "span_id": span_id,
            "model": model,
            "window_tokens": window_tokens,
            "iteration": iteration,
            "stage": stage,
            "estimated_tokens": sum(estimate_tokens(m) + 6 for m in body["messages"])
            + (estimate_tokens(body["tools"]) if body["tools"] else 0)
            + 16,
            "message_count": len(body["messages"]),
            "tool_count": len(body["tools"]),
            "capture_boundary": "memagent_model_input_after_prompt_fitting",
            **body,
        }
        return self.store._put(payload)

    def page(
        self,
        *,
        agent_id,
        memory_id=None,
        user_id=_UNSET,
        application_id=None,
        limit=200,
        cursor=None,
    ):
        page = self.provider.query_observability_records(
            MemoryType.SHARED_MEMORY,
            agent_ids=[agent_id] if memory_id is None else None,
            memory_ids=[memory_id] if memory_id is not None else None,
            record_type=RECORD_TYPE,
            user_id=user_id,
            application_id=application_id,
            limit=limit,
            cursor=cursor,
        )
        rows = []
        for row in page.get("items", []):
            payload = ObservabilityStore._payload(row)
            if not self._matches(payload, agent_id, memory_id, user_id, application_id):
                continue
            payload.setdefault(
                "timestamp", payload.get("created_at") or row.get("timestamp")
            )
            rows.append(
                {
                    k: v
                    for k, v in payload.items()
                    if k not in {"messages", "tools", "memory_evidence"}
                }
            )
        rows.sort(key=lambda r: (str(r.get("timestamp") or ""), str(r["record_id"])))
        return {"items": rows, "next_cursor": page.get("next_cursor")}

    def list(self, **filters):
        """A bounded metadata page; use page() for cursor-based traversal."""
        return self.page(**filters)["items"]

    @staticmethod
    def _matches(payload, agent_id, memory_id, user_id, application_id):
        return bool(
            payload
            and payload.get("record_type") == RECORD_TYPE
            and str(payload.get("agent_id")) == str(agent_id)
            and (memory_id is None or payload.get("memory_id") == memory_id)
            and (user_id is _UNSET or payload.get("user_id") == user_id)
            and (
                application_id is None
                or payload.get("application_id") == application_id
            )
        )

    def get(
        self,
        record_id,
        *,
        agent_id,
        memory_id=None,
        user_id=_UNSET,
        application_id=None,
    ) -> dict[str, Any] | None:
        payload = self.store._get(record_id)
        return (
            payload
            if self._matches(payload, agent_id, memory_id, user_id, application_id)
            else None
        )
