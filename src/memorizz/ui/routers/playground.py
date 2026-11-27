# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Playground pages, chat streaming, and thread APIs.

GET /playground (agent selector), GET /agents/{agent_id}/playground, POST
/agents/{agent_id}/playground/stream, POST /agents/{agent_id}/playground/compact,
GET /agents/{agent_id}/playground/thread, POST
/agents/{agent_id}/playground/config, the thread APIs, and the access an
agent's harness delegates may use from the playground (POST
/api/agents/{agent_id}/harness-access, GET .../harness-access/{grant_id}, POST
/api/agents/{agent_id}/harness-workspace), plus the thread-memory serializers,
loaders, and token-stat builders used only by these routes.

Agent instances are loaded per request via ``MemAgent.load`` — there is no
module-level agent cache, so lifetime semantics are identical to the inline
handlers this module replaced.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)

from .. import agent_status
from ..helpers import (
    SANDBOX_OFF,
    _agent_entity_memory_enabled,
    _agent_workflow_memory_enabled,
    _build_agent_nav_items,
    _build_agent_threads,
    _build_browser_control_config,
    _build_internet_provider_config,
    _build_memory_types_for_agent,
    _build_self_aware_config,
    _build_skills_marketplace_provider_config,
    _coerce_timestamp,
    _extract_agent_tools,
    _get_default_llm_model,
    _json_object,
    _list_agents,
    _load_agent_knowledge_base,
    _load_agent_last_run_map,
    _load_thread_messages,
    _normalize_browser_control_provider_name,
    _normalize_internet_provider_name,
    _normalize_llm_provider,
    _normalize_skills_marketplace_provider_name,
    _parse_bool,
    _parse_routing,
    _parse_self_aware_root_paths,
    _persist_mcp_configs_to_toolbox,
    _provider_supports_entity_memory,
    _resolve_sandbox_provider_config,
    _sandbox_is_off,
    _to_text,
    _validate_browser_control_choice,
    _validate_internet_provider_choice,
    _validate_sandbox_provider_choice,
    _validate_self_aware_config,
    _validate_skills_marketplace_provider_choice,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["playground"])

# The longest agent name the header field and the rename API accept.
AGENT_NAME_MAX = 80

# Messages served per thread request: the newest page, as the CLI shows.
THREAD_PAGE_SIZE = 200
THREAD_PAGE_MAX = 1000


def _serialize_toolbox_memory_item(document: Dict[str, Any]) -> Dict[str, str]:
    """Convert a toolbox document into a compact, UI-safe payload."""
    tool_identifier = _to_text(
        document.get("tool_id")
        or document.get("toolId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    tool_name = _to_text(
        document.get("name")
        or document.get("tool_name")
        or tool_identifier
        or "unnamed_toolbox_item"
    ).strip()
    tool_type = _to_text(document.get("tool_type") or document.get("type")).strip()
    description = _to_text(
        document.get("description") or document.get("docstring") or ""
    ).strip()
    signature = _to_text(document.get("signature") or "").strip()
    timestamp = _to_text(
        document.get("timestamp")
        or document.get("updated_at")
        or document.get("created_at")
        or document.get("createdAt")
        or ""
    ).strip()

    value_preview = ""
    if "parameters" in document and document.get("parameters") is not None:
        try:
            value_preview = json.dumps(document.get("parameters"), ensure_ascii=False)
        except Exception:
            value_preview = _to_text(document.get("parameters"))
    elif document.get("content") is not None:
        value_preview = _to_text(document.get("content"))

    return {
        "id": tool_identifier,
        "name": tool_name or "unnamed_toolbox_item",
        "tool_type": tool_type or "toolbox_item",
        "description": description,
        "signature": signature,
        "value_preview": value_preview,
        "timestamp": timestamp,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _normalize_workflow_steps(steps: Any) -> Dict[str, Any]:
    """Return workflow steps as a dictionary when possible."""
    if isinstance(steps, dict):
        return steps
    if isinstance(steps, list):
        normalized: Dict[str, Any] = {}
        for index, entry in enumerate(steps, start=1):
            key = f"Step {index}"
            normalized[key] = entry
        return normalized
    if isinstance(steps, str):
        text = steps.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            normalized = {}
            for index, entry in enumerate(parsed, start=1):
                normalized[f"Step {index}"] = entry
            return normalized
    return {}


def _serialize_workflow_memory_item(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a workflow document into a compact, UI-safe payload."""
    workflow_identifier = _to_text(
        document.get("workflow_id")
        or document.get("workflowId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    workflow_name = _to_text(
        document.get("name")
        or document.get("title")
        or workflow_identifier
        or "unnamed_workflow"
    ).strip()
    description = _to_text(document.get("description") or "").strip()
    user_query = _to_text(
        document.get("user_query") or document.get("userQuery") or ""
    ).strip()
    status = _to_text(document.get("status") or "").strip()
    outcome = _to_text(document.get("outcome") or "").strip()
    timestamp = _to_text(
        document.get("updated_at")
        or document.get("updatedAt")
        or document.get("created_at")
        or document.get("createdAt")
        or document.get("timestamp")
        or ""
    ).strip()

    steps = _normalize_workflow_steps(document.get("steps"))
    step_names = list(steps.keys())
    step_count = len(step_names)
    step_names_preview = step_names[:3]

    return {
        "id": workflow_identifier,
        "name": workflow_name or "unnamed_workflow",
        "description": description,
        "user_query": user_query,
        "status": status,
        "outcome": outcome,
        "timestamp": timestamp,
        "step_count": step_count,
        "step_names_preview": step_names_preview,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_memory(
    memory_type: Any,
    serialize: Any,
    label: str,
    agent_id: str,
    memory_id: str,
    limit: Optional[int] = None,
    *,
    require_memory_id: bool = True,
) -> List[Dict[str, Any]]:
    """Rows of one memory type relevant to the active agent thread, newest
    first. Rows of another agent or another thread are left out; with
    ``require_memory_id``, so are rows naming no thread when one is active."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        documents = provider.list_all(memory_type) or []
    except Exception as exc:
        logger.debug(
            "Failed to load %s memory for agent %s thread %s: %s",
            label,
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue
        if normalized_memory_id:
            if doc_memory_id and doc_memory_id != normalized_memory_id:
                continue
            if require_memory_id and not doc_memory_id:
                continue
        elif doc_memory_id:
            continue

        filtered.append(serialize(doc))

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _load_thread_memories(
    agent_id: str, memory_id: str
) -> Dict[str, List[Dict[str, Any]]]:
    """The thread-scoped memory the playground shows beside a conversation."""
    from ...enums.memory_type import MemoryType

    return {
        "toolbox": _load_thread_memory(
            MemoryType.TOOLBOX,
            _serialize_toolbox_memory_item,
            "toolbox",
            agent_id,
            memory_id,
            require_memory_id=False,
        ),
        "workflow": _load_thread_memory(
            MemoryType.WORKFLOW_MEMORY,
            _serialize_workflow_memory_item,
            "workflow",
            agent_id,
            memory_id,
        ),
        "entity": _load_thread_entity_memory(
            agent_id=agent_id, memory_id=memory_id, limit=None
        ),
        "summary": _load_thread_memory(
            MemoryType.SUMMARIES,
            _serialize_summary_memory_item,
            "summary",
            agent_id,
            memory_id,
        ),
        "tool_log": _load_thread_tool_log_memory(memory_id=memory_id, limit=20),
    }


def _serialize_skill_memory_item(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a skillbox document into a compact, UI-safe payload."""
    skill_identifier = _to_text(
        document.get("skill_id")
        or document.get("skillId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    skill_name = _to_text(
        document.get("name") or skill_identifier or "unnamed_skill"
    ).strip()
    description = _to_text(document.get("description") or "").strip()
    status = _to_text(document.get("status") or "").strip()
    injection_role = _to_text(document.get("injection_role") or "user").strip()
    if injection_role not in {"user", "developer"}:
        injection_role = "user"
    version = document.get("version")
    stats = document.get("stats")
    if not isinstance(stats, dict):
        stats = {}

    def _stat_count(key: str) -> int:
        try:
            return int(stats.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "id": skill_identifier,
        "name": skill_name or "unnamed_skill",
        "description": description,
        "status": status,
        "injection_role": injection_role,
        "version": version,
        "activations": _stat_count("activations"),
        "successes": _stat_count("successes"),
        "failures": _stat_count("failures"),
        "deviations": _stat_count("deviations"),
        "similarity": None,
        "demotion_reason": _to_text(document.get("demotion_reason") or "").strip(),
        "promoted_at": _to_text(document.get("promoted_at") or "").strip(),
    }


def _load_thread_skill_memory(
    agent_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load learned-skill rows for the active agent.

    Skills are agent-scoped rather than thread-scoped, so no memory_id filter
    is applied.
    """
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ...enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.SKILLBOX) or []
    except Exception as exc:
        logger.debug(
            "Failed to load skill memory for agent %s: %s",
            normalized_agent_id,
            exc,
        )
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        filtered.append(_serialize_skill_memory_item(doc))

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("promoted_at")) or 0.0,
        reverse=True,
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _serialize_entity_memory_item(document: Dict[str, Any]) -> Dict[str, str]:
    """Convert an entity-memory document into a compact, UI-safe payload."""
    entity_identifier = _to_text(
        document.get("entity_id")
        or document.get("entityId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    entity_name = _to_text(
        document.get("name")
        or document.get("entity_name")
        or entity_identifier
        or "unnamed_entity"
    ).strip()
    entity_type = _to_text(
        document.get("entity_type") or document.get("entityType") or ""
    ).strip()
    timestamp = _to_text(
        document.get("updated_at")
        or document.get("updatedAt")
        or document.get("created_at")
        or document.get("createdAt")
        or ""
    ).strip()

    raw_attributes = document.get("attributes")
    attributes_preview = ""
    attribute_pairs: List[str] = []
    attributes_obj = raw_attributes
    if isinstance(attributes_obj, str):
        try:
            parsed = json.loads(attributes_obj)
        except Exception:
            parsed = attributes_obj
        attributes_obj = parsed

    if isinstance(attributes_obj, dict):
        for key, value in attributes_obj.items():
            key_text = _to_text(key).strip()
            if not key_text:
                continue
            value_text = _to_text(value).strip()
            attribute_pairs.append(f"{key_text}: {value_text}")
    elif isinstance(attributes_obj, list):
        for entry in attributes_obj:
            if not isinstance(entry, dict):
                continue
            key_text = _to_text(entry.get("name") or entry.get("key")).strip()
            if not key_text:
                continue
            value_text = _to_text(entry.get("value")).strip()
            attribute_pairs.append(f"{key_text}: {value_text}")

    if attribute_pairs:
        attributes_preview = ", ".join(attribute_pairs[:6])
    if len(attributes_preview) > 260:
        attributes_preview = f"{attributes_preview[:257]}..."

    from ..memory_view import vector_note

    note = vector_note(document)
    return {
        "id": entity_identifier,
        "name": entity_name or "unnamed_entity",
        "entity_type": entity_type,
        "timestamp": timestamp,
        "attributes_preview": attributes_preview,
        # "partial vector" / "no vector" when search by meaning may miss it.
        "vector_note": note["label"],
        "vector_note_title": note["title"],
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_entity_memory(
    agent_id: str, memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, str]]:
    """Load entity-memory rows relevant to the active agent thread."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_agent_id:
        return []

    try:
        from ...enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.ENTITY_MEMORY) or []
    except Exception as exc:
        logger.debug(
            "Failed to load entity memory for agent %s thread %s: %s",
            normalized_agent_id,
            normalized_memory_id,
            exc,
        )
        return []

    scoped: List[Dict[str, str]] = []
    fallback_agent_scoped: List[Dict[str, str]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        doc_agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if doc_agent_id and doc_agent_id != normalized_agent_id:
            continue

        doc_memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()
        if not doc_agent_id and not doc_memory_id:
            continue

        serialized = _serialize_entity_memory_item(doc)
        if normalized_memory_id:
            if doc_memory_id == normalized_memory_id:
                scoped.append(serialized)
                continue
            if not doc_memory_id and doc_agent_id == normalized_agent_id:
                fallback_agent_scoped.append(serialized)
            continue

        # No explicit thread selected: show all agent-scoped entity rows.
        if doc_memory_id or doc_agent_id == normalized_agent_id:
            scoped.append(serialized)

    filtered = scoped if scoped else fallback_agent_scoped

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0, reverse=True
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _serialize_summary_memory_item(document: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a summaries-memory document into a compact, UI-safe payload."""
    summary_identifier = _to_text(
        document.get("summary_id")
        or document.get("summaryId")
        or document.get("_id")
        or document.get("id")
    ).strip()
    summary_type = _to_text(
        document.get("summary_type") or document.get("summaryType") or "summary"
    ).strip()
    timestamp = _to_text(
        document.get("created_at")
        or document.get("createdAt")
        or document.get("updated_at")
        or document.get("updatedAt")
        or ""
    ).strip()
    content = _to_text(document.get("content") or "").strip()
    if len(content) > 520:
        content = f"{content[:517]}..."

    return {
        "id": summary_identifier,
        "summary_type": summary_type or "summary",
        "timestamp": timestamp,
        "content": content,
        "memory_id": _to_text(
            document.get("memory_id") or document.get("memoryId")
        ).strip(),
    }


def _load_thread_tool_log_memory(
    memory_id: str, limit: Optional[int] = 20
) -> List[Dict[str, Any]]:
    """Load tool log entries for the active thread."""
    provider = _state.get("provider")
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_memory_id:
        return []

    try:
        from ...enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.TOOL_LOG) or []
    except Exception as exc:
        logger.debug(
            "Failed to load tool log memory for thread %s: %s",
            normalized_memory_id,
            exc,
        )
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue

        content = doc.get("content") or doc
        if isinstance(content, dict):
            doc_memory_id = _to_text(
                content.get("memory_id") or doc.get("memory_id") or ""
            ).strip()
        else:
            doc_memory_id = _to_text(doc.get("memory_id", "")).strip()

        if doc_memory_id and doc_memory_id != normalized_memory_id:
            continue

        # Serialize the tool log entry
        if isinstance(content, dict):
            tool_name = _to_text(content.get("tool_name", "")).strip() or "unknown"
            arguments = _to_text(content.get("arguments", "")).strip()
            result = _to_text(content.get("result", "")).strip()
            success = content.get("success", True)
            error = _to_text(content.get("error", "")).strip() or None
            timestamp = _to_text(content.get("timestamp", "")).strip()
            tool_log_id = _to_text(
                content.get("tool_log_id") or doc.get("id", "")
            ).strip()
        else:
            continue

        result_preview = result[:200] + "..." if len(result) > 200 else result

        filtered.append(
            {
                "tool_log_id": tool_log_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result_preview": result_preview,
                "success": success,
                "error": error,
                "timestamp": timestamp,
            }
        )

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0,
        reverse=True,
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


def _load_thread_trace_bundles(
    *, agent_id: str, memory_id: str, limit: Optional[int] = 100
) -> List[Dict[str, Any]]:
    """Load private observability bundles for Playground trace replay."""
    provider = _state.get("provider")
    normalized_agent_id = _to_text(agent_id).strip()
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_memory_id:
        return []

    from ...enums.memory_type import MemoryType

    try:
        query = getattr(provider, "query_observability_records", None)
        if callable(query):
            page = query(
                MemoryType.SHARED_MEMORY,
                agent_ids=[normalized_agent_id] if normalized_agent_id else None,
                memory_ids=[normalized_memory_id],
                record_type="observability_trace_bundle",
                limit=limit or 1000,
            )
            documents = page.get("items") or []
        else:
            documents = provider.list_all(MemoryType.SHARED_MEMORY) or []
    except Exception as exc:
        logger.debug("Failed to load private trace bundles: %s", exc)
        return []

    rows: List[Dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        content = _to_text(document.get("content")).strip()
        try:
            payload = json.loads(content)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("type") != "trace_bundle":
            continue
        payload_memory_id = _to_text(
            document.get("trace_memory_id") or payload.get("memory_id")
        ).strip()
        payload_agent_id = _to_text(
            document.get("agent_id") or payload.get("agent_id")
        ).strip()
        if payload_memory_id != normalized_memory_id:
            continue
        if normalized_agent_id and payload_agent_id != normalized_agent_id:
            continue
        rows.append(
            {
                **document,
                "role": "tool",
                "content": content,
                "memory_id": payload_memory_id,
                "timestamp": document.get("timestamp")
                or payload.get("timestamp")
                or document.get("updated_at"),
            }
        )

    rows.sort(key=lambda row: _coerce_timestamp(row.get("timestamp")) or 0.0)
    if limit and limit > 0:
        return rows[-limit:]
    return rows


def _serialize_thread_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert thread message payload into JSON-serializable structure."""
    if not isinstance(message, dict):
        return {
            "role": "system",
            "content": _to_text(message),
            "timestamp": "",
            "memory_id": "",
        }

    timestamp_value = message.get("timestamp")
    timestamp_text = _to_text(timestamp_value).strip() if timestamp_value else ""

    role = _to_text(message.get("role")).strip().lower() or "system"
    content = _to_text(message.get("content") or message.get("text", ""))
    serialized = {
        "role": role,
        "content": content,
        "timestamp": timestamp_text,
        "memory_id": _to_text(message.get("memory_id")).strip(),
    }

    if role == "tool":
        trace_events = _parse_trace_bundle_payload(content)
        if trace_events is not None:
            serialized["message_type"] = "trace_bundle"
            serialized["trace_events"] = trace_events
            serialized["content"] = f"Trace events ({len(trace_events)})"

    return serialized


def _parse_trace_bundle_payload(raw_content: Any) -> Optional[List[Dict[str, str]]]:
    """Decode persisted trace-bundle content from tool messages."""
    content = _to_text(raw_content).strip()
    if not content:
        return None

    try:
        payload = json.loads(content)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if _to_text(payload.get("type")).strip().lower() != "trace_bundle":
        return None

    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return []

    events: List[Dict[str, str]] = []
    for event in raw_events:
        if not isinstance(event, dict):
            continue

        trace_kind = _to_text(
            event.get("trace_kind") or event.get("kind") or "trace"
        ).strip()
        title = _to_text(event.get("title") or "Trace").strip()
        event_content = _to_text(
            event.get("content") or event.get("message") or ""
        ).strip()
        trace_id = _to_text(event.get("trace_id")).strip()

        if not title and not event_content:
            continue

        serialized_event = {
            "trace_kind": trace_kind.lower() or "trace",
            "title": title or "Trace",
            "content": event_content,
        }
        if trace_id:
            serialized_event["trace_id"] = trace_id
        for key in ("root_trace_id", "turn_id", "thread_id"):
            value = _to_text(event.get(key)).strip()
            if value:
                serialized_event[key] = value
        # What the step found, for the reply to show (pages, a table).
        if isinstance(event.get("display"), dict) and event["display"].get("kind"):
            serialized_event["display"] = event["display"]
        events.append(serialized_event)

    return events


def _agent_context_window(agent: Any) -> int:
    """The budget the agent really uses, not a generic 128k guess."""
    from ...memagent.persistence import _saved_context_cap

    live_window = getattr(agent, "_context_window_tokens", None)
    if type(live_window) is int and live_window > 0:
        return live_window
    # A stored record's cap only counts where MemAgent.load would restore it.
    cap = _saved_context_cap(agent, getattr(agent, "llm_config", None))
    if type(cap) is int and cap > 0:
        return cap
    model = getattr(agent, "model", None)
    if model is None or not hasattr(model, "get_context_window_tokens"):
        # A stored record: build its provider the way MemAgent.load would.
        llm_config = getattr(agent, "llm_config", None)
        if isinstance(llm_config, dict) and llm_config.get("provider"):
            try:
                from ...llms.llm_factory import create_llm_provider

                model = create_llm_provider(dict(llm_config))
            except Exception:
                model = None
    if model is not None and hasattr(model, "get_context_window_tokens"):
        try:
            value = model.get_context_window_tokens()
            if type(value) is int and value > 0:
                return value
        except Exception:
            pass
    return 128000


def _model_label(agent: Any) -> Optional[str]:
    model = getattr(agent, "model", None)
    name = getattr(model, "model", None) if model is not None else None
    if isinstance(name, str):
        return name
    config = getattr(agent, "llm_config", None)
    return (
        str(config.get("model"))
        if isinstance(config, dict) and config.get("model")
        else None
    )


def _model_max_context(live: Any) -> Optional[int]:
    """The model's own context length, when the provider can report it."""
    model = getattr(live, "model", None)
    if type(model).__name__ != "OllamaLLM":
        return None
    try:
        from ...llms.ollama import _model_context_length

        return _model_context_length(model.client, model._host, model.model)
    except Exception:
        return None


def _live_agent(agent: Any) -> Any:
    """A MemAgent for a stored record, so estimates match what it sends."""
    if agent is None or hasattr(agent, "_build_system_prompt"):
        return agent
    agent_id = getattr(agent, "agent_id", None)
    provider = _state.get("provider")
    if not agent_id or provider is None:
        return None
    try:
        from ...memagent import MemAgent

        return MemAgent.load(agent_id, memory_provider=provider)
    except Exception as exc:
        logger.debug("Could not load agent %s for context stats: %s", agent_id, exc)
        return None


def _build_token_stats(
    agent,
    context_window: List[Dict[str, Any]],
    toolbox_memory: Optional[List[Dict[str, Any]]] = None,
    workflow_memory: Optional[List[Dict[str, Any]]] = None,
    entity_memory: Optional[List[Dict[str, Any]]] = None,
    summary_memory: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Estimate the next request the way the agent builds it.

    Counts the system prompt, tool schemas, memory context and the history
    the agent would send after trimming, using the agent's own estimator.
    Trace records shown in the thread and summarized messages are never sent,
    so they are not counted; the toolbox rows are copies of the tools.
    """
    from ...conversation_history import is_trace_bundle_entry
    from ...enums.memory_type import MemoryType
    from ...memagent.managers.memory_manager import MemoryManager
    from ...memagent.utils.prompt_budget import estimate_tokens

    live = _live_agent(agent)
    window = _agent_context_window(live if live is not None else agent)

    rows = [
        row
        for row in context_window or []
        if isinstance(row, dict)
        and row.get("role") in {"user", "assistant"}
        and not is_trace_bundle_entry(row)
        and not MemoryManager._is_summarized_message(row)
    ]
    if live is not None:
        system_prompt = _to_text(live._build_system_prompt())
        try:
            sent = live._prepare_history_messages(rows, system_prompt, "")
        except Exception:
            sent = rows
    else:
        system_prompt = str(getattr(agent, "instruction", "") or "")
        sent = rows[-40:]

    def history_cost(messages: List[Dict[str, Any]]) -> int:
        return sum(
            estimate_tokens(_to_text(message.get("content") or "")) + 6
            for message in messages
        )

    def memory_cost(items: Optional[List[Dict[str, Any]]]) -> int:
        return estimate_tokens(items) if items else 0

    system_tokens = estimate_tokens(system_prompt)
    tools_tokens = live._estimated_tool_tokens() if live is not None else 0
    memory_tokens = (
        memory_cost(entity_memory)
        + memory_cost(summary_memory)
        + memory_cost(workflow_memory)
    )
    history_tokens = history_cost(sent)
    total_tokens = system_tokens + tools_tokens + memory_tokens + history_tokens
    projected_tokens = total_tokens - history_tokens + history_cost(rows)

    policy = getattr(live, "context_policy", None)
    compact_at = int(getattr(policy, "compact_at", 0) or 0)
    compaction_available = bool(
        live is not None and MemoryType.SUMMARIES in live.active_memory_types
    )

    def share(tokens: int) -> float:
        return round(tokens * 100 / window, 2) if window > 0 else 0

    return {
        "total_tokens": total_tokens,
        "context_window_tokens": window,
        "percentage_used": share(total_tokens),
        "projected_tokens": projected_tokens,
        "projected_percentage": share(projected_tokens),
        # What compaction cannot shrink: the prompt without any history.
        "fixed_percentage": share(total_tokens - history_tokens),
        "message_count": len(sent),
        "history_messages": sent,
        "thread_message_count": len(rows),
        "compact_at": compact_at,
        "compaction_available": compaction_available,
        "model_name": _model_label(live if live is not None else agent),
        "model_context_tokens": _model_max_context(live),
        "composition": {
            "system_prompt": system_tokens,
            "tools": tools_tokens,
            "memory": memory_tokens,
            "history": history_tokens,
        },
    }


def _entity_memory_status_error(agent: Any) -> str:
    """Return a user-facing warning when entity memory is configured but unavailable."""
    if not agent or not _agent_entity_memory_enabled(agent):
        return ""
    provider = _state.get("provider")
    if _provider_supports_entity_memory(provider):
        return ""
    return (
        "Entity memory is enabled in this agent configuration, but the active memory "
        "provider does not support ENTITY_MEMORY. "
        "The entity_memory_lookup/entity_memory_upsert tools will not be available."
    )


def _parse_skill_paths_json(
    value: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Parse skill path JSON payload from Playground config."""
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return [], None
    try:
        parsed = json.loads(raw)
    except Exception as exc:
        return None, f"Invalid skill paths JSON: {exc}"
    if not isinstance(parsed, list):
        return None, "Skill paths JSON must be an array."

    paths: List[str] = []
    seen = set()
    for item in parsed:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        paths.append(path)
        seen.add(path)
    return paths, None


def _harness_delegates(agent: Any) -> List[str]:
    """Labels of the delegates an agent hands work to that run their turns
    on a harness ("Codex researcher (codex)"); empty when it delegates
    nothing to harnesses."""
    from ...metaharness.catalog import agent_name_of, delegates_work, harness_backing

    if agent is None or not delegates_work(agent):
        return []
    provider = _state.get("provider")
    labels = []
    for entry in getattr(agent, "delegates", None) or []:
        delegate = entry
        if isinstance(entry, str):
            try:
                delegate = provider.retrieve_memagent(entry) if provider else None
            except Exception:
                delegate = None
        backing = harness_backing(delegate) if delegate is not None else None
        if backing:
            labels.append(f"{agent_name_of(delegate)} ({backing['harness']})")
    return labels


def _harness_service():
    from ..state import get_meta_harness

    try:
        return get_meta_harness()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _mark_harness_turns(
    messages: List[Dict[str, Any]], agent_id: str, memory_id: Optional[str]
) -> None:
    """Say which stored answers a harness gave, so the page can link each to
    its run. An answer is matched to the harness run of this agent and
    conversation that was asked the question before it."""
    if not memory_id or not messages:
        return
    try:
        runs = _harness_service().list_runs(limit=300)
    except Exception:
        return
    asked: Dict[str, Dict[str, Any]] = {}
    for run in runs:  # newest first: the first seen for a question is kept
        task = run.get("task") or {}
        if (
            task.get("memory_id") != memory_id
            or task.get("agent_id") != agent_id
            or task.get("mode") != "runtime"
        ):
            continue
        asked.setdefault(_to_text(task.get("task")).strip(), run)
    if not asked:
        return
    question = None
    for message in messages:
        role = _to_text(message.get("role")).lower()
        if role == "user":
            question = _to_text(message.get("content")).strip()
        elif role == "assistant" and question in asked:
            run = asked.pop(question)
            result = run.get("result") or {}
            message["harness_run"] = {
                "run_id": run.get("run_id"),
                "harness": run.get("harness"),
                "status": run.get("status"),
                "latency_ms": result.get("latency_ms"),
                "cost_usd": result.get("cost_usd"),
            }
            question = None


# The agent's own runtime is "memagent"; it cannot answer for itself.
_NOT_ANSWERING = frozenset({"memagent"})


# Probing every harness runs its CLI; the list is kept for a short while.
_ANSWER_HARNESSES: Dict[str, Any] = {"at": 0.0, "rows": None}
ANSWER_HARNESSES_SECONDS = 30.0


def _answer_harnesses() -> List[Dict[str, Any]]:
    """The harnesses a playground conversation can be answered by, ready
    ones first, each with the models it offers and why it is not ready."""
    import time

    from ...metaharness.catalog import HARNESS_LABELS

    kept = _ANSWER_HARNESSES
    if (
        kept["rows"] is not None
        and time.monotonic() - kept["at"] < ANSWER_HARNESSES_SECONDS
    ):
        return [dict(row) for row in kept["rows"]]
    try:
        rows = _harness_service().list_harnesses()
    except Exception:
        return []
    found = []
    for row in rows:
        name = str(row.get("name") or "")
        if not name or name in _NOT_ANSWERING:
            continue
        ready = bool(row.get("ready"))
        found.append(
            {
                "name": name,
                "label": HARNESS_LABELS.get(name, name),
                "ready": ready,
                "models": [str(model) for model in row.get("models") or []][:40],
                "why": ""
                if ready
                else str(row.get("remediation") or row.get("error") or "Not ready"),
            }
        )
    found.sort(key=lambda item: (not item["ready"], item["label"].lower()))
    kept.update(at=time.monotonic(), rows=[dict(row) for row in found])
    return found


def _check_answer_harness(harness: str) -> None:
    """Refuse a harness that is unknown or not ready, before the turn starts."""
    from ...metaharness.catalog import HARNESS_LABELS

    adapter = (
        None
        if harness in _NOT_ANSWERING
        else getattr(_harness_service(), "adapters", {}).get(harness)
    )
    if adapter is None:
        raise HTTPException(status_code=400, detail=f"Unknown harness: {harness}")
    state = adapter.probe().to_dict()
    if not state.get("ready"):
        why = state.get("remediation") or state.get("error") or "it is not ready"
        raise HTTPException(
            status_code=409,
            detail=f"{HARNESS_LABELS.get(harness, harness)} cannot answer: {why}",
        )


def _answer_on_harness(agent: Any, harness: str, model: Optional[str]) -> None:
    """Make this loaded agent answer its next turn through ``harness``.

    Only the instance for this request changes. MemoRizz still supplies the
    conversation's memory, the trace and approvals."""
    config = dict(getattr(agent, "harness_config", None) or {})
    if model:
        config["model"] = model
    else:
        # The agent's own model name means nothing to a harness.
        config.pop("model", None)
    agent.with_meta_harness(
        _harness_service(), mode="runtime", default_harness=harness, config=config
    )


# How long a harness turn is followed after its page left.
DETACHED_TURN_SECONDS = 45 * 60


def _follow_detached(
    stream: Any, agent_id: str, query: str, started: Any, progress: Any, run_ids: set
) -> None:
    """Keep a harness turn going after its page left, and publish it as a
    live turn so the page (or another one) shows it again and gets the
    answer. The turn is recorded in the conversation as usual when it ends."""
    import threading
    import time

    from ...live_turns import LiveTurn

    def follow() -> None:
        live = LiveTurn(agent_id, query, source="playground")
        deadline = time.monotonic() + DETACHED_TURN_SECONDS
        try:
            live.event(dict(started))
            if progress is not None:
                live.event(dict(progress))
            while time.monotonic() < deadline:
                try:
                    event = stream.poll(timeout=0.5)
                except StopIteration:
                    break
                if event is None:
                    live.tick()
                    continue
                run_ids.add(event["run_id"])
                agent_status.note_event(agent_id, event)
                live.event(dict(event))
        except Exception:
            logger.debug(
                "A detached harness turn stopped being followed", exc_info=True
            )
        finally:
            live.close()
            stream.close()
            for run_id in run_ids:
                agent_status.end_run(agent_id, run_id)

    threading.Thread(target=follow, name="playground-harness-turn", daemon=True).start()


def _harness_parent(
    service: Any, agent_id: str, grant_id: str, workspace: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """What a playground turn gives its harness delegates: the access grant
    the page holds, else the conversation's own folder. Returns the parent
    and a warning for the page when the grant or folder can't be used."""
    warning = None
    if grant_id:
        grant = service.delegate_access(grant_id, agent_id=agent_id)
        if grant is not None:
            return grant, None
        warning = (
            "The access you allowed for harness work has expired; "
            "allow it again in the inspector."
        )
    if workspace:
        from ...metaharness.security import resolve_workspace

        try:
            if service.is_scratch_workspace(workspace):
                folder = str(resolve_workspace(workspace))
            else:
                folder = str(
                    resolve_workspace(workspace, service.allowed_workspace_roots)
                )
            return {"workspace": folder}, warning
        except Exception as exc:
            warning = warning or f"That folder can't be used for harness work: {exc}"
    return None, warning


def _recall_scope(agent: Any) -> str:
    """How far an agent's conversations reach each other: "across",
    "separate" (each on its own) or "sealed" (nothing crosses, either way)."""
    policy = getattr(agent, "retrieval_policy", None) if agent else None
    if not isinstance(policy, dict):
        policy = policy.to_dict() if hasattr(policy, "to_dict") else {}
    if policy.get("conversation_scope") in ("thread", "disabled"):
        return "sealed" if policy.get("entity_scope") == "thread" else "separate"
    return "across"


def _playground_recall_policy(agent_instance: Any, memory_id: Optional[str]) -> Any:
    """The agent's recall policy with its other conversations to read from,
    or None to leave the policy as it is (sealed, thread-scoped, or no
    other conversation yet)."""
    from dataclasses import replace

    policy = getattr(agent_instance, "retrieval_policy", None)
    if getattr(policy, "conversation_scope", None) != "memory":
        return None
    current = _to_text(memory_id).strip()
    # Newest conversations first; the policy caps how many it will read.
    others = [
        _to_text(item).strip()
        for item in reversed(getattr(agent_instance, "memory_ids", None) or [])
    ]
    others = [item for item in dict.fromkeys(others) if item and item != current]
    if not others:
        return None
    return replace(policy, also_recall_from=tuple(others))


# The templates offered on a first run, in this order when they exist.
FIRST_RUN_TEMPLATES = ("research", "writing", "support", "productivity")


def _first_run() -> Dict[str, Any]:
    """What the playground shows a store with no agent: the model a first
    agent gets, a few templates and the steps that follow."""
    from .. import dashboard as dash
    from ..helpers import first_agent_model

    try:
        from ..agent_templates import template_cards

        cards = {card["key"]: card for card in template_cards() if card["builtin"]}
    except Exception:
        logger.debug("Agent templates unavailable for the first run", exc_info=True)
        cards = {}
    return {
        "model": first_agent_model(),
        "templates": [cards[key] for key in FIRST_RUN_TEMPLATES if key in cards],
        # The same five steps the Dashboard lists; none is done yet.
        "steps": dash.getting_started({}),
        "read_only": bool(_state.get("read_only")),
    }


@router.get("/playground", response_class=HTMLResponse)
async def playground_index(request: Request):
    """Show the playground page with agent selector."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    from datetime import timezone

    from .. import dashboard as dash
    from ..execution_view import build_playground_launcher

    agents = []
    last_run_by_agent: Dict[str, float] = {}
    try:
        agents = _list_agents()
        # The sidebar already reads this map; compute it once for both. It
        # comes from conversation memory, never from a trace scan.
        last_run_by_agent = _load_agent_last_run_map(agents)
    except Exception as e:
        logger.error(f"Failed to list agents for playground: {e}")

    now = datetime.now(timezone.utc)
    launcher = build_playground_launcher(
        agents,
        last_run_by_agent=last_run_by_agent,
        now=now,
        default_model=lambda provider: _get_default_llm_model(
            _normalize_llm_provider(provider or "openai")
        ),
    )
    return templates.TemplateResponse(
        "playground_select.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents": agents,
            "launcher": launcher,
            # A store with no agent yet: a first message makes the first one.
            "first_run": None if agents else _first_run(),
            "generated_at": now,
            **dash.template_helpers(),
            "active_page": "playground",
            "agents_nav": _build_agent_nav_items(
                agents=agents, last_run_by_agent=last_run_by_agent
            ),
        },
    )


@router.get("/agents/{agent_id}/playground", response_class=HTMLResponse)
async def agent_playground(
    request: Request,
    agent_id: str,
    config_error: Optional[str] = None,
    memory_id: Optional[str] = None,
):
    """Show the playground page for interacting with an agent."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)

    agent = None
    error = config_error
    context_window: List[Dict[str, Any]] = []
    toolbox_memory: List[Dict[str, Any]] = []
    workflow_memory: List[Dict[str, Any]] = []
    skill_memory: List[Dict[str, Any]] = []
    entity_memory: List[Dict[str, Any]] = []
    summary_memory: List[Dict[str, Any]] = []
    tool_log_memory: List[Dict[str, Any]] = []
    knowledge_base_memory: List[Dict[str, Any]] = []
    threads: List[Dict[str, Any]] = []
    default_memory_id = ""
    enable_entity_memory = False
    enable_workflow_memory = False
    entity_memory_status_error = ""

    try:
        agent = _state["provider"].retrieve_memagent(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        threads = _build_agent_threads(agent)
        if threads:
            default_memory_id = str(threads[0].get("memory_id", "") or "")
        else:
            memory_ids = getattr(agent, "memory_ids", None) or []
            if isinstance(memory_ids, list) and memory_ids:
                default_memory_id = _to_text(memory_ids[0]).strip()

        if memory_id:
            available = set(getattr(agent, "memory_ids", None) or []) | {
                str(thread.get("memory_id", "")) for thread in threads
            }
            if memory_id not in available:
                raise HTTPException(404, "Conversation not found")
            default_memory_id = memory_id

        if default_memory_id:
            context_window = _load_thread_messages(default_memory_id, limit=None)
            context_window.extend(
                _load_thread_trace_bundles(
                    agent_id=agent_id,
                    memory_id=default_memory_id,
                    limit=100,
                )
            )
            context_window.sort(
                key=lambda row: _coerce_timestamp(row.get("timestamp")) or 0.0
            )
            thread_memory = _load_thread_memories(agent_id, default_memory_id)
            toolbox_memory = thread_memory["toolbox"]
            workflow_memory = thread_memory["workflow"]
            entity_memory = thread_memory["entity"]
            summary_memory = thread_memory["summary"]
            tool_log_memory = thread_memory["tool_log"]
        # Knowledge base entries are agent-scoped, not thread-scoped.
        knowledge_base_memory = _load_agent_knowledge_base(agent)
        # Learned skills are agent-scoped, not thread-scoped.
        skill_memory = _load_thread_skill_memory(agent_id)
        enable_entity_memory = _agent_entity_memory_enabled(agent)
        enable_workflow_memory = _agent_workflow_memory_enabled(agent)
        entity_memory_status_error = _entity_memory_status_error(agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load agent {agent_id}: {e}")
        error = str(e)

    # Build token stats for context panel
    token_stats = (
        _build_token_stats(
            agent,
            context_window,
            toolbox_memory=toolbox_memory,
            workflow_memory=workflow_memory,
            entity_memory=entity_memory,
            summary_memory=summary_memory,
        )
        if agent
        else None
    )

    # Extract editable config fields
    llm_config = getattr(agent, "llm_config", {}) or {} if agent else {}
    llm_provider = _normalize_llm_provider(llm_config.get("provider", "openai"))
    llm_model = (
        llm_config.get("model")
        or llm_config.get("deployment_name")
        or _get_default_llm_model(llm_provider)
    )
    llm_context_window = llm_config.get("context_window_tokens")
    if type(llm_context_window) is not int or llm_context_window <= 0:
        llm_context_window = ""
    instruction = getattr(agent, "instruction", "") or "" if agent else ""
    max_steps = getattr(agent, "max_steps", 20) if agent else 20

    persona = getattr(agent, "persona", None) if agent else None
    if isinstance(persona, dict):
        persona_name = persona.get("name", "")
        persona_role = persona.get("role", "")
        persona_goals = persona.get("goals", "")
        persona_background = persona.get("background", "")
    else:
        persona_name = getattr(persona, "name", "") if persona else ""
        persona_role = getattr(persona, "role", "") if persona else ""
        persona_goals = getattr(persona, "goals", "") if persona else ""
        persona_background = getattr(persona, "background", "") if persona else ""

    # Sandbox provider: per-agent override or global default
    sandbox_provider = getattr(agent, "sandbox_provider", None) if agent else None
    if isinstance(sandbox_provider, dict):
        sandbox_provider = sandbox_provider.get("provider", "")
    # With nothing saved the agent follows the console's default; "none"
    # turns code execution off for this agent whatever the default is.
    sandbox_default = os.environ.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "").strip()
    sandbox_choice = _to_text(sandbox_provider or "").strip().lower()
    if not sandbox_choice and not sandbox_default:
        sandbox_choice = SANDBOX_OFF
    sandbox_effective = (
        "" if sandbox_choice == SANDBOX_OFF else sandbox_choice or sandbox_default
    )
    sandbox_provider = sandbox_choice
    sandbox_status_error = _validate_sandbox_provider_choice(sandbox_effective)

    browser_control_config = getattr(agent, "browser_control", None) if agent else None
    browser_control_provider = (
        _normalize_browser_control_provider_name(browser_control_config)
        or _normalize_browser_control_provider_name(
            os.environ.get("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "")
        )
        or ""
    )
    browser_control_config = _build_browser_control_config(
        browser_control_provider,
        browser_control_config if isinstance(browser_control_config, dict) else None,
    )
    browser_control_status_error = _validate_browser_control_choice(
        browser_control_provider, browser_control_config
    )

    internet_provider = (
        _normalize_internet_provider_name(
            getattr(agent, "internet_access_provider", None) if agent else None
        )
        or _normalize_internet_provider_name(
            os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER", "")
        )
        or ""
    )
    internet_provider_config = (
        getattr(agent, "internet_access_config", None) if agent else None
    )
    internet_status_error = _validate_internet_provider_choice(
        internet_provider,
        internet_provider_config,
    )

    skills_marketplace_provider = (
        _normalize_skills_marketplace_provider_name(
            getattr(agent, "skills_marketplace_provider", None) if agent else None
        )
        or _normalize_skills_marketplace_provider_name(
            os.environ.get("MEMORIZZ_DEFAULT_SKILLS_MARKETPLACE_PROVIDER", "")
        )
        or ""
    )
    skills_marketplace_provider_config = (
        getattr(agent, "skills_marketplace_config", None) if agent else None
    )
    skills_marketplace_status_error = _validate_skills_marketplace_provider_choice(
        skills_marketplace_provider,
        skills_marketplace_provider_config,
    )

    skill_paths = getattr(agent, "skill_paths", None) if agent else None
    if isinstance(skill_paths, str):
        skill_paths = [skill_paths]
    if not isinstance(skill_paths, list):
        skill_paths = []

    mcp_servers = getattr(agent, "mcp_servers", None) if agent else None
    if not isinstance(mcp_servers, list):
        mcp_servers = []

    self_aware_enabled = bool(getattr(agent, "self_aware", False)) if agent else False
    self_aware_config = getattr(agent, "self_aware_config", None) if agent else None
    if not isinstance(self_aware_config, dict):
        self_aware_config = {}
    # The label of the header chip: its own source and how far it has moved.
    self_aware_label = "Self-aware" if self_aware_enabled else ""
    if self_aware_enabled and self_aware_config.get("own_source"):
        self_aware_label = "Self-aware · own source"
        try:
            from ...memagent.self_source import SelfSource

            own = SelfSource(str(agent.agent_id))
            if own.exists():
                self_aware_label = f"Self-aware · v{own.versions()[0]['number']}"
        except Exception:
            pass
    self_aware_root_paths = self_aware_config.get("root_paths")
    if not isinstance(self_aware_root_paths, list):
        self_aware_root_paths = []
    self_aware_root_paths_text = "\n".join(
        _to_text(path).strip()
        for path in self_aware_root_paths
        if _to_text(path).strip()
    )

    harness_delegates = _harness_delegates(agent)
    harness_roots: List[str] = []
    if harness_delegates:
        from ..state import get_meta_harness

        try:
            harness_roots = list(get_meta_harness().allowed_workspace_roots or [])
        except Exception:
            harness_roots = []

    return templates.TemplateResponse(
        "playground.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents_nav": _build_agent_nav_items(active_agent_id=agent_id),
            "active_agent_id": agent_id,
            "agent": agent,
            "context_window": context_window,
            "toolbox_memory": toolbox_memory,
            "workflow_memory": workflow_memory,
            "skill_memory": skill_memory,
            "entity_memory": entity_memory,
            "summary_memory": summary_memory,
            "tool_log_memory": tool_log_memory,
            "knowledge_base_memory": knowledge_base_memory,
            "token_stats": token_stats,
            "error": error,
            "active_page": "playground",
            # Config fields
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "llm_routing": llm_config.get("routing") or {},
            "llm_context_window": llm_context_window,
            "instruction": instruction,
            "max_steps": max_steps,
            "recall_scope": _recall_scope(agent),
            "sandbox_provider": sandbox_provider,
            "sandbox_default": sandbox_default,
            "sandbox_effective": sandbox_effective,
            "sandbox_status_error": sandbox_status_error,
            "browser_control_provider": browser_control_provider,
            "browser_control_status_error": browser_control_status_error,
            "internet_provider": internet_provider,
            "internet_status_error": internet_status_error,
            "skills_marketplace_provider": skills_marketplace_provider,
            "skills_marketplace_status_error": skills_marketplace_status_error,
            "agent_tools": _extract_agent_tools(agent),
            "persona_name": persona_name,
            "persona_role": persona_role,
            "persona_goals": persona_goals,
            "persona_background": persona_background,
            "enable_entity_memory": enable_entity_memory,
            "enable_workflow_memory": enable_workflow_memory,
            "entity_memory_status_error": entity_memory_status_error,
            "skill_paths": skill_paths,
            "mcp_servers": mcp_servers,
            "sign_ins_needed": _sign_ins_needed(agent_id, agent, mcp_servers),
            "self_aware": self_aware_enabled,
            "self_aware_label": self_aware_label,
            "self_aware_root_paths": self_aware_root_paths_text,
            "self_aware_allow_writes": bool(
                self_aware_config.get("allow_writes", False)
            ),
            "self_aware_allow_deletes": bool(
                self_aware_config.get("allow_deletes", False)
            ),
            "automations_enabled": bool(getattr(agent, "automations_enabled", True))
            if agent
            else True,
            "default_timezone": _to_text(
                getattr(agent, "default_timezone", "") if agent else ""
            ).strip(),
            "threads": threads,
            "default_memory_id": default_memory_id,
            "harness_delegates": harness_delegates,
            "harness_roots": harness_roots,
        },
    )


def _sign_ins_needed(agent_id: str, agent: Any, mcp_servers: Any) -> List[Dict]:
    """The agent's connected apps that are not signed in, for the bar above
    the message box. Sent with the page: the bar is there at once, with no
    second read of the agent."""
    if not agent or not mcp_servers:
        return []
    try:
        from .capabilities import _agent_state

        return [
            row
            for row in _agent_state(agent_id, agent)
            if row.get("status") == "needs_sign_in"
        ]
    except Exception as exc:
        logger.debug("Sign-in state for %s was not read: %s", agent_id, exc)
        return []


@router.get("/api/nav/agent-status")
async def nav_agent_status(ids: str = ""):
    """What the sidebar's agents are doing: a current step or a last message."""
    if not _state["provider"]:
        return {"agents": {}}
    wanted = [value.strip() for value in ids.split(",") if value.strip()]
    if not wanted:
        return {"agents": {}}
    from ..helpers import _list_agents

    def load():
        return agent_status.nav_status(
            wanted, agent_status.known_agents(_list_agents, wanted)
        )

    try:
        statuses = await run_in_threadpool(load)
    except Exception as exc:
        logger.debug("Agent status for the sidebar failed: %s", exc)
        statuses = {}
    return {"agents": statuses}


@router.post("/agents/{agent_id}/playground/stream")
async def agent_playground_stream(request: Request, agent_id: str):
    """SSE endpoint: stream agent responses as Server-Sent Events."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    form = await request.form()
    query = str(form.get("query", "")).strip()
    memory_id = str(form.get("memory_id", "")).strip() or None
    # Optional end-user scope for multi-tenant deployments that proxy the
    # UI. Empty string == single-operator/legacy behavior.
    user_id = str(form.get("user_id", "")).strip() or None
    # Allow runtime overrides from the playground config panel
    override_model = str(form.get("llm_model", "")).strip() or None
    override_instruction = str(form.get("instruction", "")).strip() or None
    # What this agent's harness delegates may use: an access grant from the
    # inspector, else the conversation's own folder.
    harness_grant = str(form.get("harness_grant", "")).strip()
    harness_workspace = str(form.get("harness_workspace", "")).strip()
    # "Answered by": a harness that runs this turn in place of the agent's
    # own model. Chosen per conversation in the page; never saved on the agent.
    answer_harness = (
        str(form.get("answer_harness", "")).strip().lower().replace("_", "-")
    )
    answer_model = str(form.get("answer_model", "")).strip() or None
    if answer_harness:
        if _state.get("read_only"):
            raise HTTPException(status_code=403, detail="This console is read-only")
        await run_in_threadpool(_check_answer_harness, answer_harness)

    if not query:
        raise HTTPException(status_code=400, detail="No query provided")

    import asyncio
    import time

    from ...memagent import MemAgent
    from ...memagent.persistence import keep_stored_fields
    from ...streaming import agent_event_stream, check_cancelled, execution_lock

    def prepare(session):
        # Keep follow-up turns behind prior conversation/agent-state commits.
        lock = execution_lock(("ui", agent_id))
        while not lock.acquire(timeout=0.1):
            check_cancelled()
        session.stack.callback(lock.release)
        check_cancelled()
        overrides: Dict[str, Any] = {
            "streaming": True,
            "capture_context_snapshots": True,
            "capture_memory_history": True,
        }
        if override_instruction:
            overrides["instruction"] = override_instruction
        agent_instance = MemAgent.load(
            agent_id, memory_provider=_state["provider"], **overrides
        )
        # Settings this message runs with but the agent was not saved with:
        # unsaved config-panel edits and this process's default providers.
        # A save during the run keeps the stored values for them.
        run_only: List[str] = ["instruction"] if override_instruction else []
        # The stored record drives the model override and the sandbox,
        # internet and entity-memory checks below: read it once per message.
        stored_agent = _state["provider"].retrieve_memagent(agent_id)
        # Apply runtime model override if user changed the model in the playground
        if override_model:
            run_only.append("llm_config")
            try:
                from ...llms.llm_factory import create_llm_provider

                llm_config = dict(getattr(stored_agent, "llm_config", None) or {})
                llm_config["model"] = override_model

                # Carry over the API key from the already-loaded
                # provider so that saved configs (which deliberately
                # omit secrets) don't cause auth failures.
                existing_model = getattr(agent_instance, "model", None)
                if existing_model is not None:
                    for attr in ("api_key", "_api_key"):
                        key = getattr(existing_model, attr, None)
                        if key and "api_key" not in llm_config:
                            llm_config["api_key"] = key
                            break
                    # Also check the underlying client object
                    if "api_key" not in llm_config:
                        client = getattr(existing_model, "client", None)
                        if client is not None:
                            key = getattr(client, "api_key", None)
                            if key:
                                llm_config["api_key"] = key

                agent_instance.model = create_llm_provider(llm_config)
                # Override succeeded — clear any prior init error
                # carried over from MemAgent.load().
                agent_instance._llm_init_error = None
            except Exception as exc:
                # Stash the cause on the instance so chat can show
                # the real reason instead of "No LLM model configured".
                agent_instance._llm_init_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Could not apply model override: %s", exc)

        # Apply sandbox provider from agent config or global default.
        # Skip if the agent already attempted sandbox init during load()
        # — no need to retry and log the same warning twice.
        if not agent_instance.has_sandbox():
            sandbox_cfg = (
                getattr(stored_agent, "sandbox_provider", None)
                if stored_agent
                else None
            )
            if not sandbox_cfg:
                sandbox_cfg = os.environ.get("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "")
            if _sandbox_is_off(sandbox_cfg):
                # This agent runs no code, whatever the console's default.
                sandbox_cfg = ""
            resolved_sandbox_cfg = _resolve_sandbox_provider_config(sandbox_cfg)
            # Only attempt if no sandbox config was already tried during load
            stored_sandbox = (
                getattr(stored_agent, "sandbox_provider", None)
                if stored_agent
                else None
            )
            sandbox_already_attempted = bool(stored_sandbox) and not _sandbox_is_off(
                stored_sandbox
            )
            if resolved_sandbox_cfg and not sandbox_already_attempted:
                try:
                    validation_error = _validate_sandbox_provider_choice(
                        resolved_sandbox_cfg
                    )
                    if validation_error:
                        raise ValueError(validation_error)
                    agent_instance.with_sandbox_provider(resolved_sandbox_cfg)
                    run_only.append("sandbox_provider")
                except Exception as exc:
                    logger.debug(f"Sandbox provider not available: {exc}")

        # Apply internet provider from agent config or global default.
        browser_apply_error = None
        browser_config = (
            getattr(stored_agent, "browser_control", None) if stored_agent else None
        )
        stored_browser_config = browser_config
        if not browser_config:
            browser_name = _normalize_browser_control_provider_name(
                os.environ.get("MEMORIZZ_BROWSER_CONTROL_PROVIDER", "")
            )
            browser_config = _build_browser_control_config(browser_name)
        if browser_config and not agent_instance.has_browser_control():
            # MemAgent.load already attempted persisted config. Only apply
            # the global fallback here to avoid retrying the same failure.
            if not stored_browser_config:
                try:
                    agent_instance.with_browser_control(browser_config)
                    run_only.append("browser_control")
                except Exception as exc:
                    browser_apply_error = (
                        "Browser control unavailable: " + _to_text(exc).strip()
                    )
            else:
                browser_apply_error = getattr(
                    agent_instance, "_browser_control_init_error", None
                )

        if browser_apply_error:
            session.emit(
                "status",
                stage="configuration_warning",
                scope="browser_control",
                message="Browser control unavailable",
            )

        # Apply internet provider from agent config or global default.
        internet_apply_error = None
        internet_provider_name = _normalize_internet_provider_name(
            getattr(stored_agent, "internet_access_provider", None)
            if stored_agent
            else None
        )
        internet_provider_config = (
            getattr(stored_agent, "internet_access_config", None)
            if stored_agent
            else None
        )
        if not internet_provider_name:
            internet_provider_name = _normalize_internet_provider_name(
                os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER", "")
            )
            internet_provider_config = None
            run_only.extend(("internet_access_provider", "internet_access_config"))

        if internet_provider_name and not agent_instance.has_internet_access():
            try:
                from ...internet_access import create_internet_access_provider

                resolved_internet_config = _build_internet_provider_config(
                    internet_provider_name, internet_provider_config
                )
                provider_instance = create_internet_access_provider(
                    internet_provider_name,
                    resolved_internet_config or {},
                )
                if provider_instance:
                    agent_instance.with_internet_access_provider(provider_instance)
                else:
                    internet_apply_error = (
                        f"Internet provider '{internet_provider_name}' is unknown."
                    )
            except Exception as exc:
                internet_apply_error = "Internet unavailable: " + _to_text(exc).strip()
                logger.warning(f"Could not apply internet provider: {exc}")

        if internet_apply_error:
            session.emit(
                "status",
                stage="configuration_warning",
                scope="internet",
                message="Internet access unavailable",
            )
        if answer_harness:
            _answer_on_harness(agent_instance, answer_harness, answer_model)
            run_only.extend(
                (
                    "meta_harness",
                    "meta_harness_mode",
                    "default_harness",
                    "harness_config",
                )
            )
            session.emit(
                "status",
                stage="answered_by",
                harness=answer_harness,
                message=f"{answer_harness} answers this turn",
            )
        keep_stored_fields(agent_instance, *run_only)

        # Validate entity-memory runtime tool availability.
        entity_memory_warning = _entity_memory_status_error(stored_agent)
        if (
            not entity_memory_warning
            and _agent_entity_memory_enabled(stored_agent)
            and getattr(agent_instance, "tool_manager", None) is not None
        ):
            try:
                runtime_tools = set(agent_instance.tool_manager.list_tools())
            except Exception:
                runtime_tools = set()
            if (
                "entity_memory_lookup" not in runtime_tools
                or "entity_memory_upsert" not in runtime_tools
            ):
                entity_memory_warning = (
                    "Entity memory is enabled in config, but runtime entity-memory "
                    "tools are unavailable in this session."
                )
        if entity_memory_warning:
            session.emit(
                "status",
                stage="configuration_warning",
                scope="entity_memory",
                message="Entity memory tools unavailable",
            )

        # Validate persona runtime tool availability. Mirrors the
        # entity-memory check above: if the stored agent has a
        # persona configured but update_persona/read_persona didn't
        # register on the runtime, surface it instead of silently
        # leaving the LLM without its own persona-evolution tools.
        persona_warning = None
        stored_persona = (
            getattr(stored_agent, "persona", None) if stored_agent else None
        )
        if stored_persona and getattr(agent_instance, "tool_manager", None) is not None:
            try:
                runtime_tools = set(agent_instance.tool_manager.list_tools())
            except Exception:
                runtime_tools = set()
            if (
                "update_persona" not in runtime_tools
                or "read_persona" not in runtime_tools
            ):
                persona_warning = (
                    "A persona is configured on this agent, but the persona "
                    "evolution tools (update_persona, read_persona) are not "
                    "available in this session."
                )
        if persona_warning:
            session.emit(
                "status",
                stage="configuration_warning",
                scope="persona",
                message="Persona tools unavailable",
            )
        run_kwargs: Dict[str, Any] = {"user_id": user_id} if user_id else {}
        # Each playground conversation has a memory of its own, and one
        # person owns them all: unless the agent keeps conversations apart, a
        # chat recalls the agent's other conversations and is told of them.
        recall_policy = _playground_recall_policy(agent_instance, memory_id)
        if recall_policy is not None:
            run_kwargs["retrieval_policy"] = recall_policy
        if (harness_grant or harness_workspace) and (
            answer_harness or _harness_delegates(agent_instance)
        ):
            parent, warning = None, None
            try:
                from ..state import get_meta_harness

                parent, warning = _harness_parent(
                    get_meta_harness(), agent_id, harness_grant, harness_workspace
                )
            except Exception as exc:
                warning = f"Harness delegates are unavailable: {exc}"
            if warning:
                session.emit(
                    "status",
                    stage="configuration_warning",
                    scope="harness_access",
                    message=warning,
                    grant_expired=bool(harness_grant)
                    and not (parent or {}).get("grant_id"),
                )
            if parent:
                run_kwargs["tool_context"] = {"harness_parent": parent}
        return agent_instance, run_kwargs

    def finalize(session):
        current_memory_id = session.identity["memory_id"]
        agent_instance = session.agent
        if current_memory_id not in (agent_instance.memory_ids or []):
            agent_instance.memory_ids.append(current_memory_id)
        update = getattr(_state["provider"], "update_memagent_memory_ids", None)
        if callable(update):
            result = update(agent_id, agent_instance.memory_ids)
            session.persistence["adapter_state"] = (
                "failed" if result is False else "written"
            )
        else:
            session.persistence["adapter_state"] = "not_configured"

    # Continue the memory's most recent thread, as the CLI does. History loads
    # per thread, and a streamed turn without one starts a new thread — the
    # agent would lose the conversation (and its cached prefix) every turn.
    thread_kwargs: Dict[str, Any] = {}
    if memory_id:
        from ...cli.conversations import latest_thread_id

        try:
            thread_id = await asyncio.to_thread(
                latest_thread_id,
                _state["provider"],
                memory_id,
                user_id=user_id,
                agent_id=agent_id,
            )
        except Exception:
            thread_id = None
        if thread_id:
            thread_kwargs["thread_id"] = thread_id

    async def event_stream():
        stream = agent_event_stream(
            None,
            query,
            _prepare=prepare,
            _finalize=finalize,
            _agent_id=agent_id,
            memory_id=memory_id,
            **thread_kwargs,
        )
        last_frame = time.monotonic()
        sentinel = object()
        run_ids: set = set()
        # A harness turn can run for minutes. If the page goes away (a
        # reload, a closed tab) it carries on, and the page finds it again.
        started = None
        progress = None
        finished = False

        def poll():
            try:
                return stream.poll(timeout=0.5)
            except StopIteration:
                return sentinel

        try:
            while True:
                event = await asyncio.to_thread(poll)
                if event is sentinel:
                    break
                if event is None:
                    if time.monotonic() - last_frame >= 10:
                        yield ": heartbeat\n\n"
                        last_frame = time.monotonic()
                    continue
                # The sidebar shows this step beside the agent's name.
                run_ids.add(event["run_id"])
                agent_status.note_event(agent_id, event)
                if event["type"] == "run.started":
                    started = event
                elif event["type"] == "run.done":
                    finished = True
                elif event["type"] == "status" and event.get("harness_run_id"):
                    progress = event
                yield f"id: {event['run_id']}:{event['seq']}\nevent: {event['type']}\ndata: {event.to_json()}\n\n"
                last_frame = time.monotonic()
        finally:
            if answer_harness and started is not None and not finished:
                # Stopping a harness turn is asked for by name (the run is
                # cancelled through its own route), so a dropped connection
                # here is the page leaving, not a request to stop.
                _follow_detached(stream, agent_id, query, started, progress, run_ids)
            else:
                # Explicit browser abort cancels this request's producer, never another run.
                await asyncio.to_thread(stream.close)
                for run_id in run_ids:
                    agent_status.end_run(agent_id, run_id)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/api/agents/{agent_id}/answer-harnesses")
async def agent_answer_harnesses(agent_id: str):
    """The harnesses this agent's conversations can be answered by."""
    from ..helpers import _load_agent

    agent = _load_agent(agent_id)

    def field(name):
        return (
            agent.get(name) if isinstance(agent, dict) else getattr(agent, name, None)
        )

    # An agent set to run on a harness is answered by it whatever is picked
    # here, so the page names that harness instead of "the agent's model".
    runtime = None
    if field("meta_harness") and str(field("meta_harness_mode") or "") == "runtime":
        runtime = str(field("default_harness") or "auto").strip().lower() or "auto"
    return {
        "ok": True,
        "harnesses": await run_in_threadpool(_answer_harnesses),
        "runtime": runtime,
    }


@router.post("/api/agents/{agent_id}/harness-access")
async def agent_harness_access_grant(request: Request, agent_id: str):
    """Let this agent's harness delegates use a folder, the web or edits in
    the playground. Web access and edits need an approver's name."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")
    body = await _json_object(request)
    network = str(body.get("network") or "none").strip().lower()
    if network not in {"none", "full"}:
        raise HTTPException(status_code=400, detail="network must be none or full")
    write = body.get("write", False)
    if not isinstance(write, bool):
        raise HTTPException(status_code=400, detail="write must be true or false")
    approver = str(body.get("approver_id") or "").strip()
    if (network == "full" or write) and not approver:
        raise HTTPException(
            status_code=400,
            detail="Give your name to approve web access or edits",
        )
    workspace = str(body.get("workspace") or "").strip()
    agent = await run_in_threadpool(_state["provider"].retrieve_memagent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Access is for the agent's harness delegates, or for a conversation that
    # a harness answers (the page names it; it is checked like a turn's).
    answer_harness = str(body.get("answer_harness") or "").strip().lower()
    if answer_harness:
        await run_in_threadpool(_check_answer_harness, answer_harness.replace("_", "-"))
    elif not await run_in_threadpool(_harness_delegates, agent):
        raise HTTPException(
            status_code=400, detail="This agent has no delegates that run on a harness"
        )
    service = _harness_service()
    try:
        grant = await run_in_threadpool(
            lambda: service.grant_delegate_access(
                agent_id=agent_id,
                approver_id=approver or "playground",
                workspace=workspace,
                network=network,
                write=write,
            )
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        from ...metaharness.security import HarnessSecurityError

        if isinstance(exc, HarnessSecurityError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise
    return {"ok": True, "grant": grant}


@router.get("/api/agents/{agent_id}/harness-access/{grant_id}")
async def agent_harness_access_status(agent_id: str, grant_id: str):
    """A grant while it holds; 404 once it expired or belongs to another agent."""
    grant = await run_in_threadpool(
        _harness_service().delegate_access, grant_id, agent_id=agent_id
    )
    if grant is None:
        raise HTTPException(status_code=404, detail="This access has expired")
    return {"ok": True, "grant": grant}


@router.post("/api/agents/{agent_id}/harness-workspace")
async def agent_harness_workspace(agent_id: str):
    """A fresh folder for one conversation's harness delegates, so they never
    work in the folder the UI runs from."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")
    service = _harness_service()
    folder = await run_in_threadpool(service.scratch_workspace)
    return {"ok": True, "workspace": folder}


@router.post("/api/agents/{agent_id}/threads/{memory_id}")
async def agent_rename_thread(request: Request, agent_id: str, memory_id: str):
    """Name a conversation; an empty title goes back to its first question."""
    from ..helpers import _load_agent, _save_agent_fields

    agent = _load_agent(agent_id)
    if memory_id not in (getattr(agent, "memory_ids", None) or []):
        raise HTTPException(status_code=404, detail="Conversation not found")
    payload = await _json_object(request, required=False)
    title = " ".join(str(payload.get("title") or "").split())[:80]
    titles = dict(getattr(agent, "thread_titles", None) or {})
    if title:
        titles[memory_id] = title
    else:
        titles.pop(memory_id, None)
    _save_agent_fields(agent, thread_titles=titles or None)
    return {"ok": True, "memory_id": memory_id, "title": title or None}


@router.get("/api/agents/{agent_id}/live")
async def agent_live_turns(
    request: Request, agent_id: str, memory_id: str = "", wait: float = 300.0
):
    """Server-sent events for turns the CLI runs with this agent, as they
    stream; ``memory_id`` keeps one conversation's. The stream closes after
    ``wait`` seconds and the browser reconnects."""
    import asyncio
    import time

    from ...live_turns import LiveTail
    from ..helpers import _load_agent

    _load_agent(agent_id)
    tail = LiveTail(agent_id, memory_id.strip() or None)
    deadline = time.monotonic() + max(0.0, min(wait, 300.0))

    from ..shutdown import requested as stopping

    async def events():
        yield "retry: 2000\n\n"
        beat = time.monotonic()
        while True:
            for event in await run_in_threadpool(tail.poll):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            now = time.monotonic()
            # Ctrl+C ends the stream at once: the server waits for it.
            if now >= deadline or stopping() or await request.is_disconnected():
                return
            if now - beat >= 15:
                yield ": keep-alive\n\n"
                beat = now
            await asyncio.sleep(0.2)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.post("/api/agents/{agent_id}/name")
async def agent_rename(request: Request, agent_id: str):
    """Rename an agent: the name the playground, the agent list and the CLI show."""
    from ..helpers import _load_agent, _save_agent_fields

    agent = _load_agent(agent_id)
    payload = await _json_object(request, required=False)
    name = " ".join(str(payload.get("name") or "").split())[:AGENT_NAME_MAX]
    if not name:
        raise HTTPException(status_code=400, detail="The name cannot be empty")
    _save_agent_fields(agent, name=name)
    return {"ok": True, "agent_id": agent_id, "name": name}


def _delete_conversation(
    provider: Any, agent_id: str, memory_id: str
) -> Dict[str, int]:
    """Delete one conversation's messages, summaries and tool logs."""
    from ...enums.memory_type import MemoryType
    from ...memagent.managers.memory_manager import MemoryManager

    def row_id(row: Dict[str, Any]) -> Optional[str]:
        value = row.get("_id") or row.get("id")
        return str(value) if value else None

    def mine(row: Dict[str, Any]) -> bool:
        owner = row.get("agent_id")
        return not owner or str(owner) == agent_id

    counts = {"messages": 0, "summaries": 0, "tool_logs": 0}
    rows = provider.retrieve_conversation_history_ordered_by_timestamp(
        memory_id=memory_id
    )
    for row in rows or []:
        if isinstance(row, dict) and mine(row) and row_id(row):
            if provider.delete_by_id(row_id(row), MemoryType.CONVERSATION_MEMORY):
                counts["messages"] += 1
    manager = MemoryManager(provider)
    for summary in manager.load_summaries_for_thread(
        memory_id=memory_id, agent_id=agent_id, limit=1000
    ):
        # The loader also returns the agent's summaries of other
        # conversations; delete only this one's.
        if summary.get("memory_id") != memory_id:
            continue
        if summary.get("summary_id") and provider.delete_by_id(
            summary["summary_id"], MemoryType.SUMMARIES
        ):
            counts["summaries"] += 1
    for log in manager.list_tool_logs(memory_id, limit=1000) or []:
        if not isinstance(log, dict) or not mine(log):
            continue
        # Some stores key tool logs by tool_log_id rather than _id.
        for candidate in dict.fromkeys((row_id(log), log.get("tool_log_id"))):
            if candidate and provider.delete_by_id(str(candidate), MemoryType.TOOL_LOG):
                counts["tool_logs"] += 1
                break
    # Captured inputs contain conversation content. Remove this thread's
    # snapshots too; the content-free change journal can retain deletion metadata.
    from ...observability.context_snapshots import RECORD_TYPE
    from ...observability.store import ObservabilityStore

    cursor = None
    snapshot_ids = []
    while True:
        page = provider.query_observability_records(
            MemoryType.SHARED_MEMORY,
            agent_ids=[agent_id],
            record_type=RECORD_TYPE,
            limit=1000,
            cursor=cursor,
        )
        for row in page.get("items", []):
            payload = ObservabilityStore._payload(row)
            if (
                payload
                and payload.get("agent_id") == agent_id
                and payload.get("memory_id") == memory_id
            ):
                snapshot_ids.append(payload["record_id"])
        cursor = page.get("next_cursor")
        if not cursor:
            break
    for snapshot_id in snapshot_ids:
        provider.delete_by_id(snapshot_id, MemoryType.SHARED_MEMORY)
    return counts


@router.delete("/api/agents/{agent_id}/threads/{memory_id}")
async def agent_delete_thread(agent_id: str, memory_id: str):
    """Delete a conversation: its messages, summaries and tool logs."""
    from ..helpers import _load_agent, _save_agent_fields

    agent = _load_agent(agent_id)
    memory_ids = list(getattr(agent, "memory_ids", None) or [])
    if memory_id not in memory_ids:
        raise HTTPException(status_code=404, detail="Conversation not found")
    titles = dict(getattr(agent, "thread_titles", None) or {})
    titles.pop(memory_id, None)
    # Leave the list first, so a failure below never shows a half-deleted
    # conversation; the rows are then removed.
    _save_agent_fields(
        agent,
        memory_ids=[item for item in memory_ids if item != memory_id],
        thread_titles=titles or None,
    )
    try:
        counts = await run_in_threadpool(
            _delete_conversation, _state["provider"], agent_id, memory_id
        )
    except Exception as exc:
        logger.error("Deleting conversation %s failed: %s", memory_id, exc)
        raise HTTPException(
            status_code=500,
            detail="The conversation was removed from the list, "
            "but some of its messages could not be deleted",
        ) from exc
    return {"ok": True, "memory_id": memory_id, "deleted": counts}


@router.post("/api/agents/{agent_id}/context-policy")
async def agent_context_policy(request: Request, agent_id: str):
    """Change when the agent compacts its context (``compact_at``, percent).

    Compaction stores summaries, so a threshold also turns on summary memory.
    """
    from ...enums.memory_type import MemoryType
    from ...tooling import ContextPolicy
    from ..helpers import _load_agent, _save_agent_fields

    agent = _load_agent(agent_id)
    payload = await _json_object(request, required=False)
    if "compact_at" not in payload:
        raise HTTPException(status_code=400, detail="compact_at is required")
    try:
        compact_at = int(payload["compact_at"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="compact_at must be a number"
        ) from exc
    current = getattr(agent, "context_policy", None) or {}
    policy = ContextPolicy.from_value(
        {**(current if isinstance(current, dict) else {}), "compact_at": compact_at}
    )
    fields: Dict[str, Any] = {"context_policy": policy.to_dict()}
    memory_types = [
        str(getattr(item, "value", item))
        for item in getattr(agent, "memory_types", None) or []
    ]
    summaries = MemoryType.SUMMARIES.value
    if policy.compact_at and memory_types and summaries not in memory_types:
        fields["memory_types"] = memory_types + [summaries]
    _save_agent_fields(agent, **fields)
    return {
        "ok": True,
        "context_policy": policy.to_dict(),
        "summary_memory": not memory_types
        or summaries in (fields.get("memory_types") or memory_types),
    }


@router.post("/agents/{agent_id}/playground/compact")
async def agent_playground_compact(request: Request, agent_id: str):
    """Compact/summarize the context window for the given agent thread."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    form = await request.form()
    memory_id = str(form.get("memory_id", "")).strip() or None

    from ...memagent import MemAgent

    try:
        agent_instance = await run_in_threadpool(
            MemAgent.load, agent_id, memory_provider=_state["provider"]
        )

        if not agent_instance:
            return JSONResponse(
                {"ok": False, "error": "Agent not found"}, status_code=404
            )

        if memory_id:
            agent_instance._current_memory_id = memory_id

        # Same as auto-compaction: the newest messages stay word for word.
        keep = int(agent_instance.context_policy.keep_recent_messages)
        summary_ids = await run_in_threadpool(
            agent_instance.generate_summaries,
            days_back=36500,
            max_memories_per_summary=200,
            memory_id=memory_id,
            user_id=agent_instance._current_user_id,
            thread_id=agent_instance._current_thread_id,
            keep_recent=keep,
            summary_type="compaction",
        )

        return JSONResponse(
            {
                "ok": True,
                "compacted": bool(summary_ids),
                "summary_count": len(summary_ids),
                "summary_ids": summary_ids,
                "message": (
                    f"Context compacted: {len(summary_ids)} "
                    f"{'summary' if len(summary_ids) == 1 else 'summaries'} created."
                    if summary_ids
                    else "No compaction performed. No new summaries were created; "
                    "recent messages are kept unchanged."
                ),
            }
        )
    except Exception as exc:
        logger.error(f"Compact failed for agent {agent_id}: {exc}")
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


def _thread_order_key(row: Dict[str, Any]) -> float:
    """Order messages and trace bundles on one clock.

    Conversation rows store naive UTC timestamps while trace bundles store
    UTC-aware ones. Reading naive values as local time pushed every trace
    bundle after the whole conversation, so naive ISO strings count as UTC.
    """
    value = row.get("timestamp")
    if isinstance(value, str):
        text = value.strip()
        if (
            text
            and "T" in text
            and not text.endswith("Z")
            and "+" not in text[10:]
            and text.count("-") <= 2
        ):
            value = text + "+00:00"
    return _coerce_timestamp(value) or 0.0


@router.get("/agents/{agent_id}/playground/thread")
async def agent_playground_thread(
    agent_id: str, memory_id: str = "", limit: int = THREAD_PAGE_SIZE
):
    """Return the newest page of one of the agent's own conversations."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    agent = _state["provider"].retrieve_memagent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    requested_memory_id = _to_text(memory_id).strip()
    # Same membership rule as renaming or deleting a conversation.
    if requested_memory_id and requested_memory_id not in (
        getattr(agent, "memory_ids", None) or []
    ):
        raise HTTPException(status_code=404, detail="Conversation not found")
    page_size = max(1, min(int(limit or THREAD_PAGE_SIZE), THREAD_PAGE_MAX))
    messages: List[Dict[str, Any]] = []
    if requested_memory_id:
        messages = _load_thread_messages(requested_memory_id, limit=page_size)
        messages.extend(
            _load_thread_trace_bundles(
                agent_id=agent_id,
                memory_id=requested_memory_id,
                limit=100,
            )
        )
        messages.sort(key=_thread_order_key)
        logger.debug(
            "Thread %s: loaded %d raw messages for agent %s",
            requested_memory_id,
            len(messages),
            agent_id,
        )
        messages = [
            msg
            for msg in messages
            if not _to_text(msg.get("agent_id")).strip()
            or _to_text(msg.get("agent_id")).strip() == agent_id
        ]
        logger.debug(
            "Thread %s: %d messages after agent_id filter",
            requested_memory_id,
            len(messages),
        )

    serialized = [_serialize_thread_message(msg) for msg in messages]
    await run_in_threadpool(
        _mark_harness_turns, serialized, agent_id, requested_memory_id
    )
    thread_memory = _load_thread_memories(agent_id, requested_memory_id)
    toolbox_memory = thread_memory["toolbox"]
    workflow_memory = thread_memory["workflow"]
    skill_memory = _load_thread_skill_memory(agent_id)
    entity_memory = thread_memory["entity"]
    summary_memory = thread_memory["summary"]
    tool_log_memory = thread_memory["tool_log"]
    token_stats = _build_token_stats(
        agent,
        messages,
        toolbox_memory=toolbox_memory,
        workflow_memory=workflow_memory,
        entity_memory=entity_memory,
        summary_memory=summary_memory,
    )

    last_activity = "—"
    if messages:
        last_ts = _coerce_timestamp(messages[-1].get("timestamp"))
        if last_ts is not None:
            last_activity = datetime.fromtimestamp(last_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

    return {
        "memory_id": requested_memory_id,
        "messages": serialized,
        "toolbox_memory": toolbox_memory,
        "workflow_memory": workflow_memory,
        "skill_memory": skill_memory,
        "entity_memory": entity_memory,
        "summary_memory": summary_memory,
        "tool_log_memory": tool_log_memory,
        "token_stats": token_stats,
        # Same count as the conversation list: messages, not run traces.
        "message_count": sum(
            1 for row in serialized if row.get("message_type") != "trace_bundle"
        ),
        "last_activity": last_activity,
    }


def _config_error_redirect(agent_id: str, message: str) -> RedirectResponse:
    """Back to the playground with the config panel's error in the query string."""
    return RedirectResponse(
        url=f"/agents/{agent_id}/playground?config_error={quote(message)}",
        status_code=302,
    )


@router.post("/agents/{agent_id}/playground/config")
async def agent_playground_config_update(request: Request, agent_id: str):
    """Update agent config from the playground panel."""
    if not _state["provider"]:
        raise HTTPException(status_code=400, detail="Not connected")

    existing = _state["provider"].retrieve_memagent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not found")

    form = await request.form()
    new_instruction = str(form.get("instruction", "")).strip()
    new_model = str(form.get("llm_model", "")).strip()
    new_provider = str(form.get("llm_provider", "")).strip()
    new_max_steps = form.get("max_steps")
    new_sandbox_provider = str(form.get("sandbox_provider", "")).strip()
    new_browser_control_provider = _normalize_browser_control_provider_name(
        form.get("browser_control_provider", "")
    )
    new_internet_provider = _normalize_internet_provider_name(
        form.get("internet_provider", "")
    )
    raw_skills_marketplace_provider = form.get("skills_marketplace_provider")
    if raw_skills_marketplace_provider is None:
        new_skills_marketplace_provider = _normalize_skills_marketplace_provider_name(
            getattr(existing, "skills_marketplace_provider", None)
        )
    else:
        new_skills_marketplace_provider = _normalize_skills_marketplace_provider_name(
            raw_skills_marketplace_provider
        )
    new_persona_name = str(form.get("persona_name", "")).strip()
    new_persona_role = str(form.get("persona_role", "")).strip()
    new_persona_goals = str(form.get("persona_goals", "")).strip()
    new_persona_background = str(form.get("persona_background", "")).strip()
    raw_skill_paths = form.get("skill_paths_json")
    raw_enable_entity_memory = form.get("enable_entity_memory")
    raw_enable_workflow_memory = form.get("enable_workflow_memory")
    raw_self_aware = form.get("self_aware")
    raw_self_aware_root_paths = form.get("self_aware_root_paths")
    raw_self_aware_allow_writes = form.get("self_aware_allow_writes")
    raw_self_aware_allow_deletes = form.get("self_aware_allow_deletes")
    raw_automations_enabled = form.get("automations_enabled")
    raw_default_timezone = form.get("default_timezone")

    parsed_skill_paths = None
    parse_error = None

    # MCP servers are edited only on the MCP connections page, which keeps
    # their credentials encrypted; this form never replaces them.
    if raw_skill_paths is not None:
        parsed_skill_paths, parse_error = _parse_skill_paths_json(str(raw_skill_paths))
    if parse_error:
        return _config_error_redirect(agent_id, parse_error[:200])

    # Build updated fields. When the user switches providers we have
    # to start a fresh dict instead of merging onto the old one —
    # otherwise provider-specific keys (e.g. OpenAI's `max_tokens`)
    # ride along into the next provider and crash construction.
    existing_llm_config = getattr(existing, "llm_config", {}) or {}
    existing_provider = (existing_llm_config.get("provider") or "").lower()
    if new_provider and new_provider.lower() != existing_provider:
        llm_config: Dict[str, Any] = {"provider": new_provider}
    else:
        llm_config = dict(existing_llm_config)
        if new_provider:
            llm_config["provider"] = new_provider
    if new_model:
        if (llm_config.get("provider") or "").lower() == "azure":
            llm_config["deployment_name"] = new_model
        else:
            llm_config["model"] = new_model
    if "llm_routing_json" in form:
        # OpenRouter's routing rules; empty fields clear them, and another
        # provider never carries them.
        llm_config.pop("routing", None)
        if (llm_config.get("provider") or "").lower() == "openrouter":
            routing, routing_error = _parse_routing(form.get("llm_routing_json"))
            if routing_error:
                return _config_error_redirect(agent_id, routing_error)
            if routing:
                llm_config["routing"] = routing
    if "context_window_tokens" in form:
        # Blank means the provider default (for Ollama, the model's own
        # length when it fits in memory); a number pins the window.
        raw_window = str(form.get("context_window_tokens") or "").strip()
        extra = dict(llm_config.get("additional_config") or {})
        extra.pop("num_ctx", None)
        if extra or "additional_config" in llm_config:
            llm_config["additional_config"] = extra
        if raw_window.isdigit() and int(raw_window) > 0:
            llm_config["context_window_tokens"] = int(raw_window)
        else:
            llm_config.pop("context_window_tokens", None)

    persona_payload = None
    existing_persona = getattr(existing, "persona", None)
    if new_persona_name or (
        existing_persona
        and (
            (isinstance(existing_persona, dict) and existing_persona.get("name"))
            or (hasattr(existing_persona, "name") and existing_persona.name)
        )
    ):
        persona_payload = {}
        if isinstance(existing_persona, dict):
            persona_payload = dict(existing_persona)
        elif existing_persona and hasattr(existing_persona, "__dict__"):
            persona_payload = dict(existing_persona.__dict__)
        if new_persona_name:
            persona_payload["name"] = new_persona_name
        if new_persona_role:
            persona_payload["role"] = new_persona_role
        # Always update goals and background (allow clearing them)
        persona_payload["goals"] = new_persona_goals
        persona_payload["background"] = new_persona_background

    max_steps_value = getattr(existing, "max_steps", 20)
    if new_max_steps:
        try:
            max_steps_value = int(new_max_steps)
        except (ValueError, TypeError):
            pass

    enable_entity_memory_value = _parse_bool(raw_enable_entity_memory)
    enable_workflow_memory_value = _parse_bool(raw_enable_workflow_memory)
    self_aware_enabled_value = _parse_bool(raw_self_aware)
    self_aware_allow_writes_value = _parse_bool(raw_self_aware_allow_writes)
    self_aware_allow_deletes_value = _parse_bool(raw_self_aware_allow_deletes)
    self_aware_root_paths_value = _parse_self_aware_root_paths(
        _to_text(raw_self_aware_root_paths)
    )
    existing_self_aware_config = getattr(existing, "self_aware_config", None)
    if not isinstance(existing_self_aware_config, dict):
        existing_self_aware_config = None
    self_aware_config_value = _build_self_aware_config(
        root_paths=self_aware_root_paths_value,
        allow_writes=self_aware_allow_writes_value,
        allow_deletes=self_aware_allow_deletes_value,
        base_config=existing_self_aware_config,
    )
    self_aware_validation_error = _validate_self_aware_config(self_aware_config_value)
    if self_aware_validation_error:
        return _config_error_redirect(agent_id, self_aware_validation_error[:220])

    if raw_automations_enabled is None:
        automations_enabled_value = bool(getattr(existing, "automations_enabled", True))
    else:
        automations_enabled_value = _parse_bool(raw_automations_enabled)

    default_timezone_value = getattr(existing, "default_timezone", None)
    if raw_default_timezone is not None:
        default_timezone_value = _to_text(raw_default_timezone).strip() or None
    if default_timezone_value:
        try:
            from ...automation.schedule import validate_timezone_name

            validate_timezone_name(default_timezone_value)
        except Exception as exc:
            return _config_error_redirect(agent_id, str(exc)[:220])
    memory_types_value = _build_memory_types_for_agent(
        application_mode=getattr(existing, "application_mode", "assistant"),
        enable_entity_memory=enable_entity_memory_value,
        enable_workflow_memory=enable_workflow_memory_value,
        existing_memory_types=getattr(existing, "memory_types", None),
    )

    # Resolve sandbox provider: use form value, fall back to existing
    # The form's choice is the setting: empty follows the console's default,
    # "none" turns it off. A request without the field leaves it as it was.
    if "sandbox_provider" in form:
        sandbox_value = new_sandbox_provider or None
    else:
        sandbox_value = getattr(existing, "sandbox_provider", None)
    # Only validate sandbox when the user explicitly changed it;
    # carry forward the existing value without blocking unrelated edits.
    sandbox_changed = (
        new_sandbox_provider
        and new_sandbox_provider
        != _to_text(getattr(existing, "sandbox_provider", "") or "").strip()
    )
    if sandbox_changed:
        sandbox_validation_error = _validate_sandbox_provider_choice(sandbox_value)
        if sandbox_validation_error:
            return _config_error_redirect(agent_id, sandbox_validation_error[:220])

    existing_browser_control = getattr(existing, "browser_control", None)
    existing_browser_provider = _normalize_browser_control_provider_name(
        existing_browser_control
    )
    browser_control_value = new_browser_control_provider or None
    browser_control_config_value = _build_browser_control_config(
        browser_control_value,
        existing_browser_control
        if new_browser_control_provider == existing_browser_provider
        and isinstance(existing_browser_control, dict)
        else None,
    )
    if new_browser_control_provider != existing_browser_provider:
        browser_validation_error = _validate_browser_control_choice(
            browser_control_value, browser_control_config_value
        )
        if browser_validation_error:
            return _config_error_redirect(agent_id, browser_validation_error[:220])

    internet_value = new_internet_provider or None
    internet_config_value = _build_internet_provider_config(internet_value)
    internet_validation_error = _validate_internet_provider_choice(
        internet_value, internet_config_value
    )
    if internet_validation_error:
        return _config_error_redirect(agent_id, internet_validation_error[:220])

    skills_marketplace_value = new_skills_marketplace_provider or None
    existing_skills_provider = _normalize_skills_marketplace_provider_name(
        getattr(existing, "skills_marketplace_provider", None)
    )
    skills_marketplace_base_config = (
        getattr(existing, "skills_marketplace_config", None)
        if skills_marketplace_value
        and existing_skills_provider == skills_marketplace_value
        else None
    )
    skills_marketplace_config_value = _build_skills_marketplace_provider_config(
        skills_marketplace_value,
        skills_marketplace_base_config,
    )
    skills_marketplace_validation_error = _validate_skills_marketplace_provider_choice(
        skills_marketplace_value,
        skills_marketplace_config_value,
    )
    if skills_marketplace_validation_error:
        return _config_error_redirect(
            agent_id, skills_marketplace_validation_error[:220]
        )

    window_update: Dict[str, Any] = {}
    if "context_window_tokens" in form:
        # The window chosen here replaces any agent-level cap, so it is the
        # one the agent actually runs with.
        window_update = {"context_window_tokens": None, "context_window_source": None}
    retrieval_policy_value = getattr(existing, "retrieval_policy", None)
    if "recall_across_present" in form and _recall_scope(existing) != "sealed":
        # Sealing is set on the agent's own page; this is the lighter choice
        # between recalling across conversations and keeping each on its own.
        chosen = dict(retrieval_policy_value or {})
        if _parse_bool(form.get("recall_across")):
            chosen.pop("conversation_scope", None)
        else:
            chosen["conversation_scope"] = "thread"
        retrieval_policy_value = chosen or None
    updated = existing.model_copy(
        update={
            "agent_id": agent_id,
            "retrieval_policy": retrieval_policy_value,
            "instruction": new_instruction or getattr(existing, "instruction", None),
            "memory_types": memory_types_value,
            "max_steps": max_steps_value,
            "persona": persona_payload,
            "llm_config": llm_config,
            "internet_access_provider": internet_value,
            "internet_access_config": internet_config_value,
            "skills_marketplace_provider": skills_marketplace_value,
            "skills_marketplace_config": skills_marketplace_config_value,
            "sandbox_provider": sandbox_value,
            "browser_control": browser_control_config_value,
            "skill_paths": parsed_skill_paths
            if parsed_skill_paths is not None
            else getattr(existing, "skill_paths", None),
            "mcp_servers": getattr(existing, "mcp_servers", None),
            "self_aware": self_aware_enabled_value,
            "self_aware_config": self_aware_config_value,
            "automations_enabled": automations_enabled_value,
            "default_timezone": default_timezone_value,
            **window_update,
        }
    )

    try:
        _state["provider"].store_memagent(updated)
        _persist_mcp_configs_to_toolbox(
            agent_id=agent_id,
            mcp_servers=updated.mcp_servers or [],
            memory_ids=updated.memory_ids or [],
        )
    except Exception as e:
        logger.error(f"Failed to update agent config {agent_id}: {e}")
        return _config_error_redirect(agent_id, str(e)[:200])

    return RedirectResponse(url=f"/agents/{agent_id}/playground", status_code=302)
