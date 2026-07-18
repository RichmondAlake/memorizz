# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Shared helpers used by ``ui/app.py`` and the extracted ``ui/routers/``.

These functions were moved verbatim out of ``app.py`` so that router modules
can use them without importing ``ui.app`` (which would be circular: ``app.py``
imports every router at module import time). They read the connected provider
from the shared ``ui.state._state`` and never import from ``ui.app``.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from .state import _state

logger = logging.getLogger(__name__)

RECENT_NAV_AGENT_LIMIT = 6


def _read_lob_value(value: Any) -> Any:
    """Read Oracle LOB values into plain Python types when possible."""
    reader = getattr(value, "read", None)
    if callable(reader):
        try:
            return reader()
        except Exception:
            return value
    return value


def _to_text(value: Any) -> str:
    """Coerce values (including LOB/bytes) into safe UI strings."""
    value = _read_lob_value(value)
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="ignore")
    return str(value)


def _extract_agent_identifier(agent: Any) -> Optional[str]:
    """Extract a string agent identifier from dict/model values."""
    if isinstance(agent, dict):
        raw_id = agent.get("agent_id") or agent.get("agentId") or agent.get("_id")
    else:
        raw_id = (
            getattr(agent, "agent_id", None)
            or getattr(agent, "agentId", None)
            or getattr(agent, "_id", None)
        )
    if raw_id is None:
        return None
    text = _to_text(raw_id).strip()
    return text or None


def _extract_agent_memory_ids(agent: Any) -> List[str]:
    """Extract normalized memory IDs from a memagent object/dict."""
    if isinstance(agent, dict):
        raw_memory_ids = agent.get("memory_ids") or agent.get("memoryIds") or []
    else:
        raw_memory_ids = getattr(agent, "memory_ids", None) or []

    memory_ids: List[str] = []
    for memory_id in raw_memory_ids:
        text = _to_text(memory_id).strip()
        if text:
            memory_ids.append(text)
    return memory_ids


def _extract_agent_persona_name(agent: Any) -> str:
    """Extract display name for an agent."""
    if isinstance(agent, dict):
        explicit_name = _to_text(agent.get("name")).strip()
    else:
        explicit_name = _to_text(getattr(agent, "name", None)).strip()
    if explicit_name:
        return explicit_name

    if isinstance(agent, dict):
        persona = agent.get("persona")
    else:
        persona = getattr(agent, "persona", None)

    if persona:
        if isinstance(persona, dict):
            name = _to_text(persona.get("name")).strip()
        else:
            name = _to_text(getattr(persona, "name", None)).strip()
        if name:
            return name
    return "Agent"


def _parse_object_id_timestamp(value: Any) -> Optional[float]:
    """Parse Mongo ObjectId timestamps from hex/ObjectId(...) strings."""
    raw = _to_text(value).strip()
    if not raw:
        return None

    if raw.startswith("ObjectId(") and raw.endswith(")"):
        raw = raw[len("ObjectId(") : -1].strip().strip("'").strip('"')

    if len(raw) != 24:
        return None

    try:
        int(raw, 16)
    except ValueError:
        return None

    return float(int(raw[:8], 16))


def _coerce_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of mixed timestamp formats into epoch seconds."""
    value = _read_lob_value(value)
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if hasattr(value, "timestamp") and callable(getattr(value, "timestamp")):
        try:
            return float(value.timestamp())
        except Exception:
            pass

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = value.decode("utf-8", errors="ignore")

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        iso_value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return float(datetime.fromisoformat(iso_value).timestamp())
        except ValueError:
            pass

        try:
            return float(raw)
        except ValueError:
            return None

    return None


def _load_memagent_created_at_map() -> Dict[str, float]:
    """Build a creation-time lookup map keyed by agent_id."""
    provider = _state.get("provider")
    if not provider:
        return {}

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.MEMAGENT) or []
    except Exception as exc:
        logger.debug("Failed to load memagent docs for sorting: %s", exc)
        return {}

    created_by_agent: Dict[str, float] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        raw_id = doc.get("agent_id") or doc.get("agentId") or doc.get("_id")
        agent_id = _to_text(raw_id).strip() if raw_id is not None else ""
        if not agent_id:
            continue

        created_at = doc.get("created_at")
        if created_at is None:
            created_at = doc.get("createdAt")

        timestamp = _coerce_timestamp(created_at)
        if timestamp is None:
            timestamp = _parse_object_id_timestamp(raw_id)

        if timestamp is not None:
            created_by_agent[agent_id] = timestamp

    return created_by_agent


def _agent_created_timestamp(
    agent: Any, created_by_agent: Optional[Dict[str, float]] = None
) -> float:
    """Resolve an agent's creation timestamp for deterministic sorting."""
    agent_id = _extract_agent_identifier(agent)

    if agent_id and created_by_agent and agent_id in created_by_agent:
        return created_by_agent[agent_id]

    if isinstance(agent, dict):
        created_at = agent.get("created_at")
        if created_at is None:
            created_at = agent.get("createdAt")
    else:
        created_at = getattr(agent, "created_at", None)
        if created_at is None:
            created_at = getattr(agent, "createdAt", None)

    timestamp = _coerce_timestamp(created_at)
    if timestamp is not None:
        return timestamp

    if agent_id:
        object_id_timestamp = _parse_object_id_timestamp(agent_id)
        if object_id_timestamp is not None:
            return object_id_timestamp

    return 0.0


def _extract_message_timestamp(message: Any) -> Optional[float]:
    """Extract a comparable timestamp from conversation message payloads."""
    if not isinstance(message, dict):
        return None

    timestamp = _coerce_timestamp(message.get("timestamp"))
    if timestamp is not None:
        return timestamp
    timestamp = _coerce_timestamp(message.get("created_at"))
    if timestamp is not None:
        return timestamp
    return _coerce_timestamp(message.get("createdAt"))


def _retrieve_conversation_history(
    memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve conversation history with provider fallback and deterministic ordering."""
    provider = _state.get("provider")
    memory_id_value = _to_text(memory_id).strip()
    if not provider or not memory_id_value:
        return []

    history: List[Dict[str, Any]] = []

    # Preferred path: provider-level API (supported by all official providers).
    try:
        rows = (
            provider.retrieve_conversation_history_ordered_by_timestamp(
                memory_id=memory_id_value, limit=None
            )
            or []
        )
        history = [row for row in rows if isinstance(row, dict)]
    except Exception as exc:
        logger.debug(
            "Provider history API unavailable for memory '%s': %s",
            memory_id_value,
            exc,
        )

    # Compatibility fallback for custom providers exposing stores only.
    if not history:
        try:
            from ..enums.memory_type import MemoryType

            stores = getattr(provider, "stores", None) or {}
            conversation_store = stores.get(
                MemoryType.CONVERSATION_MEMORY
            ) or stores.get(MemoryType.CONVERSATION_MEMORY.value)
            if conversation_store is not None:
                rows = (
                    conversation_store.retrieve_conversation_history_ordered_by_timestamp(
                        memory_id=memory_id_value, limit=None
                    )
                    or []
                )
                history = [row for row in rows if isinstance(row, dict)]
        except Exception as exc:
            logger.debug(
                "Conversation-store fallback failed for memory '%s': %s",
                memory_id_value,
                exc,
            )

    history.sort(key=lambda row: _extract_message_timestamp(row) or 0.0)
    if limit and limit > 0:
        return history[-limit:]
    return history


def _load_last_run_map_from_conversation_docs(
    agent_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Fallback last-run map built directly from conversation documents."""
    provider = _state.get("provider")
    if not provider:
        return {}

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.CONVERSATION_MEMORY) or []
    except Exception as exc:
        logger.debug("Failed to load conversation docs for last-run fallback: %s", exc)
        return {}

    last_run_by_agent: Dict[str, float] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        if not agent_id:
            continue
        if agent_ids is not None and agent_id not in agent_ids:
            continue

        timestamp = _extract_message_timestamp(doc)
        if timestamp is None:
            continue
        if timestamp > last_run_by_agent.get(agent_id, 0.0):
            last_run_by_agent[agent_id] = timestamp

    return last_run_by_agent


def _load_agent_last_run_map(agents: Optional[List[Any]] = None) -> Dict[str, float]:
    """Build a map of last run timestamp per agent."""
    provider = _state.get("provider")
    if not provider:
        return {}

    if agents is None:
        try:
            agents = provider.list_memagents()
        except Exception as exc:
            logger.debug("Failed to list agents while loading last-run map: %s", exc)
            return {}

    known_agent_ids: Set[str] = set()
    last_run_by_agent: Dict[str, float] = {}
    for agent in agents or []:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        known_agent_ids.add(agent_id)

        latest_timestamp = 0.0
        for memory_id in _extract_agent_memory_ids(agent):
            history = _retrieve_conversation_history(memory_id=memory_id, limit=None)

            for message in history:
                timestamp = _extract_message_timestamp(message)
                if timestamp and timestamp > latest_timestamp:
                    latest_timestamp = timestamp

        if latest_timestamp > 0:
            last_run_by_agent[agent_id] = latest_timestamp

    # Historical fallback: if memory_ids are sparse/missing, infer from conversation docs.
    if known_agent_ids and len(last_run_by_agent) < len(known_agent_ids):
        fallback_map = _load_last_run_map_from_conversation_docs(known_agent_ids)
        for agent_id, timestamp in fallback_map.items():
            if timestamp > last_run_by_agent.get(agent_id, 0.0):
                last_run_by_agent[agent_id] = timestamp

    return last_run_by_agent


def _toolbox_doc_key(doc: Dict[str, Any]) -> Optional[str]:
    """Build a stable dedupe key for toolbox tool docs."""
    if not isinstance(doc, dict):
        return None

    tool_id = _to_text(
        doc.get("tool_id") or doc.get("toolId") or doc.get("_id")
    ).strip()
    name = _to_text(doc.get("name")).strip()
    signature = _to_text(doc.get("signature")).strip()

    if tool_id:
        return f"id:{tool_id}"
    if name or signature:
        return f"{name}|{signature}"
    return None


def _is_toolbox_tool_doc(doc: Dict[str, Any]) -> bool:
    """Return True when a toolbox document represents an executable tool."""
    if not isinstance(doc, dict):
        return False

    tool_type = _to_text(doc.get("tool_type") or doc.get("type")).strip().lower()
    name = _to_text(doc.get("name")).strip()

    if tool_type == "mcp_server_config":
        return False
    if name.startswith("mcp::"):
        return False
    return bool(_toolbox_doc_key(doc))


def _load_toolbox_documents() -> List[Dict[str, Any]]:
    """Load toolbox documents for tool counting."""
    provider = _state.get("provider")
    if not provider:
        return []

    try:
        from ..enums.memory_type import MemoryType

        docs = provider.list_all(MemoryType.TOOLBOX) or []
    except Exception as exc:
        logger.debug("Failed to list toolbox docs: %s", exc)
        return []

    return [doc for doc in docs if isinstance(doc, dict)]


def _count_runtime_tools_for_agent(agent_id: str) -> int:
    """Best-effort runtime tool count by loading the agent tool manager."""
    provider = _state.get("provider")
    agent_id_value = _to_text(agent_id).strip()
    if not provider or not agent_id_value:
        return 0

    try:
        from ..memagent import MemAgent

        agent_instance = MemAgent.load(agent_id_value, memory_provider=provider)
        tool_manager = getattr(agent_instance, "tool_manager", None)
        if not tool_manager:
            return 0
        tool_names = tool_manager.list_tools() or []
        return len(tool_names)
    except Exception as exc:
        logger.debug(
            "Could not compute runtime tool count for %s: %s",
            agent_id_value,
            exc,
        )
        return 0


def _normalize_internet_provider_name(value: Any) -> str:
    """Normalize internet provider values to a stable lowercase name."""
    if isinstance(value, dict):
        value = value.get("provider") or value.get("name")
    return _to_text(value).strip().lower()


def _normalize_skills_marketplace_provider_name(value: Any) -> str:
    """Normalize skills marketplace provider values to a stable lowercase name."""
    if isinstance(value, dict):
        value = value.get("provider") or value.get("name")
    return _to_text(value).strip().lower()


def _normalize_memory_type_values(
    memory_types: Optional[List[Any]],
) -> List[str]:
    """Normalize memory type enums/strings into stable string values."""
    if not memory_types:
        return []

    from ..enums.memory_type import MemoryType

    valid_values = {memory_type.value for memory_type in MemoryType}
    normalized: List[str] = []
    seen = set()
    for item in memory_types:
        value = _to_text(getattr(item, "value", item)).strip().lower()
        if not value or value not in valid_values or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _default_memory_types_for_mode(application_mode: Optional[str]) -> List[str]:
    """Return application-mode default memory types as normalized values."""
    from ..enums import ApplicationMode, ApplicationModeConfig

    normalized_mode = _to_text(application_mode).strip().lower()
    try:
        mode = (
            ApplicationModeConfig.validate_mode(normalized_mode)
            if normalized_mode
            else ApplicationMode.DEFAULT
        )
    except ValueError:
        mode = ApplicationMode.DEFAULT
    return _normalize_memory_type_values(ApplicationModeConfig.get_memory_types(mode))


def _agent_entity_memory_enabled(agent: Any) -> bool:
    """Return whether entity memory is enabled for an agent config."""
    from ..enums.memory_type import MemoryType

    configured_types = _normalize_memory_type_values(
        getattr(agent, "memory_types", None)
    )
    if not configured_types:
        configured_types = _default_memory_types_for_mode(
            getattr(agent, "application_mode", None)
        )
    return MemoryType.ENTITY_MEMORY.value in configured_types


def _provider_supports_entity_memory(provider: Any) -> bool:
    """Best-effort detection for provider entity-memory support."""
    if provider is None:
        return False

    support_fn = getattr(provider, "supports_entity_memory", None)
    if callable(support_fn):
        try:
            return bool(support_fn())
        except Exception:
            return False
    return bool(getattr(provider, "entity_memory_collection", None))


def _extract_agent_tools(agent) -> List[Dict[str, str]]:
    """Extract a simplified tools list from an agent for display in the UI."""
    if not agent:
        return []
    raw_tools = getattr(agent, "tools", None) or []
    result = []
    seen_names: Set[str] = set()
    for tool in raw_tools:
        if isinstance(tool, dict):
            name = _to_text(
                tool.get("name") or tool.get("function", {}).get("name", "Unknown")
            )
            desc = _to_text(
                tool.get("description")
                or tool.get("docstring", "")
                or tool.get("function", {}).get("description", "")
            )
            normalized_name = (name or "Unknown").strip()
            seen_names.add(normalized_name)
            result.append(
                {
                    "name": normalized_name,
                    "description": desc[:120] if desc else "",
                }
            )

    default_context_tools = [
        (
            "context_window_stats_tool",
            "Show the latest context-window token usage metrics.",
        ),
        (
            "list_summary_registry_tool",
            "List summaries generated automatically or on demand.",
        ),
        (
            "fetch_summary_tool",
            "Fetch a stored summary by summary_id.",
        ),
        (
            "autosummarize_conversation",
            "Generate conversation summaries now.",
        ),
    ]
    for tool_name, description in default_context_tools:
        if tool_name in seen_names:
            continue
        result.append({"name": tool_name, "description": description})
        seen_names.add(tool_name)

    sandbox_provider = getattr(agent, "sandbox_provider", None)
    if isinstance(sandbox_provider, dict):
        sandbox_provider = sandbox_provider.get("provider")
    sandbox_provider = _to_text(sandbox_provider).strip()
    if sandbox_provider:
        sandbox_tools = [
            (
                "execute_code",
                f"Execute code in the configured {sandbox_provider} sandbox.",
            ),
            ("sandbox_write_file", "Write a file inside the sandbox filesystem."),
            ("sandbox_read_file", "Read a file from the sandbox filesystem."),
        ]
        for tool_name, description in sandbox_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)

    internet_provider = _normalize_internet_provider_name(
        getattr(agent, "internet_access_provider", None)
    )
    if internet_provider:
        internet_tools = [
            (
                "internet_search",
                f"Search the live internet via {internet_provider}.",
            ),
            (
                "open_web_page",
                "Fetch and summarize webpage content from a URL.",
            ),
        ]
        for tool_name, description in internet_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)

    skills_marketplace_provider = _normalize_skills_marketplace_provider_name(
        getattr(agent, "skills_marketplace_provider", None)
    )
    if skills_marketplace_provider:
        tool_name = "skills_marketplace_search"
        if tool_name not in seen_names:
            result.append(
                {
                    "name": tool_name,
                    "description": (
                        "Search skills from the configured marketplace via sandbox execution."
                    ),
                }
            )
            seen_names.add(tool_name)

    if _agent_entity_memory_enabled(agent) and _provider_supports_entity_memory(
        _state.get("provider")
    ):
        entity_tools = [
            (
                "entity_memory_lookup",
                "Look up structured entity facts for the active thread.",
            ),
            (
                "entity_memory_upsert",
                "Create or update structured entity facts for the active thread.",
            ),
        ]
        for tool_name, description in entity_tools:
            if tool_name in seen_names:
                continue
            result.append({"name": tool_name, "description": description})
            seen_names.add(tool_name)
    return result


def _build_agent_tool_count_map(agents: List[Any]) -> Dict[str, int]:
    """Build per-agent tool counts using embedded tool metadata plus toolbox docs."""
    if not agents:
        return {}

    provider = _state.get("provider")
    toolbox_docs = [
        doc for doc in _load_toolbox_documents() if _is_toolbox_tool_doc(doc)
    ]

    docs_by_agent: Dict[str, List[Dict[str, Any]]] = {}
    docs_by_memory: Dict[str, List[Dict[str, Any]]] = {}
    for doc in toolbox_docs:
        agent_id = _to_text(doc.get("agent_id") or doc.get("agentId")).strip()
        memory_id = _to_text(doc.get("memory_id") or doc.get("memoryId")).strip()

        if agent_id:
            docs_by_agent.setdefault(agent_id, []).append(doc)
        if memory_id:
            docs_by_memory.setdefault(memory_id, []).append(doc)

    counts: Dict[str, int] = {}
    for agent in agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue

        seen_keys: Set[str] = set()
        memory_ids = _extract_agent_memory_ids(agent)

        # Embedded tools (already present on the memagent document).
        for tool_meta in _extract_agent_tools(agent):
            tool_name = _to_text(tool_meta.get("name")).strip()
            if tool_name:
                seen_keys.add(f"name:{tool_name}")

        # Provider-specific retrieval (Oracle) can surface tool associations
        # that may not be obvious from raw toolbox docs.
        if provider and hasattr(provider, "retrieve_tools_for_agent"):
            try:
                if isinstance(agent, dict):
                    tool_access = (
                        _to_text(
                            agent.get("tool_access") or agent.get("toolAccess")
                        ).strip()
                        or "private"
                    )
                else:
                    tool_access = (
                        _to_text(getattr(agent, "tool_access", None)).strip()
                        or "private"
                    )
                provider_tools = (
                    provider.retrieve_tools_for_agent(
                        agent_id=agent_id,
                        tool_access=tool_access,
                        query=None,
                        top_k=200,
                    )
                    or []
                )
                for doc in provider_tools:
                    if isinstance(doc, dict) and _is_toolbox_tool_doc(doc):
                        key = _toolbox_doc_key(doc)
                        if key:
                            seen_keys.add(key)
            except Exception as exc:
                logger.debug(
                    "Provider-specific tool retrieval failed for %s: %s", agent_id, exc
                )

        # Toolbox docs linked directly by agent_id.
        for doc in docs_by_agent.get(agent_id, []):
            key = _toolbox_doc_key(doc)
            if key:
                seen_keys.add(key)

        # Toolbox docs linked through memory IDs.
        for memory_id in memory_ids:
            for doc in docs_by_memory.get(memory_id, []):
                key = _toolbox_doc_key(doc)
                if key:
                    seen_keys.add(key)

        runtime_count = _count_runtime_tools_for_agent(agent_id)
        counts[agent_id] = max(len(seen_keys), runtime_count)

    return counts


def _agent_last_run_timestamp(
    agent: Any, last_run_by_agent: Optional[Dict[str, float]] = None
) -> float:
    """Resolve an agent's latest run timestamp."""
    agent_id = _extract_agent_identifier(agent)
    if agent_id and last_run_by_agent and agent_id in last_run_by_agent:
        return last_run_by_agent[agent_id]
    return 0.0


def _sort_agents_by_last_run_desc(
    agents: List[Any], last_run_by_agent: Optional[Dict[str, float]] = None
) -> List[Any]:
    """Sort agents by last run (newest first), then by creation time."""
    if not agents:
        return []

    last_run_map = last_run_by_agent or _load_agent_last_run_map(agents)
    created_by_agent = _load_memagent_created_at_map()
    indexed = [
        (
            _agent_last_run_timestamp(agent, last_run_map),
            _agent_created_timestamp(agent, created_by_agent),
            idx,
            agent,
        )
        for idx, agent in enumerate(agents)
    ]
    indexed.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [item[3] for item in indexed]


def _build_agent_nav_items(
    active_agent_id: Optional[str] = None,
    agents: Optional[List[Any]] = None,
    last_run_by_agent: Optional[Dict[str, float]] = None,
    limit: int = RECENT_NAV_AGENT_LIMIT,
) -> List[Dict[str, str]]:
    """Build recent agent list for sidebar navigation."""
    if not _state["provider"]:
        return []

    if agents is None:
        try:
            agents = _state["provider"].list_memagents()
        except Exception as exc:
            logger.error(f"Failed to list agents for navigation: {exc}")
            return []

    ordered_agents = _sort_agents_by_last_run_desc(
        agents, last_run_by_agent=last_run_by_agent
    )
    if limit > 0:
        ordered_agents = ordered_agents[:limit]

    items: List[Dict[str, str]] = []
    for agent in ordered_agents:
        agent_id = _extract_agent_identifier(agent)
        if not agent_id:
            continue
        items.append(
            {
                "agent_id": agent_id,
                "name": _extract_agent_persona_name(agent),
                "short_id": agent_id[:8],
                "is_active": agent_id == active_agent_id,
            }
        )

    return items


def _load_agent_knowledge_base(
    agent: Any, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load knowledge-base entries attached to this agent.

    Knowledge base entries are agent-scoped (not thread-scoped): each is
    linked to an agent via a `knowledge_base_id` that the agent references
    in its `knowledge_base_ids` attribute. Returns the same list regardless
    of which conversation thread is active.
    """
    provider = _state.get("provider")
    if not provider or agent is None:
        return []

    kb_ids_raw = getattr(agent, "knowledge_base_ids", None) or []
    if isinstance(kb_ids_raw, dict):
        kb_ids_raw = list(kb_ids_raw.values())
    kb_ids = {_to_text(kid).strip() for kid in kb_ids_raw if _to_text(kid).strip()}
    if not kb_ids:
        return []

    try:
        from ..enums.memory_type import MemoryType

        documents = provider.list_all(MemoryType.KNOWLEDGE_BASE) or []
    except Exception as exc:
        logger.debug("Failed to load knowledge base: %s", exc)
        return []

    filtered: List[Dict[str, Any]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        doc_kb_id = _to_text(
            doc.get("knowledge_base_id") or doc.get("knowledgeBaseId")
        ).strip()
        if not doc_kb_id or doc_kb_id not in kb_ids:
            continue
        content = _to_text(doc.get("content") or "")
        preview = content[:280] + ("…" if len(content) > 280 else "")
        filtered.append(
            {
                "knowledge_base_id": doc_kb_id,
                "namespace": _to_text(doc.get("namespace") or ""),
                "timestamp": doc.get("created_at") or doc.get("createdAt"),
                "content_preview": preview,
                "content_length": len(content),
            }
        )

    filtered.sort(
        key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0,
        reverse=True,
    )
    if limit and limit > 0:
        return filtered[:limit]
    return filtered


# -----------------------------------------------------------------------------
# LLM model catalog and shared agent-config helpers (moved verbatim from app.py)
# -----------------------------------------------------------------------------

LLM_MODEL_CATALOG: Dict[str, List[Dict[str, str]]] = {
    # Source: https://platform.openai.com/docs/models (Latest models section)
    "openai": [
        {"value": "gpt-5.2", "label": "GPT-5.2", "group": "Featured"},
        {"value": "gpt-5.1", "label": "GPT-5.1", "group": "Featured"},
        {"value": "gpt-5", "label": "GPT-5", "group": "GPT-5 Family"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini", "group": "GPT-5 Family"},
        {"value": "gpt-5-nano", "label": "GPT-5 Nano", "group": "GPT-5 Family"},
        {"value": "o3-pro", "label": "o3 Pro", "group": "Reasoning"},
        {"value": "o3", "label": "o3", "group": "Reasoning"},
        {"value": "o4-mini", "label": "o4 Mini", "group": "Reasoning"},
        {"value": "gpt-4.1", "label": "GPT-4.1", "group": "GPT-4.1 Family"},
        {
            "value": "gpt-4.1-mini",
            "label": "GPT-4.1 Mini",
            "group": "GPT-4.1 Family",
        },
        {
            "value": "gpt-4.1-nano",
            "label": "GPT-4.1 Nano",
            "group": "GPT-4.1 Family",
        },
        {"value": "gpt-4o", "label": "GPT-4o", "group": "Legacy"},
        {"value": "gpt-4o-mini", "label": "GPT-4o Mini", "group": "Legacy"},
        {"value": "gpt-4-turbo", "label": "GPT-4 Turbo", "group": "Legacy"},
    ],
    # Azure OpenAI uses deployment names. These IDs are convenient defaults.
    "azure": [
        {"value": "gpt-5.2", "label": "GPT-5.2 deployment", "group": "Current"},
        {"value": "gpt-5", "label": "GPT-5 deployment", "group": "Current"},
        {"value": "gpt-5-mini", "label": "GPT-5 Mini deployment", "group": "Current"},
        {"value": "o3", "label": "o3 deployment", "group": "Reasoning"},
        {"value": "o4-mini", "label": "o4 Mini deployment", "group": "Reasoning"},
        {"value": "gpt-4.1", "label": "GPT-4.1 deployment", "group": "Previous"},
        {"value": "gpt-4o", "label": "GPT-4o deployment", "group": "Previous"},
    ],
    # HuggingFace Hub repo IDs. Custom input still accepts arbitrary repos;
    # this list is just a convenience starting point of widely-used instruct
    # models. Availability check at /api/huggingface/installed reflects what
    # the user has actually downloaded into their HF cache.
    "huggingface": [
        # Meta Llama
        {
            "value": "meta-llama/Llama-3.3-70B-Instruct",
            "label": "Llama 3.3 70B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Llama-3.2-3B-Instruct",
            "label": "Llama 3.2 3B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Llama-3.2-1B-Instruct",
            "label": "Llama 3.2 1B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "label": "Llama 3.1 8B Instruct",
            "group": "Meta",
        },
        {
            "value": "meta-llama/Meta-Llama-3-8B-Instruct",
            "label": "Llama 3 8B Instruct",
            "group": "Meta",
        },
        # Alibaba Qwen
        {"value": "Qwen/Qwen3-32B", "label": "Qwen 3 32B", "group": "Alibaba"},
        {"value": "Qwen/Qwen3-8B", "label": "Qwen 3 8B", "group": "Alibaba"},
        {"value": "Qwen/Qwen3-4B", "label": "Qwen 3 4B", "group": "Alibaba"},
        {
            "value": "Qwen/Qwen2.5-7B-Instruct",
            "label": "Qwen 2.5 7B Instruct",
            "group": "Alibaba",
        },
        # Google Gemma 4 (April 2026, Apache 2.0, gated — accept license once on HF)
        {
            "value": "google/gemma-4-E2B-it",
            "label": "Gemma 4 E2B Instruct (2.3B eff)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-E4B-it",
            "label": "Gemma 4 E4B Instruct (4.5B eff)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-26B-A4B-it",
            "label": "Gemma 4 26B A4B Instruct (MoE)",
            "group": "Google",
        },
        {
            "value": "google/gemma-4-31B-it",
            "label": "Gemma 4 31B Instruct (Dense)",
            "group": "Google",
        },
        # Google Gemma 3
        {"value": "google/gemma-3-27b-it", "label": "Gemma 3 27B", "group": "Google"},
        {"value": "google/gemma-3-4b-it", "label": "Gemma 3 4B", "group": "Google"},
        {"value": "google/gemma-2-9b-it", "label": "Gemma 2 9B", "group": "Google"},
        # Mistral
        {
            "value": "mistralai/Mistral-7B-Instruct-v0.3",
            "label": "Mistral 7B Instruct v0.3",
            "group": "Mistral AI",
        },
        {
            "value": "mistralai/Mistral-Small-Instruct-2409",
            "label": "Mistral Small Instruct (24B)",
            "group": "Mistral AI",
        },
        # Microsoft Phi
        {"value": "microsoft/phi-4", "label": "Phi-4 (14B)", "group": "Microsoft"},
        {
            "value": "microsoft/Phi-4-mini-instruct",
            "label": "Phi-4 mini Instruct",
            "group": "Microsoft",
        },
        # DeepSeek R1 distills
        {
            "value": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "label": "DeepSeek R1 Distill Llama 8B",
            "group": "DeepSeek",
        },
    ],
    # Anthropic Claude models
    "anthropic": [
        {"value": "claude-opus-4-6", "label": "Claude Opus 4.6", "group": "Latest"},
        {"value": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "group": "Latest"},
        {
            "value": "claude-haiku-4-5-20251001",
            "label": "Claude Haiku 4.5",
            "group": "Latest",
        },
        {
            "value": "claude-sonnet-4-5-20250929",
            "label": "Claude Sonnet 4.5",
            "group": "Previous",
        },
        {"value": "claude-opus-4-5", "label": "Claude Opus 4.5", "group": "Previous"},
        {"value": "claude-sonnet-4-0", "label": "Claude Sonnet 4", "group": "Previous"},
        {"value": "claude-opus-4-0", "label": "Claude Opus 4", "group": "Previous"},
        {
            "value": "claude-3-5-sonnet-20241022",
            "label": "Claude 3.5 Sonnet",
            "group": "Legacy",
        },
        {
            "value": "claude-3-haiku-20240307",
            "label": "Claude 3 Haiku",
            "group": "Legacy",
        },
    ],
    # Ollama local models. Values are the exact Ollama tags users `ollama pull`,
    # so the dropdown can never offer a model that doesn't exist on the registry
    # (the older `gemma4` / `qwen3.6` entries here resulted in 404s at chat time).
    "ollama": [
        # Meta Llama
        {"value": "llama3.3:70b", "label": "Llama 3.3 (70B)", "group": "Meta"},
        {"value": "llama3.2:3b", "label": "Llama 3.2 (3B)", "group": "Meta"},
        {"value": "llama3.2:1b", "label": "Llama 3.2 (1B)", "group": "Meta"},
        {"value": "llama3.1:8b", "label": "Llama 3.1 (8B)", "group": "Meta"},
        # Google Gemma 4 — needs Ollama daemon >= 0.22.1 (April 28 2026 release).
        # E2B/E4B are the on-device "effective-N" sizes, the 26B MoE and 31B
        # dense are the workstation tier. Tags via `ollama pull gemma4:<size>`.
        # `gemma4:latest` aliases to e4b — surfaced as its own entry because
        # that's what `ollama pull gemma4` (no tag) writes to disk.
        {
            "value": "gemma4:latest",
            "label": "Gemma 4 (latest, = E4B)",
            "group": "Google",
        },
        {"value": "gemma4:e2b", "label": "Gemma 4 (E2B, on-device)", "group": "Google"},
        {"value": "gemma4:e4b", "label": "Gemma 4 (E4B, on-device)", "group": "Google"},
        {"value": "gemma4:26b", "label": "Gemma 4 (26B A4B, MoE)", "group": "Google"},
        {"value": "gemma4:31b", "label": "Gemma 4 (31B Dense)", "group": "Google"},
        # Google Gemma 3 — multiple sizes; gemma3n is the multimodal variant.
        {"value": "gemma3:27b", "label": "Gemma 3 (27B)", "group": "Google"},
        {"value": "gemma3:12b", "label": "Gemma 3 (12B)", "group": "Google"},
        {"value": "gemma3:4b", "label": "Gemma 3 (4B)", "group": "Google"},
        {"value": "gemma3:1b", "label": "Gemma 3 (1B)", "group": "Google"},
        {
            "value": "gemma3n:e4b",
            "label": "Gemma 3n (E4B, multimodal)",
            "group": "Google",
        },
        # Alibaba Qwen 3 — chat sizes plus the MoE and coder-specialized tags.
        {"value": "qwen3:32b", "label": "Qwen 3 (32B)", "group": "Alibaba"},
        {"value": "qwen3:14b", "label": "Qwen 3 (14B)", "group": "Alibaba"},
        {"value": "qwen3:8b", "label": "Qwen 3 (8B)", "group": "Alibaba"},
        {"value": "qwen3:4b", "label": "Qwen 3 (4B)", "group": "Alibaba"},
        {"value": "qwen3:1.7b", "label": "Qwen 3 (1.7B)", "group": "Alibaba"},
        {"value": "qwen3:0.6b", "label": "Qwen 3 (0.6B)", "group": "Alibaba"},
        {
            "value": "qwen3:30b-a3b",
            "label": "Qwen 3 (30B-A3B, MoE)",
            "group": "Alibaba",
        },
        {"value": "qwen3-coder:30b", "label": "Qwen 3 Coder (30B)", "group": "Alibaba"},
        # Mistral
        {
            "value": "mistral-small:24b",
            "label": "Mistral Small (24B)",
            "group": "Mistral AI",
        },
        {
            "value": "mistral-nemo:12b",
            "label": "Mistral Nemo (12B)",
            "group": "Mistral AI",
        },
        {"value": "mistral:7b", "label": "Mistral (7B)", "group": "Mistral AI"},
        # DeepSeek R1 reasoning
        {"value": "deepseek-r1:14b", "label": "DeepSeek R1 (14B)", "group": "DeepSeek"},
        {"value": "deepseek-r1:7b", "label": "DeepSeek R1 (7B)", "group": "DeepSeek"},
        {
            "value": "deepseek-r1:1.5b",
            "label": "DeepSeek R1 (1.5B)",
            "group": "DeepSeek",
        },
        # Microsoft
        {"value": "phi4:14b", "label": "Phi-4 (14B)", "group": "Microsoft"},
    ],
    # MLX (Apple Silicon native). mlx-community/* repos ship pre-quantized
    # weights — Google explicitly recommends MLX for Gemma 4 on Macs.
    # Requires a native arm64 Python — `pip install memorizz[mlx]` will
    # fail under Rosetta. Custom repo IDs are still accepted.
    "mlx": [
        # Google Gemma 4 (4-bit MLX quants — fast on Apple Silicon)
        {
            "value": "mlx-community/gemma-4-E2B-it-4bit",
            "label": "Gemma 4 E2B 4-bit",
            "group": "Google",
        },
        {
            "value": "mlx-community/gemma-4-E4B-it-4bit",
            "label": "Gemma 4 E4B 4-bit",
            "group": "Google",
        },
        {
            "value": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "label": "Gemma 4 26B A4B 4-bit (MoE)",
            "group": "Google",
        },
        # Llama
        {
            "value": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "label": "Llama 3.2 3B 4-bit",
            "group": "Meta",
        },
        {
            "value": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "label": "Llama 3.2 1B 4-bit",
            "group": "Meta",
        },
        # Qwen
        {
            "value": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "label": "Qwen 2.5 7B 4-bit",
            "group": "Alibaba",
        },
        {
            "value": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "label": "Qwen 2.5 3B 4-bit",
            "group": "Alibaba",
        },
        # Mistral / Microsoft
        {
            "value": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "label": "Mistral 7B Instruct 4-bit",
            "group": "Mistral AI",
        },
        {
            "value": "mlx-community/Phi-3.5-mini-instruct-4bit",
            "label": "Phi-3.5 mini 4-bit",
            "group": "Microsoft",
        },
    ],
    # OpenAI-compatible local servers (llama.cpp's `llama-server`, LM Studio,
    # vLLM, mlx_lm.server, etc.). The model name is whatever the local server
    # exposes — we forward it verbatim. Most servers accept any string; some
    # (LM Studio) echo the loaded model id back via /v1/models.
    #
    # Two groups of entries: GGUF repos for `llama-server -hf <repo>` and
    # `mlx-community/*` repos for `python -m mlx_lm.server --model <repo>`.
    # The agent-form JS detects which family the user picked and shows the
    # right startup command in the hint.
    "local-openai": [
        # GGUF — for llama.cpp's `llama-server`, LM Studio, llamafile
        {
            "value": "ggml-org/gemma-4-E2B-it-GGUF",
            "label": "Gemma 4 E2B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "ggml-org/gemma-4-E4B-it-GGUF",
            "label": "Gemma 4 E4B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/gemma-3-1b-it-GGUF",
            "label": "Gemma 3 1B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "label": "Llama 3.2 3B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        {
            "value": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "label": "Qwen 2.5 7B (GGUF, llama.cpp)",
            "group": "GGUF — llama.cpp / LM Studio",
        },
        # MLX — for `mlx_lm.server` running in a native arm64 sidecar venv.
        # Use this when memorizz itself is on an x86_64/Rosetta env: the MLX
        # process lives in a separate native arm64 Python and memorizz talks
        # to it over OpenAI-compatible HTTP. Pairs with the in-process
        # "MLX (Apple Silicon)" provider, which only works when memorizz's
        # own Python is arm64.
        {
            "value": "mlx-community/gemma-4-E2B-it-4bit",
            "label": "Gemma 4 E2B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/gemma-4-E4B-it-4bit",
            "label": "Gemma 4 E4B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "label": "Gemma 4 26B A4B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "label": "Qwen 2.5 7B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
        {
            "value": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "label": "Llama 3.2 3B 4-bit (MLX server)",
            "group": "MLX — mlx_lm.server",
        },
    ],
}

DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL_BY_PROVIDER = {
    "openai": "gpt-5.2",
    "azure": "gpt-5",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "anthropic": "claude-sonnet-4-5-20250929",
    "ollama": "llama3.1:8b",
    "mlx": "mlx-community/gemma-4-E2B-it-4bit",
    "local-openai": "ggml-org/gemma-4-E2B-it-GGUF",
}


def _normalize_llm_provider(value: Any) -> str:
    """Normalize user-configured LLM provider names."""
    provider = _to_text(value).strip().lower()
    if provider in LLM_MODEL_CATALOG:
        return provider
    return DEFAULT_LLM_PROVIDER


def _get_default_llm_provider() -> str:
    """Resolve default provider from environment with safe fallback."""
    return _normalize_llm_provider(os.environ.get("MEMORIZZ_DEFAULT_LLM_PROVIDER", ""))


def _get_default_llm_model(provider: Optional[str] = None) -> str:
    """Resolve default model/deployment from environment with provider fallback."""
    env_value = _to_text(os.environ.get("MEMORIZZ_DEFAULT_LLM_MODEL", "")).strip()
    if env_value:
        return env_value

    normalized_provider = _normalize_llm_provider(
        provider or _get_default_llm_provider()
    )
    return DEFAULT_LLM_MODEL_BY_PROVIDER.get(
        normalized_provider, DEFAULT_LLM_MODEL_BY_PROVIDER[DEFAULT_LLM_PROVIDER]
    )


def _build_internet_provider_config(
    provider_name: Optional[str], base_config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Build internet provider config with optional default API key fallback."""
    normalized_provider = _normalize_internet_provider_name(provider_name)
    if not normalized_provider:
        return None

    config: Dict[str, Any] = {}
    if isinstance(base_config, dict):
        for key, value in base_config.items():
            if key == "provider":
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            config[key] = value

    if "api_key" not in config:
        default_key = _to_text(
            os.environ.get("MEMORIZZ_DEFAULT_INTERNET_PROVIDER_API_KEY", "")
        ).strip()
        if default_key:
            config["api_key"] = default_key

    return config or None


def _validate_internet_provider_choice(
    internet_provider: Any,
    internet_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Validate internet provider configuration and runtime prerequisites.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    provider_name = _normalize_internet_provider_name(internet_provider)
    if not provider_name:
        return None

    try:
        from ..internet_access import create_internet_access_provider

        config = _build_internet_provider_config(provider_name, internet_config) or {}
        provider = create_internet_access_provider(provider_name, config)
        if not provider:
            return (
                f"Unknown internet provider '{provider_name}'. "
                "Supported providers: tavily, firecrawl, offline."
            )
        try:
            provider.close()
        except Exception:
            pass
        return None
    except Exception as exc:
        message = _to_text(exc).strip()
        if message:
            return message
        return f"Failed to initialize internet provider '{provider_name}'."


def _build_skills_marketplace_provider_config(
    provider_name: Optional[str], base_config: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """Build skills marketplace config with API key fallback from Settings env."""
    normalized_provider = _normalize_skills_marketplace_provider_name(provider_name)
    if not normalized_provider:
        return None

    config: Dict[str, Any] = {}
    if isinstance(base_config, dict):
        for key, value in base_config.items():
            if key == "provider":
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            config[key] = value

    if normalized_provider == "skillsmp":
        if "api_key" not in config:
            default_key = _to_text(os.environ.get("SKILLSMP_API_KEY", "")).strip()
            if default_key:
                config["api_key"] = default_key
        if "base_url" not in config:
            config["base_url"] = "https://skillsmp.com"
    elif normalized_provider == "vercel":
        if "github_token" not in config:
            default_token = _to_text(os.environ.get("GITHUB_TOKEN", "")).strip()
            if default_token:
                config["github_token"] = default_token

    return config or None


def _validate_skills_marketplace_provider_choice(
    skills_marketplace_provider: Any,
    skills_marketplace_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Validate skills marketplace provider configuration.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    provider_name = _normalize_skills_marketplace_provider_name(
        skills_marketplace_provider
    )
    if not provider_name:
        return None

    if provider_name not in ("skillsmp", "vercel"):
        return (
            f"Unknown skills marketplace provider '{provider_name}'. "
            "Supported providers: skillsmp, vercel."
        )

    config = (
        _build_skills_marketplace_provider_config(
            provider_name, skills_marketplace_config
        )
        or {}
    )

    if provider_name == "skillsmp":
        api_key = _to_text(config.get("api_key", "")).strip()
        if not api_key:
            return "Skills Marketplace requires an API key. Add `SKILLSMP_API_KEY` in Settings."

    # Vercel provider works without a token (public GitHub access),
    # but rate limits are better with GITHUB_TOKEN.

    return None


def _validate_sandbox_provider_choice(
    sandbox_provider: Any,
) -> Optional[str]:
    """
    Validate sandbox provider configuration and runtime prerequisites.

    Returns:
        ``None`` when valid, otherwise an actionable error message.
    """
    if not sandbox_provider:
        return None

    resolved_provider = _resolve_sandbox_provider_config(sandbox_provider)
    provider_name = ""
    provider_config: Dict[str, Any] = {}

    if isinstance(resolved_provider, dict):
        provider_name = _to_text(resolved_provider.get("provider")).strip().lower()
        provider_config = {
            key: value for key, value in resolved_provider.items() if key != "provider"
        }
    else:
        provider_name = _to_text(resolved_provider).strip().lower()

    if not provider_name:
        return None

    if provider_name == "graalpy":
        mode = _to_text(provider_config.get("mode", "")).strip().lower()
        wrapper_jar = _to_text(provider_config.get("java_wrapper_jar", "")).strip()
        if mode == "java_wrapper" and not wrapper_jar:
            return (
                "GraalPy internet access is disabled, but `GRAALPY_JAVA_WRAPPER_JAR` "
                "is not set. Add the wrapper JAR path in Settings or re-enable internet."
            )

    try:
        from ..sandbox.base import create_sandbox_provider

        provider = create_sandbox_provider(provider_name, provider_config)
        if not provider:
            return (
                f"Unknown sandbox provider '{provider_name}'. "
                "Supported providers: e2b, daytona, graalpy."
            )
        return None
    except Exception as exc:
        message = _to_text(exc).strip()
        if message:
            return message
        return f"Failed to initialize sandbox provider '{provider_name}'."


def _graalpy_internet_access_enabled() -> bool:
    """Return whether GraalPy should run with internet access."""
    raw_value = _to_text(os.environ.get("MEMORIZZ_GRAALPY_INTERNET_ACCESS", "")).strip()
    if not raw_value:
        return True
    return _parse_bool(raw_value)


def _build_graalpy_default_sandbox_config() -> Dict[str, Any]:
    """
    Build GraalPy provider config from UI settings.

    - Internet enabled  -> subprocess mode
    - Internet disabled -> java_wrapper mode with UNTRUSTED policy
    """
    config: Dict[str, Any] = {"provider": "graalpy"}

    graalpy_path = _to_text(os.environ.get("GRAALPY_PATH", "")).strip()
    if graalpy_path:
        config["graalpy_path"] = graalpy_path

    if _graalpy_internet_access_enabled():
        config["mode"] = "subprocess"
        return config

    config["mode"] = "java_wrapper"
    config["sandbox_policy"] = "UNTRUSTED"
    wrapper_jar = _to_text(os.environ.get("GRAALPY_JAVA_WRAPPER_JAR", "")).strip()
    if wrapper_jar:
        config["java_wrapper_jar"] = wrapper_jar
    return config


def _resolve_sandbox_provider_config(sandbox_provider: Any) -> Any:
    """
    Resolve sandbox provider input into the effective provider config.

    For GraalPy string inputs, this applies Settings-driven defaults so runtime
    behavior matches the Settings page toggles.
    """
    if not sandbox_provider:
        return None
    if isinstance(sandbox_provider, dict):
        return sandbox_provider

    provider_name = _to_text(sandbox_provider).strip().lower()
    if provider_name != "graalpy":
        return provider_name
    return _build_graalpy_default_sandbox_config()


def _load_thread_messages(
    memory_id: str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load and sort conversation messages for a single thread memory_id."""
    provider = _state.get("provider")
    normalized_memory_id = _to_text(memory_id).strip()
    if not provider or not normalized_memory_id:
        return []

    try:
        history = (
            provider.retrieve_conversation_history_ordered_by_timestamp(
                memory_id=normalized_memory_id, limit=limit
            )
            or []
        )
    except Exception as exc:
        logger.debug(
            "Failed to load conversation history for memory_id %s: %s",
            normalized_memory_id,
            exc,
        )
        return []

    history.sort(key=lambda item: _coerce_timestamp(item.get("timestamp")) or 0.0)
    return history


def _build_agent_threads(agent: Any) -> List[Dict[str, Any]]:
    """Build thread metadata from agent memory_ids for playground display."""
    if not _state.get("provider") or not agent:
        return []

    memory_ids = getattr(agent, "memory_ids", None) or []
    if not isinstance(memory_ids, list):
        return []

    seen: Set[str] = set()
    thread_rows: List[Dict[str, Any]] = []

    for raw_memory_id in memory_ids:
        memory_id = _to_text(raw_memory_id).strip()
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)

        history = _load_thread_messages(memory_id, limit=200)

        message_count = len(history)
        last_msg = history[-1] if history else {}
        last_role = _to_text(last_msg.get("role")).strip().lower() if last_msg else ""
        last_content = _to_text(last_msg.get("content") or last_msg.get("text", ""))
        if len(last_content) > 120:
            last_content = f"{last_content[:117]}..."

        last_ts_raw = last_msg.get("timestamp") if last_msg else None
        last_ts = _coerce_timestamp(last_ts_raw)
        if last_ts is not None:
            last_activity = datetime.fromtimestamp(last_ts).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_activity = _to_text(last_ts_raw).strip() if last_ts_raw else "—"

        thread_rows.append(
            {
                "memory_id": memory_id,
                "message_count": message_count,
                "last_role": last_role or "—",
                "last_content": last_content or "—",
                "last_activity": last_activity,
                "last_ts": last_ts or 0.0,
            }
        )

    thread_rows.sort(key=lambda item: item.get("last_ts", 0.0), reverse=True)
    return thread_rows


def _build_memory_types_for_agent(
    application_mode: Optional[str],
    enable_entity_memory: bool,
    enable_workflow_memory: bool,
    existing_memory_types: Optional[List[Any]] = None,
) -> List[str]:
    """Build memory type configuration from mode defaults + explicit memory toggles."""
    from ..enums.memory_type import MemoryType

    default_values = _default_memory_types_for_mode(application_mode)
    existing_values = _normalize_memory_type_values(existing_memory_types)

    ordered_values: List[str] = []
    seen = set()
    for memory_type in existing_values + default_values:
        if memory_type in seen:
            continue
        ordered_values.append(memory_type)
        seen.add(memory_type)

    entity_value = MemoryType.ENTITY_MEMORY.value
    if enable_entity_memory:
        if entity_value not in seen:
            ordered_values.append(entity_value)
            seen.add(entity_value)
    else:
        ordered_values = [value for value in ordered_values if value != entity_value]

    workflow_value = MemoryType.WORKFLOW_MEMORY.value
    if enable_workflow_memory:
        if workflow_value not in seen:
            ordered_values.append(workflow_value)
            seen.add(workflow_value)
    else:
        ordered_values = [value for value in ordered_values if value != workflow_value]

    summaries_value = MemoryType.SUMMARIES.value
    if summaries_value not in seen:
        ordered_values.append(summaries_value)

    return ordered_values


def _agent_workflow_memory_enabled(agent: Any) -> bool:
    """Return whether workflow memory is enabled for an agent config."""
    from ..enums.memory_type import MemoryType

    configured_types = _normalize_memory_type_values(
        getattr(agent, "memory_types", None)
    )
    if not configured_types:
        configured_types = _default_memory_types_for_mode(
            getattr(agent, "application_mode", None)
        )
    return MemoryType.WORKFLOW_MEMORY.value in configured_types


def _parse_bool(value: Optional[str]) -> bool:
    """Parse checkbox-like values into bools."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _parse_self_aware_root_paths(value: Optional[str]) -> List[str]:
    """Parse newline/comma-delimited self-aware root paths."""
    if not value:
        return []
    raw_parts = str(value).replace("\n", ",").split(",")
    roots: List[str] = []
    seen = set()
    for raw in raw_parts:
        path = _to_text(raw).strip()
        if not path or path in seen:
            continue
        roots.append(path)
        seen.add(path)
    return roots


def _build_self_aware_config(
    root_paths: Optional[List[str]],
    allow_writes: bool = False,
    allow_deletes: bool = False,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build normalized self-aware config payload."""
    config = dict(base_config) if isinstance(base_config, dict) else {}
    normalized_roots = []
    seen = set()
    for root in root_paths or []:
        text = _to_text(root).strip()
        if not text or text in seen:
            continue
        normalized_roots.append(text)
        seen.add(text)

    config["root_paths"] = normalized_roots
    config["allow_writes"] = bool(allow_writes)
    config["allow_deletes"] = bool(allow_deletes) if allow_writes else False
    config["policy_version"] = _to_text(config.get("policy_version")).strip() or "v1"

    def _coerce_limit(key: str, default: int, min_value: int, max_value: int) -> int:
        try:
            value = int(config.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(min_value, min(max_value, value))

    config["timeout_seconds"] = _coerce_limit("timeout_seconds", 30, 1, 300)
    config["max_output_chars"] = _coerce_limit("max_output_chars", 50000, 1024, 1000000)
    config["max_file_read_bytes"] = _coerce_limit(
        "max_file_read_bytes", 250000, 1024, 5000000
    )
    config["max_file_write_bytes"] = _coerce_limit(
        "max_file_write_bytes", 250000, 1, 5000000
    )
    return config


def _validate_self_aware_config(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """Validate self-aware config payload."""
    if config is None:
        return None
    if not isinstance(config, dict):
        return "Self-aware config must be an object."

    root_paths = config.get("root_paths")
    if root_paths is None:
        root_paths = []
    if not isinstance(root_paths, list):
        return "Self-aware root paths must be an array."
    for idx, root in enumerate(root_paths):
        if not isinstance(root, str) or not root.strip():
            return f"Self-aware root path #{idx + 1} must be a non-empty string."

    allow_writes = bool(config.get("allow_writes", False))
    allow_deletes = bool(config.get("allow_deletes", False))
    if allow_deletes and not allow_writes:
        return "Self-aware deletes require writes to be enabled."
    return None


def _persist_mcp_configs_to_toolbox(
    agent_id: str, mcp_servers: List[Dict[str, Any]], memory_ids: List[str]
) -> None:
    """Persist MCP server JSON payloads into toolbox memory."""
    if not _state["provider"] or not mcp_servers:
        return

    from ..enums.memory_type import MemoryType

    memory_id = memory_ids[0] if memory_ids else None

    for server in mcp_servers:
        server_name = str(server.get("name", "")).strip()
        if not server_name:
            continue
        payload = {
            "_id": f"{agent_id}:mcp:{server_name}",
            "tool_id": f"{agent_id}:mcp:{server_name}",
            "name": f"mcp::{server_name}",
            "description": f"MCP server config for {server_name}",
            "signature": "mcp_server_config(server_json)",
            "docstring": "Stored MCP server configuration JSON.",
            "tool_type": "mcp_server_config",
            "type": "mcp_server_config",
            "parameters": server,
            "agent_id": agent_id,
            "memory_id": memory_id,
        }
        try:
            _state["provider"].store(payload, memory_store_type=MemoryType.TOOLBOX)
        except Exception as exc:
            logger.warning(
                "Failed to persist MCP server config '%s' for agent %s: %s",
                server_name,
                agent_id,
                exc,
            )
