# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Tenant-safe adapter from MCP tools to Memorizz agents and providers."""

from __future__ import annotations

import inspect
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..approval import (
    ApprovalStateError,
    ApprovalStatus,
    ApprovalStore,
    default_approval_store,
)
from ..enums.memory_type import MemoryType
from ..mcp.security import redact
from .auth import RequestIdentity
from .config import EXECUTE_SCOPE, READ_SCOPE, WRITE_SCOPE, MemorizzMCPServerConfig

logger = logging.getLogger(__name__)

_TENANT_MEMORY_TYPES = frozenset(
    {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
        MemoryType.SHORT_TERM_MEMORY,
        MemoryType.WORKFLOW_MEMORY,
        MemoryType.SUMMARIES,
        MemoryType.SEMANTIC_CACHE,
        MemoryType.ENTITY_MEMORY,
        MemoryType.TOOL_LOG,
    }
)
_WRITABLE_MEMORY_TYPES = frozenset(
    {MemoryType.KNOWLEDGE_BASE, MemoryType.SHORT_TERM_MEMORY}
)
_SENSITIVE_AGENT_FIELDS = {
    "model",
    "tools",
    "llm_config",
    "embedding_config",
    "internet_access_config",
    "skills_marketplace_config",
    "sandbox_provider",
    "whatsapp_config",
    "mcp_servers",
}


class MemorizzServerError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _accepts_keyword(function: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if str(key).lower() == "embedding":
                continue
            result[str(key)] = _json_value(item)
        return redact(result)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _document_field(document: Dict[str, Any], field: str) -> Any:
    if field in document:
        return document.get(field)
    content = document.get("content")
    return content.get(field) if isinstance(content, dict) else None


def _document_user_id(document: Dict[str, Any]) -> Any:
    return _document_field(document, "user_id")


class MemorizzRuntime:
    """Lazy, synchronized access to one Memorizz deployment."""

    def __init__(
        self,
        config: MemorizzMCPServerConfig,
        *,
        provider: Any = None,
        session_builder: Optional[Callable[..., Any]] = None,
        approval_store: Optional[ApprovalStore] = None,
    ) -> None:
        self.config = config
        self._provider = provider
        self._session_builder = session_builder
        self._provider_lock = threading.RLock()
        self._agents: Dict[str, Any] = {}
        self._agent_locks: Dict[str, threading.RLock] = {}
        self._conversation_state: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self.approval_store = approval_store or default_approval_store()

    @staticmethod
    def _approval_owner(identity: RequestIdentity) -> str:
        return f"mcp-server:{identity.principal or 'anonymous'}"

    @property
    def provider(self):
        with self._provider_lock:
            if self._provider is None:
                from ..cli import agent_factory
                from ..cli.config import load_layered_env

                load_layered_env()
                warnings: List[str] = []
                self._provider = agent_factory.detect_memory_provider({}, warnings)
                for warning in warnings:
                    logger.warning("MCP server provider: %s", warning)
            return self._provider

    def _require_scope(self, identity: RequestIdentity, scope: str) -> None:
        if scope not in identity.scopes:
            raise MemorizzServerError(
                "insufficient_scope", f"This operation requires the {scope} scope"
            )
        if scope == WRITE_SCOPE and not self.config.allow_writes:
            raise MemorizzServerError(
                "writes_disabled", "Memory writes are disabled by the server operator"
            )
        if scope == EXECUTE_SCOPE and not self.config.allow_agent_execution:
            raise MemorizzServerError(
                "execution_disabled",
                "Agent execution is disabled by the server operator",
            )

    def _normalize_memory_type(
        self, value: str, identity: RequestIdentity, *, writable: bool = False
    ) -> MemoryType:
        try:
            memory_type = MemoryType(str(value or "").strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in MemoryType)
            raise MemorizzServerError(
                "invalid_memory_type", f"Unsupported memory type. Choose: {supported}"
            ) from exc
        if self.config.transport != "stdio" and memory_type not in _TENANT_MEMORY_TYPES:
            raise MemorizzServerError(
                "memory_type_not_tenant_scoped",
                f"Remote access to {memory_type.value} is not tenant scoped",
            )
        if writable and memory_type not in _WRITABLE_MEMORY_TYPES:
            choices = ", ".join(item.value for item in _WRITABLE_MEMORY_TYPES)
            raise MemorizzServerError(
                "memory_type_not_writable", f"Direct writes are limited to: {choices}"
            )
        return memory_type

    def _agent_public(self, agent: Any) -> Dict[str, Any]:
        value = _json_value(agent)
        if not isinstance(value, dict):
            value = {
                "agent_id": getattr(agent, "agent_id", None),
                "name": getattr(agent, "name", None),
            }
        for field in _SENSITIVE_AGENT_FIELDS:
            value.pop(field, None)
        return {
            key: value.get(key)
            for key in (
                "agent_id",
                "name",
                "instruction",
                "application_mode",
                "is_favorite",
                "automations_enabled",
                "default_timezone",
            )
            if value.get(key) is not None
        }

    def _is_agent_exposed(self, agent_id: str, identity: RequestIdentity) -> bool:
        exposed = self.config.exposed_agent_ids
        if self.config.transport == "stdio" and exposed is None:
            return True
        return bool(exposed and agent_id in exposed)

    def _assert_agent_exposed(self, agent_id: str, identity: RequestIdentity) -> None:
        if not self._is_agent_exposed(agent_id, identity):
            raise MemorizzServerError("agent_not_found", "Agent was not found")

    def server_info(self, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        from .. import __version__

        return {
            "ok": True,
            "name": "memorizz",
            "version": __version__,
            "principal": identity.principal,
            "authenticated": identity.authenticated,
            "scopes": sorted(identity.scopes),
            "policy": self.config.public_dict(),
            "memory_types": [
                item.value
                for item in MemoryType
                if self.config.transport == "stdio" or item in _TENANT_MEMORY_TYPES
            ],
            "directly_writable_memory_types": [
                item.value for item in _WRITABLE_MEMORY_TYPES
            ],
        }

    def list_agents(self, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        try:
            agents = list(self.provider.list_memagents() or [])
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not list agents"
            ) from exc
        visible = []
        for agent in agents:
            agent_id = str(
                getattr(agent, "agent_id", None)
                or (agent.get("agent_id") if isinstance(agent, dict) else "")
                or ""
            )
            if agent_id and self._is_agent_exposed(agent_id, identity):
                visible.append(self._agent_public(agent))
        return {"ok": True, "agents": visible, "count": len(visible)}

    def get_agent(self, agent_id: str, identity: RequestIdentity) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        normalized = str(agent_id or "").strip()
        self._assert_agent_exposed(normalized, identity)
        try:
            agent = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the agent"
            ) from exc
        if not agent:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        return {"ok": True, "agent": self._agent_public(agent)}

    def _default_session(self):
        if self._session_builder:
            return self._session_builder(memory_provider=self.provider)
        from ..cli import agent_factory
        from ..cli.config import load_layered_env

        load_layered_env()
        try:
            llm_config = agent_factory.detect_llm_config()
            return agent_factory.build_session_agent(
                memory_provider=self.provider, llm_config=llm_config
            )
        except Exception as exc:
            raise MemorizzServerError(
                "agent_configuration_error",
                "No runnable Memorizz LLM is configured for agent execution",
            ) from exc

    def _load_runnable_agent(
        self, agent_id: Optional[str], identity: RequestIdentity
    ) -> Any:
        normalized = str(agent_id or "").strip()
        if not normalized and self.config.transport != "stdio":
            exposed = sorted(self.config.exposed_agent_ids or [])
            if len(exposed) == 1:
                normalized = exposed[0]
            else:
                raise MemorizzServerError(
                    "agent_id_required",
                    "Select one of the explicitly exposed agents by agent_id",
                )
        if not normalized:
            session = self._default_session()
            agent = session.agent
            normalized = str(getattr(agent, "agent_id", ""))
            if (
                self.config.transport == "stdio"
                and self.config.exposed_agent_ids is not None
            ):
                self.config.exposed_agent_ids.add(normalized)
        else:
            self._assert_agent_exposed(normalized, identity)
            if normalized in self._agents:
                return self._agents[normalized]
            from ..memagent import MemAgent

            try:
                agent = MemAgent.load(normalized, memory_provider=self.provider)
            except Exception as exc:
                raise MemorizzServerError(
                    "agent_not_runnable", "Agent could not be loaded for execution"
                ) from exc
        self._agents[normalized] = agent
        self._agent_locks.setdefault(normalized, threading.RLock())
        return agent

    def execute_agent(
        self,
        message: str,
        identity: RequestIdentity,
        *,
        agent_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_scope(identity, EXECUTE_SCOPE)
        # Every agent turn persists conversation state, even when the selected
        # agent does not call an external mutating tool.
        self._require_scope(identity, WRITE_SCOPE)
        text = str(message or "").strip()
        if not text:
            raise MemorizzServerError("invalid_message", "Message cannot be empty")
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "message_too_large",
                f"Message exceeds {self.config.max_text_chars} characters",
            )
        agent = self._load_runnable_agent(agent_id, identity)
        resolved_agent_id = str(getattr(agent, "agent_id", "default"))
        state_key = (identity.principal or "local", resolved_agent_id)
        previous = self._conversation_state.get(state_key)
        resolved_memory_id = str(
            memory_id or (previous[0] if previous else uuid.uuid4())
        )
        resolved_thread_id = str(
            thread_id or (previous[1] if previous else uuid.uuid4())
        )
        lock = self._agent_locks.setdefault(resolved_agent_id, threading.RLock())
        with lock:
            try:
                response = agent.run(
                    text,
                    memory_id=resolved_memory_id,
                    thread_id=resolved_thread_id,
                    user_id=identity.principal,
                    context=context or {},
                    tool_context={
                        "user_id": identity.principal,
                        "mcp_principal": identity.principal,
                    },
                )
                if getattr(agent, "memory_provider", None) is not None:
                    agent.save()
            except Exception as exc:
                logger.exception("Memorizz MCP agent execution failed")
                raise MemorizzServerError(
                    "agent_execution_error", "The agent could not complete the request"
                ) from exc
        self._conversation_state[state_key] = (
            resolved_memory_id,
            resolved_thread_id,
        )
        return {
            "ok": True,
            "agent_id": resolved_agent_id,
            "memory_id": resolved_memory_id,
            "thread_id": resolved_thread_id,
            "response": str(response),
        }

    def _bounded_limit(self, limit: int) -> int:
        return max(1, min(int(limit or 20), self.config.max_result_items))

    def _tenant_rows(
        self, rows: Any, identity: RequestIdentity, limit: int
    ) -> List[Dict[str, Any]]:
        values = []
        for row in list(rows or []):
            if not isinstance(row, dict):
                continue
            if _document_user_id(row) != identity.principal:
                continue
            values.append(_json_value(row))
            if len(values) >= limit:
                break
        return values

    def list_memories(
        self,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        resolved_limit = self._bounded_limit(limit)
        list_all = self.provider.list_all
        kwargs = {}
        if _accepts_keyword(list_all, "user_id"):
            kwargs["user_id"] = identity.principal
        try:
            rows = list_all(resolved_type, **kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not list memories"
            ) from exc
        if memory_id:
            rows = [
                row
                for row in rows or []
                if isinstance(row, dict) and row.get("memory_id") == memory_id
            ]
        values = self._tenant_rows(rows, identity, resolved_limit)
        return {
            "ok": True,
            "memory_type": resolved_type.value,
            "memories": values,
            "count": len(values),
        }

    def search_memories(
        self,
        query: str,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        text = str(query or "").strip()
        if not text:
            raise MemorizzServerError("invalid_query", "Search query cannot be empty")
        resolved_type = self._normalize_memory_type(memory_type, identity)
        resolved_limit = self._bounded_limit(limit)
        retrieve = self.provider.retrieve_by_query
        kwargs = {
            "memory_store_type": resolved_type,
            "memory_id": memory_id,
            "limit": resolved_limit,
        }
        if _accepts_keyword(retrieve, "user_id"):
            kwargs["user_id"] = identity.principal
        try:
            rows = retrieve(text, **kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not search memories"
            ) from exc
        values = self._tenant_rows(rows, identity, resolved_limit)
        return {
            "ok": True,
            "query": text,
            "memory_type": resolved_type.value,
            "memories": values,
            "count": len(values),
        }

    def get_memory(
        self, record_id: str, memory_type: str, identity: RequestIdentity
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        try:
            row = self.provider.retrieve_by_id(record_id, resolved_type)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the memory"
            ) from exc
        if not isinstance(row, dict) or _document_user_id(row) != identity.principal:
            raise MemorizzServerError("memory_not_found", "Memory was not found")
        return {"ok": True, "memory": _json_value(row)}

    def remember(
        self,
        content: str,
        memory_type: str,
        identity: RequestIdentity,
        *,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        text = str(content or "").strip()
        if not text:
            raise MemorizzServerError(
                "invalid_content", "Memory content cannot be empty"
            )
        if len(text) > self.config.max_text_chars:
            raise MemorizzServerError(
                "content_too_large",
                f"Memory content exceeds {self.config.max_text_chars} characters",
            )
        resolved_type = self._normalize_memory_type(
            memory_type, identity, writable=True
        )
        safe_metadata = dict(metadata or {})
        for reserved in (
            "_id",
            "id",
            "embedding",
            "user_id",
            "memory_id",
            "memory_type",
        ):
            safe_metadata.pop(reserved, None)
        resolved_memory_id = str(memory_id or uuid.uuid4())
        document = {
            **safe_metadata,
            "content": text,
            "text": text,
            "memory_id": resolved_memory_id,
            "user_id": identity.principal,
            "memory_type": resolved_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            record_id = self.provider.store(
                document,
                memory_store_type=resolved_type,
                memory_id=resolved_memory_id,
            )
        except Exception as exc:
            logger.exception("Memorizz MCP memory store failed")
            raise MemorizzServerError(
                "provider_error", "The memory provider could not store the memory"
            ) from exc
        return {
            "ok": True,
            "record_id": str(record_id),
            "memory_id": resolved_memory_id,
            "memory_type": resolved_type.value,
        }

    def forget(
        self,
        record_id: str,
        memory_type: str,
        identity: RequestIdentity,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        existing = self.get_memory(record_id, resolved_type.value, identity)
        if not existing.get("ok"):
            raise MemorizzServerError("memory_not_found", "Memory was not found")
        arguments = {
            "record_id": str(record_id),
            "memory_type": resolved_type.value,
        }
        proposal = self.approval_store.propose(
            owner_id=self._approval_owner(identity),
            tool_name="memorizz_forget_memory",
            arguments=arguments,
            policy_reason="Permanent deletion requires a human host decision",
            checkpoint={
                "version": 1,
                "operation": "forget_memory",
                "arguments": arguments,
                "principal": identity.principal,
                "scopes": sorted(identity.scopes),
                "authenticated": identity.authenticated,
            },
            ttl_seconds=self.config.approval_ttl_seconds,
        )
        return {
            "ok": False,
            "status": "approval_required",
            "proposal": proposal.to_dict(include_arguments=True),
        }

    def _forget_authorized(
        self,
        record_id: str,
        memory_type: str,
        identity: RequestIdentity,
    ) -> Dict[str, Any]:
        self._require_scope(identity, WRITE_SCOPE)
        resolved_type = self._normalize_memory_type(memory_type, identity)
        self.get_memory(record_id, resolved_type.value, identity)
        try:
            deleted = bool(self.provider.delete_by_id(record_id, resolved_type))
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not delete the memory"
            ) from exc
        return {
            "ok": deleted,
            "record_id": record_id,
            "memory_type": resolved_type.value,
            "deleted": deleted,
        }

    def list_approvals(
        self,
        *,
        status: Optional[str] = None,
        principal: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        owner_id = f"mcp-server:{principal}" if principal else None
        return [
            item.to_dict(include_arguments=True)
            for item in self.approval_store.list(
                owner_id=owner_id, status=status, limit=limit
            )
            if item.owner_id.startswith("mcp-server:")
        ]

    def approve_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        return self.approval_store.approve(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        ).to_dict(include_arguments=True)

    def reject_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        return self.approval_store.reject(
            proposal_id,
            approver_id=approver_id,
            decision_reason=reason,
        ).to_dict(include_arguments=True)

    def cancel_proposal(
        self,
        proposal_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel a pending server proposal without executing it."""
        return self.reject_proposal(
            proposal_id,
            approver_id=approver_id,
            reason=reason or "Cancelled by host",
        )

    def resume_proposal(self, proposal_id: str) -> Dict[str, Any]:
        proposal = self.approval_store.get(proposal_id)
        if proposal is None or not proposal.owner_id.startswith("mcp-server:"):
            raise MemorizzServerError("approval_not_found", "Approval was not found")
        if proposal.status != ApprovalStatus.APPROVED:
            raise ApprovalStateError("Only an approved proposal can be resumed")
        checkpoint = dict(proposal.checkpoint or {})
        if checkpoint.get("operation") != "forget_memory":
            raise MemorizzServerError(
                "invalid_checkpoint", "Unsupported MCP server approval checkpoint"
            )
        arguments = dict(checkpoint.get("arguments") or {})
        self.approval_store.consume(
            proposal_id,
            expected_tool_name="memorizz_forget_memory",
            expected_arguments=arguments,
        )
        identity = RequestIdentity(
            principal=checkpoint.get("principal"),
            scopes=frozenset(checkpoint.get("scopes") or []),
            authenticated=bool(checkpoint.get("authenticated")),
        )
        return self._forget_authorized(
            str(arguments.get("record_id") or ""),
            str(arguments.get("memory_type") or ""),
            identity,
        )

    def list_conversations(
        self,
        agent_id: str,
        identity: RequestIdentity,
        *,
        limit: int = 20,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        normalized = str(agent_id or "").strip()
        self._assert_agent_exposed(normalized, identity)
        try:
            agent = self.provider.retrieve_memagent(normalized)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the agent"
            ) from exc
        if not agent:
            raise MemorizzServerError("agent_not_found", "Agent was not found")
        memory_ids = (
            agent.get("memory_ids")
            if isinstance(agent, dict)
            else getattr(agent, "memory_ids", None)
        )
        from ..cli.conversations import discover_conversations

        conversations = discover_conversations(
            self.provider,
            memory_ids or [],
            user_id=identity.principal,
            agent_id=normalized,
        )[: self._bounded_limit(limit)]
        values = [
            {
                "memory_id": item.memory_id,
                "thread_id": item.thread_id,
                "title": item.title,
                "preview": item.preview,
                "created_at": item.created_at.isoformat() if item.created_at else None,
                "updated_at": item.updated_at.isoformat() if item.updated_at else None,
                "message_count": item.message_count,
            }
            for item in conversations
        ]
        return {"ok": True, "conversations": values, "count": len(values)}

    def get_conversation(
        self,
        memory_id: str,
        thread_id: str,
        identity: RequestIdentity,
        *,
        limit: int = 100,
    ) -> Dict[str, Any]:
        self._require_scope(identity, READ_SCOPE)
        retrieve = self.provider.retrieve_conversation_history_ordered_by_timestamp
        kwargs = {
            "memory_id": memory_id,
            "memory_type": MemoryType.CONVERSATION_MEMORY,
            "limit": self._bounded_limit(limit),
        }
        if _accepts_keyword(retrieve, "user_id"):
            kwargs["user_id"] = identity.principal
        if _accepts_keyword(retrieve, "thread_id"):
            kwargs["thread_id"] = thread_id
        try:
            rows = retrieve(**kwargs)
        except Exception as exc:
            raise MemorizzServerError(
                "provider_error", "The memory provider could not read the conversation"
            ) from exc
        rows = [
            row
            for row in rows or []
            if isinstance(row, dict)
            and _document_field(row, "memory_id") == memory_id
            and _document_field(row, "thread_id") == thread_id
        ]
        values = self._tenant_rows(rows, identity, self._bounded_limit(limit))
        return {
            "ok": True,
            "memory_id": memory_id,
            "thread_id": thread_id,
            "messages": values,
            "count": len(values),
        }
