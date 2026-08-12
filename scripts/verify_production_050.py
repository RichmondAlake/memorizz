#!/usr/bin/env python3
"""Opt-in production verification for MemoRizz 0.5.0.

The verifier exercises the real Oracle provider and, unless explicitly
skipped, the configured OpenAI, Anthropic, Tavily, E2B, Browser Use, and hosted
MCP endpoints. Credentials are read from the process environment or a local
``.env`` file, are never accepted as command-line arguments, and are redacted
from all diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from memorizz.sandbox.base import SandboxProvider  # noqa: E402

SECRET_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "BROWSER_USE_API_KEY",
    "E2B_API_KEY",
    "GITHUB_TOKEN",
    "GOOGLE_API_KEY",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "MEMORIZZ_MCP_ENCRYPTION_KEY",
    "OPENAI_API_KEY",
    "ORACLE_ADMIN_PASSWORD",
    "ORACLE_PASSWORD",
    "SKILLSMP_API_KEY",
    "TAVILY_API_KEY",
    "TWILIO_AUTH_TOKEN",
    "VOYAGE_API_KEY",
)
SECRET_PATTERN = re.compile(
    r"(?i)(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{12,}|"
    r"tvly-[A-Za-z0-9_-]{12,}|e2b_[A-Za-z0-9_-]{12,}|"
    r"bearer\s+[A-Za-z0-9._~-]{12,})"
)


class ProbeSkipped(RuntimeError):
    """A deliberate, operator-selected skip rather than a verification failure."""


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPOSITORY_ROOT / ".env", override=False)


def _redact_text(value: Any) -> str:
    text = str(value)
    secrets = sorted(
        {
            secret
            for name in SECRET_ENV_NAMES
            if (secret := os.environ.get(name, "").strip())
        },
        key=len,
        reverse=True,
    )
    for secret in secrets:
        text = text.replace(secret, "***REDACTED***")
    return SECRET_PATTERN.sub("***REDACTED***", text)


def _redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                marker in lowered
                for marker in ("password", "secret", "token", "api_key")
            ):
                result[str(key)] = "***REDACTED***"
            else:
                result[str(key)] = _redact_value(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [_redact_value(item) for item in value]
    return _redact_text(value) if isinstance(value, str) else value


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for this live probe")
    return value


def _tool_call(name: str, arguments: Dict[str, Any], call_id: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _structured_mcp_result(value: Dict[str, Any]) -> Dict[str, Any]:
    result = value.get("result") or {}
    structured = result.get("structuredContent") or {}
    payload = structured.get("result") or {}
    return payload if isinstance(payload, dict) else {}


class InMemoryCredentialStore:
    """Process-local credential store used so verification writes no secrets."""

    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}

    def get(self, credential_ref: str) -> Dict[str, Any]:
        return dict(self.records.get(credential_ref, {}))

    def set(self, credential_ref: str, value: Dict[str, Any]) -> None:
        self.records[credential_ref] = dict(value)

    def update(self, credential_ref: str, values: Dict[str, Any]) -> None:
        current = dict(self.records.get(credential_ref, {}))
        for key, value in values.items():
            if value is None:
                current.pop(key, None)
            else:
                current[key] = value
        self.records[credential_ref] = current

    def delete(self, credential_ref: str) -> bool:
        return self.records.pop(credential_ref, None) is not None


class VerificationSandbox(SandboxProvider):
    """Minimal provider used to verify the complete builder surface."""

    provider_name = "verification"

    def __init__(self) -> None:
        super().__init__({})

    def validate_configuration(self) -> Optional[str]:
        return None

    def get_config(self) -> Dict[str, Any]:
        return {"provider": self.provider_name}

    def execute_code(self, code, language="python", timeout=30, envs=None):
        from memorizz.sandbox.models import ExecutionResult

        return ExecutionResult(stdout=[code], metadata={"provider": self.provider_name})

    def write_file(self, path, content):
        return True

    def read_file(self, path):
        return "verification"


def production_calendar_lookup(
    calendar_id: str,
    visibility: Literal["private", "public"] = "private",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read one calendar using strict, deterministic arguments."""
    return {
        "calendar_id": calendar_id,
        "visibility": visibility,
        "options": dict(options or {}),
    }


def production_weather_read(city: str) -> str:
    """Read current weather for one city."""
    return f"weather:{city}"


def production_calendar_create_event(title: str, starts_at: str) -> str:
    """Create a calendar event."""
    return f"created:{title}:{starts_at}"


def production_inventory_lookup(sku: str) -> str:
    """Read inventory for one SKU."""
    return f"stock:{sku}"


def production_small_result() -> str:
    """Return a small deterministic tool result."""
    return "inline-result"


def production_large_result() -> str:
    """Return a large deterministic tool result."""
    return "L" * 900


@dataclass
class ProbeResult:
    name: str
    category: str
    status: str
    required: bool
    duration_ms: int
    detail: Any = None
    error: Optional[str] = None


class ProductionVerifier:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.results: list[ProbeResult] = []
        self.state: Dict[str, Any] = {
            "memory_ids": set(),
            "agent_ids": set(),
            "user_ids": set(),
        }
        self.tempdir = tempfile.TemporaryDirectory(prefix="memorizz-050-verify-")
        self.temp_path = Path(self.tempdir.name)
        os.environ["MEMORIZZ_HOME"] = str(self.temp_path / "home")

    def run(
        self,
        name: str,
        category: str,
        function,
        *,
        required: bool = True,
    ) -> None:
        started = time.monotonic()
        try:
            detail = function()
            result = ProbeResult(
                name=name,
                category=category,
                status="passed",
                required=required,
                duration_ms=int((time.monotonic() - started) * 1000),
                detail=_redact_value(detail),
            )
        except ProbeSkipped as exc:
            result = ProbeResult(
                name=name,
                category=category,
                status="skipped",
                required=required,
                duration_ms=int((time.monotonic() - started) * 1000),
                detail=_redact_text(exc),
            )
        except Exception as exc:
            result = ProbeResult(
                name=name,
                category=category,
                status="failed",
                required=required,
                duration_ms=int((time.monotonic() - started) * 1000),
                error=_redact_text(f"{type(exc).__name__}: {exc}"),
            )
        self.results.append(result)
        if not self.args.json:
            symbol = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[
                result.status
            ]
            suffix = f" — {result.error}" if result.error else ""
            print(
                f"[{symbol}] {result.category}: {result.name} "
                f"({result.duration_ms} ms){suffix}",
                file=sys.stderr,
                flush=True,
            )

    @property
    def provider(self):
        provider = self.state.get("provider")
        if provider is None:
            raise RuntimeError("Oracle provider probe did not complete")
        return provider

    def remember_scope(
        self,
        *,
        memory_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        if memory_id:
            self.state["memory_ids"].add(memory_id)
        if agent_id:
            self.state["agent_ids"].add(agent_id)
        if user_id:
            self.state["user_ids"].add(user_id)

    def oracle_runtime(self) -> Dict[str, Any]:
        from memorizz.memory_provider.oracle import LocalOracleRuntime

        runtime = LocalOracleRuntime.from_env(
            provision_if_missing=False,
            container_name=self.args.oracle_container,
        )
        report = runtime.ensure_ready()
        _require(report.get("ok") is True, "Oracle Docker runtime is not ready")
        return {
            "container": report["container"],
            "state": report["state"],
            "action": report["action"],
        }

    def oracle_preflight(self) -> Dict[str, Any]:
        from memorizz.memory_provider.oracle import OracleProvider

        provider = OracleProvider.from_env(index_policy="lazy")
        report = provider.preflight()
        _require(report.get("ok") is True, f"Oracle preflight failed: {report}")
        embedding = report.get("embedding") or {}
        dimensions = report.get("vector_dimensions") or {}
        _require(
            embedding.get("provider") == "OracleInDatabaseEmbeddingProvider",
            "Oracle in-database embeddings are not active",
        )
        _require(embedding.get("dimensions") == 384, "Embedding dimension is not 384")
        _require(
            dimensions and set(dimensions.values()) == {384},
            "Oracle VECTOR columns are not consistently 384-dimensional",
        )
        _require(
            report.get("pdb_open_state") == "READ WRITE",
            "Oracle PDB is not open read/write",
        )
        _require(
            not report.get("missing_schema_privileges"),
            "Oracle schema privileges are incomplete",
        )
        self.state["provider"] = provider
        return {
            "database_product": report.get("database_product"),
            "database_version": report.get("database_version"),
            "pdb": report.get("pdb"),
            "pdb_open_state": report.get("pdb_open_state"),
            "embedding_model": embedding.get("model"),
            "embedding_dimensions": embedding.get("dimensions"),
            "vector_indexes": len(report.get("vector_indexes") or []),
            "vector_memory_size": report.get("vector_memory_size"),
            "index_policy": report.get("index_policy"),
            "exact_search_fallback": report.get("exact_search_fallback"),
        }

    def oracle_schema_and_embedding(self) -> Dict[str, Any]:
        provider = self.provider
        required_columns = {
            "SUMMARIES": {
                "SOURCE_MESSAGE_IDS",
                "PERIOD_START",
                "PERIOD_END",
                "MEMORY_UNITS_COUNT",
            },
            "CONVERSATION_MEMORY": {"SUMMARY_ID", "USER_ID"},
            "TOOLBOX": {
                "INPUT_SCHEMA",
                "TOOL_POLICY",
                "ALIASES",
                "DEPRECATED_ARGUMENTS",
                "IMPORT_REFERENCE",
                "USER_ID",
            },
            "SEMANTIC_CACHE": {"MEMORY_ID", "SESSION_ID", "USER_ID", "METADATA"},
        }
        with provider._get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT table_name, column_name
                FROM user_tab_columns
                WHERE table_name IN (
                    'SUMMARIES', 'CONVERSATION_MEMORY', 'TOOLBOX', 'SEMANTIC_CACHE'
                )
                """
            )
            actual: Dict[str, set[str]] = {}
            for table, column in cursor.fetchall():
                actual.setdefault(str(table), set()).add(str(column))
            cursor.execute(
                """
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = 'SUMMARY_MESSAGE_LINKS'
                """
            )
            link_table_count = int(cursor.fetchone()[0])
            cursor.close()
        for table, columns in required_columns.items():
            missing = columns.difference(actual.get(table, set()))
            _require(not missing, f"{table} is missing columns: {sorted(missing)}")
        _require(link_table_count == 1, "SUMMARY_MESSAGE_LINKS is missing")
        vector = provider._embedding_provider.get_embedding(
            "MemoRizz Oracle 26ai production verification"
        )
        _require(len(vector) == 384, "Oracle VECTOR_EMBEDDING returned wrong size")
        _require(any(float(item) != 0.0 for item in vector), "Embedding is all zeroes")
        return {
            "schema_tables_checked": sorted(required_columns),
            "summary_link_table": True,
            "embedding_dimensions": len(vector),
            "embedding_nonzero": True,
        }

    def toolbox_schema_round_trip(self) -> Dict[str, Any]:
        from memorizz.enums import MemoryType
        from memorizz.long_term.procedural.toolbox import Toolbox
        from memorizz.memagent import MemAgent

        token = uuid.uuid4().hex[:12]
        agent_id = f"verify-toolbox-{token}"
        user_id = f"verify-user-{token}"
        self.remember_scope(agent_id=agent_id, user_id=user_id)
        # Persist the owning agent first so Oracle can enforce the Toolbox FK.
        MemAgent(
            memory_provider=self.provider,
            agent_id=agent_id,
            automations_enabled=False,
        ).save()
        toolbox = Toolbox(self.provider, agent_id=agent_id)
        tool_id = toolbox.register_tool(
            production_calendar_lookup,
            augment=False,
            persist=True,
            user_id=user_id,
            aliases=["calendar_lookup"],
            deprecated_arguments={"calendar": "calendar_id"},
        )
        row = self.provider.retrieve_by_id(tool_id, MemoryType.TOOLBOX)
        _require(row is not None, "Persisted Toolbox row was not retrievable")
        schema = row.get("input_schema") or {}
        properties = schema.get("properties") or {}
        _require(schema.get("additionalProperties") is False, "Schema is not strict")
        _require("calendar_id" in schema.get("required", []), "Required field was lost")
        visibility = properties.get("visibility") or {}
        _require(
            visibility.get("enum") == ["private", "public"],
            "Literal enum was not preserved",
        )
        _require(visibility.get("default") == "private", "Default was not preserved")
        _require(row.get("aliases") == ["calendar_lookup"], "Aliases were lost")
        _require(
            row.get("deprecated_arguments") == {"calendar": "calendar_id"},
            "Deprecated argument mapping was lost",
        )
        _require(row.get("user_id") == user_id, "Toolbox user scope was lost")
        self.state["toolbox"] = toolbox
        return {
            "required": schema.get("required"),
            "strict": True,
            "enum_and_default_preserved": True,
            "aliases_preserved": True,
            "deprecated_arguments_preserved": True,
        }

    def summary_compaction_round_trip(self) -> Dict[str, Any]:
        from memorizz.enums import MemoryType
        from memorizz.memagent import MemAgent

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-summary-{token}"
        thread_id = f"verify-thread-{token}"
        user_id = f"verify-user-{token}"
        agent_id = f"verify-summary-agent-{token}"
        self.remember_scope(memory_id=memory_id, user_id=user_id, agent_id=agent_id)
        message_ids = []
        for role, content in (
            ("user", "Original production verification question"),
            ("assistant", "Original production verification answer"),
        ):
            message_ids.append(
                self.provider.store(
                    data={
                        "memory_id": memory_id,
                        "thread_id": thread_id,
                        "role": role,
                        "content": content,
                        "agent_id": agent_id,
                        "user_id": user_id,
                    },
                    memory_store_type=MemoryType.CONVERSATION_MEMORY,
                )
            )
        summary_id = f"summary-{token}"
        self.provider.store_summary_with_links(
            {
                "summary_id": summary_id,
                "content": "Question and answer retained losslessly.",
                "source_message_ids": message_ids,
                "period_start": 10.0,
                "period_end": 20.0,
                "memory_units_count": 2,
                "memory_id": memory_id,
                "thread_id": thread_id,
                "agent_id": agent_id,
                "user_id": user_id,
            }
        )
        summary = self.provider.retrieve_by_id(summary_id, MemoryType.SUMMARIES)
        _require(summary is not None, "Summary was not retrievable by summary_id")
        _require(
            summary.get("source_message_ids") == message_ids,
            "Summary source order was not preserved",
        )
        _require(summary.get("period_start") == 10.0, "period_start was lost")
        _require(summary.get("period_end") == 20.0, "period_end was lost")
        _require(summary.get("memory_units_count") == 2, "Message count was lost")
        for message_id in message_ids:
            message = self.provider.retrieve_by_id(
                message_id, MemoryType.CONVERSATION_MEMORY
            )
            _require(
                message and message.get("summary_id") == summary_id,
                "Conversation summary marker was not written",
            )
        agent = MemAgent(
            memory_provider=self.provider,
            agent_id=agent_id,
            automations_enabled=False,
        )
        agent._current_user_id = user_id
        expanded, error = agent.tool_manager.execute_tool(
            "expand_summary", {"summary_id": summary_id}
        )
        _require(error is None, f"Summary expansion failed: {error}")
        originals = expanded.get("original_messages") or []
        _require(
            [item.get("content") for item in originals]
            == [
                "Original production verification question",
                "Original production verification answer",
            ],
            "Summary expansion was not lossless",
        )
        rollback_id = f"rollback-{token}"
        try:
            self.provider.store_summary_with_links(
                {
                    "summary_id": rollback_id,
                    "content": "This insert must roll back",
                    "source_message_ids": [str(uuid.uuid4())],
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "user_id": user_id,
                }
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-scope summary source did not fail")
        _require(
            self.provider.retrieve_by_id(rollback_id, MemoryType.SUMMARIES) is None,
            "Failed summary insert was not rolled back",
        )
        return {
            "source_messages": len(message_ids),
            "summary_marker": True,
            "lossless_expansion": True,
            "atomic_rollback": True,
        }

    def semantic_cache_governance(self) -> Dict[str, Any]:
        from memorizz.enums import MemoryType
        from memorizz.short_term_memory.semantic_cache import (
            SemanticCache,
            SemanticCacheConfig,
        )

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-cache-{token}"
        agent_id = f"verify-cache-agent-{token}"
        user_id = f"verify-user-{token}"
        self.remember_scope(memory_id=memory_id, agent_id=agent_id, user_id=user_id)
        metadata = {
            "domain": "inventory",
            "tags": ["verification"],
            "fingerprints": {
                "model": "model-v1",
                "prompt": "prompt-v1",
                "tool_schema": "tools-v1",
                "data_version": "inventory-v1",
            },
            "admission": {"read_only": True, "deterministic": True},
        }
        cache = SemanticCache(
            config=SemanticCacheConfig(
                similarity_threshold=0.90,
                enable_memory_provider_sync=True,
                freshness_by_domain={"inventory": 300.0},
            ),
            memory_provider=self.provider,
            embedding_manager=self.provider._embedding_provider,
            agent_id=agent_id,
            memory_id=memory_id,
        )
        _require(
            cache.set(
                "inventory for SKU-050",
                "12 units",
                session_id="session-1",
                user_id=user_id,
                metadata=metadata,
            ),
            "Cache admission failed",
        )
        _require(
            cache.get(
                "inventory for SKU-050",
                session_id="session-1",
                user_id=f"other-{user_id}",
                lookup_metadata=metadata,
            )
            is None,
            "Semantic cache crossed a user boundary",
        )
        _require(
            cache.get(
                "inventory for SKU-050",
                session_id="session-1",
                user_id=user_id,
                lookup_metadata=metadata,
            )
            == "12 units",
            "Expected governed cache hit was missed",
        )
        stats = cache.statistics()
        _require(stats["writes"] == 1, "Cache write counter is incorrect")
        _require(stats["hits"] == 1, "Cache hit counter is incorrect")
        _require(stats["misses"] >= 1, "Cache miss counter is incorrect")
        provenance = stats.get("last_hit") or {}
        _require(provenance.get("user_id") == user_id, "Hit provenance lost user")
        _require(provenance.get("similarity") is not None, "Hit similarity was lost")
        removed = cache.invalidate(domains=["inventory"])
        _require(removed >= 1, "Domain invalidation removed no cache rows")
        remaining = self.provider.list_all(MemoryType.SEMANTIC_CACHE, user_id=user_id)
        _require(
            not any(row.get("memory_id") == memory_id for row in remaining),
            "Persistent domain invalidation left the Oracle row behind",
        )
        return {
            "hits": stats["hits"],
            "misses": stats["misses"],
            "writes": stats["writes"],
            "tenant_isolation": True,
            "hit_provenance": True,
            "domain_invalidation": removed,
        }

    def progressive_tool_router(self) -> Dict[str, Any]:
        from memorizz.memagent.managers.tool_manager import ToolManager
        from memorizz.tooling import SemanticToolRouter

        manager = ToolManager()
        for function in (
            production_weather_read,
            production_calendar_create_event,
            production_inventory_lookup,
        ):
            _require(manager.add_tool(function), f"Could not add {function.__name__}")
        router = SemanticToolRouter(
            manager,
            top_k=1,
            aliases={"old_weather": "production_weather_read"},
            deprecated_arguments={"production_weather_read": {"location": "city"}},
            max_attempts_per_call=2,
        )
        router.begin_turn(user_id="verification-user")
        schemas = router.schemas_for_turn(
            "weather in London", user_id="verification-user"
        )
        names = {schema["function"]["name"] for schema in schemas}
        _require(
            names == {"discover_tools", "invoke_tool", "production_weather_read"},
            f"Unexpected progressively disclosed tools: {sorted(names)}",
        )
        _require(
            all(
                schema["function"]["parameters"].get("additionalProperties") is False
                for schema in schemas
            ),
            "One or more disclosed schemas are not strict",
        )
        hidden = router.invoke_tool(
            "production_calendar_create_event",
            {"title": "Review", "starts_at": "2026-08-13T10:00:00Z"},
        )
        _require(
            hidden.get("error_code") == "invalid_tool_invocation",
            "Hidden tool dispatch was not blocked",
        )
        called = router.invoke_tool("old_weather", {"location": "London"})
        _require(called.get("result") == "weather:London", "Alias dispatch failed")
        duplicate = router.invoke_tool("production_weather_read", {"city": "London"})
        _require(
            duplicate.get("error_code") == "duplicate_tool_call",
            "Duplicate tool call was not rejected",
        )
        return {
            "registered_tools": 3,
            "disclosed_business_tools": 1,
            "strict_schemas": True,
            "hidden_dispatch_blocked": True,
            "alias_and_deprecated_argument": True,
            "duplicate_detection": True,
        }

    def durable_automation_approval(self) -> Dict[str, Any]:
        from memorizz.approval import ApprovalRequired, SQLiteApprovalStore
        from memorizz.memagent import MemAgent
        from memorizz.tooling import ContextPolicy

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-approval-{token}"
        thread_id = f"verify-thread-{token}"
        user_id = f"verify-user-{token}"
        agent_id = f"verify-approval-agent-{token}"
        self.remember_scope(memory_id=memory_id, agent_id=agent_id, user_id=user_id)
        store = SQLiteApprovalStore(self.temp_path / f"approval-{token}.sqlite3")
        agent = MemAgent(
            memory_provider=self.provider,
            agent_id=agent_id,
            approval_store=store,
            context_policy=ContextPolicy(tool_top_k=50),
            default_timezone="UTC",
        )
        _require(
            agent.automation_manager.is_enabled(),
            "Oracle automation store is not available",
        )
        agent._current_memory_id = memory_id
        agent._current_thread_id = thread_id
        query = "create a scheduled interval automation"
        agent.semantic_tool_router.begin_turn(user_id=user_id)
        schemas = agent._build_llm_tools(query, user_id=user_id)
        serialized_schemas = json.dumps(schemas)
        _require('"approved"' not in serialized_schemas, "approved leaked to schema")
        _require('"confirm"' not in serialized_schemas, "confirm leaked to schema")
        names = {schema["function"]["name"] for schema in schemas}
        _require(
            "automation_create_job" in names,
            "Automation create tool was not disclosed",
        )
        arguments = {
            "name": "MemoRizz 0.5 verification",
            "schedule_type": "interval",
            "interval_seconds": 3600,
            "timezone": "UTC",
            "query_template": "Run the MemoRizz 0.5 verification task",
            "memory_id": memory_id,
            "client_request_id": token,
        }
        try:
            agent._execute_and_record_tool_call(
                _tool_call("automation_create_job", arguments, f"approval-{token}"),
                [{"role": "user", "content": query}],
                workflow=None,
                user_id=user_id,
                query=query,
            )
        except ApprovalRequired as exc:
            proposal = exc.proposal
        else:
            raise AssertionError("Automation mutation executed without approval")
        _require(proposal.arguments == arguments, "Proposal arguments changed")
        _require(
            proposal.to_dict().get("thread_id") == thread_id,
            "Proposal thread was not retained",
        )
        _require(bool(proposal.argument_hash), "Proposal argument hash is missing")
        approved = agent.approve(
            proposal.proposal_id, approver_id="production-verifier"
        )
        _require(
            approved.get("approver_id") == "production-verifier",
            "Approver identity was not retained",
        )
        resumed = json.loads(agent.resume_approval(proposal.proposal_id))
        _require(resumed.get("ok") is True, f"Approved automation failed: {resumed}")
        job_id = resumed.get("job_id")
        _require(bool(job_id), "Automation resume returned no job ID")
        _require(
            agent.automation_manager.store.get_job(job_id) is not None,
            "Approved automation was not persisted in Oracle",
        )
        replay = json.loads(agent.resume_approval(proposal.proposal_id))
        _require(
            replay.get("error_code") == "invalid_approval_state",
            "Approval proposal was reusable",
        )
        return {
            "model_visible_boolean_gates": False,
            "argument_hash": True,
            "thread_checkpoint": True,
            "approver_audit": True,
            "oracle_automation_persisted": True,
            "single_use": True,
        }

    def size_aware_tool_results(self) -> Dict[str, Any]:
        from memorizz.memagent import MemAgent
        from memorizz.tooling import ContextPolicy, ToolResultPolicy

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-tool-log-{token}"
        thread_id = f"verify-thread-{token}"
        user_id = f"verify-user-{token}"
        agent_id = f"verify-tool-agent-{token}"
        self.remember_scope(memory_id=memory_id, agent_id=agent_id, user_id=user_id)
        agent = MemAgent(
            tools=[production_small_result, production_large_result],
            memory_provider=self.provider,
            memory_ids=[memory_id],
            agent_id=agent_id,
            automations_enabled=False,
            context_policy=ContextPolicy(tool_top_k=50),
            tool_result_policy=ToolResultPolicy(offload_above_chars=256),
        )
        agent._current_memory_id = memory_id
        agent._current_thread_id = thread_id
        messages: list[Dict[str, Any]] = []

        agent.semantic_tool_router.begin_turn(user_id=user_id)
        agent._build_llm_tools("small result", user_id=user_id)
        agent._execute_and_record_tool_call(
            _tool_call("production_small_result", {}, f"small-{token}"),
            messages,
            workflow=None,
            user_id=user_id,
            query="small result",
        )
        _require(messages[-1]["content"] == "inline-result", "Small result offloaded")
        _require(
            not self.provider.list_tool_logs(
                memory_id=memory_id, user_id=user_id, thread_id=thread_id
            ),
            "Small result created a tool-log row",
        )

        agent.semantic_tool_router.begin_turn(user_id=user_id)
        agent._build_llm_tools("large result", user_id=user_id)
        agent._execute_and_record_tool_call(
            _tool_call("production_large_result", {}, f"large-{token}"),
            messages,
            workflow=None,
            user_id=user_id,
            query="large result",
        )
        pointer = json.loads(messages[-1]["content"])
        _require(pointer.get("offloaded") is True, "Large result was not offloaded")
        _require(pointer.get("result_chars") == 900, "Pointer size audit is wrong")
        _require(bool(pointer.get("result_sha256")), "Pointer digest is missing")
        logs = self.provider.list_tool_logs(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        )
        _require(len(logs) == 1, "Large result was not stored exactly once")

        agent.semantic_tool_router.begin_turn(user_id=user_id)
        agent._build_llm_tools("retrieve tool log entry", user_id=user_id)
        agent._execute_and_record_tool_call(
            _tool_call(
                "retrieve_tool_log_entry",
                {"tool_log_id": pointer["tool_log_id"]},
                f"expand-{token}",
            ),
            messages,
            workflow=None,
            user_id=user_id,
            query="retrieve tool log entry",
        )
        expanded = json.loads(messages[-1]["content"])
        stored_result = (expanded.get("tool_log") or {}).get("result")
        _require(stored_result == "L" * 900, "Expanded result was not lossless")
        _require(
            hashlib.sha256(stored_result.encode()).hexdigest()
            == pointer["result_sha256"],
            "Expanded result does not match pointer digest",
        )
        logs_after = self.provider.list_tool_logs(
            memory_id=memory_id, user_id=user_id, thread_id=thread_id
        )
        _require(len(logs_after) == 1, "Expansion created a pointer-to-pointer log")

        agent.semantic_tool_router.begin_turn(user_id=user_id)
        agent._build_llm_tools("retrieve tool log entry", user_id=user_id)
        agent._execute_and_record_tool_call(
            _tool_call(
                "retrieve_tool_log_entry",
                {"tool_log_id": f"missing-{token}"},
                f"missing-{token}",
            ),
            messages,
            workflow=None,
            user_id=user_id,
            query="retrieve tool log entry",
        )
        failure = json.loads(messages[-1]["content"])
        _require(
            failure.get("error_code") == "tool_log_not_found",
            "Missing expansion did not return a structured failure",
        )
        return {
            "small_inline": True,
            "large_offloaded_once": True,
            "digest_verified": True,
            "expansion_reoffloaded": False,
            "structured_expansion_failure": True,
        }

    def builder_and_semantic_layer(self) -> Dict[str, Any]:
        from memorizz.approval import SQLiteApprovalStore
        from memorizz.long_term.procedural.toolbox import Toolbox
        from memorizz.memagent import MemAgent
        from memorizz.memagent.builders import MemAgentBuilder
        from memorizz.semantic_layer import SemanticCatalog
        from memorizz.tooling import ContextPolicy, ToolResultPolicy

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-builder-{token}"
        agent_id = f"verify-builder-agent-{token}"
        delegate_id = f"verify-builder-delegate-{token}"
        self.remember_scope(memory_id=memory_id, agent_id=agent_id)
        self.remember_scope(agent_id=delegate_id)
        catalog = SemanticCatalog(
            [
                {
                    "name": "commerce",
                    "version": "2026-08",
                    "entities": [
                        {
                            "name": "orders",
                            "source": "analytics.orders",
                            "primary_key": "id",
                        }
                    ],
                    "measures": [
                        {
                            "name": "revenue",
                            "entity": "orders",
                            "expression": "orders.amount",
                            "aggregation": "sum",
                        }
                    ],
                    "dimensions": [
                        {
                            "name": "region",
                            "entity": "orders",
                            "expression": "orders.region",
                        }
                    ],
                    "policies": [
                        {
                            "name": "analyst_access",
                            "entity": "orders",
                            "allowed_roles": ["analyst"],
                            "row_filter": "orders.deleted_at IS NULL",
                        }
                    ],
                    "synonyms": [
                        {"alias": "sales", "target": "revenue", "kind": "measure"}
                    ],
                }
            ]
        )
        plan = catalog.plan(
            "commerce",
            {
                "measures": ["sales"],
                "dimensions": ["region"],
                "filters": [{"field": "region", "operator": "eq", "value": "EMEA"}],
                "roles": ["analyst"],
                "limit": 50,
            },
        )
        _require("EMEA" not in plan.sql, "Semantic filter was interpolated into SQL")
        _require(plan.parameters.get("filter_0") == "EMEA", "Filter bind was lost")
        _require(plan.policies == ["analyst_access"], "Semantic policy was not applied")
        _require(bool(plan.lineage), "Semantic lineage is empty")

        delegate = MemAgent(
            memory_provider=self.provider,
            agent_id=delegate_id,
            automations_enabled=False,
        )
        toolbox = Toolbox(self.provider, agent_id=agent_id)
        approval_store = SQLiteApprovalStore(
            self.temp_path / f"builder-{token}.sqlite3"
        )
        saved_skills_key = os.environ.pop("SKILLSMP_API_KEY", None)
        try:
            agent = (
                MemAgentBuilder()
                .with_name("MemoRizz 0.5 production verifier")
                .with_instruction("Verify the complete 0.5 builder surface.")
                .with_memory_provider(self.provider)
                .with_memory_ids(memory_id)
                .with_sandbox(VerificationSandbox())
                .with_toolbox(toolbox)
                .with_skills(
                    {
                        "name": f"verification_skill_{token}",
                        "description": "Verify authored skill persistence",
                        "content": "Inspect scope, execute safely, report evidence.",
                    }
                )
                .with_skill_retrieval(True, top_k=2)
                .with_skills_marketplace(
                    "skillsmp", {"base_url": "https://skillsmp.com"}
                )
                .with_tool_result_policy(ToolResultPolicy(offload_above_chars=512))
                .with_context_policy(ContextPolicy(tool_top_k=3))
                .with_approval_store(approval_store)
                .with_delegation([delegate], mode="deterministic", plan=[])
                .with_semantic_layer(catalog)
                .with_continual_learning(False)
                .with_automations_enabled(False)
                .build_and_save()
            )
        finally:
            if saved_skills_key is not None:
                os.environ["SKILLSMP_API_KEY"] = saved_skills_key
        _require(agent.sandbox_manager.provider is not None, "Sandbox was not attached")
        _require(agent.toolbox is toolbox, "Toolbox was not attached")
        _require(agent.skillbox is not None, "Skillbox was not created")
        _require(agent.skill_retrieval is True, "Skill retrieval was not enabled")
        _require(agent.continual_learning is False, "Continual learning was enabled")
        _require(agent.delegates == [delegate], "Delegates were not attached")
        _require(agent.semantic_layer is catalog, "Semantic catalog was not attached")
        _require(agent.has_skills_marketplace(), "Skills marketplace was not attached")
        _require(
            self.provider.retrieve_memagent(agent.agent_id) is not None,
            "build_and_save did not persist the agent",
        )
        self.state["agent_ids"].add(agent.agent_id)
        return {
            "sandbox": True,
            "toolbox": True,
            "authored_skill_retrieval": True,
            "continual_learning": False,
            "tool_and_context_policies": True,
            "approval_store": True,
            "delegation": True,
            "skills_marketplace": True,
            "build_and_save": True,
            "semantic_plan_parameterized": True,
            "semantic_policy_and_lineage": True,
        }

    def graalpy_execution_boundary(self) -> Dict[str, Any]:
        from memorizz.sandbox.providers.graalpy_provider import GraalPySandboxProvider

        provider = GraalPySandboxProvider(
            graalpy_path=sys.executable,
            working_dir=str(self.temp_path / "graal-execution"),
            max_memory_mb=256,
            max_cpu_seconds=5,
            max_processes=2,
            max_file_bytes=4096,
        )
        try:
            config = provider.get_config()
            _require(
                config.get("security_boundary")
                == "bounded_execution_provider_not_strong_sandbox",
                "Subprocess mode is mislabeled as a strong sandbox",
            )
            _require(config.get("allow_network") is False, "Network is not denied")
            _require(provider.write_file("safe/value.txt", "safe"), "Safe write failed")
            _require(provider.read_file("safe/value.txt") == "safe", "Safe read failed")
            _require(
                provider.write_file("../escape.txt", "blocked") is False,
                "Path traversal was accepted",
            )
            _require(
                provider.read_file("/etc/passwd") is None,
                "Absolute host path was accepted",
            )
            safe_environment = provider._safe_environment(None)
            _require(
                not any(name in safe_environment for name in SECRET_ENV_NAMES),
                "Host secrets leaked into the subprocess environment",
            )
        finally:
            provider.close()
        wrapper_source = (
            SOURCE_ROOT / "memorizz" / "sandbox" / "MemorizzGraalSandbox.java"
        ).read_text(encoding="utf-8")
        for required in (
            "SandboxPolicy.UNTRUSTED",
            "allowCreateProcess(false)",
            "EnvironmentAccess.NONE",
            "IOAccess.NONE",
            "sandbox.MaxCPUTime",
            "sandbox.MaxHeapMemory",
            "sandbox.MaxThreads",
        ):
            _require(required in wrapper_source, f"Java wrapper is missing {required}")
        return {
            "subprocess_label": "execution_provider_only",
            "path_confinement": True,
            "environment_allowlist": True,
            "network_default": "denied",
            "host_resource_limits": True,
            "java_untrusted_wrapper_source": True,
            "prebuilt_wrapper_jar": False,
        }

    def cli_capabilities_and_preflight(self) -> Dict[str, Any]:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(SOURCE_ROOT)
        capabilities = subprocess.run(
            [sys.executable, "-m", "memorizz.cli", "capabilities", "--json"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        _require(
            capabilities.returncode == 0,
            f"CLI capabilities failed: {capabilities.stderr}",
        )
        capability_report = json.loads(capabilities.stdout)
        _require(capability_report.get("version") == "0.5.0", "CLI version mismatch")
        required_features = {
            "mcp_client",
            "mcp_server",
            "durable_approvals",
            "progressive_tool_disclosure",
            "tool_result_offloading",
            "semantic_cache_governance",
            "sandbox_e2b_v2",
            "multi_agent_orchestration",
            "oracle_summary_compaction",
            "oracle_bootstrap_preflight",
            "governed_semantic_layer",
            "browser_control",
        }
        _require(
            required_features.issubset(capability_report.get("features") or {}),
            "CLI capability report is incomplete",
        )
        preflight = subprocess.run(
            [
                sys.executable,
                "-m",
                "memorizz.cli",
                "oracle",
                "preflight",
                "--index-policy",
                "lazy",
                "--json",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        _require(
            preflight.returncode == 0,
            f"CLI Oracle preflight failed: {preflight.stderr}",
        )
        preflight_report = json.loads(preflight.stdout)
        _require(preflight_report.get("ok") is True, "CLI Oracle preflight is not OK")
        return {
            "version": capability_report["version"],
            "features_reported": len(capability_report["features"]),
            "oracle_preflight": True,
        }

    def openai_provider(self) -> Dict[str, Any]:
        if self.args.skip_external:
            raise ProbeSkipped("external providers disabled with --skip-external")
        from memorizz.llms import OpenAI

        provider = OpenAI(
            api_key=_require_env("OPENAI_API_KEY"),
            model=self.args.openai_model,
        )
        response = provider.generate_text(
            "Reply with exactly OPENAI_OK and nothing else.",
            instructions="Follow the response format exactly.",
        )
        _require("OPENAI_OK" in str(response), "OpenAI marker was not returned")
        self.state["openai_model"] = provider
        return {"provider": "openai", "model": provider.model, "response": "verified"}

    def anthropic_provider(self) -> Dict[str, Any]:
        if self.args.skip_external:
            raise ProbeSkipped("external providers disabled with --skip-external")
        from memorizz.llms import Anthropic

        provider = Anthropic(
            api_key=_require_env("ANTHROPIC_API_KEY"),
            model=self.args.anthropic_model,
            # The same provider is reused by a full MemAgent delegation turn,
            # whose tool-aware system context needs more headroom than the
            # direct one-token marker probe below.
            max_tokens=256,
            enable_prompt_caching=False,
        )
        response = provider.generate_text(
            "Reply with exactly ANTHROPIC_OK and nothing else.",
            instructions="Follow the response format exactly.",
        )
        _require("ANTHROPIC_OK" in str(response), "Anthropic marker was not returned")
        self.state["anthropic_model"] = provider
        return {
            "provider": "anthropic",
            "model": provider.model,
            "response": "verified",
        }

    def tavily_provider(self) -> Dict[str, Any]:
        if self.args.skip_external:
            raise ProbeSkipped("external providers disabled with --skip-external")
        from memorizz.internet_access.providers.tavily import TavilyProvider

        provider = TavilyProvider(api_key=_require_env("TAVILY_API_KEY"), timeout=30)
        try:
            results = provider.search(
                "MemoRizz Model Context Protocol GitHub", max_results=2
            )
            _require(bool(results), "Tavily returned no results")
            _require(
                all(item.url for item in results), "Tavily result is missing a URL"
            )
            domains = sorted(
                {
                    urlparse(item.url).hostname
                    for item in results
                    if urlparse(item.url).hostname
                }
            )
            return {"provider": "tavily", "results": len(results), "domains": domains}
        finally:
            provider.close()

    def e2b_stateful_session(self) -> Dict[str, Any]:
        if self.args.skip_external or self.args.skip_e2b:
            raise ProbeSkipped("E2B disabled by verifier options")
        from memorizz.sandbox.providers.e2b_provider import E2BSandboxProvider

        path = f"/tmp/memorizz-050-{uuid.uuid4().hex[:10]}.txt"
        with E2BSandboxProvider(
            api_key=_require_env("E2B_API_KEY"),
            session_timeout=120,
            max_execution_timeout=30,
            allow_internet_access=False,
        ) as provider:
            _require(provider.write_file(path, "write"), "E2B file write failed")
            result = provider.execute_code(
                "from pathlib import Path\n"
                f"p = Path({path!r})\n"
                "p.write_text(p.read_text() + '-execute')\n"
                "print(p.read_text())",
                timeout=30,
            )
            _require(result.exit_code == 0, f"E2B execution failed: {result.error}")
            _require(
                provider.read_file(path) == "write-execute",
                "E2B write/execute/read did not share one filesystem",
            )
            metadata = result.to_dict().get("metadata") or {}
            _require(metadata.get("provider") == "e2b", "E2B metadata was omitted")
            _require(
                metadata.get("stateful_session") is True, "E2B session not stateful"
            )
            _require(metadata.get("egress_allowed") is False, "E2B egress policy lost")
            sdk_versions = metadata.get("sdk_versions") or {}
            _require(
                sdk_versions.get("e2b") and sdk_versions.get("e2b_code_interpreter"),
                "E2B SDK compatibility metadata was omitted",
            )
            return {
                "provider": metadata.get("provider"),
                "factory": metadata.get("factory"),
                "stateful_session": True,
                "write_execute_read": True,
                "execution_timeout": metadata.get("execution_timeout"),
                "egress_allowed": False,
                "resource_policy_enforcement": metadata.get(
                    "resource_policy_enforcement"
                ),
                "sdk_versions": sdk_versions,
            }

    def browser_control_approval_and_execution(self) -> Dict[str, Any]:
        if self.args.skip_external or self.args.skip_browser:
            raise ProbeSkipped("Browser Use disabled by verifier options")
        from memorizz.approval import ApprovalRequired, SQLiteApprovalStore
        from memorizz.browser_control import BrowserUseProvider
        from memorizz.memagent import MemAgent
        from memorizz.tooling import ContextPolicy

        token = uuid.uuid4().hex[:12]
        provider = BrowserUseProvider(
            command=os.environ.get("MEMORIZZ_BROWSER_USE_COMMAND", "browser-use"),
            llm_provider=self.args.browser_llm_provider,
            model=self.args.browser_model,
            headless=True,
            use_vision=False,
            allowed_domains=["example.com"],
            prohibited_domains=[],
            block_ip_addresses=True,
            max_steps=6,
            task_timeout=self.args.browser_timeout,
        )
        issue = provider.validate_configuration()
        _require(issue is None, f"Browser Use is not ready: {issue}")

        calls: list[Dict[str, Any]] = []
        original_run_task = provider.run_task

        def tracked_run_task(task, *, max_steps=None, timeout=None):
            calls.append({"task": task, "max_steps": max_steps, "timeout": timeout})
            return original_run_task(task, max_steps=max_steps, timeout=timeout)

        provider.run_task = tracked_run_task  # type: ignore[method-assign]
        store = SQLiteApprovalStore(self.temp_path / f"browser-{token}.sqlite3")
        agent = MemAgent(
            agent_id=f"verify-browser-{token}",
            browser_control=provider,
            approval_store=store,
            context_policy=ContextPolicy(tool_top_k=50),
            automations_enabled=False,
        )
        agent._current_memory_id = f"verify-browser-memory-{token}"
        agent._current_thread_id = f"verify-browser-thread-{token}"
        user_id = f"verify-browser-user-{token}"
        query = "Use browser control to read the Example Domain page title"
        arguments = {
            "task": (
                "Open https://example.com, read the page title, and return it. "
                "Do not navigate to any other domain."
            ),
            "max_steps": 6,
        }
        agent.semantic_tool_router.begin_turn(user_id=user_id)
        schemas = agent._build_llm_tools(query, user_id=user_id)
        serialized_schemas = json.dumps(schemas)
        _require("browser_control" in serialized_schemas, "Browser tool was hidden")
        _require('"approved"' not in serialized_schemas, "approved leaked to schema")
        _require('"confirm"' not in serialized_schemas, "confirm leaked to schema")
        try:
            agent._execute_and_record_tool_call(
                _tool_call("browser_control", arguments, f"browser-{token}"),
                [{"role": "user", "content": query}],
                workflow=None,
                user_id=user_id,
                query=query,
            )
        except ApprovalRequired as exc:
            proposal = exc.proposal
        else:
            raise AssertionError("Browser task executed without durable approval")
        _require(not calls, "Browser provider ran before host approval")
        _require(proposal.arguments == arguments, "Browser proposal arguments changed")
        _require(bool(proposal.argument_hash), "Browser argument hash is missing")

        agent.approve(proposal.proposal_id, approver_id="production-verifier")
        result = json.loads(agent.resume_approval(proposal.proposal_id))
        _require(len(calls) == 1, "Approved browser task did not execute exactly once")
        _require(result.get("success") is True, f"Browser task failed: {result}")
        _require(
            "example domain" in str(result.get("output") or "").lower(),
            "Browser task did not return the Example Domain title",
        )
        hosts = {
            urlparse(str(url)).hostname
            for url in result.get("urls") or []
            if urlparse(str(url)).hostname
        }
        _require(hosts.issubset({"example.com"}), f"Browser escaped allowlist: {hosts}")
        metadata = result.get("metadata") or {}
        _require(
            metadata.get("private_worker_process") is True,
            "Browser task did not report private worker isolation",
        )
        replay = json.loads(agent.resume_approval(proposal.proposal_id))
        _require(
            replay.get("error_code") == "invalid_approval_state",
            "Browser approval was reusable",
        )
        provider.close()
        return {
            "provider": "browseruse",
            "llm_provider": self.args.browser_llm_provider,
            "model": self.args.browser_model,
            "durable_approval": True,
            "single_use": True,
            "private_worker_process": True,
            "allowed_hosts": sorted(hosts),
            "steps": result.get("steps"),
            "output": result.get("output"),
        }

    def multi_agent_orchestration(self) -> Dict[str, Any]:
        if self.args.skip_external:
            raise ProbeSkipped("external providers disabled with --skip-external")
        from memorizz.memagent import MemAgent
        from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator

        openai_model = self.state.get("openai_model")
        anthropic_model = self.state.get("anthropic_model")
        _require(openai_model is not None, "OpenAI probe did not complete")
        _require(anthropic_model is not None, "Anthropic probe did not complete")
        token = uuid.uuid4().hex[:12]
        root_id = f"verify-root-{token}"
        delegate_id = f"verify-delegate-{token}"
        memory_id = f"verify-orchestration-{token}"
        thread_id = f"verify-thread-{token}"
        user_id = f"verify-user-{token}"
        workflow_id = f"verify-workflow-{token}"
        trace_id = f"verify-trace-{token}"
        self.remember_scope(memory_id=memory_id, agent_id=root_id, user_id=user_id)
        self.remember_scope(agent_id=delegate_id)
        root = MemAgent(
            model=openai_model,
            memory_provider=self.provider,
            agent_id=root_id,
            name="OpenAI coordinator",
            instruction="Consolidate delegated verification results concisely.",
            automations_enabled=False,
        )
        delegate = MemAgent(
            model=anthropic_model,
            memory_provider=self.provider,
            agent_id=delegate_id,
            name="Anthropic verifier",
            instruction=(
                "Complete the delegated check and include the marker "
                "DELEGATE_OK in the answer."
            ),
            automations_enabled=False,
        )
        orchestrator = MultiAgentOrchestrator(
            root,
            [delegate],
            delegation_plan=[
                {
                    "task_id": "provider-check",
                    "description": (
                        "Confirm provider abstraction propagation and include DELEGATE_OK."
                    ),
                    "assigned_agent_id": delegate_id,
                    "priority": 1,
                    "dependencies": [],
                }
            ],
            persist_participants=True,
            workflow_id=workflow_id,
        )
        report = orchestrator.execute_multi_agent_workflow(
            "Verify cross-provider orchestration.",
            memory_id,
            thread_id,
            user_id=user_id,
            context={"release": "0.5.0"},
            tool_context={"request_id": token},
            trace_id=trace_id,
            return_report=True,
        )
        _require(report.get("workflow_id") == workflow_id, "Workflow ID was lost")
        _require(report.get("trace_id") == trace_id, "Trace ID was lost")
        _require(report.get("user_id") == user_id, "User scope was lost")
        tasks = report.get("tasks") or []
        _require(len(tasks) == 1, "Deterministic delegation plan was not used")
        _require(tasks[0].get("status") == "completed", f"Delegate failed: {tasks}")
        _require(
            "DELEGATE_OK" in str(tasks[0].get("result")),
            "Anthropic delegate result marker is missing",
        )
        _require(
            self.provider.retrieve_memagent(root_id) is not None
            and self.provider.retrieve_memagent(delegate_id) is not None,
            "Requested participant persistence did not occur",
        )
        return {
            "root_provider": "openai",
            "delegate_provider": "anthropic",
            "deterministic_plan": True,
            "participants_persisted": True,
            "user_context_trace_propagated": True,
            "tasks_completed": 1,
            "partial": report.get("partial"),
        }

    def _mcp_manager(self, owner_id: str, servers: list[Dict[str, Any]]):
        from memorizz.approval import SQLiteApprovalStore
        from memorizz.mcp import MCPClientManager
        from memorizz.mcp.audit import MCPAuditLogger

        safe_owner = re.sub(r"[^A-Za-z0-9_.-]", "_", owner_id)
        return MCPClientManager(
            owner_id=owner_id,
            servers=servers,
            credential_store=InMemoryCredentialStore(),
            audit_logger=MCPAuditLogger(self.temp_path / f"{safe_owner}-audit.jsonl"),
            approval_store=SQLiteApprovalStore(
                self.temp_path / f"{safe_owner}-mcp-approvals.sqlite3"
            ),
        )

    def first_party_mcp_stdio(self) -> Dict[str, Any]:
        if self.args.skip_mcp:
            raise ProbeSkipped("MCP probes disabled with --skip-mcp")
        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-mcp-stdio-{token}"
        self.remember_scope(memory_id=memory_id)
        environment = {
            "MEMORIZZ_HOME": str(self.temp_path / f"mcp-stdio-{token}"),
            "MEMORIZZ_BACKEND": "oracle",
            "MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING": "true",
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
            "ORACLE_USER": _require_env("ORACLE_USER"),
            "ORACLE_PASSWORD": _require_env("ORACLE_PASSWORD"),
            "ORACLE_DSN": _require_env("ORACLE_DSN"),
            "PYTHONPATH": str(SOURCE_ROOT),
        }
        manager = self._mcp_manager(
            f"stdio-{token}",
            [
                {
                    "name": "memorizz-server",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "memorizz.mcp_server"],
                    "cwd": str(REPOSITORY_ROOT),
                    "env": environment,
                    "timeout": 30,
                }
            ],
        )
        tools = manager.list_tools("memorizz-server")
        _require(tools.get("ok") is True, f"MCP tools/list failed: {tools}")
        _require(len(tools.get("tools") or []) == 11, "MCP server tool surface drifted")
        resources = manager.list_resources("memorizz-server")
        prompts = manager.list_prompts("memorizz-server")
        _require(resources.get("ok") is True, "MCP resources/list failed")
        _require(prompts.get("ok") is True, "MCP prompts/list failed")
        info = manager.call_tool("memorizz-server", "memorizz_server_info", {})
        _require(
            _structured_mcp_result(info).get("version") == "0.5.0",
            "MCP version mismatch",
        )
        proposed = manager.call_tool(
            "memorizz-server",
            "memorizz_store_memory",
            {
                "content": "Stored through the MemoRizz 0.5 MCP server",
                "memory_type": "knowledge_base",
                "memory_id": memory_id,
            },
        )
        _require(
            proposed.get("error_code") == "approval_required",
            "MCP mutation did not require durable approval",
        )
        proposal_id = proposed["proposal"]["proposal_id"]
        manager.approve_tool_call(proposal_id, approver_id="production-verifier")
        stored = manager.resume_tool_call(proposal_id)
        stored_payload = _structured_mcp_result(stored)
        _require(stored_payload.get("ok") is True, "MCP resume failed")
        record_id = str(stored_payload.get("record_id") or "")
        _require(record_id, "MCP store omitted its physical Oracle record ID")
        replay = manager.resume_tool_call(proposal_id)
        _require(
            replay.get("error_code") == "invalid_approval_state",
            "MCP approval was reusable",
        )
        listed = manager.call_tool(
            "memorizz-server",
            "memorizz_list_memories",
            {"memory_type": "knowledge_base", "memory_id": memory_id},
        )
        _require(
            _structured_mcp_result(listed).get("count") == 1, "MCP Oracle write missing"
        )
        fetched = manager.call_tool(
            "memorizz-server",
            "memorizz_get_memory",
            {"record_id": record_id, "memory_type": "knowledge_base"},
        )
        _require(
            _structured_mcp_result(fetched).get("memory", {}).get("_id") == record_id,
            "MCP Oracle by-ID round trip failed",
        )
        return {
            "transport": "stdio",
            "tools": 11,
            "resources": len(resources.get("resources") or []),
            "prompts": len(prompts.get("prompts") or []),
            "oracle_write_read": True,
            "oracle_by_id": True,
            "durable_approval": True,
            "single_use": True,
        }

    @staticmethod
    def _free_loopback_port() -> int:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def first_party_mcp_http(self) -> Dict[str, Any]:
        if self.args.skip_mcp:
            raise ProbeSkipped("MCP probes disabled with --skip-mcp")
        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-mcp-http-{token}"
        self.remember_scope(memory_id=memory_id)
        port = self._free_loopback_port()
        alice_token = f"alice-{uuid.uuid4().hex}"
        bob_token = f"bob-{uuid.uuid4().hex}"
        environment = dict(os.environ)
        environment.update(
            {
                "MEMORIZZ_HOME": str(self.temp_path / f"mcp-http-{token}"),
                "MEMORIZZ_BACKEND": "oracle",
                "MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING": "true",
                "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
                "ORACLE_USER": _require_env("ORACLE_USER"),
                "ORACLE_PASSWORD": _require_env("ORACLE_PASSWORD"),
                "ORACLE_DSN": _require_env("ORACLE_DSN"),
                "PYTHONPATH": str(SOURCE_ROOT),
                "MEMORIZZ_MCP_SERVER_API_KEYS": json.dumps(
                    {"alice": alice_token, "bob": bob_token}
                ),
            }
        )
        command = [
            sys.executable,
            "-m",
            "memorizz.cli",
            "mcp",
            "serve",
            "--transport",
            "streamable-http",
            "--port",
            str(port),
            "--public-url",
            f"http://127.0.0.1:{port}",
            "--allow-writes",
            "--no-agent-execution",
        ]
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                        break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Authenticated MCP server exited at startup")
                    time.sleep(0.1)
            else:
                raise RuntimeError("Authenticated MCP server did not become ready")

            def manager_for(principal: str, bearer: str):
                return self._mcp_manager(
                    f"http-{principal}-{token}",
                    [
                        {
                            "name": "remote",
                            "transport": "streamable_http",
                            "url": f"http://127.0.0.1:{port}/mcp",
                            "allow_private_network": True,
                            "auth": {"type": "bearer", "token": bearer},
                            "timeout": 30,
                        }
                    ],
                )

            alice = manager_for("alice", alice_token)
            bob = manager_for("bob", bob_token)
            _require(
                alice.test_connection("remote").get("ok") is True,
                "HTTP MCP connect failed",
            )
            proposed = alice.call_tool(
                "remote",
                "memorizz_store_memory",
                {
                    "content": "Alice-only Oracle memory",
                    "memory_type": "knowledge_base",
                    "memory_id": memory_id,
                },
            )
            _require(
                proposed.get("error_code") == "approval_required",
                "Authenticated MCP write did not require approval",
            )
            proposal_id = proposed["proposal"]["proposal_id"]
            alice.approve_tool_call(proposal_id, approver_id="production-verifier")
            stored = alice.resume_tool_call(proposal_id)
            stored_payload = _structured_mcp_result(stored)
            _require(stored_payload.get("ok") is True, "HTTP MCP write failed")
            record_id = str(stored_payload.get("record_id") or "")
            _require(record_id, "HTTP MCP store omitted its Oracle record ID")
            alice_rows = _structured_mcp_result(
                alice.call_tool(
                    "remote",
                    "memorizz_list_memories",
                    {"memory_type": "knowledge_base", "memory_id": memory_id},
                )
            )
            bob_rows = _structured_mcp_result(
                bob.call_tool(
                    "remote",
                    "memorizz_list_memories",
                    {"memory_type": "knowledge_base", "memory_id": memory_id},
                )
            )
            _require(alice_rows.get("count") == 1, "Alice cannot read her MCP row")
            _require(bob_rows.get("count") == 0, "Bob can read Alice's MCP row")
            alice_record = _structured_mcp_result(
                alice.call_tool(
                    "remote",
                    "memorizz_get_memory",
                    {"record_id": record_id, "memory_type": "knowledge_base"},
                )
            )
            _require(
                alice_record.get("memory", {}).get("_id") == record_id,
                "Alice cannot read her Oracle record by ID",
            )
            bob_record = bob.call_tool(
                "remote",
                "memorizz_get_memory",
                {"record_id": record_id, "memory_type": "knowledge_base"},
            )
            _require(
                bob_record.get("ok") is False,
                "Bob can read Alice's Oracle record by ID",
            )
            return {
                "transport": "streamable-http",
                "bearer_auth": True,
                "durable_approval": True,
                "oracle_write_read": True,
                "oracle_by_id": True,
                "tenant_isolation": True,
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def hosted_mcp_auth_boundaries(self) -> Dict[str, Any]:
        if self.args.skip_external or self.args.skip_mcp:
            raise ProbeSkipped("hosted MCP probes disabled by verifier options")
        notion = self._mcp_manager(
            "hosted-notion",
            [
                {
                    "name": "notion",
                    "transport": "streamable_http",
                    "url": "https://mcp.notion.com/mcp",
                    "auth": {"type": "none"},
                    "timeout": 20,
                    "max_retries": 0,
                }
            ],
        )
        notion_result = notion.test_connection("notion")
        _require(
            notion_result.get("ok") is False
            and notion_result.get("error_code") == "authorization_required",
            "Notion did not report the expected authentication boundary: "
            f"{notion_result}",
        )

        # Google publishes the Calendar tool catalog without credentials and
        # applies authorization when a tool accesses calendar data. Verify both
        # halves so public discovery is not misclassified as an auth bypass.
        google = self._mcp_manager(
            "hosted-google-calendar",
            [
                {
                    "name": "google-calendar",
                    "transport": "streamable_http",
                    "url": "https://calendarmcp.googleapis.com/mcp/v1",
                    "auth": {"type": "none"},
                    "timeout": 20,
                    "max_retries": 0,
                }
            ],
        )
        google_discovery = google.test_connection("google-calendar")
        _require(
            google_discovery.get("ok") is True
            and int(google_discovery.get("tool_count") or 0) > 0,
            f"Google Calendar tool discovery failed: {google_discovery}",
        )
        google_read = google.call_tool("google-calendar", "list_calendars", {})
        _require(
            google_read.get("ok") is False
            and google_read.get("error_code") == "authorization_required",
            "Google Calendar protected read did not require authorization: "
            f"{google_read}",
        )
        return {
            "notion_endpoint": notion_result["error_code"],
            "google_calendar_discovery": int(google_discovery["tool_count"]),
            "google_calendar_protected_read": google_read["error_code"],
            "oauth_success_requires_operator_credentials": True,
        }

    def scoped_cleanup_contract(self) -> Dict[str, Any]:
        from memorizz.enums import MemoryType

        token = uuid.uuid4().hex[:12]
        memory_id = f"verify-cleanup-{token}"
        user_id = f"verify-cleanup-user-{token}"
        agent_id = f"verify-cleanup-agent-{token}"
        self.provider.store(
            data={
                "content": "transactional cleanup marker",
                "memory_id": memory_id,
                "user_id": user_id,
                "agent_id": agent_id,
            },
            memory_store_type=MemoryType.SHORT_TERM_MEMORY,
        )
        report = self.provider.delete_scope(
            memory_id=memory_id,
            user_id=user_id,
            agent_ids=[agent_id],
        )
        _require(report.get("ok") is True, "delete_scope did not return success")
        _require(report.get("total_deleted", 0) >= 1, "delete_scope deleted no rows")
        remaining = self.provider.retrieve_by_query(
            {"memory_id": memory_id, "user_id": user_id},
            memory_store_type=MemoryType.SHORT_TERM_MEMORY,
            limit=10,
        )
        _require(not remaining, "delete_scope left the scoped row behind")
        return {
            "transactional": True,
            "reported_counts": report.get("counts"),
            "total_deleted": report.get("total_deleted"),
        }

    def cleanup(self) -> Dict[str, Any]:
        provider = self.state.get("provider")
        if provider is None:
            raise ProbeSkipped("Oracle provider was not created")
        total = 0
        cleanup_reports = 0
        for memory_id in sorted(self.state["memory_ids"]):
            report = provider.delete_scope(memory_id=memory_id)
            total += int(report.get("total_deleted", 0))
            cleanup_reports += 1
        for user_id in sorted(self.state["user_ids"]):
            report = provider.delete_scope(user_id=user_id)
            total += int(report.get("total_deleted", 0))
            cleanup_reports += 1
        agent_ids = sorted(self.state["agent_ids"])
        if agent_ids:
            report = provider.delete_scope(agent_ids=agent_ids)
            total += int(report.get("total_deleted", 0))
            cleanup_reports += 1
        return {
            "scope_reports": cleanup_reports,
            "rows_deleted": total,
            "memory_scopes": len(self.state["memory_ids"]),
            "user_scopes": len(self.state["user_ids"]),
            "agent_scopes": len(agent_ids),
        }

    def close(self) -> None:
        provider = self.state.get("provider")
        if provider is not None:
            try:
                provider.close()
            except Exception:
                pass
        self.tempdir.cleanup()

    def report(self) -> Dict[str, Any]:
        failed_required = [
            item for item in self.results if item.required and item.status == "failed"
        ]
        return {
            "schema": "memorizz-production-verification-v1",
            "release": "0.5.0",
            "ok": not failed_required,
            "summary": {
                "passed": sum(item.status == "passed" for item in self.results),
                "failed": sum(item.status == "failed" for item in self.results),
                "skipped": sum(item.status == "skipped" for item in self.results),
                "required_failures": len(failed_required),
            },
            "results": [asdict(item) for item in self.results],
            "limitations": {
                "notion_oauth_success": (
                    "requires an operator to authorize a Notion workspace"
                ),
                "google_calendar_oauth_success": (
                    "requires Developer Preview enrollment, a Google Cloud project, "
                    "and operator OAuth credentials"
                ),
                "graalpy_java_untrusted": (
                    "requires a matching GraalVM build and compiled wrapper JAR"
                ),
                "e2b_cpu_memory": (
                    "requires an explicit E2B template; the default probe verifies "
                    "session, timeout, and egress policy only"
                ),
                "browser_authenticated_profiles": (
                    "the default browser probe uses public example.com only; "
                    "authenticated workflows require operator-owned profiles"
                ),
            },
        }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify MemoRizz 0.5.0 against real production dependencies."
    )
    parser.add_argument(
        "--oracle-dsn",
        help="Override ORACLE_DSN (credentials are intentionally not accepted as flags).",
    )
    parser.add_argument(
        "--oracle-container",
        default=None,
        help="Existing Docker container name (or MEMORIZZ_ORACLE_CONTAINER).",
    )
    parser.add_argument("--openai-model", default="gpt-4.1-mini")
    parser.add_argument("--anthropic-model", default="claude-haiku-4-5-20251001")
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help=(
            "Skip OpenAI, Anthropic, Tavily, E2B, Browser Use, hosted MCP, "
            "and orchestration calls."
        ),
    )
    parser.add_argument("--skip-e2b", action="store_true")
    parser.add_argument("--skip-browser", action="store_true")
    parser.add_argument(
        "--browser-llm-provider",
        choices=("browseruse", "openai", "anthropic", "google"),
        default="openai",
    )
    parser.add_argument("--browser-model", default="gpt-4.1-mini")
    parser.add_argument("--browser-timeout", type=int, default=300)
    parser.add_argument("--skip-mcp", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit only JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _load_environment()
    if args.oracle_dsn:
        os.environ["ORACLE_DSN"] = args.oracle_dsn
    if args.oracle_container:
        os.environ["MEMORIZZ_ORACLE_CONTAINER"] = args.oracle_container
    else:
        args.oracle_container = os.environ.get(
            "MEMORIZZ_ORACLE_CONTAINER", "memorizz_oracle"
        )
    os.environ["MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING"] = "true"
    os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] = ""

    verifier = ProductionVerifier(args)
    try:
        verifier.run("Docker runtime readiness", "oracle", verifier.oracle_runtime)
        verifier.run(
            "database and vector preflight", "oracle", verifier.oracle_preflight
        )
        verifier.run(
            "schema migration and in-database embedding",
            "oracle",
            verifier.oracle_schema_and_embedding,
        )
        verifier.run(
            "complete Toolbox JSON Schema round-trip",
            "oracle",
            verifier.toolbox_schema_round_trip,
        )
        verifier.run(
            "atomic summary compaction and expansion",
            "oracle",
            verifier.summary_compaction_round_trip,
        )
        verifier.run(
            "semantic-cache governance and invalidation",
            "oracle",
            verifier.semantic_cache_governance,
        )
        verifier.run(
            "progressive tool disclosure and dispatch",
            "governance",
            verifier.progressive_tool_router,
        )
        verifier.run(
            "durable automation approval checkpoint",
            "governance",
            verifier.durable_automation_approval,
        )
        verifier.run(
            "size-aware tool-result expansion boundary",
            "governance",
            verifier.size_aware_tool_results,
        )
        verifier.run(
            "builder parity and governed semantic layer",
            "sdk",
            verifier.builder_and_semantic_layer,
        )
        verifier.run(
            "GraalPy execution-provider boundary",
            "sandbox",
            verifier.graalpy_execution_boundary,
        )
        verifier.run(
            "CLI capabilities and Oracle preflight",
            "cli",
            verifier.cli_capabilities_and_preflight,
        )
        verifier.run("OpenAI provider", "external", verifier.openai_provider)
        verifier.run("Anthropic provider", "external", verifier.anthropic_provider)
        verifier.run("Tavily search provider", "external", verifier.tavily_provider)
        verifier.run(
            "stateful E2B 2.x session", "external", verifier.e2b_stateful_session
        )
        verifier.run(
            "durably approved Browser Use task",
            "browser",
            verifier.browser_control_approval_and_execution,
        )
        verifier.run(
            "configured cross-provider orchestration",
            "orchestration",
            verifier.multi_agent_orchestration,
        )
        verifier.run(
            "first-party MemoRizz MCP over stdio",
            "mcp",
            verifier.first_party_mcp_stdio,
        )
        verifier.run(
            "authenticated MemoRizz MCP over HTTP",
            "mcp",
            verifier.first_party_mcp_http,
        )
        verifier.run(
            "Notion and Google Calendar authentication boundaries",
            "mcp",
            verifier.hosted_mcp_auth_boundaries,
        )
        verifier.run(
            "transactional scoped lifecycle cleanup",
            "oracle",
            verifier.scoped_cleanup_contract,
        )
        verifier.run("verification-data teardown", "cleanup", verifier.cleanup)
        report = verifier.report()
    finally:
        verifier.close()

    print(json.dumps(report, indent=None if args.json else 2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
