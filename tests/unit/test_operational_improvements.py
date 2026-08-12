from __future__ import annotations

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from memorizz import ApprovalResumeResult, MemAgent, SemanticCacheInspection
from memorizz.approval import SQLiteApprovalStore
from memorizz.enums import MemoryType
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.memory_provider.oracle.provider import OracleProvider
from memorizz.short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig
from memorizz.task_decomposition import SubTask, normalize_delegation_config


class _EmbeddingManager:
    def get_embedding(self, _query):
        return [1.0, 0.0, 0.0]


def test_subtask_round_trip_and_delegation_plan_are_json_safe():
    task = SubTask("collect", "Collect evidence", "researcher", priority=2)
    task.status = "completed"
    task.result = {"sources": 3}

    restored = SubTask.from_dict(task.to_dict())
    assert restored.to_dict() == task.to_dict()

    config = normalize_delegation_config(
        {"mode": "deterministic", "plan": [task]}, for_persistence=True
    )
    assert config["plan"] == [task.to_dict()]
    json.dumps(config)

    agent = (
        MemAgentBuilder()
        .with_delegation(mode="deterministic", plan=[task])
        .build(validate=False)
    )
    assert agent.delegation_config["plan"] == [task.to_dict()]


def test_callable_delegation_plan_is_explicitly_runtime_only():
    config = normalize_delegation_config({"plan": lambda *_args: []})
    assert callable(config["plan"])
    with pytest.raises(TypeError, match="runtime-only"):
        normalize_delegation_config(config, for_persistence=True)


def test_generate_summaries_uses_public_scope_not_last_run_state():
    class Manager:
        def __init__(self):
            self.kwargs = None

        def generate_summaries(self, **kwargs):
            self.kwargs = kwargs
            return ["summary-1"]

    agent = MemAgent()
    manager = Manager()
    agent.memory_manager = manager
    agent.memory_ids = ["configured-memory"]
    agent._current_memory_id = "last-memory"
    agent._current_user_id = "wrong-last-user"
    agent._current_thread_id = "wrong-last-thread"

    result = agent.generate_summaries(
        memory_id="requested-memory",
        user_id="alice",
        thread_id="thread-7",
    )

    assert result == ["summary-1"]
    assert manager.kwargs["memory_ids"] == ["requested-memory"]
    assert manager.kwargs["current_memory_id"] == "requested-memory"
    assert manager.kwargs["user_id"] == "alice"
    assert manager.kwargs["thread_id"] == "thread-7"

    agent.generate_summaries(memory_id="requested-memory")
    assert manager.kwargs["user_id"] is None
    assert manager.kwargs["thread_id"] is None


def test_memory_manager_summary_compaction_filters_tenant_and_thread(
    monkeypatch,
):
    now = time.time()

    class Provider:
        def __init__(self):
            self.summary = None

        def retrieve_conversation_history_ordered_by_timestamp(self, **_kwargs):
            return [
                {
                    "_id": "alice-t1",
                    "memory_id": "m1",
                    "user_id": "alice",
                    "thread_id": "t1",
                    "role": "user",
                    "content": "include",
                    "timestamp": now,
                },
                {
                    "_id": "alice-t2",
                    "memory_id": "m1",
                    "user_id": "alice",
                    "thread_id": "t2",
                    "role": "user",
                    "content": "wrong thread",
                    "timestamp": now,
                },
                {
                    "_id": "bob-t1",
                    "memory_id": "m1",
                    "user_id": "bob",
                    "thread_id": "t1",
                    "role": "user",
                    "content": "wrong tenant",
                    "timestamp": now,
                },
            ]

        def store_summary_with_links(self, value):
            self.summary = value
            return "summary-1"

    class Model:
        def generate(self, _messages):
            return "scoped summary"

    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda _text: [0.0])
    provider = Provider()
    manager = MemoryManager(provider)

    result = manager.generate_summaries(
        model=Model(),
        agent_id="agent-1",
        memory_ids=["m1"],
        current_memory_id="m1",
        user_id="alice",
        thread_id="t1",
        days_back=1,
    )

    assert result == ["summary-1"]
    assert provider.summary["source_message_ids"] == ["alice-t1"]
    assert provider.summary["user_id"] == "alice"


def test_memory_manager_omitted_tool_log_tenant_filter_is_unscoped():
    marker = object()

    class Provider:
        def __init__(self):
            self.user_id = marker

        def list_tool_logs(self, memory_id, limit=20, user_id=marker, thread_id=None):
            self.user_id = user_id
            return []

    provider = Provider()
    manager = MemoryManager(provider)
    manager.list_tool_logs("memory-1")
    assert provider.user_id is marker

    manager.list_tool_logs("memory-1", user_id=None)
    assert provider.user_id is None


def test_approval_resume_keeps_exact_tool_result_separate_from_model(tmp_path):
    calls = []
    large_value = "x" * 400

    def update_inventory(sku: str, quantity: int):
        calls.append((sku, quantity))
        return {
            "sku": sku,
            "quantity": quantity,
            "updated": True,
            "receipt": large_value,
        }

    class Model:
        def generate(self, _messages, tools=None):
            return "Unrelated commentary about yesterday's inventory."

    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    agent = MemAgent(
        model=Model(),
        tools=[update_inventory],
        approval_store=store,
        tool_result_policy={"offload_above_chars": 10},
    )

    class ToolLogs:
        def __init__(self):
            self.stored = []

        def store_tool_log(self, **value):
            self.stored.append(value)
            return "tool-log-1"

    tool_logs = ToolLogs()
    agent.memory_manager = tool_logs
    arguments = {"sku": "A-7", "quantity": 4}
    checkpoint = {
        "query": "update A-7",
        "messages": [{"role": "user", "content": "update A-7"}],
        "memory_id": "memory-1",
        "thread_id": "thread-1",
        "user_id": "alice",
        "logical_tool_name": "update_inventory",
        "logical_arguments": arguments,
        "model_tool_name": "update_inventory",
        "tool_call_id": "call-1",
        "router_state": {
            "selected": ["update_inventory"],
            "call_attempts": {},
            "successful_calls": [],
            "invocation_count": 0,
            "user_id": "alice",
        },
    }
    proposal = store.propose(
        owner_id=agent.agent_id,
        tool_name="update_inventory",
        arguments=arguments,
        policy_reason="External mutation",
        checkpoint=checkpoint,
    )
    store.approve(proposal.proposal_id, approver_id="operator@example.com")

    resumed = agent.resume_approval(proposal.proposal_id)

    assert isinstance(resumed, ApprovalResumeResult)
    assert resumed.consumed is True
    assert resumed.tool_result == {
        "quantity": 4,
        "receipt": large_value,
        "sku": "A-7",
        "updated": True,
    }
    assert resumed.assistant_response.startswith("Unrelated commentary")
    assert calls == [("A-7", 4)]
    assert json.loads(tool_logs.stored[0]["result"]) == resumed.tool_result


def test_run_stream_can_raise_provider_authentication_failures():
    class AuthenticationError(RuntimeError):
        status_code = 401

    class Model:
        def generate_stream(self, _messages, tools=None):
            raise AuthenticationError("401 invalid API key")
            yield  # pragma: no cover

    agent = MemAgent(model=Model())
    events = []
    agent.set_stream_event_callback(events.append)

    with pytest.raises(AuthenticationError):
        list(agent.run_stream("hello", raise_on_provider_error=True))

    error = next(event for event in events if event["type"] == "error")
    assert error["terminal"] is True
    assert error["error_code"] == "provider_authentication_failed"
    assert error["provider_status_code"] == 401


def test_builder_environment_presets(monkeypatch):
    from memorizz.memory_provider.oracle import LocalOracleRuntime

    calls = []

    class Runtime:
        def ensure_ready(self):
            calls.append("ready")
            return {"ok": True, "state": "running"}

    class Provider:
        def preflight(self):
            calls.append("preflight")
            return {"ok": True, "version_full": "23.26.0.0.0"}

        def close(self):
            calls.append("close")

    monkeypatch.setattr(
        LocalOracleRuntime,
        "from_env",
        classmethod(lambda cls, **kwargs: Runtime()),
    )
    monkeypatch.setattr(
        OracleProvider,
        "from_env",
        classmethod(lambda cls, **kwargs: Provider()),
    )

    builder = MemAgentBuilder().with_oracle_from_env()
    assert calls == ["ready", "preflight"]
    assert builder._environment_reports["oracle"]["preflight"]["version_full"] == (
        "23.26.0.0.0"
    )

    monkeypatch.setenv("E2B_API_KEY", "test-only-key")
    e2b_builder = MemAgentBuilder().with_e2b_from_env()
    assert e2b_builder._environment_reports["e2b"]["ok"] is True
    assert e2b_builder._sandbox_provider.api_key == "test-only-key"


def test_semantic_cache_inspection_exposes_freshness_and_invalidation_data():
    cache = SemanticCache(
        config=SemanticCacheConfig(
            ttl_hours=1,
            freshness_by_domain={"inventory": 120},
            require_fingerprint_match=False,
            enable_memory_provider_sync=False,
        ),
        embedding_manager=_EmbeddingManager(),
        agent_id="agent-1",
        memory_id="memory-1",
    )
    assert cache.set(
        "current inventory",
        "42 units",
        user_id="alice",
        metadata={"domain": "inventory", "domains": ["erp"], "tags": ["stock"]},
    )

    inspection = cache.inspect("inventory now", user_id="alice")
    assert isinstance(inspection, SemanticCacheInspection)
    assert inspection.hit is True
    assert inspection.matched_query == "current inventory"
    assert inspection.cache_key
    assert inspection.similarity == pytest.approx(1.0)
    assert inspection.ttl_seconds == 120
    assert 0 < inspection.expires_in_seconds <= 120
    assert inspection.hit_count == 0
    assert inspection.invalidation_domains == ["erp", "inventory"]
    assert inspection.invalidation_tags == ["stock"]

    bypass = cache.inspect(
        "inventory now", user_id="alice", bypass_reason="side_effecting_tool"
    )
    assert bypass.hit is False
    assert bypass.bypass_reason == "side_effecting_tool"


def test_observability_summary_is_tenant_and_agent_scoped():
    class Provider:
        def retrieve_conversation_history_ordered_by_timestamp(self, **_kwargs):
            trace = json.dumps({"type": "trace_bundle", "events": [{"kind": "tool"}]})
            return [
                {
                    "memory_id": "m1",
                    "user_id": "alice",
                    "agent_id": "agent-1",
                    "thread_id": "t1",
                    "role": "user",
                    "content": "hello",
                    "timestamp": 1,
                },
                {
                    "memory_id": "m1",
                    "user_id": "alice",
                    "agent_id": "agent-1",
                    "thread_id": "t1",
                    "role": "tool",
                    "content": trace,
                    "timestamp": 2,
                },
                {
                    "memory_id": "m1",
                    "user_id": "bob",
                    "agent_id": "agent-1",
                    "thread_id": "t1",
                    "role": "user",
                    "content": "private",
                    "timestamp": 3,
                },
            ]

        def list_tool_logs(self, **_kwargs):
            return [
                {
                    "memory_id": "m1",
                    "user_id": "alice",
                    "agent_id": "agent-1",
                    "thread_id": "t1",
                    "success": True,
                }
            ]

        def list_all(self, memory_type, user_id=None):
            if memory_type == MemoryType.WORKFLOW_MEMORY:
                return [
                    {
                        "memory_id": "m1",
                        "user_id": "alice",
                        "agent_id": "agent-1",
                        "outcome": "success",
                    }
                ]
            if memory_type == MemoryType.SUMMARIES:
                return [
                    {
                        "memory_id": "m1",
                        "user_id": "alice",
                        "agent_id": "agent-1",
                    }
                ]
            return []

        def close(self):
            pass

    agent = MemAgent(memory_provider=Provider(), agent_id="agent-1")
    summary = agent.observability_summary("m1", "alice", thread_id="t1")

    assert summary["conversation"]["row_count"] == 2
    assert summary["conversation"]["message_count"] == 1
    assert summary["conversation"]["trace_event_count"] == 1
    assert summary["tool_logs"]["success_count"] == 1
    assert summary["workflows"]["outcomes"] == {"success": 1}
    assert summary["summaries"]["count"] == 1


def test_agent_lifecycle_runs_scoped_cleanup_before_closing():
    events = []

    class Provider:
        def delete_scope(self, **scope):
            events.append(("cleanup", scope))
            return {"ok": True, "total_deleted": 2}

        def close(self):
            events.append(("close", "memory"))

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("close", self.name))

    provider = Provider()
    agent = MemAgent(memory_provider=provider)
    agent.sandbox_manager = Resource("sandbox")
    agent.browser_control_manager = Resource("browser")

    with agent.lifecycle(
        cleanup_scope={
            "memory_id": "m1",
            "user_id": "alice",
            "agent_ids": [agent.agent_id],
        }
    ) as active:
        assert active is agent

    assert events[0][0] == "cleanup"
    assert ("close", "sandbox") in events
    assert ("close", "browser") in events
    assert events[-1] == ("close", "memory")
    assert agent._last_close_report["ok"] is True


def test_oracle_preflight_reports_full_patch_version():
    class Cursor:
        def __init__(self):
            self.sql = ""

        def execute(self, sql, _params=None):
            self.sql = " ".join(str(sql).lower().split())

        def fetchone(self):
            if "product_component_version" in self.sql:
                return (
                    "Oracle Database 23ai Free",
                    "23.0.0.0.0",
                    "23.26.0.0.0",
                )
            if "version_full from v$instance" in self.sql:
                return ("23.26.0.0.0",)
            if "sys_context" in self.sql:
                return ("FREEPDB1",)
            if "open_mode from v$pdbs" in self.sql:
                return ("READ WRITE",)
            if "from v$parameter" in self.sql:
                return ("0",)
            return (0,)

        def fetchall(self):
            if "session_privs" in self.sql:
                return [("CREATE SESSION",), ("CREATE TABLE",)]
            return []

    class Connection:
        def cursor(self):
            return Cursor()

    provider = object.__new__(OracleProvider)
    provider.config = SimpleNamespace(
        dsn="localhost:1521/FREEPDB1",
        schema="MEMORIZZ",
        index_policy="lazy",
    )
    provider._embedding_provider = None

    @contextmanager
    def connection():
        yield Connection()

    provider._get_connection = connection
    provider.get_vector_schema_dimensions = lambda: {}
    provider.recommended_vector_memory_size = lambda: {"recommended": "1G"}

    report = provider.preflight()
    assert report["database_version"] == "23.0.0.0.0"
    assert report["version_full"] == "23.26.0.0.0"
    assert report["ok"] is True

    provider._embedding_provider = SimpleNamespace(
        get_default_model=lambda: "test-embedder",
        get_dimensions=lambda: 256,
    )
    provider.get_vector_schema_dimensions = lambda: {"TOOLBOX.EMBEDDING": 384}
    mismatch = provider.preflight()
    assert mismatch["ok"] is False
    assert mismatch["embedding_dimension_compatible"] is False
    assert mismatch["embedding_dimension_mismatches"] == {"TOOLBOX.EMBEDDING": 384}
    assert any(
        "embedding dimension mismatch" in item for item in mismatch["diagnostics"]
    )
