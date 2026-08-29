"""Regressions for scoped recall, shared-agent safety, and lean installs."""

from __future__ import annotations

import importlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz import MemAgent, RetrievalPolicy, capabilities
from memorizz.enums import MemoryType
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memagent.managers.memory_manager import MemoryManager
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.short_term_memory.semantic_cache import SemanticCache, SemanticCacheConfig
from memorizz.tooling import ContextPolicy


class _EmbeddingManager:
    def get_embedding(self, _text: str):
        return [1.0, 0.0]


def _tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.mark.unit
def test_retrieval_policy_separates_storage_from_automatic_recall():
    disabled = RetrievalPolicy.disabled()
    assert disabled.conversation_scope == "disabled"
    assert disabled.knowledge_base_scope == "disabled"
    assert RetrievalPolicy.from_value(False) == disabled
    assert RetrievalPolicy.from_value("thread").conversation_scope == "thread"

    agent = MemAgent(
        memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.KNOWLEDGE_BASE],
        retrieval_policy=disabled,
    )
    assert set(agent.active_memory_types) == {
        MemoryType.CONVERSATION_MEMORY,
        MemoryType.KNOWLEDGE_BASE,
    }
    assert agent.retrieval_policy == disabled

    built = (
        MemAgentBuilder()
        .with_retrieval_policy(
            {
                "conversation_scope": "thread",
                "knowledge_base_scope": "namespace",
                "knowledge_base_namespaces": ["agent-harness"],
            }
        )
        .build(validate=False)
    )
    assert built.retrieval_policy.to_dict() == {
        "conversation_scope": "thread",
        "knowledge_base_scope": "namespace",
        "knowledge_base_namespaces": ["agent-harness"],
    }


@pytest.mark.unit
def test_memory_manager_enforces_thread_and_namespace_after_provider_query():
    class BroadProvider:
        def __init__(self):
            self.calls = []

        def retrieve_by_query(self, **kwargs):
            self.calls.append(kwargs)
            return [
                {
                    "_id": "right",
                    "memory_id": "memory-1",
                    "user_id": "alice",
                    "thread_id": "thread-a",
                    "namespace": "agents",
                },
                {
                    "_id": "wrong-thread",
                    "memory_id": "memory-1",
                    "user_id": "alice",
                    "thread_id": "thread-b",
                    "namespace": "agents",
                },
                {
                    "_id": "wrong-namespace",
                    "memory_id": "memory-1",
                    "user_id": "alice",
                    "thread_id": "thread-a",
                    "content": {"namespace": "finance"},
                },
                {
                    "_id": "wrong-user",
                    "memory_id": "memory-1",
                    "user_id": "bob",
                    "thread_id": "thread-a",
                    "namespace": "agents",
                },
                {
                    "_id": "wrong-memory",
                    "memory_id": "memory-2",
                    "user_id": "alice",
                    "thread_id": "thread-a",
                    "namespace": "agents",
                },
            ]

    provider = BroadProvider()
    rows = MemoryManager(provider).retrieve_relevant_memories(
        query="what did we ingest?",
        memory_type=MemoryType.CONVERSATION_MEMORY,
        memory_id="memory-1",
        user_id="alice",
        thread_id="thread-a",
        namespace="agents",
    )

    assert [row["_id"] for row in rows] == ["right"]
    assert provider.calls[0]["thread_id"] == "thread-a"
    assert provider.calls[0]["namespace"] == "agents"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("memory_type", "scope_name", "scope_value"),
    [
        (MemoryType.CONVERSATION_MEMORY, "thread_id", "thread-a"),
        (MemoryType.KNOWLEDGE_BASE, "namespace", "agents"),
    ],
)
def test_filesystem_ranks_only_inside_explicit_retrieval_scope(
    tmp_path, memory_type, scope_name, scope_value
):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / memory_type.value,
            embedding_provider=_EmbeddingManager(),
        )
    )
    other_scope = "thread-b" if scope_name == "thread_id" else "finance"
    for index in range(6):
        provider.store(
            {
                "memory_id": "memory-1",
                "user_id": "alice",
                scope_name: other_scope,
                "content": f"distractor-{index}",
                "embedding": [1.0, 0.0],
            },
            memory_type,
        )
    wanted_id = provider.store(
        {
            "memory_id": "memory-1",
            "user_id": "alice",
            scope_name: scope_value,
            "content": "wanted",
            "embedding": [0.9, 0.1],
        },
        memory_type,
    )

    rows = provider.retrieve_by_query(
        "wanted",
        memory_type=memory_type,
        memory_id="memory-1",
        user_id="alice",
        limit=1,
        **{scope_name: scope_value},
    )

    assert [row["_id"] for row in rows] == [wanted_id]


@pytest.mark.unit
def test_summary_registry_and_expansion_are_exactly_thread_scoped():
    class Provider:
        def list_all(self, memory_type):
            assert memory_type == MemoryType.SUMMARIES
            return [
                {
                    "_id": "summary-a",
                    "memory_id": "memory-1",
                    "agent_id": "agent-1",
                    "user_id": "alice",
                    "thread_id": "thread-a",
                    "content": "the URL from thread A",
                    "period_end": 2,
                },
                {
                    "_id": "summary-b",
                    "memory_id": "memory-1",
                    "agent_id": "agent-1",
                    "user_id": "alice",
                    "thread_id": "thread-b",
                    "content": "unrelated uploaded content",
                    "period_end": 3,
                },
            ]

        def retrieve_by_id(self, summary_id, memory_type):
            return next(
                row for row in self.list_all(memory_type) if row["_id"] == summary_id
            )

    provider = Provider()
    manager = MemoryManager(provider)
    rows = manager.load_summaries_for_thread(
        "memory-1", agent_id="agent-1", user_id="alice", thread_id="thread-a"
    )
    assert [row["summary_id"] for row in rows] == ["summary-a"]
    assert rows[0]["thread_id"] == "thread-a"

    agent = MemAgent()
    agent.memory_provider = provider
    agent.memory_manager = manager
    assert agent.fetch_context_summary(
        "summary-a",
        memory_id="memory-1",
        user_id="alice",
        thread_id="thread-a",
    )
    assert (
        agent.fetch_context_summary(
            "summary-b",
            memory_id="memory-1",
            user_id="alice",
            thread_id="thread-a",
        )
        is None
    )


@pytest.mark.unit
def test_shared_agent_router_callbacks_and_cache_scope_are_context_local():
    agent = MemAgent()
    barrier = threading.Barrier(2)

    def agent_worker(suffix: str):
        events = []
        memory_id = f"memory-{suffix}"
        thread_id = f"thread-{suffix}"
        user_id = f"user-{suffix}"
        agent.set_stream_event_callback(events.append)
        agent.semantic_tool_router.begin_turn(user_id=user_id)
        agent.semantic_tool_router._selected.add(f"tool-{suffix}")
        agent._resolve_execution_state(memory_id, thread_id)
        barrier.wait(timeout=5)
        agent._emit_stream_event("trace", {"marker": suffix})
        return {
            "memory_id": agent._current_memory_id,
            "thread_id": agent._current_thread_id,
            "router": agent.semantic_tool_router.checkpoint_state(),
            "events": events,
        }

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(agent_worker, ("a", "b")))

    by_thread = {result["thread_id"]: result for result in results}
    assert by_thread["thread-a"]["memory_id"] == "memory-a"
    assert by_thread["thread-a"]["router"]["selected"] == ["tool-a"]
    assert by_thread["thread-a"]["router"]["user_id"] == "user-a"
    assert [event["marker"] for event in by_thread["thread-a"]["events"]] == ["a"]
    assert by_thread["thread-b"]["router"]["selected"] == ["tool-b"]
    assert [event["marker"] for event in by_thread["thread-b"]["events"]] == ["b"]

    cache = SemanticCache(
        config=SemanticCacheConfig(
            similarity_threshold=0.9,
            enable_memory_provider_sync=False,
        ),
        embedding_manager=_EmbeddingManager(),
        agent_id="shared-agent",
    )
    cache_barrier = threading.Barrier(2)

    def cache_worker(suffix: str):
        cache.memory_id = f"memory-{suffix}"
        assert cache.set(
            "same question",
            f"answer-{suffix}",
            session_id="same-session",
            user_id="alice",
            metadata={"fingerprints": {"request_context": "same"}},
        )
        cache_barrier.wait(timeout=5)
        return cache.memory_id, cache.get(
            "same question",
            session_id="same-session",
            user_id="alice",
            lookup_metadata={"fingerprints": {"request_context": "same"}},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        cache_results = dict(executor.map(cache_worker, ("a", "b")))
    assert cache_results == {
        "memory-a": "answer-a",
        "memory-b": "answer-b",
    }


@pytest.mark.unit
def test_routed_trace_exposes_model_and_logical_tool_names():
    def summarize_article(url: str) -> dict:
        return {"url": url, "summary": "grounded"}

    agent = MemAgent(
        tools=[summarize_article],
        context_policy=ContextPolicy(progressive_tool_disclosure=True),
    )
    events = []
    agent.set_stream_event_callback(events.append)
    agent.semantic_tool_router.begin_turn(user_id="alice")
    agent.semantic_tool_router._selected.add("summarize_article")

    messages = []
    agent._execute_and_record_tool_call(
        _tool_call(
            "invoke_tool",
            {
                "tool_name": "summarize_article",
                "arguments": {"url": "https://openai.com/example"},
            },
        ),
        messages,
        workflow=None,
        user_id="alice",
        streaming=True,
    )

    traces = [event for event in events if event.get("type") == "trace"]
    assert [trace["trace_kind"] for trace in traces] == [
        "tool_call",
        "tool_result",
    ]
    for trace in traces:
        assert trace["tool_name"] == "summarize_article"
        assert trace["logical_tool_name"] == "summarize_article"
        assert trace["model_tool_name"] == "invoke_tool"
    assert traces[-1]["success"] is True
    assert traces[-1]["duration_ms"] >= 0

    persisted = agent._build_trace_bundle_events(traces)
    persisted_result = persisted[-1]
    assert persisted_result["tool_name"] == "summarize_article"
    assert persisted_result["logical_tool_name"] == "summarize_article"
    assert persisted_result["model_tool_name"] == "invoke_tool"
    assert persisted_result["success"] is True
    assert persisted_result["duration_ms"] >= 0
    assert json.loads(messages[-1]["content"])["tool_name"] == "summarize_article"


@pytest.mark.unit
def test_semantic_cache_defaults_to_session_and_fingerprints_request_context():
    assert SemanticCacheConfig().enable_session_scoping is True
    builder = MemAgentBuilder().with_semantic_cache()
    assert builder._semantic_cache_config["scope"] == "session"

    agent = MemAgent()
    first = agent._semantic_cache_metadata(
        {"current_page": {"id": "article-a"}, "quoted_text": "alpha"}
    )
    second = agent._semantic_cache_metadata(
        {"current_page": {"id": "article-b"}, "quoted_text": "beta"}
    )
    assert "request_context" in first["fingerprints"]
    assert (
        first["fingerprints"]["request_context"]
        != second["fingerprints"]["request_context"]
    )


@pytest.mark.unit
def test_mcp_dependencies_are_optional_and_capabilities_are_explicit():
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
        import tomli as tomllib

    project_root = Path(__file__).resolve().parents[2]
    with (project_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    base = "\n".join(project["dependencies"])
    assert "mcp>=" not in base
    assert "cryptography>=" not in base
    assert "uvicorn>=" not in base
    mcp_extra = "\n".join(project["optional-dependencies"]["mcp"])
    assert "mcp>=2.0.0" in mcp_extra
    assert "cryptography>=42.0.0" in mcp_extra
    assert "uvicorn>=0.24.0" in mcp_extra

    report = capabilities()
    for feature in (
        "thread_scoped_summaries",
        "retrieval_policy",
        "concurrency_safe_run_state",
        "logical_tool_trace_names",
        "semantic_cache_session_default",
        "entity_memory",
        "personalization_context",
        "memory_supply_observability",
        "canonical_entity_identity",
        "host_completion_policy",
        "evaluation_suite",
    ):
        assert report["features"][feature]["available"] is True
    assert report["features"]["mcp_client"]["install_extra"] == "mcp"
    assert report["features"]["entity_memory"]["strict_tenant_scope"] is True
    assert report["features"]["entity_memory"]["control_plane_parity"] is True


@pytest.mark.unit
def test_credentials_module_imports_without_cryptography(monkeypatch, tmp_path):
    module_name = "memorizz.mcp.credentials"
    import memorizz.mcp as parent

    original = sys.modules.pop(module_name, None)
    original_attr = getattr(parent, "credentials", None)
    monkeypatch.setitem(sys.modules, "cryptography", None)
    monkeypatch.setitem(sys.modules, "cryptography.fernet", None)
    try:
        module = importlib.import_module(module_name)
        assert module.Fernet is None
        store = module.EncryptedFileCredentialStore(
            path=tmp_path / "credentials.enc",
            key_path=tmp_path / "credentials.key",
        )
        assert store.get("missing") == {}
        with pytest.raises(module.MCPConfigurationError, match=r"memorizz\[mcp\]"):
            store.set("server", {"token": "secret"})
    finally:
        if original is not None:
            sys.modules[module_name] = original
        else:
            sys.modules.pop(module_name, None)
        setattr(parent, "credentials", original_attr)


@pytest.mark.unit
def test_node_package_bridge_installs_the_complete_mcp_cli():
    project_root = Path(__file__).resolve().parents[2]
    postinstall = (
        project_root / "packaging" / "npm" / "scripts" / "postinstall.js"
    ).read_text(encoding="utf-8")
    launcher = (project_root / "packaging" / "npm" / "bin" / "memorizz.js").read_text(
        encoding="utf-8"
    )

    assert "memorizz[mcp]==${VERSION}" in postinstall
    assert "memorizz[mcp]==${VERSION}" in launcher
    assert '"--from", PACKAGE_SPEC, "memorizz"' in launcher
