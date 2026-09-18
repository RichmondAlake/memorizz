import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from memorizz.cli.app import app
from memorizz.enums import MemoryType
from memorizz.memory_provider.notion import provision_notion_workspace
from tests.unit.test_notion_provider import stack  # noqa: F401


def test_workspace_provisioning_exposes_all_memory_types_and_linked_ui(stack):
    provider, _, api, _ = stack
    result = provision_notion_workspace(api.parent_id, client=provider._client)
    assert result["data_source_id"] == api.data_source_id
    assert len(result["view_ids"]) == len(MemoryType) + 4
    assert {
        view["filter"]["select"]["equals"]
        for view in api.views
        if "database_id" in view
    } == {value.value for value in MemoryType}
    assert len([view for view in api.views if "create_database" in view]) == 4


def test_cli_notion_init_status_sync_and_repair(stack, monkeypatch):
    import memorizz.cli.notion_commands as commands

    provider, _, api, _ = stack
    monkeypatch.setattr(commands.cfg, "load_layered_env", lambda: None)
    monkeypatch.setattr(commands, "_provider", lambda **_: provider)
    runner = CliRunner()
    for arguments in (
        ["notion", "status"],
        ["notion", "sync", "--memory-type", "knowledge_base"],
        ["notion", "repair"],
    ):
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert "test-notion-token" not in result.output
        assert json.loads(result.output)["pending"] == 0
    with patch(
        "memorizz.memory_provider.notion.provision_notion_workspace",
        return_value={"data_source_id": api.data_source_id},
    ) as create:
        result = runner.invoke(
            app, ["notion", "init", "--parent-page-id", api.parent_id, "--no-views"]
        )
        assert result.exit_code == 0, result.output
        assert create.call_args.kwargs["create_views"] is False
        assert "MEMORIZZ_BACKEND=notion" in result.output


def test_cli_factory_does_not_fall_back_to_filesystem_for_notion(monkeypatch):
    from memorizz.cli import agent_factory

    monkeypatch.setenv("MEMORIZZ_BACKEND", "notion")
    monkeypatch.setattr(
        agent_factory,
        "_choose_embedding",
        lambda _: {
            "embedding_provider": "ollama",
            "embedding_config": {"model": "embed"},
        },
    )
    sentinel = object()
    with patch(
        "memorizz.memory_provider.notion.factory.create_notion_provider_from_env",
        return_value=sentinel,
    ) as create:
        assert agent_factory.detect_memory_provider({}, []) is sentinel
        assert create.call_args.kwargs["embedding_provider"] == "ollama"


def test_ui_notion_connect_renders_no_token_and_accepts_separate_vectors(
    stack, monkeypatch
):
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app
    from memorizz.ui.state import _state

    provider = stack[0]
    monkeypatch.setenv("NOTION_TOKEN", "private-notion-ui-token")
    monkeypatch.setenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", provider.config.data_source_id)
    with patch.dict(
        _state,
        {
            "provider": None,
            "provider_type": None,
            "connection_info": {},
            "provider_secrets": {},
        },
    ):
        client = TestClient(create_app(), follow_redirects=False)
        response = client.get("/connect")
        assert response.status_code == 200
        assert "private-notion-ui-token" not in response.text
        assert "Notion + separate semantic provider" in response.text
        assert "Configured via NOTION_TOKEN" in response.text
        with patch(
            "memorizz.memory_provider.notion.factory.create_notion_provider_from_env",
            return_value=provider,
        ) as create:
            response = client.post(
                "/connect",
                data={"provider_type": "notion", "notion_semantic_backend": "mongodb"},
            )
        assert response.status_code == 302
        assert create.call_args.kwargs["semantic_backend"] == "mongodb"
        assert _state["provider"] is provider
        assert "private-notion-ui-token" not in json.dumps(_state["connection_info"])
        calls = len(stack[2].calls)
        dashboard = client.get("/dashboard")
        assert dashboard.status_code == 200
        assert "local repair journal" in dashboard.text
        assert len(stack[2].calls) == calls, "Dashboard counts must not scan Notion"
        assert client.get("/memory/knowledge-base").status_code == 200
        usage = client.get("/traces/usage.json")
        assert usage.status_code == 200


def test_agent_config_secrets_are_not_persisted_to_notion(stack):
    provider, _, api, _ = stack
    identifier = provider.store_memagent(
        {
            "agent_id": "test-agent",
            "instruction": "test",
            "llm_config": {
                "provider": "openai",
                "model": "example",
                "api_key": "never-store-this",
            },
            "mcp_servers": [
                {
                    "name": "example",
                    "headers": {"Authorization": "Bearer never-store-this"},
                }
            ],
        }
    )
    assert "never-store-this" not in json.dumps(api.pages)
    row = provider.retrieve_by_id(identifier, MemoryType.MEMAGENT)
    assert row["llm_config"]["model"] == "example"
    assert "api_key" not in row["llm_config"]


@pytest.mark.parametrize("backend", ["filesystem", "mongodb", "oracle", "none"])
def test_env_factory_constructs_selected_vector_backend_without_other_connections(
    stack, monkeypatch, backend
):
    from memorizz.memory_provider.notion.factory import create_notion_provider_from_env

    provider = stack[0]
    monkeypatch.setenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", provider.config.data_source_id)
    monkeypatch.setenv("NOTION_TOKEN", "test-only")
    monkeypatch.setenv("MEMORIZZ_NOTION_MONGODB_URI", "mongodb://fixture.invalid")
    with patch(
        "memorizz.memory_provider.filesystem.FileSystemProvider"
    ) as filesystem, patch(
        "memorizz.memory_provider.mongodb.MongoDBProvider"
    ) as mongo, patch(
        "memorizz.memory_provider.oracle.OracleProvider"
    ) as oracle, patch(
        "memorizz.memory_provider.notion.factory.NotionProvider"
    ) as notion:
        result = create_notion_provider_from_env(
            semantic_backend=backend,
            embedding_provider="ollama",
            embedding_config={"model": "test"},
        )
        assert filesystem.called is (backend == "filesystem")
        assert mongo.called is (backend == "mongodb")
        assert oracle.from_env.called is (backend == "oracle")
        assert (notion.call_args.kwargs["semantic_provider"] is None) is (
            backend == "none"
        )
        assert result._owns_semantic_provider is (backend != "none")


def test_env_factory_closes_owned_vectors_after_failed_notion_connection(
    stack, monkeypatch
):
    from memorizz.memory_provider.notion.factory import create_notion_provider_from_env

    monkeypatch.setenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", stack[0].config.data_source_id)
    monkeypatch.setenv("NOTION_TOKEN", "test")
    with patch(
        "memorizz.memory_provider.filesystem.FileSystemProvider"
    ) as filesystem, patch(
        "memorizz.memory_provider.notion.factory.NotionProvider",
        side_effect=RuntimeError("connection failed"),
    ):
        with pytest.raises(RuntimeError):
            create_notion_provider_from_env(semantic_backend="filesystem")
        filesystem.return_value.close.assert_called_once()


def test_env_factory_invalid_backend_does_not_fall_back(stack, monkeypatch):
    from memorizz.memory_provider.notion.factory import create_notion_provider_from_env

    monkeypatch.setenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", stack[0].config.data_source_id)
    monkeypatch.setenv("NOTION_TOKEN", "test")
    with pytest.raises(ValueError, match="semantic backend"):
        create_notion_provider_from_env(semantic_backend="typo")


def test_semantic_cache_uses_notion_and_never_serves_revoked_local_hit(
    stack, monkeypatch
):
    from memorizz.short_term_memory.semantic_cache import (
        SemanticCache,
        SemanticCacheConfig,
    )

    provider, _, api, embedder = stack
    monkeypatch.setattr(
        "memorizz.short_term_memory.semantic_cache.get_embedding_manager",
        MagicMock(side_effect=AssertionError("must not initialize global embeddings")),
    )
    cache = SemanticCache(
        SemanticCacheConfig(similarity_threshold=0.8, enable_usage_tracking=False),
        memory_provider=provider,
        agent_id="agent",
        memory_id="memory",
    )
    assert cache.set("espresso", "Coffee answer", session_id="session", user_id="alice")
    assert (
        len(embedder.calls) == 1
    ), "Only the semantic provider embeds the stored query"
    assert (
        cache.get("espresso", session_id="session", user_id="alice") == "Coffee answer"
    )
    assert cache.get("espresso", session_id="session", user_id="bob") is None
    record = provider.list_all(MemoryType.SEMANTIC_CACHE)[0]
    api.hidden.add(
        provider._state.get(record["id"], MemoryType.SEMANTIC_CACHE.value)["page_id"]
    )
    assert cache.get("espresso", session_id="session", user_id="alice") is None
    assert not cache.inspect("espresso", session_id="session", user_id="alice").hit


def test_mcp_runtime_notion_memory_roundtrip_is_tenant_scoped(stack, tmp_path):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server import MemorizzMCPServerConfig, MemorizzRuntime
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import ALL_SCOPES
    from memorizz.mcp_server.runtime import MemorizzServerError

    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(),
        provider=stack[0],
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite"),
    )
    alice = RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )
    bob = RequestIdentity(
        principal="bob", scopes=frozenset(ALL_SCOPES), authenticated=True
    )
    saved = runtime.remember("coffee", "knowledge_base", alice, memory_id="memory")
    assert (
        runtime.get_memory(saved["record_id"], "knowledge_base", alice)["memory"][
            "content"
        ]
        == "coffee"
    )
    assert runtime.search_memories("espresso", "knowledge_base", alice)["count"] == 1
    assert runtime.search_memories("espresso", "knowledge_base", bob)["count"] == 0
    with pytest.raises(MemorizzServerError, match="not found"):
        runtime.get_memory(saved["record_id"], "knowledge_base", bob)


def test_ui_bounded_memory_preview_and_errors_are_not_empty_confirmations(
    stack, monkeypatch
):
    from fastapi.testclient import TestClient

    from memorizz.memory_provider.notion import NotionError
    from memorizz.ui.app import create_app
    from memorizz.ui.state import _state

    provider = stack[0]
    for index in range(102):
        provider.store(
            {"id": str(index), "content": "coffee"}, MemoryType.KNOWLEDGE_BASE
        )
    with patch.dict(
        _state, {"provider": provider, "provider_type": "notion", "connection_info": {}}
    ):
        client = TestClient(create_app())
        response = client.get("/memory/knowledge-base")
        assert response.status_code == 200
        assert "100 items shown" in response.text
        assert "more records exist" in response.text
        monkeypatch.setattr(
            provider,
            "retrieve_by_query",
            MagicMock(side_effect=NotionError("private failure context")),
        )
        response = client.get("/memory/knowledge-base")
        assert "Memory could not be loaded" in response.text
        assert "private failure context" not in response.text
        assert "There are no items" not in response.text


def test_factory_embedding_environment_is_consistent_for_ui_and_cli(stack, monkeypatch):
    from memorizz.memory_provider.notion.factory import create_notion_provider_from_env

    monkeypatch.setenv("MEMORIZZ_NOTION_DATA_SOURCE_ID", stack[0].config.data_source_id)
    monkeypatch.setenv("NOTION_TOKEN", "test")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "selected-model")
    with patch(
        "memorizz.memory_provider.filesystem.FileSystemProvider"
    ) as vectors, patch("memorizz.memory_provider.notion.factory.NotionProvider"):
        create_notion_provider_from_env(semantic_backend="filesystem")
        config = vectors.call_args.args[0]
        assert config.embedding_provider == "ollama"
        assert config.embedding_config["model"] == "selected-model"


def test_semantic_chunking_and_entity_ingestion_use_only_selected_embedder(
    stack, monkeypatch
):
    from memorizz.long_term.semantic.entity_memory.entity_memory import EntityMemory
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase

    provider = stack[0]
    monkeypatch.setattr(
        "memorizz.long_term.semantic.knowledge_base.get_embedding",
        MagicMock(side_effect=AssertionError("global embedding must not be used")),
    )
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        MagicMock(side_effect=AssertionError("global embedding must not be used")),
    )
    knowledge = KnowledgeBase(provider)
    group = knowledge.ingest_knowledge(
        "Coffee is hot. Tea comes with lemon. Espresso is strong.",
        "test",
        chunking_strategy="semantic",
        user_id="alice",
    )
    assert knowledge.retrieve_knowledge(group)
    entities = EntityMemory(provider)
    identifier = entities.upsert_entity(
        name="Coffee",
        entity_type="drink",
        attributes=[{"name": "temperature", "value": "hot"}],
        memory_id="memory",
        user_id="alice",
    )
    assert identifier
    assert provider.retrieve_by_query(
        "espresso", MemoryType.ENTITY_MEMORY, user_id="alice"
    )


def test_workflow_rehydration_from_notion_does_not_generate_global_embedding(
    stack, monkeypatch
):
    from memorizz.long_term.procedural.workflow.workflow import Workflow

    provider = stack[0]
    monkeypatch.setattr(
        "memorizz.long_term.procedural.workflow.workflow.get_embedding",
        MagicMock(side_effect=AssertionError("rehydration must not embed")),
    )
    identifier = provider.store(
        {"name": "Coffee", "description": "Order espresso", "steps": {}},
        MemoryType.WORKFLOW_MEMORY,
    )
    workflow = Workflow.from_dict(
        provider.retrieve_by_id(identifier, MemoryType.WORKFLOW_MEMORY)
    )
    assert workflow.name == "Coffee"
    assert workflow.embedding is None
