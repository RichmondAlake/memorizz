"""Real MemAgent + Notion REST transport + durable filesystem vector index."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider import (
    FileSystemConfig,
    FileSystemProvider,
    NotionConfig,
    NotionProvider,
)
from memorizz.memory_provider.notion import NotionClient, provision_notion_workspace
from tests.fixtures.notion_api import NotionAPI
from tests.unit.test_notion_provider import Embedder


class DeterministicModel:
    model = "gpt-4o-mini"

    def __init__(self):
        self.messages = []

    def get_config(self):
        return {"provider": "openai", "model": self.model}

    def generate(self, messages, tools=None, **kwargs):
        self.messages = messages
        text = json.dumps(messages, default=str)
        answer = (
            "Tea with lemon." if "tea with lemon" in text else "Coffee with oat milk."
        )
        return {"role": "assistant", "content": answer}

    def generate_stream(self, messages, tools=None, **kwargs):
        answer = self.generate(messages, tools)["content"]
        for word in answer.split(" "):
            yield {"type": "content", "content": word + " "}
        yield {"type": "done", "content": answer}

    def get_last_usage(self):
        return {
            "prompt_tokens": 24,
            "completion_tokens": 6,
            "total_tokens": 30,
            "model": self.model,
            "provider": "openai",
        }

    def get_context_window_tokens(self):
        return 8192


def provider_for(tmp_path, api, client):
    vectors = FileSystemProvider(
        FileSystemConfig(
            tmp_path / "vectors", embedding_provider=Embedder(), use_faiss=False
        )
    )
    return NotionProvider(
        NotionConfig(
            api.data_source_id,
            token="wire-test",
            state_path=tmp_path / "journal.sqlite",
        ),
        vectors,
        client=client,
    )


def test_real_agent_stream_memory_persistence_trace_and_human_edit(tmp_path):
    from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
    from memorizz.observability.usage_query import query_usage

    api = NotionAPI()
    client = NotionClient(
        "wire-test", session=api, sleep=lambda _: None, clock=lambda: 0
    )
    provider = provider_for(tmp_path, api, client)
    knowledge = KnowledgeBase(provider)
    group_id = knowledge.ingest_knowledge(
        "Alice prefers coffee with oat milk.", "preferences", user_id="alice"
    )
    rows = knowledge.retrieve_knowledge(group_id)
    assert len(rows) == 1
    memory_id = rows[0]["id"]
    # The legacy knowledge-base helper must use the selected semantic model.
    assert knowledge.retrieve_knowledge_by_query("espresso")[0]["id"] == memory_id
    # Pre-inference recall is intentionally limited to the interaction's memory
    # scope; attached document groups are also available through the KB tool.
    provider.update_by_id(
        memory_id, {"memory_id": "chat-memory"}, MemoryType.KNOWLEDGE_BASE
    )
    model = DeterministicModel()
    agent = MemAgent(
        model=model,
        memory_provider=provider,
        name="notion-e2e",
        instruction="Use the saved preferences.",
        memory_types=[MemoryType.KNOWLEDGE_BASE, MemoryType.CONVERSATION_MEMORY],
        automations_enabled=False,
        verbose=False,
    )
    # Attach the ingested group through the public knowledge API.
    knowledge.attach_to_agent(agent, group_id)
    chunks = list(
        agent.run_stream(
            "What espresso should I order?",
            memory_id="chat-memory",
            thread_id="thread",
            user_id="alice",
        )
    )
    assert "Coffee with oat milk." in "".join(chunks)
    assert (
        agent._embedding_backfill_executor is None
    ), "Notion already indexed conversations; legacy global-model backfill must not run"
    assert "Alice prefers coffee with oat milk." in json.dumps(
        model.messages, default=str
    )
    history = provider.retrieve_conversation_history_ordered_by_timestamp(
        "chat-memory", user_id="alice", thread_id="thread"
    )
    assert history, "The actual conversation persistence path must write Notion records"
    assert provider.retrieve_memagent(agent.agent_id) is not None
    traces = provider.query_observability_records(
        MemoryType.SHARED_MEMORY,
        record_type="observability_trace_bundle",
        user_id="alice",
    )
    assert traces["items"], "The actual recorder must persist a trace bundle"
    usage = query_usage(provider, filters={"user_id": "alice"})
    assert usage["totals"]["total_tokens"] == 30
    assert usage["coverage"]["read_complete"] is True
    page_id = provider._state.get(memory_id, MemoryType.KNOWLEDGE_BASE.value)["page_id"]
    api.edit(page_id, "content", "Alice prefers tea with lemon.")
    provider.sync_page(page_id)
    chunks = list(
        agent.run_stream(
            "What matcha should I order?",
            memory_id="chat-memory",
            thread_id="thread",
            user_id="alice",
        )
    )
    assert "Tea with lemon." in "".join(chunks)
    restarted = NotionProvider(
        provider.config, provider.semantic_provider, client=client
    )
    assert (
        restarted.retrieve_by_id(memory_id, MemoryType.KNOWLEDGE_BASE)["content"]
        == "Alice prefers tea with lemon."
    )
    assert restarted.retrieve_conversation_history_ordered_by_timestamp(
        "chat-memory", user_id="alice", thread_id="thread"
    )
    assert (
        restarted.retrieve_conversation_history_ordered_by_timestamp(
            "chat-memory", user_id="bob", thread_id="thread"
        )
        == []
    )
    assert restarted.index_status()["pending"] == 0, restarted._state.rows(
        pending_only=True
    )
    agent.close()


def test_wire_level_http_provision_store_search_update_delete(tmp_path):
    api = NotionAPI()

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self):
            parsed = urlparse(self.path)
            payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(payload) if payload else None
            params = {key: values[0] for key, values in parse_qs(parsed.query).items()}
            assert self.headers["Notion-Version"] == "2026-03-11"
            result = api.handle(
                self.command, parsed.path.removeprefix("/v1"), body, params
            )
            encoded = json.dumps(result.json()).encode()
            self.send_response(result.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        do_GET = do_POST = do_PATCH = handle_request

        def log_message(self, *_):
            pass

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("Loopback sockets need local execution permission")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = NotionClient(
        "wire-test",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        sleep=lambda _: None,
        clock=lambda: 0,
    )
    try:
        workspace = provision_notion_workspace(api.parent_id, client=client)
        assert workspace["data_source_id"] == api.data_source_id
        assert len(workspace["view_ids"]) == len(MemoryType) + 4
        provider = provider_for(tmp_path, api, client)
        identifier = provider.store(
            {"content": "coffee", "user_id": "alice"}, MemoryType.KNOWLEDGE_BASE
        )
        assert (
            provider.retrieve_by_query(
                "espresso", MemoryType.KNOWLEDGE_BASE, user_id="alice"
            )[0]["id"]
            == identifier
        )
        provider.update_by_id(identifier, {"content": "tea"}, MemoryType.KNOWLEDGE_BASE)
        assert (
            provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)[0][
                "content"
            ]
            == "tea"
        )
        assert provider.delete_by_id(identifier, MemoryType.KNOWLEDGE_BASE)
        assert not provider.retrieve_by_query("matcha", MemoryType.KNOWLEDGE_BASE)
        provider.close()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
