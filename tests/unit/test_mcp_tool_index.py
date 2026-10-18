"""MCP tools kept in toolbox memory and found by meaning."""

import re
import threading
from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.toolbox.indexing import argument_names, search_text
from memorizz.mcp.tool_index import MCPToolIndex, catalog_row_id, tool_row_id
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

# One dimension per meaning; a word counts toward each meaning it starts with
# one of the meaning's words ("look" and "find" mean search).
MEANINGS = [
    ("search", "look", "find"),
    ("page",),
    ("calendar",),
    ("event",),
    ("comment",),
    ("remind",),
    ("web",),
]


class WordEmbedder:
    """A tiny meaning model: texts about the same things point the same way."""

    def __init__(self):
        self.model = "words-v1"
        self.batches = []

    def get_default_model(self):
        return self.model

    def _vector(self, text):
        words = re.findall(r"[a-z]+", text.lower())
        return [
            sum(word.startswith(meaning) for word in words) + 0.01
            for meaning in MEANINGS
        ]

    def get_embedding(self, text, **kwargs):
        return self._vector(text)

    def get_embeddings(self, texts):
        self.batches.append(list(texts))
        return [self._vector(text) for text in texts]


SEARCH = {
    "name": "notion-search",
    "description": "Search the workspace for pages.",
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    "annotations": {"readOnlyHint": True},
}
CALENDAR = {
    "name": "notion-calendar-list-events",
    "description": "List calendar events.",
    "inputSchema": {"type": "object", "properties": {"start": {"type": "string"}}},
}
COMMENT = {
    "name": "notion-create-comment",
    "description": "Add a comment to a page.",
    "inputSchema": {"type": "object", "properties": {"page_id": {"type": "string"}}},
}


def _public(server, name):
    return f"{server}__{name}"


def _provider(tmp_path, embedder, use_faiss=True):
    return FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "memory",
            embedding_provider=embedder,
            lazy_vector_indexes=True,
            use_faiss=use_faiss,
        )
    )


def _row(provider, row_id):
    return provider.retrieve_by_id(row_id, memory_store_type=MemoryType.TOOLBOX)


def test_tools_are_embedded_once_and_only_changes_are_redone(tmp_path):
    embedder = WordEmbedder()
    provider = _provider(tmp_path, embedder)
    index = MCPToolIndex(provider, "agent-1")

    assert index.sync("notion", [SEARCH, CALENDAR], _public) == {
        "stored": 2,
        "removed": 0,
    }
    row = _row(provider, tool_row_id("agent-1", "notion", "notion-search"))
    assert row["name"] == "notion__notion-search"
    assert row["tool_type"] == "mcp_tool" and row["embedding"]
    assert "Arguments: query." in search_text(row)
    assert len(embedder.batches) == 1

    # An unchanged listing is one read and no writes or embeddings.
    assert index.sync("notion", [SEARCH, CALENDAR], _public) == {
        "stored": 0,
        "removed": 0,
    }
    assert len(embedder.batches) == 1

    changed = dict(SEARCH, description="Search pages and databases.")
    assert index.sync("notion", [changed], _public) == {"stored": 1, "removed": 1}
    assert embedder.batches[-1] == [search_text(_row(provider, row["_id"]))]
    assert _row(provider, tool_row_id("agent-1", "notion", CALENDAR["name"])) is None

    # A different embedding model makes every stored vector stale.
    embedder.model = "words-v2"
    assert index.sync("notion", [changed], _public)["stored"] == 1

    assert index.remove_server("notion") == 1
    assert _row(provider, row["_id"]) is None
    assert _row(provider, catalog_row_id("agent-1", "notion")) is None


def test_without_an_embedder_tools_are_stored_and_embedded_later(tmp_path):
    provider = _provider(tmp_path, None)
    provider._get_embedding_provider = lambda: None
    index = MCPToolIndex(provider, "agent-1")
    assert index.sync("notion", [SEARCH], _public)["stored"] == 1
    row_id = tool_row_id("agent-1", "notion", "notion-search")
    assert not _row(provider, row_id).get("embedding")

    embedder = WordEmbedder()
    provider._get_embedding_provider = lambda: embedder
    assert index.sync("notion", [SEARCH], _public)["stored"] == 1
    assert _row(provider, row_id)["embedding"]


def test_saving_an_agent_keeps_its_mcp_index_and_unchanged_vectors(tmp_path):
    embedder = WordEmbedder()
    provider = _provider(tmp_path, embedder)
    MCPToolIndex(provider, "agent-1").sync("notion", [SEARCH], _public)
    tool = {"name": "web_search", "description": "Search the web.", "parameters": {}}
    provider._sync_agent_tools_to_toolbox("agent-1", [tool])
    provider.update_by_id(
        "agent-1:web_search", {"embedding": [1.0] * 7}, MemoryType.TOOLBOX
    )

    # The save that mirrors the agent's own tools rewrites only those rows.
    provider._sync_agent_tools_to_toolbox("agent-1", [tool])
    assert _row(provider, tool_row_id("agent-1", "notion", "notion-search"))
    assert _row(provider, catalog_row_id("agent-1", "notion"))
    assert _row(provider, "agent-1:web_search")["embedding"] == [1.0] * 7

    provider._sync_agent_tools_to_toolbox(
        "agent-1", [dict(tool, description="Search the public web.")]
    )
    assert not _row(provider, "agent-1:web_search").get("embedding")


# Without FAISS (the base install) ranking falls back to brute-force cosine,
# which must rank per agent too: every agent with the server has the same rows.
@pytest.mark.parametrize("use_faiss", [True, False], ids=["faiss", "brute-force"])
def test_tool_rows_are_embedded_on_first_search_and_ranked_per_agent(
    tmp_path, use_faiss
):
    embedder = WordEmbedder()
    provider = _provider(tmp_path, embedder, use_faiss=use_faiss)
    MCPToolIndex(provider, "agent-1").sync("notion", [SEARCH, CALENDAR], _public)
    MCPToolIndex(provider, "agent-2").sync("notion", [SEARCH, CALENDAR], _public)
    provider._sync_agent_tools_to_toolbox(
        "agent-1", [{"name": "remind_me", "description": "Remind me later."}]
    )
    provider.store(
        {
            "_id": "agent-1:mcp:notion",
            "name": "mcp::notion",
            "tool_type": "mcp_server_config",
            "agent_id": "agent-1",
        },
        memory_store_type=MemoryType.TOOLBOX,
    )

    rows = provider.retrieve_by_query(
        "remind me about calendar events",
        memory_store_type=MemoryType.TOOLBOX,
        limit=2,
        user_id=None,
        agent_id="agent-1",
    )
    assert [row["name"] for row in rows] == [
        "notion__notion-calendar-list-events",
        "remind_me",
    ]
    assert {row["agent_id"] for row in rows} == {"agent-1"}
    # The agent's own tool was embedded for the search; the config row is not
    # a tool and stays unembedded.
    assert _row(provider, "agent-1:remind_me")["embedding"]
    assert not _row(provider, "agent-1:mcp:notion").get("embedding")


def test_keyword_fallback_ranks_tools_per_agent(tmp_path):
    provider = _provider(tmp_path, None)
    for agent in ("agent-1", "agent-2"):
        provider._sync_agent_tools_to_toolbox(
            agent, [{"name": "calendar_events", "description": "List events."}]
        )

    rows = provider.retrieve_by_query(
        "calendar_events",
        memory_store_type=MemoryType.TOOLBOX,
        limit=5,
        user_id=None,
        agent_id="agent-2",
    )
    assert [row["agent_id"] for row in rows] == ["agent-2"]


def test_argument_names_read_every_parameter_shape():
    assert argument_names({"type": "object", "properties": {"b": {}, "a": {}}}) == [
        "a",
        "b",
    ]
    assert argument_names({"b": {}, "a": {}}) == ["a", "b"]
    assert argument_names([{"name": "q"}, {"type": "x"}]) == ["q"]
    assert argument_names(None) == []


def _agent(tmp_path, monkeypatch, *, memory_types=None):
    from memorizz import MemAgent
    from memorizz.mcp import MCPClientManager

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    server = {
        "name": "notion",
        "transport": "streamable_http",
        "url": "https://mcp.notion.com/mcp",
        "auth": {"type": "none"},
    }
    owner = MCPClientManager(owner_id="agent-n", servers=[server])
    owner._remember_tool_listing(
        owner.get_server("notion"), [SEARCH, CALENDAR, COMMENT]
    )
    kwargs = {} if memory_types is None else {"memory_types": memory_types}
    agent = MemAgent(
        agent_id="agent-n",
        memory_provider=_provider(tmp_path, WordEmbedder()),
        auto_register=False,
        mcp_servers=[server],
        **kwargs,
    )
    for thread in threading.enumerate():
        if thread.name.startswith("mcp-tool-index-"):
            thread.join()
    return agent


def test_an_agent_indexes_its_mcp_tools_and_routes_to_them_by_meaning(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path, monkeypatch)
    try:
        provider = agent.memory_provider
        assert _row(provider, tool_row_id("agent-n", "notion", "notion-search"))
        router = agent.semantic_tool_router
        router.begin_turn()
        # No word of the request appears in notion-search's name or
        # description; only its meaning matches.
        assert router.preview("look up my reading notes", limit=1) == [
            "notion__notion-search"
        ]
        # Runs that act through MCP tools are kept as workflow memory.
        assert MemoryType.WORKFLOW_MEMORY in agent.active_memory_types

        # An empty cache means "not listed yet": nothing is removed.
        agent._mcp_index_in_background = False
        agent._index_mcp_tools("notion", [])
        assert _row(provider, catalog_row_id("agent-n", "notion"))
        # Disconnecting the server removes its rows.
        agent.with_mcp_servers([])
        for thread in threading.enumerate():
            if thread.name.startswith("mcp-tool-index-"):
                thread.join()
        assert _row(provider, catalog_row_id("agent-n", "notion")) is None
    finally:
        agent.close()


def test_workflow_memory_turned_on_for_mcp_is_not_saved_as_configuration(
    tmp_path, monkeypatch
):
    """Saved, it stayed on after the servers were removed."""
    agent = _agent(tmp_path, monkeypatch, memory_types=["conversation_memory"])
    try:
        assert MemoryType.WORKFLOW_MEMORY in agent.active_memory_types
        agent.save()
        saved = agent.memory_provider.retrieve_memagent(agent.agent_id)
        assert saved.memory_types == ["conversation_memory"]
    finally:
        agent.close()


def test_a_stateless_agent_does_not_gain_workflow_memory(tmp_path, monkeypatch):
    agent = _agent(tmp_path, monkeypatch, memory_types=[])
    try:
        assert agent.active_memory_types == []
    finally:
        agent.close()


def test_a_stateless_agent_reloads_stateless(tmp_path, monkeypatch):
    """Saved as "not configured", it came back with the default types."""
    from memorizz import MemAgent

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = _provider(tmp_path, WordEmbedder())
    agent = MemAgent(
        agent_id="agent-s",
        memory_provider=provider,
        auto_register=False,
        memory_types=[],
    )
    try:
        agent.save()
        assert provider.retrieve_memagent("agent-s").memory_types == []
        reloaded = MemAgent.load("agent-s", memory_provider=provider)
        assert reloaded.active_memory_types == []
        reloaded.close()
    finally:
        agent.close()
