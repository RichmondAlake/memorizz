"""MCP tools exposed one by one, so models call them without a two-step facade."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from memorizz.mcp import MCPClientManager
from memorizz.memagent.managers.tool_manager import ToolManager
from memorizz.tooling import SemanticToolRouter

pytestmark = pytest.mark.unit

SERVER = Path(__file__).parents[1] / "fixtures" / "mcp_stdio_server.py"
F1_TOOLS = [
    {
        "name": "get_f1_next_event",
        "description": "Next Formula 1 race: date, circuit and sessions.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"timezone": {"type": "string"}},
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "get_f1_standings",
        "description": "Driver and constructor standings.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _stdio_server(**overrides):
    return {
        "name": "test",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER)],
        "timeout": 10,
        **overrides,
    }


def test_successful_listings_are_cached_per_server_address(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    manager = MCPClientManager(owner_id="agent-1", servers=[_stdio_server()])
    listed = manager.list_tools("test")
    assert listed["ok"], listed

    cached = manager.cached_tools("test")
    assert {tool["name"] for tool in cached} == {
        tool["name"] for tool in listed["tools"]
    }

    # A server at a different address does not inherit the old tool list.
    moved = MCPClientManager(
        owner_id="agent-1", servers=[_stdio_server(args=[str(SERVER), "--other"])]
    )
    assert moved.cached_tools("test") == []

    blocked_name = cached[0]["name"]
    blocked = MCPClientManager(
        owner_id="agent-1", servers=[_stdio_server(blocked_tools=[blocked_name])]
    )
    assert blocked_name not in {tool["name"] for tool in blocked.cached_tools("test")}


def _agent_with_cached_f1_tools(tmp_path, monkeypatch):
    from memorizz import MemAgent

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    server = {
        "name": "f1",
        "transport": "streamable_http",
        "url": "https://racecalendar.app/mcp",
        "auth": {"type": "none"},
    }
    manager = MCPClientManager(owner_id="agent-f1", servers=[server])
    manager._remember_tool_listing(manager.get_server("f1"), F1_TOOLS)
    return MemAgent(
        agent_id="agent-f1",
        memory_provider=False,
        auto_register=False,
        mcp_servers=[server],
    )


def test_agent_registers_each_cached_tool_and_routes_calls_through_policy(
    tmp_path, monkeypatch
):
    agent = _agent_with_cached_f1_tools(tmp_path, monkeypatch)
    try:
        names = set(agent.tool_manager.list_tools())
        assert {"f1__get_f1_next_event", "f1__get_f1_standings"} <= names
        assert "mcp_call_tool" in names  # the facade stays as a fallback

        metadata = agent.tool_manager.get_tool_metadata("f1__get_f1_next_event")
        assert metadata["source"] == "mcp"
        assert "$schema" not in metadata["input_schema"]
        assert "timezone" in metadata["input_schema"]["properties"]

        calls = []
        monkeypatch.setattr(
            agent.mcp_manager,
            "call_tool",
            lambda **kwargs: calls.append(kwargs) or {"ok": True},
        )
        result, _ = agent.tool_manager.execute_tool(
            "f1__get_f1_next_event", {"timezone": "UTC"}
        )
        assert result == {"ok": True}
        assert calls == [
            {
                "server_name": "f1",
                "tool_name": "get_f1_next_event",
                "arguments": {"timezone": "UTC"},
            }
        ]

        from memorizz.memagent import persistence

        saved = persistence._serialize_tools_for_save(agent) or []
        assert not any(tool["name"].startswith("f1__") for tool in saved)

        # Removing the servers removes their tools too.
        agent.with_mcp_servers([])
        assert not any(
            name.startswith("f1__") for name in agent.tool_manager.list_tools()
        )
    finally:
        agent.close()


def test_long_tool_names_stay_within_the_provider_limit():
    from memorizz import MemAgent

    name = MemAgent._mcp_tool_name("workspace-server", "x" * 80)
    assert len(name) <= 64
    assert name == MemAgent._mcp_tool_name("workspace-server", "x" * 80)
    assert MemAgent._mcp_tool_name("my server", "get.item") == "my_server__get_item"


class _SemanticToolbox:
    """A toolbox index that knows only the built-in tools."""

    def get_most_similar_tools(self, query, limit=25, **kwargs):
        return [{"name": "memory_lookup", "user_id": None, "agent_id": None}]


def test_router_surfaces_keyword_matched_mcp_tools_beside_semantic_results():
    manager = ToolManager()

    def memory_lookup(query: str) -> str:
        """Look up the user's memory."""
        return query

    manager.add_tool(memory_lookup)
    for tool in F1_TOOLS:
        manager.add_tool(
            {
                "name": f"f1__{tool['name']}",
                "metadata": {
                    "name": f"f1__{tool['name']}",
                    "description": tool["description"],
                    "input_schema": tool["inputSchema"],
                    "source": "mcp",
                },
                "function": lambda **kwargs: kwargs,
                "type": "function",
            }
        )

    router = SemanticToolRouter(manager, toolbox=_SemanticToolbox(), top_k=4)
    router.begin_turn()
    chosen = router.preview("when is the next Formula 1 race")
    assert chosen[0] == "memory_lookup"
    assert "f1__get_f1_next_event" in chosen
    # A single weak keyword is not enough to take a slot.
    assert router.preview("what do you remember about next week") == ["memory_lookup"]

    router.disclose(["f1__get_f1_standings", "not_registered"])
    names = {
        schema["function"]["name"] for schema in router.schemas_for_turn("standings")
    }
    assert "f1__get_f1_standings" in names
    assert "not_registered" not in names


NOTION_CREATE = {
    "name": "notion-create-pages",
    "description": "Create one or more Notion pages.",
    "inputSchema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "pages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "properties": {
                            "description": 'Outside a database the only allowed property is "title".',
                            "type": "object",
                            "additionalProperties": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "content": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "required": ["pages"],
        "additionalProperties": False,
    },
}
NOTION_SEARCH = {
    "name": "notion-search",
    "description": "Search the Notion workspace. " + "It can find it for you. " * 40,
    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
    "annotations": {"readOnlyHint": True},
}


def _notion_agent(tmp_path, monkeypatch):
    from memorizz import MemAgent

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    server = {
        "name": "notion",
        "transport": "streamable_http",
        "url": "https://mcp.notion.com/mcp",
        "auth": {"type": "none"},
    }
    manager = MCPClientManager(owner_id="agent-n", servers=[server])
    manager._remember_tool_listing(
        manager.get_server("notion"), [NOTION_CREATE, NOTION_SEARCH]
    )
    return MemAgent(
        agent_id="agent-n",
        memory_provider=False,
        auto_register=False,
        mcp_servers=[server],
    )


def test_arguments_are_fitted_to_the_schema_or_explained_before_the_server(
    tmp_path, monkeypatch
):
    agent = _notion_agent(tmp_path, monkeypatch)
    manager = agent.mcp_manager
    try:
        # A misplaced field moves where the schema documents it, and REST-style
        # wrapped text becomes the plain value. Nothing else changes.
        fixed, notes = manager.repair_arguments(
            "notion",
            "notion-create-pages",
            {
                "pages": [
                    {"title": "Harness Part 2", "content": "Notes"},
                    {"properties": {"title": {"title": {"text": {"content": "B"}}}}},
                ]
            },
        )
        assert fixed == {
            "pages": [
                {"properties": {"title": "Harness Part 2"}, "content": "Notes"},
                {"properties": {"title": "B"}},
            ]
        }
        assert len(notes) == 2
        # Ambiguous text is left alone and explained instead.
        mixed = {"pages": [{"properties": {"title": {"a": "One", "b": "Two"}}}]}
        assert manager.repair_arguments("notion", "notion-create-pages", mixed) == (
            mixed,
            [],
        )
        sent = []
        monkeypatch.setattr(
            manager,
            "_call_tool_authorized",
            lambda *a, **k: sent.append(a) or {"ok": True},
        )
        bad = manager.call_tool(
            server_name="notion", tool_name="notion-create-pages", arguments=mixed
        )
        assert bad["error_code"] == "invalid_arguments"
        assert "expected string or number or null, got object" in bad["error"]
        unknown = manager.call_tool(
            server_name="notion",
            tool_name="notion-create-pages",
            arguments={"pages": [{"emoji": "x"}]},
        )
        assert "'emoji' is not allowed here" in unknown["error"]
        assert sent == []

        # The agent fits arguments before asking anyone to approve the write.
        prepared, error, repairs = agent._prepare_mcp_call(
            "notion__notion-create-pages", {"pages": [{"title": "x"}]}
        )
        assert error is None and repairs
        assert prepared == {"pages": [{"properties": {"title": "x"}}]}
        facade, error, _ = agent._prepare_mcp_call(
            "mcp_call_tool",
            {
                "server_name": "notion",
                "tool_name": "notion-create-pages",
                "arguments": {"pages": [{"title": "x"}]},
            },
        )
        assert facade["arguments"] == {"pages": [{"properties": {"title": "x"}}]}
        _, error, _ = agent._prepare_mcp_call("notion__notion-create-pages", mixed)
        assert error["error_code"] == "invalid_arguments"
    finally:
        agent.close()


def test_first_class_mcp_writes_need_approval_like_the_facade(tmp_path, monkeypatch):
    agent = _notion_agent(tmp_path, monkeypatch)
    try:
        write = agent._effective_tool_policy(
            "notion__notion-create-pages", {"pages": []}
        )
        assert write["requires_approval"] is True
        assert "notion.notion-create-pages" in write["approval_reason"]
        # Read-only annotations come from the cached listing.
        read = agent._effective_tool_policy("notion__notion-search", {"query": "x"})
        assert read["requires_approval"] is False
        facade = agent._effective_tool_policy(
            "mcp_call_tool",
            {"server_name": "notion", "tool_name": "notion-create-pages"},
        )
        assert facade["requires_approval"] is True
    finally:
        agent.close()


def test_chat_filler_does_not_outrank_the_tool_the_request_names():
    manager = ToolManager()
    for tool in (NOTION_CREATE, NOTION_SEARCH):
        manager.add_tool(
            {
                "name": f"notion__{tool['name']}",
                "metadata": {
                    "name": f"notion__{tool['name']}",
                    "description": tool["description"],
                    "input_schema": tool["inputSchema"],
                    "source": "mcp",
                },
                "function": lambda **kwargs: kwargs,
                "type": "function",
            }
        )
    router = SemanticToolRouter(manager, top_k=1)
    router.begin_turn()
    query = "can you create a new page and give me the link to open it"
    # "page" matches "pages"; "can", "you", "it" and "me" no longer count.
    assert router.preview(query) == ["notion__notion-create-pages"]


def test_misnamed_or_undisclosed_tools_get_a_way_forward():
    from memorizz.tooling import ToolNotDisclosedError, UnknownToolError

    manager = ToolManager()

    def create_pages(title: str) -> str:
        """Create pages."""
        return title

    def search_pages(query: str) -> str:
        """Search pages."""
        return query

    manager.add_tool(create_pages)
    manager.add_tool(search_pages)
    router = SemanticToolRouter(manager, top_k=1)
    router.begin_turn()
    router.schemas_for_turn("search")

    with pytest.raises(UnknownToolError, match="Did you mean 'create_pages'") as typo:
        router.normalize_invocation("create_page", {"title": "x"})
    assert typo.value.code == "unknown_tool"
    assert '"title"' in str(typo.value)  # the schema comes with the suggestion
    assert router.normalize_invocation("create_pages", {"title": "x"})[0] == (
        "create_pages"
    )

    router.begin_turn()
    router.schemas_for_turn("search")
    with pytest.raises(ToolNotDisclosedError, match="not disclosed"):
        router.normalize_invocation("create_pages", {"title": "x"})
    # Named exactly, it is disclosed with its schema for the next call.
    assert router.normalize_invocation("create_pages", {"title": "x"})[0] == (
        "create_pages"
    )
    with pytest.raises(UnknownToolError, match="discover_tools"):
        router.normalize_invocation("teleport", {})


def test_small_tool_sets_are_sent_whole_so_the_cache_prefix_holds():
    manager = ToolManager()

    def weather(city: str) -> str:
        """Weather for a city."""
        return city

    def convert(amount: float) -> str:
        """Convert currency."""
        return str(amount)

    manager.add_tool(weather)
    manager.add_tool(convert)
    router = SemanticToolRouter(manager, top_k=1, stable_under_tokens=6000)
    router.begin_turn()
    first = router.schemas_for_turn("weather in Lisbon")
    router.begin_turn()
    second = router.schemas_for_turn("convert 250 euros")
    names = [schema["function"]["name"] for schema in first]
    assert names == ["convert", "weather"] and first == second
    assert router.normalize_invocation("convert", {"amount": 1.0})[0] == "convert"

    # Past the budget, disclosure is progressive again.
    small_budget = SemanticToolRouter(manager, top_k=1, stable_under_tokens=1)
    small_budget.begin_turn()
    assert len(small_budget.schemas_for_turn("weather in Lisbon")) < 4


def test_context_policy_carries_the_stable_tool_budget():
    from memorizz.tooling import ContextPolicy

    assert ContextPolicy().stable_tool_list_tokens == 6000
    saved = ContextPolicy(stable_tool_list_tokens=0).to_dict()
    assert saved["stable_tool_list_tokens"] == 0
    assert ContextPolicy.from_value(saved).stable_tool_list_tokens == 0
