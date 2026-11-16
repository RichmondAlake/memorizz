"""Tools an environment lends an agent: there while it is, gone when it leaves,
and never part of the saved agent."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from memorizz import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", Mock())
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )


def own_tool(x: int) -> int:
    """Double a number."""
    return x * 2


def look_around() -> str:
    """See who is nearby."""
    return "nobody"


def wave_at(name: str) -> str:
    """Wave at someone."""
    return f"waved at {name}"


@pytest.fixture
def agent(memory):
    from memorizz.llms.openai import OpenAI

    return MemAgent(
        model=OpenAI(model="gpt-test"),
        memory_provider=memory,
        auto_register=False,
        tools=[own_tool],
    )


def test_entering_lends_the_tools_and_leaving_takes_them_back(agent):
    before = set(agent.tool_manager.list_tools())

    lent = agent.enter_environment("room", [look_around, wave_at])

    assert lent == ["look_around", "wave_at"]
    assert agent.environments == {"room": ["look_around", "wave_at"]}
    assert set(agent.tool_manager.list_tools()) == before | set(lent)
    assert (
        agent.tool_manager.execute_tool("wave_at", {"name": "Bo"})[0] == "waved at Bo"
    )
    # Offered on every turn: they have no toolbox row to be found by.
    assert set(lent) <= agent.semantic_tool_router.always_visible

    assert agent.leave_environment("room") == lent
    assert set(agent.tool_manager.list_tools()) == before
    assert not set(lent) & agent.semantic_tool_router.always_visible
    assert agent.environments == {}
    assert agent.leave_environment("room") == []


def test_a_save_inside_an_environment_writes_none_of_its_tools(agent, memory, tmp_path):
    agent.save()
    agent.enter_environment("room", [look_around, wave_at])

    agent.save()

    stored = {tool["name"] for tool in memory.retrieve_memagent(agent.agent_id).tools}
    assert "own_tool" in stored
    assert not {"look_around", "wave_at"} & stored
    # Nothing anywhere in the store: not the record, not toolbox memory.
    for path in (tmp_path / "memory").rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "look_around" not in text and "wave_at" not in text, path


def test_a_never_saved_agent_is_first_saved_without_them(agent, memory):
    agent.enter_environment("room", [look_around])
    agent.save()
    stored = {tool["name"] for tool in memory.retrieve_memagent(agent.agent_id).tools}
    assert "look_around" not in stored and "own_tool" in stored


def test_an_environment_cannot_shadow_the_agents_own_tool(agent):
    with pytest.raises(ValueError, match="already has a tool named 'own_tool'"):
        agent.enter_environment("room", [look_around, own_tool])
    # Nothing was lent when one tool was refused.
    assert "look_around" not in agent.tool_manager.list_tools()
    assert agent.environments == {}


def test_entering_again_replaces_the_tools(agent):
    agent.enter_environment("room", [look_around, wave_at])
    agent.enter_environment("room", [look_around])
    assert agent.environments == {"room": ["look_around"]}
    assert "wave_at" not in agent.tool_manager.list_tools()


def test_two_environments_keep_their_own_tools(agent):
    agent.enter_environment("room", [look_around])
    agent.enter_environment("garden", [wave_at])
    agent.leave_environment("room")
    assert agent.environments == {"garden": ["wave_at"]}
    assert "wave_at" in agent.tool_manager.list_tools()


def test_the_lent_tools_are_sent_to_the_model(agent):
    agent.enter_environment("room", [look_around])
    agent.semantic_tool_router.begin_turn(user_id=None)
    names = {
        json.dumps(schema)
        and (schema.get("function", {}).get("name") or schema.get("name"))
        for schema in agent._build_llm_tools("hello", None)
    }
    assert "look_around" in names


def test_a_run_that_used_only_lent_tools_is_not_kept_as_a_workflow(agent):
    from types import SimpleNamespace

    def workflow(*tools):
        return SimpleNamespace(
            steps={
                f"Step {i}: {name}": {"_id": name} for i, name in enumerate(tools, 1)
            }
        )

    agent.enter_environment("room", [look_around, wave_at])
    assert agent._used_only_environment_tools(workflow("look_around", "wave_at"))
    # A run that also used the agent's own tool is worth remembering.
    assert not agent._used_only_environment_tools(workflow("look_around", "own_tool"))
    assert agent._persist_workflow_run(workflow("look_around")) is False
    agent.leave_environment("room")
    assert not agent._used_only_environment_tools(workflow("look_around"))
