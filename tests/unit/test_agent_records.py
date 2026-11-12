"""An agent reads its own runs, tool calls and learned skills from a store
it shares with other agents."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memagent.core import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.memory_provider.base import MemoryProvider


@pytest.fixture
def provider(tmp_path):
    return FileSystemProvider(FileSystemConfig(root_path=tmp_path / "store"))


def _at(hour: int) -> str:
    return datetime(2026, 11, 12, hour, 0, tzinfo=timezone.utc).isoformat()


def _run(provider, agent_id, hour, *, thread="t1", outcome="success"):
    provider.store(
        {
            "agent_id": agent_id,
            "created_at": _at(hour),
            "origin_thread_id": thread,
            "outcome": outcome,
            "user_query": f"run at {hour}",
        },
        MemoryType.WORKFLOW_MEMORY,
    )


def _agent(provider, agent_id="ada"):
    return SimpleNamespace(memory_provider=provider, agent_id=agent_id)


def test_runs_are_this_agents_only_and_oldest_first(provider):
    _run(provider, "ada", 11)
    _run(provider, "twin", 10)
    _run(provider, "ada", 9)
    runs = MemAgent.workflow_runs(_agent(provider))
    assert [run["user_query"] for run in runs] == ["run at 9", "run at 11"]
    assert MemAgent.workflow_runs(_agent(provider, "nobody")) == []


def test_runs_filter_by_conversation_outcome_time_and_count(provider):
    _run(provider, "ada", 9, thread="monday")
    _run(provider, "ada", 10, thread="tuesday", outcome="failure")
    _run(provider, "ada", 11, thread="tuesday")
    ada = _agent(provider)
    assert len(MemAgent.workflow_runs(ada, thread_id="tuesday")) == 2
    (failed,) = MemAgent.workflow_runs(ada, outcome="failure")
    assert failed["user_query"] == "run at 10"
    since = MemAgent.workflow_runs(ada, since=_at(10))
    assert [run["user_query"] for run in since] == ["run at 10", "run at 11"]
    assert (
        len(
            MemAgent.workflow_runs(
                ada, since=datetime(2026, 11, 12, 11, tzinfo=timezone.utc)
            )
        )
        == 1
    )
    latest = MemAgent.workflow_runs(ada, limit=2, newest_first=True)
    assert [run["user_query"] for run in latest] == ["run at 11", "run at 10"]
    assert MemAgent.workflow_runs(ada, limit=0) == []


def test_order_reads_times_not_strings(provider):
    # The same instant written two ways, and a later one with an offset that
    # sorts first as text.
    provider.store(
        {
            "agent_id": "ada",
            "created_at": "2026-11-12T10:00:00+01:00",
            "user_query": "nine utc",
        },
        MemoryType.WORKFLOW_MEMORY,
    )
    provider.store(
        {
            "agent_id": "ada",
            "created_at": "2026-11-12T09:30:00+00:00",
            "user_query": "half nine utc",
        },
        MemoryType.WORKFLOW_MEMORY,
    )
    runs = MemAgent.workflow_runs(_agent(provider))
    assert [run["user_query"] for run in runs] == ["nine utc", "half nine utc"]


def test_tool_calls_and_learned_skills_are_scoped_too(provider):
    for agent_id, tool, ok, hour in (
        ("ada", "lookup", True, 9),
        ("ada", "lookup", False, 10),
        ("ada", "pay", True, 11),
        ("twin", "lookup", False, 12),
    ):
        provider.store(
            {
                "agent_id": agent_id,
                "tool_name": tool,
                "success": ok,
                "thread_id": "t1",
                "timestamp": _at(hour),
            },
            MemoryType.TOOL_LOG,
        )
    ada = _agent(provider)
    assert [call["tool_name"] for call in MemAgent.tool_calls(ada)] == [
        "lookup",
        "lookup",
        "pay",
    ]
    assert len(MemAgent.tool_calls(ada, tool="lookup")) == 2
    (refused,) = MemAgent.tool_calls(ada, failed=True)
    assert refused["tool_name"] == "lookup" and refused["success"] is False
    assert len(MemAgent.tool_calls(ada, failed=False)) == 2

    for agent_id, name, status in (
        ("ada", "Pay", "active"),
        ("ada", "Hold", "shadow"),
        ("twin", "Pay", "active"),
    ):
        provider.store(
            {"agent_id": agent_id, "name": name, "status": status}, MemoryType.SKILLBOX
        )
    assert sorted(skill["name"] for skill in MemAgent.learned_skills(ada)) == [
        "Hold",
        "Pay",
    ]
    assert [
        skill["name"] for skill in MemAgent.learned_skills(ada, status="shadow")
    ] == ["Hold"]


def test_the_scoped_read_is_on_every_provider(provider):
    _run(provider, "ada", 9)
    _run(provider, "twin", 9)
    rows = provider.list_for_agent(MemoryType.WORKFLOW_MEMORY, "ada")
    assert [row["agent_id"] for row in rows] == ["ada"]
    assert provider.list_for_agent(MemoryType.WORKFLOW_MEMORY, "") == []
    # A provider that only has list_all still serves it.
    assert MemoryProvider.list_for_agent is not None

    class Old:
        def list_all(self, memory_type):
            return provider.list_all(memory_type)

    runs = MemAgent.workflow_runs(
        SimpleNamespace(memory_provider=Old(), agent_id="ada")
    )
    assert len(runs) == 1


def test_a_time_that_is_not_one_is_refused(provider):
    with pytest.raises(ValueError):
        MemAgent.workflow_runs(_agent(provider), since="last tuesday")
