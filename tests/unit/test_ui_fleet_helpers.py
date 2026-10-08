"""Fleet pages must not load full agents or whole histories per row."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memorizz.ui import helpers
from memorizz.ui.state import _state


class _Provider:
    def __init__(self):
        self.history_limits = []

    def retrieve_conversation_history_ordered_by_timestamp(self, memory_id, limit=None):
        self.history_limits.append(limit)
        rows = [
            {"memory_id": memory_id, "timestamp": 100.0, "content": "old"},
            {"memory_id": memory_id, "timestamp": 200.0, "content": "new"},
        ]
        return rows[-limit:] if limit else rows

    def retrieve_memagent(self, agent_id):
        return SimpleNamespace(
            agent_id=agent_id, tools=[{"name": "a"}, {"name": "b"}, {"name": "c"}]
        )


@pytest.fixture()
def provider(monkeypatch):
    stub = _Provider()
    monkeypatch.setitem(_state, "provider", stub)
    return stub


@pytest.mark.unit
def test_last_run_map_reads_one_row_per_conversation(provider):
    agent = SimpleNamespace(agent_id="agent-1", memory_ids=["m1", "m2"])
    last_run = helpers._load_agent_last_run_map([agent])
    assert last_run == {"agent-1": 200.0}
    assert provider.history_limits == [1, 1]


@pytest.mark.unit
def test_history_helper_passes_limit_to_provider(provider):
    rows = helpers._retrieve_conversation_history("m1", limit=1)
    assert [row["content"] for row in rows] == ["new"]
    assert provider.history_limits == [1]


@pytest.mark.unit
def test_tool_count_uses_stored_record_not_memagent_load(provider, monkeypatch):
    from memorizz.memagent import MemAgent

    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("fleet pages must not instantiate MemAgent")

    monkeypatch.setattr(MemAgent, "load", staticmethod(_boom))
    assert helpers._count_runtime_tools_for_agent("agent-1") == 3
