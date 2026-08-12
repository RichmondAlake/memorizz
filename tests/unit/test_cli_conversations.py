"""Tests for the interactive CLI conversation resume flow."""

from datetime import datetime, timedelta, timezone
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz.cli import commands, conversations
from memorizz.cli.conversations import ConversationSummary


class _Provider:
    def __init__(self, rows_by_memory):
        self.rows_by_memory = rows_by_memory

    def retrieve_conversation_history_ordered_by_timestamp(
        self,
        memory_id,
        memory_type=None,
        limit=None,
        user_id=None,
        thread_id=None,
    ):
        rows = list(self.rows_by_memory.get(memory_id, []))
        if thread_id is not None:
            rows = [row for row in rows if row.get("thread_id") == thread_id]
        return rows[-limit:] if limit else rows


def _row(
    memory_id,
    thread_id,
    role,
    content,
    timestamp,
    *,
    user_id=None,
    agent_id="agent-1",
):
    return {
        "memory_id": memory_id,
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "agent_id": agent_id,
    }


@pytest.mark.unit
def test_discover_conversations_groups_titles_and_sorts_by_latest_activity():
    now = datetime.now(timezone.utc)
    provider = _Provider(
        {
            "memory-1": [
                _row(
                    "memory-1",
                    "thread-old",
                    "user",
                    "Plan a migration",
                    now - timedelta(days=3),
                ),
                _row(
                    "memory-1",
                    "thread-old",
                    "assistant",
                    "Here is the plan",
                    now - timedelta(days=3, seconds=-10),
                ),
                _row(
                    "memory-1",
                    "thread-new",
                    "user",
                    "Investigate the API timeout",
                    now - timedelta(hours=2),
                ),
                _row(
                    "memory-1",
                    "thread-new",
                    "assistant",
                    "I found the cause",
                    now - timedelta(hours=1),
                ),
                _row(
                    "memory-1",
                    "thread-new",
                    "tool",
                    '{"type":"trace_bundle"}',
                    now,
                ),
                _row(
                    "memory-1",
                    "other-agent",
                    "user",
                    "Do not show this",
                    now,
                    agent_id="agent-2",
                ),
            ]
        }
    )

    result = conversations.discover_conversations(
        provider,
        ["memory-1", "memory-1"],
        user_id=None,
        agent_id="agent-1",
    )

    assert [item.thread_id for item in result] == ["thread-new", "thread-old"]
    assert result[0].title == "Investigate the API timeout"
    assert result[0].preview == "Investigate the API timeout"
    assert result[0].message_count == 2


@pytest.mark.unit
def test_discover_conversations_handles_nested_rows_and_tenant_scope():
    now = datetime.now(timezone.utc)
    provider = _Provider(
        {
            "memory-1": [
                {
                    "memory_id": "memory-1",
                    "timestamp": now.isoformat(),
                    "content": {
                        "thread_id": "alice-thread",
                        "role": "user",
                        "content": "Alice private conversation",
                        "user_id": "alice",
                    },
                },
                {
                    "memory_id": "memory-1",
                    "timestamp": (now + timedelta(seconds=1)).isoformat(),
                    "content": {
                        "thread_id": "bob-thread",
                        "role": "user",
                        "content": "Bob private conversation",
                        "user_id": "bob",
                    },
                },
            ]
        }
    )

    result = conversations.discover_conversations(
        provider, ["memory-1"], user_id="alice"
    )

    assert [item.thread_id for item in result] == ["alice-thread"]


@pytest.mark.unit
def test_filter_conversations_matches_all_terms_across_title_and_ids():
    items = [
        ConversationSummary(
            memory_id="memory-orders",
            thread_id="thread-refund",
            title="Refund an order",
            preview="Send a receipt",
            created_at=None,
            updated_at=None,
            message_count=4,
        ),
        ConversationSummary(
            memory_id="memory-api",
            thread_id="thread-timeout",
            title="Investigate API latency",
            preview="Timeout in the jobs endpoint",
            created_at=None,
            updated_at=None,
            message_count=8,
        ),
    ]

    assert conversations.filter_conversations(items, "api timeout") == [items[1]]
    assert conversations.filter_conversations(items, "refund thread-refund") == [
        items[0]
    ]


@pytest.mark.unit
def test_cmd_conversations_resumes_and_persists_selected_pair(monkeypatch):
    selected = ConversationSummary(
        memory_id="memory-2",
        thread_id="thread-2",
        title="Resume [this] conversation",
        preview="latest prompt",
        created_at=None,
        updated_at=None,
        message_count=6,
    )
    agent = SimpleNamespace(
        agent_id="agent-1",
        memory_ids=["memory-1", "memory-2"],
        resume_thread=lambda memory_id, thread_id: resume_calls.append(
            (memory_id, thread_id)
        ),
    )
    output = StringIO()
    session = SimpleNamespace(
        agent=agent,
        provider=object(),
        memory_id="memory-1",
        thread_id="thread-1",
        user_id=None,
        console=Console(file=output, color_system=None, width=120),
    )
    resume_calls = []
    saved = []
    monkeypatch.setattr(
        commands.conversations, "discover_conversations", lambda *a, **k: [selected]
    )
    monkeypatch.setattr(
        commands.conversations, "pick_conversation", lambda *a, **k: selected
    )
    monkeypatch.setattr(commands.cfg, "save_state", lambda state: saved.append(state))

    commands.cmd_conversations(session, "")

    assert resume_calls == [("memory-2", "thread-2")]
    assert session.memory_id == "memory-2"
    assert session.thread_id == "thread-2"
    assert saved == [{"memory_id": "memory-2", "thread_id": "thread-2"}]
    assert "Resumed conversation Resume [this] conversation" in output.getvalue()


@pytest.mark.unit
def test_cmd_new_starts_a_thread_in_current_memory_and_persists_it(monkeypatch):
    saved = []
    agent = SimpleNamespace(
        start_new_thread=lambda memory_id: (calls.append(memory_id) or "new-thread"),
        get_current_memory_id=lambda: "current-memory",
    )
    calls = []
    session = SimpleNamespace(
        agent=agent,
        memory_id="current-memory",
        thread_id="old-thread",
        console=Console(file=StringIO(), color_system=None),
    )
    monkeypatch.setattr(commands.cfg, "save_state", lambda state: saved.append(state))

    commands.cmd_new(session, "")

    assert calls == ["current-memory"]
    assert session.thread_id == "new-thread"
    assert saved == [{"memory_id": "current-memory", "thread_id": "new-thread"}]


@pytest.mark.unit
def test_conversation_command_and_aliases_are_registered():
    assert "/conversations" in commands.command_completions()
    assert commands.ALIASES["conversation"] == "conversations"
    assert commands.ALIASES["converstations"] == "conversations"
    assert commands.ALIASES["threads"] == "conversations"
