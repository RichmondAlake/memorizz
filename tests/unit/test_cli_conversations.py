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
    from memorizz.cli import picker

    monkeypatch.setattr(picker, "picker_available", lambda console: True)

    commands.cmd_conversations(session, "")

    assert resume_calls == [("memory-2", "thread-2")]
    assert session.memory_id == "memory-2"
    assert session.thread_id == "thread-2"
    assert saved == [{"memory_id": "memory-2", "thread_id": "thread-2"}]
    assert "Resumed conversation Resume [this] conversation" in output.getvalue()


@pytest.mark.unit
def test_thread_messages_keep_full_text_and_fold_tools_into_the_reply():
    now = datetime.now(timezone.utc)
    provider = _Provider(
        {
            "memory-1": [
                # Offloaded tool results are stamped before their turn's user row.
                _row(
                    "memory-1",
                    "thread-1",
                    "tool",
                    '{"tool_name": "update_persona", "digest": "ok"}',
                    now,
                ),
                _row("memory-1", "thread-1", "tool", '{"type":"trace_bundle"}', now),
                _row(
                    "memory-1",
                    "thread-1",
                    "user",
                    "Be my coach\nand plan my week",
                    now + timedelta(seconds=1),
                ),
                _row(
                    "memory-1",
                    "thread-1",
                    "assistant",
                    "**Done.**\n\n1. Monday",
                    now + timedelta(seconds=2),
                ),
                _row(
                    "memory-1",
                    "thread-1",
                    "user",
                    "Next?",
                    now + timedelta(seconds=3),
                ),
                _row(
                    "memory-1",
                    "thread-1",
                    "assistant",
                    "Ship it.",
                    now + timedelta(seconds=4),
                ),
                _row("memory-1", "thread-2", "user", "Other thread", now),
                _row(
                    "memory-1",
                    "thread-1",
                    "user",
                    "Another agent",
                    now,
                    agent_id="agent-2",
                ),
            ]
        }
    )

    messages = conversations.thread_messages(
        provider, "memory-1", "thread-1", agent_id="agent-1"
    )

    assert [(m["role"], m["text"], m["tools"]) for m in messages] == [
        ("user", "Be my coach\nand plan my week", []),
        ("assistant", "**Done.**\n\n1. Monday", ["update_persona"]),
        ("user", "Next?", []),
        ("assistant", "Ship it.", []),
    ]


@pytest.mark.unit
def test_cmd_conversations_replays_the_resumed_thread(monkeypatch):
    from memorizz.cli import picker

    now = datetime.now(timezone.utc)
    provider = _Provider(
        {
            "memory-2": [
                _row(
                    "memory-2",
                    "thread-2",
                    "user",
                    f"question {n}",
                    now + timedelta(seconds=2 * n),
                )
                for n in range(20)
            ]
            + [
                _row(
                    "memory-2",
                    "thread-2",
                    "assistant",
                    f"**answer {n}**",
                    now + timedelta(seconds=2 * n + 1),
                )
                for n in range(20)
            ]
        }
    )
    selected = ConversationSummary(
        memory_id="memory-2",
        thread_id="thread-2",
        title="question 0",
        preview="question 19",
        created_at=now,
        updated_at=now,
        message_count=40,
    )
    output = StringIO()
    session = SimpleNamespace(
        agent=SimpleNamespace(
            agent_id="agent-1",
            memory_ids=["memory-2"],
            resume_thread=lambda memory_id, thread_id: None,
        ),
        provider=provider,
        memory_id="memory-1",
        thread_id="thread-1",
        user_id=None,
        code_mode=False,
        console=Console(file=output, color_system=None, width=120),
    )
    monkeypatch.setattr(
        commands.conversations, "discover_conversations", lambda *a, **k: [selected]
    )
    monkeypatch.setattr(
        commands.conversations, "pick_conversation", lambda *a, **k: selected
    )
    monkeypatch.setattr(commands.cfg, "save_state", lambda state: None)
    monkeypatch.setattr(picker, "picker_available", lambda console: True)

    commands.cmd_conversations(session, "")

    text = output.getvalue()
    assert "Earlier in this conversation" in text
    assert "… 10 earlier messages" in text
    assert "memorizz> question 4\n" not in text
    assert "memorizz> question 5\n" in text and "memorizz> question 19\n" in text
    # Answers render as Markdown, not raw markup.
    assert "answer 19" in text and "**answer 19**" not in text
    # The summary line closes the replay, right above the prompt.
    assert text.index("answer 19") < text.index("Resumed conversation question 0")


@pytest.mark.unit
def test_cmd_conversations_list_prints_without_opening_the_picker(monkeypatch):
    from datetime import datetime

    from memorizz.cli import picker

    rows = [
        ConversationSummary(
            memory_id="memory-1",
            thread_id=f"thread-{n}",
            title=title,
            preview="",
            created_at=None,
            updated_at=datetime(2026, 10, 9, 12, n),
            message_count=n * 2,
        )
        for n, title in [(1, "Release notes"), (2, "Harness progress")]
    ]
    output = StringIO()
    session = SimpleNamespace(
        agent=SimpleNamespace(agent_id="a", memory_ids=["memory-1"]),
        provider=object(),
        memory_id="memory-1",
        thread_id="thread-2",
        user_id=None,
        console=Console(file=output, color_system=None, width=120),
    )
    monkeypatch.setattr(
        commands.conversations, "discover_conversations", lambda *a, **k: rows
    )

    def no_picker(*args, **kwargs):
        raise AssertionError("the picker must not open")

    monkeypatch.setattr(commands.conversations, "pick_conversation", no_picker)
    monkeypatch.setattr(picker, "picker_available", lambda console: True)
    commands.cmd_conversations(session, "list")
    text = output.getvalue()
    assert "Conversations (2 of 2)" in text
    assert "▶ Harness progress" in text and "4 messages" in text
    assert "  Release notes" in text

    # Without a terminal a search prints the matches instead of a picker.
    output.truncate(0)
    output.seek(0)
    monkeypatch.setattr(picker, "picker_available", lambda console: False)
    commands.cmd_conversations(session, "release")
    assert "Conversations (1 of 2)" in output.getvalue()


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


@pytest.mark.unit
def test_cmd_history_prints_the_whole_thread_untruncated():
    now = datetime.now(timezone.utc)
    long_answer = "Listen up. " + "word " * 200 + "the end."
    provider = _Provider(
        {
            "memory-1": [
                _row(
                    "memory-1",
                    "thread-1",
                    "tool",
                    '{"tool_name": "update_persona", "digest": "{\'updated\': True}"}',
                    now,
                ),
                _row("memory-1", "thread-1", "user", "update your persona", now),
                _row(
                    "memory-1",
                    "thread-1",
                    "assistant",
                    long_answer,
                    now + timedelta(seconds=1),
                ),
            ]
            + [
                _row(
                    "memory-1",
                    "thread-1",
                    role,
                    f"{role} {n}",
                    now + timedelta(seconds=2 + 2 * n + (role == "assistant")),
                )
                for n in range(30)
                for role in ("user", "assistant")
            ]
        }
    )
    output = StringIO()
    session = SimpleNamespace(
        agent=SimpleNamespace(agent_id="agent-1"),
        provider=provider,
        memory_id="memory-1",
        thread_id="thread-1",
        user_id=None,
        code_mode=False,
        console=Console(file=output, color_system=None, width=120),
    )

    commands.cmd_history(session, "")

    text = output.getvalue()
    assert "Conversation history · 62 messages" in text
    assert "earlier messages" not in text
    assert "memorizz> update your persona" in text
    assert "Tools: update_persona" in text and "digest" not in text
    assert "the end." in text
    assert "memorizz> user 29" in text and "assistant 29" in text

    output.truncate(0)
    output.seek(0)
    session.thread_id = None
    commands.cmd_history(session, "")
    assert "No conversation history yet." in output.getvalue()
