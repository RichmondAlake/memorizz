# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The sidebar's agent rows: a current step while a reply streams, the last
message otherwise, and the shortcut that folds the sidebar."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.enums.memory_type import MemoryType
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import agent_status, state
from memorizz.ui.app import create_app

UI = Path(__file__).resolve().parents[2] / "src" / "memorizz" / "ui"


def _say(provider, memory_id, role, content, second):
    provider.store(
        {
            "role": role,
            "content": content,
            "memory_id": memory_id,
            "agent_id": "agent-1",
            "user_id": None,
            "timestamp": f"2026-11-18T10:00:{second:02d}+00:00",
        },
        MemoryType.CONVERSATION_MEMORY,
    )


@pytest.fixture()
def store(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.store_memagent(
        MemAgentModel(agent_id="agent-1", name="Vera", memory_ids=["m1"])
    )
    provider.store_memagent(MemAgentModel(agent_id="agent-2", name="Noor"))
    agent_status.reset()
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": False,
        },
    ):
        yield provider
    agent_status.reset()
    provider.close()


@pytest.fixture()
def client(store):
    return TestClient(create_app(), follow_redirects=False)


def _status(client, ids="agent-1,agent-2"):
    response = client.get("/api/nav/agent-status", params={"ids": ids})
    assert response.status_code == 200
    return response.json()["agents"]


@pytest.mark.unit
def test_an_idle_agent_shows_the_last_thing_said(client, store):
    _say(store, "m1", "user", "Where is my order?", 1)
    _say(store, "m1", "assistant", "It ships **tomorrow**,\nby courier.", 2)

    agents = _status(client)

    assert agents["agent-1"]["state"] == "idle"
    assert agents["agent-1"]["text"] == "It ships tomorrow, by courier."
    assert agents["agent-1"]["at"] > 0
    # An agent that has never spoken has a row, and nothing on it.
    assert agents["agent-2"] == {"state": "idle", "text": "", "at": None}


@pytest.mark.unit
def test_a_question_still_unanswered_is_marked_as_the_users(client, store):
    _say(store, "m1", "user", "Any update?", 1)

    assert _status(client)["agent-1"]["text"] == "You: Any update?"


@pytest.mark.unit
def test_a_streaming_reply_shows_its_step_then_gives_way_to_the_answer(client, store):
    _say(store, "m1", "assistant", "Earlier answer", 1)
    assert _status(client)["agent-1"]["text"] == "Earlier answer"

    def event(kind, **fields):
        return {"type": kind, "run_id": "run-1", **fields}

    agent_status.note_event("agent-1", event("run.started"))
    assert _status(client)["agent-1"]["text"] == "Getting ready"
    agent_status.note_event("agent-1", event("status", stage="provider_request"))
    assert _status(client)["agent-1"]["text"] == "Thinking"
    agent_status.note_event("agent-1", event("tool.started", tool_name="find_order"))
    running = _status(client)["agent-1"]
    assert running["state"] == "running"
    assert running["text"] == "Running find_order"
    agent_status.note_event("agent-1", event("answer.delta", delta="It"))
    assert _status(client)["agent-1"]["text"] == "Writing a reply"

    # The answer is stored before the run ends; ending it drops the cached line.
    _say(store, "m1", "assistant", "It ships tomorrow.", 5)
    agent_status.note_event("agent-1", event("run.done", status="completed"))

    done = _status(client)["agent-1"]
    assert done["state"] == "idle"
    assert done["text"] == "It ships tomorrow."


@pytest.mark.unit
def test_a_call_awaiting_approval_is_flagged_live_and_after_the_run(client, store):
    agent_status.note_event(
        "agent-1",
        {
            "type": "approval.required",
            "run_id": "run-2",
            "proposal": {"tool_name": "refund_order", "proposal_id": "p1"},
        },
    )
    live = _status(client)["agent-1"]
    assert live == {**live, "state": "waiting", "text": "Needs approval: refund_order"}

    # The run pauses there. What is left in the thread is the waiting note.
    _say(
        store,
        "m1",
        "assistant",
        "[Waiting for approval] I asked to run refund_order with these "
        "arguments (proposal p1). It has not run.",
        9,
    )
    agent_status.end_run("agent-1", "run-2")

    after = _status(client)["agent-1"]
    assert after["state"] == "waiting"
    assert after["text"] == "Needs approval: refund_order"


@pytest.mark.unit
def test_status_is_only_given_for_agents_in_this_store(client):
    assert _status(client, ids="agent-1,someone-else").keys() == {"agent-1"}
    assert _status(client, ids="") == {}


@pytest.mark.unit
def test_the_stream_route_follows_its_run_and_lets_go_when_it_ends(store):
    source = (UI / "routers" / "playground.py").read_text()
    stream = source[source.index("async def event_stream():") :]
    stream = stream[: stream.index("return StreamingResponse(")]
    assert "agent_status.note_event(agent_id, event)" in stream
    # Also on a browser abort, where no run.done event is ever read.
    assert "agent_status.end_run(agent_id, run_id)" in stream.split("finally:")[1]


@pytest.mark.unit
def test_sidebar_agent_rows_carry_a_status_line(client):
    page = client.get("/agents").text
    assert 'data-nav-agent="agent-1"' in page
    assert 'class="nav-agent-status" hidden' in page
    script = (UI / "static" / "js" / "app.js").read_text()
    assert "/api/nav/agent-status" in script
    # The lines are text, never markup: a message cannot inject into the page.
    body = script[script.index("function initializeAgentNavStatus") :]
    body = body[: body.index("function initializeBenchmarkMonitor")]
    assert "innerHTML" not in body


@pytest.mark.unit
def test_command_b_folds_the_sidebar():
    script = (UI / "static" / "js" / "app.js").read_text()
    handler = script[script.index("// ⌘B on a Mac") :]
    handler = handler[: handler.index("initializeAgentNavStatus(storage)")]
    assert 'event.key.toLowerCase() !== "b"' in handler
    assert "setSidebarCollapsed(!body.classList" in handler
    # Bold in a rich-text editor keeps its keys.
    assert "contenteditable" in handler


@pytest.mark.unit
def test_the_sidebar_does_not_list_the_agents_on_every_poll(client, store, monkeypatch):
    listed = []
    original = store.list_memagents
    monkeypatch.setattr(
        store,
        "list_memagents",
        lambda *a, **k: (listed.append(1), original(*a, **k))[1],
    )

    for _ in range(4):
        assert set(_status(client)) == {"agent-1", "agent-2"}
    assert len(listed) == 1

    # An agent made since then is found at once, not after the list expires.
    store.store_memagent(MemAgentModel(agent_id="agent-3", name="Idris"))
    assert "agent-3" in _status(client, "agent-1,agent-3")
    assert len(listed) == 2

    # A finished run may have given its agent a new conversation.
    agent_status.end_run("agent-1", "run-1")
    _status(client)
    assert len(listed) == 3
