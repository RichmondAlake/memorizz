# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Persona learning in the UI: the agent form, the control plane panel and the
playground's memory overview."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz import FileSystemConfig, FileSystemProvider
from memorizz.enums import MemoryType
from memorizz.long_term.semantic.persona.persona import Persona
from memorizz.memagent.models import MemAgentModel

pytestmark = pytest.mark.unit

BAKERY = "Advise on running a small bakery."


class _Model:
    model = "background"

    def __init__(self, reply=None):
        self.reply = reply
        self.prompts = []

    def generate_text(self, prompt, instructions=None):
        self.prompts.append(prompt)
        return json.dumps(
            self.reply
            or {
                "changes": [
                    {
                        "field": "goals",
                        "text": BAKERY,
                        "evidence": [
                            {"conversation": "C1", "quote": "I run a small bakery"},
                            {"conversation": "C2", "quote": "my bakery again"},
                            {"conversation": "C3", "quote": "my bakery shifts"},
                            {"conversation": "C3", "quote": "my bakery shifts"},
                        ],
                    }
                ]
            }
        )

    def get_config(self):
        return {"provider": "ollama", "model": "background"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding",
        lambda text, **kwargs: [0.0] * 8,
    )


@pytest.fixture()
def seeded(tmp_path):
    pytest.importorskip("fastapi")
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "persona-ui",
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-1",
            name="Support agent",
            memory_ids=["memory-1"],
            application_mode="assistant",
            llm_config={"provider": "ollama", "model": "gemma4"},
            background_llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
            persona=Persona(name="Zed", goals="Keep the harbour safe."),
            persona_learning={"enabled": True, "every_n_turns": 5, "auto_apply": False},
        )
    )
    for thread, text in (
        ("t1", "I run a small bakery."),
        ("t2", "For my bakery again: how do I cut waste?"),
        ("t3", "On my bakery shifts: who starts first?"),
    ):
        provider.store(
            {
                "role": "user",
                "content": text,
                "memory_id": "memory-1",
                "thread_id": thread,
                "conversation_id": thread,
                "user_id": None,
                "agent_id": "agent-1",
                "timestamp": "2026-10-01T10:00:00",
            },
            MemoryType.CONVERSATION_MEMORY,
        )
    return provider


@pytest.fixture()
def client(seeded):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app
    from memorizz.ui.routers import learning_control_plane as route

    with patch.dict(
        state._state,
        {"provider": seeded, "provider_type": "filesystem", "connection_info": {}},
    ), patch.dict(route._last_actions, {}, clear=True), patch(
        "memorizz.learning.consolidation.model_for_agent",
        lambda record: _Model(),
    ):
        yield TestClient(create_app(), follow_redirects=False)


def _scope(**extra):
    return {
        "agent_id": "agent-1",
        "memory_id": "",
        "user_id": "",
        "thread_id": "",
        **extra,
    }


def _page(client):
    return client.get("/learning-control-plane?agent_id=agent-1").text


def _review(client):
    response = client.post(
        "/learning-control-plane/persona-review",
        data=_scope(persona_memory_id="memory-1"),
    )
    assert response.status_code == 303


def _persona(provider):
    return provider.retrieve_memagent("agent-1").persona


def test_the_panel_says_what_it_does_and_when(client):
    html = _page(client)
    assert 'id="persona-learning"' in html
    assert "ollama/qwen2.5:7b" in html
    assert "Every 5 turns" in html and "waits for review" in html
    assert "name and role never change" in html


def test_review_shows_the_change_and_its_quotes_before_anything_happens(client, seeded):
    _review(client)
    html = _page(client)
    assert "A change was proposed" in html
    assert BAKERY in html
    assert "I run a small bakery" in html
    assert _persona(seeded).version == 1


def test_apply_and_undo_from_the_page(client, seeded):
    original = _persona(seeded).goals
    _review(client)
    client.post(
        "/learning-control-plane/persona-apply", data=_scope(approved_by="richmond")
    )
    persona = _persona(seeded)
    assert persona.version == 2
    assert persona.goals.endswith(BAKERY)
    html = _page(client)
    assert "Persona is now version 2" in html
    assert "Changes it made" in html and "richmond" in html

    client.post(
        "/learning-control-plane/persona-undo",
        data=_scope(approved_by="richmond", version="2"),
    )
    assert _persona(seeded).goals == original
    assert "Undone. Persona is now version 3" in _page(client)


def test_turning_a_change_down_keeps_it_from_coming_back(client, seeded):
    _review(client)
    client.post(
        "/learning-control-plane/persona-dismiss", data=_scope(approved_by="richmond")
    )
    assert _persona(seeded).version == 1
    assert "Waiting for review" not in _page(client)
    assert seeded.retrieve_memagent("agent-1").persona_learning_state["declined"] == [
        BAKERY
    ]


def test_an_action_that_cant_run_says_why(client):
    client.post(
        "/learning-control-plane/persona-apply", data=_scope(approved_by="richmond")
    )
    assert "No persona change is waiting" in _page(client)


def test_an_agent_with_no_persona_gets_no_panel(client, seeded):
    from memorizz.memagent import persistence

    persistence.update_agent_record(seeded, "agent-1", persona=None)
    assert 'id="persona-learning"' not in _page(client)


def test_a_multi_tenant_agent_is_told_why_it_doesnt_learn(client, seeded):
    from memorizz.memagent import persistence

    persistence.update_agent_record(seeded, "agent-1", multi_tenant=True)
    html = _page(client)
    assert "doesn't learn from any one of them" in html
    assert "/learning-control-plane/persona-review" not in html


def test_the_memory_overview_reports_the_schedule_and_a_waiting_change(client):
    _review(client)
    with patch("memorizz.memagent.core.MemAgent.save"):
        body = client.get("/api/agents/agent-1/memory-overview").json()
    assert body["persona_learning"]["every_n_turns"] == 5
    assert body["pending_persona_change"] is True


def _form(**overrides):
    return {
        "agent_name": "Form agent",
        "instruction": "You remember things.",
        "application_mode": "assistant",
        "max_steps": "10",
        "tool_access": "private",
        "llm_provider": "ollama",
        "llm_model": "gemma4",
        "llm_config_json": '{"base_url": "http://127.0.0.1:11434"}',
        "persona_name": "Zed",
        "persona_role": "Harbour pilot",
        **overrides,
    }


def test_the_agent_form_round_trips_the_settings(client, seeded):
    response = client.post(
        "/agents/new",
        data=_form(
            persona_learning="on",
            persona_learning_every_n_turns="12",
            persona_learning_auto_apply="on",
        ),
    )
    assert response.status_code in (302, 303)
    agent_id = response.headers["location"].split("/agents/")[1].split("/")[0]
    saved = seeded.retrieve_memagent(agent_id)
    assert saved.persona_learning["every_n_turns"] == 12
    assert saved.persona_learning["auto_apply"] is True

    html = client.get(f"/agents/{agent_id}/edit").text
    assert 'id="persona_learning" name="persona_learning" value="on" checked' in html
    assert 'value="12"' in html

    client.post(f"/agents/{agent_id}/edit", data=_form())
    assert seeded.retrieve_memagent(agent_id).persona_learning is None


def test_editing_an_agent_keeps_what_learning_has_done(client, seeded):
    _review(client)
    client.post(
        "/agents/agent-1/edit",
        data=_form(persona_learning="on", persona_learning_every_n_turns="9"),
    )
    saved = seeded.retrieve_memagent("agent-1")
    assert saved.persona_learning["every_n_turns"] == 9
    assert saved.persona_learning["auto_apply"] is False
    assert saved.persona_learning_state["pending"]["addition"] == BAKERY
