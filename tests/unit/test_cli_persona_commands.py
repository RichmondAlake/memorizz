# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""/persona keeps what it sets, and /persona-learning runs persona learning."""

from __future__ import annotations

import io
import json
import time
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz.cli import commands, memory_upkeep
from memorizz.enums import MemoryType
from memorizz.long_term.semantic.persona.persona import Persona
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

LLM = {"provider": "ollama", "model": "qwen2.5:7b"}
BAKERY = "Advise on running a small bakery."


class _Model:
    def __init__(self, reply=None):
        self.reply = reply if reply is not None else {"changes": []}

    def generate_text(self, prompt):
        return json.dumps(self.reply)


def _bakery_change():
    return {
        "changes": [
            {
                "field": "goals",
                "text": BAKERY,
                "evidence": [
                    {"conversation": "C1", "quote": "I run a small bakery"},
                    {"conversation": "C2", "quote": "my bakery again"},
                    {"conversation": "C3", "quote": "my bakery shifts"},
                ],
            }
        ]
    }


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding",
        lambda text, **kwargs: [0.0] * 8,
    )


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


def _session(provider, **settings):
    agent = MemAgent(llm_config=LLM, memory_provider=provider, **settings)
    agent.save()
    return SimpleNamespace(
        agent=agent,
        provider=provider,
        console=Console(file=io.StringIO(), width=200),
        memory_id="memory-1",
        user_id=None,
        thread_id="t1",
        llm_config=LLM,
    )


def _printed(session) -> str:
    return session.console.file.getvalue()


def _reload(session):
    return MemAgent.load(session.agent.agent_id, memory_provider=session.provider)


def _talk(session):
    for thread, text in (
        ("t1", "I run a small bakery."),
        ("t2", "For my bakery again: how do I cut waste?"),
        ("t3", "On my bakery shifts: who starts first?"),
    ):
        session.provider.store(
            {
                "memory_id": "memory-1",
                "thread_id": thread,
                "conversation_id": thread,
                "user_id": None,
                "agent_id": session.agent.agent_id,
                "role": "user",
                "content": text,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            MemoryType.CONVERSATION_MEMORY,
        )


# -- /persona ----------------------------------------------------------------


def test_a_persona_set_in_the_chat_is_there_after_a_relaunch(provider):
    session = _session(provider)
    commands.cmd_persona(session, "Ada | tutor me in maths | a patient teacher")

    loaded = _reload(session)
    persona = loaded.persona_manager.current_persona
    assert persona.name == "Ada"
    assert "tutor me in maths" in persona.goals
    assert "a patient teacher" in persona.background
    assert "You are Ada" in loaded._build_system_prompt()
    assert "update_persona" in loaded.tool_manager.list_tools()


def test_changing_a_persona_keeps_what_it_learned(provider):
    session = _session(provider, persona=Persona(name="Zed", role="Harbour pilot"))
    original = session.agent.persona_manager.current_persona
    persona_id = original.persona_id
    session.agent.tool_manager.get_tool_callable("update_persona")(
        updates={"background": "Years at sea."}, reason="The user asked"
    )

    commands.cmd_persona(session, "Ada | tutor me in maths")

    persona = _reload(session).persona_manager.current_persona
    assert persona.persona_id == persona_id
    assert persona.name == "Ada"
    assert persona.goals == "tutor me in maths"
    assert persona.background == "Years at sea."
    assert persona.role == "Harbour pilot"
    assert persona.version == 3
    assert len(persona.evolution_history) == 2
    assert "Persona set" in _printed(session)


def test_setting_the_same_persona_again_is_no_error(provider):
    session = _session(provider, persona=Persona(name="Zed"))
    commands.cmd_persona(session, "Zed")
    assert "Persona set" in _printed(session)
    assert _reload(session).persona_manager.current_persona.version == 1


def test_setting_a_persona_changes_nothing_else_on_the_saved_agent(provider):
    session = _session(provider, instruction="Saved instruction.")
    session.agent.instruction = "This session only."
    commands.cmd_persona(session, "Ada")
    assert provider.retrieve_memagent(session.agent.agent_id).instruction == (
        "Saved instruction."
    )


def test_a_save_that_fails_isnt_reported_as_done(provider, monkeypatch):
    session = _session(provider)

    def refuse(*args, **kwargs):
        raise RuntimeError("the store is read-only")

    monkeypatch.setattr(commands, "save_agent_fields", refuse)
    commands.cmd_persona(session, "Ada")
    assert "could not be saved" in _printed(session)
    assert "the store is read-only" in _printed(session)
    assert "Persona set →" not in _printed(session)


def test_a_persona_name_is_shown_as_typed(provider):
    session = _session(provider)
    commands.cmd_persona(session, "[Ada]")
    assert "[Ada]" in _printed(session)


def test_reset_after_set_stays_reset(provider):
    session = _session(provider)
    commands.cmd_persona(session, "Ada")
    commands.cmd_persona_reset(session, "")
    assert _reload(session).persona_manager.current_persona is None


# -- /persona-learning -------------------------------------------------------


def test_turning_it_on_needs_a_persona(provider):
    session = _session(provider)
    memory_upkeep.cmd_persona_learning(session, "on")
    assert "no persona to grow" in _printed(session)
    assert not session.agent.persona_learning.enabled


def test_on_and_off_are_saved(provider):
    session = _session(provider, persona=Persona(name="Zed"))
    memory_upkeep.cmd_persona_learning(session, "on review")
    assert "on [review]" in _printed(session) and "undo [version]" in _printed(session)
    loaded = _reload(session)
    assert loaded.persona_learning.enabled
    assert loaded.persona_learning.auto_apply is False
    assert "waits for review" in _printed(session)

    memory_upkeep.cmd_persona_learning(session, "off")
    assert not _reload(session).persona_learning.enabled


def test_now_shows_the_evidence_and_applies_on_a_yes(provider, monkeypatch):
    session = _session(provider, persona=Persona(name="Zed"))
    _talk(session)
    session.agent.with_background_model(_Model(_bakery_change()))
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    memory_upkeep.cmd_persona_learning(session, "now")

    printed = _printed(session)
    assert BAKERY in printed
    assert "I run a small bakery" in printed
    assert "Persona is now version 2" in printed
    assert _reload(session).persona_manager.current_persona.goals.endswith(BAKERY)


def test_now_says_why_nothing_happened(provider):
    session = _session(provider, persona=Persona(name="Zed"))
    memory_upkeep.cmd_persona_learning(session, "now")
    assert "Not enough conversations yet" in _printed(session)


def test_a_no_leaves_the_change_waiting(provider, monkeypatch):
    session = _session(provider, persona=Persona(name="Zed"))
    _talk(session)
    session.agent.with_background_model(_Model(_bakery_change()))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    memory_upkeep.cmd_persona_learning(session, "now")

    assert "Left waiting" in _printed(session)
    assert session.agent.persona_learning_status()["pending"]
    memory_upkeep.cmd_persona_learning(session, "")
    assert "Waiting for review" in _printed(session)
    memory_upkeep.cmd_persona_learning(session, "approve")
    assert session.agent.persona_manager.current_persona.version == 2


def test_undo_from_the_chat(provider, monkeypatch):
    session = _session(provider, persona=Persona(name="Zed"))
    _talk(session)
    session.agent.with_background_model(_Model(_bakery_change()))
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    memory_upkeep.cmd_persona_learning(session, "now")
    memory_upkeep.cmd_persona_learning(session, "undo")

    assert "Undone" in _printed(session)
    assert not session.agent.persona_manager.current_persona.goals.endswith(BAKERY)


def test_nothing_to_undo_is_said_plainly(provider):
    session = _session(provider, persona=Persona(name="Zed"))
    memory_upkeep.cmd_persona_learning(session, "undo")
    assert "No background change to undo" in _printed(session)


def test_the_command_is_listed():
    assert "persona-learning" in commands.COMMANDS
