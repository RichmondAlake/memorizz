"""/background-model and /consolidate in the CLI."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from rich.console import Console

from memorizz.cli import memory_upkeep
from memorizz.enums import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-cli-upkeep-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


class _Model:
    def __init__(self, name: str = "gemma4", facts: Optional[List[Dict]] = None):
        self.model = name
        self.facts = facts or []

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        if "durable facts" in prompt:
            return json.dumps({"facts": self.facts})
        return json.dumps({"retire": []})

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "ollama", "model": self.model}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None


def _session(work_dir: Path, model: _Model) -> SimpleNamespace:
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )
    agent = MemAgent(
        model=model,
        memory_provider=provider,
        memory_ids=["memory-1"],
        application_mode="assistant",
        auto_register=False,
        agent_id="agent-1",
    )
    agent.save()
    return SimpleNamespace(
        agent=agent,
        provider=provider,
        console=Console(record=True, width=200),
        memory_id="memory-1",
        thread_id="t1",
        user_id=None,
        llm_config={"provider": "ollama", "model": "gemma4"},
    )


def test_background_model_is_shown_set_saved_and_turned_off(work_dir, monkeypatch):
    monkeypatch.setattr(
        "memorizz.memagent.core.create_llm_provider",
        lambda config: _Model(config["model"]),
    )
    session = _session(work_dir, _Model())
    memory_upkeep.cmd_background_model(session, "")
    assert "the chat model" in session.console.export_text()

    memory_upkeep.cmd_background_model(session, "qwen2.5:3b")
    assert session.agent.background_model.model == "qwen2.5:3b"
    stored = session.provider.retrieve_memagent("agent-1")
    assert stored.background_llm_config == {"provider": "ollama", "model": "qwen2.5:3b"}

    memory_upkeep.cmd_background_model(session, "off")
    assert session.agent.background_model is session.agent.model
    assert session.provider.retrieve_memagent("agent-1").background_llm_config is None
    session.agent.close()


def test_consolidate_shows_the_plan_and_applies_it_on_yes(work_dir, monkeypatch):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Lisbon",
                "evidence": "I live in Lisbon",
            }
        ]
    )
    session = _session(work_dir, model)
    session.agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    memory_upkeep.cmd_consolidate(session, "")
    text = session.console.export_text()
    assert "add user.city = Lisbon" in text and "Nothing changed" in text
    record = EntityMemory(session.provider).get_entity_by_name(
        "user", memory_id="memory-1"
    )
    assert record is None

    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    memory_upkeep.cmd_consolidate(session, "")
    text = session.console.export_text()
    assert "Applied 1 changes" in text
    plan_id = text.split("/consolidate undo ")[-1].split()[0]
    record = EntityMemory(session.provider).get_entity_by_name(
        "user", memory_id="memory-1"
    )
    assert {a["name"]: a["value"] for a in record["attributes"]} == {"city": "Lisbon"}

    memory_upkeep.cmd_consolidate(session, f"undo {plan_id}")
    record = EntityMemory(session.provider).get_entity_by_name(
        "user", memory_id="memory-1"
    )
    assert record["attributes"] == []
    session.agent.close()


def test_a_tidy_memory_says_so(work_dir):
    session = _session(work_dir, _Model())
    memory_upkeep.cmd_consolidate(session, "")
    assert "nothing to change" in session.console.export_text()
    session.agent.close()
