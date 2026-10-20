"""Learning control-plane page: plan, apply and undo memory consolidation,
and the playground's memory overview."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

from memorizz import FileSystemConfig, FileSystemProvider
from memorizz.enums import MemoryType

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Model:
    model = "background"

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        self.prompts.append(prompt)
        if "durable facts" in prompt:
            return json.dumps(
                {
                    "facts": [
                        {
                            "entity": "user",
                            "attribute": "city",
                            "value": "Lisbon",
                            "evidence": "I moved to Lisbon",
                        }
                    ]
                }
            )
        return json.dumps({"same_fact": [], "same_entity": []})

    def get_config(self):
        return {"provider": "ollama", "model": "background"}


@pytest.fixture()
def seeded(tmp_path):
    pytest.importorskip("fastapi")
    from memorizz.memagent.models import MemAgentModel

    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "consolidation-ui",
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
            background_llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
            memory_consolidation={"enabled": True, "every_n_turns": 5},
        )
    )
    for entity_id, attribute in (("ada-1", "role"), ("ada-2", "known_for")):
        provider.store(
            data={
                "_id": entity_id,
                "entity_id": entity_id,
                "name": "Ada Lovelace",
                "entity_type": "person",
                "attributes": [{"name": attribute, "value": "x"}],
                "relations": [],
                "metadata": {},
                "memory_id": "memory-1",
                "user_id": None,
                "created_at": "2026-09-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
            },
            memory_store_type=MemoryType.ENTITY_MEMORY,
        )
    for record_id, when in (("s-old", 1.0), ("s-new", 2.0)):
        provider.store(
            data={
                "_id": record_id,
                "memory_id": "memory-1",
                "agent_id": "agent-1",
                "thread_id": "t1",
                "user_id": None,
                "content": "Same summary.",
                "created_at": when,
            },
            memory_store_type=MemoryType.SUMMARIES,
        )
    for role, text in (
        ("user", "I moved to Lisbon last month."),
        ("assistant", "Welcome!"),
    ):
        provider.store(
            data={
                "role": role,
                "content": text,
                "memory_id": "memory-1",
                "thread_id": "t1",
                "user_id": None,
                "agent_id": "agent-1",
                "timestamp": "2026-10-01T10:00:00",
            },
            memory_store_type=MemoryType.CONVERSATION_MEMORY,
        )
    return provider


def _client(provider):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    patcher = patch.dict(
        state._state,
        {"provider": provider, "provider_type": "filesystem", "connection_info": {}},
    )
    patcher.start()
    return TestClient(create_app(), follow_redirects=False), patcher


def _scope(**extra):
    return {
        "agent_id": "agent-1",
        "memory_id": "",
        "user_id": "",
        "thread_id": "",
        **extra,
    }


def test_plan_apply_and_undo_consolidation_from_the_page(seeded):
    from memorizz.long_term.semantic.entity_memory import EntityMemory
    from memorizz.ui.routers import learning_control_plane as route

    model = _Model()
    client, patcher = _client(seeded)
    try:
        with patch.dict(route._last_actions, {}, clear=True), patch(
            "memorizz.learning.consolidation.model_for_agent", lambda record: model
        ):
            html = client.get("/learning-control-plane?agent_id=agent-1").text
            assert "Memory consolidation" in html
            assert "ollama/qwen2.5:7b" in html and "background model" in html
            assert "Every 5 turns" in html
            assert 'value="memory-1"' in html

            response = client.post(
                "/learning-control-plane/consolidation-plan",
                data=_scope(consolidation_memory_id="memory-1"),
            )
            assert response.status_code in (302, 303)
            action = route._last_actions["agent-1"]
            assert action["kind"] == "consolidation-plan", action
            report = action["report"]
            assert report["thread_ids"] == ["t1"]  # the most recent conversations
            assert report["counts"] == {
                "merge_entities": 1,
                "suppress_duplicate": 1,
                "retire_attribute": 0,
                "add_fact": 1,
            }

            html = client.get("/learning-control-plane?agent_id=agent-1").text
            assert "Apply 3 change(s)" in html
            assert "Ada Lovelace" in html and "user.city" in html
            assert "“I moved to Lisbon”" in html
            assert "Add fact" in html and "Update fact" not in html

            response = client.post(
                "/learning-control-plane/consolidation-apply",
                data=_scope(plan_id=report["plan_id"], approved_by="operator-7"),
            )
            assert response.status_code in (302, 303)
            applied = route._last_actions["agent-1"]
            assert applied["kind"] == "consolidation-applied", applied
            assert applied["report"]["applied"]["add_fact"] == 1
            entities = EntityMemory(seeded)
            assert len(entities.list_entities(memory_id="memory-1")) == 2  # Ada + user
            assert (
                seeded.retrieve_by_id("s-new", MemoryType.SUMMARIES)["retention_state"]
                == "suppressed"
            )

            html = client.get("/learning-control-plane?agent_id=agent-1").text
            assert "Applied plans" in html and report["plan_id"][:24] in html
            assert 'action="/learning-control-plane/consolidation-restore"' in html

            response = client.post(
                "/learning-control-plane/consolidation-restore",
                data=_scope(
                    plan_id=report["plan_id"],
                    approved_by="operator-7",
                    consolidation_memory_id="memory-1",
                ),
            )
            assert route._last_actions["agent-1"]["kind"] == "consolidation-restored"
            names = sorted(
                record["name"]
                for record in entities.list_entities(memory_id="memory-1")
            )
            assert names == ["Ada Lovelace", "Ada Lovelace", "user"]
            assert (
                seeded.retrieve_by_id("s-new", MemoryType.SUMMARIES)["retention_state"]
                == "active"
            )
            html = client.get("/learning-control-plane?agent_id=agent-1").text
            assert "Undone" in html

            # The timeline labels the changes.
            history = client.get(
                "/traces/memory-history?agent_id=agent-1&memory_id=memory-1"
            ).json()
            labels = {
                event["label"]
                for event in history["events"]
                if str(event.get("source", "")).startswith("consolidation:")
            }
            assert "Merged duplicates of Ada Lovelace" in labels
            assert "Added user.city" in labels
            assert "Undid memory consolidation" in labels
    finally:
        patcher.stop()


def test_a_scheduled_plan_waiting_for_review_opens_from_the_page(seeded):
    from memorizz.learning.consolidation import MemoryConsolidator
    from memorizz.memagent.core import _shared_upkeep
    from memorizz.ui.routers import learning_control_plane as route

    plan = MemoryConsolidator(seeded, agent_id="agent-1").plan(memory_id="memory-1")
    pending = _shared_upkeep(seeded, "agent-1").plans
    pending[plan.plan_id] = plan
    client, patcher = _client(seeded)
    try:
        with patch.dict(route._last_actions, {}, clear=True):
            html = client.get("/learning-control-plane?agent_id=agent-1").text
            assert "Waiting for review" in html
            client.post(
                "/learning-control-plane/consolidation-review",
                data=_scope(plan_id=plan.plan_id),
            )
            assert route._last_actions["agent-1"]["report"]["plan_id"] == plan.plan_id
            client.post(
                "/learning-control-plane/consolidation-apply",
                data=_scope(plan_id=plan.plan_id, approved_by="operator-7"),
            )
            assert plan.plan_id not in pending
    finally:
        patcher.stop()


def test_the_playground_memory_overview(seeded, monkeypatch):
    monkeypatch.setattr(
        "memorizz.memagent.core.create_llm_provider", lambda config: _Model()
    )
    client, patcher = _client(seeded)
    try:
        data = client.get("/api/agents/agent-1/memory-overview").json()
        assert "entity_memory" in data["memory_types"]
        assert data["entity_memory"] is True and data["sealed"] is False
        assert data["background_model"] == {"provider": "ollama", "model": "qwen2.5:7b"}
        assert data["memory_consolidation"]["every_n_turns"] == 5
        html = client.get("/agents/agent-1/playground").text
        assert 'id="pg-memory-overview"' in html and "How this agent remembers" in html
    finally:
        patcher.stop()
