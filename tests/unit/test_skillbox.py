# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Unit tests for the Skill model and Skillbox store."""

import hashlib
import re
from pathlib import Path
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.skillbox import Skill, Skillbox, SkillStatus
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def stable_embedding(text: str, dims: int = 16) -> List[float]:
    """Deterministic bag-of-words embedding for retrieval tests."""
    vector = [0.0] * dims
    for token in re.findall(r"[a-z]+", str(text).lower()):
        digest = int(hashlib.sha1(token.encode()).hexdigest(), 16)
        vector[digest % dims] += 1.0
    norm = sum(component**2 for component in vector) ** 0.5 or 1.0
    return [component / norm for component in vector]


class StableEmbeddingProvider:
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        return stable_embedding(text)

    def get_provider_info(self):
        return {"provider": "stable-test", "model": "bow", "dimensions": 16}


@pytest.fixture()
def patched_embeddings(monkeypatch):
    from memorizz.long_term.procedural.skillbox import skill as skill_mod
    from memorizz.long_term.procedural.skillbox import skillbox as skillbox_mod

    monkeypatch.setattr(skill_mod, "get_embedding", stable_embedding)
    monkeypatch.setattr(skillbox_mod, "get_embedding", stable_embedding)
    return stable_embedding


@pytest.fixture()
def fs_provider(tmp_path):
    config = FileSystemConfig(
        root_path=Path(tmp_path) / "skillbox-store",
        embedding_provider=StableEmbeddingProvider(),
        lazy_vector_indexes=True,
    )
    return FileSystemProvider(config)


def _skill(name="Refund lookup and issue", **overrides) -> Skill:
    defaults = dict(
        name=name,
        description="Use when the user asks to refund a completed order.",
        content=(
            "---\nname: x\n---\n# x\n\n## Procedure\n"
            "1. Call `lookup_order` with the order identifier.\n"
        ),
        preconditions=["Order identifier present"],
        tools_used=["lookup_order", "issue_refund"],
        queries=["refund my order", "give me my money back"],
        agent_id="agent-1",
        source_canonical_hash="hash-1",
        status=SkillStatus.ACTIVE,
        baseline={"executions": 6, "success_rate": 0.83},
    )
    defaults.update(overrides)
    return Skill(**defaults)


class TestSkillModel:
    def test_embedding_input_is_applicability_only(self, patched_embeddings):
        """The single most likely silent-quality bug in the system: the
        skill embedding must encode WHEN to apply, never HOW it was done.
        """
        skill = _skill()
        embedding_input = skill.embedding_input()
        assert skill.description in embedding_input
        assert "Order identifier present" in embedding_input
        assert "refund my order" in embedding_input
        # Mechanism must be absent: no content, no procedure, no tool list.
        assert "Procedure" not in embedding_input
        assert "lookup_order" not in embedding_input
        assert skill.content not in embedding_input

    def test_round_trip_serialization(self, patched_embeddings):
        skill = _skill()
        clone = Skill.from_dict(skill.to_dict())
        assert clone.skill_id == skill.skill_id
        assert clone.status == SkillStatus.ACTIVE
        assert clone.preconditions == skill.preconditions
        assert clone.tools_used == skill.tools_used
        assert clone.embedding == skill.embedding
        assert clone.baseline == skill.baseline

    def test_loading_without_embedding_never_regenerates(self, monkeypatch):
        from memorizz.long_term.procedural.skillbox import skill as skill_mod

        def _explode(*_args, **_kwargs):
            raise AssertionError("embedding API must not be called on load")

        monkeypatch.setattr(skill_mod, "get_embedding", _explode)
        document = {
            "name": "n",
            "description": "d",
            "content": "c",
            "status": "active",
        }
        loaded = Skill.from_dict(document)
        assert loaded.embedding is None

    def test_default_stats_shape(self, patched_embeddings):
        skill = _skill()
        assert skill.stats["activations"] == 0
        assert skill.stats["deviations"] == 0
        assert skill.stats["recent_outcomes"] == []


class TestSkillboxStore:
    def test_round_trip_through_filesystem_provider(
        self, fs_provider, patched_embeddings
    ):
        box = Skillbox(fs_provider, agent_id="agent-1")
        skill = _skill()
        box.add_skill(skill)

        by_id = box.get_skill_by_id(skill.skill_id)
        assert by_id is not None and by_id.name == skill.name
        by_name = box.get_skill_by_name(skill.name)
        assert by_name is not None and by_name.skill_id == skill.skill_id

    def test_status_transition_writes_demotion_audit(
        self, fs_provider, patched_embeddings
    ):
        box = Skillbox(fs_provider, agent_id="agent-1")
        skill = _skill()
        box.add_skill(skill)

        assert box.set_status(skill.skill_id, SkillStatus.DEMOTED, "drift")
        demoted = box.get_skill_by_id(skill.skill_id)
        assert demoted.status == SkillStatus.DEMOTED
        assert demoted.demotion_reason == "drift"
        assert demoted.demoted_at is not None

    def test_update_skill_preserves_stored_embedding(
        self, fs_provider, patched_embeddings
    ):
        """Loaded skills often have embedding=None (providers exclude the
        vector); a full-document write-back would destroy retrieval."""
        box = Skillbox(fs_provider, agent_id="agent-1")
        skill = _skill()
        box.add_skill(skill)

        loaded = box.get_skill_by_id(skill.skill_id)
        loaded.embedding = None  # simulate an embedding-excluding projection
        loaded.stats["activations"] = 3
        assert box.update_skill(loaded)

        raw = fs_provider.list_all(memory_store_type=MemoryType.SKILLBOX)
        stored = next(d for d in raw if d.get("skill_id") == skill.skill_id)
        assert stored.get("embedding"), "embedding must survive stat updates"
        assert stored["stats"]["activations"] == 3

    def test_get_active_skill_for_hash_and_cache_invalidation(
        self, fs_provider, patched_embeddings
    ):
        box = Skillbox(fs_provider, agent_id="agent-1")
        skill = _skill()
        box.add_skill(skill)

        assert box.get_active_skill_for_hash("hash-1").skill_id == skill.skill_id
        assert box.get_active_skill_for_hash("other-hash") is None

        box.set_status(skill.skill_id, SkillStatus.DEMOTED, "drift")
        assert box.get_active_skill_for_hash("hash-1") is None

    def test_retrieval_is_scored_and_status_filtered(
        self, fs_provider, patched_embeddings
    ):
        box = Skillbox(fs_provider, agent_id="agent-1")
        refund = _skill()
        calendar = _skill(
            name="Calendar scheduling",
            description="Use when the user wants to schedule a meeting.",
            queries=["schedule a meeting", "book time on my calendar"],
            source_canonical_hash="hash-2",
        )
        demoted = _skill(
            name="Old refund flow",
            status=SkillStatus.DEMOTED,
            source_canonical_hash="hash-3",
        )
        for entry in (refund, calendar, demoted):
            box.add_skill(entry)

        hits = box.retrieve_skills_by_query(
            "please refund my order", limit=2, min_similarity=0.3
        )
        assert hits, "expected at least one scored hit"
        assert hits[0].skill.name == refund.name
        assert all(hit.skill.status == SkillStatus.ACTIVE for hit in hits)
        assert all(0.0 <= hit.similarity <= 1.0001 for hit in hits)

    def test_min_similarity_gate_blocks_weak_matches(
        self, fs_provider, patched_embeddings
    ):
        box = Skillbox(fs_provider, agent_id="agent-1")
        box.add_skill(_skill())
        hits = box.retrieve_skills_by_query(
            "completely unrelated astrophysics question zzz",
            limit=2,
            min_similarity=0.95,
        )
        assert hits == []

    def test_agent_scoping(self, fs_provider, patched_embeddings):
        theirs = Skillbox(fs_provider, agent_id="agent-2")
        theirs.add_skill(_skill(agent_id="agent-2"))
        mine = Skillbox(fs_provider, agent_id="agent-1")
        assert mine.list_skills() == []
