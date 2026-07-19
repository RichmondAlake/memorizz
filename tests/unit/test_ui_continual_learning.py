# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Functional tests for the continual-learning UI: trajectory classes on
the workflows page and skill lifecycle actions on the skills page.

Promotion from the UI is human-*triggered*, never human-*exempted* — the
tests pin that the eligibility gates still apply to "Distill now".
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.long_term.procedural.workflow.canonicalization import (  # noqa: E402
    canonical_hash,
    canonical_signature,
)
from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

AGENT_ID = "cl-ui-agent"

DISTILLED_SKILL_MD = """---
name: Refund lookup and issue
description: Use when the user asks to refund a completed order.
preconditions:
  - Order identifier is present in the query
tools: [lookup_order, issue_refund]
---

# Refund lookup and issue

## When to apply
When the user requests a refund for an identified order.
When NOT to apply: pending orders.

## Procedure
1. Call `lookup_order` with the order identifier.
2. Call `issue_refund` with order_id and amount.
"""


class ScriptedLLM:
    def generate_text(self, prompt, instructions=None):
        if instructions and "YES or NO" in instructions:
            return "YES - matches the runs."
        return DISTILLED_SKILL_MD


def _steps(order_id):
    base = datetime.now()
    return {
        "Step 1: lookup_order": {
            "arguments": {"order_id": order_id},
            "result": "ok",
            "timestamp": base.isoformat(),
            "error": None,
        },
        "Step 2: issue_refund": {
            "arguments": {"order_id": order_id, "amount": 5},
            "result": "ok",
            "timestamp": (base + timedelta(seconds=1)).isoformat(),
            "error": None,
        },
    }


def _seed_workflows(provider, count, queries=None, promoted_skill_id=None):
    """Store `count` successful runs of the SAME canonical class."""
    stored_hash = None
    for index in range(count):
        steps = _steps(f"ID-{index}")
        signature = canonical_signature(steps)
        stored_hash = canonical_hash(signature)
        provider.store(
            {
                "name": "Tool Execution for Query",
                "workflow_id": f"wf-{index}",
                "agent_id": AGENT_ID,
                "user_query": (queries[index] if queries else f"refund order {index}"),
                "outcome": "success",
                "created_at": (
                    datetime.now() - timedelta(minutes=count - index)
                ).isoformat(),
                "steps": steps,
                "canonical_hash": stored_hash,
                "canonical_signature": signature,
                "step_count": len(signature),
                "promoted_skill_id": promoted_skill_id,
                "skills_activated": [],
                "embedding": [0.0],
            },
            memory_store_type=MemoryType.WORKFLOW_MEMORY,
        )
    return stored_hash


def _seed_skill(
    provider,
    status,
    source_hash="hash-x",
    skill_id="skill-1",
    injection_role="user",
):
    provider.store(
        {
            "skill_id": skill_id,
            "agent_id": AGENT_ID,
            "name": "Refund lookup and issue",
            "description": "Use when the user asks to refund a completed order.",
            "content": DISTILLED_SKILL_MD,
            "preconditions": ["Order identifier is present"],
            "tools_used": ["lookup_order", "issue_refund"],
            "queries": ["refund my order"],
            "status": status,
            "injection_role": injection_role,
            "version": 1,
            "source_canonical_hash": source_hash,
            "baseline": {"executions": 5, "success_rate": 1.0},
            "stats": {"activations": 3, "successes": 3, "failures": 0, "deviations": 0},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "embedding": [0.0],
        },
        memory_store_type=MemoryType.SKILLBOX,
    )


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "cl-ui", lazy_vector_indexes=True)
    )
    # Agent whose stored config drives manager construction from the UI.
    provider.store_memagent(
        MemAgentModel(
            agent_id=AGENT_ID,
            name="CL UI Agent",
            instruction="x",
            continual_learning=True,
            llm_config={"provider": "openai", "model": "gpt-4.1-mini"},
            continual_learning_config={"min_executions": 5, "min_distinct_queries": 2},
        )
    )
    # Tools must resolve by name for the distillation validation gate.
    for tool in ("lookup_order", "issue_refund"):
        provider.store(
            {"name": tool, "docstring": tool, "embedding": [0.0]},
            memory_store_type=MemoryType.TOOLBOX,
        )
    patcher = patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    )
    patcher.start()
    yield provider
    patcher.stop()


@pytest.fixture(autouse=True)
def _stable_skill_embeddings(monkeypatch):
    from memorizz.long_term.procedural.skillbox import skill as skill_mod

    monkeypatch.setattr(skill_mod, "get_embedding", lambda text, **k: [0.0])


class TestWorkflowClassesPage:
    @pytest.mark.unit
    def test_sidebar_exposes_continual_learning_navigation(self, client, connected):
        page = client.get("/memory/workflows").text

        assert "<span>Continual Learning</span>" in page
        assert '<a href="/memory/workflows"' in page
        assert "<span>Workflow Trajectories</span>" in page
        assert '<a href="/memory/skills"' in page
        assert "<span>Learned Skills</span>" in page

    @pytest.mark.unit
    def test_groups_runs_into_classes_with_gate_status(self, client, connected):
        _seed_workflows(connected, 5)
        resp = client.get("/memory/workflows")
        assert resp.status_code == 200
        page = resp.text
        assert "lookup_order → issue_refund" in page
        assert "5 runs" in page
        assert "✓ executions" in page
        assert "✓ success rate" in page
        assert "✓ query diversity" in page
        assert "✓ recency" in page
        assert "Distill now" in page
        assert "Run promotion cycle" in page

    @pytest.mark.unit
    def test_ineligible_class_shows_failing_gate_not_distill(self, client, connected):
        _seed_workflows(connected, 2)  # below min_executions=5
        page = client.get("/memory/workflows").text
        assert "✗ executions" in page
        assert "Distill now" not in page

    @pytest.mark.unit
    def test_promoted_class_shows_badge_and_suppressed_runs(self, client, connected):
        _seed_workflows(connected, 5, promoted_skill_id="skill-1")
        _seed_skill(connected, "active", skill_id="skill-1")
        page = client.get("/memory/workflows").text
        assert "PROMOTED" in page
        assert "suppressed" in page
        assert "Distill now" not in page


class TestPromotionActions:
    @pytest.mark.unit
    def test_distill_now_promotes_eligible_class(self, client, connected):
        stored_hash = _seed_workflows(connected, 5)
        with patch(
            "memorizz.ui.routers.continual_learning.create_llm_provider",
            return_value=ScriptedLLM(),
        ):
            resp = client.post(f"/continual-learning/{AGENT_ID}/distill/{stored_hash}")
        assert resp.status_code == 303

        skills = connected.list_all(memory_store_type=MemoryType.SKILLBOX)
        assert len(skills) == 1
        assert skills[0]["status"] == "active"
        assert skills[0]["source_canonical_hash"] == stored_hash

        # Report panel shows the promotion; source runs are now suppressed.
        page = client.get("/memory/workflows").text
        assert "Promoted skill" in page
        assert "PROMOTED" in page

    @pytest.mark.unit
    def test_distill_now_still_enforces_gates(self, client, connected):
        stored_hash = _seed_workflows(connected, 2)  # ineligible
        with patch(
            "memorizz.ui.routers.continual_learning.create_llm_provider",
            return_value=ScriptedLLM(),
        ):
            resp = client.post(f"/continual-learning/{AGENT_ID}/distill/{stored_hash}")
        assert resp.status_code == 303
        assert connected.list_all(memory_store_type=MemoryType.SKILLBOX) == []
        page = client.get("/memory/workflows").text
        assert "Skipped" in page and "executions" in page

    @pytest.mark.unit
    def test_cycle_without_llm_reports_gracefully(self, client, connected):
        _seed_workflows(connected, 5)
        with patch(
            "memorizz.ui.routers.continual_learning.create_llm_provider",
            side_effect=RuntimeError("no key"),
        ):
            resp = client.post(f"/continual-learning/{AGENT_ID}/cycle")
        assert resp.status_code == 303
        page = client.get("/memory/workflows").text
        assert "no LLM provider configured" in page


class TestSkillLifecycleActions:
    @pytest.mark.unit
    def test_skills_page_shows_status_and_actions(self, client, connected):
        _seed_skill(
            connected,
            "shadow",
            skill_id="skill-shadow",
            injection_role="developer",
        )
        page = client.get("/memory/skills").text
        assert "SHADOW" in page
        assert "authority: developer" in page
        assert "Activate as developer" in page
        assert "Demote" not in page  # only ACTIVE skills can be demoted
        assert "SKILL.md" in page

    @pytest.mark.unit
    def test_activate_shadow_skill_stamps_workflows(self, client, connected):
        stored_hash = _seed_workflows(connected, 5)
        _seed_skill(
            connected,
            "shadow",
            source_hash=stored_hash,
            skill_id="s-act",
            injection_role="developer",
        )

        resp = client.post("/continual-learning/skills/s-act/activate")
        assert resp.status_code == 303

        skill = connected.list_all(memory_store_type=MemoryType.SKILLBOX)[0]
        assert skill["status"] == "active"
        assert skill["injection_role"] == "developer"
        workflows = connected.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
        assert all(w.get("promoted_skill_id") == "s-act" for w in workflows)

    @pytest.mark.unit
    def test_demote_active_skill_releases_workflows(self, client, connected):
        stored_hash = _seed_workflows(connected, 5, promoted_skill_id="s-dem")
        _seed_skill(connected, "active", source_hash=stored_hash, skill_id="s-dem")

        resp = client.post("/continual-learning/skills/s-dem/demote")
        assert resp.status_code == 303

        skill = connected.list_all(memory_store_type=MemoryType.SKILLBOX)[0]
        assert skill["status"] == "demoted"
        assert "manually demoted" in (skill.get("demotion_reason") or "")
        workflows = connected.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
        assert all(not w.get("promoted_skill_id") for w in workflows)
