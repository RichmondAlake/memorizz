"""Learning control-plane page: plan, approve, review and reverse memory suppression."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from memorizz import FileSystemConfig, FileSystemProvider, LearningControlPlane
from memorizz.enums import MemoryType

DAY = 86_400.0


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from memorizz.memagent.models import MemAgentModel

    monkeypatch.setenv("MEMORIZZ_RETENTION_MIN_SCORE", "0.3")
    monkeypatch.setenv("MEMORIZZ_RETENTION_GRACE_DAYS", "30")
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "lcp", lazy_vector_indexes=True, use_faiss=False
        )
    )
    provider.store_memagent(
        MemAgentModel(
            agent_id="agent-1",
            name="Support agent",
            learning_control_plane=True,
            memory_ids=["kb-1"],
        )
    )
    provider.store(
        {
            "_id": "stale-fact",
            "name": "stale-fact",
            "content": "An old, never-recalled detail",
            "agent_id": "agent-1",
            "memory_id": "kb-1",
            "timestamp": time.time() - 120 * DAY,
            "importance": 0.1,
            "embedding": [1.0, 0.0],
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    plane = LearningControlPlane(
        provider,
        agent_id="agent-1",
        config={"enabled": True, "compile_async": False, "compile_every_n_events": 0},
    )
    plane.begin_run("hello", scope={"memory_id": "kb-1", "run_id": "run-1"})
    plane.close()
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


@pytest.mark.unit
def test_plan_approve_review_and_restore_round_trip(seeded):
    from memorizz.ui.routers import learning_control_plane as route

    client, patcher = _client(seeded)
    try:
        with patch.dict(route._last_actions, {}, clear=True):
            html = client.get(
                "/learning-control-plane?agent_id=agent-1&memory_id=kb-1"
            ).text
            assert "Plan memory retention (dry run)" in html
            assert 'action="/learning-control-plane/retention-plan"' in html

            response = client.post(
                "/learning-control-plane/retention-plan",
                data={
                    "agent_id": "agent-1",
                    "memory_id": "kb-1",
                    "user_id": "",
                    "thread_id": "",
                },
            )
            assert response.status_code in (302, 303)
            action = route._last_actions["agent-1"]
            assert action["kind"] == "forget-plan"
            assert action["report"]["plan_kind"] == "retention"
            assert action["report"]["candidate_count"] == 1
            plan_id = action["report"]["plan_id"]

            html = client.get(
                "/learning-control-plane?agent_id=agent-1&memory_id=kb-1"
            ).text
            assert "Approve reversible suppression" in html
            assert "Suppress 1 record(s)" in html
            assert f'name="plan_id" value="{plan_id}"' in html

            response = client.post(
                "/learning-control-plane/forget-apply",
                data={
                    "agent_id": "agent-1",
                    "plan_id": plan_id,
                    "approved_by": "operator-7",
                    "reason": "quarterly tidy",
                    "memory_id": "kb-1",
                    "user_id": "",
                    "thread_id": "",
                },
            )
            assert response.status_code in (302, 303)
            assert route._last_actions["agent-1"]["kind"] == "forget-apply"
            assert (
                "Suppressed 1 memory record(s)"
                in route._last_actions["agent-1"]["message"]
            )
            row = seeded.retrieve_by_id("stale-fact", MemoryType.KNOWLEDGE_BASE)
            assert row["retention_state"] == "suppressed"
            assert row["content"] == "An old, never-recalled detail"

            html = client.get(
                "/learning-control-plane?agent_id=agent-1&memory_id=kb-1"
            ).text
            assert "Suppressed memories" in html
            assert "stale-fact" in html and "operator-7" in html
            assert 'action="/learning-control-plane/unsuppress"' in html

            response = client.post(
                "/learning-control-plane/unsuppress",
                data={
                    "agent_id": "agent-1",
                    "record_id": "stale-fact",
                    "memory_type": "knowledge_base",
                    "approved_by": "operator-7",
                    "memory_id": "kb-1",
                    "user_id": "",
                    "thread_id": "",
                },
            )
            assert response.status_code in (302, 303)
            assert route._last_actions["agent-1"]["kind"] == "unsuppress"
            row = seeded.retrieve_by_id("stale-fact", MemoryType.KNOWLEDGE_BASE)
            assert row["retention_state"] == "active"
            html = client.get(
                "/learning-control-plane?agent_id=agent-1&memory_id=kb-1"
            ).text
            assert "Suppressed memories" not in html
    finally:
        patcher.stop()


@pytest.mark.unit
def test_artifact_forgetting_route_is_unchanged(seeded):
    from memorizz.ui.routers import learning_control_plane as route

    client, patcher = _client(seeded)
    try:
        with patch.dict(route._last_actions, {}, clear=True):
            response = client.post(
                "/learning-control-plane/forget-plan",
                data={
                    "agent_id": "agent-1",
                    "memory_id": "kb-1",
                    "user_id": "",
                    "thread_id": "",
                },
            )
            assert response.status_code in (302, 303)
            action = route._last_actions["agent-1"]
            assert action["kind"] == "forget-plan"
            assert "plan_kind" not in action["report"]
    finally:
        patcher.stop()
