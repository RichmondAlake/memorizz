"""Per-agent Forgetting overrides are editable in the agent form and persist on the policy."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers.agents_crud import (
    _forgetting_form_values,
    _retrieval_policy_with_forgetting,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "ui-forgetting", lazy_vector_indexes=True
        )
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


def _form(**overrides):
    form = {
        "agent_name": "Forgetting Agent",
        "instruction": "You remember things.",
        "application_mode": "assistant",
        "max_steps": "10",
        "tool_access": "private",
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
    }
    form.update(overrides)
    return form


@pytest.mark.unit
def test_policy_merge_keeps_only_provided_overrides():
    merged = _retrieval_policy_with_forgetting(
        {"conversation_scope": "memory"},
        {
            "scoring_alpha_importance": "2",
            "scoring_recency_anchor": "created",
            "retention_enabled": "on",
            "retention_min_retention": "0.2",
            "scoring_alpha_recency": "",
            "retention_grace_days": "abc",
        },
    )
    assert merged["conversation_scope"] == "memory"
    assert merged["scoring"] == {"alpha_importance": 2.0, "recency_anchor": "created"}
    assert merged["retention"] == {"enabled": True, "min_retention": 0.2}
    # Blank everything -> override keys removed (inherit global).
    cleared = _retrieval_policy_with_forgetting(merged, {})
    assert "scoring" not in cleared and "retention" not in cleared
    assert _retrieval_policy_with_forgetting(None, {}) is None
    values = _forgetting_form_values(merged)
    assert (
        values["scoring_alpha_importance"] == "2.0"
        and values["retention_enabled"] == "on"
    )
    assert values["scoring_alpha_recency"] == ""


@pytest.mark.unit
def test_create_and_edit_round_trip_forgetting_overrides(client, connected):
    response = client.post(
        "/agents/new",
        data=_form(
            forgetting_scoring_alpha_importance="2",
            forgetting_scoring_recency_anchor="created",
            forgetting_retention_enabled="on",
            forgetting_retention_min_retention="0.2",
            forgetting_retention_grace_days="45",
        ),
    )
    assert response.status_code in (302, 303), response.text[:300]
    agent = connected.list_memagents()[0]
    policy = agent.retrieval_policy
    assert policy["scoring"] == {"alpha_importance": 2.0, "recency_anchor": "created"}
    assert policy["retention"] == {
        "enabled": True,
        "min_retention": 0.2,
        "grace_days": 45.0,
    }

    html = client.get(f"/agents/{agent.agent_id}/edit").text
    assert "Forgetting (per-agent overrides)" in html
    assert 'name="forgetting_scoring_alpha_importance" value="2.0"' in html
    assert 'value="created" selected' in html

    response = client.post(
        f"/agents/{agent.agent_id}/edit",
        data=_form(forgetting_scoring_alpha_relevance="0.5"),
    )
    assert response.status_code in (302, 303), response.text[:300]
    agent = connected.retrieve_memagent(agent.agent_id)
    assert agent.retrieval_policy["scoring"] == {"alpha_relevance": 0.5}
    assert "retention" not in agent.retrieval_policy


@pytest.mark.unit
def test_new_agent_form_renders_the_group_with_inherit_defaults(client, connected):
    html = client.get("/agents/new").text
    assert "Forgetting (per-agent overrides)" in html
    assert 'name="forgetting_scoring_recency_decay_per_hour" value=""' in html
