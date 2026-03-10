"""Unit tests for UI self-awareness helper functions."""

import pytest

from memorizz.memagent.models import MemAgentModel
from memorizz.ui.app import (
    _build_agent_form_data,
    _build_self_aware_config,
    _parse_self_aware_root_paths,
    _validate_self_aware_config,
)


@pytest.mark.unit
def test_parse_self_aware_root_paths_handles_newlines_commas_and_duplicates():
    parsed = _parse_self_aware_root_paths(" /repo \n/repo\nsrc, src , ./pkg ")
    assert parsed == ["/repo", "src", "./pkg"]


@pytest.mark.unit
def test_build_self_aware_config_enforces_delete_dependency():
    cfg = _build_self_aware_config(
        root_paths=["/repo", "/repo", ""],
        allow_writes=False,
        allow_deletes=True,
    )
    assert cfg["root_paths"] == ["/repo"]
    assert cfg["allow_writes"] is False
    assert cfg["allow_deletes"] is False
    assert cfg["policy_version"] == "v1"


@pytest.mark.unit
def test_validate_self_aware_config_rejects_invalid_root_entries():
    error = _validate_self_aware_config(
        {"root_paths": ["ok", ""], "allow_writes": True}
    )
    assert error is not None
    assert "root path" in error.lower()


@pytest.mark.unit
def test_build_agent_form_data_includes_self_aware_fields():
    agent = MemAgentModel(
        instruction="test",
        self_aware=True,
        self_aware_config={
            "root_paths": ["/repo", "src"],
            "allow_writes": True,
            "allow_deletes": False,
        },
    )

    form_data = _build_agent_form_data(agent)
    assert form_data["self_aware"] is True
    assert form_data["self_aware_root_paths"] == "/repo\nsrc"
    assert form_data["self_aware_allow_writes"] is True
    assert form_data["self_aware_allow_deletes"] is False
