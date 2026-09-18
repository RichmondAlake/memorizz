"""UI settings remain secret-free and truthful about saved versus active state."""

import os
from unittest.mock import patch

import pytest
from dotenv import dotenv_values

from memorizz import _env_io as env


@pytest.fixture
def settings_client(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app
    from memorizz.ui.state import _state

    monkeypatch.chdir(tmp_path)
    with patch.dict(
        os.environ, {"MEMORIZZ_HOME": str(tmp_path / "home")}, clear=True
    ), patch.dict(
        _state,
        {"provider": object(), "provider_type": "filesystem", "connection_info": {}},
    ):
        yield TestClient(create_app(), follow_redirects=False)


def test_ui_displays_dynamic_path_and_does_not_prefill_tokens(
    settings_client, tmp_path
):
    from memorizz.ui.app import _build_settings_sections

    with patch.dict(
        os.environ,
        {
            "NOTION_TOKEN": "private-ui-token",
            "MEMORIZZ_ENV_FILE": str(tmp_path / "custom.env"),
        },
    ):
        response = settings_client.get("/settings")
        assert response.status_code == 200
        assert str(tmp_path / "custom.env") in response.text
        assert "Active UI memory provider" in response.text
        assert "does not reconnect" in response.text
        assert "memorizz notion connect" in response.text
        assert "private-ui-token" not in response.text
        fields = [
            field
            for section in _build_settings_sections()
            for field in section["fields"]
        ]
        token = next(field for field in fields if field["env"] == "NOTION_TOKEN")
        assert token["current_value"] == "" and token["is_set"] is True


def test_ui_warns_about_project_shadow_and_does_not_switch_provider(
    settings_client, tmp_path
):
    from memorizz.ui.state import _state

    provider = _state["provider"]
    env.update_env_file(tmp_path / ".env", {"MEMORIZZ_BACKEND": "filesystem"})
    response = settings_client.post(
        "/settings",
        data={"MEMORIZZ_BACKEND": "notion", "NOTION_TOKEN": "private-ui-token"},
    )
    assert response.status_code == 200
    assert "project .env takes precedence" in response.text
    assert "restart CLI/MCP processes" in response.text
    assert "private-ui-token" not in response.text
    assert dotenv_values(env.resolve_env_file())["NOTION_TOKEN"] == "private-ui-token"
    assert _state["provider"] is provider
    assert _state["provider_type"] == "filesystem"


def test_ui_failed_save_does_not_claim_persistence(settings_client):
    with patch(
        "memorizz._env_io.update_env_file", side_effect=OSError("private-ui-token")
    ):
        response = settings_client.post(
            "/settings", data={"NOTION_TOKEN": "private-ui-token"}
        )
    assert "NOT saved" in response.text
    assert "private-ui-token" not in response.text
    assert not env.resolve_env_file().exists()


@pytest.mark.parametrize(
    "data",
    [
        {"NOTION_TOKEN": "private\0token"},
        {"MEMORIZZ_BACKEND": "private-bad-value"},
        {"MEMORIZZ_NOTION_DATA_SOURCE_ID": "private-bad-value"},
    ],
)
def test_ui_invalid_updates_are_not_applied_or_saved(settings_client, data):
    response = settings_client.post("/settings", data=data)
    assert response.status_code == 200
    assert not env.resolve_env_file().exists()
    assert all(key not in os.environ for key in data)
    assert all(value not in response.text for value in data.values())
