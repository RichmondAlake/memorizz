"""The Settings page exposes the forgetting mechanism's global defaults."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from dotenv import dotenv_values


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


@pytest.mark.unit
def test_settings_page_shows_the_forgetting_section(settings_client):
    html = settings_client.get("/settings").text
    assert "Forgetting mechanism" in html
    for env in (
        "MEMORIZZ_IMPORTANCE_RATER",
        "MEMORIZZ_RECENCY_DECAY_PER_HOUR",
        "MEMORIZZ_RECENCY_ANCHOR",
        "MEMORIZZ_ALPHA_IMPORTANCE",
        "MEMORIZZ_RETENTION_ENABLED",
        "MEMORIZZ_RETENTION_MIN_SCORE",
        "MEMORIZZ_REFLECTION_THRESHOLD",
    ):
        assert f'name="{env}"' in html, env


@pytest.mark.unit
def test_forgetting_defaults_are_saved_to_the_env_file(settings_client, tmp_path):
    response = settings_client.post(
        "/settings",
        data={
            "MEMORIZZ_RECENCY_DECAY_PER_HOUR": "0.99",
            "MEMORIZZ_IMPORTANCE_RATER": "llm",
            "MEMORIZZ_RETENTION_ENABLED": "true",
            "MEMORIZZ_RETENTION_GRACE_DAYS": "45",
        },
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code in (200, 302, 303), response.text[:300]
    env_files = list((tmp_path / "home").rglob(".env"))
    assert env_files, "no .env written under MEMORIZZ_HOME"
    saved = dotenv_values(env_files[0])
    assert saved["MEMORIZZ_RECENCY_DECAY_PER_HOUR"] == "0.99"
    assert saved["MEMORIZZ_IMPORTANCE_RATER"] == "llm"
    assert saved["MEMORIZZ_RETENTION_ENABLED"] == "true"
    assert saved["MEMORIZZ_RETENTION_GRACE_DAYS"] == "45"
