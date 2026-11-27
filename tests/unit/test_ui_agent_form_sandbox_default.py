"""A new agent's form starts on a sandbox that can run, or on none."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def console(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_ENV_FILE", str(tmp_path / "home" / ".env"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    values = {
        "provider": provider,
        "provider_type": "filesystem",
        "connection_info": {"path": str(tmp_path / "memory")},
        "read_only": False,
    }
    try:
        with patch.dict(state._state, values):
            yield TestClient(create_app(), follow_redirects=False), provider
    finally:
        provider.close()


def _selected_sandbox(page: str) -> str:
    select = re.search(r'<select id="sandbox_provider".*?</select>', page, re.S).group(
        0
    )
    return re.search(r'<option value="([^"]*)"\s+selected', select).group(1)


def test_a_default_sandbox_that_is_not_set_up_is_not_preselected(console, monkeypatch):
    client, provider = console
    from memorizz.ui.routers import agents_crud

    monkeypatch.setenv("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "graalpy")
    monkeypatch.setattr(
        agents_crud,
        "_validate_sandbox_provider_choice",
        lambda value: "GraalPy needs a graalpy executable" if value else None,
    )

    page = client.get("/agents/new").text

    assert _selected_sandbox(page) == ""
    assert "The default sandbox (graalpy) is not set up" in page
    # The form as it loads can be saved.
    created = client.post(
        "/agents/new",
        data={
            "agent_name": "Plain",
            "instruction": "Be brief.",
            "llm_provider": "openai",
            "llm_model": "gpt-6.2-sol",
            "sandbox_provider": _selected_sandbox(page),
            "application_mode": "assistant",
        },
    )
    assert created.status_code in (302, 303), created.text[:300]
    assert [agent.name for agent in provider.list_memagents()] == ["Plain"]


def test_a_default_sandbox_that_can_run_is_preselected(console, monkeypatch):
    client, _ = console
    from memorizz.ui.routers import agents_crud

    monkeypatch.setenv("MEMORIZZ_DEFAULT_SANDBOX_PROVIDER", "graalpy")
    monkeypatch.setattr(
        agents_crud, "_validate_sandbox_provider_choice", lambda _: None
    )

    page = client.get("/agents/new").text

    assert _selected_sandbox(page) == "graalpy"
    assert "sandbox-default-note" not in page
