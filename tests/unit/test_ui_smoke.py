# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Whole-app smoke tests for the local UI.

Every user-facing page must render without a server error, both while
disconnected and against a real (filesystem) provider. This is the safety
net that lets app.py be refactored — route extractions and template edits
that break a page fail here instead of in someone's browser.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

# Pages that must render for a DISCONNECTED session (200, or a redirect to
# /connect — never a 5xx).
PAGES = [
    "/",
    "/connect",
    "/dashboard",
    "/settings",
    "/playground",
    "/agents",
    "/agents/new",
    "/automations",
    "/traces",
    "/observability",
    "/vercel-skills",
    "/evalground",
    "/api/status",
    "/api/personas",
    "/api/persona-presets",
    "/evalground/runs/active",
]

MEMORY_PAGES = [
    "personas",
    "toolbox",
    "skills",
    "conversations",
    "workflows",
    "knowledge-base",
    "short-term",
    "entity",
    "summaries",
    "shared",
    "cache",
]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def fs_provider(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "ui-smoke", lazy_vector_indexes=True
        )
    )
    yield provider


@pytest.mark.unit
@pytest.mark.parametrize("path", PAGES)
def test_page_renders_disconnected(client, path):
    with patch.dict(
        state._state,
        {"provider": None, "provider_type": None, "connection_info": {}},
    ):
        resp = client.get(path)
    assert resp.status_code < 500, f"{path} -> {resp.status_code}"


@pytest.mark.unit
@pytest.mark.parametrize("path", PAGES)
def test_page_renders_connected(client, fs_provider, path):
    with patch.dict(
        state._state,
        {
            "provider": fs_provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs_provider.root_path)},
        },
    ):
        resp = client.get(path)
    assert resp.status_code < 500, f"{path} -> {resp.status_code}"


@pytest.mark.unit
@pytest.mark.parametrize("slug", MEMORY_PAGES)
def test_memory_pages_render(client, fs_provider, slug):
    with patch.dict(
        state._state,
        {
            "provider": fs_provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(fs_provider.root_path)},
        },
    ):
        resp = client.get(f"/memory/{slug}")
    assert resp.status_code == 200, f"/memory/{slug} -> {resp.status_code}"


@pytest.mark.unit
def test_unknown_memory_type_is_404(client, fs_provider):
    with patch.dict(
        state._state,
        {"provider": fs_provider, "provider_type": "filesystem"},
    ):
        resp = client.get("/memory/nonexistent")
    assert resp.status_code == 404
