# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Every console page shares one header, one title and the shared stylesheet.

Pages drifted when each built its own header (all-caps eyebrows, boxed
heroes, marketing headlines). The shared ``page_header`` macro in
``_ui.html`` keeps them uniform; this guards against new pages opting out.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

CONSOLE_PAGES = [
    "/dashboard",
    "/agents",
    "/agents/new",
    "/playground",
    "/settings",
    "/automations",
    "/traces",
    "/traces/health",
    "/traces/compare",
    "/evalground",
    "/evalground/compare",
    "/harnesses",
    "/mcp",
    "/vercel-skills",
    "/learning-control-plane",
    "/persona-evolution",
    "/memory/conversations",
    "/memory/workflows",
    "/memory/skills",
]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield provider


@pytest.mark.unit
@pytest.mark.parametrize("path", CONSOLE_PAGES)
def test_console_pages_share_one_header_and_stylesheet(client, connected, path):
    response = client.get(path)
    assert response.status_code == 200, f"{path} -> {response.status_code}"
    html = response.text
    assert html.count('class="page-header"') == 1, path
    assert 'class="page-header-text"' in html, path
    assert len(re.findall(r"<h1[\s>]", html)) == 1, path
    assert "/static/css/console.css" in html, path
    for crumb in re.findall(r'<span class="page-eyebrow">([^<]+)</span>', html):
        # Breadcrumbs read in sentence case, never as shouted labels.
        assert crumb != crumb.upper(), f"{path}: all-caps breadcrumb {crumb!r}"
