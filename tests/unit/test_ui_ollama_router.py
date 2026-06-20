# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Route tests for the Ollama UI endpoints.

These were the first UI routes pulled out of the monolithic ui/app.py into a
dedicated APIRouter (memorizz.ui.routers.ollama). The tests pin the on-the-wire
behavior so the extraction (and future ones) can't silently change it. The
Ollama daemon is never contacted — urllib.request.urlopen is mocked.
"""

import json
import urllib.error
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui.app import create_app  # noqa: E402


class _FakeResp:
    """Minimal stand-in for the urlopen() context manager."""

    def __init__(self, data: bytes = b""):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.mark.unit
def test_installed_reachable(client):
    tags = json.dumps(
        {"models": [{"name": "llama3.1:8b"}, {"model": "qwen2.5:7b"}]}
    ).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResp(tags)):
        resp = client.get("/api/ollama/installed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is True
    assert "llama3.1:8b" in body["models"]
    assert "qwen2.5:7b" in body["models"]


@pytest.mark.unit
def test_installed_unreachable(client):
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        resp = client.get("/api/ollama/installed")
    assert resp.status_code == 200
    assert resp.json()["reachable"] is False


@pytest.mark.unit
def test_pull_success(client):
    ok = json.dumps({"status": "success"}).encode()
    with patch("urllib.request.urlopen", return_value=_FakeResp(ok)):
        resp = client.post("/api/ollama/pull", data={"name": "llama3.1:8b"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "message": "Pulled llama3.1:8b"}


@pytest.mark.unit
def test_pull_http_error(client):
    err = urllib.error.HTTPError("http://x/api/pull", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        resp = client.post("/api/ollama/pull", data={"name": "does-not-exist"})
    assert resp.status_code == 404
    assert resp.json()["ok"] is False


@pytest.mark.unit
def test_delete_success(client):
    with patch("urllib.request.urlopen", return_value=_FakeResp(b"")):
        resp = client.delete("/api/ollama/models/llama3.1:8b")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "message": "Removed llama3.1:8b"}
