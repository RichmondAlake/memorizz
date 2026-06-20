# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Route tests for the HuggingFace model-cache UI endpoints.

The huggingface_hub SDK is never really called: it's faked via sys.modules so
the tests run identically whether or not the optional dependency is installed.
"""

import sys
import types
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def _repo(repo_id, repo_type="model", revs=("abc123",)):
    return types.SimpleNamespace(
        repo_id=repo_id,
        repo_type=repo_type,
        revisions=[types.SimpleNamespace(commit_hash=h) for h in revs],
    )


def _fake_hf(repos=None, snapshot_path="/tmp/hf/models--gpt2"):
    """Build fake huggingface_hub + huggingface_hub.constants modules."""
    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda repo_id, **k: snapshot_path

    class _Strategy:
        expected_freed_size = 123

        def execute(self):
            return None

    class _Cache:
        def __init__(self, repos):
            self.repos = repos

        def delete_revisions(self, *_):
            return _Strategy()

    hf.scan_cache_dir = lambda: _Cache(repos or [])

    consts = types.ModuleType("huggingface_hub.constants")
    consts.HF_HUB_CACHE = "/tmp/hf-cache"
    hf.constants = consts
    return {"huggingface_hub": hf, "huggingface_hub.constants": consts}


@pytest.mark.unit
def test_installed_no_sdk(client):
    with patch.dict(
        sys.modules, {"huggingface_hub": None, "huggingface_hub.constants": None}
    ):
        resp = client.get("/api/huggingface/installed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is False
    assert "not installed" in body["error"]


@pytest.mark.unit
def test_installed_with_models(client):
    repos = [
        _repo("gpt2"),
        _repo("some/dataset", repo_type="dataset"),
        _repo("bert-base"),
    ]
    with patch.dict(sys.modules, _fake_hf(repos)):
        resp = client.get("/api/huggingface/installed")
    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    # sorted, dataset repo excluded
    assert body["models"] == ["bert-base", "gpt2"]
    assert body["cache_dir"] == "/tmp/hf-cache"


@pytest.mark.unit
def test_pull_no_sdk(client):
    with patch.dict(sys.modules, {"huggingface_hub": None}):
        resp = client.post("/api/huggingface/pull", data={"repo_id": "gpt2"})
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


@pytest.mark.unit
def test_pull_success(client):
    with patch.dict(sys.modules, _fake_hf(snapshot_path="/tmp/hf/models--gpt2")):
        resp = client.post("/api/huggingface/pull", data={"repo_id": "gpt2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["path"] == "/tmp/hf/models--gpt2"


@pytest.mark.unit
def test_delete_not_found(client):
    with patch.dict(sys.modules, _fake_hf(repos=[])):
        resp = client.delete("/api/huggingface/models/gpt2")
    assert resp.status_code == 404
    assert resp.json()["ok"] is False


@pytest.mark.unit
def test_delete_success(client):
    with patch.dict(sys.modules, _fake_hf(repos=[_repo("gpt2")])):
        resp = client.delete("/api/huggingface/models/gpt2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["freed_bytes"] == 123
