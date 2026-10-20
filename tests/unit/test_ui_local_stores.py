"""The connect page finds the FileSystem stores on this computer."""

import json
import os
from unittest.mock import patch

import pytest

from memorizz.ui.local_stores import detect_local_stores

pytestmark = pytest.mark.unit


def _store(root, agents):
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "index.json").write_text("{}")
    for index in range(agents):
        (root / "agents" / f"agent-{index}.json").write_text(json.dumps({}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORIZZ_MEMORY_ROOT", raising=False)
    return tmp_path


def test_the_clis_store_comes_first_then_other_stores_with_agent_counts(home):
    _store(home / "memory", 2)
    _store(home / "data", 0)
    (home / "audit").mkdir()  # not a store
    stores = detect_local_stores()
    assert [(s["path"], s["agents"], s["exists"], s["default"]) for s in stores] == [
        (str((home / "memory").resolve()), 2, True, True),
        (str((home / "data").resolve()), 0, True, False),
    ]


def test_a_store_named_by_the_environment_is_the_default_even_before_it_exists(
    home, monkeypatch
):
    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(home / "elsewhere"))
    _store(home / "memory", 1)
    stores = detect_local_stores()
    assert stores[0]["path"] == str((home / "elsewhere").resolve())
    assert stores[0]["default"] and not stores[0]["exists"]
    assert stores[1]["agents"] == 1 and not stores[1]["default"]


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    with patch.dict(
        state._state, {"provider": None, "provider_type": None, "connection_info": {}}
    ):
        yield TestClient(create_app())


def test_the_connect_page_starts_on_the_store_with_agents(home, client):
    _store(home / "memory", 3)
    _store(home / "data", 0)
    unset = {"MEMORIZZ_BACKEND": "", "MONGODB_URI": "", "ORACLE_USER": ""}
    with patch.dict(os.environ, unset):
        page = client.get("/connect").text
    assert '<option value="filesystem" selected>' in page
    memory = str((home / "memory").resolve())
    assert f'id="filesystem_path" placeholder="{memory}" value="{memory}"' in page
    assert "3 agents · the CLI's store" in page
    assert f'data-path="{(home / "data").resolve()}"' in page and "empty" in page


def test_a_database_in_the_environment_still_comes_first(home, client):
    _store(home / "memory", 3)
    with patch.dict(
        os.environ, {"MEMORIZZ_BACKEND": "", "MONGODB_URI": "mongodb://db.test"}
    ):
        page = client.get("/connect").text
    assert '<option value="mongodb" selected>' in page
    assert "FileSystem stores on this computer" in page and "3 agents" in page


def test_without_local_agents_the_connect_page_keeps_its_old_choice(home, client):
    unset = {"MEMORIZZ_BACKEND": "", "MONGODB_URI": "", "ORACLE_USER": ""}
    with patch.dict(os.environ, unset):
        page = client.get("/connect").text
    assert '<option value="oracle" selected>' in page
    assert "not created yet" in page
