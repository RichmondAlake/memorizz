# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The page for an agent's own source: turning it on, choosing the coding
model, and reading versions and errors."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.memagent import self_repair
from memorizz.memagent.models import MemAgentModel
from memorizz.memagent.self_source import SelfSource
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.routers import self_source as page


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    provider.store_memagent(MemAgentModel(agent_id="agent-1", name="Vera"))
    page._jobs.clear()
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": False,
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)
    provider.close()


BASE = "/api/agents/agent-1/source"


@pytest.mark.unit
def test_an_agent_starts_without_a_copy_and_gets_one_on_request(client):
    assert client.get("/agents/agent-1/source").status_code == 200
    before = client.get(BASE).json()
    assert before["enabled"] is False and before["versions"] == []

    after = client.post(BASE + "/enable").json()

    assert after["enabled"] is True
    assert after["source"]["versions"] == 0
    assert [v["installed"] for v in after["versions"]] == [True]
    stored = state._state["provider"].retrieve_memagent("agent-1")
    assert stored.self_aware is True
    assert stored.self_aware_config["own_source"] is True


@pytest.mark.unit
def test_the_coding_model_is_saved_without_any_key(client):
    client.post(BASE + "/enable")

    saved = client.post(
        BASE + "/coding-model",
        json={"provider": "ollama", "model": "flint", "host": "http://10.0.0.5:11434"},
    ).json()

    assert saved["coding_model"] == {
        "provider": "ollama",
        "model": "flint",
        "host": "http://10.0.0.5:11434",
    }
    assert client.get(BASE).json()["coding_model"]["model"] == "flint"
    # Turning own source on earlier is not undone by choosing a model.
    assert client.get(BASE).json()["enabled"] is True
    bad = client.post(
        BASE + "/coding-model",
        json={"provider": "ollama", "model": "x", "host": "ftp://x"},
    )
    assert bad.status_code == 400
    cleared = client.post(
        BASE + "/coding-model", json={"provider": "ollama", "model": ""}
    )
    assert cleared.json()["coding_model"] is None


@pytest.mark.unit
def test_versions_errors_diffs_and_the_patch_are_served(client):
    client.post(BASE + "/enable")
    source = SelfSource("agent-1")
    source.edit_file(
        "memorizz/_env_io.py",
        'raw = os.environ.get("MEMORIZZ_HOME")',
        'raw = os.environ.get("MEMORIZZ_HOME") or None',
    )
    kept = source.save_version("Treat an empty home as unset")
    self_repair.ErrorLog("agent-1").record({"source": "exception", "error": "Boom"})

    data = client.get(BASE).json()
    assert [v["number"] for v in data["versions"]] == [1, 0]
    assert data["errors"][0]["error"] == "Boom"
    assert (
        "or None"
        in client.get(BASE + "/diff", params={"version": kept["id"]}).json()["diff"]
    )
    assert client.get(BASE + "/diff", params={"version": "; rm -rf"}).status_code == 400
    patch_file = client.get(BASE + "/patch")
    assert "attachment" in patch_file.headers["content-disposition"]
    assert "or None" in patch_file.text

    back = client.post(BASE + "/restore", json={"version": "installed"}).json()
    assert back["version"]["number"] == 2
    assert client.get(BASE + "/patch").text == ""


@pytest.mark.unit
def test_fixing_needs_a_copy_and_a_writable_store(client):
    assert client.post(BASE + "/recommend", json={}).status_code == 409
    assert client.post(BASE + "/fix", json={}).status_code == 409
    client.post(BASE + "/enable")
    with patch.dict(state._state, {"read_only": True}):
        assert client.post(BASE + "/fix", json={}).status_code == 403
        assert client.post(BASE + "/enable").status_code == 403


@pytest.mark.unit
def test_a_run_reports_its_steps_and_only_one_runs_at_a_time(client):
    client.post(BASE + "/enable")
    seen = {}

    class _Agent:
        def recommend_self_fix(self, focus, on_step=None):
            on_step("Reading its errors")
            seen["step"] = client.get(BASE).json()["job"]["step"]
            seen["busy"] = client.post(BASE + "/recommend", json={}).status_code
            return {"recommendation": "Problem: none", "failures": 0}

        def close(self, **_):
            pass

    with patch.object(page, "_live_agent", return_value=_Agent()):
        started = client.post(BASE + "/recommend", json={}).json()["job"]
        assert started["state"] == "running"
        for _ in range(200):
            job = client.get(BASE).json()["job"]
            if job["state"] != "running":
                break
            import time

            time.sleep(0.02)

    assert job["state"] == "done"
    assert job["result"]["recommendation"] == "Problem: none"
    assert seen == {"step": "Reading its errors", "busy": 409}


@pytest.mark.unit
def test_the_diff_page_gets_a_change_as_files_and_numbered_lines(client):
    client.post(BASE + "/enable")
    source = SelfSource("agent-1")
    source.edit_file(
        "memorizz/_env_io.py",
        "return Path(base).expanduser()",
        "return Path(base).expanduser().resolve()",
    )
    source.create_file("memorizz/_note.py", "NOTE = 1\n")

    assert client.get("/agents/agent-1/source/diff").status_code == 200
    waiting = client.get(BASE + "/diff?version=working").json()["files"]

    by_path = {file["path"]: file for file in waiting}
    assert by_path["memorizz/_note.py"]["status"] == "added"
    changed = by_path["memorizz/_env_io.py"]
    assert (changed["insertions"], changed["deletions"]) == (1, 1)
    lines = changed["hunks"][0]["lines"]
    removed = next(line for line in lines if line["kind"] == "del")
    added = next(line for line in lines if line["kind"] == "add")
    assert removed["text"].strip() == "return Path(base).expanduser()"
    assert added["text"].strip() == "return Path(base).expanduser().resolve()"
    # The old line and the new one sit at the same place in the file.
    assert removed["old"] == added["new"] and removed["new"] is None

    kept = source.save_version("Resolve the home path")
    one = client.get(BASE + f"/diff?version={kept['id']}").json()["files"]
    everything = client.get(BASE + "/diff?version=all").json()["files"]
    assert {file["path"] for file in one} == set(by_path)
    assert {file["path"] for file in everything} == set(by_path)
    assert client.get(BASE + "/diff?version=working").json()["files"] == []


@pytest.mark.unit
def test_a_whole_file_is_served_before_and_after_a_change(client):
    client.post(BASE + "/enable")
    source = SelfSource("agent-1")
    source.edit_file(
        "memorizz/_env_io.py",
        "return Path(base).expanduser()",
        "return Path(base).expanduser().resolve()",
    )
    kept = source.save_version("Resolve the home path")
    file_url = BASE + "/file?path=memorizz/_env_io.py"

    assert client.get("/agents/agent-1/source/file").status_code == 200
    after = client.get(file_url + f"&version={kept['id']}").json()
    before = client.get(file_url + f"&version={kept['id']}&side=old").json()
    since_install = client.get(file_url + "&version=all&side=old").json()

    (line,) = after["changed"]
    assert after["text"].splitlines()[line - 1].strip() == (
        "return Path(base).expanduser().resolve()"
    )
    assert before["changed"] == [line]
    assert before["text"].splitlines()[line - 1].strip() == (
        "return Path(base).expanduser()"
    )
    assert since_install["text"] == before["text"]
    # Nothing is waiting to be kept, so the current file has no marks.
    assert client.get(file_url).json()["changed"] == []

    outside = client.get(BASE + "/file?path=../../../etc/passwd")
    assert outside.status_code == 400
    assert client.get(BASE + "/file?path=memorizz/nope.py").status_code == 400
    assert client.get(file_url + "&version=HEAD;ls").status_code == 400


@pytest.mark.unit
def test_a_served_endpoint_gets_its_own_key_and_never_the_openai_one(
    client, monkeypatch
):
    from memorizz.memagent import MemAgent
    from memorizz.memagent.self_repair import SelfRepair

    client.post(BASE + "/enable")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    refused = client.post(
        BASE + "/coding-model",
        json={"provider": "openai", "model": "flint", "api_key_env": "sk-or-123 x"},
    )
    assert refused.status_code == 400

    saved = client.post(
        BASE + "/coding-model",
        json={
            "provider": "openai",
            "model": "recursive/flint-v1.0:free",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
    ).json()["coding_model"]
    assert saved == {
        "provider": "openai",
        "model": "recursive/flint-v1.0:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    }
    assert client.get(BASE).json()["coding_key_missing"] == "OPENROUTER_API_KEY"

    agent = MemAgent.load("agent-1", memory_provider=state._state["provider"])
    repair = SelfRepair(agent)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        repair.coding_model()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    model = repair.coding_model()
    assert model.client.api_key == "sk-or-test"
    assert str(model.client.base_url).startswith("https://openrouter.ai/api/v1")
    assert client.get(BASE).json()["coding_key_missing"] is None

    # No variable named: another host still never sees the OpenAI key.
    client.post(
        BASE + "/coding-model",
        json={
            "provider": "openai",
            "model": "flint",
            "base_url": "http://10.0.0.5:8000/v1",
        },
    )
    agent = MemAgent.load("agent-1", memory_provider=state._state["provider"])
    assert SelfRepair(agent).coding_model().client.api_key == "sk-local"


@pytest.mark.unit
def test_the_source_page_says_which_coding_model_keys_are_set(client, monkeypatch):
    client.post(BASE + "/enable")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    page_html = client.get("/agents/agent-1/source").text
    assert 'id="cm-option-openrouter"' in page_html
    assert 'id="cm-option-own"' in page_html

    keys = client.get(BASE).json()["coding_keys"]
    assert keys["openrouter"] is True and keys["anthropic"] is False

    client.post(
        BASE + "/coding-model",
        json={"provider": "anthropic", "model": "claude-sonnet-5-5"},
    )
    assert client.get(BASE).json()["coding_key_missing"] == "ANTHROPIC_API_KEY"
