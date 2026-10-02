"""Agent form: delegates (other saved agents an agent hands work to)."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz import MemAgent  # noqa: E402
from memorizz.memagent.models import MemAgentModel  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402

pytestmark = pytest.mark.unit

LLM = {"provider": "openai", "model": "gpt-4.1-mini", "api_key": "sk-test"}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "store", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": store,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(store.root_path)},
        },
    ):
        yield store


def _agent(store, name, delegates=None):
    model = MemAgentModel(
        name=name,
        instruction=f"You are {name}.",
        llm_config={"provider": "openai", "model": "gpt-4.1-mini"},
        delegates=delegates,
    )
    saved = store.store_memagent(model)
    return model.agent_id or getattr(saved, "agent_id", None) or str(saved)


def _form(**values):
    form = {
        "agent_name": "Coordinator",
        "instruction": "Split the work.",
        "application_mode": "assistant",
        "max_steps": "10",
        "tool_access": "private",
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
        "delegation_form": "1",
        "delegation_enabled": "1",
        "delegation_root_fallback": "1",
        "delegation_consolidation": "model",
    }
    form.update(values)
    return form


def test_delegates_are_chosen_saved_and_shown_again(client, provider):
    researcher = _agent(provider, "Researcher")
    writer = _agent(provider, "Writer")

    page = client.get("/agents/new").text
    assert "Delegates" in page and 'name="delegate_ids"' in page
    assert f'value="{researcher}"' in page and f'value="{writer}"' in page

    response = client.post(
        "/agents/new",
        data=_form(
            delegate_ids=[researcher, writer],
            delegation_max_workers="2",
            delegation_consolidation="deterministic",
        ),
    )
    assert response.status_code in (302, 303), response.text[:400]
    coordinator = next(
        agent for agent in provider.list_memagents() if agent.name == "Coordinator"
    )
    assert coordinator.delegates == [researcher, writer]
    config = coordinator.delegation_config
    assert config["enabled"] is True and config["mode"] == "auto"
    assert config["max_workers"] == 2
    assert config["consolidation_strategy"] == "deterministic"
    assert config["allow_root_fallback"] is True

    # The edit form shows the selection, and the agent is not its own choice.
    edit = client.get(f"/agents/{coordinator.agent_id}/edit").text
    assert edit.count('name="delegate_ids"') == 2
    assert (
        f'value="{researcher}" checked' in edit and f'value="{writer}" checked' in edit
    )

    # Unticking one and switching delegation off is saved; other config stays.
    response = client.post(
        f"/agents/{coordinator.agent_id}/edit",
        data=_form(delegate_ids=[writer], delegation_enabled=""),
    )
    assert response.status_code in (302, 303), response.text[:400]
    saved = provider.retrieve_memagent(coordinator.agent_id)
    assert saved.delegates == [writer]
    assert saved.delegation_config["enabled"] is False

    # A post without the Delegates section (another client) leaves them alone.
    form = _form()
    form.pop("delegation_form")
    client.post(f"/agents/{coordinator.agent_id}/edit", data=form)
    assert provider.retrieve_memagent(coordinator.agent_id).delegates == [writer]


def test_delegate_loops_are_refused(client, provider):
    a = _agent(provider, "Agent A")
    b = _agent(provider, "Agent B", delegates=[a])

    # A can't delegate to itself, nor to B, which already delegates back to A.
    own = client.post(
        f"/agents/{a}/edit", data=_form(agent_name="Agent A", delegate_ids=[a])
    )
    assert own.status_code == 200 and "can&#39;t be its own delegate" in own.text
    loop = client.post(
        f"/agents/{a}/edit", data=_form(agent_name="Agent A", delegate_ids=[b])
    )
    assert loop.status_code == 200
    assert "would loop" in loop.text and "Agent B" in loop.text
    assert not provider.retrieve_memagent(a).delegates

    bad = client.post(
        f"/agents/{a}/edit",
        data=_form(agent_name="Agent A", delegate_ids=[b], delegation_max_workers="99"),
    )
    assert "between 1 and 8" in bad.text


def test_loading_a_saved_delegate_loop_does_not_hang(provider, caplog):
    a = _agent(provider, "Loop A")
    b = _agent(provider, "Loop B", delegates=[a])
    # Written directly, as an older release or another client could have.
    record = provider.retrieve_memagent(a)
    provider.store_memagent(record.model_copy(update={"delegates": [b]}))

    with patch("memorizz.memagent.core.create_llm_provider") as create:
        create.return_value = None
        agent = MemAgent.load(a, provider)
    assert [delegate.agent_id for delegate in agent.delegates] == [b]
    # B's delegate would be A again: skipped rather than loaded forever.
    assert agent.delegates[0].delegates == []
    assert "already in this delegate chain" in caplog.text


def test_a_harness_is_added_as_a_delegate_and_marked(client, provider):
    coordinator = _agent(provider, "Coordinator")

    made = client.post(
        "/api/harness-delegates",
        json={"harness": "codex", "model": "gpt-6-luna", "coordinator_id": coordinator},
    )
    assert made.status_code == 200, made.text
    agent = made.json()["agent"]
    assert agent["name"] == "Codex delegate (gpt-6-luna)"
    assert agent["runs_on"] == {
        "harness": "codex",
        "label": "Codex",
        "model": "gpt-6-luna",
    }
    saved = provider.retrieve_memagent(agent["id"])
    assert saved.meta_harness is True and saved.meta_harness_mode == "runtime"
    assert saved.default_harness == "codex"
    # No workspace: it works in the workspace of the run it is part of.
    assert saved.harness_config == {"model": "gpt-6-luna"}
    # It loads with the coordinator's model, so it loads without warnings.
    assert saved.llm_config["model"] == "gpt-4.1-mini"

    plain = client.post(
        "/api/harness-delegates", json={"harness": "pi", "name": "Summariser"}
    )
    assert plain.json()["agent"]["name"] == "Summariser"
    assert provider.retrieve_memagent(plain.json()["agent"]["id"]).harness_config == {}

    for bad in (
        {"harness": "memagent"},
        {"harness": "auto"},
        {"harness": "codex", "model": "a\nb"},
    ):
        assert client.post("/api/harness-delegates", json=bad).status_code == 400

    # The coordinator's Delegates list says what each harness delegate runs on.
    edit = client.get(f"/agents/{coordinator}/edit").text
    assert "runs on Codex · gpt-6-luna" in edit and "runs on pi" in edit
    assert "delegate-choice--harness" in edit
    response = client.post(
        f"/agents/{coordinator}/edit",
        data=_form(delegate_ids=[agent["id"], plain.json()["agent"]["id"]]),
    )
    assert response.status_code in (302, 303), response.text[:400]
    assert provider.retrieve_memagent(coordinator).delegates == [
        agent["id"],
        plain.json()["agent"]["id"],
    ]


def test_the_harness_model_is_saved_and_other_harness_settings_kept(
    client, provider, tmp_path
):
    workspace = tmp_path / "project"
    workspace.mkdir()
    response = client.post(
        "/agents/new",
        data=_form(
            agent_name="Runner",
            meta_harness_mode="runtime",
            default_harness="claude-code",
            harness_workspace=str(workspace),
            harness_model="claude-sonnet-5-5",
        ),
    )
    assert response.status_code in (302, 303), response.text[:400]
    runner = next(a for a in provider.list_memagents() if a.name == "Runner")
    assert runner.harness_config["model"] == "claude-sonnet-5-5"
    assert runner.harness_config["workspace"] == str(workspace.resolve())

    edit = client.get(f"/agents/{runner.agent_id}/edit").text
    assert 'id="harness_model" name="harness_model" value="claude-sonnet-5-5"' in edit

    # Clearing the model keeps the workspace and any other harness settings.
    record = provider.retrieve_memagent(runner.agent_id)
    record.harness_config = {**record.harness_config, "budget": {"max_steps": 12}}
    provider.store_memagent(record)
    response = client.post(
        f"/agents/{runner.agent_id}/edit",
        data=_form(
            agent_name="Runner",
            meta_harness_mode="runtime",
            default_harness="claude-code",
            harness_workspace=str(workspace),
            harness_model="",
        ),
    )
    assert response.status_code in (302, 303), response.text[:400]
    config = provider.retrieve_memagent(runner.agent_id).harness_config
    assert "model" not in config
    assert config["workspace"] == str(workspace.resolve()) and config["budget"] == {
        "max_steps": 12
    }


def test_harness_options_list_harnesses_and_their_models(client, provider):
    from types import SimpleNamespace

    service = SimpleNamespace(
        adapters={},
        list_harnesses=lambda: [
            {"name": "codex", "ready": True, "models": ["gpt-6-astra"], "metadata": {}},
            {"name": "memagent", "ready": True, "models": [], "metadata": {}},
            {"name": "hermes", "ready": False, "models": [], "metadata": {}},
        ],
    )
    with patch(
        "memorizz.ui.routers.agents_crud.get_meta_harness", return_value=service
    ), patch("memorizz.llms.model_lists.latest_models_for", return_value={}):
        body = client.get("/api/harness-delegates/options").json()
    assert body["ok"] is True
    # memagent can't be a harness delegate of itself; unready ones are listed but off.
    assert body["harnesses"] == [
        {"name": "codex", "label": "Codex", "ready": True},
        {"name": "hermes", "label": "Hermes", "ready": False},
    ]
    assert body["models"]["codex"]["default"] == "gpt-6-astra"
    assert body["models"]["codex"]["groups"][0]["models"] == ["gpt-6-astra"]
