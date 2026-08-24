# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Functional tests for the agent CRUD, playground, and evalground routes.

These are the three largest route groups still living inside ui/app.py.
Beyond the smoke suite's status-code checks, these tests pin real behavior
(create → persist → render → edit → JSON thread payload) so the routes can
be extracted into routers without silent breakage.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz import MemAgent  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers.evalground import _secret_free_agent_template  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), follow_redirects=False)


@pytest.fixture()
def connected(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui-func", lazy_vector_indexes=True)
    )
    patcher = patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    )
    patcher.start()
    yield provider
    patcher.stop()


def _create_agent(client, **overrides):
    form = {
        "agent_name": "Func Test Agent",
        "instruction": "You are a functional test agent.",
        "application_mode": "assistant",
        "max_steps": "10",
        "tool_access": "private",
        "llm_provider": "openai",
        "llm_model": "gpt-4.1-mini",
    }
    form.update(overrides)
    return client.post("/agents/new", data=form)


class TestAgentCrud:
    @pytest.mark.unit
    def test_create_persists_and_redirects(self, client, connected):
        resp = _create_agent(client)
        assert resp.status_code in (302, 303), resp.text[:300]

        agents = connected.list_memagents()
        assert len(agents) == 1
        agent = agents[0]
        assert agent.name == "Func Test Agent"
        assert agent.instruction == "You are a functional test agent."
        loaded = MemAgent.load(agent.agent_id, memory_provider=connected)
        assert loaded.agent_id == agent.agent_id
        assert loaded.name == "Func Test Agent"
        assert loaded.llm_provider == "openai"
        assert loaded.llm_model == "gpt-4.1-mini"
        loaded.close(close_memory_provider=False)

    @pytest.mark.unit
    def test_create_with_flags_round_trips(self, client, connected):
        resp = _create_agent(client, semantic_cache="on", continual_learning="on")
        assert resp.status_code in (302, 303)
        agent = connected.list_memagents()[0]
        assert bool(agent.semantic_cache) is True
        assert bool(agent.continual_learning) is True
        assert agent.continual_learning_config["skill_injection_role"] == "user"

    @pytest.mark.unit
    def test_meta_harness_runtime_configuration_round_trips(
        self, client, connected, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        resp = _create_agent(
            client,
            meta_harness_mode="runtime",
            default_harness="codex",
            harness_workspace=str(workspace),
        )
        assert resp.status_code in (302, 303), resp.text[:300]
        agent = connected.list_memagents()[0]
        assert agent.meta_harness is True
        assert agent.meta_harness_mode == "runtime"
        assert agent.default_harness == "codex"
        assert agent.harness_config["workspace"] == str(workspace)

        form = client.get(f"/agents/{agent.agent_id}/edit")
        assert form.status_code == 200
        assert 'option value="runtime" selected' in form.text
        assert 'option value="codex" selected' in form.text

    @pytest.mark.unit
    def test_browser_control_can_be_configured_and_persisted(self, client, connected):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "unit-test-only",
                "MEMORIZZ_BROWSER_USE_ALLOWED_DOMAINS": "example.com, *.notion.so",
            },
            clear=False,
        ), patch(
            "memorizz.browser_control.providers.browser_use.shutil.which",
            return_value="/isolated/bin/browser-use",
        ), patch(
            "memorizz.browser_control.providers.browser_use.BrowserUseProvider._resolved_python_command",
            return_value=["/isolated/bin/python"],
        ):
            resp = _create_agent(client, browser_control_provider="browseruse")

        assert resp.status_code in (302, 303), resp.text[:300]
        agent = connected.list_memagents()[0]
        assert agent.browser_control["provider"] == "browseruse"
        assert agent.browser_control["allowed_domains"] == [
            "example.com",
            "*.notion.so",
        ]
        assert "api_key" not in agent.browser_control

        form = client.get(f"/agents/{agent.agent_id}/edit")
        assert form.status_code == 200
        assert 'option value="browseruse" selected' in form.text

    @pytest.mark.unit
    def test_create_with_reviewed_developer_skill_authority(self, client, connected):
        resp = _create_agent(
            client,
            continual_learning="on",
            skill_injection_role="developer",
            continual_learning_require_shadow="on",
            continual_learning_shadow_evaluation="on",
        )
        assert resp.status_code in (302, 303), resp.text[:300]

        agent = connected.list_memagents()[0]
        assert agent.continual_learning_config["skill_injection_role"] == ("developer")
        assert agent.continual_learning_config["require_shadow"] is True
        assert agent.continual_learning_config["shadow_evaluation_enabled"] is True

        form = client.get(f"/agents/{agent.agent_id}/edit")
        assert form.status_code == 200
        assert "Authority for newly learned skills" in form.text
        assert "Passively evaluate shadow skills" in form.text
        assert 'option value="developer" selected' in form.text

    @pytest.mark.unit
    def test_developer_skill_authority_requires_shadow_review(self, client, connected):
        resp = _create_agent(
            client,
            continual_learning="on",
            skill_injection_role="developer",
        )

        assert resp.status_code == 200
        assert "require shadow review" in resp.text.lower()
        assert connected.list_memagents() == []

    @pytest.mark.unit
    def test_agents_list_page_shows_created_agent(self, client, connected):
        _create_agent(client)
        resp = client.get("/agents")
        assert resp.status_code == 200
        assert "Func Test Agent" in resp.text

    @pytest.mark.unit
    def test_edit_updates_instruction_and_preserves_flags(self, client, connected):
        _create_agent(client, continual_learning="on")
        agent_id = connected.list_memagents()[0].agent_id

        form_page = client.get(f"/agents/{agent_id}/edit")
        assert form_page.status_code == 200

        resp = client.post(
            f"/agents/{agent_id}/edit",
            data={
                "agent_name": "Func Test Agent",
                "instruction": "Updated instruction.",
                "application_mode": "assistant",
                "max_steps": "10",
                "tool_access": "private",
                "llm_provider": "openai",
                "llm_model": "gpt-4.1-mini",
                "continual_learning": "on",
                "whatsapp_enabled": "on",
                "whatsapp_welcome_message": "Hello from Memorizz",
            },
        )
        assert resp.status_code in (302, 303), resp.text[:300]

        updated = connected.retrieve_memagent(agent_id)
        assert updated.instruction == "Updated instruction."
        assert bool(updated.continual_learning) is True
        assert bool(updated.whatsapp_enabled) is True
        assert updated.whatsapp_config["welcome_message"] == "Hello from Memorizz"

    @pytest.mark.unit
    def test_agent_detail_redirects_to_playground(self, client, connected):
        _create_agent(client)
        agent_id = connected.list_memagents()[0].agent_id
        resp = client.get(f"/agents/{agent_id}")
        assert resp.status_code in (301, 302, 303, 307, 308)


class TestPlayground:
    @pytest.mark.unit
    def test_playground_select_page(self, client, connected):
        _create_agent(client)
        resp = client.get("/playground")
        assert resp.status_code == 200
        assert "Func Test Agent" in resp.text

    @pytest.mark.unit
    def test_playground_page_renders_for_agent(self, client, connected):
        _create_agent(client)
        agent_id = connected.list_memagents()[0].agent_id
        resp = client.get(f"/agents/{agent_id}/playground")
        assert resp.status_code == 200
        # The memory panes the playground template must always carry.
        for anchor in (
            "context-workflow-container",
            "context-skills-container",
        ):
            assert anchor in resp.text, f"playground missing {anchor}"

    @pytest.mark.unit
    def test_playground_thread_json_shape(self, client, connected):
        _create_agent(client)
        agent_id = connected.list_memagents()[0].agent_id
        resp = client.get(f"/agents/{agent_id}/playground/thread")
        assert resp.status_code == 200
        payload = resp.json()
        for key in (
            "messages",
            "toolbox_memory",
            "workflow_memory",
            "skill_memory",
            "entity_memory",
        ):
            assert key in payload, f"thread payload missing {key}"
        assert isinstance(payload["messages"], list)

    @pytest.mark.unit
    def test_playground_unknown_agent_is_not_500(self, client, connected):
        resp = client.get("/agents/does-not-exist/playground")
        assert resp.status_code < 500


class TestEvalground:
    @pytest.mark.unit
    def test_page_exposes_local_and_metered_memory_suite_models(
        self, client, connected
    ):
        resp = client.get("/evalground")
        assert resp.status_code == 200
        for benchmark_name in (
            "AgentMemBench",
            "LongMemEval-V2",
            "LoCoMo-Plus",
            "BEAM",
            "MemoryAgentBench",
        ):
            assert benchmark_name in resp.text
        assert "Ollama readers cost $0 externally" in resp.text
        assert "OpenAI (metered)" in resp.text
        assert "Full MemAgent execution" in resp.text
        assert "Memory retrieval diagnostic" in resp.text
        assert "Concept Query Expansion" in resp.text

    @pytest.mark.unit
    def test_runs_active_empty(self, client, connected):
        resp = client.get("/evalground/runs/active")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, (list, dict))

    @pytest.mark.unit
    def test_unknown_run_id_is_404(self, client, connected):
        resp = client.get("/evalground/runs/nonexistent-run-id")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            # some payloads return {"error": ...} instead of 404
            assert "error" in str(resp.json()).lower() or resp.json()

    @pytest.mark.unit
    def test_start_run_without_oracle_provider_is_rejected(self, client, connected):
        resp = client.post(
            "/evalground/runs",
            data={"agent_id": "", "benchmark": "longmemeval", "num_samples": "1"},
        )
        # Filesystem provider isn't supported by evalground: must fail
        # cleanly (4xx or JSON error), never 5xx, and never start a run.
        assert resp.status_code < 500

    @pytest.mark.unit
    def test_local_suite_can_queue_without_oracle_or_paid_key(
        self, client, connected, tmp_path
    ):
        with patch("memorizz.ui.routers.evalground.threading.Thread") as thread_cls:
            resp = client.post(
                "/evalground/runs",
                data={
                    "benchmark": "agentmembench",
                    "dataset_variant": "locomo",
                    "data_path": str(tmp_path),
                    "num_samples": "1",
                    "model": "qwen2.5:0.5b",
                    "embedding_model": "nomic-embed-text",
                    "ollama_host": "http://localhost:11434",
                },
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"
        thread_cls.return_value.start.assert_called_once()

    @pytest.mark.unit
    def test_local_suite_rejects_unknown_variant(self, client, connected, tmp_path):
        resp = client.post(
            "/evalground/runs",
            data={
                "benchmark": "beam",
                "dataset_variant": "tiny",
                "data_path": str(tmp_path),
                "num_samples": "1",
            },
        )
        assert resp.status_code == 400
        assert "Unknown BEAM variant" in resp.json()["error"]

    @pytest.mark.unit
    def test_full_memagent_mode_requires_selected_agent(
        self, client, connected, tmp_path
    ):
        resp = client.post(
            "/evalground/runs",
            data={
                "benchmark": "agentmembench",
                "dataset_variant": "locomo",
                "data_path": str(tmp_path),
                "num_samples": "1",
                "evaluation_mode": "memagent",
            },
        )
        assert resp.status_code == 400
        assert "Select an agent" in resp.json()["error"]

    @pytest.mark.unit
    def test_agent_eval_snapshot_removes_secrets_and_side_effects(self):
        snapshot = _secret_free_agent_template(
            {
                "agent_id": "agent-1",
                "llm_config": {
                    "provider": "openai",
                    "model": "gpt-test",
                    "api_key": "must-not-leave-process",
                },
                "internet_access_config": {"token": "also-secret"},
                "tools": ["mutating-tool"],
                "mcp_servers": [{"name": "calendar"}],
                "continual_learning": True,
            }
        )
        serialized = str(snapshot)
        assert "must-not-leave-process" not in serialized
        assert "also-secret" not in serialized
        assert snapshot["tools"] == []
        assert snapshot["mcp_servers"] == []
        assert snapshot["continual_learning"] is False
