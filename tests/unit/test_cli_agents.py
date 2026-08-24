"""CLI coverage for explicit MemAgent creation and discovery."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from memorizz.cli.app import app
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

runner = CliRunner()


def _isolated_env(tmp_path):
    return {
        "MEMORIZZ_HOME": str(tmp_path),
        "MEMORIZZ_BACKEND": "",
        "MEMORIZZ_DEFAULT_LLM_PROVIDER": "",
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "AZURE_OPENAI_API_KEY": "",
        "OLLAMA_HOST": "http://127.0.0.1:1",
    }


@pytest.mark.unit
def test_cli_create_list_and_show_agent(tmp_path):
    env = _isolated_env(tmp_path)

    def initialize_cache_without_network(cache_manager, *_args, **_kwargs):
        cache_manager.cache_instance = object()

    with patch(
        "memorizz.memagent.managers.cache_manager.CacheManager._initialize_cache",
        initialize_cache_without_network,
    ):
        created = runner.invoke(
            app,
            [
                "agents",
                "create",
                "--name",
                "CLI Agent",
                "--instruction",
                "Created through the CLI.",
                "--memory-id",
                "cli-memory",
                "--semantic-cache",
                "--no-llm",
                "--set-default",
                "--json",
            ],
            env=env,
        )
    assert created.exit_code == 0, created.output
    payload = json.loads(created.output)
    agent_id = payload["agent"]["agent_id"]
    assert payload["agent"]["name"] == "CLI Agent"
    assert payload["agent"]["memory_ids"] == ["cli-memory"]
    assert payload["agent"]["semantic_cache"] is True
    assert payload["default_agent"] is True

    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["agent_id"] == agent_id

    listed = runner.invoke(app, ["agents", "list", "--json"], env=env)
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)["agents"][0]["agent_id"] == agent_id

    shown = runner.invoke(app, ["agents", "show", agent_id, "--json"], env=env)
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["agent"]["instruction"] == (
        "Created through the CLI."
    )

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    persisted = provider.retrieve_memagent(agent_id)
    assert persisted is not None
    assert persisted.name == "CLI Agent"


@pytest.mark.unit
def test_cli_rejects_invalid_agent_mode_without_persisting(tmp_path):
    result = runner.invoke(
        app,
        [
            "agents",
            "create",
            "--name",
            "Invalid",
            "--application-mode",
            "unknown",
            "--no-llm",
        ],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "Invalid application mode" in result.output
    assert not (tmp_path / "memory" / "memagent").exists()


@pytest.mark.unit
def test_cli_rejects_explicit_unavailable_llm_without_persisting(tmp_path):
    result = runner.invoke(
        app,
        [
            "agents",
            "create",
            "--name",
            "Missing LLM",
            "--llm-provider",
            "openai",
        ],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "could not be initialized" in result.output
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    assert provider.list_memagents() == []


@pytest.mark.unit
def test_cli_persists_meta_harness_agent_configuration(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = runner.invoke(
        app,
        [
            "agents",
            "create",
            "--name",
            "Harness Agent",
            "--no-llm",
            "--harness-mode",
            "runtime",
            "--default-harness",
            "codex",
            "--harness-workspace",
            str(workspace),
            "--json",
        ],
        env=_isolated_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["agent"]["meta_harness"] is True
    assert payload["agent"]["meta_harness_mode"] == "runtime"
    assert payload["agent"]["default_harness"] == "codex"

    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    persisted = provider.retrieve_memagent(payload["agent"]["agent_id"])
    assert persisted.meta_harness_mode == "runtime"
    assert persisted.harness_config["workspace"] == str(workspace.resolve())
    assert persisted.harness_config["permissions"]["allowed_roots"] == [
        str(workspace.resolve())
    ]
