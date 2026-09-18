"""Guided setup uses real provisioning against a deterministic Notion API."""

import json
import os
from unittest.mock import patch

import pytest
from dotenv import dotenv_values
from typer.testing import CliRunner

from memorizz import _env_io as env
from memorizz.cli import agent_factory
from memorizz.cli.app import app
from memorizz.memory_provider.notion import NotionClient, NotionWriteUncertain
from tests.fixtures.notion_api import NotionAPI, Response


@pytest.fixture
def wizard(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    clean = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("MEMORIZZ_", "NOTION_", "OPENAI_", "AZURE_OPENAI_", "ORACLE_", "MONGODB_")
        )
    }
    clean["MEMORIZZ_HOME"] = str(tmp_path / "home")
    with patch.dict(os.environ, clean, clear=True):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(env, "_ENV_SOURCES", {})
        api = NotionAPI()
        client = NotionClient(
            "wizard-private-token", session=api, sleep=lambda _: None, clock=lambda: 0
        )
        monkeypatch.setattr(
            "memorizz.memory_provider.notion.client.NotionClient", lambda token: client
        )
        monkeypatch.setattr(
            "memorizz.cli.memory_commands.secret_prompt",
            lambda label: "wizard-private-token",
        )
        # The test parent is only present in this fake account, never a real page.
        api.pages[api.parent_id] = {
            "id": api.parent_id,
            "object": "page",
            "in_trash": False,
        }
        yield api, client
        client.close()


def test_connect_existing_library_checks_schema_without_remote_writes(wizard):
    api, _ = wizard
    result = CliRunner().invoke(
        app, ["notion", "connect"], input=f"existing\n{api.data_source_id}\nnone\n\ny\n"
    )
    assert result.exit_code == 0, result.output
    saved = dotenv_values(env.resolve_env_file())
    assert saved["NOTION_TOKEN"] == "wizard-private-token"
    assert saved["MEMORIZZ_BACKEND"] == "notion"
    assert saved["MEMORIZZ_NOTION_DATA_SOURCE_ID"] == api.data_source_id
    assert saved["MEMORIZZ_NOTION_SEMANTIC_BACKEND"] == "none"
    assert "MEMORIZZ_BACKEND" not in os.environ
    assert all(method == "GET" for method, *_ in api.calls)
    assert (
        "read-only check" in result.output
        and "Semantic search is disabled" in result.output
    )
    assert "wizard-private-token" not in result.output


def test_provider_prompt_retries_invalid_choice_and_accepts_uppercase(wizard):
    api, _ = wizard
    result = CliRunner().invoke(
        app,
        ["memory", "configure"],
        input=f"invalid\nNOTION\nexisting\n{api.data_source_id}\nnone\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert "Choose one of:" in result.output
    assert dotenv_values(env.resolve_env_file())["MEMORIZZ_BACKEND"] == "notion"


def test_provider_prompt_eof_cancels_without_configuration_writes(wizard):
    result = CliRunner().invoke(app, ["memory", "configure"], input="")
    assert result.exit_code == 1
    assert "Cancelled; current agent unchanged" in result.output
    assert not env.resolve_env_file().exists()


def test_database_url_resolves_data_source_not_view_id(wizard):
    api, _ = wizard
    api.failures = [
        Response(404, {"code": "object_not_found"}),
        Response(
            200, {"id": api.database_id, "data_sources": [{"id": api.data_source_id}]}
        ),
    ]
    result = CliRunner().invoke(
        app,
        ["memory", "configure", "notion"],
        input=f"existing\nhttps://www.notion.so/{api.database_id.replace('-', '')}?v=unrelated\nnone\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert (
        dotenv_values(env.resolve_env_file())["MEMORIZZ_NOTION_DATA_SOURCE_ID"]
        == api.data_source_id
    )
    assert [call[1] for call in api.calls][:3] == [
        "/data_sources/" + api.database_id,
        "/databases/" + api.database_id,
        "/data_sources/" + api.data_source_id,
    ]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_access_failure_never_saves_or_provisions(wizard, status):
    api, _ = wizard
    api.failures = [
        Response(
            status, {"code": "object_not_found", "message": "wizard-private-token"}
        )
    ] * 2
    result = CliRunner().invoke(
        app, ["notion", "connect"], input=f"existing\n{api.data_source_id}\n"
    )
    assert result.exit_code == 1
    assert "share the exact page/database" in result.output
    assert "wizard-private-token" not in result.output
    assert not env.resolve_env_file().exists()
    assert all(method == "GET" for method, *_ in api.calls)


def test_incompatible_database_is_not_migrated(wizard):
    api, _ = wizard
    del api.properties["Content"]
    result = CliRunner().invoke(
        app, ["notion", "connect"], input=f"existing\n{api.data_source_id}\n"
    )
    assert result.exit_code == 2
    assert "No schema was changed" in result.output
    assert not env.resolve_env_file().exists()
    assert all(method == "GET" for method, *_ in api.calls)


def test_create_cancelled_before_any_remote_write(wizard):
    api, _ = wizard
    result = CliRunner().invoke(
        app,
        ["notion", "connect"],
        input=f"create\n{api.parent_id}\nMy memory\nnone\n\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "nothing saved or created" in result.output
    assert not env.resolve_env_file().exists()
    assert len(api.pages) == 1
    assert all(method == "GET" for method, *_ in api.calls)


def test_confirmed_creation_provisions_all_views_saves_manifest_and_configuration(
    wizard,
):
    api, _ = wizard
    result = CliRunner().invoke(
        app,
        ["notion", "connect", "--project"],
        input=f"create\n{api.parent_id}\nMy memory\nnone\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(api.views) == 17
    saved = dotenv_values(".env")
    assert saved["MEMORIZZ_NOTION_DATA_SOURCE_ID"] == api.data_source_id
    from pathlib import Path

    manifest = next(Path.cwd().glob("notion-setup-*.json"))
    assert json.loads(manifest.read_text())["data_source_id"] == api.data_source_id
    assert "wizard-private-token" not in manifest.read_text() + result.output
    assert "Restart required" in result.output and "not migrated" in result.output


def test_partial_creation_exposes_recovery_ids_and_never_retries(wizard, monkeypatch):
    api, _ = wizard
    calls = []

    def uncertain(*args, **kwargs):
        calls.append(True)
        exc = NotionWriteUncertain("private-token-that-must-not-appear")
        exc.notion_workspace = {"root_page_id": api.parent_id}
        raise exc

    monkeypatch.setattr(
        "memorizz.memory_provider.notion.provision_notion_workspace", uncertain
    )
    result = CliRunner().invoke(
        app,
        ["notion", "connect"],
        input=f"create\n{api.parent_id}\nMy memory\nnone\n\ny\n",
    )
    assert result.exit_code == 1
    assert (
        api.parent_id in result.output
        and "do not blindly rerun create" in result.output
    )
    assert "private-token-that-must-not-appear" not in result.output
    assert calls == [True]
    assert not env.resolve_env_file().exists()


def test_local_save_failure_after_creation_keeps_resource_ids(wizard, monkeypatch):
    api, _ = wizard

    def fail(*args):
        raise PermissionError("wizard-private-token")

    monkeypatch.setattr(env, "update_env_file", fail)
    result = CliRunner().invoke(
        app,
        ["notion", "connect"],
        input=f"create\n{api.parent_id}\nMy memory\nnone\n\ny\n",
    )
    assert result.exit_code == 2
    assert api.data_source_id in result.output
    assert len(list(env.memorizz_home().glob("notion-setup-*.json"))) == 1
    assert not env.resolve_env_file().exists()
    assert "MEMORIZZ_BACKEND" not in os.environ
    assert "wizard-private-token" not in result.output


def test_filesystem_vector_setup_retains_no_stale_embedding_dimensions(
    wizard, monkeypatch
):
    api, _ = wizard
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "old-model")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "1536")
    result = CliRunner().invoke(
        app,
        ["notion", "connect"],
        input=f"existing\n{api.data_source_id}\nfilesystem\n\nollama\nnomic-embed-text\n\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    saved = dotenv_values(env.resolve_env_file())
    assert saved["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "ollama"
    assert saved["MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS"] == ""
    assert "No model download or paid embedding call" in result.output
    assert all(method == "GET" for method, *_ in api.calls)


@pytest.mark.parametrize("backend", ["filesystem", "mongodb", "oracle"])
def test_other_provider_wizards_are_save_only(wizard, backend):
    details = {
        "filesystem": "\n",
        "mongodb": "\n",
        "oracle": "app_user\nlocalhost:1521/DB\n",
    }[backend]
    result = CliRunner().invoke(
        app,
        ["memory", "configure", backend],
        input=details + "ollama\nnomic-embed-text\n\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert dotenv_values(env.resolve_env_file())["MEMORIZZ_BACKEND"] == backend
    assert "MEMORIZZ_BACKEND" not in os.environ
    assert wizard[0].calls == []


@pytest.mark.parametrize("backend", ["mongodb", "oracle", "misspelled"])
def test_selected_backend_never_silently_falls_back(wizard, monkeypatch, backend):
    monkeypatch.setenv("MEMORIZZ_BACKEND", backend)
    for key in ("MONGODB_URI", "ORACLE_DSN", "ORACLE_USER", "ORACLE_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="memorizz memory configure"):
        agent_factory.detect_memory_provider({}, [])


def test_mongodb_uses_configured_embedding_defaults(wizard, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_BACKEND", "mongodb")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(
        agent_factory,
        "_choose_embedding",
        lambda _: {
            "embedding_provider": "ollama",
            "embedding_config": {"model": "embed"},
        },
    )
    with patch("memorizz.memory_provider.mongodb.MongoDBProvider") as create:
        agent_factory.detect_memory_provider({}, [])
        config = create.call_args.args[0]
        assert config.embedding_provider == "ollama"
        assert config.embedding_config["model"] == "embed"
