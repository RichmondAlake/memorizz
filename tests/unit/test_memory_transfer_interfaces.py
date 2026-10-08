import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from memorizz import MemoryArchive, MemoryType
from memorizz.cli.app import app
from memorizz.mcp_server import (
    MemorizzMCPServerConfig,
    MemorizzRuntime,
    create_memorizz_mcp_server,
)
from memorizz.mcp_server.auth import RequestIdentity
from memorizz.mcp_server.config import ALL_SCOPES, READ_SCOPE
from memorizz.mcp_server.runtime import MemorizzServerError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app


def _provider(path):
    return FileSystemProvider(FileSystemConfig(root_path=path, use_faiss=False))


def test_cli_export_preview_apply(tmp_path, monkeypatch):
    from memorizz.cli import agent_factory

    source = _provider(tmp_path / "source")
    source.store(
        {"content": "Portable evidence", "memory_id": "project"},
        MemoryType.KNOWLEDGE_BASE,
    )
    monkeypatch.setattr(
        agent_factory,
        "detect_memory_provider",
        lambda *_: _provider(tmp_path / "source"),
    )
    runner, archive = CliRunner(), tmp_path / "portable.memorizz.json"
    result = runner.invoke(
        app, ["memory", "export", str(archive), "--memory-id", "project"]
    )
    assert result.exit_code == 0, result.output
    monkeypatch.setattr(
        agent_factory,
        "detect_memory_provider",
        lambda *_: _provider(tmp_path / "target"),
    )
    preview = runner.invoke(app, ["memory", "import", str(archive)])
    assert preview.exit_code == 0 and json.loads(preview.stdout)["dry_run"]
    assert not _provider(tmp_path / "target").list_all(MemoryType.KNOWLEDGE_BASE)
    applied = runner.invoke(app, ["memory", "import", str(archive), "--apply"])
    assert applied.exit_code == 0 and json.loads(applied.stdout)["imported"] == 1


def test_plugin_archive_follows_project_namespace_and_user(tmp_path, monkeypatch):
    from memorizz.cli import plugin_commands

    monkeypatch.delenv("MEMORIZZ_PLUGIN_REMOTE_URL", raising=False)
    monkeypatch.setattr(plugin_commands, "_remote_url", lambda: None)
    monkeypatch.setattr(plugin_commands, "_user_id", lambda: "alice")
    source_project = tmp_path / "project-a"
    target_project = tmp_path / "project-b"
    source_project.mkdir()
    target_project.mkdir()
    source_id = plugin_commands.project_memory_id(source_project)
    source = _provider(tmp_path / "source")
    for user in ("alice", "bob"):
        source.store(
            {"content": f"{user}'s notes", "user_id": user, "memory_id": source_id},
            MemoryType.KNOWLEDGE_BASE,
        )
    monkeypatch.setattr(
        plugin_commands, "_store", lambda: _provider(tmp_path / "source")
    )
    archive = tmp_path / "project.memorizz.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plugin", "export-memory", str(archive), "--workspace", str(source_project)],
    )
    assert result.exit_code == 0, result.output
    assert "bob's notes" not in archive.read_text()
    monkeypatch.setattr(
        plugin_commands, "_store", lambda: _provider(tmp_path / "target")
    )
    result = runner.invoke(
        app,
        [
            "plugin",
            "import-memory",
            str(archive),
            "--workspace",
            str(target_project),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    row = _provider(tmp_path / "target").list_all(MemoryType.KNOWLEDGE_BASE)[0]
    assert (
        row["memory_id"] == plugin_commands.project_memory_id(target_project)
        and row["user_id"] == "alice"
    )


def test_mcp_tools_tenant_scope_read_preview_and_write_policy(tmp_path):
    provider = _provider(tmp_path / "memory")
    for user in ("alice", "bob", None):
        provider.store(
            {"content": str(user), "user_id": user}, MemoryType.KNOWLEDGE_BASE
        )
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=False), provider=provider
    )
    identity = RequestIdentity("alice", frozenset([READ_SCOPE]), True)
    result = runtime.export_memories(identity)
    assert result["ok"] and result["archive"]["manifest"]["record_count"] == 1
    assert runtime.import_memories(result["archive"], identity, id_strategy="new")["ok"]
    with pytest.raises(MemorizzServerError, match="scope"):
        runtime.import_memories(
            result["archive"], identity, dry_run=False, id_strategy="new"
        )
    with pytest.raises(MemorizzServerError, match="disabled"):
        runtime.import_memories(
            result["archive"],
            RequestIdentity("alice", frozenset(ALL_SCOPES), True),
            dry_run=False,
            id_strategy="new",
        )
    names = create_memorizz_mcp_server(runtime=runtime)._tool_manager._tools
    assert "memorizz_export_memories" in names and "memorizz_import_memories" in names


def test_mcp_remote_archive_cannot_restore_global_configuration(tmp_path):
    source = _provider(tmp_path / "source")
    source.store(
        {"name": "Shared", "content": "Global", "user_id": "alice"},
        MemoryType.SHARED_MEMORY,
    )
    archive = MemoryArchive(source).export(user_id="alice")
    config = MemorizzMCPServerConfig(
        transport="http", allow_anonymous_http=True, allow_writes=True
    )
    runtime = MemorizzRuntime(config, provider=_provider(tmp_path / "target"))
    with pytest.raises(MemorizzServerError, match="not authorized"):
        runtime.import_memories(
            archive, RequestIdentity("alice", frozenset(ALL_SCOPES), True)
        )


def test_ui_download_preview_restore_and_invalid_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MEMORIZZ_UI_AUTH_ACCOUNTS", raising=False)
    monkeypatch.delenv("MEMORIZZ_UI_READ_ONLY", raising=False)
    provider = _provider(tmp_path / "memory")
    provider.store({"content": "UI evidence"}, MemoryType.KNOWLEDGE_BASE)
    client = TestClient(create_app())
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": False,
        },
    ):
        page = client.get("/memory-transfer")
        assert page.status_code == 200 and "Preview import" in page.text
        download = client.get("/api/memory-transfer/export")
        assert (
            download.status_code == 200
            and ".memorizz.json" in download.headers["content-disposition"]
        )
        assert download.headers["cache-control"] == "no-store"
        archive = download.json()
        body = {"archive": archive, "id_strategy": "new"}
        preview = client.post("/api/memory-transfer/import", json=body)
        assert preview.status_code == 200 and preview.json()["dry_run"]
        applied = client.post(
            "/api/memory-transfer/import", json={**body, "dry_run": False}
        )
        assert applied.status_code == 200 and applied.json()["ok"]
        assert len(provider.list_all(MemoryType.KNOWLEDGE_BASE)) == 2
        archive["version"] = 42
        assert (
            client.post(
                "/api/memory-transfer/import", json={"archive": archive}
            ).status_code
            == 400
        )


def test_ui_read_only_allows_export_but_denies_import(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    provider = _provider(tmp_path / "memory")
    client = TestClient(create_app())
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {},
            "read_only": True,
        },
    ):
        assert client.get("/api/memory-transfer/export").status_code == 200
        assert (
            client.post(
                "/api/memory-transfer/import",
                json={"archive": MemoryArchive(provider).export(), "dry_run": False},
            ).status_code
            == 403
        )
