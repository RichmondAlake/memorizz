"""CLI coverage for MCP connection configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from memorizz import __version__
from memorizz.cli.app import app

SERVER = Path(__file__).parents[1] / "fixtures" / "mcp_stdio_server.py"
runner = CliRunner()


@pytest.mark.unit
def test_cli_add_list_test_and_remove_stdio_server(tmp_path):
    env = {"MEMORIZZ_HOME": str(tmp_path)}
    added = runner.invoke(
        app,
        [
            "mcp",
            "add",
            "local",
            "--transport",
            "stdio",
            "--command",
            sys.executable,
            "--arg",
            str(SERVER),
        ],
        env=env,
    )
    assert added.exit_code == 0, added.output
    config_file = next((tmp_path / "mcp_servers").glob("*.json"))
    config = json.loads(config_file.read_text())
    assert config["servers"][0]["name"] == "local"

    listed = runner.invoke(app, ["mcp", "list", "--json"], env=env)
    assert listed.exit_code == 0, listed.output
    assert "local" in listed.output

    tested = runner.invoke(app, ["mcp", "test", "local", "--json"], env=env)
    assert tested.exit_code == 0, tested.output
    assert '"tool_count": 2' in tested.output

    removed = runner.invoke(app, ["mcp", "remove", "local"], env=env)
    assert removed.exit_code == 0, removed.output
    assert json.loads(config_file.read_text())["servers"] == []


@pytest.mark.unit
def test_cli_notion_and_google_presets_are_safe(tmp_path):
    env = {"MEMORIZZ_HOME": str(tmp_path)}
    notion = runner.invoke(app, ["mcp", "add", "notion", "--preset", "notion"], env=env)
    assert notion.exit_code == 0, notion.output

    google_missing_client = runner.invoke(
        app,
        ["mcp", "add", "calendar", "--preset", "google-calendar"],
        env=env,
    )
    assert google_missing_client.exit_code == 2

    google = runner.invoke(
        app,
        [
            "mcp",
            "add",
            "calendar",
            "--preset",
            "google-calendar",
            "--client-id",
            "test-client.apps.googleusercontent.com",
            "--client-secret",
            "encrypted-secret",
        ],
        env=env,
    )
    assert google.exit_code == 0, google.output
    public_text = next((tmp_path / "mcp_servers").glob("*.json")).read_text()
    assert "encrypted-secret" not in public_text
    assert "calendarmcp.googleapis.com/mcp/v1" in public_text
    assert b"encrypted-secret" not in (tmp_path / "mcp_credentials.enc").read_bytes()


@pytest.mark.unit
def test_cli_capability_report_matches_current_mcp_surface():
    result = runner.invoke(app, ["capabilities", "--json"])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["version"] == __version__
    assert report["features"]["mcp_client"]["available"] is True
    assert report["features"]["mcp_server"]["available"] is True
    assert report["features"]["mcp_server"]["tool_count"] == 24
    assert report["features"]["mcp_server"]["strict_input_schemas"] is True
    assert report["features"]["agent_creation"] == {
        "available": True,
        "cli": True,
        "mcp_local_stdio": True,
        "mcp_remote_http": False,
        "sdk": True,
        "ui": True,
    }
    assert report["features"]["headless_runtime"] == {
        "available": True,
        "cli": True,
        "display_server_required": False,
        "mcp_stdio": True,
        "sdk": True,
        "ui_dependency_required": False,
    }
    assert (
        "approval_decisions"
        in report["features"]["agent_surface_parity"]["trusted_host_only"]
    )
    assert report["features"]["durable_approvals"]["single_use"] is True
