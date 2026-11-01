"""Missing optional extras become instructions, and ``memorizz doctor`` lists them."""

import asyncio
import json
import re
import shlex
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from typer.testing import CliRunner

import memorizz
from memorizz import extras
from memorizz.cli.app import app as cli_app

pytestmark = pytest.mark.unit

ORACLE_HINT = (
    "Oracle needs the oracle extra: pip install 'memorizz[oracle]' "
    "(or uv tool install --force 'memorizz[ui,oracle]')"
)


def _oracle_provider_import_error():
    """What importing the Oracle provider raises without oracledb."""
    try:
        try:
            raise ModuleNotFoundError("No module named 'oracledb'", name="oracledb")
        except ImportError:
            raise ImportError(
                "oracledb package is required for Oracle provider. "
                "Install it with: pip install oracledb"
            )
    except ImportError as exc:
        return exc


def test_extras_come_from_pyproject_optional_dependencies():
    declared = extras.declared_extras()
    for name in ("oracle", "mongodb", "ui", "mcp", "filesystem", "notion"):
        assert name in declared
    rows = {row["extra"]: row for row in extras.extras_report()}
    assert rows["oracle"]["modules"] == ["oracledb"]
    assert rows["mongodb"]["modules"] == ["pymongo"]
    assert rows["filesystem"]["modules"] == ["faiss"]
    assert set(rows["ui"]["modules"]) == {"fastapi", "uvicorn", "jinja2", "multipart"}
    assert rows["notion"]["status"] == "built-in"
    # The everything-bundle and tooling are not features.
    assert not {"all", "dev", "docs"} & set(rows)


def test_an_extra_is_installed_only_when_all_its_modules_import(monkeypatch):
    missing = {"oracledb", "jinja2"}
    monkeypatch.setattr(extras, "_importable", lambda module: module not in missing)
    rows = {row["extra"]: row for row in extras.extras_report()}
    assert rows["oracle"]["status"] == "missing"
    assert rows["oracle"]["install"] == "pip install 'memorizz[oracle]'"
    assert rows["ui"]["status"] == "partial" and rows["ui"]["missing"] == ["jinja2"]
    assert rows["mongodb"]["status"] == "installed"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            ModuleNotFoundError("No module named 'oracledb'", name="oracledb"),
            ORACLE_HINT,
        ),
        (_oracle_provider_import_error(), ORACLE_HINT),
        (
            ModuleNotFoundError("No module named 'pymongo'", name="pymongo"),
            "MongoDB needs the mongodb extra: pip install 'memorizz[mongodb]' "
            "(or uv tool install --force 'memorizz[ui,mongodb]')",
        ),
        (
            ModuleNotFoundError("No module named 'bson'", name="bson"),
            "MongoDB needs the mongodb extra: pip install 'memorizz[mongodb]' "
            "(or uv tool install --force 'memorizz[ui,mongodb]')",
        ),
        (ValueError("Oracle requires user, password, and DSN"), None),
        (ModuleNotFoundError("No module named 'nothing_we_ship'"), None),
    ],
)
def test_missing_driver_messages(error, message):
    assert extras.missing_extra_message(error) == message


def test_the_cli_names_the_extra_without_the_ui():
    error = ModuleNotFoundError("No module named 'oracledb'", name="oracledb")
    assert extras.missing_extra_message(error, with_ui=False) == (
        "Oracle needs the oracle extra: pip install 'memorizz[oracle]' "
        "(or uv tool install --force 'memorizz[oracle]')"
    )


def test_the_connect_page_shows_the_extra_to_install(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from memorizz.ui import connection, state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORIZZ_UI_READ_ONLY", raising=False)
    with (
        patch.dict(state._state, {"provider": None, "provider_type": None}),
        patch.object(
            connection, "build_provider", side_effect=_oracle_provider_import_error()
        ),
    ):
        page = TestClient(create_app()).post(
            "/connect",
            data={
                "provider_type": "oracle",
                "oracle_user": "EVAL",
                "oracle_password": "pw",
                "oracle_dsn": "db/FREEPDB1",
            },
        )
    assert page.status_code == 200
    assert (
        "Oracle needs the oracle extra: pip install &#39;memorizz[oracle]&#39; "
        "(or uv tool install --force &#39;memorizz[ui,oracle]&#39;)"
    ) in page.text


def test_cli_provider_detection_raises_the_instruction(monkeypatch):
    from memorizz.cli import agent_factory

    monkeypatch.setattr(
        agent_factory,
        "_detect_memory_provider",
        lambda *_: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'oracledb'", name="oracledb")
        ),
    )
    with pytest.raises(ModuleNotFoundError) as caught:
        agent_factory.detect_memory_provider({}, [])
    assert str(caught.value).startswith("Oracle needs the oracle extra")


def test_doctor_lists_extras_and_checks_the_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORIZZ_BACKEND", "oracle")
    monkeypatch.setattr(extras, "_importable", lambda module: module != "oracledb")
    result = CliRunner().invoke(cli_app, ["doctor", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.output)
    rows = {row["extra"]: row for row in report["extras"]}
    assert rows["oracle"]["installed"] is False and rows["ui"]["installed"] is True
    assert report["backend"]["install"] == (
        "pip install 'memorizz[oracle]' (or uv tool install --force 'memorizz[ui,oracle]')"
    )

    monkeypatch.setattr(extras, "_importable", lambda module: True)
    result = CliRunner().invoke(cli_app, ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Optional extras" in result.output
    assert "Memory backend: oracle (MEMORIZZ_BACKEND)" in result.output


UV_RECEIPT = """[tool]
requirements = [{ name = "memorizz", extras = ["ui", "oracle"], specifier = "==0.17.11" }]
python = "3.12"
"""


@pytest.fixture
def uv_tool(tmp_path, monkeypatch):
    """Run as a uv tool: a receipt beside the interpreter, as uv writes it."""
    monkeypatch.delenv("MEMORIZZ_INSTALL_METHOD", raising=False)

    def install(receipt=UV_RECEIPT):
        (tmp_path / "uv-receipt.toml").write_text(receipt)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))

    return install


def test_a_uv_tool_adds_an_extra_to_the_ones_it_has(uv_tool):
    uv_tool()
    assert shlex.split(extras.install_command(["mcp"])) == [
        "uv",
        "tool",
        "install",
        "--force",
        "--refresh-package",
        "memorizz",
        "--python",
        "3.12",
        "memorizz[ ui,oracle,mcp]==0.17.11",
    ]


def test_a_uv_tool_does_not_repeat_an_extra_it_has(uv_tool):
    uv_tool()
    assert "memorizz[ ui,oracle]==0.17.11" in extras.install_command(["oracle"])


def test_an_unpinned_uv_tool_stays_unpinned(uv_tool):
    uv_tool(UV_RECEIPT.replace(', specifier = "==0.17.11"', ""))
    assert extras.install_command(["mcp"]).endswith("'memorizz[ ui,oracle,mcp]'")


def test_a_uv_tool_with_no_extras_gains_the_one_it_needs(uv_tool):
    uv_tool(UV_RECEIPT.replace('extras = ["ui", "oracle"], ', ""))
    assert extras.install_command(["mcp"]).endswith("'memorizz[ mcp]==0.17.11'")


def test_a_uv_tool_is_not_sent_to_pip(uv_tool):
    uv_tool()
    message = extras.missing_extra_message(ImportError("No module named 'oracledb'"))
    assert message.startswith("Oracle needs the oracle extra: uv tool install --force")
    assert "pip install" not in message and "(or " not in message


def test_doctor_gives_a_uv_tool_one_command(uv_tool, tmp_path, monkeypatch):
    uv_tool(UV_RECEIPT.replace('"ui", "oracle"', '"ui"'))
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORIZZ_BACKEND", "oracle")
    monkeypatch.setattr(extras, "_importable", lambda module: module != "oracledb")
    result = CliRunner().invoke(cli_app, ["doctor", "--json"])
    report = json.loads(result.output)
    assert "memorizz[ ui,oracle]==0.17.11" in report["backend"]["install"]
    assert "pip install" not in report["backend"]["install"]


def test_an_npm_install_keeps_the_pip_hint(uv_tool, monkeypatch):
    # The launcher's runtime has a receipt, but it isn't the person's own tool.
    uv_tool()
    monkeypatch.setenv("MEMORIZZ_INSTALL_METHOD", "npm")
    assert extras.install_command(["mcp"]) == "pip install 'memorizz[mcp]'"


def test_an_mcp_connection_tells_a_uv_tool_how_to_add_the_extra(uv_tool, monkeypatch):
    from memorizz.mcp import MCPClientManager, MCPConfigurationError

    uv_tool()
    monkeypatch.setitem(sys.modules, "mcp", None)
    manager = MCPClientManager(
        owner_id="agent-1",
        servers=[{"name": "notion", "transport": "stdio", "command": "x"}],
    )

    async def connect():
        async with manager._client(manager._servers["notion"]):
            pass

    with pytest.raises(MCPConfigurationError) as error:
        asyncio.run(connect())
    assert "memorizz[ ui,oracle,mcp]==0.17.11" in str(error.value)
    assert "pip install" not in str(error.value)


def test_encrypted_credentials_tell_a_uv_tool_how_to_add_the_extra(
    uv_tool, monkeypatch
):
    from memorizz.mcp import MCPConfigurationError, credentials

    uv_tool()
    monkeypatch.setattr(credentials, "Fernet", None)
    with pytest.raises(MCPConfigurationError) as error:
        credentials._require_cryptography()
    assert "memorizz[ ui,oracle,mcp]==0.17.11" in str(error.value)


def test_rich_cannot_delete_the_extras_from_a_uv_hint(uv_tool):
    # Printed without escaping, "[ui,oracle,mcp]" would be read as a style tag.
    # What was left would reinstall the tool with no extras and remove them.
    uv_tool()
    command = extras.install_command(["mcp"])
    output = StringIO()
    Console(file=output, color_system=None, width=200).print(f"Install: {command}")
    assert command in output.getvalue()


def test_no_install_hint_is_written_out_by_hand():
    # Only install_command knows how this install adds an extra.
    written_out = re.compile(r"(pip install|uv tool install)\W*(--force\W*)?memorizz\[")
    package = Path(memorizz.__file__).parent
    found = [
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.suffix in {".py", ".html", ".js"}
        and path != package / "extras.py"
        and written_out.search(path.read_text(encoding="utf-8"))
    ]
    assert found == []


def test_fine_tuning_reports_how_to_add_its_extra(uv_tool):
    from memorizz.finetune.service import environment

    uv_tool()
    assert "memorizz[ ui,oracle,finetune]==0.17.11" in environment()["install"]


def test_pages_can_show_how_to_add_an_extra(uv_tool):
    from memorizz.ui.state import templates

    uv_tool()
    page = templates.env.from_string(
        "{{ install_command(['huggingface'])|tojson }}"
    ).render()
    assert "memorizz[ ui,oracle,huggingface]==0.17.11" in page
    assert "'" not in page  # it sits inside a script's string
