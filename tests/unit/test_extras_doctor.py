"""Missing optional extras become instructions, and ``memorizz doctor`` lists them."""

import json
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

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
