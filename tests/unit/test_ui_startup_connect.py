"""``memorizz ui`` connects at startup from its flags or the environment,
through the connect form's own code path."""

import os
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from memorizz.cli import operations  # noqa: E402
from memorizz.cli.app import app as cli_app  # noqa: E402
from memorizz.ui import connection, state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.security import ReadOnlyProviderProxy  # noqa: E402

pytestmark = pytest.mark.unit

ENV_KEYS = (
    "MEMORIZZ_BACKEND",
    "MEMORIZZ_MEMORY_ROOT",
    "MEMORIZZ_UI_AUTO_CONNECT",
    "MEMORIZZ_UI_READ_ONLY",
    "MEMORIZZ_UI_AUTH_TOKEN",
    "MEMORIZZ_UI_AUTH_ACCOUNTS",
    "ORACLE_USER",
    "ORACLE_PASSWORD",
    "ORACLE_DSN",
    "ORACLE_SCHEMA",
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "NOTION_TOKEN",
    "MEMORIZZ_NOTION_DATA_SOURCE_ID",
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    # Recorded first, so values the code under test sets are undone too. The
    # real os.environ stays in place: FAISS reads KMP_DUPLICATE_LIB_OK from it.
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.chdir(tmp_path)
    with patch.dict(
        state._state,
        {
            "provider": None,
            "provider_type": None,
            "connection_info": {},
            "provider_secrets": {},
            "read_only": False,
            "embedding_check": None,
            "startup_connect": None,
        },
    ):
        yield tmp_path


# --- What the environment names ------------------------------------------------


def test_nothing_named_means_no_startup_connection(home):
    assert connection.resolve_startup_connection({}) is None
    assert connection.resolve_startup_connection({"ORACLE_USER": "u"}) is None


def test_backend_and_its_own_variables_name_the_store(home):
    env = {
        "MEMORIZZ_BACKEND": "oracle",
        "ORACLE_USER": "EVAL",
        "ORACLE_PASSWORD": "pw",
        "ORACLE_DSN": "db:1521/FREEPDB1",
        "ORACLE_SCHEMA": "EVAL_SCHEMA",
    }
    startup = connection.resolve_startup_connection(env)
    assert startup["provider_type"] == "oracle"
    assert startup["params"] == {
        "oracle_user": "EVAL",
        "oracle_password": "pw",
        "oracle_dsn": "db:1521/FREEPDB1",
        "oracle_schema": "EVAL_SCHEMA",
    }
    mongo = connection.resolve_startup_connection(
        {
            "MEMORIZZ_BACKEND": "mongodb",
            "MONGODB_URI": "mongodb://db",
            "MONGODB_DB_NAME": "m",
        }
    )
    assert mongo["params"] == {"mongodb_uri": "mongodb://db", "mongodb_db_name": "m"}
    store = connection.resolve_startup_connection(
        {"MEMORIZZ_BACKEND": "filesystem", "MEMORIZZ_MEMORY_ROOT": str(home / "fs")}
    )
    assert store["params"] == {"filesystem_path": str(home / "fs")}


def test_the_provider_conventions_apply_without_memorizz_backend(home):
    oracle = {"ORACLE_USER": "u", "ORACLE_PASSWORD": "p", "ORACLE_DSN": "d"}
    assert connection.resolve_startup_connection(oracle)["provider_type"] == "oracle"
    mongo = {"MONGODB_URI": "mongodb://db"}
    assert connection.resolve_startup_connection(mongo)["provider_type"] == "mongodb"
    root = {"MEMORIZZ_MEMORY_ROOT": str(home / "fs")}
    assert connection.resolve_startup_connection(root)["params"] == {
        "filesystem_path": str(home / "fs")
    }
    # The CLI's default store alone does not connect the UI at startup.
    assert connection.resolve_startup_connection({"OPENAI_API_KEY": "x"}) is None


def test_an_incomplete_or_unknown_backend_says_why(home):
    startup = connection.resolve_startup_connection(
        {"MEMORIZZ_BACKEND": "oracle", "ORACLE_USER": "u", "ORACLE_DSN": "d"}
    )
    assert startup["error"] == (
        "MEMORIZZ_BACKEND is oracle, but ORACLE_PASSWORD is not set, so the UI "
        "did not connect at startup."
    )
    unknown = connection.resolve_startup_connection({"MEMORIZZ_BACKEND": "redis"})
    assert "MEMORIZZ_BACKEND=redis is not a memory provider" in unknown["error"]


def test_auto_connect_can_be_turned_off(home):
    env = {"MEMORIZZ_BACKEND": "filesystem", "MEMORIZZ_UI_AUTO_CONNECT": "false"}
    assert connection.resolve_startup_connection(env) is None


# --- Connecting at startup -----------------------------------------------------


def test_the_ui_connects_at_startup_from_the_environment(home, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_BACKEND", "filesystem")
    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(home / "store"))
    app = create_app(startup_connect="env")
    with TestClient(app, follow_redirects=False) as client:
        assert state._state["provider_type"] == "filesystem"
        assert client.get("/").headers["location"] == "/dashboard"
        page = client.get("/dashboard").text
    assert '<span class="ui-conn-provider">FileSystem</span>' in page


def test_startup_uses_the_connect_forms_code_path(home, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_READ_ONLY", "true")
    startup = {
        "provider_type": "filesystem",
        "source": "command line",
        "params": {"filesystem_path": str(home / "store")},
    }
    real = connection.connect_provider
    with patch.object(connection, "connect_provider", side_effect=real) as connect:
        with TestClient(create_app(startup_connect=startup)):
            pass
    args, kwargs = connect.call_args
    assert args[0] == "filesystem" and kwargs["read_only"] is True
    assert kwargs["source"] == "startup"
    # Read-only handling is the form's: the provider is wrapped.
    assert isinstance(state._state["provider"], ReadOnlyProviderProxy)
    assert state._state["connection_info"]["access"] == "read-only"


def test_a_failed_startup_connection_still_starts_and_shows_why(home, monkeypatch):
    startup = {
        "provider_type": "oracle",
        "source": "command line",
        "params": {
            "oracle_user": "EVAL",
            "oracle_dsn": "db:1521/FREEPDB1",
            "oracle_password": "never-render-me",
        },
    }
    failure = ModuleNotFoundError("No module named 'oracledb'", name="oracledb")
    with patch.object(connection, "build_provider", side_effect=failure):
        with TestClient(
            create_app(startup_connect=startup), follow_redirects=False
        ) as client:
            assert client.get("/").headers["location"] == "/connect"
            page = client.get("/connect").text
    assert "Could not connect at startup." in page
    assert (
        "Oracle needs the oracle extra: pip install &#39;memorizz[oracle]&#39;" in page
    )
    assert '<option value="oracle" selected>' in page
    assert 'value="db:1521/FREEPDB1"' in page and 'value="EVAL"' in page
    assert "never-render-me" not in page
    # The secret is dropped once used.
    assert "oracle_password" not in startup["params"]


def test_a_form_connection_clears_the_startup_error(home):
    state._state["startup_connect"] = {
        "provider_type": "oracle",
        "error": "boom",
        "fields": {},
    }
    client = TestClient(create_app(), follow_redirects=False)
    response = client.post(
        "/connect",
        data={"provider_type": "filesystem", "filesystem_path": str(home / "store")},
    )
    assert response.status_code == 302
    assert state._state["startup_connect"] is None
    assert "boom" not in client.get("/connect").text


# --- memorizz ui flags -----------------------------------------------------------


def test_flags_build_the_startup_connection_with_the_password_from_the_env(home):
    startup = operations.ui_startup_from_flags(
        "oracle",
        user="EVAL",
        dsn="db:1521/FREEPDB1",
        environ={"ORACLE_PASSWORD": "from-env"},
        prompt=lambda label: pytest.fail("asked for a password that is set"),
    )
    assert startup == {
        "provider_type": "oracle",
        "source": "command line",
        "params": {
            "oracle_user": "EVAL",
            "oracle_dsn": "db:1521/FREEPDB1",
            "oracle_password": "from-env",
        },
    }


def test_flags_ask_for_a_missing_password_and_read_other_variables(home):
    asked = []
    startup = operations.ui_startup_from_flags(
        None,
        dsn="db/FREEPDB1",
        password_env="EVAL_DB_PASSWORD",
        environ={"ORACLE_USER": "EVAL", "EVAL_DB_PASSWORD": ""},
        prompt=lambda label: asked.append(label) or "typed",
    )
    assert asked == ["Oracle password for EVAL"]
    assert startup["provider_type"] == "oracle"
    assert startup["params"]["oracle_password"] == "typed"
    mongo = operations.ui_startup_from_flags(
        "mongodb", db="evals", environ={"MONGODB_URI": "mongodb://db"}
    )
    assert mongo["params"] == {
        "mongodb_uri": "mongodb://db",
        "mongodb_db_name": "evals",
    }
    assert operations.ui_startup_from_flags(None, environ={}) is None
    with pytest.raises(ValueError, match="Unknown --backend"):
        operations.ui_startup_from_flags("redis", environ={})


def test_memorizz_ui_passes_flags_and_read_only_to_the_server(home, monkeypatch):
    monkeypatch.setenv("ORACLE_PASSWORD", "from-env")
    with patch.object(operations, "run_local", return_value=True) as run:
        result = CliRunner().invoke(
            cli_app,
            [
                "ui",
                "--backend",
                "oracle",
                "--dsn",
                "db:1521/FREEPDB1",
                "--user",
                "EVAL",
                "--read-only",
                "--port",
                "8796",
            ],
        )
    assert result.exit_code == 0, result.output
    kwargs = run.call_args.kwargs
    assert kwargs["read_only"] is True and kwargs["port"] == 8796
    assert kwargs["startup_connect"]["params"]["oracle_password"] == "from-env"
    assert "from-env" not in result.output

    with patch.object(operations, "run_local", return_value=True) as run:
        CliRunner().invoke(cli_app, ["ui"])
    assert run.call_args.kwargs["startup_connect"] == "env"
    with patch.object(operations, "run_local", return_value=True) as run:
        CliRunner().invoke(cli_app, ["ui", "--no-connect"])
    assert run.call_args.kwargs["startup_connect"] is None


def test_run_local_read_only_sets_the_switch_the_server_reads(home, monkeypatch):
    with patch("memorizz.ui.run_server") as server:
        assert operations.run_local(port=8796, startup_connect=None, read_only=True)
    assert os.environ["MEMORIZZ_UI_READ_ONLY"] == "true"
    assert server.call_args.kwargs == {
        "host": "127.0.0.1",
        "port": 8796,
        "startup_connect": None,
    }
