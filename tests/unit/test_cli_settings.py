"""CLI / REPL configuration contracts with isolated homes and no network."""

import getpass
import os
import types
import warnings

import pytest
from dotenv import dotenv_values
from prompt_toolkit.document import Document
from typer.testing import CliRunner

from memorizz import _env_io as env
from memorizz.cli import commands
from memorizz.cli import settings_commands as settings
from memorizz.cli.app import app
from memorizz.cli.repl import SafeFileHistory, SlashCompleter


@pytest.fixture
def setup_env(tmp_path, monkeypatch):
    # Direct dotenv/os.environ writes must not escape the fixture, including
    # variables that did not exist when monkeypatch.delenv was called.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for key in set(os.environ) | settings.PUBLIC_SETTINGS | {"NOTION_TOKEN"}:
        if key.startswith(("MEMORIZZ_", "NOTION_", "OPENAI_", "AZURE_OPENAI_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(env, "_ENV_SOURCES", {})
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_config_set_is_saved_only_and_reloads_on_next_launch(setup_env):
    result = CliRunner().invoke(app, ["config", "set", "MEMORIZZ_BACKEND", "notion"])
    assert result.exit_code == 0, result.output
    assert "MEMORIZZ_BACKEND" not in os.environ
    assert dotenv_values(env.resolve_env_file())["MEMORIZZ_BACKEND"] == "notion"
    assert "Restart required" in result.output and "unchanged" in result.output
    env.load_layered_env()
    assert os.environ["MEMORIZZ_BACKEND"] == "notion"


def test_secret_hidden_not_argument_or_output(setup_env, monkeypatch):
    secret = 'private "token" #value'
    monkeypatch.setattr(getpass, "getpass", lambda *args: secret)
    result = CliRunner().invoke(app, ["config", "set", "NOTION_TOKEN"])
    assert result.exit_code == 0, result.output
    assert secret not in result.output
    assert dotenv_values(env.resolve_env_file())["NOTION_TOKEN"] == secret
    result = CliRunner().invoke(app, ["config", "get", "NOTION_TOKEN"])
    assert result.exit_code == 0
    assert "hidden" in result.output and secret not in result.output


@pytest.mark.parametrize("key", ["NOTION_TOKEN", "UNKNOWN_INTEGRATION", "MONGODB_URI"])
def test_inline_secrets_rejected_without_echoing(setup_env, key):
    result = CliRunner().invoke(app, ["config", "set", key, "private-password"])
    assert result.exit_code == 2
    assert "hidden input" in result.output
    assert "private-password" not in result.output
    assert not env.resolve_env_file().exists()


def test_getpass_echo_fallback_is_disabled(setup_env, monkeypatch):
    def warning(*args):
        warnings.warn("cannot hide", getpass.GetPassWarning)
        pytest.fail("Must abort before echoed input")

    monkeypatch.setattr(getpass, "getpass", warning)
    result = CliRunner().invoke(app, ["config", "set", "NOTION_TOKEN"])
    assert result.exit_code == 2 and "Hidden input is unavailable" in result.output
    assert not env.resolve_env_file().exists()


def test_project_target_and_override_warning(setup_env, monkeypatch):
    env.update_env_file(setup_env / ".env", {"MEMORIZZ_BACKEND": "filesystem"})
    result = CliRunner().invoke(app, ["config", "set", "MEMORIZZ_BACKEND", "notion"])
    assert "project .env takes precedence" in result.output
    result = CliRunner().invoke(
        app, ["config", "set", "MEMORIZZ_BACKEND", "notion", "--project"]
    )
    assert result.exit_code == 0, result.output
    assert "project .env takes precedence" not in result.output
    assert dotenv_values(setup_env / ".env")["MEMORIZZ_BACKEND"] == "notion"
    assert os.environ["MEMORIZZ_BACKEND"] == "filesystem"


def test_path_honors_env_override_and_custom_file_explains_loading(
    setup_env, monkeypatch
):
    custom = setup_env / "custom.env"
    result = CliRunner().invoke(app, ["config", "path", "--env-file", str(custom)])
    assert result.exit_code == 0 and "not automatically loaded" in result.output
    monkeypatch.setenv("MEMORIZZ_ENV_FILE", str(custom))
    result = CliRunner().invoke(app, ["config", "path"])
    assert str(custom) in result.output
    result = CliRunner().invoke(
        app, ["config", "set", "MEMORIZZ_BACKEND", "filesystem"]
    )
    assert result.exit_code == 0 and custom.is_file()


@pytest.mark.parametrize(
    "args",
    [
        ["MEMORIZZ_BACKEND", "invalid"],
        ["MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "-1"],
        ["MEMORIZZ_NOTION_DATA_SOURCE_ID", "bad-id"],
        ["KEY=private-password"],
        ["MEMORIZZ_HOME", "elsewhere"],
        ["MEMORIZZ_BACKEND", "notion", "--project", "--env-file", "x"],
    ],
)
def test_invalid_settings_are_not_saved(setup_env, args):
    result = CliRunner().invoke(app, ["config", "set", *args])
    assert result.exit_code == 2, result.output
    assert "private-password" not in result.output
    assert not env.resolve_env_file().exists()


def test_help_and_keys_are_available_without_llm_startup(setup_env, monkeypatch):
    monkeypatch.setattr(
        "memorizz.cli.agent_factory.build_session_agent",
        lambda *a, **k: pytest.fail("No agent should start"),
    )
    for args in (
        ["--help"],
        ["config", "--help"],
        ["config", "keys"],
        ["memory", "configure", "--help"],
        ["notion", "connect", "--help"],
    ):
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output


def test_login_notion_maps_token_and_does_not_switch_provider(setup_env, monkeypatch):
    from tests.unit.test_cli_login import _Console

    secret = "private-notion-token"
    monkeypatch.setattr(getpass, "getpass", lambda *args: secret)
    original = object()
    session = types.SimpleNamespace(console=_Console(), provider=original)
    commands.cmd_login(session, "notion")
    assert dotenv_values(env.resolve_env_file())["NOTION_TOKEN"] == secret
    assert "NOTION" not in dotenv_values(env.resolve_env_file())
    assert session.provider is original
    output = " ".join(session.console.lines)
    assert "Restart" in output and "/memory-provider notion" in output
    assert secret not in output


def test_repl_uses_same_settings_editor(setup_env, capsys):
    from tests.unit.test_cli_login import _Console

    session = types.SimpleNamespace(console=_Console())
    commands.cmd_config(session, "set MEMORIZZ_BACKEND notion --project")
    assert dotenv_values(setup_env / ".env")["MEMORIZZ_BACKEND"] == "notion"
    assert "Restart required" in capsys.readouterr().out


def test_repl_argument_errors_do_not_escape_or_echo_private_input(setup_env, capsys):
    from tests.unit.test_cli_login import _Console

    session = types.SimpleNamespace(console=_Console())
    commands.cmd_config(session, "get OPENAI_API_KEY --private-token")
    output = " ".join(session.console.lines) + capsys.readouterr().out
    assert "Invalid arguments" in output
    assert "private-token" not in output


def test_memory_setup_repl_help_has_no_extra_configure_verb(setup_env, capsys):
    from tests.unit.test_cli_login import _Console

    session = types.SimpleNamespace(console=_Console())
    commands.cmd_memory_provider(session, "--help")
    output = capsys.readouterr().out
    assert "/memory-provider" in output and "PROVIDER" in output
    assert "/memory-provider configure" not in output
    assert not env.resolve_env_file().exists()


def test_existing_memory_abbreviations_still_resolve(setup_env, monkeypatch):
    from tests.unit.test_cli_login import _Console

    received = []
    monkeypatch.setitem(
        commands.COMMANDS,
        "memory",
        commands.Command(lambda session, args: received.append(args), "", ""),
    )
    session = types.SimpleNamespace(console=_Console())
    for name in ("mem", "memo", "memor", "memory"):
        commands.dispatch(f"/{name} thread-id", session)
    assert received == ["thread-id"] * 4


@pytest.mark.parametrize(
    "line",
    [
        "/login notion private-token",
        "/LO private-token",
        "/config set NOTION_TOKEN private-token",
        " /co set UNKNOWN private-token",
        "/memory-provider private-token",
        "/memory-p private-token",
    ],
)
def test_sensitive_commands_not_retained_in_history(setup_env, line):
    history = SafeFileHistory(str(setup_env / "history"))
    history.append_string(line)
    history.store_string(line)
    assert history.get_strings() == []
    assert not (setup_env / "history").exists()


def test_old_sensitive_history_is_filtered_without_deleting_file(setup_env):
    path = setup_env / "history"
    path.write_text("\n# old\n+/login notion private-token\n\n# old\n+hello\n")
    history = SafeFileHistory(str(path))
    assert list(history.load_history_strings()) == ["hello"]
    assert "private-token" in path.read_text()  # Do not silently rewrite user history.


def test_completion_offers_settings_and_memory_provider_without_values():
    completer = SlashCompleter(commands.command_completions())
    for prefix, expected in (
        ("/config set MEMORIZZ_NOTION_D", "/config set MEMORIZZ_NOTION_DATA_SOURCE_ID"),
        ("/login n", "/login notion"),
        ("/memory-provider n", "/memory-provider notion"),
    ):
        assert expected in [
            row.text for row in completer.get_completions(Document(prefix), None)
        ]


@pytest.mark.parametrize(
    "value",
    [
        "f8346b05-8564-497f-9240-6bb512b078c1",
        "https://app.notion.com/p/Name-f8346b058564497f92406bb512b078c1?source=copy_link",
        "https://www.notion.so/f8346b058564497f92406bb512b078c1?v=unrelated-view",
    ],
)
def test_notion_identifier_accepts_real_urls_and_ids(value):
    assert settings.notion_identifier(value) == "f8346b05-8564-497f-9240-6bb512b078c1"


@pytest.mark.parametrize(
    "value",
    [
        "https://notion.so.attacker.invalid/f8346b058564497f92406bb512b078c1",
        "http://notion.so/f8346b058564497f92406bb512b078c1",
        "https://attacker.invalid/private-token",
        "private-token",
    ],
)
def test_notion_identifier_rejects_other_hosts_without_reflecting_input(value):
    with pytest.raises(ValueError) as caught:
        settings.notion_identifier(value)
    assert "private-token" not in str(caught.value)
