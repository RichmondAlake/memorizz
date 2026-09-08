"""Release notices stay useful without blocking or polluting CLI output."""

import json
import shlex
from io import StringIO
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import requests
from rich.console import Console
from typer.testing import CliRunner

from memorizz.cli import repl, updates
from memorizz.cli.app import app

pytestmark = pytest.mark.unit


@pytest.fixture
def release_api(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    for name in ("CI", "MEMORIZZ_NO_UPDATE_CHECK", "MEMORIZZ_INSTALL_METHOD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(updates.sys, "prefix", str(tmp_path / "venv"))
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = {
        "info": {"version": "0.10.0", "requires_python": ">=3.10", "yanked": False}
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(updates.requests, "get", get)
    return get, response


@pytest.mark.parametrize(
    ("installed", "latest", "expected"),
    [
        ("0.2.2", "0.9.0", True),
        ("0.9.0", "0.10.0", True),
        ("0.9.9", "0.9.10", True),
        ("0.9.0", "0.9.0", False),
        ("1.0.0", "0.9.0", False),
        ("0.9.0+local", "0.9.0", False),
        ("0.10.0rc1", "0.10.0", True),
        ("0.9.0", "0.10.0rc1", False),
        ("0.9.0", "0.10.0.dev1", False),
        ("0.9.0", "0.10.0+local", False),
        ("0.9.0", "0.9.0.post1", True),
        ("unknown", "0.10.0", False),
        ("0.9.0", "invalid", False),
    ],
)
def test_only_newer_stable_versions_are_announced(
    release_api, installed, latest, expected
):
    _, response = release_api
    response.json.return_value["info"]["version"] = latest

    notice = updates.check_for_update(installed)

    assert bool(notice) is expected
    if expected:
        assert f"memorizz {installed} → {latest}" in notice
        assert "Upgrade:" in notice


def test_daily_cache_refreshes_and_stops_notifying_after_upgrade(release_api, tmp_path):
    get, response = release_api
    assert updates.check_for_update("0.9.0")
    assert updates.check_for_update("0.9.0")
    assert updates.check_for_update("0.10.0") is None
    get.assert_called_once()
    assert get.call_args.args == (updates.PYPI_URL,)
    assert get.call_args.kwargs["timeout"] == (2, 2)

    cache_path = tmp_path / "update-check.json"
    cache = json.loads(cache_path.read_text())
    cache["checked_at"] -= 24 * 60 * 60
    cache_path.write_text(json.dumps(cache))
    response.json.return_value["info"]["version"] = "0.11.0"

    assert "0.10.0 → 0.11.0" in updates.check_for_update("0.10.0")
    assert get.call_count == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {"version": "0.10.0", "yanked": True},
        {"version": "0.10.0", "requires_python": ">=99"},
        {"version": "0.10.0", "requires_python": "broken"},
        {"version": None},
        {},
        [],
    ],
)
def test_unavailable_or_malformed_releases_are_quiet(release_api, metadata):
    _, response = release_api
    response.json.return_value = {"info": metadata}
    assert updates.check_for_update("0.9.0") is None


@pytest.mark.parametrize(
    "error",
    [
        requests.Timeout(),
        requests.ConnectionError(),
        requests.HTTPError(),
        ValueError(),
    ],
)
def test_network_and_json_failures_are_quiet(release_api, error):
    get, _ = release_api
    get.side_effect = error
    assert updates.check_for_update("0.9.0") is None


def test_known_update_remains_available_offline(release_api, tmp_path):
    get, _ = release_api
    assert updates.check_for_update("0.9.0")
    cache_path = tmp_path / "update-check.json"
    cache = json.loads(cache_path.read_text())
    cache["checked_at"] = 0
    cache_path.write_text(json.dumps(cache))
    get.side_effect = requests.ConnectionError()

    assert "0.9.0 → 0.10.0" in updates.check_for_update("0.9.0")
    assert updates.check_for_update("0.10.0") is None


@pytest.mark.parametrize("cache", ["broken json", "[]", '{"checked_at": "never"}'])
def test_corrupted_cache_is_replaced(release_api, tmp_path, cache):
    (tmp_path / "update-check.json").write_text(cache)
    assert updates.check_for_update("0.9.0")
    assert (
        json.loads((tmp_path / "update-check.json").read_text())["latest_version"]
        == "0.10.0"
    )


def test_cache_is_refreshed_for_a_different_python(release_api, tmp_path):
    get, _ = release_api
    assert updates.check_for_update("0.9.0")
    cache_path = tmp_path / "update-check.json"
    cache = json.loads(cache_path.read_text())
    cache["python_version"] = "3.9.0"
    cache_path.write_text(json.dumps(cache))

    assert updates.check_for_update("0.9.0")
    assert get.call_count == 2


def test_unwritable_cache_does_not_hide_update(release_api, monkeypatch):
    monkeypatch.setattr(updates.cfg, "ensure_home", Mock(side_effect=PermissionError()))
    assert updates.check_for_update("0.9.0")


@pytest.mark.parametrize("name", ["MEMORIZZ_NO_UPDATE_CHECK", "CI"])
def test_opt_out_skips_network_and_cache(release_api, monkeypatch, tmp_path, name):
    get, _ = release_api
    monkeypatch.setenv(name, "true")
    assert updates.check_for_update("0.9.0") is None
    get.assert_not_called()
    assert not (tmp_path / "update-check.json").exists()


@pytest.mark.parametrize(
    ("marker", "command"),
    [
        ("uv-receipt.toml", "uv tool upgrade memorizz"),
        ("pipx_metadata.json", "pipx upgrade memorizz"),
    ],
)
def test_managed_python_upgrade_commands(release_api, tmp_path, marker, command):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    (prefix / marker).touch()
    assert updates.upgrade_command() == command


def test_npm_command_takes_precedence_over_managed_python(
    release_api, monkeypatch, tmp_path
):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").touch()
    monkeypatch.setenv("MEMORIZZ_INSTALL_METHOD", "npm")
    assert updates.upgrade_command() == "npm install -g memorizz@latest"


def test_homebrew_upgrade_command(release_api, monkeypatch, tmp_path):
    monkeypatch.setattr(
        updates.sys, "prefix", str(tmp_path / "Cellar/memorizz/0.9.0/libexec")
    )
    assert updates.upgrade_command() == "brew upgrade memorizz"


def test_pip_command_targets_and_quotes_running_python(release_api, monkeypatch):
    executable = "/Applications/Python Env/bin/python"
    monkeypatch.setattr(updates.sys, "executable", executable)
    assert shlex.split(updates.upgrade_command()) == [
        executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "memorizz",
    ]


@pytest.fixture
def workers(monkeypatch):
    threads = []

    def start_thread(**kwargs):
        thread = Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(updates, "Thread", start_thread)
    yield threads
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()


def test_slow_lookup_does_not_block_session_or_exit(release_api, workers, monkeypatch):
    started, release = Event(), Event()
    console = SimpleNamespace(is_terminal=True, is_dumb_terminal=False, print=Mock())

    def slow_check():
        started.set()
        release.wait(timeout=2)
        return "Update available"

    monkeypatch.setattr(updates, "check_for_update", slow_check)
    try:
        with updates.update_notifier(console):
            assert started.wait(timeout=1)
            assert workers[0].daemon
        # Context exit completes even while the network worker is blocked.
        assert workers[0].is_alive()
    finally:
        release.set()
    workers[0].join(timeout=2)
    console.print.assert_not_called()


@pytest.mark.parametrize(("is_terminal", "is_dumb"), [(False, False), (True, True)])
def test_noninteractive_notifier_skips_network(
    release_api, workers, is_terminal, is_dumb
):
    get, _ = release_api
    console = SimpleNamespace(is_terminal=is_terminal, is_dumb_terminal=is_dumb)
    with updates.update_notifier(console):
        pass
    assert not workers
    get.assert_not_called()


def test_repl_shows_notice_while_waiting_for_input(release_api, workers, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_INSTALL_METHOD", "npm")
    monkeypatch.setenv("TERM", "xterm-256color")
    release_api[1].json.return_value["info"]["version"] = "99.0.0"
    output = StringIO()
    printed = Event()

    class RecordingConsole(Console):
        def print(self, *args, **kwargs):
            super().print(*args, **kwargs)
            if "Update available" in output.getvalue():
                printed.set()

    console = RecordingConsole(
        file=output, force_terminal=True, color_system=None, width=160
    )

    def prompt(_):
        assert printed.wait(timeout=2), output.getvalue()
        return "/exit"

    monkeypatch.setattr(
        repl, "PromptSession", Mock(return_value=SimpleNamespace(prompt=prompt))
    )
    monkeypatch.setattr(repl.commands, "dispatch", Mock(return_value=False))
    session = SimpleNamespace(
        console=console,
        code_mode=False,
        provider=None,
        provider_name="ollama",
        model_name="qwen2.5:7b",
        agent=SimpleNamespace(),
        warnings=[],
    )

    repl.run_repl(session)

    assert "Update available:" in output.getvalue()
    assert "npm install -g memorizz@latest" in output.getvalue()
    repl.commands.dispatch.assert_called_once_with("/exit", session)


@pytest.mark.parametrize(
    "args", [["--help"], ["--version"], ["capabilities", "--json"]]
)
def test_machine_and_metadata_commands_never_check_for_updates(release_api, args):
    get, _ = release_api
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Update available" not in result.output
    get.assert_not_called()
