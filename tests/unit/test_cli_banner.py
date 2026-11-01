"""Tests for the interactive CLI startup banner."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz.cli import repl


@pytest.mark.unit
@pytest.mark.parametrize(
    ("manager", "expected_status"),
    [(object(), "enabled"), (None, "disabled")],
)
def test_banner_reports_installed_version_and_live_learning_state(
    monkeypatch, manager, expected_status
):
    monkeypatch.setattr(repl, "__version__", "9.8.7")
    output = StringIO()
    console = Console(file=output, color_system=None, width=120)
    session = SimpleNamespace(
        code_mode=False,
        provider=SimpleNamespace(root_path="/tmp/memorizz-memory"),
        provider_name="ollama",
        model_name="qwen2.5:7b",
        agent=SimpleNamespace(continual_learning_manager=manager),
        warnings=[],
    )

    repl._banner(console, session)

    rendered = output.getvalue()
    assert "memorizz 9.8.7 — memory assistant" in rendered
    assert f"learning: continual {expected_status}" in rendered


@pytest.mark.unit
def test_banner_shows_the_upgrade_command_with_its_extras(monkeypatch):
    command = "uv tool install --force 'memorizz[mcp,ui]==9.9.9'"
    monkeypatch.setattr(
        repl, "known_update", lambda: {"latest": "9.9.9", "command": command}
    )
    output = StringIO()
    console = Console(file=output, color_system=None, width=160)
    session = SimpleNamespace(
        code_mode=False,
        provider=SimpleNamespace(root_path="/tmp/memorizz-memory"),
        provider_name="ollama",
        model_name="qwen2.5:7b",
        agent=SimpleNamespace(continual_learning_manager=None),
        warnings=[],
    )

    repl._banner(console, session)

    assert command in output.getvalue()


@pytest.mark.unit
def test_banner_warnings_keep_their_square_brackets():
    hint = "Oracle needs the oracle extra: pip install 'memorizz[oracle]'"
    output = StringIO()
    console = Console(file=output, color_system=None, width=160)
    session = SimpleNamespace(
        code_mode=False,
        provider=SimpleNamespace(root_path="/tmp/memorizz-memory"),
        provider_name="ollama",
        model_name="qwen2.5:7b",
        agent=SimpleNamespace(continual_learning_manager=None),
        warnings=[hint],
    )

    repl._banner(console, session)

    assert hint in output.getvalue()
