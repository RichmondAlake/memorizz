"""Text the CLI didn't write itself is shown as it is, square brackets included."""

import ast
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from memorizz import cli
from memorizz.cli import agent_factory, commands

pytestmark = pytest.mark.unit

_FAILURE_NAMES = {"e", "err", "error", "exc", "exception", "warning"}


def _unescaped_failures(tree):
    """Rich print calls that put a failure's text into their markup as it is."""
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if call.func.attr != "print" or "console" not in ast.unparse(call.func.value):
            continue
        if any(keyword.arg == "markup" for keyword in call.keywords):
            continue
        for argument in call.args:
            for value in ast.walk(argument):
                if isinstance(value, ast.FormattedValue):
                    shown = ast.unparse(value.value)
                    leaf = shown.rsplit(".", 1)[-1]
                    named = leaf in _FAILURE_NAMES or "['error']" in shown
                    if named and "escape(" not in shown:
                        yield call.lineno, shown


def test_the_cli_escapes_failures_it_prints():
    found = [
        f"{path.name}:{line} {shown}"
        for path in sorted(Path(cli.__file__).parent.glob("*.py"))
        for line, shown in _unescaped_failures(ast.parse(path.read_text()))
    ]
    assert found == []


def test_a_failure_keeps_its_square_brackets(monkeypatch):
    def refuse(_name):
        raise RuntimeError("Install it: pip install 'memorizz[ui,oracle]'")

    monkeypatch.setattr(agent_factory, "make_browser_control_config", refuse)
    output = StringIO()
    session = SimpleNamespace(
        agent=SimpleNamespace(),
        console=Console(file=output, color_system=None, width=200),
    )
    commands.cmd_browser(session, "on")
    assert "pip install 'memorizz[ui,oracle]'" in output.getvalue()
