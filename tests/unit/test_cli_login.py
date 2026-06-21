# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""/login ollama -> Ollama Cloud sign-in (for ':cloud' models)."""

import shutil
import subprocess
import types

import pytest

from memorizz.cli import commands


class _Console:
    def __init__(self):
        self.lines = []

    def print(self, *a, **k):
        self.lines.append(" ".join(str(x) for x in a))


@pytest.mark.unit
def test_login_lists_ollama():
    assert "ollama" in [p[0] for p in commands._LOGIN_PROVIDERS]


@pytest.mark.unit
def test_login_ollama_runs_signin(monkeypatch):
    calls = {}
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/local/bin/ollama")

    class _Result:
        returncode = 0

    def _run(cmd, *a, **k):
        calls["cmd"] = cmd
        return _Result()

    monkeypatch.setattr(subprocess, "run", _run)
    session = types.SimpleNamespace(console=_Console())
    commands.cmd_login(session, "ollama")
    assert calls["cmd"] == ["ollama", "signin"]


@pytest.mark.unit
def test_login_ollama_missing_cli(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    def _boom(*a, **k):  # must NOT be called when the CLI is absent
        raise AssertionError("subprocess.run should not run without the ollama CLI")

    monkeypatch.setattr(subprocess, "run", _boom)
    session = types.SimpleNamespace(console=_Console())
    commands.cmd_login(session, "ollama")
    out = "\n".join(session.console.lines).lower()
    assert "not found" in out and "ollama signin" in out
