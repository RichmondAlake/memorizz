"""Fresh-process configuration and actual terminal echo suppression; no services."""

import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest
from dotenv import dotenv_values

import memorizz


def _runtime(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime = dict(os.environ)
    for key in list(runtime):
        if key.startswith(("MEMORIZZ_", "NOTION_", "OPENAI_", "AZURE_OPENAI_")):
            del runtime[key]
    runtime["MEMORIZZ_HOME"] = str(tmp_path / "home")
    runtime["PYTHONPATH"] = str(Path(memorizz.__file__).resolve().parent.parent)
    return runtime


def test_fresh_process_setting_save_load_and_precedence(tmp_path):
    runtime = _runtime(tmp_path)

    def cli(*arguments, extra_env=None):
        result = subprocess.run(
            [sys.executable, "-m", "memorizz", *arguments],
            env={**runtime, **(extra_env or {})},
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return result.stdout

    assert "Restart required" in cli("config", "set", "MEMORIZZ_BACKEND", "notion")
    loaded = cli("config", "get", "MEMORIZZ_BACKEND")
    assert "MEMORIZZ_BACKEND: 'notion'" in loaded
    assert str(tmp_path / "home" / ".env") in loaded
    cli("config", "set", "MEMORIZZ_BACKEND", "filesystem", "--project")
    assert "MEMORIZZ_BACKEND: 'filesystem'" in cli("config", "get", "MEMORIZZ_BACKEND")
    exported = cli(
        "config", "get", "MEMORIZZ_BACKEND", extra_env={"MEMORIZZ_BACKEND": "mongodb"}
    )
    assert "MEMORIZZ_BACKEND: 'mongodb'" in exported
    assert "process environment" in exported


@pytest.mark.skipif(os.name == "nt", reason="POSIX pseudo-terminal contract")
def test_actual_terminal_hides_credentials_and_next_process_redacts(tmp_path):
    import pty

    runtime = _runtime(tmp_path)
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-m", "memorizz", "config", "set", "NOTION_TOKEN"],
        env=runtime,
        cwd=tmp_path,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    output = b""
    sent = False
    secret = b"terminal-test-private-token"
    deadline = time.monotonic() + 15
    try:
        while time.monotonic() < deadline:
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break  # PTY EOF on Linux.
                if not chunk:
                    break
                output += chunk
                if b"(hidden):" in output and not sent:
                    os.write(master, secret + b"\n")
                    sent = True
            if process.poll() is not None:
                break
        assert sent, output.decode(errors="replace")
        assert process.wait(timeout=5) == 0, output.decode(errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
    assert secret not in output, "The actual terminal echoed the credential"
    assert dotenv_values(tmp_path / "home" / ".env")["NOTION_TOKEN"] == secret.decode()
    result = subprocess.run(
        [sys.executable, "-m", "memorizz", "config", "get", "NOTION_TOKEN"],
        env=runtime,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "set; hidden" in result.stdout
    assert secret.decode() not in result.stdout + result.stderr
