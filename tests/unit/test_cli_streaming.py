"""Real stdout-pipe proofs; provider remains blocked after the first delta."""

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parents[1] / "integration" / "streaming_fixture.py"


def launch(tmp_path, *args):
    return subprocess.Popen(
        [sys.executable, str(FIXTURE), "cli", "run", *args, "hello"],
        env={**os.environ, "MEMORIZZ_STREAM_FIXTURE_DIR": str(tmp_path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def read_until(process, predicate):
    data = b""
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + 15
        while not predicate(data):
            assert time.monotonic() < deadline, (data, process.poll())
            if selector.select(timeout=0.1):
                chunk = os.read(process.stdout.fileno(), 65536)
                assert chunk, (data, process.stderr.read().decode())
                data += chunk
    return data


@pytest.mark.parametrize("args", [[], ["--stream"], ["--output", "jsonl"]])
def test_cli_flushes_before_provider_finishes(tmp_path, args):
    process = launch(tmp_path, *args)
    try:
        first = read_until(process, lambda data: b"Hello " in data)
        assert not (tmp_path / "provider_complete").exists()
        assert process.poll() is None
        (tmp_path / "release").touch()
        tail, errors = process.communicate(timeout=15)
        assert process.returncode == 0, errors.decode()
        if "jsonl" in args:
            events = [json.loads(line) for line in (first + tail).splitlines()]
            assert events[0]["type"] == "run.started"
            assert events[-1]["type"] == "run.done"
            assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
            assert (
                "".join(e["delta"] for e in events if e["type"] == "answer.delta")
                == "Hello 世界 \n"
            )
        else:
            assert (first + tail).decode() == "Hello 世界 \n\n"
            assert b"agent ready" in errors
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("stop", ["interrupt", "pipe", "error"])
def test_cli_partial_stop_and_exit_status(tmp_path, stop):
    process = launch(tmp_path)
    try:
        first = read_until(process, lambda data: b"Hello " in data)
        if stop == "interrupt":
            process.send_signal(signal.SIGINT)
        elif stop == "pipe":
            process.stdout.close()
            (tmp_path / "release").touch()
        else:
            (tmp_path / "fail").touch()
            (tmp_path / "release").touch()
        process.wait(timeout=15)
        assert process.returncode == {"interrupt": 130, "pipe": 141, "error": 1}[stop]
        assert first == b"Hello "
        assert (tmp_path / "provider_closed").exists()
        if stop != "pipe":
            assert not (tmp_path / "persist_started").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("terminal", ["dumb", "xterm-256color"])
def test_repl_renders_while_provider_is_blocked(tmp_path, terminal):
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE), "repl"],
        env={
            **os.environ,
            "MEMORIZZ_STREAM_FIXTURE_DIR": str(tmp_path),
            "TERM": terminal,
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        first = read_until(process, lambda data: b"Hello" in data)
        assert not (tmp_path / "provider_complete").exists()
        assert process.poll() is None
        (tmp_path / "release").touch()
        tail, errors = process.communicate(timeout=10)
        assert process.returncode == 0, errors.decode()
        assert "世界" in (first + tail).decode()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
