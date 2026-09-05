"""Live synthetic stdio/loopback delivery, opt-in metadata and cancellation."""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from memorizz.mcp.streaming import MCPEventDispatcher

FIXTURE = Path(__file__).with_name("streaming_fixture.py")


def environment(root):
    return {
        **os.environ,
        "MEMORIZZ_STREAM_FIXTURE_DIR": str(root),
        "PYTHONPATH": str(Path(__file__).parents[2] / "src"),
    }


def free_port():
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("Loopback binding requires sandbox approval")
        return sock.getsockname()[1]


def wait_ready(process, port):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read().decode())
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError("Fixture did not start")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


async def exercise(client, dispatcher, root, *, cancel=False):
    events, first = [], asyncio.Event()

    def receive(event):
        events.append(event)
        if event["type"] == "answer.delta":
            first.set()

    task = asyncio.create_task(
        dispatcher.call_tool(
            client,
            {"agent_id": "stream-fixture", "message": "hello"},
            receive,
        )
    )
    try:
        try:
            await asyncio.wait_for(first.wait(), timeout=5)
        except TimeoutError:
            if task.done():
                raise AssertionError(("No early answer", task.result(), events))
            raise AssertionError(("No early answer; task still running", events))
        assert not task.done()
        assert not (root / "provider_complete").exists()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            deadline = time.monotonic() + 5
            while not (root / "provider_closed").exists():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.02)
            assert not (root / "persist_started").exists()
            return events
        (root / "release").touch()
        result = await asyncio.wait_for(task, timeout=15)
        assert not result.is_error, result
        payload = result.structured_content
        if "result" in payload:
            payload = payload["result"]
        assert payload["response"] == "Hello 世界 \n"
        assert payload["status"] == "completed"
        assert events[-1]["type"] == "run.done"
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert (
            "".join(e["delta"] for e in events if e["type"] == "answer.delta")
            == payload["response"]
        )
        return events
    finally:
        if not task.done():
            task.cancel()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_mcp_stdio_receives_deltas_before_result(tmp_path, cancel, mode):
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        dispatcher = MCPEventDispatcher()
        transport = stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(FIXTURE), "mcp-stdio"],
                env=environment(tmp_path),
            )
        )
        async with Client(
            transport, mode=mode, message_handler=dispatcher.receive
        ) as client:
            await exercise(client, dispatcher, tmp_path, cancel=cancel)

    asyncio.run(run())


@pytest.mark.parametrize("progress", [False, True])
def test_mcp_default_and_progress_only_clients_keep_final_result(tmp_path, progress):
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        transport = stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=[str(FIXTURE), "mcp-stdio"],
                env=environment(tmp_path),
            )
        )
        async with Client(transport, mode="auto") as client:
            updates = []

            async def callback(value, total, message):
                updates.append((value, message))

            task = asyncio.create_task(
                client.call_tool(
                    "memorizz_execute_agent",
                    {"agent_id": "stream-fixture", "message": "hello"},
                    **({"progress_callback": callback} if progress else {}),
                )
            )
            deadline = time.monotonic() + 5
            while not (tmp_path / "provider_waiting").exists():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.02)
            assert not task.done()
            if progress:
                assert updates
                assert "Hello" not in str(updates)
            (tmp_path / "release").touch()
            result = await asyncio.wait_for(task, 10)
            assert not result.is_error, result
            assert "Hello 世界" in str(result.structured_content)

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_mcp_authenticated_http_streaming(tmp_path, cancel, mode):
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client

    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE), "mcp-http", str(port)],
        env=environment(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_ready(process, port)

        async def run():
            dispatcher = MCPEventDispatcher()
            async with httpx2.AsyncClient(
                headers={
                    "Authorization": "Bearer alice-synthetic-stream-token-long-enough"
                }
            ) as http:
                transport = streamable_http_client(
                    f"http://127.0.0.1:{port}/mcp", http_client=http
                )
                async with Client(
                    transport, mode=mode, message_handler=dispatcher.receive
                ) as client:
                    events = await exercise(client, dispatcher, tmp_path, cancel=cancel)
            if not cancel:
                # Bob has a different session, principal and default conversation.
                other = MCPEventDispatcher()
                async with httpx2.AsyncClient(
                    headers={
                        "Authorization": "Bearer bob-synthetic-stream-token-long-enough"
                    }
                ) as http:
                    transport = streamable_http_client(
                        f"http://127.0.0.1:{port}/mcp", http_client=http
                    )
                    async with Client(
                        transport, mode=mode, message_handler=other.receive
                    ) as client:
                        bob_events = []
                        await other.call_tool(
                            client,
                            {"agent_id": "stream-fixture", "message": "bob"},
                            bob_events.append,
                        )
                        assert bob_events[0]["memory_id"] != events[0]["memory_id"]
                        assert bob_events[0]["run_id"] != events[0]["run_id"]
                        assert len({event["run_id"] for event in bob_events}) == 1

        asyncio.run(run())
    finally:
        stop(process)


@pytest.mark.parametrize("cancel", [False, True])
def test_ui_socket_streams_before_provider_completion(tmp_path, cancel):
    import json

    import httpx

    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE), "ui", str(port)],
        env=environment(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        wait_ready(process, port)
        events = []
        with httpx.Client(
            headers={"Authorization": "Bearer stream-fixture-token"}, timeout=10
        ) as client:
            with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/agents/stream-fixture/playground/stream",
                data={"query": "hello", "memory_id": "socket-thread"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-cache, no-transform"
                assert response.headers["x-accel-buffering"] == "no"
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    events.append(event)
                    if event["type"] == "answer.delta" and event["delta"] == "Hello ":
                        assert not (tmp_path / "provider_complete").exists()
                        if cancel:
                            break
                        (tmp_path / "release").touch()
            if cancel:
                deadline = time.monotonic() + 5
                while not (tmp_path / "provider_closed").exists():
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                assert not (tmp_path / "persist_started").exists()
            else:
                assert events[-1]["type"] == "run.done"
                assert events[-1]["status"] == "completed", events
                assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
                assert (
                    "".join(e["delta"] for e in events if e["type"] == "answer.delta")
                    == "Hello 世界 \n"
                )
    finally:
        stop(process)
