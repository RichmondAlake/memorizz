"""Isolation and cleanup guards for the opt-in standalone MongoDB gate."""

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.integration import run_observability_mongodb as launcher


@pytest.mark.parametrize("mismatch", [None, "pid", "path", "missing_path"])
def test_launcher_verifies_process_and_storage_before_using_endpoint(
    tmp_path, mismatch
):
    client = Mock()
    storage = {"dbPath": str(tmp_path)}
    if mismatch == "path":
        storage["dbPath"] = str(tmp_path / "unowned")
    elif mismatch == "missing_path":
        storage = {}
    client.admin.command.side_effect = [
        {"pid": 2 if mismatch == "pid" else 1, "version": "test-version"},
        {"parsed": {"storage": storage}},
    ]
    process = SimpleNamespace(pid=1)
    if mismatch:
        with pytest.raises(RuntimeError, match="not owned"):
            launcher.verify_process(client, process, tmp_path.resolve())
    else:
        assert (
            launcher.verify_process(client, process, tmp_path.resolve())
            == "test-version"
        )


@pytest.mark.parametrize("failure", [None, "pytest", "ownership", "timeout", "client"])
def test_launcher_stops_only_owned_child_on_success_and_failure(monkeypatch, failure):
    listener = Mock()
    listener.getsockname.return_value = ("127.0.0.1", 49152)
    socket_context = Mock(
        __enter__=Mock(return_value=listener), __exit__=Mock(return_value=False)
    )
    monkeypatch.setattr(launcher.socket, "socket", Mock(return_value=socket_context))
    process = Mock(pid=1234)
    process.poll.return_value = None
    if failure == "timeout":
        process.wait.side_effect = [subprocess.TimeoutExpired("owned mongod", 30), 0]
    popen = Mock(return_value=process)
    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    client = Mock()
    client_factory = Mock(return_value=client)
    if failure == "client":
        client_factory.side_effect = RuntimeError("client construction failed")
    monkeypatch.setattr(launcher, "MongoClient", client_factory)
    ownership = Mock(return_value="test-version")
    if failure == "ownership":
        ownership.side_effect = RuntimeError("not owned")
    monkeypatch.setattr(launcher, "verify_process", ownership)
    test_command = Mock(return_value=SimpleNamespace(returncode=0))
    if failure == "pytest":
        test_command.return_value.returncode = 1
    elif failure == "timeout":
        test_command.side_effect = subprocess.TimeoutExpired("synthetic pytest", 300)
    monkeypatch.setattr(launcher.subprocess, "run", test_command)
    monkeypatch.setenv(
        "MEMORIZZ_OBS_TEST_MONGODB_URI", "mongodb://never-connect.invalid"
    )
    if failure in {"ownership", "client"}:
        with pytest.raises(RuntimeError):
            launcher.run(sys.executable)
        test_command.assert_not_called()
    elif failure == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            launcher.run(sys.executable)
    else:
        assert launcher.run(sys.executable) == (1 if failure == "pytest" else 0)
        command = test_command.call_args.args[0]
        assert command[command.index("-k") + 1] == "mongodb"
        assert (
            "127.0.0.1:49152"
            in test_command.call_args.kwargs["env"]["MEMORIZZ_OBS_TEST_MONGODB_URI"]
        )
    command = popen.call_args.args[0]
    assert command[command.index("--bind_ip") + 1] == "127.0.0.1"
    assert "--fork" not in command and "--nounixsocket" in command
    directory = launcher.Path(command[command.index("--dbpath") + 1])
    assert "memorizz-obs-live-" in directory.name and not directory.exists()
    process.terminate.assert_called_once_with()
    if failure == "timeout":
        process.kill.assert_called_once_with()
    else:
        process.kill.assert_not_called()
    if failure != "client":
        client.close.assert_called_once_with()
