"""Local launcher target checks are read-only and fail closed."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def launcher():
    path = Path(__file__).parents[2] / "examples/observability/local_oracle.py"
    spec = importlib.util.spec_from_file_location("local_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "bad", [None, "name", "label", "image", "port", "dsn", "schema", "mount"]
)
def test_local_oracle_verification_refuses_other_targets(monkeypatch, bad):
    module = launcher()
    state = {
        "container_id": "fixture",
        "image_id": "image",
        "dsn": "127.0.0.1:1529/FREEPDB1",
        "dev_user": "MEMORIZZ_DEV",
        "test_user": "MEMORIZZ_OBS_TEST_LOCAL",
        "data_directory": str(module.STATE.parent / "oradata"),
    }
    item = {
        "Name": "/" + module.NAME,
        "Image": "image",
        "Config": {
            "Labels": {"io.memorizz.purpose": "isolated-observability-development"}
        },
        "HostConfig": {
            "PortBindings": {"1521/tcp": [{"HostIp": "127.0.0.1", "HostPort": "1529"}]}
        },
        "Mounts": [
            {"Source": state["data_directory"], "Destination": "/opt/oracle/oradata"}
        ],
    }
    if bad == "name":
        item["Name"] = "/another-container"
    elif bad == "label":
        item["Config"]["Labels"] = {}
    elif bad == "image":
        item["Image"] = "other"
    elif bad == "port":
        item["HostConfig"]["PortBindings"]["1521/tcp"][0]["HostIp"] = "0.0.0.0"
    elif bad == "dsn":
        state["dsn"] = "remote:1521/PROD"
    elif bad == "schema":
        state["test_user"] = "MEMORIZZ_DEV"
    elif bad == "mount":
        item["Mounts"][0]["Source"] = "/unrelated"
    before = deepcopy(item)
    calls = []

    def docker(*args):
        calls.append(args)
        return json.dumps([item])

    monkeypatch.setattr(module, "docker", docker)
    if bad:
        with pytest.raises(RuntimeError, match="refused"):
            module.verify(state)
    else:
        assert module.verify(state) == item
    assert calls == [("inspect", "fixture")]
    assert before == item
