# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The connect page lists the Oracle and MongoDB databases on this computer,
found in Docker by the port they publish, without reading any secret."""

import json

import pytest

from memorizz.ui import local_databases

CONTAINERS = {
    "acme-oracle-free": (
        "running",
        "Up 9 hours",
        "gvenzl/oracle-free:23",
        {"1521/tcp": "1521"},
    ),
    "observability": (
        "running",
        "Up 4 hours",
        "sha256:f40109d95628c75c8",
        {"1521/tcp": "1529"},
    ),
    "pri-db-free": (
        "exited",
        "Exited (137) 4 days ago",
        "container-registry.oracle.com/database/free:latest",
        {"1521/tcp": "1525"},
    ),
    "mongo": ("running", "Up 2 days", "mongo:6", {"27017/tcp": "27018"}),
    "web": ("running", "Up 1 hour", "nginx:1", {"80/tcp": "8080"}),
}


@pytest.fixture()
def docker(monkeypatch):
    calls = []

    def run(args, timeout=30):
        calls.append(args)
        if args[1] == "ps":
            return (
                0,
                "\n".join(f"{n}\t{c[0]}\t{c[1]}" for n, c in CONTAINERS.items()),
                "",
            )
        lines = []
        for name in args[4:]:
            _state, _status, image, ports = CONTAINERS[name]
            bindings = {p: [{"HostIp": "", "HostPort": h}] for p, h in ports.items()}
            lines.append(f"/{name}\t{image}\t{json.dumps(bindings)}")
        return 0, "\n".join(lines), ""

    monkeypatch.setattr("memorizz.ui.docker_oracle._run", run)
    monkeypatch.setattr(local_databases, "_cache", {"at": 0.0, "value": None})
    for name in ("ORACLE_DSN", "ORACLE_USER", "MONGODB_URI"):
        monkeypatch.delenv(name, raising=False)
    return calls


@pytest.mark.unit
def test_databases_in_containers_are_found_by_the_port_they_publish(docker):
    found = local_databases.detect_local_databases(
        current={"dsn": "localhost:1529/FREEPDB1"}
    )
    oracle = [
        (db["label"], db["dsn"], db["state"], db["current"]) for db in found["oracle"]
    ]
    assert oracle == [
        ("acme-oracle-free", "localhost:1521/FREEPDB1", "running", False),
        ("observability", "localhost:1529/FREEPDB1", "running", True),
        ("pri-db-free", "localhost:1525/FREEPDB1", "stopped", False),
    ], "running first; an image known only by its hash still counts"
    assert found["oracle"][2]["hint"] == "docker start pri-db-free"
    assert [db["uri"] for db in found["mongodb"]] == ["mongodb://localhost:27018"]
    inspect = next(args for args in docker if args[1] == "inspect")
    assert "Env" not in inspect[3], "a container's environment holds its passwords"


@pytest.mark.unit
def test_the_environment_adds_its_addresses_without_secrets(docker, monkeypatch):
    monkeypatch.setenv("ORACLE_DSN", "db.example.com:1521/PROD")
    monkeypatch.setenv("ORACLE_USER", "reader")
    monkeypatch.setenv(
        "MONGODB_URI", "mongodb+srv://u:s3cret@cluster0.example.net/?x=1"
    )
    found = local_databases.detect_local_databases()
    assert found["oracle"][0] == {
        "label": "ORACLE_DSN",
        "source": "environment",
        "dsn": "db.example.com:1521/PROD",
        "user": "reader",
        "current": False,
    }
    env = found["mongodb"][0]
    assert env["uri"] == "" and "s3cret" not in json.dumps(found)


@pytest.mark.unit
def test_no_docker_means_no_list(monkeypatch):
    monkeypatch.setattr(
        "memorizz.ui.docker_oracle._run", lambda args, timeout=30: (-1, "", "missing")
    )
    monkeypatch.setattr(local_databases, "_cache", {"at": 0.0, "value": None})
    monkeypatch.delenv("ORACLE_DSN", raising=False)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert local_databases.detect_local_databases() == {"oracle": [], "mongodb": []}


@pytest.mark.unit
def test_the_connect_page_lists_them_under_their_provider(
    docker, tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    html = TestClient(create_app()).get("/connect").text
    oracle = html.index('id="oracle-fields"')
    mongodb = html.index('id="mongodb-fields"')
    filesystem = html.index('id="filesystem-fields"')
    listed = html.index("Oracle databases on this computer")
    assert oracle < listed < mongodb
    assert 'data-dsn="localhost:1525/FREEPDB1"' in html
    assert "docker start pri-db-free" in html
    assert mongodb < html.index("MongoDB databases on this computer") < filesystem
    assert html.index("FileSystem stores on this computer") > filesystem
