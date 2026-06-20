# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Route tests for the Oracle-via-Docker UI endpoints.

Docker is never touched: the memorizz.ui.docker_oracle helper that every handler
delegates to is mocked, so these run on any machine.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui.app import create_app  # noqa: E402

MOD = "memorizz.ui.docker_oracle"


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.mark.unit
def test_status_docker_unavailable(client):
    with patch(f"{MOD}.docker_available", return_value=False):
        resp = client.get("/api/docker/oracle/status")
    assert resp.status_code == 200
    assert resp.json()["state"] == "docker-unavailable"


@pytest.mark.unit
def test_status_running(client):
    with patch(f"{MOD}.docker_available", return_value=True), patch(
        f"{MOD}.list_oracle_containers",
        return_value=[{"name": "memorizz_oracle", "state": "running"}],
    ):
        resp = client.get("/api/docker/oracle/status")
    assert resp.json()["state"] == "running"


@pytest.mark.unit
def test_status_absent(client):
    with patch(f"{MOD}.docker_available", return_value=True), patch(
        f"{MOD}.list_oracle_containers", return_value=[]
    ):
        resp = client.get("/api/docker/oracle/status")
    assert resp.json()["state"] == "absent"


@pytest.mark.unit
def test_runtime(client):
    with patch(
        f"{MOD}.detect_runtime", return_value={"available": True, "kind": "docker"}
    ):
        resp = client.get("/api/docker/oracle/runtime")
    assert resp.status_code == 200
    assert resp.json()["available"] is True


@pytest.mark.unit
def test_start_docker_unavailable(client):
    with patch(f"{MOD}.docker_available", return_value=False):
        resp = client.post("/api/docker/oracle/start", data={"container_name": "x"})
    assert resp.status_code == 503
    assert resp.json()["ok"] is False


@pytest.mark.unit
def test_start_absent_404(client):
    with patch(f"{MOD}.docker_available", return_value=True), patch(
        f"{MOD}.get_container_state", return_value="absent"
    ):
        resp = client.post("/api/docker/oracle/start", data={"container_name": "nope"})
    assert resp.status_code == 404


@pytest.mark.unit
def test_start_success(client):
    with patch(f"{MOD}.docker_available", return_value=True), patch(
        f"{MOD}.get_container_state", return_value="stopped"
    ), patch(f"{MOD}.start_container", return_value=(True, "started")):
        resp = client.post("/api/docker/oracle/start", data={"container_name": "x"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "message": "started"}


@pytest.mark.unit
def test_create_docker_unavailable(client):
    with patch(f"{MOD}.docker_available", return_value=False):
        resp = client.post(
            "/api/docker/oracle/create",
            data={
                "oracle_user": "u",
                "oracle_password": "p",
                "oracle_dsn": "localhost:1521/FREEPDB1",
            },
        )
    assert resp.status_code == 503


@pytest.mark.unit
def test_logs_stream_attaches_and_closes(client):
    with patch(
        f"{MOD}.stream_container_logs", return_value=iter(["DATABASE IS READY"])
    ):
        resp = client.get("/api/docker/oracle/logs/stream?name=test&tail=5")
    assert resp.status_code == 200
    body = resp.text
    assert "event: attach" in body
    assert "event: close" in body
    assert "DATABASE IS READY" in body
