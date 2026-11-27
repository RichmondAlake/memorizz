"""Tests for memorizz.cloud (connect, definition, publish, status, MemAgent.publish)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

import memorizz
from memorizz import MemAgent, Persona, cloud
from memorizz.cloud import CloudError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_cloud_state(monkeypatch):
    cloud.disconnect()
    monkeypatch.delenv("MEMORIZZ_CLOUD_URL", raising=False)
    monkeypatch.delenv("MEMORIZZ_CLOUD_KEY", raising=False)
    yield
    cloud.disconnect()


def test_cloud_module_is_exported_from_memorizz():
    assert memorizz.cloud is cloud
    assert "cloud" in memorizz.__all__


def test_connect_validates_url_and_reads_env_without_returning_key(monkeypatch):
    info = cloud.connect("https://platform.memorizz.dev/", key="wk_secret_123")
    assert info == {"url": "https://platform.memorizz.dev"}
    assert "wk_secret_123" not in json.dumps(info)

    cloud.disconnect()
    monkeypatch.setenv("MEMORIZZ_CLOUD_URL", "https://platform.memorizz.dev")
    monkeypatch.setenv("MEMORIZZ_CLOUD_KEY", "wk_from_env")
    assert cloud.connect() == {"url": "https://platform.memorizz.dev"}

    with pytest.raises(CloudError, match="expected https://"):
        cloud.connect("platform.memorizz.dev", key="wk_abc")

    with pytest.raises(CloudError, match="credentials in the MemoRizz Cloud URL"):
        cloud.connect("https://user:pass@platform.memorizz.dev", key="wk_abc")


def test_definition_extracts_only_allowed_fields_and_drops_local_capabilities(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "store", use_faiss=False)
    )
    persona = Persona.from_dict(
        {
            "name": "Ada",
            "role": "Claims assistant",
            "goals": "Resolve claims quickly",
            "background": "Worked in marine insurance for 8 years",
        }
    )

    def secret_local_tool(x: str) -> str:
        """A local tool that must never be uploaded."""
        return x

    agent = MemAgent(
        name="Ada",
        instruction="You are Ada, a concise claims assistant.",
        persona=persona,
        memory_provider=provider,
        tools=[secret_local_tool],
        memory_ids=["private-mem-1"],
        voice=True,
    )
    defn = cloud.definition(agent)
    assert set(defn.keys()) == {"name", "instruction", "persona", "client"}
    assert defn["name"] == "Ada"
    assert defn["instruction"] == "You are Ada, a concise claims assistant."
    assert defn["persona"] == {
        "role": "Claims assistant",
        "goals": "Resolve claims quickly",
        "background": "Worked in marine insurance for 8 years",
    }
    assert defn["client"].startswith("memorizz ")
    for forbidden in (
        "tools",
        "model",
        "llm_config",
        "memory_ids",
        "voice",
        "mcp_servers",
        "sandbox_provider",
        "delegates",
    ):
        assert forbidden not in defn

    dropped = cloud.dropped_capabilities(agent)
    assert "local tools" in dropped
    assert "voice" in dropped
    assert "existing memory" in dropped


def test_definition_checks_missing_name_and_field_lengths():
    with pytest.raises(CloudError) as exc:
        cloud.definition({"instruction": "No name here"})
    assert exc.value.code == "name_required"

    with pytest.raises(CloudError) as exc:
        cloud.definition({"name": "Ada", "instruction": "a" * 4001})
    assert exc.value.code == "instruction_too_long"

    with pytest.raises(CloudError) as exc:
        cloud.definition({"name": "Ada", "persona": {"goals": "g" * 601}})
    assert exc.value.code == "persona_field_too_long"


def test_publish_and_status_round_trip_and_never_store_key_on_agent(tmp_path, capsys):
    captured = {}

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.ok = 200 <= status_code < 300
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(method, url, *, headers=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        if method == "POST":
            return FakeResponse(
                200,
                {
                    "ok": True,
                    "agent_id": "agent-123",
                    "name": "Ada",
                    "in_world": True,
                    "updated": False,
                    "watch": "https://platform.memorizz.dev/simuground",
                },
            )
        return FakeResponse(
            200,
            {
                "ok": True,
                "published": True,
                "agent_id": "agent-123",
                "name": "Ada",
                "in_world": True,
            },
        )

    cloud.connect("https://platform.memorizz.dev", key="wk_never_leak_me")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "store", use_faiss=False)
    )
    agent = MemAgent(name="Ada", instruction="You are Ada.", memory_provider=provider)

    with patch("requests.request", side_effect=fake_request):
        res = agent.publish()
        st = cloud.status()

    assert res["agent_id"] == "agent-123"
    assert st["in_world"] is True
    assert captured["url"] == "https://platform.memorizz.dev/simuground/api/publish"
    assert captured["headers"]["Authorization"] == "Bearer wk_never_leak_me"

    out = capsys.readouterr().out
    assert "Published Ada (agent-123)" in out
    assert "https://platform.memorizz.dev/simuground" in out
    assert "wk_never_leak_me" not in out
    # The workshop key must never be written onto the agent instance.
    assert "wk_never_leak_me" not in repr(agent.__dict__)


@pytest.mark.parametrize(
    "status_code,body,expected_code,expected_phrase",
    [
        (
            401,
            {"error": "invalid_key", "detail": "bad"},
            "invalid_key",
            "workshop key was rejected",
        ),
        (
            401,
            {"error": "key_expired", "detail": "Expired on 2026-12-06"},
            "key_expired",
            "Expired on 2026-12-06",
        ),
        (401, {"error": "key_revoked", "detail": "revoked"}, "key_revoked", "revoked"),
        (
            429,
            {"error": "rate_limited", "detail": "Wait 10s"},
            "rate_limited",
            "Wait 10s",
        ),
        (
            409,
            {"error": "world_full", "detail": "Holds 50 at most"},
            "world_full",
            "50 at most",
        ),
        (
            400,
            {"error": "instruction_too_long", "detail": "At most 4,000 characters"},
            "instruction_too_long",
            "4,000",
        ),
        (500, {"error": "server_error", "detail": "Oops"}, "server_error", "Oops"),
    ],
)
def test_publish_turns_server_errors_into_actionable_messages(
    status_code, body, expected_code, expected_phrase
):
    cloud.connect("https://platform.memorizz.dev", key="wk_test_secret")

    class FakeResp:
        ok = False

        def __init__(self):
            self.status_code = status_code

        def json(self):
            return body

    with patch("requests.request", return_value=FakeResp()):
        with pytest.raises(CloudError) as exc:
            cloud.publish({"name": "Ada", "instruction": "Hi"}, quiet=True)
    assert exc.value.code == expected_code
    assert exc.value.status_code == status_code
    assert expected_phrase in str(exc.value)
    assert "wk_test_secret" not in str(exc.value)


def test_publish_turns_network_errors_into_actionable_messages():
    cloud.connect("https://platform.memorizz.dev", key="wk_test_secret")

    with patch("requests.request", side_effect=requests.Timeout("slow")):
        with pytest.raises(CloudError) as exc:
            cloud.publish({"name": "Ada", "instruction": "Hi"}, quiet=True)
    assert exc.value.code == "timeout" and "Timed out" in str(exc.value)

    with patch("requests.request", side_effect=requests.ConnectionError("offline")):
        with pytest.raises(CloudError) as exc:
            cloud.publish({"name": "Ada", "instruction": "Hi"}, quiet=True)
    assert exc.value.code == "connection_error" and "Could not reach" in str(exc.value)
