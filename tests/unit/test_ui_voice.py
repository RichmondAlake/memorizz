"""Voice in the UI: provider status, Settings keys, the agent form values and
the endpoint a Vapi call asks for each reply."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz.ui.state import _state  # noqa: E402
from memorizz.voice import clear_brain, set_brain  # noqa: E402
from memorizz.voice.providers.vapi import webhook_token  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    from memorizz.ui.app import create_app

    monkeypatch.chdir(tmp_path)
    with patch.dict(
        os.environ, {"MEMORIZZ_HOME": str(tmp_path / "home")}, clear=True
    ), patch.dict(
        _state,
        {"provider": object(), "provider_type": "filesystem", "connection_info": {}},
    ):
        yield TestClient(create_app(), follow_redirects=False)


def test_status_lists_every_provider_with_livekit_as_the_default(client):
    body = client.get("/api/voice/status").json()
    assert body["default"] == "livekit"
    providers = {row["provider"]: row for row in body["providers"]}
    assert set(providers) == {"livekit", "elevenlabs", "vapi", "openai"}
    assert providers["elevenlabs"]["ready"] is False
    os.environ["ELEVENLABS_API_KEY"] = "key"
    ready = client.get("/api/voice/status").json()["providers"]
    assert {row["provider"]: row["ready"] for row in ready}["elevenlabs"] is True


def test_settings_has_a_voice_section_and_never_shows_a_secret(client):
    os.environ["ELEVENLABS_API_KEY"] = "sk_very_secret"
    page = client.get("/settings").text
    for name in (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "ELEVENLABS_API_KEY",
        "VAPI_API_KEY",
        "VAPI_PUBLIC_KEY",
        "MEMORIZZ_PUBLIC_URL",
    ):
        assert f'name="{name}"' in page
    assert 'id="settings-voice"' in page
    assert "sk_very_secret" not in page


def test_the_agent_form_keeps_voice_settings_it_does_not_show():
    from memorizz.ui.routers.agents_crud import _voice_form_values, _voice_value

    assert _voice_value(None, None, None) == (False, None)
    assert _voice_value("on", "", "") == (True, {"provider": "livekit"})
    assert _voice_value("on", "elevenlabs", " Rachel ") == (
        True,
        {"provider": "elevenlabs", "voice": "Rachel"},
    )
    stored = {"provider": "vapi", "options": {"public_url": "https://x.example"}}
    # Same provider: the public address set in code survives an edit.
    assert _voice_value("on", "vapi", "", stored)[1] == stored
    # Another provider: its options do not carry over.
    assert _voice_value("on", "openai", "nova", stored)[1] == {
        "provider": "openai",
        "voice": "nova",
    }
    # Switched off, the choice is kept for next time.
    assert _voice_value(None, "vapi", "", stored) == (False, stored)
    assert _voice_form_values()["voice_provider"] == "livekit"


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]


def test_a_vapi_call_gets_the_agents_reply_as_a_chat_completion_stream(client):
    os.environ["VAPI_API_KEY"] = "private"
    heard = []
    brain = lambda said: heard.append(said) or "**Octopuses**, mostly."  # noqa: E731
    set_brain("agent-1", brain)
    try:
        path = (
            f"/webhook/voice/vapi/agent-1/{webhook_token('agent-1')}/chat/completions"
        )
        response = client.post(
            path,
            json={
                "model": "memorizz",
                "stream": True,
                "messages": [
                    {"role": "system", "content": "ignored"},
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "what do you study"},
                ],
            },
            # Vapi's servers send no browser Origin and no UI session.
            headers={"Origin": "https://api.vapi.ai"},
        )
    finally:
        clear_brain("agent-1", brain)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = _events(response)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"]["content"] == "Octopuses, mostly."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert response.text.rstrip().endswith("data: [DONE]")
    assert heard == ["what do you study"]


def test_a_vapi_call_can_ask_without_streaming(client):
    os.environ["VAPI_API_KEY"] = "private"
    brain = lambda said: "Four."  # noqa: E731
    set_brain("agent-1", brain)
    try:
        response = client.post(
            f"/webhook/voice/vapi/agent-1/{webhook_token('agent-1')}",
            json={"messages": [{"role": "user", "content": "two plus two"}]},
        )
    finally:
        clear_brain("agent-1", brain)
    assert response.json()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Four.",
    }


def test_the_vapi_endpoint_refuses_a_wrong_token_or_path(client):
    os.environ["VAPI_API_KEY"] = "private"
    token = webhook_token("agent-1")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert (
        client.post(
            f"/webhook/voice/vapi/agent-1/nope/chat/completions", json=body
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"/webhook/voice/vapi/agent-2/{token}/chat/completions", json=body
        ).status_code
        == 404
    )
    assert (
        client.post(f"/webhook/voice/vapi/agent-1/{token}/admin", json=body).status_code
        == 404
    )
    # No private key configured: no token is valid.
    del os.environ["VAPI_API_KEY"]
    assert (
        client.post(
            f"/webhook/voice/vapi/agent-1/{token}/chat/completions", json=body
        ).status_code
        == 404
    )


def test_other_posts_still_need_the_same_origin(client):
    response = client.post(
        "/api/agents/agent-1/voice/speak",
        json={"text": "hi"},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403
