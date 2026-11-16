"""Voice: the configuration, each provider's requests, and the agent option."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.voice import (
    VoiceConfig,
    VoiceError,
    VoiceNotConfigured,
    VoiceUnsupported,
    create_voice_provider,
    speakable,
    voice_status,
)
from memorizz.voice.providers import elevenlabs as elevenlabs_module
from memorizz.voice.providers import vapi as vapi_module
from memorizz.voice.providers.elevenlabs import ElevenLabsVoice
from memorizz.voice.providers.openai import OpenAIVoice
from memorizz.voice.providers.vapi import VapiVoice

pytestmark = pytest.mark.unit

VOICE_ENV = (
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "ELEVENLABS_API_KEY",
    "VAPI_API_KEY",
    "VAPI_PUBLIC_KEY",
    "MEMORIZZ_PUBLIC_URL",
)


@pytest.fixture(autouse=True)
def no_voice_keys(monkeypatch):
    for name in VOICE_ENV:
        monkeypatch.delenv(name, raising=False)
    elevenlabs_module._voice_cache.clear()


class FakeHTTP:
    """Records requests and answers from a queue of (status, json, content)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def _answer(self, method, url, **kwargs):
        self.calls.append(SimpleNamespace(method=method, url=url, **kwargs))
        status, body, content = self.answers.pop(0)
        return SimpleNamespace(
            status_code=status,
            json=lambda: body,
            content=content,
            text=json.dumps(body),
        )

    def get(self, url, **kwargs):
        return self._answer("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._answer("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self._answer("PATCH", url, **kwargs)


# ---- configuration ---------------------------------------------------------


def test_livekit_is_the_default_provider():
    assert VoiceConfig.from_value(None).provider == "livekit"
    assert VoiceConfig.from_value(True).to_dict() == {"provider": "livekit"}
    assert create_voice_provider(None).name == "livekit"


def test_a_provider_name_or_a_dict_selects_the_provider():
    assert VoiceConfig.from_value("ElevenLabs").provider == "elevenlabs"
    config = VoiceConfig.from_value(
        {"provider": "vapi", "voice_id": "v1", "public_url": "https://x.example"}
    )
    assert config.to_dict() == {
        "provider": "vapi",
        "voice": "v1",
        "options": {"public_url": "https://x.example"},
    }
    # A saved configuration reads back to itself.
    assert VoiceConfig.from_value(config.to_dict()) == config


def test_unknown_providers_and_credentials_are_refused():
    with pytest.raises(ValueError, match="Unknown voice provider"):
        VoiceConfig.from_value("carrier-pigeon")
    with pytest.raises(ValueError, match="must not hold credentials"):
        VoiceConfig.from_value({"provider": "elevenlabs", "api_key": "sk-secret"})
    with pytest.raises(ValueError, match="speed"):
        VoiceConfig.from_value({"speed": 9})


def test_status_names_what_each_provider_is_missing():
    by_name = {row["provider"]: row for row in voice_status()}
    assert set(by_name) == {"livekit", "elevenlabs", "vapi", "openai"}
    assert by_name["elevenlabs"]["ready"] is False
    assert by_name["elevenlabs"]["missing_env"] == ["ELEVENLABS_API_KEY"]
    assert by_name["vapi"]["realtime"] and not by_name["vapi"]["tts"]
    assert by_name["livekit"]["realtime"] and by_name["livekit"]["stt"]


def test_speech_text_drops_code_and_markup():
    assert speakable("**Hi** there\n```py\nprint(1)\n```") == "Hi there (code omitted)"
    long = "One sentence. " * 600
    cut = speakable(long, limit=100)
    assert len(cut) <= 100 and cut.endswith(".")


# ---- ElevenLabs ------------------------------------------------------------


def test_elevenlabs_speaks_with_the_accounts_first_voice(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    http = FakeHTTP(
        (200, {"voices": [{"voice_id": "abc", "name": "Rachel"}]}, b""),
        (200, {}, b"MP3"),
        (200, {}, b"MP3"),
    )
    voice = ElevenLabsVoice(VoiceConfig.from_value("elevenlabs"), http=http)

    audio = voice.synthesize("Hello **there**")

    assert (audio.data, audio.mime_type) == (b"MP3", "audio/mpeg")
    listing, speech = http.calls
    assert listing.url.endswith("/v2/voices")
    assert speech.url.endswith("/v1/text-to-speech/abc")
    assert speech.headers["xi-api-key"] == "key"
    assert speech.json == {"text": "Hello there", "model_id": "eleven_flash_v2_5"}
    # The voice is looked up once per key, not on every sentence.
    voice.synthesize("Again")
    assert len(http.calls) == 3


def test_elevenlabs_uses_a_configured_voice_id_without_a_lookup(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    http = FakeHTTP((200, {}, b"MP3"))
    config = VoiceConfig.from_value(
        {"provider": "elevenlabs", "voice": "JBFqnCBsd6RMkjVDRZzb", "speed": 1.1}
    )
    ElevenLabsVoice(config, http=http).synthesize("Hi")
    assert http.calls[0].url.endswith("/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb")
    assert http.calls[0].json["voice_settings"] == {"speed": 1.1}


def test_elevenlabs_transcribes_a_browser_recording(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    http = FakeHTTP((200, {"text": " hello world "}, b""))
    heard = ElevenLabsVoice(http=http).transcribe(b"RIFF", "audio/webm;codecs=opus")
    assert heard == "hello world"
    call = http.calls[0]
    assert call.url.endswith("/v1/speech-to-text")
    assert call.data == {"model_id": "scribe_v3"}
    assert call.files["file"] == ("speech.webm", b"RIFF", "audio/webm")


def test_a_provider_failure_is_a_readable_voice_error(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "key")
    http = FakeHTTP((401, {"detail": {"message": "Invalid API key"}}, b""))
    with pytest.raises(VoiceError, match="401: Invalid API key") as raised:
        ElevenLabsVoice(http=http).transcribe(b"x")
    assert raised.value.code == "voice_unauthorized"


def test_a_provider_without_keys_says_which_to_set():
    with pytest.raises(VoiceNotConfigured, match="ELEVENLABS_API_KEY") as raised:
        ElevenLabsVoice().synthesize("Hi")
    assert raised.value.missing == ["ELEVENLABS_API_KEY"]


# ---- OpenAI ----------------------------------------------------------------


def test_openai_speaks_and_transcribes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    http = FakeHTTP((200, {}, b"MP3"), (200, {"text": "hi"}, b""))
    voice = OpenAIVoice(
        VoiceConfig.from_value({"provider": "openai", "voice": "nova"}), http=http
    )

    assert voice.synthesize("Hello").data == b"MP3"
    assert voice.transcribe(b"audio", "audio/mp4") == "hi"

    speech, heard = http.calls
    assert speech.url.endswith("/audio/speech")
    assert speech.json["voice"] == "nova" and speech.json["input"] == "Hello"
    assert speech.headers["Authorization"] == "Bearer sk-test"
    assert heard.files["file"][0] == "speech.mp4"


# ---- Vapi ------------------------------------------------------------------


def test_vapi_needs_a_public_https_address(monkeypatch):
    monkeypatch.setenv("VAPI_API_KEY", "private")
    monkeypatch.setenv("VAPI_PUBLIC_KEY", "public")
    assert "MEMORIZZ_PUBLIC_URL" in VapiVoice().status()["problem"]
    monkeypatch.setenv("MEMORIZZ_PUBLIC_URL", "http://localhost:8765")
    assert "https://" in VapiVoice().status()["problem"]
    monkeypatch.setenv("MEMORIZZ_PUBLIC_URL", "https://localhost")
    assert "internet" in VapiVoice().status()["problem"]
    monkeypatch.setenv("MEMORIZZ_PUBLIC_URL", "https://ui.example.com/")
    assert VapiVoice().status()["ready"] is True


def test_vapi_creates_a_stored_assistant_that_calls_this_agent(monkeypatch):
    monkeypatch.setenv("VAPI_API_KEY", "private")
    monkeypatch.setenv("VAPI_PUBLIC_KEY", "public")
    monkeypatch.setenv("MEMORIZZ_PUBLIC_URL", "https://ui.example.com")
    http = FakeHTTP((200, [], b""), (201, {"id": "asst_1"}, b""))
    voice = VapiVoice(VoiceConfig.from_value("vapi"), http=http)

    details = voice.realtime_session(
        agent_id="agent-1", agent_name="Ada", reply=lambda said: "hi"
    )

    # The browser gets the public key and the assistant, never the private key.
    assert details == {
        "provider": "vapi",
        "session_id": "vapi-agent-1",
        "public_key": "public",
        "assistant_id": "asst_1",
    }
    created = http.calls[1]
    assert created.method == "POST" and created.url.endswith("/assistant")
    assert created.headers["Authorization"] == "Bearer private"
    model = created.json["model"]
    token = vapi_module.webhook_token("agent-1")
    assert model["provider"] == "custom-llm"
    assert model["url"] == f"https://ui.example.com/webhook/voice/vapi/agent-1/{token}"
    assert vapi_module.valid_webhook_token("agent-1", token)
    assert not vapi_module.valid_webhook_token("agent-2", token)


def test_vapi_updates_the_assistant_it_already_made(monkeypatch):
    monkeypatch.setenv("VAPI_API_KEY", "private")
    monkeypatch.setenv("VAPI_PUBLIC_KEY", "public")
    monkeypatch.setenv("MEMORIZZ_PUBLIC_URL", "https://ui.example.com")
    name = vapi_module.assistant_name("agent-1")
    http = FakeHTTP((200, [{"id": "asst_9", "name": name}], b""), (200, {}, b""))
    assistant = VapiVoice(http=http).ensure_assistant("agent-1", "Ada")
    assert assistant == "asst_9"
    assert http.calls[1].method == "PATCH"
    assert http.calls[1].url.endswith("/assistant/asst_9")


def test_vapi_has_no_turn_by_turn_speech():
    with pytest.raises(VoiceUnsupported):
        VapiVoice().synthesize("Hi")
    with pytest.raises(VoiceUnsupported):
        VapiVoice().transcribe(b"x")


# ---- the agent option ------------------------------------------------------


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", Mock())
    return FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )


def _agent(memory, **options):
    from memorizz.llms.openai import OpenAI

    return MemAgent(
        model=OpenAI(model="gpt-test"),
        memory_provider=memory,
        auto_register=False,
        **options,
    )


def test_voice_is_off_by_default(memory):
    agent = _agent(memory)
    assert agent.voice is False and agent.voice_config is None
    with pytest.raises(VoiceError, match="voice=True") as raised:
        agent.speak("Hi")
    assert raised.value.code == "voice_disabled"


def test_voice_true_uses_livekit(memory):
    agent = _agent(memory, voice=True)
    status = agent.voice_status()
    assert status["provider"] == "livekit" and status["enabled"] is True


def test_voice_settings_survive_save_and_load(memory):
    agent = _agent(
        memory, voice=True, voice_config={"provider": "elevenlabs", "voice": "Rachel"}
    )
    agent.save()
    stored = memory.retrieve_memagent(agent.agent_id)
    assert stored.voice is True
    assert stored.voice_config == {"provider": "elevenlabs", "voice": "Rachel"}

    loaded = MemAgent.load(agent.agent_id, memory_provider=memory)
    assert loaded.voice is True
    assert loaded.voice_provider.name == "elevenlabs"
    assert loaded.voice_provider.config.voice == "Rachel"


def test_a_stale_stored_voice_configuration_does_not_stop_a_load(memory):
    from memorizz.memagent.persistence import update_agent_record

    agent = _agent(memory, voice=True)
    agent.save()
    update_agent_record(memory, agent.agent_id, voice_config={"provider": "gone"})
    loaded = MemAgent.load(agent.agent_id, memory_provider=memory)
    assert loaded.voice is True and loaded.voice_config is None


def test_with_voice_switches_provider_and_off(memory):
    agent = _agent(memory).with_voice(config="openai")
    assert agent.voice_provider.name == "openai"
    agent.with_voice(False)
    assert agent.voice_config == {"provider": "openai"}  # kept for next time
    with pytest.raises(VoiceError):
        agent.voice_provider


def test_a_spoken_turn_transcribes_answers_and_speaks(memory, monkeypatch):
    agent = _agent(memory, voice=True, voice_config="openai")
    provider = Mock()
    provider.transcribe.return_value = "what time is it"
    provider.synthesize.return_value = "AUDIO"
    agent._voice_provider = provider
    monkeypatch.setattr(agent, "run", lambda query, **kwargs: f"You asked: {query}")

    turn = agent.run_voice(b"recording", "audio/webm", memory_id="m1")

    assert turn == {
        "heard": "what time is it",
        "text": "You asked: what time is it",
        "audio": "AUDIO",
    }
    provider.transcribe.assert_called_once_with(b"recording", "audio/webm")
    # Silence is not sent to the model.
    provider.transcribe.return_value = ""
    assert agent.run_voice(b"hum")["audio"] is None


def test_the_builder_sets_voice(memory):
    from memorizz.memagent.builders import MemAgentBuilder

    builder = MemAgentBuilder().with_voice("elevenlabs", voice="Rachel")
    assert builder._voice is True
    assert builder._voice_config == {"provider": "elevenlabs", "voice": "Rachel"}
    assert MemAgentBuilder().with_voice()._voice_config is None


# ---- LiveKit (needs the voice extra) ---------------------------------------


def test_livekit_names_its_missing_keys_or_packages():
    status = create_voice_provider("livekit").status()
    assert status["ready"] is False
    assert status["missing_env"] == [
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
    ]


def test_livekit_model_descriptors_split_into_model_and_suffix():
    from memorizz.voice.providers.livekit import split_model

    assert split_model("cartesia/sonic-3:voice-1", "x") == (
        "cartesia/sonic-3",
        "voice-1",
    )
    assert split_model(None, "assemblyai/universal-3-6-pro") == (
        "assemblyai/universal-3-6-pro",
        None,
    )


def test_livekit_asks_the_agent_for_each_reply():
    pytest.importorskip("livekit.agents")
    import asyncio

    from livekit.agents import llm

    from memorizz.voice.providers import livekit as livekit_module

    context = llm.ChatContext.empty()
    context.add_message(role="assistant", content="Hi, Ada here.")
    context.add_message(role="user", content="what do you study")
    assert livekit_module.last_user_text(context) == "what do you study"

    heard = []
    agent = livekit_module._brain_agent(
        "Ada", lambda said: heard.append(said) or "**Octopuses.**"
    )

    async def collect():
        return [chunk async for chunk in agent.llm_node(context, [], None)]

    assert asyncio.run(collect()) == ["Octopuses."]
    assert heard == ["what do you study"]
    # A session only answers a user turn when it has a language model.
    assert isinstance(livekit_module._brain_llm(), llm.LLM)


def test_livekit_decodes_a_recording_to_pcm():
    pytest.importorskip("av")
    import io
    import math
    import struct
    import wave

    from memorizz.voice.providers.livekit import decode_to_pcm

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(44100)
        recording.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(44100)
            )
        )
    pcm = decode_to_pcm(buffer.getvalue())
    assert abs(len(pcm) - 32000) < 400  # one second at 16 kHz, 16 bit
    with pytest.raises(VoiceError, match="could not be decoded"):
        decode_to_pcm(b"not audio")
