"""OpenAI: text to speech and transcription over its audio API.

Not a realtime call: each turn is one transcription request and one speech
request, which is enough for push-to-talk conversations.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..base import VoiceAudio, VoiceError, VoiceProvider, audio_filename, speakable
from ..config import VoiceConfig
from . import _http

API = "https://api.openai.com/v1"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_STT_MODEL = "gpt-4o-mini-transcribe"
DEFAULT_VOICE = "alloy"


class OpenAIVoice(VoiceProvider):
    name = "openai"
    label = "OpenAI"
    supports_tts = True
    supports_stt = True
    required_env = ("OPENAI_API_KEY",)

    def __init__(self, config: Optional[VoiceConfig] = None, *, http: Any = None):
        super().__init__(config)
        self._http = _http.session(http)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}

    def _base(self) -> str:
        return str(self.config.option("base_url", API)).rstrip("/")

    def synthesize(self, text: str) -> VoiceAudio:
        self.require_ready()
        spoken = speakable(text)
        if not spoken:
            raise VoiceError("Nothing to say", code="voice_empty_text")
        body: Dict[str, Any] = {
            "model": self.config.tts_model or DEFAULT_TTS_MODEL,
            "voice": self.config.voice or DEFAULT_VOICE,
            "input": spoken,
            "response_format": "mp3",
        }
        if self.config.speed:
            body["speed"] = self.config.speed
        instructions = self.config.option("instructions")
        if instructions:
            body["instructions"] = str(instructions)
        response = _http.check(
            self._http.post(
                f"{self._base()}/audio/speech",
                headers=self._headers(),
                json=body,
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        return VoiceAudio(response.content, "audio/mpeg")

    def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> str:
        self.require_ready()
        if not audio:
            raise VoiceError("No audio was recorded", code="voice_empty_audio")
        kind = (mime_type or "audio/webm").split(";")[0].strip()
        data = {"model": self.config.stt_model or DEFAULT_STT_MODEL}
        if self.config.language:
            data["language"] = self.config.language
        response = _http.check(
            self._http.post(
                f"{self._base()}/audio/transcriptions",
                headers=self._headers(),
                data=data,
                files={"file": (audio_filename(kind), audio, kind)},
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        return str(response.json().get("text") or "").strip()
