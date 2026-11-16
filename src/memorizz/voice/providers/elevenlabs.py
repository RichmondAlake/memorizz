"""ElevenLabs: text to speech and transcription over its REST API."""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, Optional

from ..base import VoiceAudio, VoiceError, VoiceProvider, audio_filename, speakable
from ..config import VoiceConfig
from . import _http

API = "https://api.elevenlabs.io"
DEFAULT_TTS_MODEL = "eleven_flash_v2_5"
DEFAULT_STT_MODEL = "scribe_v3"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

_VOICE_ID = re.compile(r"^[A-Za-z0-9]{20}$")
# api key -> {name or "": voice id}; a voice list rarely changes within a run.
_voice_cache: Dict[str, Dict[str, str]] = {}
_voice_lock = threading.Lock()


class ElevenLabsVoice(VoiceProvider):
    name = "elevenlabs"
    label = "ElevenLabs"
    supports_tts = True
    supports_stt = True
    required_env = ("ELEVENLABS_API_KEY",)

    def __init__(self, config: Optional[VoiceConfig] = None, *, http: Any = None):
        super().__init__(config)
        self._http = _http.session(http)

    def _headers(self) -> Dict[str, str]:
        return {"xi-api-key": os.environ.get("ELEVENLABS_API_KEY", "")}

    def _base(self) -> str:
        return str(self.config.option("base_url", API)).rstrip("/")

    def resolve_voice(self) -> str:
        """The voice id to speak with: the configured id, a voice found by
        name, or the account's first voice (no id exists on every account)."""
        wanted = self.config.voice or ""
        if _VOICE_ID.match(wanted):
            return wanted
        key = os.environ.get("ELEVENLABS_API_KEY", "")
        with _voice_lock:
            cached = _voice_cache.get(key, {}).get(wanted.casefold())
        if cached:
            return cached
        params: Dict[str, Any] = {"page_size": 20}
        if wanted:
            params["search"] = wanted
        response = _http.check(
            self._http.get(
                f"{self._base()}/v2/voices",
                headers=self._headers(),
                params=params,
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        voices = response.json().get("voices") or []
        if wanted:
            exact = [
                v
                for v in voices
                if str(v.get("name", "")).casefold() == wanted.casefold()
            ]
            voices = exact or voices
        if not voices:
            raise VoiceError(
                f"ElevenLabs has no voice named {wanted!r}"
                if wanted
                else "This ElevenLabs account has no voices",
                code="voice_not_found",
            )
        voice_id = str(voices[0]["voice_id"])
        with _voice_lock:
            _voice_cache.setdefault(key, {})[wanted.casefold()] = voice_id
        return voice_id

    def synthesize(self, text: str) -> VoiceAudio:
        self.require_ready()
        spoken = speakable(text)
        if not spoken:
            raise VoiceError("Nothing to say", code="voice_empty_text")
        body: Dict[str, Any] = {
            "text": spoken,
            "model_id": self.config.tts_model or DEFAULT_TTS_MODEL,
        }
        if self.config.language:
            body["language_code"] = self.config.language
        if self.config.speed:
            body["voice_settings"] = {"speed": self.config.speed}
        output_format = str(self.config.option("output_format", DEFAULT_OUTPUT_FORMAT))
        response = _http.check(
            self._http.post(
                f"{self._base()}/v1/text-to-speech/{self.resolve_voice()}",
                headers={**self._headers(), "Content-Type": "application/json"},
                params={"output_format": output_format},
                json=body,
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        mime = "audio/mpeg" if output_format.startswith("mp3") else "audio/ogg"
        return VoiceAudio(response.content, mime)

    def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> str:
        self.require_ready()
        if not audio:
            raise VoiceError("No audio was recorded", code="voice_empty_audio")
        kind = (mime_type or "audio/webm").split(";")[0].strip()
        data = {"model_id": self.config.stt_model or DEFAULT_STT_MODEL}
        if self.config.language:
            data["language_code"] = self.config.language
        response = _http.check(
            self._http.post(
                f"{self._base()}/v1/speech-to-text",
                headers=self._headers(),
                data=data,
                files={"file": (audio_filename(kind), audio, kind)},
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        return str(response.json().get("text") or "").strip()
