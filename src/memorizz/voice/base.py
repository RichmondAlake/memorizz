"""The contract every voice provider meets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import VoiceConfig

#: A reply function: what the person said -> what the agent answers.
Brain = Callable[[str], str]


class VoiceError(RuntimeError):
    """A voice request failed. ``code`` is stable; the message is for people."""

    code = "voice_error"

    def __init__(self, message: str, *, code: Optional[str] = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


class VoiceNotConfigured(VoiceError):
    """The provider's credentials or packages are missing."""

    code = "voice_not_configured"

    def __init__(self, message: str, missing: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.missing = list(missing or [])


class VoiceUnsupported(VoiceError):
    """The provider cannot do this (for example Vapi has no plain transcription)."""

    code = "voice_unsupported"


@dataclass
class VoiceAudio:
    """Synthesised speech, ready to send to a browser or write to a file."""

    data: bytes
    mime_type: str = "audio/mpeg"

    @property
    def extension(self) -> str:
        return {
            "audio/mpeg": "mp3",
            "audio/wav": "wav",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
        }.get(self.mime_type, "bin")


#: Longest text one synthesis call accepts; longer replies are cut at a sentence.
MAX_SPEECH_CHARS = 4000


def speakable(text: str, limit: int = MAX_SPEECH_CHARS) -> str:
    """``text`` trimmed to what is worth saying aloud (no code fences, bounded)."""
    import re

    cleaned = re.sub(r"```.*?```", " (code omitted) ", str(text or ""), flags=re.S)
    cleaned = re.sub(r"[*_#`>]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= limit:
        return cleaned
    cut = cleaned[:limit]
    stop = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[: stop + 1] if stop > limit // 2 else cut


class VoiceProvider:
    """One voice service.

    A provider says what it can do through three flags. ``supports_tts`` and
    ``supports_stt`` are the turn-by-turn path (text in, audio out; audio in,
    text out). ``supports_realtime`` is a live call the browser joins, where
    the provider handles listening, turn detection and speaking and asks the
    agent for each reply.
    """

    name = ""
    label = ""
    supports_tts = False
    supports_stt = False
    supports_realtime = False
    #: Environment variables that must be set.
    required_env: Tuple[str, ...] = ()
    #: Importable modules the provider needs, and the extra that installs them.
    required_modules: Tuple[str, ...] = ()
    extra = ""

    def __init__(self, config: Optional[VoiceConfig] = None) -> None:
        self.config = config or VoiceConfig(provider=self.name)

    # ---- readiness -------------------------------------------------------

    def missing_env(self) -> List[str]:
        return [name for name in self.required_env if not os.environ.get(name)]

    def missing_modules(self) -> List[str]:
        import importlib.util

        missing = []
        for module in self.required_modules:
            try:
                found = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.append(module)
        return missing

    def status(self) -> Dict[str, Any]:
        """Whether the provider can be used now, and what is missing if not."""
        missing_env = self.missing_env()
        missing_modules = self.missing_modules()
        problem = None
        if missing_modules:
            problem = self._install_hint()
        elif missing_env:
            problem = "Set " + ", ".join(missing_env) + " in Settings."
        return {
            "provider": self.name,
            "label": self.label,
            "ready": not missing_env and not missing_modules,
            "missing_env": missing_env,
            "missing_modules": missing_modules,
            "problem": problem,
            "tts": self.supports_tts,
            "stt": self.supports_stt,
            "realtime": self.supports_realtime,
        }

    def _install_hint(self) -> str:
        from ..extras import install_command

        command = install_command([self.extra]) if self.extra else ""
        return f"{self.label} voice needs its packages: {command}".strip()

    def require_ready(self) -> None:
        status = self.status()
        if not status["ready"]:
            raise VoiceNotConfigured(
                status["problem"] or f"{self.label} voice is not configured",
                status["missing_env"] + status["missing_modules"],
            )

    # ---- turn by turn ----------------------------------------------------

    def synthesize(self, text: str) -> VoiceAudio:
        raise VoiceUnsupported(
            f"{self.label} speaks only inside a live call; it has no text to speech "
            "request."
        )

    def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> str:
        raise VoiceUnsupported(
            f"{self.label} listens only inside a live call; it has no transcription "
            "request."
        )

    # ---- live call -------------------------------------------------------

    def realtime_session(
        self,
        *,
        agent_id: str,
        agent_name: str,
        reply: Brain,
        greeting: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Start a live call and return what the browser needs to join it."""
        raise VoiceUnsupported(f"{self.label} has no live call.")

    def end_realtime_session(self, session_id: str) -> None:
        """Stop a live call started by :meth:`realtime_session` (no-op if gone)."""

    def close(self) -> None:
        """Release what the provider keeps open."""


def audio_filename(mime_type: str) -> str:
    """A filename whose extension matches a browser recording's MIME type."""
    kind = (mime_type or "").split(";")[0].strip().lower()
    extension = {
        "audio/webm": "webm",
        "video/webm": "webm",
        "audio/ogg": "ogg",
        "audio/opus": "opus",
        "audio/mp4": "mp4",
        "audio/x-m4a": "m4a",
        "audio/m4a": "m4a",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/flac": "flac",
    }.get(kind, "webm")
    return f"speech.{extension}"
