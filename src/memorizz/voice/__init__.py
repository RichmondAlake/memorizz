"""Voice for agents: speaking, listening and live calls.

``MemAgent(voice=True)`` gives an agent a voice through LiveKit, the default
provider. ``voice_config`` picks another provider or tunes it::

    MemAgent(voice=True, voice_config={"provider": "elevenlabs", "voice": "Rachel"})
    MemAgent(voice=True, voice_config={"provider": "vapi"})

Credentials are read from the environment (or set in the UI's Settings); they
are never part of an agent's saved configuration.

==========  =========================================================
Provider    Needs
==========  =========================================================
livekit     LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET and the
            ``voice`` extra
elevenlabs  ELEVENLABS_API_KEY
vapi        VAPI_API_KEY, VAPI_PUBLIC_KEY and MEMORIZZ_PUBLIC_URL
openai      OPENAI_API_KEY
==========  =========================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import (
    Brain,
    VoiceAudio,
    VoiceError,
    VoiceNotConfigured,
    VoiceProvider,
    VoiceUnsupported,
    speakable,
)
from .brain import clear_brain, get_brain, set_brain
from .config import DEFAULT_VOICE_PROVIDER, VOICE_PROVIDERS, VoiceConfig, voice_settings


def _provider_class(name: str) -> Type[VoiceProvider]:
    if name == "livekit":
        from .providers.livekit import LiveKitVoice

        return LiveKitVoice
    if name == "elevenlabs":
        from .providers.elevenlabs import ElevenLabsVoice

        return ElevenLabsVoice
    if name == "vapi":
        from .providers.vapi import VapiVoice

        return VapiVoice
    if name == "openai":
        from .providers.openai import OpenAIVoice

        return OpenAIVoice
    raise ValueError(f"Unknown voice provider {name!r}")


def create_voice_provider(config: Any = None) -> VoiceProvider:
    """The provider a voice configuration names (LiveKit when none is named)."""
    resolved = VoiceConfig.from_value(config)
    return _provider_class(resolved.provider)(resolved)


def voice_status() -> List[Dict[str, Any]]:
    """Every provider with whether it is ready to use and what it can do."""
    return [create_voice_provider(name).status() for name in VOICE_PROVIDERS]


__all__ = [
    "Brain",
    "DEFAULT_VOICE_PROVIDER",
    "VOICE_PROVIDERS",
    "VoiceAudio",
    "VoiceConfig",
    "VoiceError",
    "VoiceNotConfigured",
    "VoiceProvider",
    "VoiceUnsupported",
    "clear_brain",
    "create_voice_provider",
    "get_brain",
    "set_brain",
    "speakable",
    "voice_settings",
    "voice_status",
]
