"""Voice settings of an agent: which provider speaks and listens, and how."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

#: Providers an agent can use, the default first.
VOICE_PROVIDERS = ("livekit", "elevenlabs", "vapi", "openai")
DEFAULT_VOICE_PROVIDER = "livekit"

_FIELDS = ("provider", "voice", "tts_model", "stt_model", "language", "speed")
_ALIASES = {
    "voice_id": "voice",
    "model": "tts_model",
    "tts": "tts_model",
    "stt": "stt_model",
    "eleven_labs": "elevenlabs",
    "11labs": "elevenlabs",
}
# Credentials come from the environment (Settings), never from a saved agent.
_SECRET_WORDS = ("api_key", "apikey", "secret", "token", "password")


@dataclass(frozen=True)
class VoiceConfig:
    """How one agent speaks and listens.

    ``provider`` picks the service; the other fields are optional and each
    provider falls back to its own default. ``options`` holds settings only one
    provider understands, for example ``{"public_url": "https://..."}`` for
    Vapi or ``{"agent_name": "..."}`` for LiveKit.
    """

    provider: str = DEFAULT_VOICE_PROVIDER
    voice: Optional[str] = None
    tts_model: Optional[str] = None
    stt_model: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "VoiceConfig":
        """Build a config from None/True (defaults), a provider name, a dict
        or another ``VoiceConfig``. Raises ValueError on anything else, on an
        unknown provider and on a credential in the dict."""
        if value is None or value is True or value is False:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(provider=_provider(value))
        if not isinstance(value, dict):
            raise ValueError(
                "voice_config must be a provider name, a dict or a VoiceConfig"
            )
        known: Dict[str, Any] = {}
        options: Dict[str, Any] = dict(value.get("options") or {})
        for key, item in value.items():
            if key == "options":
                continue
            name = _ALIASES.get(str(key), str(key))
            if name in _FIELDS:
                known[name] = item
            else:
                options[str(key)] = item
        for key in options:
            if any(word in key.lower() for word in _SECRET_WORDS):
                raise ValueError(
                    f"voice_config must not hold credentials ({key}); "
                    "set the provider's key in the environment or in Settings"
                )
        speed = known.get("speed")
        if speed in ("", None):
            speed = None
        else:
            try:
                speed = float(speed)
            except (TypeError, ValueError):
                raise ValueError("voice_config speed must be a number") from None
            if not 0.25 <= speed <= 4.0:
                raise ValueError("voice_config speed must be between 0.25 and 4")
        return cls(
            provider=_provider(known.get("provider") or DEFAULT_VOICE_PROVIDER),
            voice=_text(known.get("voice")),
            tts_model=_text(known.get("tts_model")),
            stt_model=_text(known.get("stt_model")),
            language=_text(known.get("language")),
            speed=speed,
            options=options,
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-safe dict holding only what was set (plus the provider)."""
        data: Dict[str, Any] = {"provider": self.provider}
        for name in _FIELDS[1:]:
            item = getattr(self, name)
            if item is not None:
                data[name] = item
        if self.options:
            data["options"] = dict(self.options)
        return data

    def option(self, name: str, default: Any = None) -> Any:
        item = self.options.get(name)
        return default if item in (None, "") else item


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _provider(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "").replace(" ", "")
    name = _ALIASES.get(name, name)
    if name not in VOICE_PROVIDERS:
        raise ValueError(
            f"Unknown voice provider {value!r}; choose one of "
            + ", ".join(VOICE_PROVIDERS)
        )
    return name


def voice_settings(
    current_enabled: Any = False,
    current_config: Any = None,
    *,
    enabled: Optional[bool] = None,
    provider: Optional[str] = None,
    voice: Optional[str] = None,
) -> "tuple[bool, Optional[Dict[str, Any]]]":
    """An agent's ``(voice, voice_config)`` after a change.

    Each argument left as None keeps what the agent has. Naming a provider or
    a voice without ``enabled`` turns voice on (asking for a voice means
    wanting one); ``voice=""`` goes back to the provider's default voice.
    Changing provider drops the settings that belonged to the old one.
    Raises ValueError for an unknown provider.
    """
    config: Dict[str, Any] = (
        dict(current_config) if isinstance(current_config, dict) else {}
    )
    if provider is not None and str(provider).strip():
        chosen = _provider(provider)
        if config.get("provider", DEFAULT_VOICE_PROVIDER) != chosen:
            config = {}
        config["provider"] = chosen
    if voice is not None:
        if str(voice).strip():
            config["voice"] = str(voice).strip()
        else:
            config.pop("voice", None)
    if enabled is None:
        changed = provider is not None or (voice is not None and str(voice).strip())
        is_on = True if changed else bool(current_enabled)
    else:
        is_on = bool(enabled)
    if not config:
        return is_on, None
    return is_on, VoiceConfig.from_value(config).to_dict()
