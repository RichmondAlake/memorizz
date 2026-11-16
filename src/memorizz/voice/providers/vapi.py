"""Vapi: a live voice call whose language model is the MemoRizz agent.

Vapi runs the call (listening, turn taking, speaking) on its own servers and
asks a "custom LLM" for each reply. That URL is this process's
``/webhook/voice/vapi/<agent>/<token>/chat/completions``, so it must be
reachable from the internet over https: set ``MEMORIZZ_PUBLIC_URL`` (or
``voice_config={"public_url": ...}``) to a tunnel or deployed address.

A browser key may not name a model URL, so the URL lives on a stored assistant
created with the private key; the browser only receives the assistant's id.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from ..base import Brain, VoiceError, VoiceNotConfigured, VoiceProvider
from ..brain import set_brain
from ..config import VoiceConfig
from . import _http

API = "https://api.vapi.ai"
WEBHOOK_PREFIX = "/webhook/voice/vapi"


def webhook_token(agent_id: str) -> str:
    """The path token that proves a chat-completions call came from the
    assistant this process configured (HMAC of the agent id, private key)."""
    secret = os.environ.get("VAPI_API_KEY", "")
    if not secret:
        return ""
    return hmac.new(
        secret.encode(), str(agent_id).encode(), hashlib.sha256
    ).hexdigest()[:40]


def valid_webhook_token(agent_id: str, token: str) -> bool:
    expected = webhook_token(agent_id)
    return bool(expected) and hmac.compare_digest(expected, str(token or ""))


def public_url(config: Optional[VoiceConfig] = None) -> str:
    """The https address Vapi can reach this process on, or ''."""
    configured = (config.option("public_url") if config else None) or os.environ.get(
        "MEMORIZZ_PUBLIC_URL", ""
    )
    return str(configured or "").strip().rstrip("/")


def _public_url_problem(url: str) -> Optional[str]:
    if not url:
        return (
            "Vapi calls the agent from its own servers: set MEMORIZZ_PUBLIC_URL in "
            "Settings to a public https address for this UI (for example a tunnel)."
        )
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "https":
        return "MEMORIZZ_PUBLIC_URL must start with https:// for Vapi."
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        return "MEMORIZZ_PUBLIC_URL must be reachable from the internet, not localhost."
    if parts.query or parts.fragment or parts.username:
        return "MEMORIZZ_PUBLIC_URL must not carry credentials, a query or a fragment."
    return None


class VapiVoice(VoiceProvider):
    name = "vapi"
    label = "Vapi"
    supports_realtime = True
    required_env = ("VAPI_API_KEY", "VAPI_PUBLIC_KEY")

    def __init__(self, config: Optional[VoiceConfig] = None, *, http: Any = None):
        super().__init__(config)
        self._http = _http.session(http)

    def status(self) -> Dict[str, Any]:
        status = super().status()
        if status["ready"]:
            problem = _public_url_problem(public_url(self.config))
            if problem:
                status.update(ready=False, problem=problem)
                status["missing_env"] = ["MEMORIZZ_PUBLIC_URL"]
        return status

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ.get('VAPI_API_KEY', '')}",
            "Content-Type": "application/json",
        }

    def _base(self) -> str:
        return str(self.config.option("base_url", API)).rstrip("/")

    def assistant_payload(
        self, agent_id: str, agent_name: str, greeting: Optional[str] = None
    ) -> Dict[str, Any]:
        url = f"{public_url(self.config)}{WEBHOOK_PREFIX}/{agent_id}/{webhook_token(agent_id)}"
        payload: Dict[str, Any] = {
            "name": assistant_name(agent_id),
            "model": {
                "provider": "custom-llm",
                "url": url,
                "model": "memorizz",
                # Only the messages: no call or customer records leave Vapi.
                "metadataSendMode": "off",
            },
            "firstMessage": greeting or f"Hi, this is {agent_name}.",
        }
        voice_provider = self.config.option("voice_provider")
        if self.config.voice and voice_provider:
            payload["voice"] = {
                "provider": str(voice_provider),
                "voiceId": self.config.voice,
            }
        return payload

    def ensure_assistant(
        self, agent_id: str, agent_name: str, greeting: Optional[str] = None
    ) -> str:
        """Create or update the stored assistant for this agent; return its id."""
        payload = self.assistant_payload(agent_id, agent_name, greeting)
        assistant_id = self.config.option("assistant_id") or self._find_assistant(
            payload["name"]
        )
        if assistant_id:
            _http.check(
                self._http.patch(
                    f"{self._base()}/assistant/{assistant_id}",
                    headers=self._headers(),
                    json=payload,
                    timeout=_http.TIMEOUT,
                ),
                self.label,
            )
            return str(assistant_id)
        response = _http.check(
            self._http.post(
                f"{self._base()}/assistant",
                headers=self._headers(),
                json=payload,
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        created = response.json().get("id")
        if not created:
            raise VoiceError("Vapi did not return an assistant id")
        return str(created)

    def _find_assistant(self, name: str) -> Optional[str]:
        response = _http.check(
            self._http.get(
                f"{self._base()}/assistant",
                headers=self._headers(),
                params={"limit": 100},
                timeout=_http.TIMEOUT,
            ),
            self.label,
        )
        body = response.json()
        rows: List[Dict[str, Any]] = body if isinstance(body, list) else []
        for row in rows:
            if row.get("name") == name:
                return str(row.get("id"))
        return None

    def realtime_session(
        self,
        *,
        agent_id: str,
        agent_name: str,
        reply: Brain,
        greeting: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = self.status()
        if not status["ready"]:
            raise VoiceNotConfigured(status["problem"], status["missing_env"])
        assistant_id = self.ensure_assistant(agent_id, agent_name, greeting)
        set_brain(agent_id, reply)
        return {
            "provider": self.name,
            "session_id": f"vapi-{agent_id}",
            "public_key": os.environ.get("VAPI_PUBLIC_KEY", ""),
            "assistant_id": assistant_id,
        }


def assistant_name(agent_id: str) -> str:
    """The stored assistant's name (Vapi allows 40 characters)."""
    return f"mz-{agent_id}"[:40]
