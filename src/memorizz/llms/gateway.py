# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Model gateways: one endpoint, one key, many models.

:class:`ModelGateway` talks to any service that speaks the OpenAI chat
protocol on its own address with its own key. :class:`OpenRouter` is that
gateway with OpenRouter's address, key and routing built in, and
:class:`Gemini` is Google's own OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Union

from .openai import OpenAI
from .routing import ModelRouting
from .tool_metadata import JsonPromptToolMetadataMixin

logger = logging.getLogger(__name__)

RoutingLike = Union[ModelRouting, Mapping[str, Any], None]


class ModelGateway(JsonPromptToolMetadataMixin, OpenAI):
    """A model served by a gateway that speaks the OpenAI chat protocol.

    Parameters
    ----------
    model:
        The model's name on the gateway.
    base_url:
        The gateway's address, such as ``https://gateway.example/v1``.
    api_key_env:
        The environment variable that holds the gateway's key. Only this
        name is saved with an agent, never the key.
    api_key:
        The key itself, for this process only. Without a key or a variable
        name the gateway is sent a placeholder, never ``OPENAI_API_KEY``.
    headers:
        Extra headers sent with every request.
    extra_body:
        Extra fields added to every chat request, for a gateway's own options.

    Everything else (``temperature``, ``max_tokens``, ``context_window_tokens``
    and so on) is as on :class:`~memorizz.llms.OpenAI`.
    """

    provider_name = "gateway"
    DEFAULT_BASE_URL: Optional[str] = None
    DEFAULT_API_KEY_ENV: Optional[str] = None

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        **options: Any,
    ):
        address = str(base_url or self.DEFAULT_BASE_URL or "").strip().rstrip("/")
        if not address:
            raise ValueError("A model gateway needs its address (base_url)")
        if not str(model or "").strip():
            raise ValueError("A model gateway needs the name of a model")
        self.api_key_env = str(api_key_env or self.DEFAULT_API_KEY_ENV or "") or None
        key = api_key or (os.getenv(self.api_key_env) if self.api_key_env else None)
        if not key and self.api_key_env:
            raise ValueError(
                f"{type(self).__name__} needs a key: set {self.api_key_env} "
                "or pass api_key"
            )
        self._headers = {str(k): str(v) for k, v in dict(headers or {}).items()}
        self._extra_body = deepcopy(dict(extra_body or {}))
        # Gateways are driven through chat completions only; a key is always
        # passed so the OpenAI provider never reaches for OPENAI_API_KEY.
        options.pop("api_mode", None)
        super().__init__(
            api_key=key or "sk-local",
            model=str(model).strip(),
            base_url=address,
            api_mode="chat_completions",
            default_headers=self._headers or None,
            **options,
        )

    # --- what is saved with an agent ---

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["provider"] = self.provider_name
        if self.base_url == (self.DEFAULT_BASE_URL or "").rstrip("/"):
            config.pop("base_url", None)
        if self.api_key_env and self.api_key_env != self.DEFAULT_API_KEY_ENV:
            config["api_key_env"] = self.api_key_env
        if self._extra_body:
            config["extra_body"] = deepcopy(self._extra_body)
        return config

    # --- requests ---

    def _request_body(self) -> Dict[str, Any]:
        """Fields added to every chat request."""
        return deepcopy(self._extra_body)

    def _create_chat_completion(self, kwargs: Dict[str, Any]) -> Any:
        body = self._request_body()
        if body:
            kwargs = {**kwargs, "extra_body": {**body, **kwargs.get("extra_body", {})}}
        return super()._create_chat_completion(kwargs)

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        """Plain text for a prompt, through the same chat request an agent's
        turns use, so the gateway's rules apply here too."""
        messages: List[Dict[str, Any]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
        answer = self.generate(messages)
        return answer if isinstance(answer, str) else (answer or "")


_context_lock = threading.Lock()
# model -> context length, from OpenRouter's public model list.
_context_cache: Dict[str, int] = {}
_context_asked_at = 0.0


def openrouter_context_windows(refresh: bool = False) -> Dict[str, int]:
    """Each OpenRouter model's context length. Asked once an hour at most,
    and empty when the list cannot be reached."""
    global _context_asked_at
    with _context_lock:
        fresh = time.monotonic() - _context_asked_at < 3600
        if (_context_asked_at and fresh and not refresh) or os.getenv(
            "MEMORIZZ_OFFLINE"
        ):
            return dict(_context_cache)
        _context_asked_at = time.monotonic()
    try:
        import requests

        rows = (
            requests.get(f"{OpenRouter.DEFAULT_BASE_URL}/models", timeout=4)
            .json()
            .get("data", [])
        )
        found = {
            str(row["id"]): int(row["context_length"])
            for row in rows
            if row.get("id") and row.get("context_length")
        }
    except Exception as exc:
        logger.debug("OpenRouter's model list could not be read: %s", exc)
        found = {}
    with _context_lock:
        if found:
            _context_cache.clear()
            _context_cache.update(found)
        return dict(_context_cache)


class OpenRouter(ModelGateway):
    """A model on `OpenRouter <https://openrouter.ai>`_.

    The key is read from ``OPENROUTER_API_KEY``. ``routing`` says which
    providers may answer and which models to fall back to::

        OpenRouter(
            "anthropic/claude-sonnet-5.5",
            routing=ModelRouting(
                fallback_models=["openai/gpt-6.2-sol"],
                sort="throughput",
                data_collection="deny",
            ),
        )

    ``app_name`` and ``app_url`` identify the application to OpenRouter.
    """

    provider_name = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
    DEFAULT_APP_NAME = "MemoRizz"
    DEFAULT_APP_URL = "https://github.com/RichmondAlake/memorizz"

    def __init__(
        self,
        model: str = "openrouter/auto",
        routing: RoutingLike = None,
        api_key: Optional[str] = None,
        app_name: Optional[str] = None,
        app_url: Optional[str] = None,
        **options: Any,
    ):
        self.routing = ModelRouting.coerce(routing) or ModelRouting()
        self.app_name = app_name or self.DEFAULT_APP_NAME
        self.app_url = app_url or self.DEFAULT_APP_URL
        headers = {
            "HTTP-Referer": self.app_url,
            "X-OpenRouter-Title": self.app_name,
            **dict(options.pop("headers", None) or {}),
        }
        super().__init__(model, api_key=api_key, headers=headers, **options)

    def with_routing(self, routing: RoutingLike) -> "OpenRouter":
        """Replace the routing rules (``None`` clears them)."""
        self.routing = ModelRouting.coerce(routing) or ModelRouting()
        return self

    def _infer_context_window_tokens(self, model: str) -> int:
        return openrouter_context_windows().get(str(model), 128_000)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        if self.routing:
            config["routing"] = self.routing.to_dict()
        if self.app_name != self.DEFAULT_APP_NAME:
            config["app_name"] = self.app_name
        if self.app_url != self.DEFAULT_APP_URL:
            config["app_url"] = self.app_url
        return config

    def _create_chat_completion(self, kwargs: Dict[str, Any]) -> Any:
        if kwargs.get("stream"):
            # A streamed answer reports its tokens and charge in a last chunk.
            kwargs = {**kwargs, "stream_options": {"include_usage": True}}
        return super()._create_chat_completion(kwargs)

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = super()._extract_usage(response)
        charged = getattr(getattr(response, "usage", None), "cost", None)
        if usage is not None and isinstance(charged, (int, float)):
            # What OpenRouter charged for this call, in dollars.
            usage["cost_usd"] = float(charged)
        return usage

    def _request_body(self) -> Dict[str, Any]:
        body = super()._request_body()
        # Ask for the call's charge to be reported with its token counts.
        body["usage"] = {"include": True}
        rules = self.routing.to_request()
        fallbacks = rules.pop("fallback_models", None)
        if fallbacks:
            # OpenRouter takes the whole chain, the agent's model first.
            body["models"] = [self.model, *[m for m in fallbacks if m != self.model]]
        body.update(rules)
        return body


# Sent for a tool call that carries no thought signature (one another model
# made, or one restored without it), so Gemini accepts the history. Google
# documents this value for exactly that case.
GEMINI_NO_SIGNATURE = "skip_thought_signature_validator"


def _has_thought_signature(call: Mapping[str, Any]) -> bool:
    extra = call.get("extra_content")
    google = extra.get("google") if isinstance(extra, Mapping) else None
    return bool(isinstance(google, Mapping) and google.get("thought_signature"))


def _gemini_error_detail(exc: BaseException, model: str) -> Optional[str]:
    """What Google said went wrong, short enough to show a person."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    try:
        body: Any = json.loads(response.text)
    except Exception:
        body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str) or not message.strip():
        return None
    status = getattr(exc, "status_code", None)
    if status == 429 and "free_tier" in message and "limit: 0" in message:
        return (
            f"Google does not offer {model} on the Gemini API free tier. Turn "
            "on billing for the key's Google project at aistudio.google.com, "
            "or choose another Gemini model."
        )
    if status == 429 and "free_tier" in message:
        limit = re.search(
            r"limit: (\d+), model: [^,]+, description: ([^.\n]+)", message
        )
        if limit:
            # "Generate content requests per minute limit exceeded on Free Tier"
            what = limit.group(2).replace(" limit exceeded on Free Tier", "").strip()
            return (
                f"This key reached Google's free-tier limit for {model} "
                f"({limit.group(1)}: {what[:1].lower() + what[1:]}). Wait and "
                "try again, or turn on billing for the key's Google project."
            )
    return message.strip().splitlines()[0].strip()[:400]


class Gemini(ModelGateway):
    """A Gemini model on the `Gemini API <https://ai.google.dev>`_.

    The key is read from ``GEMINI_API_KEY``, then ``GOOGLE_API_KEY``::

        Gemini("gemini-4-hydrogen", reasoning_effort="low")

    Requests go to Google's OpenAI-compatible endpoint, so tools, streaming
    and ``reasoning_effort`` work as on :class:`~memorizz.llms.OpenAI`.
    Gemini 3 and 4 sign each step that calls a tool and reject a history
    that leaves the signature out; it is kept on the tool call and sent
    back, and a call that has none is marked so the request is accepted.
    """

    provider_name = "gemini"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
    DEFAULT_API_KEY_ENV = "GEMINI_API_KEY"
    FALLBACK_API_KEY_ENV = "GOOGLE_API_KEY"
    DEFAULT_MODEL = "gemini-4-hydrogen"
    # Tool calls keep Google's own fields (the thought signature).
    keeps_tool_call_extras = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        **options: Any,
    ):
        name = str(model or "").strip()
        if name.startswith("models/"):
            # The model list names them "models/gemini-..."; requests don't.
            name = name[len("models/") :]
        if not api_key and not options.get("api_key_env"):
            api_key = (
                os.getenv(self.DEFAULT_API_KEY_ENV)
                or os.getenv(self.FALLBACK_API_KEY_ENV)
                or None
            )
        super().__init__(name, api_key=api_key, **options)

    def _infer_context_window_tokens(self, model: str) -> int:
        # Every current Gemini text model takes 1,048,576 tokens.
        return 1_048_576 if str(model).lower().startswith("gemini-") else 128_000

    @staticmethod
    def _with_signatures(messages: Any) -> Any:
        """The history with every tool-calling step signed: the first call of
        a step that has no signature gets the documented stand-in."""
        if not isinstance(messages, list):
            return messages
        signed: List[Any] = []
        for message in messages:
            calls = message.get("tool_calls") if isinstance(message, dict) else None
            if not calls or not isinstance(calls[0], dict):
                signed.append(message)
                continue
            if _has_thought_signature(calls[0]):
                signed.append(message)
                continue
            extra = dict(calls[0].get("extra_content") or {})
            extra["google"] = {
                **dict(extra.get("google") or {}),
                "thought_signature": GEMINI_NO_SIGNATURE,
            }
            first = {**calls[0], "extra_content": extra}
            signed.append({**message, "tool_calls": [first, *calls[1:]]})
        return signed

    def _create_chat_completion(self, kwargs: Dict[str, Any]) -> Any:
        kwargs = {**kwargs, "messages": self._with_signatures(kwargs.get("messages"))}
        if kwargs.get("stream"):
            # A streamed answer reports its tokens in a last chunk.
            kwargs.setdefault("stream_options", {"include_usage": True})
        try:
            return super()._create_chat_completion(kwargs)
        except Exception as exc:
            detail = _gemini_error_detail(exc, str(kwargs.get("model") or self.model))
            if detail:
                # Google answers with a list, which the client cannot read a
                # message from: without this the error is a bare status code.
                status = getattr(exc, "status_code", None)
                text = f"Error code: {status} - {detail}" if status else detail
                exc.detail = detail  # type: ignore[attr-defined]
                exc.args = (text,)
                if hasattr(exc, "message"):
                    exc.message = text  # type: ignore[attr-defined]
            raise


__all__ = [
    "GEMINI_NO_SIGNATURE",
    "Gemini",
    "ModelGateway",
    "OpenRouter",
    "openrouter_context_windows",
]
