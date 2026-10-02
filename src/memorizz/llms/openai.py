# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import hashlib
import json
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import openai

from .llm_provider import LLMProvider, ResponseMetadataMixin
from .message_roles import developer_messages_to_system
from .tool_metadata import ResponsesToolMetadataMixin

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    pass


class OpenAI(ResponsesToolMetadataMixin, ResponseMetadataMixin, LLMProvider):
    """
    A class for interacting with the OpenAI API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        context_window_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Any] = None,
        prompt_cache_retention: Optional[str] = None,
        api_mode: str = "chat_completions",
        additional_config: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        """
        Initialize the OpenAI client.

        Parameters:
        -----------
        api_key : str
            The API key for the OpenAI API. Local OpenAI-compatible servers
            (llama.cpp, LM Studio, vLLM) ignore this; pass any non-empty
            string to satisfy the SDK's required-arg check.
        model : str, optional
            The model to use for the OpenAI API.
        base_url : str, optional
            Override the API endpoint. Set this to point at a local
            OpenAI-compatible server (e.g. ``http://127.0.0.1:8080/v1``
            for llama.cpp's ``llama-server``, or LM Studio's local URL).
            Falls back to the official OpenAI endpoint when unset.
        reasoning_effort : str, optional
            Reasoning effort forwarded to the selected API. GPT-5.6
            function tools on this endpoint can use ``"none"``; Responses
            API text helpers keep their provider defaults. GPT-6 Astra
            requires at least ``"low"`` and Responses for function tools.
        max_completion_tokens : int, optional
            Maximum generated tokens for current reasoning models. For the
            GPT-5.5, GPT-5.6 and GPT-6 families, the legacy ``max_tokens``
            argument is translated to this parameter automatically for
            backwards compatibility.
        prompt_cache_retention : str, optional
            OpenAI prompt-cache retention policy (``"in_memory"`` or
            ``"24h"``). Only sent to the official OpenAI endpoint; local
            OpenAI-compatible servers don't understand it.
        api_mode : str, optional
            Tool-loop endpoint: ``"chat_completions"`` (legacy/default) or
            ``"responses"``. The Responses API is required for GPT-5.6
            function tools combined with non-zero reasoning effort and for
            all GPT-6 Astra function tools.
        """
        # Local OpenAI-compatible servers don't authenticate but the SDK
        # still requires a non-empty api_key. Substitute a placeholder so
        # users running llama.cpp / LM Studio don't need to set anything.
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url and not api_key:
            api_key = "sk-local"

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.base_url = base_url
        normalized_api_mode = str(api_mode or "chat_completions").strip().lower()
        if normalized_api_mode not in {"chat_completions", "responses"}:
            raise ValueError("api_mode must be 'chat_completions' or 'responses'")
        if base_url and normalized_api_mode == "responses":
            raise ValueError(
                "api_mode='responses' is only supported by the official OpenAI endpoint"
            )
        self.api_mode = normalized_api_mode
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens(model)
        )
        self.reasoning_effort = reasoning_effort
        self._last_usage: Optional[Dict[str, int]] = None
        # Prompt-cache routing key (see OpenAI's prompt-caching guide). Set
        # per conversation thread by MemAgent via ``set_prompt_cache_key`` so
        # requests sharing a prefix land on the same cache shard.
        self._prompt_cache_key: Optional[str] = None
        self._prompt_cache_retention = prompt_cache_retention
        self._request_options: Dict[str, Any] = {}
        # Current reasoning models reject the legacy ``max_tokens`` field.
        # Preserve the public constructor while translating to the API name.
        uses_completion_limit = model.lower().startswith(
            ("gpt-5.5", "gpt-5.6", "gpt-6")
        )
        if uses_completion_limit:
            if max_completion_tokens is None:
                max_completion_tokens = max_tokens
            max_tokens = None

        request_options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "reasoning_effort": reasoning_effort,
            "response_format": response_format,
        }
        for key, value in request_options.items():
            if value is not None:
                self._request_options[key] = value

        if additional_config:
            allowed_keys = {
                "temperature",
                "max_tokens",
                "max_completion_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "reasoning_effort",
                "response_format",
                "logprobs",
                "top_logprobs",
            }
            for key, value in additional_config.items():
                if key in allowed_keys and value is not None:
                    self._request_options[key] = value
            # Cache retention rides its own attribute (not _request_options)
            # so the base_url guard in _apply_cache_options still applies.
            if additional_config.get("prompt_cache_retention"):
                self._prompt_cache_retention = additional_config[
                    "prompt_cache_retention"
                ]

    def _infer_context_window_tokens(self, model: str) -> int:
        """Best-effort mapping of well-known OpenAI models to their context window."""
        normalized = model.lower() if model else ""
        if normalized == "gpt-6-astra" or normalized.startswith("gpt-6-astra-"):
            return 1_050_000
        context_map = {
            # GPT-5 family
            "gpt-5.5": 1_050_000,
            "gpt-5.5-pro": 1_050_000,
            "gpt-5.6": 1_050_000,
            "gpt-5.6-sol": 1_050_000,
            "gpt-5.6-terra": 1_050_000,
            "gpt-5.6-luna": 1_050_000,
            "gpt-5.2": 1_047_576,
            "gpt-5.2-pro": 1_047_576,
            "gpt-5.1": 1_047_576,
            "gpt-5": 128_000,
            "gpt-5-mini": 1_047_576,
            "gpt-5-nano": 1_047_576,
            # GPT-4.1 family
            "gpt-4.1": 1_047_576,
            "gpt-4.1-mini": 1_047_576,
            "gpt-4.1-nano": 1_047_576,
            # Reasoning models
            "o3": 200_000,
            "o3-pro": 200_000,
            "o4-mini": 200_000,
            "o3-mini": 200_000,
            # GPT-4o family
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "gpt-4-turbo": 128_000,
            "gpt-4o-realtime-preview": 128_000,
        }
        return context_map.get(normalized, 128_000)

    def get_config(self) -> Dict[str, Any]:
        """Returns a serializable configuration for the OpenAI provider."""
        config: Dict[str, Any] = {"provider": "openai", "model": self.model}
        config["context_window_tokens"] = self.context_window_tokens
        if self._request_options:
            config["additional_config"] = deepcopy(self._request_options)
        if self._prompt_cache_retention is not None:
            config["prompt_cache_retention"] = self._prompt_cache_retention
        if self.api_mode != "chat_completions":
            config["api_mode"] = self.api_mode
        if self.base_url:
            config["base_url"] = self.base_url
        if self.reasoning_effort is not None:
            config["reasoning_effort"] = self.reasoning_effort
        return config

    def set_prompt_cache_key(self, key: Optional[str]) -> None:
        """Set the prompt-cache routing key for subsequent requests.

        OpenAI routes requests to cache shards by a hash of the prompt prefix
        combined with this key; keeping it stable per conversation thread
        maximizes cache hits. ``None`` clears it.
        """
        if not key:
            self._prompt_cache_key = None
            return
        normalized = str(key)
        # The official API accepts at most 64 characters. Thread, tenant and
        # agent identifiers can easily exceed that, so retain deterministic
        # cache affinity without leaking or truncating identifying suffixes.
        if len(normalized) > 64:
            normalized = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        self._prompt_cache_key = normalized

    def _apply_cache_options(self, kwargs: Dict[str, Any]) -> None:
        """Attach prompt-caching parameters for the official OpenAI endpoint.

        Skipped for custom ``base_url`` targets (llama.cpp, LM Studio, vLLM):
        those servers don't implement OpenAI's prompt-cache routing and some
        reject unknown parameters.
        """
        if getattr(self, "base_url", None):
            self._last_prompt_cache_metadata = {}
            return
        key = self._prompt_cache_key or self._instruction_cache_key(kwargs)
        if key:
            kwargs["prompt_cache_key"] = key
        if self._prompt_cache_retention:
            kwargs["prompt_cache_retention"] = self._prompt_cache_retention
        self._apply_explicit_cache_boundaries(kwargs)
        from .prompt_cache import cache_request_metadata

        self._last_prompt_cache_metadata = cache_request_metadata(kwargs, "openai")

    def _instruction_cache_key(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Default routing key for callers that did not set one.

        Responses requests without a key were not observed to reuse a shared
        prefix, so single-shot callers (Evalground readers and judges, tool
        metadata helpers) route by model and reusable instructions instead.
        """
        parts = [kwargs.get("instructions")]
        items = kwargs.get("input", kwargs.get("messages"))
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("role") not in {
                "system",
                "developer",
            }:
                break
            parts.append(item.get("content"))
        if not any(parts):
            return None
        digest = hashlib.sha256(
            json.dumps([self.model, parts], sort_keys=True, default=str).encode()
        ).hexdigest()
        return "memorizz:" + digest[:40]

    def _apply_explicit_cache_boundaries(self, kwargs: Dict[str, Any]) -> None:
        """Keep reusable prefix boundaries on GPT-5.6/6 requests.

        These models write implicitly only at the latest message, which
        carries MemAgent's per-turn volatile context, and read only from
        explicit breakpoints on input content (assistant output breakpoints
        are not honored for reads). Two input boundaries keep reuse alive
        across both Chat Completions and Responses:

        1. the end of the leading system/developer instructions;
        2. the last user message before the final one — stable history that
           the next turn's boundary finds by lookback, so each turn re-bills
           only its newest exchange instead of the whole conversation.

        Tool-loop continuations already reuse the implicit write. Caller
        breakpoints are respected as-is; caller messages are never mutated.
        """
        if not self.model.startswith(("gpt-5.6", "gpt-6")):
            return
        field = "input" if "input" in kwargs else "messages"
        items = kwargs.get(field)
        if field == "input" and isinstance(items, str) and kwargs.get("instructions"):
            items = [
                {"role": "developer", "content": kwargs.pop("instructions")},
                {"role": "user", "content": items},
            ]
        if not isinstance(items, list):
            return
        if any(
            isinstance(block, dict) and block.get("prompt_cache_breakpoint")
            for item in items
            if isinstance(item, dict)
            for key in ("content", "output")
            for block in (item.get(key) if isinstance(item.get(key), list) else [])
        ):
            return
        items = deepcopy(items)
        text_type = "input_text" if field == "input" else "text"

        def mark(item: Any) -> bool:
            content = item.get("content")
            if isinstance(content, str) and content:
                item["content"] = [{"type": text_type, "text": content}]
            for block in reversed(item.get("content") or []):
                if (
                    isinstance(block, dict)
                    and block.get("type") == text_type
                    and block.get("text")
                ):
                    block["prompt_cache_breakpoint"] = {"mode": "explicit"}
                    return True
            return False

        leading = 0
        while (
            leading < len(items)
            and isinstance(items[leading], dict)
            and items[leading].get("role") in {"system", "developer"}
        ):
            leading += 1
        marked = any(mark(items[index]) for index in reversed(range(leading)))
        user_turns = [
            index
            for index in range(leading, len(items))
            if isinstance(items[index], dict) and items[index].get("role") == "user"
        ]
        if len(user_turns) >= 2:
            marked = mark(items[user_turns[-2]]) or marked
        if marked:
            kwargs[field] = items

    def _create_chat_completion(self, kwargs: Dict[str, Any]) -> Any:
        """Call chat.completions.create, dropping cache params on old SDKs.

        Older ``openai`` SDK releases raise ``TypeError`` for the
        prompt-cache parameters; retry once without them rather than failing
        the whole request.
        """
        try:
            return self.client.chat.completions.create(**kwargs)
        except TypeError:
            trimmed = {
                k: v
                for k, v in kwargs.items()
                if k not in ("prompt_cache_key", "prompt_cache_retention")
            }
            if len(trimmed) == len(kwargs):
                raise
            self._last_prompt_cache_metadata[
                "prompt_cache_warning"
            ] = "cache_options_dropped"
            self._last_prompt_cache_metadata.pop("prompt_cache_key", None)
            return self.client.chat.completions.create(**trimmed)

    @staticmethod
    def _responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate the internal Chat Completions history to Responses items."""

        items: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user")
            if role == "tool":
                output = message.get("content", "")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False, default=str)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": str(message.get("tool_call_id") or ""),
                        "output": output,
                    }
                )
                continue

            content = message.get("content")
            if content not in (None, ""):
                items.append({"role": role, "content": content})

            for call in message.get("tool_calls") or []:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                call_id = call.get("id", "") if isinstance(call, dict) else ""
                items.append(
                    {
                        "type": "function_call",
                        "call_id": str(call_id),
                        "name": str(function.get("name") or "unknown"),
                        "arguments": str(function.get("arguments") or "{}"),
                    }
                )
        return items

    @staticmethod
    def _responses_tools(
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Translate Chat Completions function schemas to Responses schemas."""

        if not tools:
            return None
        translated: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                translated.append(dict(tool))
                continue
            function = dict(tool.get("function") or {})
            translated.append(
                {
                    "type": "function",
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "parameters": function.get("parameters")
                    or {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    # Do not opt schemas into OpenAI strict mode implicitly.
                    # MemoRizz allows defaulted optional parameters, whereas
                    # Responses strict mode requires every property to appear
                    # in ``required`` (using nullable types for optionals).
                    # MemoRizz still performs its own exact signature binding
                    # and additionalProperties enforcement before dispatch.
                    "strict": bool(function.get("strict", False)),
                }
            )
        return translated

    def _extract_responses_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        try:
            prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            extracted: Dict[str, int] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            input_details = getattr(usage, "input_tokens_details", None)
            cached = (
                getattr(input_details, "cached_tokens", None) if input_details else None
            )
            if cached is not None:
                extracted["cached_tokens"] = cached
            writes = getattr(input_details, "cache_write_tokens", None)
            if writes is not None:
                extracted["cache_write_tokens"] = writes
            output_details = getattr(usage, "output_tokens_details", None)
            reasoning = (
                getattr(output_details, "reasoning_tokens", None)
                if output_details
                else None
            )
            if reasoning is not None:
                extracted["reasoning_tokens"] = reasoning
            return extracted
        except Exception:
            return None

    def _responses_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: str,
    ) -> Any:
        """Generate through Responses while preserving MemoRizz's tool contract."""

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": self._responses_input(messages),
            "store": False,
        }
        translated_tools = self._responses_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools
            kwargs["tool_choice"] = tool_choice

        max_output_tokens = self._request_options.get("max_completion_tokens")
        if max_output_tokens is None:
            max_output_tokens = self._request_options.get("max_tokens")
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        for key in ("temperature", "top_p"):
            value = self._request_options.get(key)
            if value is not None:
                kwargs[key] = value
        self._apply_cache_options(kwargs)
        return kwargs

    def _generate_responses(self, messages, tools, tool_choice):
        kwargs = self._responses_kwargs(messages, tools, tool_choice)
        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.responses.create(**kwargs)
        self._last_usage = self._extract_responses_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=kwargs.get("max_output_tokens")
            or kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens"),
        )

        function_calls = [
            item
            for item in getattr(response, "output", []) or []
            if getattr(item, "type", None) == "function_call"
        ]
        if function_calls:
            from types import SimpleNamespace

            tool_calls = [
                SimpleNamespace(
                    id=str(getattr(item, "call_id", "") or getattr(item, "id", "")),
                    type="function",
                    function=SimpleNamespace(
                        name=str(getattr(item, "name", "") or "unknown"),
                        arguments=str(getattr(item, "arguments", "") or "{}"),
                    ),
                )
                for item in function_calls
            ]
            message = SimpleNamespace(
                content=getattr(response, "output_text", None) or None,
                tool_calls=tool_calls,
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                response_id=getattr(response, "id", None),
            )
        return getattr(response, "output_text", "") or ""

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "store": False,
        }
        if instructions:
            kwargs["instructions"] = instructions

        max_output_tokens = self._request_options.get("max_completion_tokens")
        if max_output_tokens is None:
            max_output_tokens = self._request_options.get("max_tokens")
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if self.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        for key in ("temperature", "top_p"):
            value = self._request_options.get(key)
            if value is not None:
                kwargs[key] = value
        self._apply_cache_options(kwargs)
        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.responses.create(**kwargs)
        self._last_usage = self._extract_responses_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=kwargs.get("max_output_tokens")
            or kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens"),
        )
        return response.output_text

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """
        Generate a response using OpenAI's chat completions API.

        Parameters:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content'.
                Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
            tools (Optional[List[Dict[str, Any]]]): List of tool definitions for function calling.
            tool_choice (str): Controls which (if any) function is called ("auto", "none", or specific function).

        Returns:
            Any: Either a string (final response) or the full response object (if tool calls are present).
        """
        if getattr(self, "api_mode", "chat_completions") == "responses":
            return self._generate_responses(messages, tools, tool_choice)

        provider_messages = (
            developer_messages_to_system(messages) if self.base_url else messages
        )
        kwargs = {"model": self.model, "messages": provider_messages}

        # Add tools if provided
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self._request_options:
            kwargs.update(self._request_options)
        self._apply_cache_options(kwargs)

        self._last_usage = None
        self._last_response_metadata = {}
        response = self._create_chat_completion(kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=kwargs.get("max_output_tokens")
            or kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens"),
        )

        # If there are tool calls, return the full response object
        if response.choices[0].message.tool_calls:
            return response

        # Otherwise return just the text content
        return response.choices[0].message.content

    def generate_stream(self, messages, tools=None, tool_choice="auto"):
        """Native provider deltas, completed tool calls, usage and terminal state."""
        from .streaming import chat_events, responses_events

        self._last_usage = None
        self._last_response_metadata = {}
        if getattr(self, "api_mode", "chat_completions") == "responses":
            kwargs = self._responses_kwargs(messages, tools, tool_choice)
            kwargs["stream"] = True
            yield from responses_events(
                self,
                self.client.responses.create(**kwargs),
                max_output_tokens=kwargs.get("max_output_tokens"),
            )
            return
        kwargs = {
            "model": self.model,
            "messages": developer_messages_to_system(messages)
            if self.base_url
            else messages,
            "stream": True,
        }
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice)
        kwargs.update(self._request_options)
        self._apply_cache_options(kwargs)
        if not self.base_url:
            kwargs.setdefault("stream_options", {"include_usage": True})
        yield from chat_events(
            self,
            self._create_chat_completion(kwargs),
            max_output_tokens=kwargs.get("max_completion_tokens")
            or kwargs.get("max_tokens"),
        )

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        try:
            extracted = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
            # Prompt-cache observability: how much of the prompt was served
            # from OpenAI's prefix cache (billed at the cached-input rate).
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details else None
            if cached is not None:
                extracted["cached_tokens"] = cached
            writes = getattr(details, "cache_write_tokens", None)
            if writes is not None:
                extracted["cache_write_tokens"] = writes
            return extracted
        except Exception:
            return None

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens
