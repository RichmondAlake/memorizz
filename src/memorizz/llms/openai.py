# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

import openai

from .llm_provider import LLMProvider
from .message_roles import developer_messages_to_system

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# Use TYPE_CHECKING for forward references to avoid circular imports
if TYPE_CHECKING:
    pass


class OpenAI(LLMProvider):
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
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Any] = None,
        prompt_cache_retention: Optional[str] = None,
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
            Reasoning effort forwarded to Chat Completions. GPT-5.6
            function tools on this endpoint can use ``"none"``; Responses
            API text helpers keep their provider defaults.
        prompt_cache_retention : str, optional
            OpenAI prompt-cache retention policy (``"in_memory"`` or
            ``"24h"``). Only sent to the official OpenAI endpoint; local
            OpenAI-compatible servers don't understand it.
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
        request_options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
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
        context_map = {
            # GPT-5 family
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
        self._prompt_cache_key = key or None

    def _apply_cache_options(self, kwargs: Dict[str, Any]) -> None:
        """Attach prompt-caching parameters for the official OpenAI endpoint.

        Skipped for custom ``base_url`` targets (llama.cpp, LM Studio, vLLM):
        those servers don't implement OpenAI's prompt-cache routing and some
        reject unknown parameters.
        """
        if self.base_url:
            return
        if self._prompt_cache_key:
            kwargs["prompt_cache_key"] = self._prompt_cache_key
        if self._prompt_cache_retention:
            kwargs["prompt_cache_retention"] = self._prompt_cache_retention

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
            return self.client.chat.completions.create(**trimmed)

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """
        Get the metadata for a tool.

        Parameters:
        -----------
        func : Callable
            The function to get the metadata for.

        Returns:
        --------
        Dict[str, Any]
        """
        # We'll import ToolSchemaType here to avoid circular imports
        from ..long_term.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert metadata augmentation assistant specializing in JSON schema discovery "
                "and documentation enhancement.\n\n"
                f"**IMPORTANT**: Use the function name exactly as provided (`{func_name}`) and do NOT rename it."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                f"Generate enriched metadata for the function `{func_name}`.\n\n"
                f"- Docstring: {docstring}\n"
                f"- Signature: {signature}\n\n"
                "Enhance the metadata by:\n"
                "• Expanding the docstring into a detailed description.\n"
                "• Writing clear natural‐language descriptions for each parameter, including type, purpose, and constraints.\n"
                "• Identifying which parameters are required.\n"
                "• (Optional) Suggesting example queries or use cases.\n\n"
                "Produce a JSON object that strictly adheres to the ToolSchemaType structure."
            ),
        }

        response = self.client.responses.parse(
            model=self.model, input=[system_msg, user_msg], text_format=ToolSchemaType
        )

        return response.output_parsed

    def augment_docstring(self, docstring: str) -> str:
        """
        Augment the docstring with an LLM generated description.

        Parameters:
        -----------
        docstring : str
            The docstring to augment.

        Returns:
        --------
        str
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Augment the docstring {docstring} by adding more details and examples.",
        )

        return response.output_text

    def generate_queries(self, docstring: str) -> List[str]:
        """
        Generate queries for the tool.

        Parameters:
        -----------
        docstring : str
            The docstring to generate queries for.

        Returns:
        --------
        List[str]
        """
        response = self.client.responses.create(
            model=self.model,
            input=f"Generate queries for the docstring {docstring} by adding some examples of queries that can be used to leverage the tool.",
        )

        return response.output_text

    def generate_text(self, prompt: str, instructions: str = None) -> str:
        """
        Generate text using OpenAI's API.

        Parameters:
            prompt (str): The prompt to generate text from.
            instructions (str): The instructions to use for the generation.

        Returns:
            str: The generated text.
        """
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt
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

        response = self._create_chat_completion(kwargs)
        self._last_usage = self._extract_usage(response)

        # If there are tool calls, return the full response object
        if response.choices[0].message.tool_calls:
            return response

        # Otherwise return just the text content
        return response.choices[0].message.content

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a response using OpenAI's chat completions API.

        Yields dictionaries:
            - {"type": "content", "content": "..."} for text delta chunks
            - {"type": "reasoning", "content": "..."} for reasoning/thinking traces (when exposed by model)
            - {"type": "tool_calls", "response": <reconstructed response>} when tool calls are detected
            - {"type": "done", "content": "<full accumulated text>"} at the end
        """

        def _flatten_reasoning_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts = [_flatten_reasoning_text(item).strip() for item in value]
                return "\n".join([part for part in parts if part])
            if isinstance(value, dict):
                keys = (
                    "text",
                    "content",
                    "summary",
                    "reasoning",
                    "reasoning_content",
                    "value",
                )
                parts = []
                for key in keys:
                    if key in value:
                        part = _flatten_reasoning_text(value.get(key)).strip()
                        if part:
                            parts.append(part)
                if parts:
                    return "\n".join(parts)
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)

            text_attr = getattr(value, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                return text_attr

            content_attr = getattr(value, "content", None)
            if content_attr is not None:
                flattened = _flatten_reasoning_text(content_attr).strip()
                if flattened:
                    return flattened

            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump(exclude_none=True)
                    flattened = _flatten_reasoning_text(dumped).strip()
                    if flattened:
                        return flattened
                except Exception:
                    pass

            return str(value)

        def _extract_reasoning_delta(delta: Any) -> str:
            candidates: List[Any] = []
            for attr in (
                "reasoning_content",
                "reasoning",
                "thinking",
                "reasoning_summary",
            ):
                value = getattr(delta, attr, None)
                if value:
                    candidates.append(value)

            model_dump = getattr(delta, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump(exclude_none=True)
                except Exception:
                    dumped = {}
                if isinstance(dumped, dict):
                    for key in (
                        "reasoning_content",
                        "reasoning",
                        "thinking",
                        "reasoning_summary",
                    ):
                        value = dumped.get(key)
                        if value:
                            candidates.append(value)

            for value in candidates:
                text = _flatten_reasoning_text(value).strip()
                if text:
                    return text
            return ""

        provider_messages = (
            developer_messages_to_system(messages) if self.base_url else messages
        )
        kwargs = {
            "model": self.model,
            "messages": provider_messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self._request_options:
            kwargs.update(self._request_options)
        self._apply_cache_options(kwargs)
        # Ask the official endpoint to append a final usage chunk so
        # streaming turns report prompt/cached token counts like
        # non-streaming ones. Skipped for local servers, some of which
        # reject stream_options.
        if not self.base_url:
            kwargs.setdefault("stream_options", {"include_usage": True})

        stream = self._create_chat_completion(kwargs)

        accumulated_content = ""
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}

        for chunk in stream:
            # The final usage chunk has an empty choices list — capture it
            # before the skip below.
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                extracted = self._extract_usage(chunk)
                if extracted:
                    self._last_usage = extracted
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Emit reasoning/thinking traces when provider exposes them.
            reasoning_text = _extract_reasoning_delta(delta)
            if reasoning_text:
                yield {"type": "reasoning", "content": reasoning_text}

            # Accumulate text content
            if delta.content:
                accumulated_content += delta.content
                yield {"type": "content", "content": delta.content}

            # Accumulate tool calls across chunks
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_acc[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            entry["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["function"][
                                "arguments"
                            ] += tc_delta.function.arguments

            # Stream finish: keep consuming — the final usage chunk (empty
            # choices) arrives after the finish_reason chunk.
            if chunk.choices[0].finish_reason:
                continue

        # If we accumulated tool calls, yield them as a reconstructed response-like object
        if tool_calls_acc:
            # Build a lightweight object that matches the structure _execute_llm_interaction expects
            from types import SimpleNamespace

            tool_calls_list = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                tool_calls_list.append(
                    SimpleNamespace(
                        id=tc["id"],
                        type="function",
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                )

            message_ns = SimpleNamespace(
                content=accumulated_content or None,
                tool_calls=tool_calls_list,
            )
            response_ns = SimpleNamespace(choices=[SimpleNamespace(message=message_ns)])
            yield {"type": "tool_calls", "response": response_ns}
        else:
            yield {"type": "done", "content": accumulated_content}

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
            return extracted
        except Exception:
            return None

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens
