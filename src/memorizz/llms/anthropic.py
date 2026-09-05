# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import copy
import inspect
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

_MAX_CACHE_CONTROL_BREAKPOINTS = 4
_CACHEABLE_CONTENT_BLOCK_TYPES = frozenset(
    {
        "bash_code_execution_tool_result",
        "code_execution_tool_result",
        "container_upload",
        "document",
        "image",
        "search_result",
        "server_tool_use",
        "text",
        "text_editor_code_execution_tool_result",
        "tool_reference",
        "tool_result",
        "tool_search_tool_result",
        "tool_use",
        "web_fetch_tool_result",
        "web_search_tool_result",
    }
)


class Anthropic(LLMProvider):
    """LLM provider for Anthropic's Claude models.

    Uses the ``anthropic`` Python SDK to call the Messages API.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to the ``ANTHROPIC_API_KEY`` env var.
    model : str
        Model identifier (e.g. ``"claude-sonnet-4-5-20250929"``).
    context_window_tokens : int, optional
        Override the inferred context-window size.
    temperature : float, optional
        Sampling temperature (0.0–1.0).
    max_tokens : int
        Maximum tokens to generate per response (default 4096).
    top_p : float, optional
        Nucleus-sampling probability mass.
    top_k : int, optional
        Top-K sampling.
    enable_prompt_caching : bool
        Attach ``cache_control`` breakpoints (system prompt + latest message)
        so repeated prefixes — every tool-loop iteration and every follow-up
        turn — are billed at Anthropic's cached-input rate instead of full
        price. Enabled by default; requires no server-side setup.
    additional_config : dict, optional
        Extra keyword arguments forwarded to ``messages.create``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        context_window_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        enable_prompt_caching: bool = True,
        additional_config: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "The anthropic package is required for the Anthropic provider. "
                'Install it with: pip install "memorizz[anthropic]"'
            )

        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens(model)
        )
        self._last_usage: Optional[Dict[str, int]] = None
        self._max_tokens = max_tokens
        self._enable_prompt_caching = enable_prompt_caching

        # Build optional request parameters
        self._request_options: Dict[str, Any] = {}
        opts = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
        for key, value in opts.items():
            if value is not None:
                self._request_options[key] = value

        if additional_config:
            allowed = {"temperature", "top_p", "top_k", "stop_sequences", "metadata"}
            for key, value in additional_config.items():
                if key in allowed and value is not None:
                    self._request_options[key] = value
            if "enable_prompt_caching" in additional_config:
                self._enable_prompt_caching = bool(
                    additional_config["enable_prompt_caching"]
                )

    # ------------------------------------------------------------------
    # Context-window inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_context_window_tokens(model: str) -> int:
        normalized = (model or "").lower()
        # All Claude 3+ models support 200K context
        context_map = {
            "claude-opus-4-6": 200_000,
            "claude-sonnet-4-6": 200_000,
            "claude-sonnet-4-5-20250929": 200_000,
            "claude-sonnet-4-5": 200_000,
            "claude-haiku-4-5-20251001": 200_000,
            "claude-haiku-4-5": 200_000,
            "claude-opus-4-5": 200_000,
            "claude-opus-4-1": 200_000,
            "claude-sonnet-4-0": 200_000,
            "claude-opus-4-0": 200_000,
            "claude-3-5-sonnet-20241022": 200_000,
            "claude-3-5-haiku-20241022": 200_000,
            "claude-3-opus-20240229": 200_000,
            "claude-3-sonnet-20240229": 200_000,
            "claude-3-haiku-20240307": 200_000,
        }
        return context_map.get(normalized, 200_000)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "anthropic", "model": self.model}

    # ------------------------------------------------------------------
    # Tool metadata helpers (delegate to a simple LLM call)
    # ------------------------------------------------------------------

    def get_tool_metadata(self, func: Callable) -> Dict[str, Any]:
        """Generate tool metadata by introspecting the function signature."""
        sig = inspect.signature(func)
        docstring = func.__doc__ or ""
        func_name = func.__name__

        prompt = (
            f"Generate enriched metadata for the function `{func_name}`.\n\n"
            f"- Docstring: {docstring}\n"
            f"- Signature: {sig}\n\n"
            "Produce a JSON object with keys: name (string, must be '{func_name}'), "
            "description (string), parameters (object with properties, each having "
            "type and description), and required (list of required param names).\n"
            "Return ONLY the JSON."
        )
        raw = self.generate_text(prompt, instructions="Return valid JSON only.")
        return self._safe_json_parse(raw)

    def augment_docstring(self, docstring: str) -> str:
        return self.generate_text(
            f"Augment the docstring by adding more details and examples:\n\n{docstring}",
        )

    def generate_queries(self, docstring: str) -> List[str]:
        raw = self.generate_text(
            f"Generate example user queries for a tool with this docstring:\n\n{docstring}\n\n"
            "Return a JSON array of strings."
        )
        parsed = self._safe_json_parse(raw)
        if isinstance(parsed, list):
            return parsed
        return [raw]

    # ------------------------------------------------------------------
    # Simple text generation
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if instructions:
            kwargs["system"] = instructions
        kwargs.update(self._request_options)

        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.messages.create(**kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=kwargs.get("max_tokens"),
        )
        return self._text_from_response(response)

    # ------------------------------------------------------------------
    # Core generate (chat-completions style)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """Generate a response, optionally with tool calling.

        Returns either a plain string (no tool calls) or a response-like
        ``SimpleNamespace`` object that matches the shape MemAgent expects
        (``response.choices[0].message.tool_calls``).
        """
        kwargs = self._build_request_kwargs(messages, tools, tool_choice)

        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.messages.create(**kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=kwargs.get("max_tokens"),
        )

        # Check for tool calls
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if tool_use_blocks:
            return self._wrap_tool_response(response)

        return self._text_from_response(response)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream a response from the Anthropic Messages API.

        Yields the same dict shapes as the OpenAI provider:
        - ``{"type": "content", "content": "..."}``
        - ``{"type": "reasoning", "content": "..."}``
        - ``{"type": "tool_calls", "response": <SimpleNamespace>}``
        - ``{"type": "done", "content": "<accumulated text>"}``
        """
        kwargs = self._build_request_kwargs(messages, tools, tool_choice)

        self._last_usage = None
        self._last_response_metadata = {}
        accumulated_content = ""
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        current_block_index = -1
        current_block_type = None
        # Usage snapshot for THIS stream only — keeps a missing message_start
        # from merging fresh output tokens into a previous call's usage.
        stream_usage: Optional[Dict[str, Any]] = None

        from contextlib import ExitStack

        from ..streaming import check_cancelled
        from .response_metadata import response_metadata
        from .streaming import ProviderStreamError, closing_provider_stream

        stopped = False
        with ExitStack() as stack:
            stream = stack.enter_context(self.client.messages.stream(**kwargs))
            stack.enter_context(closing_provider_stream(stream))
            for event in stream:
                check_cancelled()
                event_type = getattr(event, "type", None)

                if event_type == "message_stop":
                    stopped = True
                elif event_type == "error":
                    raise ProviderStreamError("provider_stream_error")
                elif event_type == "message_start":
                    # message_start carries the authoritative input-side usage
                    # (including cache_read/cache_creation); output tokens
                    # arrive later via message_delta.
                    message_obj = getattr(event, "message", None)
                    self._last_response_metadata.update(
                        response_metadata(
                            message_obj, max_output_tokens=kwargs.get("max_tokens")
                        )
                    )
                    usage = getattr(message_obj, "usage", None)
                    if usage is not None:
                        extracted = self._usage_to_dict(usage)
                        if extracted:
                            stream_usage = extracted
                            self._last_usage = extracted

                elif event_type == "content_block_start":
                    current_block_index = getattr(
                        event, "index", current_block_index + 1
                    )
                    block = getattr(event, "content_block", None)
                    if block:
                        current_block_type = getattr(block, "type", None)
                        if current_block_type == "tool_use":
                            tool_calls_acc[current_block_index] = {
                                "id": getattr(block, "id", ""),
                                "type": "function",
                                "function": {
                                    "name": getattr(block, "name", ""),
                                    "arguments": "",
                                },
                            }

                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", "")

                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            accumulated_content += text
                            yield {"type": "content", "content": text}

                    elif delta_type == "thinking_delta":
                        thinking = getattr(delta, "thinking", "")
                        if thinking:
                            yield {"type": "reasoning", "content": thinking}

                    elif delta_type == "input_json_delta":
                        partial_json = getattr(delta, "partial_json", "")
                        if current_block_index in tool_calls_acc and partial_json:
                            tool_calls_acc[current_block_index]["function"][
                                "arguments"
                            ] += partial_json

                elif event_type == "message_delta":
                    self._last_response_metadata.update(
                        response_metadata(getattr(event, "delta", None))
                    )
                    usage = getattr(event, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", None)
                        # Merge output tokens into the usage snapshot captured
                        # at message_start instead of clobbering it.
                        merged = dict(stream_usage or {})
                        merged["completion_tokens"] = output_tokens
                        prompt_tokens = merged.get("prompt_tokens")
                        if prompt_tokens is not None and output_tokens is not None:
                            merged["total_tokens"] = prompt_tokens + output_tokens
                        else:
                            merged.setdefault("prompt_tokens", None)
                            merged.setdefault("total_tokens", None)
                        self._last_usage = merged

        if not stopped:
            raise ProviderStreamError("provider_stream_incomplete")
        if self._last_response_metadata.get("finish_reason") in {
            "max_tokens",
            "refusal",
            "pause_turn",
        }:
            raise ProviderStreamError(
                "provider_" + self._last_response_metadata["finish_reason"]
            )
        if self._last_usage:
            yield {"type": "usage", "usage": self._last_usage}

        self._last_response_metadata.update(
            response_metadata(None, text=accumulated_content)
        )
        # Yield final result
        if tool_calls_acc:
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

    # ------------------------------------------------------------------
    # Prompt caching
    # ------------------------------------------------------------------

    def _build_request_kwargs(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Dict[str, Any]:
        """Build one provider-owned Anthropic request.

        Conversion deep-copies all caller-owned message/tool data before
        prompt-cache metadata is attached. Keeping request assembly here also
        guarantees that streaming and non-streaming calls apply the same
        cache-breakpoint policy.
        """
        system_content, api_messages = self._split_system(messages)
        api_messages = self._convert_messages(api_messages)

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "messages": api_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)
        kwargs.update(copy.deepcopy(self._request_options))
        self._apply_cache_control(kwargs)
        self._validate_cache_control_budget(kwargs)
        return kwargs

    def _apply_cache_control(self, kwargs: Dict[str, Any]) -> None:
        """Attach ephemeral ``cache_control`` breakpoints to the request.

        Anthropic prompt caching is opt-in and prefix-based (rendered order:
        tools → system → messages). Up to three MemoRizz-managed breakpoints
        cover the agent loop within Anthropic's four-breakpoint request budget:

        1. On the system prompt — caches the tool schemas *and* the system
           text together, so they're read (not re-billed) on every tool-loop
           iteration and every follow-up turn.
        2. On the last content block of the FINAL message — the read point
           for the next tool-loop iteration within the same turn (the final
           user message, volatile block included, is byte-identical across
           iterations of one turn).
        3. On the last content block of the SECOND-TO-LAST message — the
           read point for the NEXT turn. The final message carries the
           per-turn volatile block, which history re-renders as the raw
           query on later turns, so a breakpoint there can never match
           across turns; the second-to-last message is stable history and
           accrues incremental cross-turn cache hits (Anthropic looks back
           up to 20 blocks to find the previous entry).

        Reads bill at ~0.1x input price, writes at 1.25x — with the tool loop
        re-sending the whole prefix every iteration this pays for itself on
        the very next request. No-ops when ``enable_prompt_caching`` is off.
        """
        if not self._enable_prompt_caching:
            return

        existing = self._count_cache_control_breakpoints(kwargs)
        if existing > _MAX_CACHE_CONTROL_BREAKPOINTS:
            self._raise_cache_control_overflow(existing)
        remaining = _MAX_CACHE_CONTROL_BREAKPOINTS - existing

        def mark_block(block: Dict[str, Any]) -> None:
            nonlocal remaining
            if block.get("cache_control") is not None or remaining <= 0:
                return
            block["cache_control"] = {"type": "ephemeral"}
            remaining -= 1

        system_content = kwargs.get("system")
        if isinstance(system_content, str) and system_content.strip() and remaining > 0:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            remaining -= 1
        elif isinstance(system_content, list):
            system_block = self._last_cacheable_content_block(system_content)
            if system_block is not None:
                mark_block(system_block)

        messages = kwargs.get("messages") or []
        selected = 0
        for msg in reversed(messages):
            if selected >= 2:
                break
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if not content.strip():
                    continue
                selected += 1
                if remaining > 0:
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                    remaining -= 1
            elif isinstance(content, list) and content:
                last_block = self._last_cacheable_content_block(content)
                if last_block is None:
                    continue
                selected += 1
                mark_block(last_block)

        self._validate_cache_control_budget(kwargs)

    @staticmethod
    def _last_cacheable_content_block(
        content: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Return the last block that Anthropic permits as a breakpoint."""
        for block in reversed(content):
            if (
                isinstance(block, dict)
                and block.get("type") in _CACHEABLE_CONTENT_BLOCK_TYPES
            ):
                return block
        return None

    @staticmethod
    def _count_cache_control_breakpoints(kwargs: Dict[str, Any]) -> int:
        """Count only API-level cache breakpoints in a request.

        This deliberately does not recurse into tool schemas, ``tool_use``
        inputs, or document sources: a user payload may legitimately contain
        a key named ``cache_control`` without it being an Anthropic cache
        breakpoint.
        """

        def has_marker(value: Any) -> int:
            return int(
                isinstance(value, dict) and value.get("cache_control") is not None
            )

        def count_content_markers(content: Any) -> int:
            if not isinstance(content, (list, tuple)):
                return 0
            count = 0
            for block in content:
                if not isinstance(block, dict):
                    continue
                count += has_marker(block)
                # Tool results can contain their own typed content blocks.
                # Follow only the content-block hierarchy, never arbitrary
                # inputs, schemas, sources, or other user payloads.
                count += count_content_markers(block.get("content"))
            return count

        count = int(kwargs.get("cache_control") is not None)
        count += sum(has_marker(tool) for tool in kwargs.get("tools") or [])

        system_content = kwargs.get("system")
        count += count_content_markers(system_content)

        for message in kwargs.get("messages") or []:
            if not isinstance(message, dict):
                continue
            count += count_content_markers(message.get("content"))

        return count

    @classmethod
    def _validate_cache_control_budget(cls, kwargs: Dict[str, Any]) -> None:
        count = cls._count_cache_control_breakpoints(kwargs)
        if count > _MAX_CACHE_CONTROL_BREAKPOINTS:
            cls._raise_cache_control_overflow(count)

    @staticmethod
    def _raise_cache_control_overflow(count: int) -> None:
        raise ValueError(
            "Anthropic accepts at most "
            f"{_MAX_CACHE_CONTROL_BREAKPOINTS} cache_control breakpoints; "
            f"the request contains {count}. Remove caller-supplied breakpoints "
            "until the request contains four or fewer."
        )

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        return self._usage_to_dict(usage)

    @staticmethod
    def _usage_to_dict(usage: Any) -> Optional[Dict[str, int]]:
        """Normalize an Anthropic usage object, including cache metrics.

        Anthropic's ``input_tokens`` EXCLUDES cached tokens, so the true
        prompt size (what context-window accounting needs) is
        ``input + cache_read + cache_creation``. The raw cache fields are
        surfaced alongside so callers can compute hit rates.
        """
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", None) or 0

        prompt_tokens = input_tokens
        if input_tokens is not None:
            prompt_tokens = input_tokens + cache_read + cache_creation

        total = None
        if prompt_tokens is not None and output_tokens is not None:
            total = prompt_tokens + output_tokens

        extracted: Dict[str, int] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total,
        }
        if cache_read:
            extracted["cached_tokens"] = cache_read
            extracted["cache_read_input_tokens"] = cache_read
        if cache_creation:
            extracted["cache_creation_input_tokens"] = cache_creation
        return extracted

    def get_last_response_metadata(self) -> Dict[str, Any]:
        from .response_metadata import last_response_metadata

        return last_response_metadata(self)

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system(messages: List[Dict[str, Any]]):
        """Separate application-authority messages from the conversation.

        Anthropic's API takes ``system`` as a top-level parameter, not as a
        portable ``developer`` message role. This helper maps both MemoRizz
        ``system`` and reviewed ``developer`` instructions into the top-level
        system parameter and returns ``(system_content, remaining_messages)``.
        System content remains a string when all instruction messages are
        text, or becomes an owned content-block list when callers supply
        structured blocks.
        """
        system_parts: List[Any] = []
        remaining: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if content:
                    system_parts.append(copy.deepcopy(content))
            else:
                remaining.append(msg)

        if not system_parts:
            return None, remaining
        if all(isinstance(part, str) for part in system_parts):
            return "\n\n".join(system_parts), remaining

        system_blocks: List[Dict[str, Any]] = []
        for part in system_parts:
            if isinstance(part, str):
                system_blocks.append({"type": "text", "text": part})
            elif isinstance(part, (list, tuple)):
                system_blocks.extend(part)
        return system_blocks or None, remaining

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tool messages to Anthropic's content-block format.

        memorizz's core loop records tool calls/results in OpenAI's shape::

            {"role": "assistant", "content": ..., "tool_calls": [
                {"id": ..., "function": {"name": ..., "arguments": "<json>"}}]}
            {"role": "tool", "tool_call_id": ..., "content": "..."}

        Anthropic has no ``tool`` role: a tool call is a ``tool_use`` block in the
        assistant message, and its result is a ``tool_result`` block inside the
        *following user* message. Without this, the follow-up request that carries
        tool results back 400s with ``Unexpected role "tool"``.

        Consecutive tool results (parallel tool calls) are merged into one user
        message, as Anthropic requires. Plain messages retain their structure
        but are deep-copied so request metadata cannot mutate caller history.
        """
        converted: List[Dict[str, Any]] = []
        pending_results: List[Dict[str, Any]] = []

        def flush_results():
            if pending_results:
                converted.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for original_msg in messages:
            # Plain messages must not alias MemAgent's reusable conversation
            # history; cache annotations are request-specific metadata.
            msg = copy.deepcopy(original_msg)
            if isinstance(msg.get("content"), tuple):
                msg["content"] = list(msg["content"])
            role = msg.get("role")

            if role == "tool":
                content = msg.get("content")
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": "" if content is None else str(content),
                    }
                )
                continue

            # A non-tool message closes any open run of tool results.
            flush_results()

            if role == "assistant" and msg.get("tool_calls"):
                blocks: List[Dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {}) or {}
                    raw_args = fn.get("arguments", "")
                    if isinstance(raw_args, dict):
                        parsed = raw_args
                    else:
                        try:
                            parsed = json.loads(raw_args) if raw_args else {}
                        except (json.JSONDecodeError, TypeError):
                            parsed = {}
                    if not isinstance(parsed, dict):
                        parsed = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": parsed,
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append(msg)

        flush_results()
        return converted

    @staticmethod
    def _convert_tools(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-format tool definitions to Anthropic format.

        OpenAI format::

            {"type": "function", "function": {"name": ..., "description": ...,
             "parameters": {...}}}

        Anthropic format::

            {"name": ..., "description": ..., "input_schema": {...}}
        """
        converted = []
        for original_tool in tools:
            tool = copy.deepcopy(original_tool)
            func = tool.get("function", tool)
            converted_tool = {
                "name": func.get("name", "unknown"),
                "description": func.get("description", ""),
                "input_schema": func.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            cache_control = func.get("cache_control", tool.get("cache_control"))
            if cache_control is not None:
                converted_tool["cache_control"] = cache_control
            converted.append(converted_tool)
        return converted

    @staticmethod
    def _convert_tool_choice(tool_choice: str) -> Dict[str, Any]:
        """Convert OpenAI-style tool_choice string to Anthropic format."""
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        # Specific tool name
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _text_from_response(response: Any) -> str:
        """Extract text content from an Anthropic Message response."""
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _wrap_tool_response(self, response: Any) -> Any:
        """Wrap an Anthropic response with tool_use blocks into the
        ``SimpleNamespace`` shape that MemAgent expects
        (``response.choices[0].message.tool_calls``).
        """
        tool_calls = []
        text_parts = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    SimpleNamespace(
                        id=block.id,
                        type="function",
                        function=SimpleNamespace(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )
            elif block.type == "text":
                text_parts.append(block.text)

        message_ns = SimpleNamespace(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message_ns)])

    @staticmethod
    def _safe_json_parse(text: str) -> Any:
        """Try to parse JSON from text, handling markdown code fences."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
