# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import json
import logging
import os
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, Tuple

from .llm_provider import LLMProvider, ResponseMetadataMixin
from .message_roles import developer_messages_to_system
from .tool_metadata import JsonPromptToolMetadataMixin

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_WINDOW_TOKENS = 8192
# With nothing configured, a local model gets its full context length when the
# attention cache for it fits in AUTO_KV_MEMORY_SHARE of this machine's RAM,
# otherwise the largest standard size that fits, and never less than
# AUTO_CONTEXT_WINDOW_TOKENS (or the model's length, if shorter). A MemAgent's
# instructions and tool schemas alone take about 6-7k tokens.
AUTO_CONTEXT_WINDOW_TOKENS = 16384
AUTO_KV_MEMORY_SHARE = 0.25
_STANDARD_WINDOWS = (262144, 131072, 65536, 32768)
_MODEL_PROFILES: Dict[Tuple[str, str], Dict[str, Any]] = {}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _kv_cache_bytes(info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Estimated f16 attention-cache bytes: (per context token, fixed).

    Read from the daemon's model metadata. Sliding-window layers cost a fixed
    amount, layers without attention (state-space) and layers that reuse an
    earlier layer's cache cost nothing. Unknown layouts return ``None``.
    """
    arch = info.get("general.architecture")
    if not arch:
        arch = next(
            (k[: -len(".block_count")] for k in info if k.endswith(".block_count")),
            None,
        )

    def get(name: str) -> Any:
        return info.get(f"{arch}.{name}")

    layers, heads = get("block_count"), get("attention.head_count")
    kv_heads = get("attention.head_count_kv")
    if type(layers) is not int or layers <= 0:
        return None
    if kv_heads is None:
        kv_heads = heads
    key = get("attention.key_length")
    if key is None and type(heads) is int and heads and get("embedding_length"):
        key = int(get("embedding_length")) // heads
    value = get("attention.value_length") or key
    if not isinstance(key, int) or not isinstance(value, int):
        return None
    key_swa = get("attention.key_length_swa") or key
    value_swa = get("attention.value_length_swa") or value
    window = get("attention.sliding_window")
    pattern = get("attention.sliding_window_pattern")
    shared = get("attention.shared_kv_layers") or 0
    per_token = fixed = 0
    for layer in range(max(layers - int(shared), 0)):
        count = kv_heads[layer] if isinstance(kv_heads, list) else kv_heads
        if not isinstance(count, int) or count <= 0:
            continue
        sliding = (
            isinstance(window, int)
            and isinstance(pattern, list)
            and layer < len(pattern)
            and bool(pattern[layer])
        )
        if sliding:
            fixed += count * (key_swa + value_swa) * 2 * window
        else:
            per_token += count * (key + value) * 2
    return (per_token, fixed) if per_token else None


def _model_profile(client: Any, host: str, model: str) -> Dict[str, Any]:
    """Context length and cache cost the daemon reports for ``model``."""
    key = (str(host), str(model))
    if key not in _MODEL_PROFILES:
        profile: Dict[str, Any] = {"context_length": None, "kv": None}
        try:
            info = client.show(model)
            model_info = getattr(info, "modelinfo", None)
            if model_info is None and isinstance(info, dict):
                model_info = info.get("model_info") or info.get("modelinfo")
            model_info = dict(model_info or {})
            for name, value in model_info.items():
                if str(name).endswith(".context_length") and type(value) is int:
                    profile["context_length"] = value
                    break
            profile["kv"] = _kv_cache_bytes(model_info)
        except Exception as exc:
            logger.debug("Could not read the context length of %s: %s", model, exc)
        _MODEL_PROFILES[key] = profile
    return _MODEL_PROFILES[key]


def _model_context_length(client: Any, host: str, model: str) -> Optional[int]:
    """The context length the daemon reports for ``model`` (cached per host)."""
    return _model_profile(client, host, model)["context_length"]


def _local_memory_bytes(host: str) -> Optional[int]:
    """This machine's RAM, when the daemon runs on it."""
    from urllib.parse import urlparse

    text = str(host or "")
    name = urlparse(text if "://" in text else f"http://{text}").hostname or ""
    if name not in _LOCAL_HOSTS and not name.endswith(".localhost"):
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def _auto_context_window(client: Any, host: str, model: str) -> Optional[int]:
    """The window used when none is configured (see AUTO_KV_MEMORY_SHARE)."""
    profile = _model_profile(client, host, model)
    length = profile["context_length"]
    if not length:
        return None
    floor = min(length, AUTO_CONTEXT_WINDOW_TOKENS)
    memory = _local_memory_bytes(host)
    if not memory or not profile["kv"]:
        return floor
    per_token, fixed = profile["kv"]
    budget = memory * AUTO_KV_MEMORY_SHARE - fixed
    for size in (length, *(s for s in _STANDARD_WINDOWS if s < length)):
        if size <= floor:
            break
        if size * per_token <= budget:
            return size
    return floor


# Only daemon options belong in persisted configuration, never credentials or
# arbitrary client kwargs that a caller put in additional_config.
_PERSISTED_OPTIONS = frozenset(
    "num_keep seed num_predict top_k top_p min_p typical_p repeat_last_n "
    "temperature repeat_penalty presence_penalty frequency_penalty mirostat "
    "mirostat_tau mirostat_eta penalize_newline stop num_ctx num_batch num_gpu "
    "main_gpu use_mmap num_thread".split()
)


class OllamaLLM(JsonPromptToolMetadataMixin, ResponseMetadataMixin, LLMProvider):
    """LLM provider for locally-hosted models via Ollama.

    Uses the ``ollama`` Python SDK to communicate with a running Ollama
    server (default ``http://localhost:11434``).

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``"llama3.1"``, ``"mistral"``, ``"gemma4:e4b"``).
        Gemma 4 tags require the Ollama daemon at v0.22.1+ (April 28 2026).
    host : str, optional
        Ollama server URL. Falls back to ``OLLAMA_HOST`` env var, then
        ``http://localhost:11434``.
    temperature : float, optional
        Sampling temperature.
    num_predict : int, optional
        Maximum tokens to generate (default 4096).
    top_p : float, optional
        Nucleus-sampling probability mass.
    top_k : int, optional
        Top-K sampling.
    seed : int, optional
        Random seed for reproducibility.
    context_window_tokens : int, optional
        Context window sent to Ollama as ``options.num_ctx``. Defaults to
        ``additional_config['num_ctx']``, then ``OLLAMA_CONTEXT_LENGTH``, then
        the model's own context length when its attention cache fits in a
        quarter of this machine's RAM (otherwise the largest standard size that
        fits, at least 16,384), then 8,192. The same value is used for
        MemAgent's history budget.
    timeout : float, optional
        Request timeout in seconds.
    think : bool, optional
        Whether to ask the daemon to emit a separate ``thinking`` chain-of-thought
        trace before the final ``content``. Defaults to ``False`` because models
        like Gemma 4 (Ollama 0.22.1+) ship with thinking on by default and can
        spend 30–60s on the trace before any visible reply, which makes the
        playground appear stuck. Pass ``True`` (or ``additional_config={"think":
        True}``) when you actually want the reasoning trace surfaced.
    additional_config : dict, optional
        Extra options forwarded to the ``options`` dict in Ollama requests.
        A top-level ``think`` key is special-cased and overrides the ``think``
        constructor arg.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: int = 4096,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        context_window_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        think: Optional[bool] = None,
        additional_config: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError(
                "The ollama package is required for the Ollama provider. "
                'Install it with: pip install "memorizz[ollama]"'
            )

        if host is None:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        client_kwargs: Dict[str, Any] = {"host": host}
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = _ollama.Client(**client_kwargs)
        self.model = model
        self._host = host
        self._timeout = timeout
        context_window = context_window_tokens
        if context_window is None:
            context_window = (additional_config or {}).get("num_ctx")
        # Only a configured window is saved with the agent, so a default
        # never turns into a pinned setting.
        self._configured_context_window = context_window
        if context_window is None:
            context_window = os.getenv("OLLAMA_CONTEXT_LENGTH")
        if context_window is None:
            context_window = (
                _auto_context_window(self.client, host, model)
                or DEFAULT_CONTEXT_WINDOW_TOKENS
            )
        try:
            if isinstance(context_window, bool) or not isinstance(
                context_window, (int, str)
            ):
                raise ValueError
            self.context_window_tokens = int(context_window)
            if self.context_window_tokens <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(
                "Ollama context_window_tokens must be a positive integer"
            ) from None
        self._last_usage: Optional[Dict[str, int]] = None

        # Build options dict for Ollama requests
        self._options: Dict[str, Any] = {}
        opts = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "num_predict": num_predict,
        }
        for key, value in opts.items():
            if value is not None:
                self._options[key] = value

        # `think` is a top-level chat parameter, not an option. Pull it out of
        # additional_config (if present) and let it override the constructor
        # arg, so existing UI configs that surface it via additional_config
        # keep working.
        self.think: Optional[bool] = think
        if additional_config:
            cfg = dict(additional_config)
            if "think" in cfg:
                think_override = cfg.pop("think")
                if think_override is not None:
                    self.think = bool(think_override)
            for key, value in cfg.items():
                if value is not None:
                    self._options[key] = value

        # Reporting a large model limit without setting num_ctx lets the
        # daemon silently truncate prompts using its much smaller default.
        # An explicit context_window_tokens wins over additional_config.
        self._options["num_ctx"] = self.context_window_tokens

        # Reasoning models (qwen3 / deepseek-r1 / qwq / magistral / *thinking*)
        # must run with `think` enabled — otherwise Ollama returns truncated or
        # empty content and no separable reasoning trace. Auto-enable when the
        # caller didn't specify, so reasoning + the full answer both surface.
        if self.think is None:
            _name = (self.model or "").lower()
            self.think = any(
                hint in _name
                for hint in ("qwen3", "deepseek-r1", "qwq", "magistral", "thinking")
            )

    def _chat_kwargs(self, **extra: Any) -> Dict[str, Any]:
        """Build the kwarg dict for ``ollama.Client.chat`` calls.

        Centralized so options / think / model fields stay consistent between
        ``generate_text``, ``generate``, and ``generate_stream``.
        """
        kwargs: Dict[str, Any] = {"model": self.model, **extra}
        if "messages" in kwargs:
            kwargs["messages"] = self._normalize_messages(kwargs["messages"])
        if self._options:
            kwargs["options"] = self._options
        if self.think is not None:
            kwargs["think"] = self.think
        return kwargs

    @staticmethod
    def _normalize_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Make MemAgent's OpenAI-shaped messages valid for the Ollama SDK.

        MemAgent stores assistant tool calls with ``function.arguments`` as a
        JSON *string* (OpenAI's wire shape). The ``ollama`` Python SDK's
        ``Message`` model validates ``arguments`` as a dict and rejects the
        string with a Pydantic error:

            tool_calls.0.function.arguments
              Input should be a valid dictionary [type=dict_type, ...]

        We walk the messages once and parse any string-shaped arguments to
        dict before they hit the SDK. We don't mutate the caller's list —
        the upstream MemAgent loop still expects the OpenAI shape next time
        around.
        """
        out: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                out.append(msg)
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                out.append(msg)
                continue
            new_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    new_calls.append(tc)
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    new_calls.append(tc)
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        # Ollama can't use unparseable args; an empty dict at
                        # least lets the conversation continue instead of
                        # crashing the whole stream.
                        logger.warning(
                            "Tool call arguments not valid JSON; sending "
                            "empty dict to Ollama. raw=%r",
                            args,
                        )
                        parsed = {}
                    new_fn = {**fn, "arguments": parsed}
                    new_calls.append({**tc, "function": new_fn})
                else:
                    new_calls.append(tc)
            out.append({**msg, "tool_calls": new_calls})
        return developer_messages_to_system(out)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "provider": "ollama",
            "model": self.model,
            "host": self._host,
            "additional_config": deepcopy(
                {
                    k: v
                    for k, v in self._options.items()
                    if k in _PERSISTED_OPTIONS and k != "num_ctx"
                }
            ),
        }
        if self._configured_context_window is not None:
            cfg["context_window_tokens"] = self.context_window_tokens
        if self._timeout is not None:
            cfg["timeout"] = self._timeout
        if self.think is not None:
            cfg["think"] = self.think
        return cfg

    # ------------------------------------------------------------------
    # Tool metadata helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Simple text generation
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        kwargs = self._chat_kwargs(messages=messages)
        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.chat(**kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=(kwargs.get("options") or {}).get("num_predict"),
        )
        # Thinking models (Gemma 4, deepseek-r1, …) leave ``content`` empty
        # and put the answer-prefix in ``thinking`` when generation truncates
        # mid-trace. Fall back so callers don't get an empty string.
        msg = response.message
        text = (getattr(msg, "content", None) or "").strip()
        if not text:
            text = (getattr(msg, "thinking", None) or "").strip()
        return text

    # ------------------------------------------------------------------
    # Core generate (chat-completions style)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        """Generate a response, optionally with tool calling.

        Returns either a plain string (no tool calls) or a response-like
        ``SimpleNamespace`` object that matches the shape MemAgent expects.
        """
        kwargs = self._chat_kwargs(messages=messages)
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        self._last_usage = None
        self._last_response_metadata = {}
        response = self.client.chat(**kwargs)
        self._last_usage = self._extract_usage(response)
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            response,
            max_output_tokens=(kwargs.get("options") or {}).get("num_predict"),
        )

        # Check for tool calls
        tool_calls = getattr(response.message, "tool_calls", None)
        if tool_calls:
            return self._wrap_tool_response(response)

        return response.message.content or ""

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream a response from the Ollama chat API.

        Yields the same dict shapes as the OpenAI provider.
        """
        kwargs = self._chat_kwargs(messages=messages, stream=True)
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        self._last_usage = None
        self._last_response_metadata = {}
        accumulated_content = ""
        tool_calls_acc: List[Dict[str, Any]] = []

        from ..streaming import check_cancelled
        from .response_metadata import response_metadata
        from .streaming import ProviderStreamError, closing_provider_stream

        finished = False
        with closing_provider_stream(self.client.chat(**kwargs)) as stream:
            for chunk in stream:
                check_cancelled()
                self._last_response_metadata.update(response_metadata(chunk))
                msg = (
                    chunk.message
                    if hasattr(chunk, "message")
                    else chunk.get("message", {})
                )

                # Text content
                text = getattr(msg, "content", None) or (
                    msg.get("content") if isinstance(msg, dict) else None
                )
                if text:
                    accumulated_content += text
                    yield {"type": "content", "content": text}

                # Thinking/reasoning traces
                thinking = getattr(msg, "thinking", None)
                if thinking:
                    yield {"type": "reasoning", "content": thinking}

                # Tool calls
                tc_list = getattr(msg, "tool_calls", None) or (
                    msg.get("tool_calls") if isinstance(msg, dict) else None
                )
                if tc_list:
                    for tc in tc_list:
                        func = (
                            tc.function
                            if hasattr(tc, "function")
                            else tc.get("function", {})
                        )
                        name = getattr(func, "name", None) or (
                            func.get("name") if isinstance(func, dict) else ""
                        )
                        args = getattr(func, "arguments", None) or (
                            func.get("arguments") if isinstance(func, dict) else {}
                        )
                        tool_calls_acc.append({"name": name, "arguments": args})

                # Check for done
                done = getattr(chunk, "done", None) or (
                    chunk.get("done") if isinstance(chunk, dict) else False
                )
                if done:
                    finished = True
                    # Extract final usage from the last chunk
                    self._last_usage = self._extract_usage(chunk)
                    self._last_response_metadata.update(
                        response_metadata(
                            chunk,
                            text=accumulated_content,
                            max_output_tokens=(kwargs.get("options") or {}).get(
                                "num_predict"
                            ),
                        )
                    )

        if not finished:
            raise ProviderStreamError(
                "provider_stream_incomplete", self.get_last_response_metadata()
            )
        if self._last_response_metadata.get("finish_reason") == "length":
            raise ProviderStreamError(
                "provider_length", self.get_last_response_metadata()
            )
        if self._last_usage:
            yield {"type": "usage", "usage": self._last_usage}

        if tool_calls_acc:
            tool_calls_list = []
            for i, tc in enumerate(tool_calls_acc):
                # Ollama returns arguments already parsed as dict
                args = tc["arguments"]
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls_list.append(
                    SimpleNamespace(
                        id=f"ollama_tc_{i}",
                        type="function",
                        function=SimpleNamespace(
                            name=tc["name"],
                            arguments=args_str,
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
    # Usage tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
        prompt_tokens = getattr(response, "prompt_eval_count", None)
        if prompt_tokens is None and isinstance(response, dict):
            prompt_tokens = response.get("prompt_eval_count")

        completion_tokens = getattr(response, "eval_count", None)
        if completion_tokens is None and isinstance(response, dict):
            completion_tokens = response.get("eval_count")

        if prompt_tokens is None and completion_tokens is None:
            return None

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
        }

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tools(
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Ensure tools are in Ollama's expected format.

        Ollama accepts the same OpenAI-style tool format::

            {"type": "function", "function": {"name": ..., "description": ...,
             "parameters": {...}}}

        If tools are already in this format, return them as-is.
        """
        converted = []
        for tool in tools:
            if "type" in tool and "function" in tool:
                # Already in OpenAI/Ollama format
                converted.append(tool)
            elif "name" in tool:
                # Bare function spec → wrap it
                converted.append({"type": "function", "function": tool})
            else:
                converted.append(tool)
        return converted

    def _wrap_tool_response(self, response: Any) -> Any:
        """Wrap an Ollama response with tool calls into the SimpleNamespace shape
        that MemAgent expects."""
        tool_calls = []
        for i, tc in enumerate(response.message.tool_calls):
            func = tc.function if hasattr(tc, "function") else tc.get("function", {})
            name = getattr(func, "name", None) or (
                func.get("name") if isinstance(func, dict) else ""
            )
            args = getattr(func, "arguments", None) or (
                func.get("arguments") if isinstance(func, dict) else {}
            )
            # Ollama returns arguments as a dict; MemAgent expects a JSON string
            args_str = json.dumps(args) if isinstance(args, dict) else str(args)
            tool_calls.append(
                SimpleNamespace(
                    id=f"ollama_tc_{i}",
                    type="function",
                    function=SimpleNamespace(name=name, arguments=args_str),
                )
            )

        content = getattr(response.message, "content", None) or ""
        message_ns = SimpleNamespace(
            content=content if content else None,
            tool_calls=tool_calls if tool_calls else None,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message_ns)])
