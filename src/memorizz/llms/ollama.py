# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Generator, List, Optional

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OllamaLLM(LLMProvider):
    """LLM provider for locally-hosted models via Ollama.

    Uses the ``ollama`` Python SDK to communicate with a running Ollama
    server (default ``http://localhost:11434``).

    Parameters
    ----------
    model : str
        Ollama model name (e.g. ``"llama3.1"``, ``"mistral"``, ``"gemma3"``).
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
        Override context-window size (default 128 000).
    timeout : float, optional
        Request timeout in seconds.
    additional_config : dict, optional
        Extra options forwarded to the ``options`` dict in Ollama requests.
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
        additional_config: Optional[Dict[str, Any]] = None,
        **_ignored: Any,
    ):
        try:
            import ollama as _ollama
        except ImportError:
            raise ImportError(
                "The ollama package is required for the Ollama provider. "
                "Install it with: pip install ollama"
            )

        if host is None:
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        client_kwargs: Dict[str, Any] = {"host": host}
        if timeout is not None:
            client_kwargs["timeout"] = timeout

        self.client = _ollama.Client(**client_kwargs)
        self.model = model
        self._host = host
        self.context_window_tokens = context_window_tokens or 128_000
        self._last_usage: Optional[Dict[str, int]] = None
        self._num_predict = num_predict

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

        if additional_config:
            for key, value in additional_config.items():
                if value is not None:
                    self._options[key] = value

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {"provider": "ollama", "model": self.model, "host": self._host}

    # ------------------------------------------------------------------
    # Tool metadata helpers
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
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self._options:
            kwargs["options"] = self._options

        response = self.client.chat(**kwargs)
        self._last_usage = self._extract_usage(response)
        return response.message.content or ""

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
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        if self._options:
            kwargs["options"] = self._options

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.chat(**kwargs)
        self._last_usage = self._extract_usage(response)

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
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self._options:
            kwargs["options"] = self._options
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        accumulated_content = ""
        tool_calls_acc: List[Dict[str, Any]] = []

        for chunk in self.client.chat(**kwargs):
            msg = (
                chunk.message if hasattr(chunk, "message") else chunk.get("message", {})
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
            tc_list = getattr(msg, "tool_calls", None)
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
                # Extract final usage from the last chunk
                self._last_usage = self._extract_usage(chunk)

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
