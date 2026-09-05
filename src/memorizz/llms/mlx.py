# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""MLX-backed LLM provider for Apple Silicon.

Google explicitly recommends MLX for running Gemma 4 on Apple Silicon
(see https://huggingface.co/blog/gemma4) — it's faster than running
``transformers`` on PyTorch's MPS backend and uses less unified memory.

This provider wraps ``mlx_lm.load`` / ``mlx_lm.stream_generate`` with the
same interface as :class:`HuggingFaceLLM` so the rest of the agent
runtime treats it as a drop-in.

Constraints:
- ``mlx`` ships only as native arm64 wheels on macOS. A Python
  environment running under Rosetta (x86_64) will fail to install it.
  Use a native arm64 Python (``conda create -n mem_arm python=3.11``
  on an Apple Silicon machine) to avoid the install error.
- Tool/function calling is not implemented. Like HuggingFaceLLM, this
  provider degrades to text-only generation and logs a warning when
  tools are passed.
"""

import inspect
import json
import logging
from typing import Any, Dict, Generator, List, Optional

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class MLXLLM(LLMProvider):
    """Run Apple-MLX-quantized causal LMs through ``mlx_lm``.

    Pre-quantized weights live under the ``mlx-community`` namespace on
    Hugging Face (e.g. ``mlx-community/gemma-4-E2B-it-4bit``). The
    standard HF cache directory is reused, so the same models that
    appear under "Available offline" via the HF cache scanner work here
    too.
    """

    def __init__(
        self,
        model: str = "mlx-community/gemma-4-E2B-it-4bit",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        adapter_path: Optional[str] = None,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        context_window_tokens: Optional[int] = None,
    ):
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for the MLX LLM provider. Install it via "
                "`pip install memorizz[mlx]` on a native arm64 Apple Silicon "
                "Python (Rosetta x86_64 envs cannot install mlx)."
            ) from exc

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.adapter_path = adapter_path
        self.tokenizer_config = tokenizer_config or {}

        load_kwargs: Dict[str, Any] = {}
        if adapter_path:
            load_kwargs["adapter_path"] = adapter_path
        if tokenizer_config:
            load_kwargs["tokenizer_config"] = tokenizer_config

        self._model, self._tokenizer = load(model, **load_kwargs)
        self.client = self._model
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens()
        )
        self._last_usage: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Required protocol methods
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            "provider": "mlx",
            "model": self.model,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "adapter_path": self.adapter_path,
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        if tools:
            logger.warning(
                "MLXLLM does not support tool calling; ignoring %d tool(s) "
                "and generating text only.",
                len(tools),
            )

        from mlx_lm import generate

        prompt = self._build_prompt(messages)
        sampler = self._build_sampler()
        self._last_usage = None
        self._last_response_metadata = {}
        output = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            sampler=sampler,
        )
        text = output.strip() if isinstance(output, str) else str(output).strip()
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(text)
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            None, text=text, max_output_tokens=self.max_new_tokens
        )
        return text

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream tokens with the same event shape as the OpenAI provider."""
        if tools:
            logger.warning(
                "MLXLLM does not support tool calling; ignoring %d tool(s) "
                "and streaming text only.",
                len(tools),
            )

        from mlx_lm import stream_generate

        prompt = self._build_prompt(messages)
        sampler = self._build_sampler()

        self._last_usage = None
        self._last_response_metadata = {}
        accumulated: List[str] = []
        last_response: Any = None
        from ..streaming import check_cancelled
        from .streaming import closing_provider_stream

        with closing_provider_stream(
            stream_generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                sampler=sampler,
            )
        ) as stream:
            for response in stream:
                check_cancelled()
                last_response = response
                chunk = getattr(response, "text", None)
                if chunk is None and isinstance(response, dict):
                    chunk = response.get("text")
                if chunk:
                    accumulated.append(chunk)
                    yield {"type": "content", "content": chunk}

        full = "".join(accumulated)

        # mlx-lm's GenerationResponse exposes prompt_tokens / generation_tokens
        prompt_tokens = (
            getattr(last_response, "prompt_tokens", None)
            if last_response is not None
            else None
        ) or self._count_tokens(prompt)
        completion_tokens = (
            getattr(last_response, "generation_tokens", None)
            if last_response is not None
            else None
        ) or self._count_tokens(full)
        self._last_usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
        }
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            last_response, text=full, max_output_tokens=self.max_new_tokens
        )
        yield {"type": "usage", "usage": self._last_usage}
        yield {"type": "done", "content": full}

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
        return self.generate(messages)

    def get_last_response_metadata(self) -> Dict[str, Any]:
        from .response_metadata import last_response_metadata

        return last_response_metadata(self)

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens

    # ------------------------------------------------------------------
    # Tool/metadata helpers — mirror HuggingFaceLLM (text-only fallback)
    # ------------------------------------------------------------------

    def augment_docstring(self, docstring: str) -> str:
        instructions = (
            "You improve terse docstrings. Expand with helpful detail and examples."
        )
        return self.generate_text(docstring, instructions=instructions)

    def generate_queries(self, docstring: str) -> List[str]:
        prompt = (
            "Generate three short example queries or tasks that would use the "
            "following tool:\n\n"
            f"{docstring}"
        )
        raw_output = self.generate_text(prompt)
        lines = [line.strip(" -•") for line in raw_output.splitlines() if line.strip()]
        return [line for line in lines if line]

    def get_tool_metadata(self, func: Any) -> Dict[str, Any]:
        from ..long_term.procedural.toolbox.tool_schema import ToolSchemaType

        docstring = func.__doc__ or ""
        signature = str(inspect.signature(func))
        func_name = func.__name__

        prompt = (
            "You produce JSON metadata for Python functions.\n"
            "The JSON must strictly follow this schema:\n"
            "{"
            '"type": "function", '
            '"function": {'
            '"name": str, '
            '"description": str, '
            '"parameters": [{"name": str, "description": str, "type": str, "required": bool}], '
            '"required": [str], '
            '"queries": [str]'
            "}"
            "}\n\n"
            f"Function name: {func_name}\n"
            f"Signature: {signature}\n"
            f"Docstring: {docstring}\n"
            "Return a JSON object only."
        )

        raw_output = self.generate_text(prompt)
        metadata_dict = self._safe_json_parse(raw_output)
        tool_schema = ToolSchemaType.model_validate(metadata_dict)
        return tool_schema.model_dump()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages with the model's chat template when available.

        Same logic as HuggingFaceLLM: prefer the tokenizer's Jinja
        ``chat_template`` so the model sees the special tokens it was
        trained on (and stops cleanly at the right EOS), and fold any
        leading ``system`` message into the first user turn for
        templates that reject the system role (Gemma).
        """
        tokenizer = self._tokenizer
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            normalized = self._normalize_messages(messages)
            try:
                return tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc:
                logger.debug(
                    "apply_chat_template failed (%s); falling back to bracket prompt.",
                    exc,
                )

        prompt_lines: List[str] = []
        for message in messages:
            role = (message.get("role") or "user").lower()
            content = message.get("content", "")
            if role in ("system", "developer"):
                prompt_lines.append(f"[system]\n{content}\n")
            elif role == "assistant":
                prompt_lines.append(f"[assistant]\n{content}\n")
            else:
                prompt_lines.append(f"[user]\n{content}\n")
        prompt_lines.append("[assistant]\n")
        return "\n".join(prompt_lines)

    @staticmethod
    def _normalize_messages(
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        cleaned: List[Dict[str, str]] = []
        pending_system: Optional[str] = None
        for message in messages:
            role = (message.get("role") or "user").lower()
            content = message.get("content") or ""
            if not content:
                continue
            if role in ("system", "developer"):
                pending_system = (
                    f"{pending_system}\n\n{content}" if pending_system else content
                )
                continue
            if role not in ("user", "assistant", "tool"):
                role = "user"
            if pending_system and role == "user":
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            cleaned.append({"role": role, "content": content})
        if pending_system and not cleaned:
            cleaned.append({"role": "user", "content": pending_system})
        return cleaned

    def _build_sampler(self):
        """Return an mlx-lm sampler matching configured temperature/top_p."""
        try:
            from mlx_lm.sample_utils import make_sampler
        except ImportError:
            return None
        return make_sampler(temp=self.temperature, top_p=self.top_p)

    def _infer_context_window_tokens(self) -> Optional[int]:
        tokenizer = self._tokenizer
        if tokenizer is not None:
            max_length = getattr(tokenizer, "model_max_length", None)
            if max_length and max_length < 10**9:
                return int(max_length)
        return None

    def _count_tokens(self, text: str) -> int:
        tokenizer = self._tokenizer
        if tokenizer is not None and text:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                logger.debug("Falling back to whitespace token counting for MLXLLM")
        return max(1, len(text.split())) if text else 0

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        snippet = text.strip()
        start = snippet.find("{")
        end = snippet.rfind("}")
        if start != -1 and end != -1:
            snippet = snippet[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse JSON output: %s", snippet)
            raise ValueError("LLM response was not valid JSON") from exc
