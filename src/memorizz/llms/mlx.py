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

import logging
from typing import Any, Dict, Generator, List, Optional

from .llm_provider import LLMProvider, ResponseMetadataMixin
from .local_chat import LocalChatMixin, chat_prompt
from .tool_metadata import SchemaPromptToolMetadataMixin

logger = logging.getLogger(__name__)


class MLXLLM(
    SchemaPromptToolMetadataMixin, LocalChatMixin, ResponseMetadataMixin, LLMProvider
):
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
            "context_window_tokens": self.context_window_tokens,
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

        prompt = chat_prompt(self._tokenizer, messages)
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

        prompt = chat_prompt(self._tokenizer, messages)
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
