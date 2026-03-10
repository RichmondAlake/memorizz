# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import json
import logging
from typing import Any, Dict, List, Optional

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLMProvider):
    """
    Lightweight wrapper around Hugging Face text-generation models.

    Uses the `transformers` pipeline API so any causal LM hosted on the
    Hugging Face Hub (or locally) can serve as the agent's LLM.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer: Optional[str] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        auth_token: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        context_window_tokens: Optional[int] = None,
    ):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the Hugging Face LLM provider. "
                "Install it via `pip install memorizz[huggingface]`."
            ) from exc

        self.model = model
        self.tokenizer_id = tokenizer or model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.auth_token = auth_token
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.pipeline_kwargs = pipeline_kwargs or {}

        pipeline_params: Dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer_id,
            "trust_remote_code": self.trust_remote_code,
            "return_full_text": False,
        }
        if auth_token:
            pipeline_params["token"] = auth_token
        if revision:
            pipeline_params["revision"] = revision

        resolved_device = self._resolve_device(self.device)
        if resolved_device is not None:
            pipeline_params["device"] = resolved_device

        pipeline_params.update(self.pipeline_kwargs)

        self._pipeline = pipeline("text-generation", **pipeline_params)
        self.client = self._pipeline
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens()
        )
        self._last_usage: Optional[Dict[str, int]] = None

    def _resolve_device(self, device: Optional[str]) -> Optional[int]:
        """Convert device configuration to pipeline-friendly format."""
        if device is None:
            return None
        if isinstance(device, int):
            return device
        device_lower = device.lower()
        if device_lower in {"cpu", "mps"}:
            return -1
        if device_lower.startswith("cuda"):
            return 0
        try:
            return int(device_lower)
        except ValueError:
            logger.warning("Unrecognized device '%s'. Defaulting to CPU.", device)
            return -1

    def get_config(self) -> Dict[str, Any]:
        """Return serializable configuration for persistence."""
        return {
            "provider": "huggingface",
            "model": self.model,
            "tokenizer": self.tokenizer_id,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "trust_remote_code": self.trust_remote_code,
            "revision": self.revision,
        }

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat-style messages into a simple text prompt."""
        prompt_lines: List[str] = []
        for message in messages:
            role = message.get("role", "user").lower()
            content = message.get("content", "")
            if role == "system":
                prompt_lines.append(f"[system]\n{content}\n")
            elif role == "assistant":
                prompt_lines.append(f"[assistant]\n{content}\n")
            else:
                prompt_lines.append(f"[user]\n{content}\n")
        prompt_lines.append("[assistant]\n")
        return "\n".join(prompt_lines)

    def _run_generation(self, prompt: str, **overrides) -> str:
        """Execute the underlying pipeline with sensible defaults."""
        generation_kwargs = {
            "max_new_tokens": overrides.get("max_new_tokens", self.max_new_tokens),
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "do_sample": overrides.get(
                "do_sample", overrides.get("temperature", self.temperature) > 0
            ),
        }
        if "stop" in overrides:
            generation_kwargs["stop"] = overrides["stop"]

        outputs = self.client(prompt, **generation_kwargs)
        if not outputs:
            return ""
        text = outputs[0].get("generated_text", "")
        return text.strip()

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Any:
        if tools:
            raise NotImplementedError(
                "Tool calling is not supported by the Hugging Face provider."
            )

        prompt = self._messages_to_prompt(messages)
        output_text = self._run_generation(prompt)
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(output_text)
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return output_text

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
        return self.generate(messages)

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
        from ..long_term_memory.procedural.toolbox.tool_schema import ToolSchemaType

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

    def _safe_json_parse(self, text: str) -> Dict[str, Any]:
        """Attempt to parse JSON even if wrapped with commentary."""
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

    def _infer_context_window_tokens(self) -> Optional[int]:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            try:
                max_length = getattr(tokenizer, "model_max_length", None)
                if max_length and max_length < 10**9:
                    return int(max_length)
            except Exception:
                pass
        return None

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                logger.debug(
                    "Falling back to whitespace token counting for HuggingFace LLM"
                )
        return max(1, len(text.split())) if text else 0

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens
