# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from ._hf_offline import enable_hf_offline_env, is_hf_offline
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
        local_files_only: Optional[bool] = None,
    ):
        # Detect offline before importing transformers so the env var
        # propagates into huggingface_hub's import-time constants.
        if local_files_only is None:
            local_files_only = is_hf_offline()
        if local_files_only:
            enable_hf_offline_env()

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
        # Fall back to the standard HF env vars when the caller didn't pass
        # an explicit token. Without this, gated repos (Llama, Gemma, etc.)
        # 401 on first download even when the user has set HF_TOKEN in
        # Settings — the saved llm_config doesn't carry secrets.
        self.auth_token = (
            auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.pipeline_kwargs = pipeline_kwargs or {}
        self.local_files_only = bool(local_files_only)

        pipeline_params: Dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer_id,
            "trust_remote_code": self.trust_remote_code,
            "return_full_text": False,
        }
        if self.auth_token:
            pipeline_params["token"] = self.auth_token
        if revision:
            pipeline_params["revision"] = revision

        resolved_device, resolved_dtype = self._resolve_device_and_dtype(self.device)
        if resolved_device is not None:
            pipeline_params["device"] = resolved_device
        if resolved_dtype is not None and "torch_dtype" not in self.pipeline_kwargs:
            pipeline_params["torch_dtype"] = resolved_dtype

        if self.local_files_only:
            # Forwarded to AutoModel.from_pretrained / AutoTokenizer.from_pretrained
            # so neither side issues a HEAD request to huggingface.co. Without
            # this, huggingface_hub retries each failed network call 5x with
            # exponential backoff (~23s per file) before giving up — the desktop
            # UI hangs for 60–90s on every uncached load when offline.
            model_kwargs = dict(pipeline_params.get("model_kwargs") or {})
            model_kwargs["local_files_only"] = True
            pipeline_params["model_kwargs"] = model_kwargs

        pipeline_params.update(self.pipeline_kwargs)

        try:
            self._pipeline = pipeline("text-generation", **pipeline_params)
        except OSError as exc:
            if self.local_files_only:
                raise OSError(
                    f"HuggingFace model '{self.model}' is not cached locally and "
                    "the host appears to be offline. Connect to the internet "
                    "and pull the model from Settings, or pick a model already "
                    "in the local cache."
                ) from exc
            raise
        self.client = self._pipeline
        self.context_window_tokens = (
            context_window_tokens or self._infer_context_window_tokens()
        )
        self._last_usage: Optional[Dict[str, int]] = None

    def _resolve_device_and_dtype(self, device: Optional[str]):
        """Pick a (pipeline-device, torch_dtype) pair for this host.

        Apple Silicon MPS doesn't support BFloat16, but most modern
        instruct models (Gemma 3, Llama 3.x) ship with bfloat16 weights
        and crash with `TypeError: BFloat16 is not supported on MPS`
        on first forward. Solution: when MPS is in play we explicitly
        force float16 so weights are cast at load time.

        On CUDA we default to bfloat16 (better numerics, no MPS issue).
        On CPU we leave dtype alone (float32, slow but always correct).
        """
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore

        # Auto-pick when caller didn't specify: MPS on Apple Silicon,
        # CUDA on systems with a GPU, otherwise CPU.
        if device is None:
            if torch is not None:
                if (
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                ):
                    return "mps", torch.float16
                if torch.cuda.is_available():
                    return 0, torch.bfloat16
            return -1, None

        if isinstance(device, int):
            dtype = None
            if torch is not None:
                if device >= 0 and torch.cuda.is_available():
                    dtype = torch.bfloat16
            return device, dtype

        device_lower = str(device).lower()
        if device_lower == "cpu":
            return -1, None
        if device_lower == "mps":
            dtype = torch.float16 if torch is not None else None
            return "mps", dtype
        if device_lower.startswith("cuda"):
            dtype = torch.bfloat16 if torch is not None else None
            return 0, dtype
        try:
            return int(device_lower), None
        except ValueError:
            logger.warning("Unrecognized device '%s'. Defaulting to CPU.", device)
            return -1, None

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
        """Convert chat-style messages into a model-native prompt.

        Modern instruct models (Gemma, Llama 3, Qwen, Mistral-Instruct) ship
        a Jinja ``chat_template`` on their tokenizer. Using it produces the
        exact special tokens the model was trained on, which:
          - bounds the response with the right end-of-turn token, so the
            model stops cleanly instead of leaking continuation chatter;
          - avoids the bracketed ``[user]/[assistant]`` shim which Gemma
            isn't trained on and treats as ordinary text.

        Some models (notably Gemma) reject a leading ``system`` role —
        we fold the system message into the first user turn as a fallback.
        Bare base models without a chat template fall back to the simple
        bracket shim.
        """
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
        if tokenizer is not None and chat_template:
            normalized = self._normalize_messages_for_chat_template(messages)
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
            role = message.get("role", "user").lower()
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
    def _normalize_messages_for_chat_template(
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Coerce the message list into a shape every chat template accepts.

        - Drops messages with empty content (chat templates usually error).
        - Folds a leading ``system`` message into the first user turn for
          models (Gemma) whose templates don't accept system roles.
        - Coerces unknown roles to ``user``.
        """
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
            logger.warning(
                "HuggingFaceLLM does not support tool calling; ignoring %d tool(s) "
                "and generating text only.",
                len(tools),
            )

        prompt = self._messages_to_prompt(messages)
        self._last_usage = None
        self._last_response_metadata = {}
        output_text = self._run_generation(prompt)
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(output_text)
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            None, text=output_text, max_output_tokens=self.max_new_tokens
        )
        return output_text

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream text generation token-by-token via TextIteratorStreamer.

        HuggingFace pipelines don't natively yield chunks, so we run the
        pipeline call on a worker thread and pump tokens through a
        ``TextIteratorStreamer`` that the main thread iterates over. The
        event shapes match the OpenAI/Ollama providers so the agent
        runtime treats this provider identically.

        Tool calling is not supported on raw text-generation pipelines —
        we log and drop any tool list rather than fail.
        """
        if tools:
            logger.warning(
                "HuggingFaceLLM does not support tool calling; ignoring %d tool(s) "
                "and streaming text only.",
                len(tools),
            )

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HuggingFace streaming."
            ) from exc

        prompt = self._messages_to_prompt(messages)
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is None:
            # No tokenizer means we can't stream — fall back to one-shot
            # generate and yield as a single content event.
            text = self._run_generation(prompt)
            yield {"type": "content", "content": text}
            yield {"type": "done", "content": text}
            return

        from queue import Empty, Full, Queue
        from threading import Event

        from transformers import StoppingCriteria, StoppingCriteriaList

        from ..streaming import (
            StreamCancelled,
            check_cancelled,
            current_cancellation,
            start_owned_worker,
        )

        cancelled, finished = Event(), Event()
        channel = Queue(maxsize=64)
        terminal = object()
        token = current_cancellation.get()
        unregister = token.register(cancelled.set) if token else lambda: None

        class StopWhenCancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        class BoundedStreamer(TextIteratorStreamer):
            def on_finalized_text(self, text, stream_end=False):
                for value in [text, terminal] if stream_end else [text]:
                    while not cancelled.is_set():
                        try:
                            channel.put(value, timeout=0.05)
                            break
                        except Full:
                            continue
                    else:
                        raise StreamCancelled()

        streamer = BoundedStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.1
        )
        gen_error = []
        self._last_usage = None
        self._last_response_metadata = {}

        def generate():
            try:
                self._pipeline(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([StopWhenCancelled()]),
                )
            except BaseException as exc:
                gen_error.append(exc)
            finally:
                finished.set()

        worker = start_owned_worker(generate, name="memorizz-huggingface-stream")
        accumulated = []
        try:
            while True:
                check_cancelled()
                try:
                    chunk = channel.get(timeout=0.1)
                except Empty:
                    if finished.is_set():
                        if gen_error:
                            raise gen_error[0]
                        from .streaming import ProviderStreamError

                        raise ProviderStreamError("provider_stream_incomplete")
                    continue
                if chunk is terminal:
                    # A final streamer callback can precede pipeline failure.
                    while not finished.wait(0.1):
                        check_cancelled()
                    if gen_error:
                        raise gen_error[0]
                    break
                if chunk:
                    accumulated.append(chunk)
                    yield {"type": "content", "content": chunk}
        finally:
            cancelled.set()
            unregister()
            worker.join(timeout=2)
        full = "".join(accumulated)
        prompt_tokens = self._count_tokens(prompt)
        completion_tokens = self._count_tokens(full)
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        from .response_metadata import response_metadata

        self._last_response_metadata = response_metadata(
            None, text=full, max_output_tokens=self.max_new_tokens
        )
        yield {"type": "usage", "usage": self._last_usage}
        yield {"type": "done", "content": full}

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

    def get_last_response_metadata(self) -> Dict[str, Any]:
        from .response_metadata import last_response_metadata

        return last_response_metadata(self)

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens
