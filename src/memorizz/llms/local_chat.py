"""Prompt building shared by the in-process models (HuggingFace, MLX)."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def normalize_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
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


def chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    """Convert chat-style messages into a model-native prompt.

    Modern instruct models (Gemma, Llama 3, Qwen, Mistral-Instruct) ship
    a Jinja ``chat_template`` on their tokenizer. Using it produces the
    exact special tokens the model was trained on, which:
      - bounds the response with the right end-of-turn token, so the
        model stops cleanly instead of leaking continuation chatter;
      - avoids the bracketed ``[user]/[assistant]`` shim which Gemma
        isn't trained on and treats as ordinary text.

    Some models (notably Gemma) reject a leading ``system`` role, so the
    system message is folded into the first user turn. Bare base models
    without a chat template fall back to the simple bracket shim.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                normalize_chat_messages(messages),
                tokenize=False,
                add_generation_prompt=True,
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


class LocalChatMixin:
    """Methods the in-process models share. They set ``_last_usage`` and
    ``context_window_tokens`` and implement ``generate``."""

    def generate_text(self, prompt: str, instructions: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if instructions:
            messages.append({"role": "system", "content": instructions})
        messages.append({"role": "user", "content": prompt})
        return self.generate(messages)

    def get_last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage

    def get_context_window_tokens(self) -> Optional[int]:
        return self.context_window_tokens


__all__ = ["LocalChatMixin", "chat_prompt", "normalize_chat_messages"]
