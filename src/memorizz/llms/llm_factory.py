# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

# src/memorizz/llms/llm_factory.py

import inspect
import logging
from typing import Any, Dict

from .llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _filter_kwargs_for_class(config: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Drop keys the target provider's __init__ doesn't accept.

    Saved llm_configs accumulate keys when users switch providers — e.g.
    ``max_tokens`` from a previous OpenAI run still rides along when the
    agent later switches to HuggingFace. Splatting the full dict into a
    constructor that doesn't accept those keys raises TypeError, which
    used to surface as the generic "No LLM model configured" error. We
    inspect the target class signature and silently drop keys that aren't
    in its parameter list — unless the class accepts **kwargs, in which
    case we trust the constructor to handle them.
    """
    sig = inspect.signature(cls.__init__)
    accepts_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_keyword:
        return {k: v for k, v in config.items() if k != "provider"}

    accepted = {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {}
    dropped = []
    for key, value in config.items():
        if key == "provider":
            continue
        if key in accepted:
            filtered[key] = value
        else:
            dropped.append(key)
    if dropped:
        logger.info(
            "Dropped unsupported kwargs for %s: %s",
            cls.__name__,
            ", ".join(sorted(dropped)),
        )
    return filtered


def create_llm_provider(config: Dict[str, Any]) -> LLMProvider:
    """
    Factory function to create an LLM provider instance from a configuration dictionary.

    Parameters:
    -----------
    config : Dict[str, Any]
        A dictionary containing the provider name and its specific parameters.
        Example for OpenAI: {"provider": "openai", "model": "gpt-4o"}
        Example for Azure: {"provider": "azure", "deployment_name": "my-gpt4"}
        Example for Anthropic: {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}
        Example for Ollama: {"provider": "ollama", "model": "llama3.1"}

    Returns:
    --------
    LLMProvider
        An instance of the specified LLM provider.

    Raises:
    -------
    ValueError
        If the provider specified in the config is unknown.
    """
    provider_name = config.get("provider", "openai").lower()
    if provider_name == "openai":
        from .openai import OpenAI

        return OpenAI(**_filter_kwargs_for_class(config, OpenAI))

    elif provider_name == "azure":
        from .azure import AzureOpenAI

        return AzureOpenAI(**_filter_kwargs_for_class(config, AzureOpenAI))

    elif provider_name == "huggingface":
        from .huggingface import HuggingFaceLLM

        return HuggingFaceLLM(**_filter_kwargs_for_class(config, HuggingFaceLLM))

    elif provider_name == "anthropic":
        from .anthropic import Anthropic

        return Anthropic(**_filter_kwargs_for_class(config, Anthropic))

    elif provider_name == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(**_filter_kwargs_for_class(config, OllamaLLM))

    elif provider_name == "mlx":
        from .mlx import MLXLLM

        return MLXLLM(**_filter_kwargs_for_class(config, MLXLLM))

    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")
