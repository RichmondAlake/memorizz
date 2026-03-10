# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

# src/memorizz/llms/llm_factory.py

from typing import Any, Dict

from .azure import AzureOpenAI
from .huggingface import HuggingFaceLLM
from .llm_provider import LLMProvider
from .openai import OpenAI


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
        # Create a copy of the config and remove the 'provider' key
        openai_config = config.copy()
        openai_config.pop("provider", None)
        return OpenAI(**openai_config)

    elif provider_name == "azure":
        # Create a copy of the config and remove the 'provider' key
        azure_config = config.copy()
        azure_config.pop("provider", None)
        return AzureOpenAI(
            azure_endpoint=azure_config.get("azure_endpoint"),
            api_version=azure_config.get("api_version"),
            deployment_name=azure_config.get("deployment_name"),
        )

    elif provider_name == "huggingface":
        huggingface_config = config.copy()
        huggingface_config.pop("provider", None)
        return HuggingFaceLLM(**huggingface_config)

    elif provider_name == "anthropic":
        from .anthropic import Anthropic

        anthropic_config = config.copy()
        anthropic_config.pop("provider", None)
        return Anthropic(**anthropic_config)

    elif provider_name == "ollama":
        from .ollama import OllamaLLM

        ollama_config = config.copy()
        ollama_config.pop("provider", None)
        return OllamaLLM(**ollama_config)

    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")
