# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""LLM providers, lazily loaded so `import memorizz` doesn't pull provider SDKs.

Each class is imported on first access via module ``__getattr__`` (PEP 562), so
`from memorizz.llms import OpenAI` still works but importing the package costs
nothing until a provider is actually used.
"""

import importlib

_LAZY = {
    "OpenAI": ".openai",
    "AzureOpenAI": ".azure",
    "HuggingFaceLLM": ".huggingface",
    "Anthropic": ".anthropic",
    "OllamaLLM": ".ollama",
    "MLXLLM": ".mlx",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    mod = importlib.import_module(module, __name__)
    return getattr(mod, name)


__all__ = list(_LAZY)
