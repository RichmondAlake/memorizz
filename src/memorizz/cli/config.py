# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""CLI-facing configuration: canonical paths, env loading, and defaults.

Thin wrapper over :mod:`memorizz._env_io` so the rest of the CLI imports paths
and env helpers from one place, plus the small set of CLI default constants.
"""

from .._env_io import (
    apply_env_updates,
    ensure_home,
    history_file,
    load_layered_env,
    memorizz_home,
    memory_root,
    resolve_env_file,
    update_env_file,
)

__all__ = [
    "apply_env_updates",
    "ensure_home",
    "history_file",
    "load_layered_env",
    "memorizz_home",
    "memory_root",
    "resolve_env_file",
    "update_env_file",
    "DEFAULT_OLLAMA_LLM",
    "DEFAULT_OLLAMA_EMBED_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_AZURE_MODEL",
    "MEMORY_ASSISTANT_INSTRUCTION",
    "OLLAMA_DEFAULT_HOST",
]

# --- Provider defaults (all overridable via MEMORIZZ_DEFAULT_LLM_MODEL etc.) ---

#: Fallback Ollama chat model when none is configured and the daemon has models
#: but we cannot otherwise choose one.
DEFAULT_OLLAMA_LLM = "llama3.1"

#: Default Ollama embedding model for the fully-local stack (dim 768).
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"

#: Default cloud models, matching the README quickstart / library docstrings.
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_AZURE_MODEL = "gpt-4o"

#: Default Ollama daemon host (also overridable via OLLAMA_HOST).
OLLAMA_DEFAULT_HOST = "http://localhost:11434"

#: System instruction for the default "memory assistant" mode.
MEMORY_ASSISTANT_INSTRUCTION = (
    "You are memorizz, a helpful assistant with persistent long-term memory. "
    "Remember the salient facts, preferences, and context the user shares, and "
    "recall them naturally in later turns. Be concise and direct."
)
