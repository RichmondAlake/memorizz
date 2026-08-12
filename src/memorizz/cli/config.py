# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""CLI-facing configuration: canonical paths, env loading, and defaults.

Thin wrapper over :mod:`memorizz._env_io` so the rest of the CLI imports paths
and env helpers from one place, plus the small set of CLI default constants.
"""

import json

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
    "state_file",
    "load_state",
    "save_state",
    "clear_state",
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

#: System instruction for the default "memory assistant" mode. Written to steer
#: even small local models toward using memory and chaining web tools correctly.
MEMORY_ASSISTANT_INSTRUCTION = (
    "You are memorizz, a helpful assistant with persistent long-term memory.\n"
    "- Remember salient facts, preferences, and context the user shares, and "
    "recall them naturally in later turns and future sessions.\n"
    "- For questions about the user, yourself, or this conversation (e.g. "
    '"what did I say", "what was my first question", "what\'s my name"), '
    "answer from your conversation history and memory. Do NOT search the web for "
    "these, and never invent a source.\n"
    "- Use internet_search ONLY for current or external information. After "
    "searching, call open_web_page on the most relevant result to read the full "
    "article before answering, then cite the source title and URL.\n"
    "- Be concise and direct."
)


# --- Persistent CLI state (default agent id + memory/thread ids) -------------
# So that each `memorizz` launch reuses ONE persistent MemAgent and its active
# conversation, instead of creating a throwaway agent per session.


def state_file():
    """Path to the CLI state file holding the persistent agent + conversation."""
    return memorizz_home() / "state.json"


def load_state():
    """Load persisted CLI state (agent/memory/thread ids). Returns {} if absent."""
    path = state_file()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_state(updates):
    """Merge non-None values into the CLI state file (creating it)."""
    ensure_home()
    state = load_state()
    state.update({k: v for k, v in updates.items() if v is not None})
    state_file().write_text(json.dumps(state, indent=2))


def clear_state(keys):
    """Remove the given keys from the CLI state file (e.g. after a memory wipe)."""
    state = load_state()
    changed = False
    for key in keys:
        if key in state:
            del state[key]
            changed = True
    if changed:
        ensure_home()
        state_file().write_text(json.dumps(state, indent=2))
