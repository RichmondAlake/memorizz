# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Build a session MemAgent from the environment, defaulting to a local stack.

The detection order favors an explicit choice, then cloud keys, then a fully
local Ollama + FileSystem stack so that ``memorizz`` works with zero API keys
when an Ollama daemon is running.

All heavy imports (``memorizz`` core, providers) are deferred into functions so
the CLI's ``--help`` / ``init`` / ``--version`` paths never pay for them.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import config as cfg
from . import ollama_probe


class NoProviderConfigured(Exception):
    """Raised when no LLM provider can be auto-detected (no key, no Ollama)."""


class NeedsOllamaPull(Exception):
    """Raised when Ollama is reachable but has no models pulled."""

    def __init__(self, host: str):
        self.host = host
        super().__init__(
            f"Ollama is running at {host} but has no models pulled. "
            f"Try: ollama pull {cfg.DEFAULT_OLLAMA_LLM}"
        )


@dataclass
class Session:
    """Mutable state for one interactive (or one-shot) CLI session."""

    agent: Any
    provider: Any
    llm_config: Dict[str, Any]
    code_mode: bool = False
    memory_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    console: Any = None
    warnings: Optional[List[str]] = None

    @property
    def provider_name(self) -> str:
        return str(self.llm_config.get("provider", "?"))

    @property
    def model_name(self) -> str:
        return str(
            self.llm_config.get("model")
            or self.llm_config.get("deployment_name")
            or "?"
        )

    def sync_ids(self) -> None:
        """Pull the active memory/thread ids off the agent after a turn."""
        try:
            self.memory_id = self.agent.get_current_memory_id() or self.memory_id
            self.thread_id = self.agent.get_current_thread_id() or self.thread_id
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# LLM detection
# --------------------------------------------------------------------------- #


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _default_model(provider: str) -> str:
    override = _env("MEMORIZZ_DEFAULT_LLM_MODEL")
    if override:
        return override
    return {
        "openai": cfg.DEFAULT_OPENAI_MODEL,
        "anthropic": cfg.DEFAULT_ANTHROPIC_MODEL,
        "azure": cfg.DEFAULT_AZURE_MODEL,
        "ollama": cfg.DEFAULT_OLLAMA_LLM,
    }.get(provider, cfg.DEFAULT_OPENAI_MODEL)


def _pick_ollama_model(models: List[str]) -> Optional[str]:
    """Choose a chat model from the daemon's tags, avoiding embedding models."""
    preferred = _env("MEMORIZZ_DEFAULT_LLM_MODEL")
    if preferred:
        if preferred in models:
            return preferred
        for m in models:
            if m.split(":")[0] == preferred.split(":")[0]:
                return m
    chat = [m for m in models if "embed" not in m.lower()]
    pool = chat or models
    for fav in (
        "llama3.1",
        "llama3.2",
        "llama3",
        "qwen2.5",
        "qwen",
        "mistral",
        "gemma",
    ):
        for m in pool:
            if m.split(":")[0] == fav or m.startswith(fav):
                return m
    return pool[0] if pool else None


def config_for_provider(provider: str) -> Dict[str, Any]:
    """Build an llm_config dict for an explicitly chosen provider."""
    provider = provider.lower()
    if provider == "ollama":
        host = ollama_probe.resolve_host()
        models = ollama_probe.list_models(host)
        model = _pick_ollama_model(models or []) or _default_model("ollama")
        return {"provider": "ollama", "model": model, "host": host}
    if provider == "azure":
        return {
            "provider": "azure",
            "deployment_name": _env("AZURE_OPENAI_DEPLOYMENT")
            or _default_model("azure"),
        }
    return {"provider": provider, "model": _default_model(provider)}


def detect_llm_config() -> Dict[str, Any]:
    """Auto-detect an llm_config from the environment.

    Order: explicit ``MEMORIZZ_DEFAULT_LLM_PROVIDER`` > Anthropic key > OpenAI key
    > Azure key > local Ollama. Raises :class:`NeedsOllamaPull` or
    :class:`NoProviderConfigured` when nothing is usable.
    """
    explicit = _env("MEMORIZZ_DEFAULT_LLM_PROVIDER")
    if explicit:
        return config_for_provider(explicit)

    if _env("ANTHROPIC_API_KEY"):
        return {"provider": "anthropic", "model": _default_model("anthropic")}
    if _env("OPENAI_API_KEY"):
        return {"provider": "openai", "model": _default_model("openai")}
    if _env("AZURE_OPENAI_API_KEY"):
        return config_for_provider("azure")

    # No cloud key — probe a local Ollama daemon.
    host = ollama_probe.resolve_host()
    result = ollama_probe.probe(host)
    if result["reachable"]:
        models = list(result["models"])  # type: ignore[arg-type]
        if models:
            model = _pick_ollama_model(models) or _default_model("ollama")
            return {"provider": "ollama", "model": model, "host": host}
        raise NeedsOllamaPull(host)

    raise NoProviderConfigured()


# --------------------------------------------------------------------------- #
# Memory provider detection
# --------------------------------------------------------------------------- #


def _choose_embedding(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """Pick an embedding provider for the FileSystem store.

    Returns kwargs for ``FileSystemConfig`` (``embedding_provider`` /
    ``embedding_config``). Falls back to no embeddings (FAISS/brute-force) when
    nothing local or cloud is available.
    """
    explicit = _env("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER")
    if explicit:
        return {"embedding_provider": explicit, "embedding_config": {}}

    # Fully-local: if the LLM is Ollama (or Ollama is reachable) and the embed
    # model is pulled, use Ollama embeddings so nothing hits the network.
    host = llm_config.get("host") if llm_config.get("provider") == "ollama" else None
    host = host or ollama_probe.resolve_host()
    models = ollama_probe.list_models(host)
    if models is not None:
        embed_model = (
            _env("MEMORIZZ_DEFAULT_EMBEDDING_MODEL") or cfg.DEFAULT_OLLAMA_EMBED_MODEL
        )
        has_embed = any(m.split(":")[0] == embed_model.split(":")[0] for m in models)
        if has_embed:
            return {
                "embedding_provider": "ollama",
                "embedding_config": {"model": embed_model, "host": host},
            }

    # Cloud fallback: OpenAI embeddings if a key is present.
    if _env("OPENAI_API_KEY"):
        return {
            "embedding_provider": "openai",
            "embedding_config": {"model": "text-embedding-3-small"},
        }

    # Last resort: no embeddings; FileSystem degrades to brute-force / no vector.
    return {"embedding_provider": None, "embedding_config": None}


def detect_memory_provider(llm_config: Dict[str, Any], warnings: List[str]) -> Any:
    """Build a MemoryProvider, defaulting to a local FileSystem store."""
    backend = (_env("MEMORIZZ_BACKEND") or "").lower()

    if backend == "mongodb" and _env("MONGODB_URI"):
        from ..memory_provider.mongodb import MongoDBConfig, MongoDBProvider

        return MongoDBProvider(
            MongoDBConfig(
                uri=_env("MONGODB_URI"),
                db_name=_env("MONGODB_DB_NAME") or "memorizz",
            )
        )

    if backend == "oracle" and _env("ORACLE_DSN"):
        from ..memory_provider.oracle import OracleConfig, OracleProvider

        return OracleProvider(
            OracleConfig(
                user=_env("ORACLE_USER"),
                password=_env("ORACLE_PASSWORD"),
                dsn=_env("ORACLE_DSN"),
                schema=_env("ORACLE_SCHEMA"),
            )
        )

    # Default: zero-config local FileSystem store under ~/.memorizz/memory.
    from ..memory_provider import FileSystemConfig, FileSystemProvider

    embed = _choose_embedding(llm_config)
    if embed.get("embedding_provider") is None:
        warnings.append(
            "No embedding model available — semantic recall is degraded "
            "(brute-force). Run `ollama pull nomic-embed-text` or set OPENAI_API_KEY."
        )
    root = cfg.memory_root()
    return FileSystemProvider(
        FileSystemConfig(
            root_path=root,
            embedding_provider=embed.get("embedding_provider"),
            embedding_config=embed.get("embedding_config"),
        )
    )


# --------------------------------------------------------------------------- #
# Agent assembly
# --------------------------------------------------------------------------- #


def build_session_agent(
    code_mode: bool = False,
    llm_config: Optional[Dict[str, Any]] = None,
    memory_provider: Optional[Any] = None,
    instruction: Optional[str] = None,
) -> Session:
    """Detect config and build a ready-to-run :class:`Session`."""
    from ..memagent.builders.agent_builder import MemAgentBuilder

    warnings: List[str] = []
    resolved_llm = llm_config or detect_llm_config()
    provider = memory_provider or detect_memory_provider(resolved_llm, warnings)

    builder = (
        MemAgentBuilder()
        .with_instruction(instruction or cfg.MEMORY_ASSISTANT_INSTRUCTION)
        .with_application_mode("assistant")
        .with_llm_config(resolved_llm)
        .with_memory_provider(provider)
        .with_max_steps(20)
    )
    if code_mode:
        builder = builder.with_self_aware(True, {"allow_writes": True})

    agent = builder.build()
    return Session(
        agent=agent,
        provider=provider,
        llm_config=resolved_llm,
        code_mode=code_mode,
        warnings=warnings,
    )
