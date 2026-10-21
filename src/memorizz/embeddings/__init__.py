# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Cap for the per-manager embedding memo below. Sized to comfortably hold a
# session's worth of repeated lookups (the same user query is embedded by the
# semantic cache, episodic recall, knowledge-base recall, and entity search
# within a single agent turn) without growing unbounded.
_EMBEDDING_CACHE_MAX_ENTRIES = 512


class EmbeddingProvider(Enum):
    """Enumeration of supported embedding providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    VOYAGEAI = "voyageai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedding provider with configuration.

        Parameters:
        -----------
        config : Optional[Dict[str, Any]]
            Provider-specific configuration parameters
        """
        self.config = config or {}

    @abstractmethod
    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding for the given text.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional provider-specific parameters

        Returns:
        --------
        List[float]
            The embedding vector
        """

    @abstractmethod
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings produced by this provider.

        Returns:
        --------
        int
            Number of dimensions in the embedding vector
        """

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model name for this provider.

        Returns:
        --------
        str
            Default model identifier
        """


class EmbeddingManager:
    """
    Central manager for embedding providers with configuration support.
    Implements the Factory pattern for provider creation.
    """

    def __init__(
        self,
        provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENAI,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the embedding manager.

        Parameters:
        -----------
        provider : Union[str, EmbeddingProvider]
            The embedding provider to use
        config : Optional[Dict[str, Any]]
            Configuration for the selected provider
        """
        if isinstance(provider, str):
            try:
                provider = EmbeddingProvider(provider.lower())
            except ValueError:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        self.provider_type = provider
        self.config = config or {}
        self._provider = self._create_provider()
        # LRU memo of text → embedding. Embeddings are deterministic for a
        # fixed provider+model (both pinned per manager instance), so repeated
        # embeds of the same text within a session are pure waste — one agent
        # turn used to embed the identical query string up to five times
        # (semantic cache, episodic recall, KB recall, toolbox, entity search).
        self._embedding_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._embedding_cache_lock = threading.Lock()

    def _create_provider(self) -> BaseEmbeddingProvider:
        """Create and return the appropriate embedding provider instance."""
        if self.provider_type == EmbeddingProvider.OPENAI:
            from .openai import OpenAIEmbeddingProvider

            return OpenAIEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.AZURE:
            from .azure import AzureOpenAIEmbeddingProvider

            return AzureOpenAIEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.OLLAMA:
            from .ollama import OllamaEmbeddingProvider

            return OllamaEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.VOYAGEAI:
            from .voyageai import VoyageAIEmbeddingProvider

            return VoyageAIEmbeddingProvider(self.config)
        elif self.provider_type == EmbeddingProvider.HUGGINGFACE:
            from .huggingface import HuggingFaceEmbeddingProvider

            return HuggingFaceEmbeddingProvider(self.config)
        else:
            raise ValueError(f"Provider {self.provider_type} not implemented")

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using the configured provider.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional provider-specific parameters

        Returns:
        --------
        List[float]
            The embedding vector
        """
        # Only memoize the plain-text call shape (optionally marked as a
        # query): other kwargs may override the model/dimensions, which would
        # make cached vectors wrong.
        cacheable = (
            isinstance(text, str) and bool(text) and set(kwargs) <= {"input_type"}
        )
        key = _cache_key(text, kwargs.get("input_type")) if cacheable else None
        if cacheable:
            with self._embedding_cache_lock:
                cached = self._embedding_cache.get(key)
                if cached is not None:
                    self._embedding_cache.move_to_end(key)
                    return list(cached)

        if set(kwargs) == {"input_type"}:
            result = (
                embed_as_query(self._provider.get_embedding, text)
                if kwargs["input_type"] == "query"
                else _without_input_type(self._provider.get_embedding, text, kwargs)
            )
        else:
            result = self._provider.get_embedding(text, **kwargs)

        if cacheable and isinstance(result, list) and result:
            with self._embedding_cache_lock:
                self._embedding_cache[key] = list(result)
                self._embedding_cache.move_to_end(key)
                while len(self._embedding_cache) > _EMBEDDING_CACHE_MAX_ENTRIES:
                    self._embedding_cache.popitem(last=False)
        return result

    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embed a batch while preserving the single-item LRU semantics.

        Providers may expose an optimized ``get_embeddings`` implementation;
        otherwise this method falls back to the stable single-item contract.
        """
        values = [str(text) for text in texts]
        if not values:
            return []
        provider_batch = getattr(self._provider, "get_embeddings", None)
        if set(kwargs) - {"input_type"} or not callable(provider_batch):
            return [self.get_embedding(text, **kwargs) for text in values]
        input_type = kwargs.get("input_type")

        results: List[Optional[List[float]]] = [None] * len(values)
        missing_indexes: List[int] = []
        missing_texts: List[str] = []
        with self._embedding_cache_lock:
            for index, text in enumerate(values):
                cached = self._embedding_cache.get(_cache_key(text, input_type))
                if cached is None:
                    missing_indexes.append(index)
                    missing_texts.append(text)
                else:
                    self._embedding_cache.move_to_end(_cache_key(text, input_type))
                    results[index] = list(cached)

        if missing_texts:
            generated = _without_input_type(provider_batch, missing_texts, kwargs)
            if len(generated) != len(missing_texts):
                raise RuntimeError(
                    "Embedding provider returned a different batch length "
                    f"({len(generated)} != {len(missing_texts)})"
                )
            with self._embedding_cache_lock:
                for index, text, embedding in zip(
                    missing_indexes, missing_texts, generated
                ):
                    value = list(embedding)
                    results[index] = value
                    if text:
                        key = _cache_key(text, input_type)
                        self._embedding_cache[key] = value
                        self._embedding_cache.move_to_end(key)
                while len(self._embedding_cache) > _EMBEDDING_CACHE_MAX_ENTRIES:
                    self._embedding_cache.popitem(last=False)
        return [list(item or []) for item in results]

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings from the current provider."""
        return self._provider.get_dimensions()

    def get_default_model(self) -> str:
        """Get the default model for the current provider."""
        return self._provider.get_default_model()

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the current provider configuration.

        Returns:
        --------
        Dict[str, Any]
            Provider information including type, model, and dimensions
        """
        return {
            "provider": self.provider_type.value,
            "model": self.get_default_model(),
            "dimensions": self.get_dimensions(),
            # Logged when a store starts; keys stay out of it.
            "config": {
                key: value
                for key, value in self.config.items()
                if key not in _SECRET_CONFIG_KEYS
            },
        }


_SECRET_CONFIG_KEYS = frozenset({"api_key", "apiKey", "token", "password"})


def shorten_embedding(vector: List[float], size: int) -> List[float]:
    """The first ``size`` values of ``vector``, scaled back to unit length.

    Matryoshka-trained models (OpenAI's text-embedding-3, nomic-embed-text,
    EmbeddingGemma, Qwen3-Embedding) keep most of their quality when cut
    this way; it is what their servers do when asked for fewer dimensions.
    """
    if len(vector) < size:
        raise ValueError(
            f"The model returned {len(vector)} dimensions, fewer than the "
            f"{size} configured; set dimensions to at most {len(vector)}."
        )
    if len(vector) == size:
        return list(vector)
    head = [float(value) for value in vector[:size]]
    norm = sum(value * value for value in head) ** 0.5
    return [value / norm for value in head] if norm else head


def embed_as_query(embed: Callable[..., List[float]], text: str) -> List[float]:
    """Call ``embed(text, input_type="query")``; embedders that take only the
    text (custom providers, test doubles) are called with the text alone."""
    try:
        return embed(text, input_type="query")
    except TypeError as exc:
        if "input_type" not in str(exc) and "argument" not in str(exc):
            raise
        return embed(text)


def _without_input_type(
    call: Callable[..., Any], value: Any, kwargs: Dict[str, Any]
) -> Any:
    if not kwargs:
        return call(value)
    try:
        return call(value, **kwargs)
    except TypeError as exc:
        if set(kwargs) != {"input_type"} or (
            "input_type" not in str(exc) and "argument" not in str(exc)
        ):
            raise
        return call(value)


def _cache_key(text: str, input_type: Optional[str]) -> str:
    # Queries and stored documents embed differently on asymmetric models
    # (Voyage, EmbeddingGemma, Qwen3-Embedding, MemoRizz fine-tuned models).
    return f"{input_type}\x00{text}" if input_type else text


# Global embedding manager instance (can be reconfigured)
_global_embedding_manager: Optional[EmbeddingManager] = None


def set_global_embedding_manager(manager: EmbeddingManager) -> EmbeddingManager:
    """
    Set the global embedding manager.

    This allows custom providers (e.g., Oracle/Mongo builders) to keep the
    module-level helpers in sync with the configured provider.
    """
    global _global_embedding_manager
    _global_embedding_manager = manager
    logger.info(
        "Set global embedding provider: %s",
        manager.get_provider_info(),
    )
    return manager


def configure_embeddings(
    provider: Union[str, EmbeddingProvider] = EmbeddingProvider.OPENAI,
    config: Optional[Dict[str, Any]] = None,
) -> EmbeddingManager:
    """
    Configure the global embedding provider.

    Parameters:
    -----------
    provider : Union[str, EmbeddingProvider]
        The embedding provider to use globally
    config : Optional[Dict[str, Any]]
        Configuration for the selected provider

    Returns:
    --------
    EmbeddingManager
        The configured embedding manager
    """
    manager = EmbeddingManager(provider, config)
    return set_global_embedding_manager(manager)


def get_embedding_manager() -> EmbeddingManager:
    """
    Get the global embedding manager, creating a default one if none exists.

    Returns:
    --------
    EmbeddingManager
        The global embedding manager instance
    """
    global _global_embedding_manager
    if _global_embedding_manager is None:
        _global_embedding_manager = _default_embedding_manager()
    return _global_embedding_manager


def embedding_settings_from_env() -> tuple:
    """``(provider, config)`` from the MEMORIZZ_DEFAULT_EMBEDDING_* settings
    (``("", {})`` when no provider is set). ``openai`` with
    MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL talks to an OpenAI-compatible server."""
    import os

    return embedding_settings_from(os.environ)


def embedding_settings_from(values: Dict[str, str]) -> tuple:
    """``(provider, config)`` from MEMORIZZ_DEFAULT_EMBEDDING_* values (the
    environment, or settings saved elsewhere, such as before a swap)."""
    name = (
        str(values.get("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "") or "").strip().lower()
    )
    config: Dict[str, Any] = {}
    if not name:
        return "", config
    model = str(values.get("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "") or "").strip()
    if model:
        config["model"] = model
    dimensions = str(
        values.get("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "") or ""
    ).strip()
    if dimensions.isdigit():
        key = "output_dimension" if name == "voyageai" else "dimensions"
        config[key] = int(dimensions)
        if name == "huggingface":
            # Matryoshka-trained models (EmbeddingGemma, Qwen3-Embedding)
            # keep their quality when cut to a smaller size.
            config["truncate_dim"] = int(dimensions)
    if name == "openai":
        config.update(openai_endpoint_settings(values))
    return name, config


def openai_endpoint_settings(values: Dict[str, str]) -> Dict[str, str]:
    """``base_url`` and ``api_key`` for an OpenAI-compatible embedding server
    (Ollama's ``/v1``, vLLM, LM Studio) from MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL
    and MEMORIZZ_DEFAULT_EMBEDDING_API_KEY; empty without a base URL."""
    base_url = str(values.get("MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL", "") or "").strip()
    if not base_url:
        return {}
    settings = {"base_url": base_url}
    api_key = str(values.get("MEMORIZZ_DEFAULT_EMBEDDING_API_KEY", "") or "").strip()
    if api_key:
        settings["api_key"] = api_key
    return settings


def _default_embedding_manager() -> EmbeddingManager:
    """The provider named by MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER, else OpenAI.

    Uses the same variables as the Oracle provider: _PROVIDER, _MODEL and
    _DIMENSIONS, so a local setup (``ollama`` with ``nomic-embed-text``) needs
    no OpenAI key.
    """
    name, config = embedding_settings_from_env()
    if name:
        try:
            manager = EmbeddingManager(name, config)
            logger.info("Using the default embedding provider: %s", name)
            return manager
        except ValueError as exc:
            logger.warning("Ignoring MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER: %s", exc)
    logger.info("No global embedding manager configured, using default OpenAI provider")
    return EmbeddingManager()


# Convenience functions for backward compatibility
def get_embedding(text: str, **kwargs) -> List[float]:
    """
    Generate embedding using the globally configured provider.
    Provides backward compatibility for existing code.

    Parameters:
    -----------
    text : str
        The text to embed
    **kwargs
        Additional provider-specific parameters

    Returns:
    --------
    List[float]
        The embedding vector
    """
    return get_embedding_manager().get_embedding(text, **kwargs)


def get_query_embedding(text: str, **kwargs) -> List[float]:
    """Embed a search query. Asymmetric models (Voyage, EmbeddingGemma,
    Qwen3-Embedding, models fine-tuned in MemoRizz) embed queries with their
    query prompt and stored memories with their document prompt; others
    ignore the distinction."""
    return get_embedding_manager().get_embedding(text, input_type="query", **kwargs)


def reset_embedding_manager() -> None:
    """Forget the global embedding manager so the next call rebuilds it from
    the current ``MEMORIZZ_DEFAULT_EMBEDDING_*`` settings."""
    global _global_embedding_manager
    _global_embedding_manager = None


def get_embedding_dimensions(model: Optional[str] = None) -> int:
    """
    Get embedding dimensions from the globally configured provider.
    Provides backward compatibility for existing code.

    Parameters:
    -----------
    model : Optional[str]
        Model parameter (for backward compatibility, may be ignored)

    Returns:
    --------
    int
        The number of dimensions in the embedding vector
    """
    dimensions = get_embedding_manager().get_dimensions()
    provider_info = get_embedding_manager().get_provider_info()
    logger.debug(
        f"Inferred embedding dimensions: {dimensions} "
        f"(provider: {provider_info['provider']}, model: {provider_info['model']})"
    )
    return dimensions
