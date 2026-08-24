# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

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
        # Only memoize the plain-text call shape: kwargs may override the
        # model/dimensions, which would make cached vectors wrong.
        cacheable = isinstance(text, str) and bool(text) and not kwargs
        if cacheable:
            with self._embedding_cache_lock:
                cached = self._embedding_cache.get(text)
                if cached is not None:
                    self._embedding_cache.move_to_end(text)
                    return list(cached)

        result = self._provider.get_embedding(text, **kwargs)

        if cacheable and isinstance(result, list) and result:
            with self._embedding_cache_lock:
                self._embedding_cache[text] = list(result)
                self._embedding_cache.move_to_end(text)
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
        if kwargs or not callable(provider_batch):
            return [self.get_embedding(text, **kwargs) for text in values]

        results: List[Optional[List[float]]] = [None] * len(values)
        missing_indexes: List[int] = []
        missing_texts: List[str] = []
        with self._embedding_cache_lock:
            for index, text in enumerate(values):
                cached = self._embedding_cache.get(text)
                if cached is None:
                    missing_indexes.append(index)
                    missing_texts.append(text)
                else:
                    self._embedding_cache.move_to_end(text)
                    results[index] = list(cached)

        if missing_texts:
            generated = provider_batch(missing_texts)
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
                        self._embedding_cache[text] = value
                        self._embedding_cache.move_to_end(text)
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
            "config": self.config,
        }


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
        logger.info(
            "No global embedding manager configured, using default OpenAI provider"
        )
        _global_embedding_manager = EmbeddingManager()
    return _global_embedding_manager


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
