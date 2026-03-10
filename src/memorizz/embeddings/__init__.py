# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


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
        return self._provider.get_embedding(text, **kwargs)

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


def infer_embedding_dimensions(
    provider: Union[str, EmbeddingProvider],
    model: str,
    config: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Infer embedding dimensions for a given provider and model without instantiating.
    This is useful for schema creation and validation.

    Parameters:
    -----------
    provider : Union[str, EmbeddingProvider]
        The embedding provider name or enum
    model : str
        The model name
    config : Optional[Dict[str, Any]]
        Optional configuration (e.g., dimensions override for OpenAI)

    Returns:
    --------
    int
        The inferred number of dimensions

    Examples:
    --------
    >>> infer_embedding_dimensions("openai", "text-embedding-3-small")
    1536
    >>> infer_embedding_dimensions("openai", "text-embedding-3-small", {"dimensions": 512})
    512
    >>> infer_embedding_dimensions("ollama", "nomic-embed-text")
    768
    """
    provider_str = (
        provider.value if isinstance(provider, EmbeddingProvider) else provider
    )

    # OpenAI provider
    if provider_str in ["openai", "azure"]:
        from .openai.provider import OpenAIEmbeddingProvider

        # Check if dimensions are explicitly set in config
        if config and "dimensions" in config:
            dimensions = config["dimensions"]
            logger.debug(
                f"Inferred dimensions from config: {dimensions} "
                f"(provider: {provider_str}, model: {model})"
            )
            return dimensions

        # Use model's default/max dimensions
        max_dims = OpenAIEmbeddingProvider.MODEL_DIMENSIONS.get(model)
        if max_dims:
            # For OpenAI 3-small/3-large, default to max unless specified
            dimensions = config.get("dimensions", max_dims) if config else max_dims
            logger.debug(
                f"Inferred dimensions from model: {dimensions} "
                f"(provider: {provider_str}, model: {model})"
            )
            return dimensions
        else:
            # Fallback for unknown models
            logger.warning(
                f"Unknown OpenAI model {model}, defaulting to 1536 dimensions"
            )
            return 1536

    # Ollama provider
    elif provider_str == "ollama":
        from .ollama.provider import OllamaEmbeddingProvider

        dimensions = OllamaEmbeddingProvider.MODEL_DIMENSIONS.get(model, 768)
        logger.debug(
            f"Inferred dimensions from model: {dimensions} "
            f"(provider: {provider_str}, model: {model})"
        )
        return dimensions

    # VoyageAI provider
    elif provider_str == "voyageai":
        from .voyageai.provider import VoyageAIEmbeddingProvider

        # VoyageAI models have configurable dimensions
        if config and "output_dimension" in config:
            dimensions = config["output_dimension"]
            logger.debug(
                f"Inferred dimensions from config: {dimensions} "
                f"(provider: {provider_str}, model: {model})"
            )
            return dimensions

        # Use model's default dimensions
        text_models = VoyageAIEmbeddingProvider.TEXT_MODELS
        if model in text_models:
            dimensions = text_models[model]["default_dimensions"]
            logger.debug(
                f"Inferred dimensions from model: {dimensions} "
                f"(provider: {provider_str}, model: {model})"
            )
            return dimensions
        else:
            logger.warning(
                f"Unknown VoyageAI model {model}, defaulting to 1024 dimensions"
            )
            return 1024

    # Hugging Face provider
    elif provider_str == "huggingface":
        from .huggingface.provider import HuggingFaceEmbeddingProvider

        if config and "dimensions" in config:
            dimensions = config["dimensions"]
            logger.debug(
                "Inferred dimensions from config: %s (provider: %s, model: %s)",
                dimensions,
                provider_str,
                model,
            )
            return dimensions

        dims = HuggingFaceEmbeddingProvider.MODEL_DIMENSIONS.get(model)
        if dims:
            logger.debug(
                "Inferred dimensions from model metadata: %s (provider: %s, model: %s)",
                dims,
                provider_str,
                model,
            )
            return dims
        else:
            logger.warning(
                "Unknown Hugging Face embedding model %s. Defaulting to 768 dimensions.",
                model,
            )
            return 768

    # Unknown provider - try to get from global config or raise
    else:
        logger.warning(
            f"Unknown provider {provider_str}, attempting to infer from global config"
        )
        try:
            return get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Could not infer dimensions: {e}")
            raise ValueError(
                f"Cannot infer dimensions for provider '{provider_str}' and model '{model}'. "
                "Please specify dimensions explicitly in config."
            )
