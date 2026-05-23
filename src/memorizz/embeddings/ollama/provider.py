# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
from typing import Any, Dict, List

from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider implementation."""

    # Common Ollama embedding models and their typical dimensions
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed": 1024,
        "all-minilm": 384,
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Ollama embedding provider.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - model: str (default: "nomic-embed-text")
            - base_url: str (default: "http://localhost:11434")
            - timeout: int (default: 30)
        """
        super().__init__(config)

        # Set default configuration
        self.model = self.config.get("model", "nomic-embed-text")
        self.base_url = self.config.get("base_url", "http://localhost:11434")
        self.timeout = self.config.get("timeout", 30)

        # Determine dimensions - use known value or probe the model
        self.dimensions = self.MODEL_DIMENSIONS.get(self.model, 768)  # Default fallback

        # Initialize the embeddings client
        self._init_client()

        # Probe actual dimensions on first use
        self._dimensions_probed = False

        logger.info(
            f"Initialized Ollama provider with model={self.model}, base_url={self.base_url}"
        )

    def _init_client(self):
        """Initialize the Ollama embeddings client."""
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "langchain_ollama is required for Ollama embeddings. "
                "Install it with: pip install langchain-ollama"
            )

        # langchain-ollama >=1.0 dropped the `timeout` constructor arg.
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model, base_url=self.base_url, timeout=self.timeout
            )
        except (TypeError, ValueError):
            self.embeddings = OllamaEmbeddings(model=self.model, base_url=self.base_url)

    def _probe_dimensions(self):
        """Probe the actual dimensions of the model by generating a test embedding."""
        if self._dimensions_probed:
            return

        try:
            test_embedding = self.embeddings.embed_query("test")
            if test_embedding:
                actual_dimensions = len(test_embedding)
                if actual_dimensions != self.dimensions:
                    logger.info(
                        f"Model {self.model} actual dimensions: {actual_dimensions}, updating from assumed {self.dimensions}"
                    )
                    self.dimensions = actual_dimensions
                self._dimensions_probed = True
        except Exception as e:
            logger.warning(f"Could not probe dimensions for {self.model}: {str(e)}")
            # Keep the assumed dimensions
            self._dimensions_probed = True

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using Ollama.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional parameters:
            - model: str (override default model - requires reinitializing client)

        Returns:
        --------
        List[float]
            The embedding vector
        """
        # Check if model override is requested
        model = kwargs.get("model", self.model)
        original_model = self.model
        overridden = model != self.model
        if overridden:
            self.model = model
            self._init_client()
            self._dimensions_probed = False

        # Clean the text
        text = text.replace("\n", " ")

        try:
            # Probe dimensions on first use
            if not self._dimensions_probed:
                self._probe_dimensions()

            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {str(e)}")
            raise
        finally:
            if overridden:
                self.model = original_model
                self._init_client()
                self._dimensions_probed = False

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        if not self._dimensions_probed:
            self._probe_dimensions()
        return self.dimensions

    def get_default_model(self) -> str:
        """Get the default model name for this provider."""
        return self.model

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of commonly available Ollama embedding models."""
        return list(cls.MODEL_DIMENSIONS.keys())

    @classmethod
    def get_model_dimensions(cls, model: str) -> int:
        """Get known dimensions for a specific model."""
        return cls.MODEL_DIMENSIONS.get(model, 768)  # Default fallback
