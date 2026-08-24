# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
from typing import Any, Dict, List

import openai

from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation."""

    # Model configuration with their dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,  # Default for 3-small is 1536, but can be reduced
        "text-embedding-3-large": 3072,  # Default for 3-large is 3072, but can be reduced
        "text-embedding-ada-002": 1536,  # Fixed dimensions
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OpenAI embedding provider.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - model: str (default: "text-embedding-3-small")
            - dimensions: int (default: 256, only for text-embedding-3-* models)
            - api_key: str (optional, uses env var if not provided)
            - base_url: str (optional, for custom endpoints)
        """
        super().__init__(config)

        # Set default configuration
        self.model = self.config.get("model", "text-embedding-3-small")
        self.dimensions = self.config.get("dimensions", 256)

        # Validate model and dimensions
        if self.model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported OpenAI model: {self.model}. Supported models: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        # For ada-002, dimensions cannot be customized
        if self.model == "text-embedding-ada-002" and self.dimensions != 1536:
            logger.warning(
                f"Model {self.model} has fixed dimensions of 1536. Ignoring custom dimensions parameter."
            )
            self.dimensions = 1536

        # For 3-small and 3-large, validate dimensions are within allowed range
        if self.model in ["text-embedding-3-small", "text-embedding-3-large"]:
            max_dims = self.MODEL_DIMENSIONS[self.model]
            if self.dimensions > max_dims:
                raise ValueError(
                    f"Dimensions {self.dimensions} exceed maximum {max_dims} for model {self.model}"
                )

        # Initialize OpenAI client
        client_kwargs = {}
        if "api_key" in self.config:
            client_kwargs["api_key"] = self.config["api_key"]
        if "base_url" in self.config:
            client_kwargs["base_url"] = self.config["base_url"]

        self.client = openai.OpenAI(**client_kwargs)

        logger.info(
            f"Initialized OpenAI provider with model={self.model}, dimensions={self.dimensions}"
        )

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using OpenAI's API.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional parameters:
            - model: str (override default model)
            - dimensions: int (override default dimensions)

        Returns:
        --------
        List[float]
            The embedding vector
        """
        # Allow per-call overrides
        model = kwargs.get("model", self.model)
        dimensions = kwargs.get("dimensions", self.dimensions)

        # Clean the text
        text = text.replace("\n", " ")

        try:
            # For ada-002, don't pass dimensions parameter
            if model == "text-embedding-ada-002":
                response = self.client.embeddings.create(input=[text], model=model)
            else:
                response = self.client.embeddings.create(
                    input=[text], model=model, dimensions=dimensions
                )

            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings in one API request and restore input ordering."""
        if not texts:
            return []
        model = kwargs.get("model", self.model)
        dimensions = kwargs.get("dimensions", self.dimensions)
        cleaned = [str(text).replace("\n", " ") for text in texts]
        request: Dict[str, Any] = {"input": cleaned, "model": model}
        if model != "text-embedding-ada-002":
            request["dimensions"] = dimensions
        try:
            response = self.client.embeddings.create(**request)
            rows = sorted(response.data, key=lambda row: int(row.index))
            return [list(row.embedding) for row in rows]
        except Exception as e:
            logger.error("Error generating OpenAI embedding batch: %s", e)
            raise

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        return self.dimensions

    def get_default_model(self) -> str:
        """Get the default model name for this provider."""
        return self.model
