# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import os
from typing import Any, Dict, List

import openai

from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)


class AzureOpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Azure OpenAI embedding provider implementation."""

    # Model configuration with their dimensions. This refers to the base models.
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,  # Default for 3-small is 1536, but can be reduced
        "text-embedding-3-large": 3072,  # Default for 3-large is 3072, but can be reduced
        "text-embedding-ada-002": 1536,  # Fixed dimensions
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Azure OpenAI embedding provider.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - deployment_name: str (Required. Your Azure deployment name)
            - azure_endpoint: str (Required. The endpoint URL for your Azure resource)
            - api_version: str (Required. The API version, e.g., "2023-05-15")
            - api_key: str (Required. Your Azure API key)
            - base_model: str (The underlying model type, e.g. "text-embedding-3-small". Used for validation. Defaults to "text-embedding-3-small")
            - dimensions: int (default: 256, only for text-embedding-3-* models)
        """
        super().__init__(config)

        # Azure-specific configuration
        self.deployment_name = self.config.get("deployment_name")
        self.azure_endpoint = self.config.get("azure_endpoint")
        self.api_version = self.config.get("api_version")
        self.api_key = self.config.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")

        if not all(
            [self.deployment_name, self.azure_endpoint, self.api_version, self.api_key]
        ):
            raise ValueError(
                "Configuration must include 'deployment_name', 'azure_endpoint', 'api_version', and 'api_key' for AzureOpenAIEmbeddingProvider."
            )

        # Set model and dimension configuration for validation purposes
        self.base_model = self.config.get("base_model", "text-embedding-3-small")
        self.dimensions = self.config.get("dimensions", 256)

        # Validate base_model and dimensions
        if self.base_model not in self.MODEL_DIMENSIONS:
            raise ValueError(
                f"Unsupported Azure OpenAI base_model: {self.base_model}. Supported base models: {list(self.MODEL_DIMENSIONS.keys())}"
            )

        # For ada-002, dimensions cannot be customized
        if self.base_model == "text-embedding-ada-002" and self.dimensions != 1536:
            logger.warning(
                f"Base model {self.base_model} has fixed dimensions of 1536. Ignoring custom dimensions parameter."
            )
            self.dimensions = 1536

        # For v3 models, validate dimensions are within allowed range
        if self.base_model in ["text-embedding-3-small", "text-embedding-3-large"]:
            max_dims = self.MODEL_DIMENSIONS[self.base_model]
            if self.dimensions > max_dims:
                raise ValueError(
                    f"Dimensions {self.dimensions} exceed maximum {max_dims} for base model {self.base_model}"
                )

        # Initialize Azure OpenAI client
        self.client = openai.AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            api_key=self.api_key,
        )

        logger.info(
            f"Initialized AzureOpenAI provider with deployment={self.deployment_name}, base_model={self.base_model}, dimensions={self.dimensions}"
        )

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using Azure OpenAI's API.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional parameters:
            - model: str (override default deployment name)
            - dimensions: int (override default dimensions)

        Returns:
        --------
        List[float]
            The embedding vector
        """
        # Allow per-call overrides. 'model' kwarg overrides the deployment name.
        deployment_name = kwargs.get("model", self.deployment_name)
        dimensions = kwargs.get("dimensions", self.dimensions)

        # Clean the text
        text = text.replace("\n", " ")

        try:
            # For ada-002, don't pass dimensions parameter
            # This check is based on the configured base_model, not the deployment name.
            if self.base_model == "text-embedding-ada-002":
                response = self.client.embeddings.create(
                    input=[text],
                    model=deployment_name,  # In Azure, 'model' is the deployment name
                )
            else:
                response = self.client.embeddings.create(
                    input=[text], model=deployment_name, dimensions=dimensions
                )

            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating Azure OpenAI embedding: {str(e)}")
            raise

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        return self.dimensions

    def get_default_model(self) -> str:
        """Get the default deployment name for this provider."""
        return self.deployment_name
