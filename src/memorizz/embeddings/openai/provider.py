# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import openai

from .. import BaseEmbeddingProvider, shorten_embedding

logger = logging.getLogger(__name__)

# Suppress httpx logs to reduce noise from API requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# The client needs a key; local OpenAI-compatible servers ignore it.
_NO_KEY = "unused"


def _compatible_endpoint(base_url: Optional[str]) -> bool:
    """A base URL that is not OpenAI's own API."""
    if not base_url:
        return False
    host = (urlparse(str(base_url)).hostname or "").lower()
    return host != "api.openai.com"


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider, also for OpenAI-compatible servers.

    With a ``base_url`` (or ``OPENAI_BASE_URL``) that is not OpenAI's API,
    any model the server serves is accepted: Ollama's ``/v1``, vLLM, LM
    Studio and the like.
    """

    # Model configuration with their dimensions
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,  # Default for 3-small is 1536, but can be reduced
        "text-embedding-3-large": 3072,  # Default for 3-large is 3072, but can be reduced
        "text-embedding-ada-002": 1536,  # Fixed dimensions
    }

    # A model other than OpenAI's, served by a compatible server.
    compatible = False
    # Whether that server is asked for ``dimensions`` (it may refuse).
    _send_dimensions = True

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize OpenAI embedding provider.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - model: str (default: "text-embedding-3-small")
            - dimensions: int (default: 256 for text-embedding-3-* models;
              for another server's model, its full size unless set)
            - api_key: str (optional; OpenAI's models use OPENAI_API_KEY when
              absent, another server's models are sent no key)
            - base_url: str (optional, for custom endpoints and
              OpenAI-compatible servers such as http://localhost:11434/v1)
        """
        super().__init__(config)

        # Set default configuration
        self.model = self.config.get("model", "text-embedding-3-small")
        base_url = self.config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.compatible = self.model not in self.MODEL_DIMENSIONS

        if self.compatible:
            if not _compatible_endpoint(base_url):
                raise ValueError(
                    f"Unsupported OpenAI model: {self.model}. Supported models: "
                    f"{list(self.MODEL_DIMENSIONS.keys())}. For an OpenAI-compatible "
                    "server (Ollama, vLLM, LM Studio), set base_url."
                )
            configured = self.config.get("dimensions")
            # Unset: the server's full size, learned from its first answer.
            self.dimensions = int(configured) if configured else None
            self._send_dimensions = self.dimensions is not None
        else:
            self.dimensions = self.config.get("dimensions", 256)

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
        elif self.compatible:
            # Never send the OpenAI key to another server.
            client_kwargs["api_key"] = _NO_KEY
        if "base_url" in self.config:
            client_kwargs["base_url"] = self.config["base_url"]

        self.client = openai.OpenAI(**client_kwargs)

        logger.info(
            f"Initialized OpenAI provider with model={self.model}, dimensions={self.dimensions}"
        )

    def _embed(
        self, texts: List[str], model: str, dimensions: Optional[int]
    ) -> List[List[float]]:
        """One request for ``texts``, in input order, ``dimensions`` long."""
        openai_model = model in self.MODEL_DIMENSIONS
        request: Dict[str, Any] = {"input": texts, "model": model}
        if dimensions is not None and (
            model != "text-embedding-ada-002" if openai_model else self._send_dimensions
        ):
            request["dimensions"] = dimensions
        try:
            response = self.client.embeddings.create(**request)
        except openai.BadRequestError:
            if openai_model or "dimensions" not in request:
                raise
            # The server does not take ``dimensions``: take the full vectors
            # and cut them here, as servers that do take it would.
            del request["dimensions"]
            response = self.client.embeddings.create(**request)
            logger.info(
                "%s does not take an embedding size; cutting its vectors to %s",
                model,
                dimensions,
            )
            self._send_dimensions = False
        rows = sorted(response.data, key=lambda row: getattr(row, "index", 0) or 0)
        vectors = [list(row.embedding) for row in rows]
        if openai_model:
            return vectors
        if dimensions is None:
            if vectors and self.dimensions is None and model == self.model:
                self.dimensions = len(vectors[0])
            return vectors
        return [shorten_embedding(vector, dimensions) for vector in vectors]

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
            return self._embed([text], model, dimensions)[0]
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
        try:
            return self._embed(cleaned, model, dimensions)
        except Exception as e:
            logger.error("Error generating OpenAI embedding batch: %s", e)
            raise

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        if self.dimensions is None:
            # Another server's model at its full size: ask it once.
            self.get_embedding("test")
        return self.dimensions

    def get_default_model(self) -> str:
        """Get the default model name for this provider."""
        return self.model
