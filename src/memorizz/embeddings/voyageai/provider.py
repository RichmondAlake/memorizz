# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
from typing import Any, Dict, List, Union

from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class VoyageAIEmbeddingProvider(BaseEmbeddingProvider):
    """VoyageAI embedding provider implementation with support for text, multimodal, and contextualized embeddings."""

    # Model configuration with their default dimensions and supported output dimensions
    TEXT_MODELS = {
        # Voyage 3 series with configurable dimensions
        "voyage-3-large": {
            "default_dimensions": 1024,
            "supported_dimensions": [256, 512, 1024, 2048],
            "context_length": 32000,
            "description": "Best general-purpose and multilingual retrieval quality",
        },
        "voyage-3.5": {
            "default_dimensions": 1024,
            "supported_dimensions": [256, 512, 1024, 2048],
            "context_length": 32000,
            "description": "Optimized for general-purpose and multilingual retrieval quality",
        },
        "voyage-3.5-lite": {
            "default_dimensions": 1024,
            "supported_dimensions": [256, 512, 1024, 2048],
            "context_length": 32000,
            "description": "Optimized for latency and cost",
        },
        "voyage-code-3": {
            "default_dimensions": 1024,
            "supported_dimensions": [256, 512, 1024, 2048],
            "context_length": 32000,
            "description": "Optimized for code retrieval",
        },
        # Fixed dimension models
        "voyage-finance-2": {
            "default_dimensions": 1024,
            "supported_dimensions": [1024],
            "context_length": 32000,
            "description": "Optimized for finance retrieval and RAG",
        },
        "voyage-law-2": {
            "default_dimensions": 1024,
            "supported_dimensions": [1024],
            "context_length": 16000,
            "description": "Optimized for legal retrieval and RAG",
        },
        "voyage-code-2": {
            "default_dimensions": 1536,
            "supported_dimensions": [1536],
            "context_length": 16000,
            "description": "Previous generation code embeddings",
        },
    }

    MULTIMODAL_MODELS = {
        "voyage-multimodal-3": {
            "default_dimensions": 1024,
            "supported_dimensions": [1024],
            "context_length": 32000,
            "description": "Rich multimodal embedding model for text and images",
        }
    }

    CONTEXTUALIZED_MODELS = {
        "voyage-context-3": {
            "default_dimensions": 1024,
            "supported_dimensions": [256, 512, 1024, 2048],
            "context_length": 32000,
            "description": "Contextualized chunk embeddings for documents",
        }
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize VoyageAI embedding provider.

        Parameters:
        -----------
        config : Dict[str, Any]
            Configuration dictionary with keys:
            - model: str (default: "voyage-3.5")
            - output_dimension: int (default: model's default, must be supported)
            - output_dtype: str (default: "float", options: "float", "int8", "uint8", "binary", "ubinary")
            - input_type: str (optional, options: "query", "document", None)
            - api_key: str (optional, uses env var VOYAGE_API_KEY if not provided)
            - base_url: str (optional, for custom endpoints)
            - embedding_type: str (default: "text", options: "text", "multimodal", "contextualized")
        """
        super().__init__(config)

        # Set default configuration
        self.embedding_type = self.config.get("embedding_type", "text")
        self.model = self.config.get("model", self._get_default_model())
        self.output_dtype = self.config.get("output_dtype", "float")
        self.input_type = self.config.get("input_type", None)

        # Validate embedding type and model
        self._validate_model_and_type()

        # Set dimensions based on model and user preference
        model_info = self._get_model_info()
        requested_dimensions = self.config.get("output_dimension")

        if requested_dimensions is not None:
            if requested_dimensions not in model_info["supported_dimensions"]:
                raise ValueError(
                    f"Dimensions {requested_dimensions} not supported for model {self.model}. "
                    f"Supported dimensions: {model_info['supported_dimensions']}"
                )
            self.dimensions = requested_dimensions
        else:
            self.dimensions = model_info["default_dimensions"]

        # Validate output dtype
        self._validate_output_dtype()

        # Initialize VoyageAI client
        self._init_client()

        logger.info(
            f"Initialized VoyageAI provider: model={self.model}, dimensions={self.dimensions}, type={self.embedding_type}"
        )

    def _get_default_model(self) -> str:
        """Get default model based on embedding type."""
        if self.embedding_type == "multimodal":
            return "voyage-multimodal-3"
        elif self.embedding_type == "contextualized":
            return "voyage-context-3"
        else:
            return "voyage-3.5"

    def _validate_model_and_type(self):
        """Validate that the model is compatible with the embedding type."""
        if self.embedding_type == "text":
            if self.model not in self.TEXT_MODELS:
                raise ValueError(
                    f"Model {self.model} not supported for text embeddings. Available: {list(self.TEXT_MODELS.keys())}"
                )
        elif self.embedding_type == "multimodal":
            if self.model not in self.MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} not supported for multimodal embeddings. Available: {list(self.MULTIMODAL_MODELS.keys())}"
                )
        elif self.embedding_type == "contextualized":
            if self.model not in self.CONTEXTUALIZED_MODELS:
                raise ValueError(
                    f"Model {self.model} not supported for contextualized embeddings. Available: {list(self.CONTEXTUALIZED_MODELS.keys())}"
                )
        else:
            raise ValueError(f"Unsupported embedding type: {self.embedding_type}")

    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information based on embedding type."""
        if self.embedding_type == "text":
            return self.TEXT_MODELS[self.model]
        elif self.embedding_type == "multimodal":
            return self.MULTIMODAL_MODELS[self.model]
        elif self.embedding_type == "contextualized":
            return self.CONTEXTUALIZED_MODELS[self.model]

    def _validate_output_dtype(self):
        """Validate output data type support."""
        supported_dtypes = ["float"]

        # Advanced dtypes are supported by newer models
        if self.model in [
            "voyage-3-large",
            "voyage-3.5",
            "voyage-3.5-lite",
            "voyage-code-3",
            "voyage-context-3",
        ]:
            supported_dtypes.extend(["int8", "uint8", "binary", "ubinary"])

        if self.output_dtype not in supported_dtypes:
            raise ValueError(
                f"Output dtype '{self.output_dtype}' not supported for model {self.model}. Supported: {supported_dtypes}"
            )

    def _init_client(self):
        """Initialize the VoyageAI client."""
        try:
            import voyageai

            # Initialize client with API key
            api_key = self.config.get("api_key")
            if api_key:
                self.client = voyageai.Client(api_key=api_key)
            else:
                # Uses VOYAGE_API_KEY environment variable
                self.client = voyageai.Client()

        except ImportError:
            raise ImportError(
                "voyageai package is required for VoyageAI embeddings. "
                "Install it with: pip install voyageai"
            )

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """
        Generate embedding using VoyageAI.

        Parameters:
        -----------
        text : str
            The text to embed
        **kwargs
            Additional parameters:
            - model: str (override default model)
            - output_dimension: int (override default dimensions)
            - output_dtype: str (override default output type)
            - input_type: str (override default input type)

        Returns:
        --------
        List[float]
            The embedding vector
        """
        # Allow per-call overrides
        model = kwargs.get("model", self.model)
        output_dimension = kwargs.get("output_dimension", self.dimensions)
        output_dtype = kwargs.get("output_dtype", self.output_dtype)
        input_type = kwargs.get("input_type", self.input_type)

        try:
            # Prepare parameters for the API call
            embed_params = {
                "texts": [text],
                "model": model,
                "output_dtype": output_dtype,
            }

            # Add optional parameters
            if input_type is not None:
                embed_params["input_type"] = input_type

            if output_dimension != self._get_model_info()["default_dimensions"]:
                embed_params["output_dimension"] = output_dimension

            # Generate embedding
            if self.embedding_type == "text":
                result = self.client.embed(**embed_params)
            elif self.embedding_type == "multimodal":
                # For multimodal, convert text to the expected format
                result = self.client.multimodal_embed(
                    inputs=[[text]], model=model, input_type=input_type
                )
            elif self.embedding_type == "contextualized":
                # For contextualized, wrap in the expected format
                result = self.client.contextualized_embed(
                    inputs=[[text]],
                    model=model,
                    input_type=input_type,
                    output_dimension=output_dimension,
                    output_dtype=output_dtype,
                )
                return result.results[0].embeddings[0]

            return result.embeddings[0]

        except Exception as e:
            logger.error(f"Error generating VoyageAI embedding: {str(e)}")
            raise

    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Parameters:
        -----------
        texts : List[str]
            The texts to embed
        **kwargs
            Additional parameters (same as get_embedding)

        Returns:
        --------
        List[List[float]]
            List of embedding vectors
        """
        # Allow per-call overrides
        model = kwargs.get("model", self.model)
        output_dimension = kwargs.get("output_dimension", self.dimensions)
        output_dtype = kwargs.get("output_dtype", self.output_dtype)
        input_type = kwargs.get("input_type", self.input_type)

        try:
            # Prepare parameters for the API call
            embed_params = {
                "texts": texts,
                "model": model,
                "output_dtype": output_dtype,
            }

            # Add optional parameters
            if input_type is not None:
                embed_params["input_type"] = input_type

            if output_dimension != self._get_model_info()["default_dimensions"]:
                embed_params["output_dimension"] = output_dimension

            # Generate embeddings
            if self.embedding_type == "text":
                result = self.client.embed(**embed_params)
                return result.embeddings
            elif self.embedding_type == "multimodal":
                # For multimodal, convert texts to the expected format
                inputs = [[text] for text in texts]
                result = self.client.multimodal_embed(
                    inputs=inputs, model=model, input_type=input_type
                )
                return result.embeddings
            elif self.embedding_type == "contextualized":
                # For contextualized, wrap each text in the expected format
                inputs = [[text] for text in texts]
                result = self.client.contextualized_embed(
                    inputs=inputs,
                    model=model,
                    input_type=input_type,
                    output_dimension=output_dimension,
                    output_dtype=output_dtype,
                )
                return [res.embeddings[0] for res in result.results]

        except Exception as e:
            logger.error(f"Error generating VoyageAI embeddings: {str(e)}")
            raise

    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings produced by this provider."""
        return self.dimensions

    def get_default_model(self) -> str:
        """Get the default model name for this provider."""
        return self.model

    def get_embedding_type(self) -> str:
        """Get the embedding type (text, multimodal, contextualized)."""
        return self.embedding_type

    @classmethod
    def get_available_models(cls, embedding_type: str = "text") -> List[str]:
        """Get list of available VoyageAI models for a specific embedding type."""
        if embedding_type == "text":
            return list(cls.TEXT_MODELS.keys())
        elif embedding_type == "multimodal":
            return list(cls.MULTIMODAL_MODELS.keys())
        elif embedding_type == "contextualized":
            return list(cls.CONTEXTUALIZED_MODELS.keys())
        else:
            # Return all models
            return (
                list(cls.TEXT_MODELS.keys())
                + list(cls.MULTIMODAL_MODELS.keys())
                + list(cls.CONTEXTUALIZED_MODELS.keys())
            )

    @classmethod
    def get_model_info(cls, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        all_models = {
            **cls.TEXT_MODELS,
            **cls.MULTIMODAL_MODELS,
            **cls.CONTEXTUALIZED_MODELS,
        }
        return all_models.get(model, {})

    @classmethod
    def get_supported_dimensions(cls, model: str) -> List[int]:
        """Get supported dimensions for a specific model."""
        model_info = cls.get_model_info(model)
        return model_info.get("supported_dimensions", [])

    def multimodal_embed(self, inputs: List[List[Union[str, Any]]], **kwargs):
        """
        Generate multimodal embeddings for inputs containing text and images.

        Parameters:
        -----------
        inputs : List[List[Union[str, PIL.Image.Image]]]
            List of multimodal inputs (text and images)
        **kwargs
            Additional parameters

        Returns:
        --------
        Multimodal embeddings result
        """
        if self.embedding_type != "multimodal":
            raise ValueError("multimodal_embed() requires embedding_type='multimodal'")

        try:
            return self.client.multimodal_embed(
                inputs=inputs,
                model=kwargs.get("model", self.model),
                input_type=kwargs.get("input_type", self.input_type),
            )
        except Exception as e:
            logger.error(f"Error generating VoyageAI multimodal embeddings: {str(e)}")
            raise

    def contextualized_embed(self, inputs: List[List[str]], **kwargs):
        """
        Generate contextualized embeddings for document chunks.

        Parameters:
        -----------
        inputs : List[List[str]]
            List of document chunks
        **kwargs
            Additional parameters

        Returns:
        --------
        Contextualized embeddings result
        """
        if self.embedding_type != "contextualized":
            raise ValueError(
                "contextualized_embed() requires embedding_type='contextualized'"
            )

        try:
            return self.client.contextualized_embed(
                inputs=inputs,
                model=kwargs.get("model", self.model),
                input_type=kwargs.get("input_type", self.input_type),
                output_dimension=kwargs.get("output_dimension", self.dimensions),
                output_dtype=kwargs.get("output_dtype", self.output_dtype),
            )
        except Exception as e:
            logger.error(
                f"Error generating VoyageAI contextualized embeddings: {str(e)}"
            )
            raise
