# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import logging
from typing import Any, Dict, List, Optional

from ...llms._hf_offline import enable_hf_offline_env, is_hf_offline
from .. import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by Hugging Face models.

    Uses `sentence-transformers` under the hood so any SentenceTransformer
    compatible repository (local or remote) can be loaded.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # Common community models with their known embedding sizes
    MODEL_DIMENSIONS = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat/multilingual-e5-small": 384,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = self.config.get("model", self.DEFAULT_MODEL)
        self.device = self.config.get(
            "device"
        )  # e.g. "cpu", "cuda", "mps", or int gpu id
        self.batch_size = self.config.get("batch_size", 32)
        self.normalize_embeddings = self.config.get("normalize_embeddings", False)
        self.cache_folder = self.config.get("cache_folder")
        self.revision = self.config.get("revision")
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        self.auth_token = self.config.get("auth_token")
        self._dimensions_override = self.config.get("dimensions")

        local_only = self.config.get("local_files_only")
        if local_only is None:
            local_only = is_hf_offline()
        if local_only:
            enable_hf_offline_env()
        self.local_files_only = bool(local_only)

        self._model_cache: Dict[str, Any] = {}
        self._dimensions_cache: Dict[str, int] = {}

        self._model_cache[self.model_name] = self._load_model(self.model_name)
        self._dimensions_cache[self.model_name] = self._infer_dimensions(
            self.model_name
        )

        logger.info(
            "Initialized HuggingFaceEmbeddingProvider model=%s dimensions=%s",
            self.model_name,
            self._dimensions_cache[self.model_name],
        )

    def _load_model(self, model_name: str):
        """Instantiate a SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the Hugging Face embedding "
                "provider. Install it via `pip install memorizz[huggingface]`."
            ) from exc

        kwargs: Dict[str, Any] = {
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.cache_folder:
            kwargs["cache_folder"] = self.cache_folder
        if self.revision:
            kwargs["revision"] = self.revision
        if self.auth_token:
            kwargs["use_auth_token"] = self.auth_token
        if self.local_files_only:
            kwargs["local_files_only"] = True

        try:
            model = SentenceTransformer(model_name, **kwargs)
        except OSError as exc:
            if self.local_files_only:
                raise OSError(
                    f"HuggingFace embedding model '{model_name}' is not cached "
                    "locally and the host appears to be offline. Connect to "
                    "the internet to download it, or pick a model already in "
                    "the local cache."
                ) from exc
            raise
        return model

    def _infer_dimensions(self, model_name: str) -> int:
        """Infer embedding size using cache, override, or provider defaults."""
        if self._dimensions_override:
            return int(self._dimensions_override)

        if model_name in self._dimensions_cache:
            return self._dimensions_cache[model_name]

        if model_name in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[model_name]

        model = self._model_cache.get(model_name)
        if model is None:
            model = self._load_model(model_name)
            self._model_cache[model_name] = model

        try:
            dims = int(model.get_sentence_embedding_dimension())
            self._dimensions_cache[model_name] = dims
            return dims
        except Exception as exc:
            logger.warning(
                "Could not infer embedding size for model %s: %s. Falling back to 768.",
                model_name,
                exc,
            )
            return 768

    def _get_model(self, model_name: Optional[str] = None):
        """Return a cached model or load it if needed."""
        target_model = model_name or self.model_name
        if target_model not in self._model_cache:
            self._model_cache[target_model] = self._load_model(target_model)
            self._dimensions_cache[target_model] = self._infer_dimensions(target_model)
        return self._model_cache[target_model]

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        """Generate an embedding for the provided text."""
        clean_text = text.replace("\n", " ")
        model_name = kwargs.get("model", self.model_name)
        model = self._get_model(model_name)

        normalize = kwargs.get("normalize_embeddings", self.normalize_embeddings)

        try:
            embedding = model.encode(
                clean_text,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as exc:
            logger.error(
                "Error generating Hugging Face embedding with model %s: %s",
                model_name,
                exc,
            )
            raise

    def get_dimensions(self) -> int:
        """Return the embedding dimension for the configured model."""
        return self._infer_dimensions(self.model_name)

    def get_default_model(self) -> str:
        """Return the default model identifier."""
        return self.model_name
