"""Text-only model configuration, task prompts and vector dimensions."""

import sys
from types import SimpleNamespace

import pytest

from memorizz.embeddings.huggingface.provider import HuggingFaceEmbeddingProvider


def test_text_only_model_loads_retrieval_prompts_and_consistent_truncation(monkeypatch):
    loads, encodes = [], []

    class Model:
        def __init__(self, name, **kwargs):
            loads.append((name, kwargs))
            self.dim = kwargs.get("truncate_dim", 768)

        def get_embedding_dimension(self):
            return self.dim

        def encode(self, text, **kwargs):
            encodes.append((text, kwargs))
            return SimpleNamespace(tolist=lambda: [0.0] * self.dim)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    provider = HuggingFaceEmbeddingProvider(
        {"model": "google/embeddinggemma-2", "dimensions": 256}
    )
    assert loads[0][1]["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert loads[0][1]["truncate_dim"] == 256
    assert len(provider.get_embedding("doc")) == provider.get_dimensions() == 256
    provider.get_embedding("query", input_type="query")
    assert [call[1]["prompt_name"] for call in encodes] == ["Document", "SearchQuery"]
    assert all(call[1]["normalize_embeddings"] for call in encodes)
    provider.get_embedding("explicit", prompt_name="Other")
    assert encodes[-1][1]["prompt_name"] == "Other"


def test_other_models_keep_existing_normalization_and_no_task_prompt(monkeypatch):
    calls = []

    class Model:
        def encode(self, text, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(tolist=lambda: [0.0] * 384)

    monkeypatch.setattr(HuggingFaceEmbeddingProvider, "_load_model", lambda *a: Model())
    provider = HuggingFaceEmbeddingProvider()
    provider.get_embedding("text")
    assert "prompt_name" not in calls[0]
    assert calls[0]["normalize_embeddings"] is False
    with pytest.raises(ValueError, match="dimensions"):
        HuggingFaceEmbeddingProvider(
            {"model": "google/embeddinggemma-2", "dimensions": 300}
        )
