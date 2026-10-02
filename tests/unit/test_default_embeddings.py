"""A local embedding default, and turns that survive a missing provider."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_embedding_provider_comes_from_the_environment(monkeypatch):
    from memorizz import embeddings

    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS", "768")
    manager = embeddings._default_embedding_manager()
    assert manager.provider_type == embeddings.EmbeddingProvider.OLLAMA
    assert manager.config == {"model": "nomic-embed-text", "dimensions": 768}

    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "not-a-provider")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fallback = embeddings._default_embedding_manager()
    assert fallback.provider_type == embeddings.EmbeddingProvider.OPENAI


def test_workflow_is_kept_without_an_embedding_when_none_is_available(monkeypatch):
    from memorizz.long_term.procedural.workflow import workflow as module

    def no_provider(text):
        raise RuntimeError("Missing credentials")

    monkeypatch.setattr(module, "get_embedding", no_provider)
    captured = module.Workflow(name="turn", description="question")
    assert captured.embedding is None


def test_ollama_embeddings_use_ollama_host(monkeypatch):
    """Like the Ollama LLM provider: OLLAMA_HOST, unless base_url is given."""
    pytest.importorskip("ollama")
    from memorizz.embeddings.ollama.provider import OllamaEmbeddingProvider

    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.internal:11434")
    assert OllamaEmbeddingProvider({}).base_url == "http://ollama.internal:11434"
    given = OllamaEmbeddingProvider({"base_url": "http://other:11434"})
    assert given.base_url == "http://other:11434"
    monkeypatch.delenv("OLLAMA_HOST")
    assert OllamaEmbeddingProvider({}).base_url == "http://localhost:11434"
