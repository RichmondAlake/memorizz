from types import SimpleNamespace

import pytest

from memorizz.embeddings import EmbeddingManager
from memorizz.embeddings.openai.provider import OpenAIEmbeddingProvider


class _BatchProvider:
    def __init__(self):
        self.calls = []

    def get_embeddings(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text))] for text in texts]


def _manager_with(provider):
    manager = object.__new__(EmbeddingManager)
    manager._provider = provider
    manager._embedding_cache = __import__("collections").OrderedDict()
    manager._embedding_cache_lock = __import__("threading").Lock()
    return manager


def test_embedding_manager_batches_only_cache_misses_and_preserves_order():
    provider = _BatchProvider()
    manager = _manager_with(provider)

    assert manager.get_embeddings(["aa", "bbb"]) == [[2.0], [3.0]]
    assert manager.get_embeddings(["bbb", "c", "aa"]) == [[3.0], [1.0], [2.0]]
    assert provider.calls == [["aa", "bbb"], ["c"]]


def test_embedding_manager_rejects_wrong_batch_length():
    provider = _BatchProvider()
    provider.get_embeddings = lambda texts: []
    manager = _manager_with(provider)

    with pytest.raises(RuntimeError, match="different batch length"):
        manager.get_embeddings(["one"])


def test_openai_batch_restores_response_index_order():
    provider = object.__new__(OpenAIEmbeddingProvider)
    provider.model = "text-embedding-3-small"
    provider.dimensions = 2
    provider.client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(
                data=[
                    SimpleNamespace(index=1, embedding=[2.0, 2.0]),
                    SimpleNamespace(index=0, embedding=[1.0, 1.0]),
                ]
            )
        )
    )

    assert provider.get_embeddings(["first", "second"]) == [
        [1.0, 1.0],
        [2.0, 2.0],
    ]
