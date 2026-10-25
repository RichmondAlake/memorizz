"""OpenAI-compatible embedding servers, and shorter Ollama vectors."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from memorizz.embeddings import (
    EmbeddingManager,
    embedding_settings_from,
    shorten_embedding,
)
from memorizz.embeddings.ollama.provider import OllamaEmbeddingProvider
from memorizz.embeddings.openai.provider import OpenAIEmbeddingProvider

pytestmark = pytest.mark.unit

LOCAL = "http://localhost:11434/v1"


class _Server:
    """Stands in for an OpenAI-compatible ``/v1/embeddings``."""

    def __init__(self, size=5, takes_dimensions=True):
        self.size = size
        self.takes_dimensions = takes_dimensions
        self.requests = []
        self.client_kwargs = {}

    def create(self, **request):
        self.requests.append(dict(request))
        if "dimensions" in request and not self.takes_dimensions:
            raise openai.BadRequestError(
                "Unrecognized request argument supplied: dimensions",
                response=httpx.Response(400, request=httpx.Request("POST", LOCAL)),
                body=None,
            )
        size = request.get("dimensions") or self.size
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=[1.0] * size)
                for i, _ in enumerate(request["input"])
            ]
        )


@pytest.fixture
def server(monkeypatch):
    server = _Server()

    def client(**kwargs):
        server.client_kwargs = kwargs
        return SimpleNamespace(embeddings=SimpleNamespace(create=server.create))

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-not-for-others")
    return server


def test_another_servers_model_needs_a_base_url(server):
    with pytest.raises(ValueError, match="set base_url"):
        OpenAIEmbeddingProvider({"model": "nomic-embed-text"})
    with pytest.raises(ValueError, match="set base_url"):
        OpenAIEmbeddingProvider(
            {"model": "nomic-embed-text", "base_url": "https://api.openai.com/v1"}
        )
    # OpenAI's own models keep their checks.
    with pytest.raises(ValueError, match="exceed maximum"):
        OpenAIEmbeddingProvider({"model": "text-embedding-3-small", "dimensions": 2000})


def test_a_compatible_server_gets_any_model_its_full_size_and_no_openai_key(server):
    provider = OpenAIEmbeddingProvider({"model": "nomic-embed-text", "base_url": LOCAL})

    assert server.client_kwargs == {"api_key": "unused", "base_url": LOCAL}
    assert provider.get_dimensions() == 5  # learned from the server
    assert provider.get_embedding("hi") == [1.0] * 5
    assert all("dimensions" not in request for request in server.requests)


def test_its_own_key_is_sent(server):
    OpenAIEmbeddingProvider(
        {"model": "bge-m3", "base_url": "https://gpu.example/v1", "api_key": "own"}
    )
    assert server.client_kwargs["api_key"] == "own"


def test_openai_base_url_in_the_environment_counts(server, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", LOCAL)
    provider = OpenAIEmbeddingProvider({"model": "nomic-embed-text"})
    assert provider.compatible and server.client_kwargs == {"api_key": "unused"}


def test_a_configured_size_is_requested(server):
    provider = OpenAIEmbeddingProvider(
        {"model": "nomic-embed-text", "base_url": LOCAL, "dimensions": 3}
    )
    assert provider.get_embeddings(["a", "b"]) == [[1.0] * 3, [1.0] * 3]
    assert server.requests[-1]["dimensions"] == 3
    assert provider.get_dimensions() == 3


def test_a_server_that_refuses_dimensions_gets_its_vectors_cut(server):
    server.takes_dimensions = False
    provider = OpenAIEmbeddingProvider(
        {"model": "bge-m3", "base_url": LOCAL, "dimensions": 4}
    )

    vector = provider.get_embedding("hi")
    assert vector == pytest.approx([0.5] * 4)  # cut to 4, unit length
    provider.get_embedding("again")
    # Asked once, then no longer.
    assert ["dimensions" in r for r in server.requests] == [True, False, False]


def test_a_server_that_ignores_dimensions_is_cut_too(server):
    server.create = lambda **request: SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[3.0, 4.0, 12.0])]
    )
    provider = OpenAIEmbeddingProvider(
        {"model": "bge-m3", "base_url": LOCAL, "dimensions": 2}
    )
    assert provider.get_embedding("hi") == pytest.approx([0.6, 0.8])


def test_a_size_larger_than_the_model_is_an_error(server):
    server.takes_dimensions = False
    provider = OpenAIEmbeddingProvider(
        {"model": "bge-m3", "base_url": LOCAL, "dimensions": 9}
    )
    with pytest.raises(ValueError, match="at most 5"):
        provider.get_embedding("hi")


def test_shorten_embedding():
    assert shorten_embedding([3.0, 4.0, 1.0], 2) == pytest.approx([0.6, 0.8])
    assert shorten_embedding([3.0, 4.0], 2) == [3.0, 4.0]
    with pytest.raises(ValueError):
        shorten_embedding([1.0], 2)


def test_settings_carry_the_server_and_its_key():
    values = {
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "openai",
        "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": "nomic-embed-text",
        "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS": "256",
        "MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL": LOCAL,
        "MEMORIZZ_DEFAULT_EMBEDDING_API_KEY": "own",
    }
    assert embedding_settings_from(values) == (
        "openai",
        {
            "model": "nomic-embed-text",
            "dimensions": 256,
            "base_url": LOCAL,
            "api_key": "own",
        },
    )
    # The key only goes with a server; other providers ignore both.
    del values["MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL"]
    assert "api_key" not in embedding_settings_from(values)[1]
    values["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] = "ollama"
    values["MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL"] = LOCAL
    assert "base_url" not in embedding_settings_from(values)[1]


def test_provider_info_leaves_keys_out(server):
    manager = EmbeddingManager(
        "openai", {"model": "bge-m3", "base_url": LOCAL, "api_key": "own"}
    )
    assert manager.get_provider_info()["config"] == {
        "model": "bge-m3",
        "base_url": LOCAL,
    }


def test_oracle_sends_a_compatible_server_its_own_key(monkeypatch):
    from memorizz.memory_provider.oracle.provider import OracleProvider

    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL", LOCAL)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-not-for-others")
    monkeypatch.delenv("MEMORIZZ_DEFAULT_EMBEDDING_API_KEY", raising=False)
    _, config = OracleProvider._resolve_embedding_defaults_from_env()
    assert config["base_url"] == LOCAL and "api_key" not in config


def test_cli_checks_the_server_url():
    from memorizz.cli.settings_commands import validate_setting

    key = "MEMORIZZ_DEFAULT_EMBEDDING_BASE_URL"
    assert validate_setting(key, " http://localhost:11434/v1 ") == LOCAL
    assert validate_setting(key, "") == ""
    with pytest.raises(ValueError, match="http"):
        validate_setting(key, "localhost:11434")


class _Ollama:
    def __init__(self, size=8, takes_dimensions=True):
        self.size = size
        self.takes_dimensions = takes_dimensions
        self.calls = []

    def embed(self, model, input, **kwargs):
        self.calls.append(("embed", kwargs))
        if "dimensions" in kwargs and not self.takes_dimensions:
            raise TypeError("embed() got an unexpected keyword argument 'dimensions'")
        size = kwargs.get("dimensions") or self.size
        if size > self.size:
            raise RuntimeError(
                f"dimensions {size} exceeds embedding length {self.size} (status code: 400)"
            )
        return SimpleNamespace(embeddings=[[2.0] * size])

    def embeddings(self, model, prompt):
        self.calls.append(("embeddings", {}))
        return SimpleNamespace(embedding=[2.0] * self.size)


def _ollama(monkeypatch, client, **config):
    monkeypatch.setattr(
        OllamaEmbeddingProvider,
        "_init_client",
        lambda self: setattr(self, "client", client),
    )
    return OllamaEmbeddingProvider(config)


def test_ollama_asks_for_fewer_dimensions(monkeypatch):
    client = _Ollama()
    provider = _ollama(monkeypatch, client, dimensions=4)
    assert provider.get_dimensions() == 4
    assert provider.get_embedding("hi") == [2.0] * 4
    assert client.calls[-1] == ("embed", {"dimensions": 4})


def test_an_older_ollama_package_gets_vectors_cut_here(monkeypatch):
    client = _Ollama(takes_dimensions=False)
    provider = _ollama(monkeypatch, client, dimensions=4)
    assert provider.get_embedding("hi") == pytest.approx([0.5] * 4)


def test_ollama_says_when_the_model_is_smaller(monkeypatch):
    provider = _ollama(monkeypatch, _Ollama(), dimensions=16)
    with pytest.raises(ValueError, match="cannot shorten nomic-embed-text to 16"):
        provider.get_dimensions()


def test_ollama_without_a_size_is_unchanged(monkeypatch):
    client = _Ollama()
    provider = _ollama(monkeypatch, client)
    assert provider.get_embedding("hi") == [2.0] * 8
    assert provider.get_dimensions() == 8
    assert {name for name, _ in client.calls} == {"embeddings"}


class _ShortOllama(_Ollama):
    """/api/embeddings refuses long text, as Ollama does past the context."""

    def embeddings(self, model, prompt):
        if len(prompt) > 100:
            self.calls.append(("embeddings", {}))
            raise RuntimeError(
                "the input length exceeds the context length (status code: 500)"
            )
        return super().embeddings(model, prompt)


def test_ollama_embeds_the_start_of_text_past_the_context(monkeypatch):
    client = _ShortOllama()
    provider = _ollama(monkeypatch, client)
    assert provider.get_embedding("short") == [2.0] * 8
    assert provider.get_embedding("long " * 100) == [2.0] * 8
    assert client.calls[-2:] == [("embeddings", {}), ("embed", {"truncate": True})]


def test_ollama_still_raises_other_errors(monkeypatch):
    class _Down(_Ollama):
        def embeddings(self, model, prompt):
            raise ConnectionError("Failed to connect to Ollama")

    provider = _ollama(monkeypatch, _Down())
    provider._dimensions_probed = True
    with pytest.raises(ConnectionError):
        provider.get_embedding("hi")
