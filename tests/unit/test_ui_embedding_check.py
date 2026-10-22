"""At connect, the store's vector size is compared with the UI's embedder:
the store's recorded embedder is adopted when it can run here, else the UI
warns. Nothing is written or re-embedded, and no embedding API is called."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from memorizz import embeddings  # noqa: E402
from memorizz.enums.memory_type import MemoryType  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import embedding_check, state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.security import ReadOnlyProviderProxy  # noqa: E402

pytestmark = pytest.mark.unit

SPEC_EXAMPLE = (
    "This store holds 256-dimension vectors from openai text-embedding-3-small; "
    "the UI is configured for ollama nomic-embed-text (768). Searches will not "
    "match until you set OPENAI_API_KEY, then connect again. Nothing was "
    "re-embedded."
)


class Inner:
    """An embedding provider that must never be asked to embed or probe."""

    def __init__(self, model, dimensions):
        self.model = model
        self.dimensions = dimensions

    def get_dimensions(self):  # pragma: no cover - the check never probes
        raise AssertionError("probed the embedder")

    def get_embedding(self, *_args, **_kwargs):  # pragma: no cover
        raise AssertionError("called the embedder")


def manager(name, model, dimensions):
    return SimpleNamespace(
        provider_type=SimpleNamespace(value=name), _provider=Inner(model, dimensions)
    )


class FakeOracle:
    """Declares VECTOR sizes the way OracleProvider does."""

    def __init__(self, declared, embedder, agents=()):
        self.declared = declared
        self._embedding_provider = embedder
        self.agents = list(agents)

    def get_vector_schema_dimensions(self):
        return dict(self.declared)

    def list_all(self, memory_type):
        assert memory_type == MemoryType.MEMAGENT
        return list(self.agents)


@pytest.fixture(autouse=True)
def keep_global_embedder(monkeypatch):
    monkeypatch.setattr(embeddings, "_global_embedding_manager", None)


def test_oracle_mismatch_without_records_says_what_to_set(monkeypatch):
    provider = FakeOracle(
        {"CONVERSATION_MEMORY.EMBEDDING": 256, "KNOWLEDGE_BASE.EMBEDDING": 256},
        manager("ollama", "nomic-embed-text", 768),
    )
    result = embedding_check.check_store_embeddings(provider, "oracle")
    assert result["status"] == "mismatch"
    assert result["source"] == "Oracle VECTOR columns"
    assert result["dimensions_by_store"] == {
        "conversation_memory": 256,
        "knowledge_base": 256,
    }
    assert result["message"] == (
        "This store holds 256-dimension vectors; the UI is configured for ollama "
        "nomic-embed-text (768). Searches will not match until you set "
        "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER, MEMORIZZ_DEFAULT_EMBEDDING_MODEL and "
        "MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS to the model that wrote them "
        "(256 dimensions), then connect again. Nothing was re-embedded."
    )


def test_recorded_embedder_without_its_key_is_named(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agents = [
        {
            "agent_id": "a1",
            "embedding_provider": "openai",
            "embedding_config": {"model": "text-embedding-3-small", "dimensions": 256},
        }
    ]
    provider = FakeOracle(
        {"CONVERSATION_MEMORY.EMBEDDING": 256},
        manager("ollama", "nomic-embed-text", 768),
        agents,
    )
    result = embedding_check.check_store_embeddings(provider, "oracle")
    assert result["status"] == "mismatch"
    assert result["message"] == SPEC_EXAMPLE
    # Nothing was swapped in.
    assert provider._embedding_provider._provider.model == "nomic-embed-text"


def test_recorded_embedder_is_adopted_when_it_can_run(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    agents = [
        {
            "embedding_provider": "openai",
            "embedding_config": {"model": "text-embedding-3-small", "dimensions": 256},
        },
        {"embedding_provider": None},
    ]
    original = manager("ollama", "nomic-embed-text", 768)
    provider = FakeOracle({"CONVERSATION_MEMORY.EMBEDDING": 256}, original, agents)
    result = embedding_check.check_store_embeddings(
        ReadOnlyProviderProxy(provider), "oracle"
    )
    assert result["status"] == "adopted"
    assert result["adopted"] == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 256,
    }
    assert result["message"].startswith(
        "This store holds 256-dimension vectors from openai text-embedding-3-small "
        "(256), so the UI searches with it instead of ollama nomic-embed-text (768)"
    )
    # The provider under the read-only proxy searches with it now.
    adopted = provider._embedding_provider
    assert adopted is not original and adopted.get_dimensions() == 256
    assert embeddings._global_embedding_manager is adopted


def test_same_size_but_a_different_recorded_model_is_still_a_mismatch(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    agents = [
        {
            "embedding_provider": "voyageai",
            "embedding_config": {"model": "voyage-3", "output_dimension": 256},
        }
    ]
    provider = FakeOracle(
        {"SUMMARIES.EMBEDDING": 256}, manager("ollama", "nomic-embed-text", 256), agents
    )
    result = embedding_check.check_store_embeddings(provider, "oracle")
    assert result["status"] == "mismatch"
    assert "from voyageai voyage-3" in result["message"]


def test_matching_store_and_empty_store(monkeypatch):
    provider = FakeOracle(
        {"CONVERSATION_MEMORY.EMBEDDING": 768},
        manager("ollama", "nomic-embed-text", 768),
    )
    assert (
        embedding_check.check_store_embeddings(provider, "oracle")["status"] == "match"
    )
    empty = FakeOracle({}, manager("ollama", "nomic-embed-text", 768))
    assert embedding_check.check_store_embeddings(empty, "oracle")["status"] == "empty"


def test_filesystem_store_reads_one_stored_embedding_per_store(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "store", lazy_vector_indexes=True)
    )
    provider.store(
        {"content": "remembered", "memory_id": "m1", "embedding": [0.1, 0.2, 0.3]},
        MemoryType.CONVERSATION_MEMORY,
    )
    sizes, source = embedding_check.store_dimensions(provider)
    assert sizes == {"conversation_memory": 3} and source == "stored embeddings"
    # The test suite's stub embedder makes 1-dimension vectors.
    result = embedding_check.check_store_embeddings(provider, "filesystem")
    assert result["status"] == "mismatch"
    assert result["message"].startswith(
        "This store holds 3-dimension vectors; the UI is configured for test test-stub (1)."
    )


class FakeCollection:
    def __init__(self, name, indexes=None, sample=None, fail_indexes=False):
        self.name = name
        self.indexes = indexes or []
        self.sample = sample
        self.fail_indexes = fail_indexes

    def list_search_indexes(self):
        if self.fail_indexes:
            raise RuntimeError("not Atlas")
        return list(self.indexes)

    def aggregate(self, pipeline, maxTimeMS=None):
        assert maxTimeMS and pipeline[1] == {"$limit": 1}
        return [{"size": self.sample}] if self.sample else []


class FakeDB(dict):
    def list_collection_names(self):
        return list(self)


def test_mongodb_reads_vector_indexes_then_stored_embeddings():
    db = FakeDB(
        {
            "knowledge_base": FakeCollection(
                "knowledge_base",
                indexes=[
                    {
                        "name": "vector_index",
                        "type": "vectorSearch",
                        "latestDefinition": {
                            "fields": [
                                {
                                    "type": "vector",
                                    "path": "embedding",
                                    "numDimensions": 1024,
                                },
                                {"type": "filter", "path": "memory_id"},
                            ]
                        },
                    }
                ],
            ),
            "conversation_memory": FakeCollection(
                "conversation_memory", sample=1024, fail_indexes=True
            ),
            "unrelated": FakeCollection("unrelated", sample=3),
        }
    )
    provider = SimpleNamespace(db=db, client=object(), _embedding_provider=None)
    sizes, source = embedding_check.store_dimensions(provider)
    assert sizes == {"knowledge_base": 1024, "conversation_memory": 1024}
    assert source == "stored embeddings and vector search indexes"


def test_the_oracle_providers_own_refusal_reads_the_same_way():
    message = (
        "Oracle VECTOR dimension mismatch: the configured embedding provider outputs "
        "384 dimensions, but schema EVAL declares CONVERSATION_MEMORY.EMBEDDING=256, "
        "KNOWLEDGE_BASE.EMBEDDING=256. Use a fresh schema or migrate every affected "
        "VECTOR column; Oracle cannot reshape existing vector columns automatically."
    )
    assert embedding_check.oracle_mismatch_error(message) == (
        "This store holds 256-dimension vectors; the UI is configured for oracle "
        "in-database ALL_MINILM_L12_V2 (384). The UI did not connect. Searches will "
        "not match until you set MEMORIZZ_ORACLE_IN_DATABASE_EMBEDDING=false and "
        "set MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER, MEMORIZZ_DEFAULT_EMBEDDING_MODEL "
        "and MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS to the model that wrote them "
        "(256 dimensions), then connect again. Nothing was re-embedded."
    )
    assert embedding_check.oracle_mismatch_error("ORA-12541: no listener") is None


def test_the_dashboard_and_header_show_the_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "store", lazy_vector_indexes=True)
    )
    check = {"status": "mismatch", "message": SPEC_EXAMPLE}
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"path": str(tmp_path / "store")},
            "embedding_check": check,
        },
    ):
        client = TestClient(create_app())
        dashboard = client.get("/dashboard").text
        connect = client.get("/connect").text
    expected = SPEC_EXAMPLE.replace("'", "&#39;")
    assert f'id="embedding-check">{expected}</p>' in dashboard
    assert 'class="ui-conn-warn" href="/connect#embedding-check"' in dashboard
    assert f'id="embedding-check">{expected}</div>' in connect
    assert "Connected to <strong>FileSystem</strong>" in connect


def test_connecting_records_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORIZZ_UI_READ_ONLY", raising=False)
    with patch.dict(state._state, {"provider": None, "provider_type": None}):
        client = TestClient(create_app(), follow_redirects=False)
        with patch.object(
            embedding_check,
            "check_store_embeddings",
            return_value={"status": "mismatch", "message": "differs"},
        ) as check:
            response = client.post(
                "/connect",
                data={
                    "provider_type": "filesystem",
                    "filesystem_path": str(tmp_path / "s"),
                },
            )
        assert response.status_code == 302
        assert check.call_args.args[1] == "filesystem"
        assert state._state["embedding_check"]["message"] == "differs"
