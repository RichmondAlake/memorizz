import sys

from memorizz.benchmarks.longmemeval_v2 import (
    _clean_accessibility_tree,
    _retrieval_query,
    load_longmemeval_v2_memory_registry,
    rank_lexical_documents,
    register_longmemeval_v2_backend,
    trajectory_chunks,
    trajectory_digest,
)


def test_registry_loader_skips_unselected_optional_backends(tmp_path, monkeypatch):
    package_root = tmp_path / "memory_modules"
    package_root.mkdir()
    (package_root / "memory.py").write_text(
        "class Memory:\n"
        "    pass\n\n"
        "MEMORY_TYPES = {}\n\n"
        "def register_memory(cls):\n"
        "    MEMORY_TYPES[cls.memory_type] = cls\n"
        "    return cls\n\n"
        "def build_memory(config):\n"
        "    return MEMORY_TYPES[config['memory_type']](config['memory_params'])\n"
        "\nfrom .no_retrieval import NoRetrievalMemory\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "memory_modules.memory", raising=False)
    monkeypatch.delitem(sys.modules, "memory_modules", raising=False)

    module = load_longmemeval_v2_memory_registry(tmp_path)

    assert module.Memory.__module__ == "memory_modules.memory"
    assert module.MEMORY_TYPES == {}
    assert "memory_modules.no_retrieval" not in sys.modules


def test_trajectory_chunks_preserve_ui_labels_and_state_provenance():
    trajectory = {
        "id": "traj-1",
        "domain": "enterprise",
        "environment": "workarena",
        "goal": "Inspect filters",
        "outcome": "success",
        "states": [
            {
                "state_index": 2,
                "step": 1,
                "action": "click Filters",
                "thought": "Read all choices",
                "accessibility_tree": "\n".join(
                    [
                        "[1] generic, visible",
                        "[2] option 'Incident Mobile', selected=False",
                        "[3] StaticText 'Incident Portal'",
                    ]
                ),
            }
        ],
    }

    chunks = trajectory_chunks(trajectory, max_chars=2_000)

    assert chunks
    combined = "\n".join(chunks)
    assert "trajectory_id: traj-1" in combined
    assert "state_index: 2" in combined
    assert "Incident Mobile" in combined
    assert "Incident Portal" in combined
    assert "generic" not in combined


def test_trajectory_digest_is_bounded_and_keeps_outcome():
    trajectory = {
        "id": "traj-2",
        "goal": "Complete task",
        "outcome": "failure",
        "states": [{"step": 1, "action": "save", "thought": "permission denied"}],
    }

    digest = trajectory_digest(trajectory, max_chars=120)

    assert len(digest) <= 120
    assert "Outcome: failure" in digest


def test_accessibility_tree_budget_preserves_late_menu_options():
    tree = "\n".join(
        [f"StaticText 'prefix {index}'" for index in range(500)]
        + ["option 'Incident Portal'", "option 'My Open Incidents'"]
    )

    cleaned = _clean_accessibility_tree(tree, max_chars=1_000)

    assert len(cleaned) == 1_000
    assert "Incident Portal" in cleaned
    assert "My Open Incidents" in cleaned
    assert "middle omitted" in cleaned


def test_lexical_lane_recovers_exact_ui_label_document():
    rows = [
        {"content": "generic incident page", "trajectory_id": "a", "chunk_index": 0},
        {
            "content": "Filters dropdown option Incident Portal and My Open Incidents",
            "trajectory_id": "b",
            "chunk_index": 2,
        },
    ]

    ranked = rank_lexical_documents(
        "which Filters dropdown option labels contain Incident?", rows, limit=1
    )

    assert ranked[0]["trajectory_id"] == "b"


def test_lexical_lane_can_diversify_long_trajectory_sources():
    rows = [
        {
            "content": f"Incidents list Filters dropdown common term {index}",
            "trajectory_id": "dominant",
            "chunk_index": index,
        }
        for index in range(5)
    ] + [
        {
            "content": "Filters option Incident Portal",
            "trajectory_id": "answer-bearing",
            "chunk_index": 7,
        }
    ]

    ranked = rank_lexical_documents(
        "Incidents list Filters dropdown option Incident",
        rows,
        limit=3,
        max_per_trajectory=2,
    )

    assert len(ranked) == 3
    assert sum(row["trajectory_id"] == "dominant" for row in ranked) == 2
    assert any(row["trajectory_id"] == "answer-bearing" for row in ranked)


def test_retrieval_query_removes_evaluator_format_instruction():
    value = (
        "Which labels contain Incident?\n\n"
        "Mark your final answer (one short phrase) in \\boxed{}."
    )

    assert _retrieval_query(value) == "Which labels contain Incident?"


def test_official_backend_uses_provider_batch_storage(tmp_path, monkeypatch):
    package_root = tmp_path / "memory_modules"
    package_root.mkdir()
    (package_root / "memory.py").write_text(
        "class Memory:\n"
        "    def __init__(self, memory_params):\n"
        "        self.memory_params = memory_params\n\n"
        "MEMORY_TYPES = {}\n\n"
        "def register_memory(cls):\n"
        "    MEMORY_TYPES[cls.memory_type] = cls\n"
        "    return cls\n\n"
        "def build_memory(config):\n"
        "    return MEMORY_TYPES[config['memory_type']](config['memory_params'])\n"
        "\nfrom .no_retrieval import NoRetrievalMemory\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "memory_modules.memory", raising=False)
    monkeypatch.delitem(sys.modules, "memory_modules", raising=False)
    load_longmemeval_v2_memory_registry(tmp_path)

    class FakeEmbeddingManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_embeddings(self, texts):
            return [[float(index)] for index, _ in enumerate(texts)]

    class FakeOpenAI:
        def __init__(self, *, model, **_kwargs):
            self.model = model
            self._last_usage = {}

        def generate(self, *_args, **_kwargs):
            return "unused"

        def get_last_usage(self):
            return self._last_usage

    class FakeProvider:
        def __init__(self):
            self.batches = []

        def store_many(self, rows, memory_type, **_kwargs):
            self.batches.append((memory_type, list(rows)))
            return [str(index) for index, _ in enumerate(rows)]

        def store(self, *_args, **_kwargs):
            raise AssertionError("benchmark ingestion must use store_many")

    provider = FakeProvider()
    monkeypatch.setattr("memorizz.embeddings.EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        "memorizz.embeddings.set_global_embedding_manager", lambda _: None
    )
    monkeypatch.setattr("memorizz.llms.openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr(
        "memorizz.benchmarks.common.create_benchmark_memory_provider",
        lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    backend_type = register_longmemeval_v2_backend()
    backend = backend_type(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "embedding_batch_size": 2,
            "enable_summaries": False,
        }
    )
    backend.insert({"id": "t1", "goal": "one", "states": []})
    backend.insert({"id": "t2", "goal": "two", "states": []})

    assert len(provider.batches) == 2
    assert [len(rows) for _, rows in provider.batches] == [2, 2]
