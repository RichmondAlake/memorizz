"""Putting fine-tuned models to work: routing writes, reranking recall,
swapping embedding models (with re-embedding) and the comparison cards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.finetune import FineTuning, FineTuningError, routing
from memorizz.finetune import swap as swapping
from memorizz.finetune.models import Example, TrainingRun
from memorizz.finetune.service import RANKING_METHOD
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


@pytest.fixture
def ft(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    # setenv first so teardown restores each variable even when a swap sets
    # it later (delenv alone records nothing for an unset variable).
    for key in swapping.EMBEDDING_ENV + ("MEMORIZZ_RERANKER_MODEL",):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr("memorizz.embeddings._global_embedding_manager", None)
    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )
    return FineTuning(provider, home=str(tmp_path / "home" / "finetune"))


def _items(score, n=20):
    """Per-example scores (1 right, 0 wrong) averaging ``score``."""
    right = round(score * n)
    return {f"x{i}": 1.0 if i < right else 0.0 for i in range(n)}


def _run(
    ft,
    tmp_path,
    kind="decision",
    task="memory_routing",
    base="Qwen/Qwen3-Embedding-0.6B",
    before=0.2,
    after=0.8,
    meta=None,
):
    dataset = ft.create_dataset(kind, task)
    run = TrainingRun(
        dataset_id=dataset.dataset_id,
        kind=kind,
        task=task,
        base_model=base,
        backbone={
            "decision": "embedding",
            "embedding": "embedding",
            "reranker": "cross_encoder",
        }[kind],
        status="succeeded",
        metrics_before={
            "accuracy": before,
            "ndcg@10": before,
            "per_item": _items(before),
        },
        metrics_after={"accuracy": after, "ndcg@10": after, "per_item": _items(after)},
    )
    run.output_dir = str(tmp_path / "runs" / run.run_id)
    model = Path(run.output_dir) / "model"
    model.mkdir(parents=True)
    if meta:
        (model / "memorizz_model.json").write_text(json.dumps(meta))
    run.progress = {"model_dir": str(model)}
    ft.store.save_run(run)
    return run


# -- routing writes ------------------------------------------------------------


@pytest.mark.unit
def test_routing_places_a_write_only_where_the_caller_may_write(monkeypatch):
    allowed = {"knowledge_base", "short_term_memory"}
    monkeypatch.delenv(routing.MIN_CONFIDENCE_ENV, raising=False)
    assert routing.place(None, allowed)["memory_type"] == "knowledge_base"
    sure = {"memory_type": "short_term_memory", "confidence": 0.9, "run_id": "r1"}
    assert routing.place(sure, allowed) | {} == {
        "routed_to": "short_term_memory",
        "confidence": 0.9,
        "run_id": "r1",
        "memory_type": "short_term_memory",
        "applied": True,
        "reason": "Routed to short_term_memory.",
    }
    skip = routing.place({"memory_type": "do_not_store", "confidence": 0.8}, allowed)
    assert skip["memory_type"] is None and skip["applied"]
    unsure = routing.place({"memory_type": "do_not_store", "confidence": 0.3}, allowed)
    assert unsure["memory_type"] == "knowledge_base" and not unsure["applied"]
    entity = routing.place({"memory_type": "entity_memory", "confidence": 0.9}, allowed)
    assert entity["memory_type"] == "knowledge_base"
    assert "memorizz_upsert_entity" in entity["next_step"]
    monkeypatch.setenv(routing.MIN_CONFIDENCE_ENV, "0.95")
    assert routing.place(sure, allowed)["memory_type"] == "knowledge_base"


@pytest.mark.unit
def test_mcp_store_memory_auto_follows_the_active_router(tmp_path, monkeypatch):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import ALL_SCOPES, MemorizzMCPServerConfig
    from memorizz.mcp_server.runtime import MemorizzRuntime

    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )
    runtime = MemorizzRuntime(
        MemorizzMCPServerConfig(allow_writes=True),
        provider=provider,
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    monkeypatch.setattr(runtime, "_embed_or_none", lambda text: None)
    identity = RequestIdentity(
        principal="alice", scopes=frozenset(ALL_SCOPES), authenticated=True
    )
    answers = {
        "remember the deploy runs at 9": {
            "memory_type": "short_term_memory",
            "confidence": 0.9,
            "run_id": "r1",
        },
        "ok thanks": {
            "memory_type": "do_not_store",
            "confidence": 0.97,
            "run_id": "r1",
        },
    }
    monkeypatch.setattr(
        routing, "route", lambda provider, text, **context: answers[text]
    )
    stored = runtime.remember("remember the deploy runs at 9", "auto", identity)
    assert stored["memory_type"] == "short_term_memory" and stored["routing"]["applied"]
    row = provider.retrieve_by_id(stored["record_id"], MemoryType.SHORT_TERM_MEMORY)
    assert row["routing"] == {
        "routed_to": "short_term_memory",
        "confidence": 0.9,
        "run_id": "r1",
        "applied": True,
    }
    skipped = runtime.remember("ok thanks", "auto", identity)
    assert skipped["stored"] is False and skipped["routing"]["memory_type"] is None
    monkeypatch.setattr(routing, "route", lambda provider, text, **context: None)
    plain = runtime.remember("a fact", "auto", identity)
    assert plain["memory_type"] == "knowledge_base" and not plain["routing"]["applied"]


# -- reranking -----------------------------------------------------------------


@pytest.mark.unit
def test_a_reranker_reorders_recall_but_a_broken_one_does_not_break_it():
    from memorizz.memagent.utils.context_dedup import dedupe_and_select

    rows = [
        (
            "knowledge_base",
            {"_id": "a", "content": "The office is in Lagos.", "score": 0.9},
        ),
        (
            "knowledge_base",
            {"_id": "b", "content": "Deploys run at 9 on Fridays.", "score": 0.5},
        ),
    ]
    plain = dedupe_and_select(rows, max_items=1)
    assert plain[0]["id"] == "a"
    reranked = dedupe_and_select(
        rows,
        max_items=1,
        rerank=lambda texts: [0.1 if "Lagos" in t else 0.95 for t in texts],
    )
    assert reranked[0]["id"] == "b"

    def broken(texts):
        raise RuntimeError("model gone")

    assert dedupe_and_select(rows, max_items=1, rerank=broken)[0]["id"] == "a"


@pytest.mark.unit
def test_rerank_rows_uses_the_configured_model(monkeypatch, tmp_path):
    from memorizz.finetune import reranking

    monkeypatch.setattr(
        reranking,
        "rerank_for",
        lambda query: (lambda texts: [len(t) / 10 for t in texts]),
    )
    rows = [{"content": "short"}, {"content": "a much longer memory"}]
    ranked = reranking.rerank_rows("q", rows, lambda row: row["content"])
    assert [r["content"] for r in ranked] == ["a much longer memory", "short"]
    assert ranked[0]["rerank_score"] == 2.0
    monkeypatch.setenv(reranking.RERANKER_ENV, str(tmp_path / "missing"))
    monkeypatch.setattr(reranking, "rerank_for", reranking.rerank_for)
    assert reranking.configured_reranker() is None


# -- embeddings ----------------------------------------------------------------


@pytest.mark.unit
def test_query_and_document_embeddings_are_cached_apart():
    from memorizz.embeddings import EmbeddingManager

    class _Provider:
        calls = []

        def get_embedding(self, text, **kwargs):
            self.calls.append(kwargs.get("input_type"))
            return [1.0, 0.0] if kwargs.get("input_type") == "query" else [0.0, 1.0]

    manager = EmbeddingManager.__new__(EmbeddingManager)
    import threading
    from collections import OrderedDict

    manager._provider = _Provider()
    manager._embedding_cache = OrderedDict()
    manager._embedding_cache_lock = threading.Lock()
    assert manager.get_embedding("deploy time", input_type="query") == [1.0, 0.0]
    assert manager.get_embedding("deploy time") == [0.0, 1.0]
    assert manager.get_embedding("deploy time", input_type="query") == [1.0, 0.0]
    assert _Provider.calls == ["query", None]


@pytest.mark.unit
def test_a_fine_tuned_model_folder_brings_its_own_prompts(tmp_path):
    from memorizz.embeddings.huggingface.provider import HuggingFaceEmbeddingProvider

    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "memorizz_model.json").write_text(
        json.dumps(
            {
                "family": "qwen3-embedding",
                "prompts": {"query": "Instruct: find\nQuery:", "document": ""},
            }
        )
    )
    provider = HuggingFaceEmbeddingProvider.__new__(HuggingFaceEmbeddingProvider)
    provider.model_name = str(folder)
    provider.finetuned = HuggingFaceEmbeddingProvider._finetuned_meta(str(folder))
    assert provider._default_prompt(str(folder), "query") == "query"
    assert provider._default_prompt(str(folder), None) is None  # empty document prompt
    assert provider._default_prompt("google/embeddinggemma-2", "query") == "SearchQuery"


@pytest.mark.unit
def test_reembedding_rebuilds_each_store_with_its_own_text(ft):
    provider = ft.provider
    kb = provider.store(
        {"content": "Deploys run at 9.", "embedding": [0.0, 0.0]},
        MemoryType.KNOWLEDGE_BASE,
    )
    entity = provider.store(
        {
            "entity_id": "e1",
            "name": "Priya",
            "entity_type": "person",
            "attributes": [{"name": "prefers", "value": "calls"}],
            "embedding": [0.0, 0.0],
        },
        MemoryType.ENTITY_MEMORY,
    )
    provider.store({"content": ""}, MemoryType.KNOWLEDGE_BASE)

    class _Manager:
        seen = []

        def get_embeddings(self, texts):
            self.seen.extend(texts)
            return [[float(len(t)), 1.0] for t in texts]

    report = swapping.reembed(
        provider,
        _Manager(),
        stores=[MemoryType.KNOWLEDGE_BASE, MemoryType.ENTITY_MEMORY],
    )
    assert report["knowledge_base"] == {
        "rows": 2,
        "updated": 1,
        "skipped": 1,
        "failed": 0,
    }
    assert report["entity_memory"]["updated"] == 1
    assert "Name: Priya\nType: person\nprefers: calls" in _Manager.seen
    assert provider.retrieve_by_id(kb, MemoryType.KNOWLEDGE_BASE)["embedding"] == [
        17.0,
        1.0,
    ]
    assert (
        provider.retrieve_by_id(entity, MemoryType.ENTITY_MEMORY)["embedding"][1] == 1.0
    )


# -- swapping ------------------------------------------------------------------


@pytest.mark.unit
def test_swap_router_and_reranker_in_and_back(ft, tmp_path, monkeypatch):
    router = _run(ft, tmp_path)
    assert ft.swap("memory_routing", router.run_id)["run_id"] == router.run_id
    assert ft.active()["memory_routing"]["run_id"] == router.run_id
    ft.swap_back("memory_routing")
    assert "memory_routing" not in ft.active()

    reranker = _run(
        ft,
        tmp_path,
        kind="reranker",
        task="relevance",
        base="cross-encoder/ms-marco-MiniLM-L6-v2",
    )
    monkeypatch.setattr(
        "memorizz.finetune.reranking.load_scorer",
        lambda path: (lambda q, ps: [0.5] * len(ps)),
    )
    ft.swap("reranking", reranker.run_id, persist=False)
    import os

    assert os.environ["MEMORIZZ_RERANKER_MODEL"] == ft.model_dir(reranker.run_id)
    assert ft.active()["reranker"]["run_id"] == reranker.run_id
    with pytest.raises(FineTuningError, match="not one for memory routing"):
        ft.swap("memory_routing", reranker.run_id)
    ft.swap_back("reranking", persist=False)
    assert "MEMORIZZ_RERANKER_MODEL" not in os.environ and "reranker" not in ft.active()


@pytest.mark.unit
def test_swap_embedding_in_place_fits_the_store_then_swaps_back(
    ft, tmp_path, monkeypatch
):
    """Stores without a staged slot switch, then re-embed in place."""
    import os

    from memorizz.finetune import staging

    monkeypatch.setattr(staging, "supported", lambda provider: False)
    monkeypatch.setattr(
        staging,
        "capabilities",
        lambda provider: {"staged": False, "fixed_size": True, "resize": False},
    )
    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    os.environ.update(
        {
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "ollama",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": "nomic-embed-text",
        }
    )

    class _Manager:
        def __init__(self, size):
            self.size = size

        def get_dimensions(self):
            return self.size

        def get_embedding(self, text, **kwargs):
            return [0.1] * self.size

    state = {"manager": _Manager(256)}
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_manager", lambda: state["manager"]
    )

    def apply(values, persist=None):
        for key in swapping.EMBEDDING_ENV:
            if values.get(key):
                os.environ[key] = values[key]
            else:
                os.environ.pop(key, None)

    monkeypatch.setattr(swapping, "apply_settings", apply)
    reembedded = []
    monkeypatch.setattr(
        swapping,
        "reembed",
        lambda provider, manager, progress=None: reembedded.append(manager)
        or {"knowledge_base": {"rows": 1, "updated": 1, "skipped": 0, "failed": 0}},
    )
    out = ft.swap("embedding", run.run_id, persist=False)
    assert out["dimensions"] == 256 and not out["staged"]
    assert out["reembedded"]["knowledge_base"]["updated"] == 1
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "huggingface"
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS"] == "256"
    assert ft.active()["embedding"]["run_id"] == run.run_id
    card = next(c for c in ft.overview() if c["use"] == "embedding")
    assert card["in_use"]["run_id"] == run.run_id and card["candidate"]["in_use"]

    ft.swap_back("embedding", persist=False)
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "ollama"
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_MODEL"] == "nomic-embed-text"
    assert "embedding" not in ft.active() and len(reembedded) == 2

    state["manager"] = _Manager(1024)
    ft._store_size(fresh=True)
    with pytest.raises(
        FineTuningError, match="1024-dimension vectors and the model makes 768"
    ):
        ft.swap("embedding", run.run_id, persist=False)


class _SizedModel:
    """An embedding model whose vectors say what they were made from: every
    value is the length of the text."""

    def __init__(self, size, provider=None, live=None, store=None):
        self.size, self.texts = size, []
        self.provider, self.live, self.store = provider, live, store

    def get_dimensions(self):
        return self.size

    def get_provider_info(self):
        return {"provider": "test", "model": f"sized-{self.size}"}

    def get_embedding(self, text, **kwargs):
        return [1.0] * self.size

    def get_embeddings(self, texts):
        if self.provider is not None:
            # Search keeps reading the live vectors while new ones are made.
            row = self.provider.retrieve_by_id(self.store, MemoryType.KNOWLEDGE_BASE)
            assert len(row["embedding"]) == self.live["size"]
        self.texts.extend(texts)
        return [[float(len(t))] * self.size for t in texts]


@pytest.mark.unit
def test_staged_swap_switches_at_once_and_swaps_back_with_the_kept_vectors(
    ft, tmp_path, monkeypatch
):
    import os

    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    os.environ.update(
        {
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "ollama",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": "nomic-embed-text",
        }
    )
    provider = ft.provider
    kb = provider.store(
        {"content": "Deploys run at 9.", "embedding": [0.25] * 256},
        MemoryType.KNOWLEDGE_BASE,
    )
    other = provider.store(
        {"content": "Standups are at 10.", "embedding": [0.25] * 256},
        MemoryType.KNOWLEDGE_BASE,
    )
    live = {"size": 256}
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_manager", lambda: _SizedModel(live["size"])
    )
    built = {}

    def manager_for(values):
        size = int(values.get("MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS") or 256)
        built[values["MEMORIZZ_DEFAULT_EMBEDDING_MODEL"]] = model = _SizedModel(
            size, provider, live, kb
        )
        return model

    monkeypatch.setattr(swapping, "manager_for", manager_for)

    # Any size fits a filesystem store: the model's full size by default.
    plan = ft.embedding_plan(run.run_id)
    assert [(o["dimensions"], o["kind"]) for o in plan["options"]] == [
        (768, "resize"),
        (256, "cut"),
    ]
    assert plan["default"] == 768 and plan["staged"]

    out = ft.swap("embedding", run.run_id, persist=False)
    assert out["staged"] and out["dimensions"] == 768
    assert out["reembedded"]["stores"]["knowledge_base"]["embedded"] == 2
    row = provider.retrieve_by_id(kb, MemoryType.KNOWLEDGE_BASE)
    assert row["embedding"] == [17.0] * 768
    assert provider.vector_previous() == {"dimensions": 256, "rows": 2}
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_DIMENSIONS"] == "768"
    assert ft.active()["embedding"]["run_id"] == run.run_id
    live["size"] = 768

    # A memory edited after the swap: swapping back re-embeds only that one.
    provider.update_by_id(
        other,
        {"content": "Standups moved to 11.", "embedding": [21.0] * 768},
        MemoryType.KNOWLEDGE_BASE,
    )
    back = ft.swap_back("embedding", persist=False)
    assert back["reused"]
    counts = back["reembedded"]["stores"]["knowledge_base"]
    assert counts["kept"] == 1 and counts["embedded"] == 1
    assert built["nomic-embed-text"].texts == ["Standups moved to 11."]
    # The kept vector comes back exactly.
    assert provider.retrieve_by_id(kb, MemoryType.KNOWLEDGE_BASE)["embedding"] == (
        [0.25] * 256
    )
    edited = provider.retrieve_by_id(other, MemoryType.KNOWLEDGE_BASE)
    assert edited["embedding"] == [21.0] * 256
    assert provider.vector_previous() is None
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "ollama"
    assert "embedding" not in ft.active()


@pytest.mark.unit
def test_a_staged_swap_that_fails_changes_nothing(ft, tmp_path, monkeypatch):
    import os

    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    os.environ.update(
        {
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "ollama",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": "nomic-embed-text",
        }
    )
    kb = ft.provider.store(
        {"content": "Deploys run at 9.", "embedding": [0.25] * 256},
        MemoryType.KNOWLEDGE_BASE,
    )
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_manager", lambda: _SizedModel(256)
    )

    class _Broken(_SizedModel):
        def get_embeddings(self, texts):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(swapping, "manager_for", lambda values: _Broken(768))
    with pytest.raises(
        FineTuningError, match="stopped before switching; nothing changed"
    ):
        ft.swap("embedding", run.run_id, persist=False)
    row = ft.provider.retrieve_by_id(kb, MemoryType.KNOWLEDGE_BASE)
    assert row["embedding"] == [0.25] * 256
    assert ft.provider.vector_previous() is None
    assert not (Path(ft.provider.root_path) / ".vector-staging" / "next").exists()
    assert os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "ollama"
    assert "embedding" not in ft.active()


@pytest.mark.unit
def test_embedding_plan_keeps_a_fixed_size_store_or_changes_its_size(
    ft, tmp_path, monkeypatch
):
    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    sizes = {"store": 256}
    monkeypatch.setattr(
        swapping, "store_dimensions", lambda provider, manager: sizes["store"]
    )
    monkeypatch.setattr("memorizz.embeddings.get_embedding_manager", lambda: None)

    def plan(staging_info):
        ft.provider.VECTOR_STAGING = staging_info
        ft._store_size(fresh=True)
        out = ft.embedding_plan(run.run_id)
        return [(o["dimensions"], o["kind"]) for o in out["options"]], out

    oracle = {"fixed_size": True, "resize": True}
    mongodb = {"fixed_size": True, "resize": False}
    # Keep the store's size by default (cut), or change the store to 768.
    options, out = plan(oracle)
    assert options == [(256, "cut"), (768, "resize")] and out["default"] == 256
    assert ft._scored_size(ft.run(run.run_id)) == 256
    sizes["store"] = 768
    assert plan(oracle)[0] == [(768, "keep")]
    assert ft._scored_size(ft.run(run.run_id)) is None
    # Too big to cut to, or not a size EmbeddingGemma makes: change it.
    for store in (1024, 300):
        sizes["store"] = store
        assert plan(oracle)[0] == [(768, "resize")]
    # A store that cannot change size cannot take it at all.
    sizes["store"] = 1024
    options, out = plan(mongodb)
    assert options == [] and "cannot change size" in out["reason"]
    with pytest.raises(FineTuningError, match="cannot change size"):
        ft.swap("embedding", run.run_id, persist=False)
    sizes["store"] = 256
    plan(oracle)
    with pytest.raises(FineTuningError, match="at 256 or 768 dimensions, not 512"):
        ft.swap("embedding", run.run_id, persist=False, dimensions=512)


@pytest.mark.unit
def test_a_card_is_scored_at_the_size_the_swap_would_store(ft, tmp_path, monkeypatch):
    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    monkeypatch.setattr(swapping, "store_dimensions", lambda provider, manager: 256)
    monkeypatch.setattr("memorizz.embeddings.get_embedding_manager", lambda: None)
    ft.provider.VECTOR_STAGING = {"fixed_size": True, "resize": True}
    for eval_id, model, value in (
        ("full", {"type": "run", "run_id": run.run_id, "label": "ft"}, 0.9),
        (
            "cut",
            {"type": "run", "run_id": run.run_id, "label": "ft", "dimensions": 256},
            0.8,
        ),
        (
            "base-cut",
            {
                "type": "base",
                "base_model": "google/embeddinggemma-2",
                "label": "base",
                "dimensions": 256,
            },
            0.5,
        ),
        (
            "base-full",
            {"type": "base", "base_model": "google/embeddinggemma-2", "label": "base"},
            0.7,
        ),
    ):
        ft.store.save_eval(
            {
                "eval_id": eval_id,
                "dataset_id": run.dataset_id,
                "dataset_name": "Retrieval",
                "split": "test",
                "count": 20,
                "model": model,
                "metrics": {"ndcg@10": value, "per_item": _items(value, 20)},
                "method": RANKING_METHOD,
                "created_at": "2026-10-13T00:00:00",
            }
        )
    card = next(c for c in ft.overview() if c["use"] == "embedding")["candidate"]
    evidence = card["evidence"]
    assert evidence["dimensions"] == 256
    # The 256-dimension scores, not the full-size ones (0.9 and 0.7).
    assert evidence["fine_tuned"] == 0.8 and evidence["base"] == 0.5

    # Compare scores the model and its base at the size a swap would keep.
    calls = []
    monkeypatch.setattr(ft, "evaluate", lambda dataset_id, **kw: calls.append(kw) or {})
    ft.compare(run.run_id)
    assert any(c.get("run_id") and c.get("dimensions") == 256 for c in calls)
    assert any(c.get("base_model") and c.get("dimensions") == 256 for c in calls)
    assert any(c.get("run_id") and not c.get("dimensions") for c in calls)


@pytest.mark.unit
def test_embedding_scores_cut_vectors_and_normalise_them_again():
    import torch

    from memorizz.finetune.scoring import embedding_scorer

    vectors = {"q": [3.0, 4.0, 100.0], "a": [3.0, 4.0, -100.0], "b": [4.0, -3.0, 100.0]}

    class _Model:
        def encode(self, texts, prompt=None, normalize_embeddings=False, **kwargs):
            out = torch.tensor([vectors[t] for t in texts])
            return (
                torch.nn.functional.normalize(out, dim=-1)
                if normalize_embeddings
                else out
            )

    full = embedding_scorer(_Model(), {})("q", ["a", "b"])
    cut = embedding_scorer(_Model(), {}, dimensions=2)("q", ["a", "b"])
    assert full[1] > full[0]  # the third value dominates at full size
    assert cut == pytest.approx([1.0, 0.0])  # [3, 4] vs [3, 4] and [4, -3]


@pytest.mark.unit
def test_cards_compare_with_the_model_in_use_and_say_if_it_helped(ft, tmp_path):
    router = _run(ft, tmp_path, before=0.1, after=0.85)
    weaker = _run(
        ft, tmp_path, kind="embedding", task="retrieval", base="google/embeddinggemma-2"
    )
    for model, value in (
        ({"type": "run", "run_id": weaker.run_id, "label": "ft"}, 0.6),
        (
            {"type": "base", "base_model": "google/embeddinggemma-2", "label": "base"},
            0.5,
        ),
        ({"type": "in_use", "label": "nomic (in use now)"}, 0.7),
    ):
        ft.store.save_eval(
            {
                "eval_id": "e" + str(value),
                "dataset_id": weaker.dataset_id,
                "dataset_name": "Retrieval",
                "split": "held_out",
                "count": 30,
                "model": model,
                "metrics": {"ndcg@10": value, "per_item": _items(value, 30)},
                "method": RANKING_METHOD,
                "created_at": "2026-10-12T00:00:00",
            }
        )
    # A score from before scoring matched training is not compared.
    ft.store.save_eval(
        {
            "eval_id": "e-old",
            "dataset_id": weaker.dataset_id,
            "dataset_name": "Retrieval",
            "split": "held_out",
            "count": 90,
            "model": {"type": "in_use", "label": "nomic (in use now)"},
            "metrics": {"ndcg@10": 0.1},
            "created_at": "2026-10-11T00:00:00",
        }
    )
    assert "e-old" not in {e["eval_id"] for e in ft.evaluations()}
    cards = {c["use"]: c for c in ft.overview()}
    assert set(cards) == {"memory_routing", "embedding", "reranking"}
    routing_card = cards["memory_routing"]["candidate"]
    assert routing_card["better"] and routing_card["gain_points"] == 75.0
    assert routing_card["compared_with"] == "the untrained base model"
    embedding = cards["embedding"]["candidate"]
    assert (
        embedding["evidence"]["frozen"]
        and embedding["compared_with"] == "the model in use now"
    )
    assert embedding["gain_points"] == -10.0 and not embedding["better"]
    # 18 vs 21 right of 30 (3 differ): that could be chance.
    assert embedding["verdict"] == "unclear"
    assert embedding["evidence"]["significance"]["p"] == 0.25
    # 17 vs 2 right of 20 in training (15 differ): unlikely to be chance.
    assert routing_card["verdict"] == "better"
    assert routing_card["evidence"]["significance"]["p"] < 0.001
    assert cards["reranking"]["candidate"] is None
    assert cards["reranking"]["in_use"]["label"] == "Vector search only (no reranker)"
    assert router.run_id == routing_card["run_id"]
    assert (
        embedding["version"] == "v1" and embedding["label"] == "v1 · embeddinggemma-2"
    )
    # Results shows the same comparison as the card.
    verdict = ft.verdict(weaker.run_id)
    assert (
        verdict["evidence"] == embedding["evidence"] and verdict["card"] == "embedding"
    )


@pytest.mark.unit
def test_embedders_that_take_only_text_still_work_for_queries():
    from memorizz.embeddings import EmbeddingManager, embed_as_query

    assert embed_as_query(lambda text: [1.0], "deploy time") == [1.0]

    class _Legacy:
        def get_embedding(self, text):
            return [0.5]

    import threading
    from collections import OrderedDict

    manager = EmbeddingManager.__new__(EmbeddingManager)
    manager._provider = _Legacy()
    manager._embedding_cache = OrderedDict()
    manager._embedding_cache_lock = threading.Lock()
    assert manager.get_embedding("q", input_type="query") == [0.5]

    def broken(text, **kwargs):
        raise TypeError("unsupported operand type(s)")

    with pytest.raises(TypeError, match="operand"):
        embed_as_query(broken, "q")


@pytest.mark.unit
def test_deleting_a_dataset_removes_its_runs_and_evaluations(ft, tmp_path):
    run = _run(ft, tmp_path)
    ft.store.save_eval(
        {
            "eval_id": "fteval-1",
            "dataset_id": run.dataset_id,
            "dataset_name": "Routing",
            "split": "test",
            "count": 3,
            "model": {"type": "run", "run_id": run.run_id, "label": "x"},
            "metrics": {"accuracy": 1.0},
            "created_at": "2026-10-12",
        }
    )
    assert ft.evaluations(run.dataset_id)
    assert ft.delete_dataset(run.dataset_id)
    assert ft.evaluations(run.dataset_id) == [] and ft.store.get_run(run.run_id) is None


@pytest.mark.unit
def test_entity_text_accepts_every_attribute_shape_and_odd_rows_are_counted():
    as_mapping = {
        "name": "Priya",
        "entity_type": "person",
        "attributes": {"prefers": "calls"},
        "relations": [],
        "metadata": None,
    }
    as_json = {"name": "Priya", "attributes": '[{"name": "prefers", "value": "calls"}]'}
    assert "prefers: calls" in swapping.embedding_text(
        MemoryType.ENTITY_MEMORY, as_mapping
    )
    assert "prefers: calls" in swapping.embedding_text(
        MemoryType.ENTITY_MEMORY, as_json
    )

    class _Provider:
        def list_all(self, memory_type):
            return [{"_id": "a", "content": "fine"}, {"_id": "b", "content": object()}]

        def update_by_id(self, record_id, data, memory_type):
            return True

    class _Manager:
        def get_embeddings(self, texts):
            return [[1.0] for _ in texts]

    def explode(memory_type, row):
        if row["_id"] == "b":
            raise ValueError("odd row")
        return "fine"

    import memorizz.finetune.swap as module

    original = module.embedding_text
    module.embedding_text = explode
    try:
        report = swapping.reembed(
            _Provider(), _Manager(), stores=[MemoryType.KNOWLEDGE_BASE]
        )
    finally:
        module.embedding_text = original
    assert report["knowledge_base"] == {
        "rows": 2,
        "updated": 1,
        "skipped": 0,
        "failed": 1,
    }


@pytest.mark.unit
def test_a_failed_embedding_swap_is_undone(ft, tmp_path, monkeypatch):
    """In place (a store without a staged slot): switched back and re-embedded."""
    import os

    from memorizz.finetune import staging

    monkeypatch.setattr(staging, "supported", lambda provider: False)
    monkeypatch.setattr(
        staging,
        "capabilities",
        lambda provider: {"staged": False, "fixed_size": True, "resize": False},
    )

    run = _run(
        ft,
        tmp_path,
        kind="embedding",
        task="retrieval",
        base="google/embeddinggemma-2",
        meta={"family": "embeddinggemma", "dimensions": 768},
    )
    os.environ.update(
        {
            "MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER": "ollama",
            "MEMORIZZ_DEFAULT_EMBEDDING_MODEL": "nomic-embed-text",
        }
    )

    class _Manager:
        def get_dimensions(self):
            return 768

        def get_embedding(self, text, **kwargs):
            return [0.1] * 768

    monkeypatch.setattr("memorizz.embeddings.get_embedding_manager", lambda: _Manager())
    monkeypatch.setattr(
        swapping,
        "apply_settings",
        lambda values, persist=None: [
            os.environ.__setitem__(k, v) if v else os.environ.pop(k, None)
            for k, v in values.items()
        ],
    )
    calls = []

    def reembed(provider, manager, progress=None):
        calls.append(os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"])
        if len(calls) == 1:
            raise RuntimeError("database went away")
        return {}

    monkeypatch.setattr(swapping, "reembed", reembed)
    with pytest.raises(FineTuningError, match="undone: database went away"):
        ft.swap("embedding", run.run_id, persist=False)
    assert calls == ["huggingface", "ollama"]
    assert (
        os.environ["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] == "ollama"
        and "embedding" not in ft.active()
    )


@pytest.mark.unit
def test_faiss_and_pytorch_can_share_a_process_on_macos(monkeypatch):
    import importlib
    import importlib.util
    import sys

    from memorizz.memory_provider.filesystem.provider import FileSystemProvider

    threads, loaded = [], []
    fake = type("faiss", (), {"omp_set_num_threads": staticmethod(threads.append)})
    monkeypatch.setattr(
        importlib, "import_module", lambda name: loaded.append(name) or fake
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "")
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK")
    assert FileSystemProvider._load_faiss() is fake
    assert threads == [1]
    # PyTorch's OpenMP runtime starts first: with FAISS's first, loading a
    # second language model crashed the process.
    assert loaded == ["torch", "faiss"]
    import os

    assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"


@pytest.mark.unit
def test_versions_are_numbered_per_job_and_can_be_named(ft, tmp_path):
    first = _run(ft, tmp_path)
    failed = _run(ft, tmp_path)
    failed.status = "failed"
    ft.store.save_run(failed)
    second = _run(ft, tmp_path)
    second.parent_run_id = first.run_id
    second.progress["replay"] = [{"dataset_id": first.dataset_id, "examples": 4}]
    ft.store.save_run(second)
    embedder = _run(ft, tmp_path, kind="embedding", task="retrieval")
    runs = {r["run_id"]: r for r in ft.runs()}
    assert runs[first.run_id]["label"] == "v1" and runs[second.run_id]["label"] == "v2"
    assert runs[failed.run_id]["version"] is None
    assert runs[second.run_id]["lineage"] == "from v1 + replay"
    assert runs[embedder.run_id]["label"] == "v1"  # its own job
    assert ft.rename_run(first.run_id, " round one ")["label"] == "round one"
    assert ft.run(second.run_id)["lineage"] == "from round one + replay"
    assert ft.rename_run(first.run_id, "")["label"] == "v1"


@pytest.mark.unit
def test_a_test_set_is_scored_on_but_never_trained_on(ft, tmp_path):
    run = _run(ft, tmp_path)
    frozen = ft.create_dataset("decision", "memory_routing", name="Frozen")
    assert ft.update_dataset(frozen.dataset_id, test_set=True).test_set
    with pytest.raises(FineTuningError, match="test set"):
        ft.start_run(frozen.dataset_id)
    with pytest.raises(FineTuningError, match="replaying it would train on it"):
        ft._snapshot(
            ft._dataset(run.dataset_id),
            [{"dataset_id": frozen.dataset_id, "fraction": 1}],
        )
    rows = {d["dataset_id"]: d for d in ft.datasets()}
    assert rows[frozen.dataset_id]["test_set"]
    assert rows[run.dataset_id]["runs"]["latest"]["label"] == "v1"
    assert rows[run.dataset_id]["runs"]["latest"]["after"] == 0.8
    with pytest.raises(FineTuningError, match="needs a name"):
        ft.update_dataset(frozen.dataset_id, name="  ")


@pytest.mark.unit
def test_dataset_counts_come_from_one_listing(ft, monkeypatch):
    a = ft.create_dataset("decision", "memory_routing", name="A")
    b = ft.create_dataset("embedding", "retrieval", name="B")
    ft.store.add_examples(
        [
            Example(dataset_id=a.dataset_id, input={"state": "x"}, status="accepted"),
            Example(dataset_id=a.dataset_id, input={"state": "y"}),
            Example(
                dataset_id=b.dataset_id,
                input={"query": "q", "positive": "p"},
                status="accepted",
                split="test",
            ),
        ]
    )
    calls = []
    real = ft.store._list
    monkeypatch.setattr(
        ft.store, "_list", lambda *a, **k: calls.append(a) or real(*a, **k)
    )
    rows = {d["name"]: d["counts"] for d in ft.datasets()}
    assert rows["A"] == {
        "total": 2,
        "status": {"accepted": 1, "pending": 1},
        "split": {"train": 1},
    }
    assert rows["B"]["split"] == {"test": 1}
    assert [c for c in calls if c[0] == "finetune_example"] == [("finetune_example",)]


@pytest.mark.unit
def test_embedding_and_reranker_models_are_put_to_work_by_swap(ft, tmp_path):
    run = _run(ft, tmp_path, kind="embedding", task="retrieval")
    with pytest.raises(FineTuningError, match="swap embedding"):
        ft.activate(run.run_id, "embedding")
    reranker = _run(ft, tmp_path, kind="reranker", task="relevance")
    with pytest.raises(FineTuningError, match="swap reranking"):
        ft.activate(reranker.run_id, "reranker")


@pytest.mark.unit
def test_retrieval_is_scored_the_way_training_scores_it(ft, monkeypatch):
    def examples(dataset, n, split="test"):
        return [
            Example(
                dataset_id=dataset.dataset_id,
                input={"query": f"q{i}", "positive": f"m{i}", "negatives": [f"n{i}"]},
                status="accepted",
                split=split,
            )
            for i in range(n)
        ]

    seen = []

    def ranker(dataset, **kwargs):
        def score(query, passages):
            seen.append(list(passages))
            return [1.0] + [0.0] * (len(passages) - 1)

        return score, {"type": "base", "label": "base"}

    monkeypatch.setattr(ft, "_ranker", ranker)
    embedding = ft.create_dataset("embedding", "retrieval")
    ft.store.add_examples(examples(embedding, 3) + examples(embedding, 12, "train"))
    result = ft.evaluate(embedding.dataset_id, base_model="x", split="test")
    # Each test query searches every memory in the dataset (15 + 15 look-alikes,
    # with names shared between splits counted once).
    assert result["count"] == 3 and result["candidates"] == len(seen[0]) == 24
    assert result["method"] == RANKING_METHOD
    assert result["metrics"]["ndcg@10"] == 1.0

    seen.clear()
    reranker = ft.create_dataset("reranker", "relevance")
    ft.store.add_examples(examples(reranker, 3) + examples(reranker, 12, "train"))
    ft.evaluate(reranker.dataset_id, base_model="x", split="test")
    # Its memory, its look-alike and 9 other memories.
    assert [len(c) for c in seen] == [11, 11, 11]
    assert all(c[0] == f"m{i}" and c[1] == f"n{i}" for i, c in enumerate(seen))


@pytest.mark.unit
def test_a_version_is_compared_with_the_version_in_use(ft, tmp_path):
    older = _run(ft, tmp_path)
    newer = _run(ft, tmp_path)
    newer.parent_run_id = older.run_id
    ft.store.save_run(newer)
    for model, value in (
        ({"type": "run", "run_id": older.run_id, "label": "a"}, 0.8),
        ({"type": "run", "run_id": newer.run_id, "label": "b"}, 0.6),
        ({"type": "base", "base_model": newer.base_model, "label": "base"}, 0.1),
    ):
        ft.store.save_eval(
            {
                "eval_id": f"e-{value}",
                "dataset_id": older.dataset_id,
                "dataset_name": "Frozen",
                "split": "test",
                "count": 35,
                "model": model,
                "metrics": {"accuracy": value, "per_item": _items(value, 35)},
                "created_at": "2026-10-12T00:00:00",
            }
        )
    ft.swap("memory_routing", older.run_id)
    verdict = ft.verdict(newer.run_id)
    assert verdict["evidence"]["in_use"]["model"] == "v1 · Qwen3-Embedding-0.6B"
    assert verdict["compared_with"] == "the model in use now"
    assert verdict["gain_points"] == -20.0 and not verdict["better"]
    # 7 examples differ, all against it: p = 2 x 0.5**7.
    assert (
        verdict["verdict"] == "worse"
        and verdict["evidence"]["significance"]["p"] == 0.0156
    )
    card = next(c for c in ft.overview() if c["use"] == "memory_routing")
    assert card["candidate"]["run_id"] == older.run_id and card["candidate"]["in_use"]
    # Scores on a dataset's "all" split include what a model trained on: not used.
    ft.store.save_eval(
        {
            "eval_id": "e-all",
            "dataset_id": older.dataset_id,
            "dataset_name": "Frozen",
            "split": "all",
            "count": 35,
            "model": {"type": "run", "run_id": newer.run_id, "label": "b"},
            "metrics": {"accuracy": 1.0, "per_item": _items(1.0, 35)},
            "created_at": "2026-10-13T00:00:00",
        }
    )
    assert ft.verdict(newer.run_id)["evidence"]["fine_tuned"] == 0.6
    ft.store.delete_eval("e-all")
    # Without a shared set, a continued run is compared with where it started.
    ft.store.delete_eval("e-0.6")
    fallback = ft.verdict(newer.run_id)["evidence"]
    assert fallback["base_kind"] == "parent" and fallback["base_label"] == "v1"
    assert ft.verdict(newer.run_id)["compared_with"] == "the version it started from"


@pytest.mark.unit
def test_a_verdict_needs_a_significance_test(ft, tmp_path):
    run = _run(ft, tmp_path, before=0.4, after=0.6)
    v = ft.verdict(run.run_id)
    # 12 vs 8 right of 20, the 4 that differ all in its favour: p = 0.125.
    assert v["gain_points"] == 20.0 and v["verdict"] == "unclear" and not v["better"]
    assert v["evidence"]["significance"] == {
        "n": 20,
        "diff": 0.2,
        "differ": 4,
        "p": 0.125,
    }
    assert v["training_significance"] == v["evidence"]["significance"]
    # Scores kept before per-example results existed cannot be tested.
    run.metrics_before.pop("per_item")
    ft.store.save_run(run)
    untested = ft.verdict(run.run_id)
    assert untested["verdict"] == "untested" and not untested["better"]
    assert untested["evidence"]["significance"] is None
    # A newer score on the same set replaces an older one.
    for eval_id, value, when in (
        ("old", 0.9, "2026-10-01"),
        ("new", 0.95, "2026-10-02"),
    ):
        ft.store.save_eval(
            {
                "eval_id": eval_id,
                "dataset_id": run.dataset_id,
                "dataset_name": "Set",
                "split": "test",
                "count": 20,
                "model": {"type": "run", "run_id": run.run_id, "label": "x"},
                "metrics": {"accuracy": value, "per_item": _items(value)},
                "created_at": when,
            }
        )
    ft.store.save_eval(
        {
            "eval_id": "base",
            "dataset_id": run.dataset_id,
            "dataset_name": "Set",
            "split": "test",
            "count": 20,
            "model": {"type": "base", "base_model": run.base_model, "label": "b"},
            "metrics": {"accuracy": 0.3, "per_item": _items(0.3)},
            "created_at": "2026-10-02",
        }
    )
    latest = ft.verdict(run.run_id)
    assert latest["evidence"]["fine_tuned"] == 0.95 and latest["gain_points"] == 65.0
    assert latest["verdict"] == "better" and latest["evidence"]["frozen"]


@pytest.mark.unit
def test_training_data_is_checked_before_a_run(ft):
    d = ft.create_dataset("decision", "memory_routing", name="Routing")
    rows = [
        Example(
            dataset_id=d.dataset_id,
            input={"state": f"note {i}"},
            label={"memory_type": {"choice": "knowledge_base"}},
            status="accepted",
            split="validation" if i >= 10 else "train",
            group=f"g{i}",
        )
        for i in range(14)
    ]
    ft.store.add_examples(rows)
    data = ft._own_snapshot(ft._dataset(d.dataset_id))
    # With no test split, half of validation becomes the test set, so
    # calibration (fitted on validation) is never scored on its own rows.
    assert len(data["test"]) == 2 and len(data["validation"]) == 2
    assert not {r["example_id"] for r in data["test"]} & {
        r["example_id"] for r in data["validation"]
    }
    # An option removed after labelling: a clear error, not a crash later.
    spec = ft._dataset(d.dataset_id).spec
    spec["questions"]["memory_type"]["criteria"].pop("knowledge_base")
    ft.update_dataset(d.dataset_id, spec=spec)
    with pytest.raises(
        FineTuningError, match="14 accepted examples .*knowledge_base ×14"
    ):
        ft._snapshot(ft._dataset(d.dataset_id))


@pytest.mark.unit
def test_the_registry_lists_every_version_per_job(ft, tmp_path):
    first = _run(ft, tmp_path)
    second = _run(ft, tmp_path)
    second.parent_run_id = first.run_id
    ft.store.save_run(second)
    failed = _run(ft, tmp_path, kind="embedding", task="retrieval")
    failed.status, failed.error = "failed", "out of memory"
    ft.store.save_run(failed)
    banking = ft.create_dataset(
        "decision",
        "custom",
        name="Banking",
        spec={
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Classify the banking intent.",
                    "criteria": {"a": "A", "b": "B"},
                }
            }
        },
    )
    custom = TrainingRun(
        dataset_id=banking.dataset_id,
        kind="decision",
        task="custom",
        base_model="m",
        backbone="embedding",
        status="succeeded",
    )
    ft.store.save_run(custom)
    ft.swap("memory_routing", second.run_id)
    reg = ft.registry()
    jobs = {j["key"]: j for j in reg["jobs"]}
    routing = jobs["memory_routing"]
    assert [v["label"] for v in routing["versions"]] == ["v2", "v1"]
    assert (
        routing["versions"][0]["in_use"]
        and routing["versions"][0]["lineage"] == "from v1"
    )
    assert routing["versions"][0]["model"]["on_disk"]
    assert routing["versions"][0]["training"]["start"] == "v1"
    assert jobs["embedding"]["failed"][0]["error"] == "out of memory"
    assert any(
        b["model_id"] == "google/embeddinggemma-2" for b in jobs["embedding"]["bases"]
    )
    # A custom question is its own job, numbered on its own.
    banking_job = next(j for j in reg["jobs"] if j["key"].startswith("custom:"))
    assert banking_job["title"] == "Classify the banking intent (2 options)"
    assert [v["label"] for v in banking_job["versions"]] == ["v1"] and not banking_job[
        "memory_job"
    ]
    assert reg["totals"]["versions"] == 3 and reg["totals"]["failed"] == 1


@pytest.mark.unit
def test_a_comparison_pools_every_test_set_both_models_were_scored_on(ft, tmp_path):
    older = _run(ft, tmp_path)
    newer = _run(ft, tmp_path)
    sets = [ft.create_dataset("decision", "memory_routing", name=n) for n in ("A", "B")]
    for d in sets:
        ft.update_dataset(d.dataset_id, test_set=True)
    # Same score on A; the newer version is far better on B.
    scores = {
        ("A", "old"): 1.0,
        ("A", "new"): 1.0,
        ("B", "old"): 0.4,
        ("B", "new"): 0.9,
    }
    for d in sets:
        for who, run in (("old", older), ("new", newer)):
            value = scores[(d.name, who)]
            ft.store.save_eval(
                {
                    "eval_id": f"{d.name}-{who}",
                    "dataset_id": d.dataset_id,
                    "dataset_name": d.name,
                    "split": "test",
                    "count": 30 if d.name == "A" else 20,
                    "model": {"type": "run", "run_id": run.run_id, "label": who},
                    "metrics": {
                        "accuracy": value,
                        "per_item": _items(value, 30 if d.name == "A" else 20),
                    },
                    "created_at": "2026-10-13T00:00:00",
                }
            )
    ft.swap("memory_routing", older.run_id)
    v = ft.verdict(newer.run_id)
    e = v["evidence"]
    assert [s["name"] for s in e["sets"]] == ["A", "B"] and e["count"] == 50
    assert e["fine_tuned"] == pytest.approx((30 + 18) / 50)
    assert e["in_use"]["value"] == pytest.approx((30 + 8) / 50)
    assert v["gain_points"] == 20.0 and v["verdict"] == "better"


@pytest.mark.unit
def test_mongodb_stages_in_its_own_collection_and_keeps_the_previous_vectors():
    mongomock = pytest.importorskip("mongomock")
    from memorizz.finetune import staging
    from memorizz.memory_provider.mongodb.vector_staging import (
        STAGING_COLLECTION,
        MongoDBVectorStaging,
    )

    db = mongomock.MongoClient()["memorizz"]
    kb = db[MemoryType.KNOWLEDGE_BASE.value]
    first = kb.insert_one({"content": "Deploys run at 9.", "embedding": [0.5, 0.5]})
    kb.insert_one(
        {"_id": "note-2", "content": "Standups at 10.", "embedding": [0.5, 0.5]}
    )

    class _Store(MongoDBVectorStaging):
        def __init__(self):
            self.db = db

        def _collection(self, memory_type):
            return db[memory_type.value]

        def list_all(self, memory_type):
            return [
                {**row, "_id": str(row["_id"])} for row in db[memory_type.value].find()
            ]

    class _Model:
        def get_embeddings(self, texts):
            # The documents keep their live vectors until the switch.
            assert kb.find_one({"_id": first.inserted_id})["embedding"] == [0.5, 0.5]
            return [[float(len(t)), 1.0] for t in texts]

    store = _Store()
    report = staging.migrate(
        store, _Model(), dimensions=2, stores=[MemoryType.KNOWLEDGE_BASE]
    )
    assert report["stores"]["knowledge_base"]["embedded"] == 2
    assert kb.find_one({"_id": first.inserted_id})["embedding"] == [17.0, 1.0]
    assert "embedding_next" not in kb.find_one({"_id": "note-2"})
    assert store.vector_previous() == {"dimensions": 2, "rows": 2}
    assert db[STAGING_COLLECTION].count_documents({"slot": "next"}) == 0

    # Back again with the kept vectors: nothing changed, nothing embedded.
    class _Old:
        def get_embeddings(self, texts):
            raise AssertionError("the kept vectors are reused")

    back = staging.migrate(
        store,
        _Old(),
        dimensions=2,
        reuse_previous=True,
        stores=[MemoryType.KNOWLEDGE_BASE],
    )
    assert back["stores"]["knowledge_base"]["kept"] == 2
    assert kb.find_one({"_id": "note-2"})["embedding"] == [0.5, 0.5]
