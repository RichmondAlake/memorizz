"""Fine-tuning metrics: ties, queries with nothing to rank against,
answers that never came, and the paired interval behind every verdict."""

from __future__ import annotations

import math

import pytest

from memorizz.finetune.evaluate import (
    classification_metrics,
    paired_test,
    ranking_metrics,
)
from memorizz.finetune.scoring import memory_pool, rank_held_out


@pytest.mark.unit
def test_tied_scores_are_averaged_not_credited_to_the_right_answer():
    # The right memory listed first, every score equal: a coin flip, not 1.0.
    flat = ranking_metrics([[(0.5, 1.0), (0.5, 0.0)]])
    assert flat["mrr@10"] == 0.75  # (1 + 1/2) / 2
    assert flat["recall@1"] == 0.5
    expected_ndcg = (1 + 1 / math.log2(3)) / 2
    assert flat["ndcg@10"] == round(expected_ndcg, 4)
    # Without ties nothing changes.
    clear = ranking_metrics([[(0.9, 1.0), (0.1, 0.0)], [(0.1, 1.0), (0.9, 0.0)]])
    assert clear["mrr@10"] == 0.75 and clear["recall@1"] == 0.5


@pytest.mark.unit
def test_a_query_with_only_right_answers_is_not_counted():
    out = ranking_metrics(
        [[(0.2, 1.0)], [(0.9, 1.0), (0.1, 0.0)]], ids=["only", "real"]
    )
    assert out["queries"] == 1 and out["skipped_no_negative"] == 1
    assert out["per_query"] == {"real": 1.0}


@pytest.mark.unit
def test_an_answer_that_never_came_counts_as_wrong():
    uniform = [[1 / 3] * 3] * 3
    out = classification_metrics(
        uniform, [0, 0, 1], ["a", "b", "c"], ids=["p", "q", "r"], abstained=[True] * 3
    )
    assert out["accuracy"] == 0.0 and out["abstained"] == 3
    assert out["per_item"] == {"p": 0.0, "q": 0.0, "r": 0.0}
    # Without the flag, a tie would have been read as option "a".
    assert classification_metrics(uniform, [0, 0, 1], ["a", "b", "c"])["accuracy"] > 0


@pytest.mark.unit
def test_the_paired_test_separates_a_real_gain_from_chance():
    # Right/wrong scores: McNemar's exact test on the items that differ.
    clear = paired_test(
        {str(i): 1.0 for i in range(30)}, {str(i): float(i < 10) for i in range(30)}
    )
    assert clear["differ"] == 20 and clear["p"] < 0.001
    four = paired_test(
        {str(i): float(i < 12) for i in range(20)},
        {str(i): float(i < 8) for i in range(20)},
    )
    assert four["p"] == 0.125  # 2 x 0.5**4: 4 wins and no losses could be chance
    # Graded scores (nDCG per query): exact sign flips over 3 differences.
    graded = paired_test(
        {"a": 1.0, "b": 0.8, "c": 0.5, "d": 0.4, "e": 0.3},
        {"a": 0.5, "b": 0.6, "c": 0.5, "d": 0.5, "e": 0.3},
    )
    # +0.5, +0.2, -0.1: totals with |sum| >= 0.6 are +-0.8 and +-0.6 -> 4 of 8
    assert graded["differ"] == 3 and graded["p"] == 0.5
    assert paired_test({"a": 1.0}, {"a": 0.0}) is None  # too few to say


@pytest.mark.unit
def test_retrieval_ranks_against_every_memory_and_keeps_per_query_scores():
    rows = [
        {
            "example_id": f"e{i}",
            "input": {"query": f"q{i}", "positive": f"m{i}", "negatives": [f"n{i}"]},
        }
        for i in range(3)
    ]
    pool = memory_pool({"test": rows})
    assert pool == ["m0", "n0", "m1", "n1", "m2", "n2"]
    seen = []

    def score(query, passages):
        seen.append(passages)
        return [1.0 if p == "m" + query[1:] else 0.0 for p in passages]

    out = rank_held_out("embedding", rows, pool, score)
    assert [len(c) for c in seen] == [6, 6, 6] and out["ndcg@10"] == 1.0
    assert out["per_item"] == {"e0": 1.0, "e1": 1.0, "e2": 1.0} and out["count"] == 3


@pytest.mark.unit
def test_held_out_examples_are_whole_generation_groups():
    from memorizz.finetune.models import Example
    from memorizz.finetune.synthetic import stratified_splits

    rows = [
        Example(
            dataset_id="d",
            input={"state": f"{label}{g}{i}"},
            label={"q": {"choice": label}},
            group=f"{label}:{g}",
        )
        for label in "abc"
        for g in range(5)
        for i in range(4)
    ]
    stratified_splits(rows, lambda e: e.label["q"]["choice"])
    splits_per_group = {}
    for e in rows:
        splits_per_group.setdefault(e.group, set()).add(e.split)
    assert all(len(s) == 1 for s in splits_per_group.values())
    for label in "abc":
        mine = [e.split for e in rows if e.label["q"]["choice"] == label]
        assert {"train", "validation", "test"} <= set(mine)


@pytest.mark.unit
def test_listings_are_cached_until_a_write_or_the_next_refresh(tmp_path):
    import time

    from memorizz.finetune.models import Dataset
    from memorizz.finetune.store import FineTuneStore
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )
    store = FineTuneStore(provider, home=str(tmp_path / "home"))
    other = FineTuneStore(provider, home=str(tmp_path / "home"))  # another process
    store.save_dataset(Dataset(name="A", kind="decision", task="custom"))
    assert [d.name for d in store.list_datasets()] == ["A"]
    # A write through this store shows at once.
    store.save_dataset(Dataset(name="B", kind="decision", task="custom"))
    assert {d.name for d in store.list_datasets()} == {"A", "B"}
    # Another writer's change is not read until the listing is refreshed.
    other.save_dataset(Dataset(name="C", kind="decision", task="custom"))
    assert {d.name for d in store.list_datasets()} == {"A", "B"}
    store.LIST_TTL = 0.0
    store.list_datasets()  # stale: served, refreshed in the background
    for _ in range(50):
        if {d.name for d in store.list_datasets()} == {"A", "B", "C"}:
            break
        time.sleep(0.05)
    assert {d.name for d in store.list_datasets()} == {"A", "B", "C"}
    # What a caller gets is a copy: changing it never changes the cache.
    store.list_datasets()[0].spec["x"] = 1
    assert all("x" not in d.spec for d in store.list_datasets())


@pytest.mark.unit
def test_saving_a_run_keeps_the_examples_cached(tmp_path, monkeypatch):
    from memorizz.finetune.models import Example, TrainingRun
    from memorizz.finetune.store import FineTuneStore
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )
    store = FineTuneStore(provider, home=str(tmp_path / "home"))
    store.add_examples([Example(dataset_id="d", input={"state": "x"})])
    assert len(store.examples("d")) == 1
    reads = []
    real = store._list_uncached
    monkeypatch.setattr(store, "_list_uncached", lambda *a: reads.append(a) or real(*a))
    store.save_run(
        TrainingRun(
            dataset_id="d",
            kind="decision",
            task="custom",
            base_model="m",
            backbone="embedding",
        )
    )
    assert len(store.examples("d")) == 1 and reads == []  # served from the cache
    store.add_examples([Example(dataset_id="d", input={"state": "y"})])
    assert len(store.examples("d")) == 2 and len(reads) == 1  # read again
