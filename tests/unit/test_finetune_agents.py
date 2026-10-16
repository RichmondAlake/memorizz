"""Per-agent routers and rerankers, and what a swap touches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.finetune import (
    FineTuning,
    FineTuningError,
    agent_models,
    clients,
    reranking,
    routing,
)
from memorizz.finetune import swap as swapping
from memorizz.finetune.models import TrainingRun
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

AGENTS = [
    {"agent_id": "a1", "name": "Alpha", "memory_ids": ["m1"]},
    {"agent_id": "a2", "name": "Beta", "memory_ids": ["m2"]},
]


@pytest.fixture
def ft(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    for key in swapping.EMBEDDING_ENV + ("MEMORIZZ_RERANKER_MODEL",):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(clients, "_DIR", tmp_path / "clients")
    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )
    provider.list_memagents = lambda: [dict(a) for a in AGENTS]
    agent_models.forget()
    yield FineTuning(provider, home=str(tmp_path / "home" / "finetune"))
    agent_models.forget()


def _run(ft, tmp_path, kind="decision", task="memory_routing"):
    dataset = ft.create_dataset(kind, task)
    run = TrainingRun(
        dataset_id=dataset.dataset_id,
        kind=kind,
        task=task,
        base_model="Qwen/Qwen3-Embedding-0.6B",
        backbone="embedding" if kind != "reranker" else "cross_encoder",
        status="succeeded",
        metrics_after={"accuracy": 0.9},
    )
    run.output_dir = str(tmp_path / "runs" / run.run_id)
    (Path(run.output_dir) / "model").mkdir(parents=True)
    run.progress = {"model_dir": str(Path(run.output_dir) / "model")}
    ft.store.save_run(run)
    return run


@pytest.mark.unit
def test_an_agent_can_use_its_own_router_or_none(ft, tmp_path, monkeypatch):
    store_router, own_router = _run(ft, tmp_path), _run(ft, tmp_path)
    ft.swap("memory_routing", store_router.run_id)
    used = []

    def decide(run_id, state, questions=None):
        used.append(run_id)
        return {
            "memory_type": {
                "type": "choice",
                "choice": "entity_memory",
                "confidence": 0.9,
            }
        }

    monkeypatch.setattr(ft, "decide", decide)
    monkeypatch.setattr(routing, "_service", lambda provider: ft)
    ft.set_agent_model("a1", "memory_routing", own_router.run_id)
    assert (
        routing.route(ft.provider, "Priya prefers calls", memory_id="m1")["run_id"]
        == own_router.run_id
    )
    assert (
        routing.route(ft.provider, "Priya prefers calls", memory_id="m2")["run_id"]
        == store_router.run_id
    )
    assert (
        routing.route(ft.provider, "Priya prefers calls")["run_id"]
        == store_router.run_id
    )
    ft.set_agent_model("a1", "memory_routing", "off")
    assert routing.route(ft.provider, "Priya prefers calls", agent_id="a1") is None
    ft.set_agent_model("a1", "memory_routing", "default")
    assert (
        routing.route(ft.provider, "x", agent_id="a1")["run_id"] == store_router.run_id
    )
    assert used == [
        own_router.run_id,
        store_router.run_id,
        store_router.run_id,
        store_router.run_id,
    ]


@pytest.mark.unit
def test_an_agent_can_use_its_own_reranker_or_none(ft, tmp_path, monkeypatch):
    own = _run(ft, tmp_path, kind="reranker", task="relevance")
    loaded = []
    monkeypatch.setattr(
        reranking,
        "scorer_for",
        lambda path: loaded.append(path) or (lambda q, ps: [1.0] * len(ps)),
    )
    assert (
        reranking.rerank_for("q", provider=ft.provider, agent_id="a1") is None
    )  # nothing chosen anywhere
    ft.set_agent_model("a1", "reranking", own.run_id)
    assert reranking.rerank_for("q", provider=ft.provider, agent_id="a1")(
        ["a", "b"]
    ) == [1.0, 1.0]
    assert loaded == [ft.model_dir(own.run_id)]
    assert (
        reranking.rerank_for("q", provider=ft.provider, memory_id="m2") is None
    )  # Beta follows the store
    ft.set_agent_model("a1", "reranking", "off")
    monkeypatch.setenv("MEMORIZZ_RERANKER_MODEL", str(tmp_path))
    assert reranking.rerank_for("q", provider=ft.provider, agent_id="a1") is None
    assert reranking.rerank_for("q", provider=ft.provider, agent_id="a2") is not None


@pytest.mark.unit
def test_agent_choices_are_checked(ft, tmp_path):
    router = _run(ft, tmp_path)
    embedder = _run(ft, tmp_path, kind="embedding", task="retrieval")
    with pytest.raises(FineTuningError, match="one model per store"):
        ft.set_agent_model("a1", "embedding", embedder.run_id)
    with pytest.raises(FineTuningError, match="not one for reranking"):
        ft.set_agent_model("a1", "reranking", router.run_id)
    with pytest.raises(FineTuningError, match="No agent nobody"):
        ft.set_agent_model("nobody", "memory_routing", router.run_id)
    chosen = ft.set_agent_model("a2", "memory_routing", router.run_id)["choices"][
        "memory_routing"
    ]
    assert chosen["mode"] == "run" and chosen["run_id"] == router.run_id
    reg = {j["key"]: j for j in ft.registry()["jobs"]}
    assert [a["name"] for a in reg["memory_routing"]["agents"]] == ["Alpha", "Beta"]
    assert reg["memory_routing"]["agents"][1]["choice"]["run_id"] == router.run_id
    assert reg["embedding"]["agents_total"] == 2


@pytest.mark.unit
def test_a_swap_lists_the_store_agents_and_other_processes(ft, tmp_path):
    router = _run(ft, tmp_path)
    ft.set_agent_model("a2", "memory_routing", router.run_id)
    clients.register("ui", ft.provider, port=8805)
    store = clients.store_identity(ft.provider)
    # Another process (our parent stands in for it) on the same store.
    other = {
        "pid": os.getppid(),
        "kind": "mcp",
        "store": store,
        "embedding": {"model": "nomic-embed-text"},
    }
    (tmp_path / "clients" / f"{os.getppid()}.json").write_text(json.dumps(other))
    # One on another store, and one that has exited.
    (tmp_path / "clients" / "1.json").write_text(
        json.dumps({**other, "pid": 1, "store": {"type": "oracle", "id": "X@db"}})
    )
    (tmp_path / "clients" / "999999.json").write_text(
        json.dumps({**other, "pid": 999999})
    )
    impact = ft.swap_impact("memory_routing")
    assert impact["store"] == {
        "type": "filesystem",
        "id": str((tmp_path / "mem").resolve()),
    }
    assert impact["agents"]["total"] == 2 and impact["agents"]["follow_store"] == 1
    assert [a["name"] for a in impact["agents"]["own_choice"]] == ["Beta"]
    assert [p["pid"] for p in impact["processes"]] == [
        os.getppid()
    ]  # not ours, not another store's
    assert not impact["restart_needed"]
    assert not (tmp_path / "clients" / "999999.json").exists()  # exited: dropped
    embedding = ft.swap_impact("embedding")
    assert embedding["restart_needed"] and embedding["reembeds"]
    assert embedding["agents"]["own_choice"] == []  # one embedding model per store


@pytest.mark.unit
def test_store_names_never_carry_credentials():
    oracle = SimpleNamespace(
        config=SimpleNamespace(user="app", dsn="db:1521/FREEPDB1", password="s3cret")
    )
    assert clients.store_identity(oracle) == {
        "type": "oracle",
        "id": "APP@db:1521/FREEPDB1",
    }
    mongo = SimpleNamespace(
        config=SimpleNamespace(
            uri="mongodb+srv://me:s3cret@cluster0.x.net/?retryWrites=true",
            db_name="memorizz",
        )
    )
    name = clients.store_identity(mongo)
    assert name == {
        "type": "mongodb",
        "id": "cluster0.x.net/memorizz",
    } and "s3cret" not in json.dumps(name)


@pytest.mark.unit
def test_older_processes_on_the_same_database_server_are_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(clients, "_DIR", tmp_path / "clients")
    store = {"type": "oracle", "id": "APP@127.0.0.1:1529/FREEPDB1"}
    assert clients._server(store) == "127.0.0.1:1529"
    assert clients._server({"type": "filesystem", "id": "/x"}) is None

    def fake_run(args, **kwargs):
        if args[0] == "ps":
            out = "101 python memorizz ui --port 8767\n102 python memorizz mcp serve\n103 python other-app\n"
        else:  # lsof: who is connected to 127.0.0.1:1529
            assert "-iTCP@127.0.0.1:1529" in args
            out = "101\n"
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(clients.subprocess, "run", fake_run)
    found = clients.unregistered(store)
    assert found == [
        {"pid": 101, "kind": "ui", "port": 8767, "same_server": True},
        {"pid": 102, "kind": "mcp", "port": None, "same_server": False},
    ]


@pytest.mark.unit
def test_a_runs_curves_and_the_data_it_trained_on(ft, tmp_path):
    from memorizz.finetune.models import Example

    run = _run(ft, tmp_path)
    folder = Path(run.output_dir)
    events = [
        {
            "type": "eval",
            "data": {
                "phase": "before",
                "metrics": {"accuracy": 0.2, "per_item": {"a": 0}},
            },
        }
    ]
    for step in range(1, 901):  # loss falls, then levels off
        events.append(
            {
                "type": "step",
                "data": {
                    "step": step,
                    "total": 900,
                    "epoch": step / 300,
                    "loss": 2.0 * 0.995**step + 0.1,
                },
            }
        )
        if step % 300 == 0:
            accuracy = {300: 0.6, 600: 0.8, 900: 0.7}[step]
            events.append(
                {
                    "type": "eval",
                    "data": {
                        "phase": "validation",
                        "epoch": step // 300,
                        "step": step,
                        "metrics": {"accuracy": accuracy, "confusion": {}},
                    },
                }
            )
    events.append(
        {"type": "eval", "data": {"phase": "after", "metrics": {"accuracy": 0.75}}}
    )
    (folder / "events.jsonl").write_text(
        "\n".join(json.dumps({"timestamp": "t", **e}) for e in events) + "\n"
    )
    c = ft.curves(run.run_id)
    assert (
        len(c["steps"]) <= 401 and c["steps"][-1]["step"] == 900
    )  # thinned, last step kept
    assert [v["epoch"] for v in c["validation"]] == [1, 2, 3]
    assert "confusion" not in c["validation"][0]["metrics"]  # numbers only
    assert c["before"]["accuracy"] == 0.2 and "per_item" not in c["before"]
    notes = " ".join(c["convergence"])
    assert "Loss fell from" in notes and "levelled off" in notes
    assert "peaked at epoch 2" in notes  # 80% then 70%: may be overfitting

    replayed = ft.create_dataset("decision", "memory_routing", name="Earlier round")
    rows = [
        {
            "example_id": "e1",
            "split": "train",
            "input": {"state": "note one"},
            "label": {"memory_type": {"choice": "knowledge_base"}},
        },
        {
            "example_id": "e2",
            "split": "test",
            "input": {"state": "note two"},
            "label": {"memory_type": {"choice": "entity_memory"}},
        },
        {
            "example_id": "r1",
            "split": "train",
            "replay": replayed.dataset_id,
            "input": {"state": "old note"},
            "label": {"memory_type": {"choice": "knowledge_base"}},
        },
    ]
    (folder / "data.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ft.store.add_examples(
        [
            Example(
                dataset_id=run.dataset_id,
                input={"state": "added later"},
                label={"memory_type": {"choice": "summaries"}},
                status="accepted",
                example_id="e9",
            )
        ]
    )
    data = ft.run_data(run.run_id)
    assert data["summary"]["splits"] == {"train": 2, "test": 1}
    assert data["summary"]["replayed"] == {"Earlier round": 1}
    assert data["summary"]["labels"]["knowledge_base"] == {"train": 2}
    assert data["since"] == {
        "added": 1,
        "removed": 2,
    }  # e9 added; e1, e2 no longer accepted here
    only_replayed = ft.run_data(run.run_id, source="replay")
    assert [r["example_id"] for r in only_replayed["rows"]] == ["r1"] and only_replayed[
        "rows"
    ][0]["replayed_from"] == "Earlier round"
    assert [r["example_id"] for r in ft.run_data(run.run_id, split="test")["rows"]] == [
        "e2"
    ]


@pytest.mark.unit
def test_speed_is_measured_next_to_what_runs_now(ft, tmp_path, monkeypatch):
    import time

    from memorizz.finetune.models import Example

    older, newer = _run(ft, tmp_path), _run(ft, tmp_path)
    for run in (older, newer):
        ft.store.add_examples(
            [
                Example(
                    dataset_id=run.dataset_id,
                    input={"state": f"note {i}"},
                    label={"memory_type": {"choice": "knowledge_base"}},
                    status="accepted",
                    split="test",
                )
                for i in range(8)
            ]
        )
    cost = {older.run_id: 0.004, newer.run_id: 0.001}
    calls = []

    def timer(card, *, run=None, settings=None, path=""):
        calls.append((card, run["run_id"]))
        return (lambda state: time.sleep(cost[run["run_id"]])), "cpu"

    monkeypatch.setattr(ft, "_timer", timer)
    ft.swap("memory_routing", older.run_id)
    result = ft.speed(newer.run_id, samples=6)
    rows = {r["role"]: r for r in result["rows"]}
    assert set(rows) == {"fine-tuned", "in use now"}  # never its base: same size
    assert rows["fine-tuned"]["n"] == 6
    assert rows["fine-tuned"]["median_ms"] < rows["in use now"]["median_ms"]
    assert rows["fine-tuned"]["p95_ms"] >= rows["fine-tuned"]["median_ms"]
    assert ("memory_routing", older.run_id) in calls
    assert result["unit"] == "per memory routed"
    # Kept with the run.
    assert ft.verdict(newer.run_id)["speed"]["rows"] == result["rows"]
    # Once it is in use, it is timed next to what Swap back would restore:
    # here no router at all, which adds no time.
    ft.swap("memory_routing", newer.run_id)
    again = {r["role"]: r for r in ft.speed(newer.run_id, samples=6)["rows"]}
    assert again["before the swap"]["n"] == 0
    assert again["before the swap"]["label"] == "No router"
    assert again["before the swap"]["device"] == "nothing runs"
