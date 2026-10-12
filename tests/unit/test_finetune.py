"""Fine-tuning: tasks, formats, synthetic data, provider-held datasets, runs,
CLI, MCP and UI, without downloading a model."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.finetune import FineTuning, FineTuningError
from memorizz.finetune.bases import (
    decision_options,
    decision_query,
    lm_messages,
    resolve_base,
)
from memorizz.finetune.evaluate import (
    classification_metrics,
    fit_temperature,
    ranking_metrics,
    softmax,
)
from memorizz.finetune.formats import render, systemone_row, unsloth_bundle
from memorizz.finetune.models import Example
from memorizz.finetune.store import PARTITION, FineTuneStore
from memorizz.finetune.synthetic import (
    JevVerifier,
    Teacher,
    generate_decision_examples,
    generate_pairs,
    parse_json,
    plan_slots,
    state_text,
    stratified_splits,
)
from memorizz.finetune.tasks import (
    answer_for,
    gold_key,
    memory_routing_spec,
    only_question,
    options,
    relevance_spec,
    validate_question,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def _provider(tmp_path: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path / "mem"), lazy_vector_indexes=True)
    )


@pytest.fixture
def ft(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    return FineTuning(_provider(tmp_path), home=str(tmp_path / "home" / "finetune"))


# -- tasks and formats -----------------------------------------------------------


@pytest.mark.unit
def test_routing_question_lists_memory_stores_and_a_do_not_store_option():
    qid, question = only_question(memory_routing_spec())
    keys = [k for k, _ in options(question)]
    assert qid == "memory_type" and keys[-1] == "do_not_store"
    assert {"knowledge_base", "entity_memory", "conversation_memory"} <= set(keys)
    assert "agents" not in keys and "semantic_cache" not in keys
    assert gold_key(question, answer_for(question, "entity_memory")) == "entity_memory"
    _, relevance = only_question(relevance_spec())
    assert [k for k, _ in options(relevance)] == ["false", "true"]
    assert gold_key(relevance, {"noul": 0.8}) == "true"
    with pytest.raises(ValueError):
        validate_question(
            "q", {"type": "choice", "instructions": "x", "criteria": {"only": "one"}}
        )
    with pytest.raises(ValueError):
        validate_question(
            "q", {"type": "score", "instructions": "x", "criteria": ["a"]}
        )


@pytest.mark.unit
def test_each_base_model_family_sees_its_own_input_format():
    _, q = only_question(memory_routing_spec())
    qwen = decision_query("qwen3-embedding", q, "Priya prefers calls")
    assert qwen.startswith("Instruct: Which MemoRizz memory store") and qwen.endswith(
        "\nQuery:Priya prefers calls"
    )
    gemma = decision_query("embeddinggemma", q, "Priya prefers calls")
    assert gemma.startswith("task: classification | query: Which MemoRizz")
    assert decision_options("embeddinggemma", q)[0].startswith(
        "title: conversation_memory | text: "
    )
    assert decision_options("qwen3-embedding", q)[0].startswith("conversation_memory: ")
    messages, shown = lm_messages(
        q,
        {"source": "chat", "content": "hi"},
        order=list(range(len(shown_keys := [k for k, _ in options(q)])))[::-1],
    )
    assert shown == shown_keys[::-1]
    assert "State:\nsource: chat\ncontent: hi" in messages[1]["content"]
    assert f"A: {shown[0]}:" in messages[1]["content"]
    assert (
        resolve_base("google/embeddinggemma-2", "decision").family == "embeddinggemma"
    )
    assert resolve_base("/models/my-qwen3-0.6b", "decision").backbone == "lm"
    assert resolve_base("anything", "reranker").backbone == "cross_encoder"


@pytest.mark.unit
def test_exports_match_systemone_unsloth_and_trainer_formats():
    spec = memory_routing_spec()
    row = {
        "example_id": "e1",
        "split": "test",
        "input": {"state": "Priya prefers calls"},
        "label": {"memory_type": {"choice": "entity_memory"}},
    }
    so = systemone_row(spec, row)
    gold = json.loads(so["gold"])["memory_type"]
    assert (
        gold["choice"] == "entity_memory"
        and gold["probabilities"]["entity_memory"] == 1.0
    )
    assert (
        json.loads(so["questions"])["memory_type"]["type"] == "choice"
        and so["split"] == "test"
    )
    qwen = resolve_base("Qwen/Qwen3-Embedding-0.6B", "decision")
    jev = render("jevembed", "decision", spec, qwen, row)[0]
    assert jev["answer"] == "entity_memory" and jev["options"][jev["label"]].startswith(
        "entity_memory:"
    )
    chat = render(
        "chat", "decision", spec, resolve_base("Qwen/Qwen3-0.6B", "decision"), row
    )[0]
    letter = chat["messages"][-1]["content"]
    assert f"{letter}: entity_memory:" in chat["messages"][1]["content"]
    triple = {
        "input": {
            "query": "When is the demo?",
            "positive": "Friday 3pm.",
            "negatives": ["Paris is in France."],
        }
    }
    gemma = resolve_base("google/embeddinggemma-2", "embedding")
    st = render(
        "sentence-transformers", "embedding", {"instruction": "x"}, gemma, triple
    )[0]
    assert st == {
        "anchor": "task: search result | query: When is the demo?",
        "positive": "title: none | text: Friday 3pm.",
        "negative": "title: none | text: Paris is in France.",
    }
    ce = render(
        "cross-encoder",
        "reranker",
        {},
        resolve_base("cross-encoder/ms-marco-MiniLM-L6-v2", "reranker"),
        triple,
    )
    assert [r["label"] for r in ce] == [1.0, 0.0]
    with pytest.raises(ValueError):
        render("cross-encoder", "decision", spec, qwen, row)
    bundle = unsloth_bundle("unsloth/Qwen3.5-4B")
    assert (
        'BASE_MODEL = "unsloth/Qwen3.5-4B"' in bundle["train_unsloth.py"]
        and "FastDecisionModel" in bundle["train_unsloth.py"]
    )


# -- metrics ---------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_cover_accuracy_calibration_and_ranking():
    probs = [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]]
    m = classification_metrics(probs, [0, 1, 1, 1], ["a", "b"])
    assert m["accuracy"] == 0.75 and m["confusion"]["matrix"] == [[1, 0], [1, 2]]
    assert 0 < m["ece"] < 1 and m["nll"] > 0 and "a" in m["per_class"]
    logits = [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
    assert fit_temperature(logits, [0, 1, 1, 0]) > fit_temperature(logits, [0, 1, 0, 1])
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    r = ranking_metrics([[(0.9, 1), (0.5, 0)], [(0.2, 1), (0.8, 0)], [(0.1, 0)]])
    assert r["queries"] == 2 and r["mrr@10"] == 0.75 and r["recall@1"] == 0.5


# -- synthetic data --------------------------------------------------------------


@pytest.mark.unit
def test_teacher_replies_are_parsed_and_unwrapped():
    assert parse_json('```json\n{"states": ["a"]}\n```') == {"states": ["a"]}
    assert parse_json('Sure! {"labels": []} hope that helps') == {"labels": []}
    with pytest.raises(ValueError):
        parse_json("no json here")
    assert state_text({"state": " wrapped "}) == "wrapped"
    assert (
        state_text({"id": 1, "body": "the longest string wins", "x": "no"})
        == "the longest string wins"
    )
    slots = plan_slots({"a": 5, "b": 2}, phenomena=["p1", "p2"])
    assert [(s["target"], s["count"]) for s in slots] == [("a", 4), ("a", 1), ("b", 2)]
    examples = [
        Example(
            dataset_id="d",
            input={"state": str(i)},
            label={"q": {"choice": "x" if i < 10 else "y"}},
        )
        for i in range(13)
    ]
    stratified_splits(examples, lambda e: e.label["q"]["choice"])
    x = [e.split for e in examples[:10]]
    assert (
        x.count("test") == 2
        and x.count("validation") == 1
        and examples[10].split == "train"
        and examples[12].split == "test"
    )


class _ScriptedModel:
    """Stands in for an LLM: answers generate and verify prompts by rule."""

    def __init__(self, wrong_label: str = ""):
        self.wrong_label = wrong_label
        self.calls = 0

    def generate_text(self, prompt, instructions=None, **kwargs):
        self.calls += 1
        if "Write " in prompt and "states" in prompt:
            target = prompt.split('the correct option is "')[1].split('"')[0]
            count = int(prompt.split("Write ")[1].split(" ")[0])
            # One wrapped state per batch: the parser must unwrap it.
            states = [{"state": f"{target} wrapped {self.calls}"}] + [
                f"{target} case {self.calls}-{i}" for i in range(count - 1)
            ]
            return json.dumps({"states": states})
        listing = prompt.split("Examples:\n")[1]
        labels = []
        for i, chunk in enumerate(listing.split("\n\n[")):
            text = chunk.split("\n", 1)[1] if "\n" in chunk else ""
            answer = text.split(" ")[0]
            if answer == self.wrong_label:
                answer = "do_not_store"
            labels.append({"id": i + 1, "answer": answer, "unambiguous": True})
        return json.dumps({"labels": labels})


def _teacher(model) -> Teacher:
    teacher = Teacher({"provider": "ollama", "model": "fake"}, model=model)
    teacher.model.generate_text = model.generate_text  # bypass the measured wrapper
    return teacher


@pytest.mark.unit
def test_decision_generation_balances_labels_and_tops_up_rejected_ones():
    spec = {
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"alpha": "A", "beta": "B", "do_not_store": "C"},
            }
        }
    }
    model = _ScriptedModel(wrong_label="beta")  # the verifier never agrees on beta
    examples, report = generate_decision_examples(
        "ds", spec, _teacher(model), count=12, max_rounds=3
    )
    per_label = report["per_label"]
    assert (
        per_label["alpha"] == 4
        and per_label["do_not_store"] == 4
        and per_label["beta"] == 0
    )
    assert report["short"] == ["beta"] and report["rounds"] == 3
    assert report["rejected"]["verifier_disagreed"] >= 4
    assert all(e.verified and e.source == "synthetic" for e in examples)
    assert any("wrapped" in e.input["state"] for e in examples)
    assert {e.split for e in examples if e.label["q"]["choice"] == "alpha"} >= {
        "train",
        "test",
    }


class _PairModel:
    def generate_text(self, prompt, instructions=None, **kwargs):
        if "Passages:" in prompt:
            texts = [
                chunk.split("\n", 1)[1].split("\n\n")[0]
                for chunk in prompt.split("\n[")[1:]
            ]
            return json.dumps(
                {
                    "items": [
                        {
                            "id": i + 1,
                            "query": f"what is in {t}?",
                            "hard_negative": f"a related but useless note {i}",
                        }
                        for i, t in enumerate(texts)
                    ]
                }
            )
        n = prompt.count("Query:")
        verdicts = []
        for i, block in enumerate(prompt.split("\n\n[")[:n]):
            verdicts.append({"id": i + 1, "relevant": "useless" not in block})
        return json.dumps({"verdicts": verdicts})


@pytest.mark.unit
def test_pair_generation_keeps_confirmed_positives_and_true_negatives():
    passages = [
        {"id": f"m{i}", "text": f"memory number {i} with details"} for i in range(5)
    ]
    triples, report = generate_pairs(passages, _teacher(_PairModel()), per_call=4)
    assert report["kept"] == 5 and all(
        t["verified"] and t["negatives"] for t in triples
    )
    assert {t["source_id"] for t in triples} == {f"m{i}" for i in range(5)}


@pytest.mark.unit
def test_jev_verifier_reads_systemone_answers(monkeypatch):
    calls = []

    class _Response:
        ok, status_code = True, 200

        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    class _Session:
        def post(self, url, headers, json, timeout):
            calls.append((url, json["model"], json["questions"]))
            answers = {}
            for qid, question in json["questions"].items():
                answers[qid] = (
                    {"type": "noul", "noul": 0.9}
                    if question["type"] == "noul"
                    else {
                        "type": "choice",
                        "probabilities": {"alpha": 0.7, "beta": 0.3},
                    }
                )
            return _Response(
                {"answers": answers, "usage": {"input_tokens": 100 * len(answers)}}
            )

    import requests

    monkeypatch.setattr(requests, "Session", _Session)
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    verifier = JevVerifier(min_confidence=0.75)
    question = {
        "type": "choice",
        "instructions": "Pick",
        "criteria": {"alpha": "A", "beta": "B"},
    }
    verdicts = verifier.decide_many(question, [f"state {i}" for i in range(10)])
    assert (
        len(verdicts) == 10
        and verdicts[0]["answer"] == "alpha"
        and verdicts[0]["unambiguous"] is False
    )  # 0.7 < 0.75
    # Eight items per request, each in its own question.
    assert sorted(len(c[2]) for c in calls) == [2, 8]
    assert (
        "state 3" in calls[0][2]["item_3"]["instructions"]
        or "state 3" in calls[1][2]["item_3"]["instructions"]
    )
    assert verifier.judge_pairs([("q", "p")]) == [True]
    assert calls[0][0].endswith("/v1/systemone") and calls[0][1] == "jev-latest"
    assert verifier.usage()["input_tokens"] == 1100 and verifier.usage()["cost_usd"] > 0
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(ValueError):
        JevVerifier()


# -- storage in the memory provider -------------------------------------------------


@pytest.mark.unit
def test_datasets_live_in_the_memory_provider_as_private_records(tmp_path):
    provider = _provider(tmp_path)
    store = FineTuneStore(provider, home=tmp_path / "ft")
    ft = FineTuning(provider, home=str(tmp_path / "ft"))
    dataset = ft.create_dataset("decision", "memory_routing", name="Routing")
    ft.add_example(
        dataset.dataset_id,
        input={"state": "Priya prefers calls"},
        label={"memory_type": {"choice": "entity_memory"}},
    )
    rows = provider.list_all("shared_memory")
    assert {row["record_type"] for row in rows} == {
        "finetune_dataset",
        "finetune_example",
    }
    assert all(
        row["scope"] == "private" and row["agent_id"] == PARTITION for row in rows
    )
    assert store.get_dataset(dataset.dataset_id).name == "Routing"
    assert len(store.examples(dataset.dataset_id)) == 1
    # A second process (or the UI after a restart) sees the same data.
    assert (
        FineTuning(provider, home=str(tmp_path / "ft")).datasets()[0]["counts"]["total"]
        == 1
    )
    files = store.run_files("r1").ensure()
    files.append_event("step", {"step": 1})
    files.append_event("step", {"step": 2})
    assert [e["data"]["step"] for e in files.events(after=1)] == [2]
    files.write_state({"phase": "running"})
    assert files.state()["phase"] == "running" and not files.cancel_requested
    files.request_cancel()
    assert files.cancel_requested
    assert (
        ft.delete_dataset(dataset.dataset_id)
        and provider.list_all("shared_memory") == []
    )


@pytest.mark.unit
def test_examples_are_validated_against_the_question(ft):
    dataset = ft.create_dataset("decision", "memory_routing")
    with pytest.raises(FineTuningError, match="not an option"):
        ft.add_example(
            dataset.dataset_id,
            input={"state": "x"},
            label={"memory_type": {"choice": "nowhere"}},
        )
    with pytest.raises(FineTuningError, match="input.state"):
        ft.add_example(
            dataset.dataset_id,
            input={"text": "x"},
            label={"memory_type": {"choice": "knowledge_base"}},
        )
    retrieval = ft.create_dataset("embedding", "retrieval")
    with pytest.raises(FineTuningError, match="query"):
        ft.add_example(retrieval.dataset_id, input={"query": "q"})
    with pytest.raises(FineTuningError, match="Unknown task"):
        ft.create_dataset("decision", "nonsense")
    example = ft.add_example(
        dataset.dataset_id,
        input={"state": "x"},
        label={"memory_type": {"choice": "knowledge_base"}},
    )
    assert ft.relabel(example.example_id, "tool_log").label == {
        "memory_type": {"choice": "tool_log"}
    }


@pytest.mark.unit
def test_stored_memories_become_routing_examples_labelled_by_store(ft, tmp_path):
    from memorizz.enums.memory_type import MemoryType

    provider = ft.provider
    provider.store(
        {"content": "Leeds is in West Yorkshire, England.", "memory_id": "m"},
        MemoryType.KNOWLEDGE_BASE,
    )
    provider.store(
        {
            "role": "user",
            "content": "Can we move the demo to Friday?",
            "memory_id": "m",
        },
        MemoryType.CONVERSATION_MEMORY,
    )
    dataset = ft.create_dataset("decision", "memory_routing")
    report = ft.import_memories(dataset.dataset_id)
    assert report["by_store"] == {"knowledge_base": 1, "conversation_memory": 1}
    labels = {
        e["input"]["state"]: e["label"]["memory_type"]["choice"]
        for e in ft.examples(dataset.dataset_id)
    }
    assert labels["Leeds is in West Yorkshire, England."] == "knowledge_base"
    assert labels["user: Can we move the demo to Friday?"] == "conversation_memory"
    # A tool is routed by its name and signature, not its description alone.
    from memorizz.finetune.service import _memory_text

    tool = {
        "name": "search_web",
        "signature": "(query: str)",
        "description": "Search the web.",
    }
    assert (
        _memory_text("toolbox", tool) == "Tool search_web(query: str): Search the web."
    )
    # Importing again adds nothing, and fine-tuning's own records are never imported.
    assert ft.import_memories(dataset.dataset_id)["added"] == 0
    assert ft.examples(dataset.dataset_id, status="pending")


# -- training runs (with a stand-in trainer) ------------------------------------------


def _seed(ft, n=14):
    dataset = ft.create_dataset("decision", "memory_routing")
    for i in range(n):
        label = ["knowledge_base", "entity_memory"][i % 2]
        ft.add_example(
            dataset.dataset_id,
            input={"state": f"example {i}"},
            label={"memory_type": {"choice": label}},
            split="train",
        )
    return dataset


@pytest.mark.unit
def test_a_run_trains_on_a_snapshot_and_records_metrics_in_the_provider(
    ft, monkeypatch
):
    from memorizz.finetune import worker

    seen = {}

    def fake_trainer(config, data, out_dir, report):
        seen.update(config=config, counts={k: len(v) for k, v in data.items()})
        report.step(1, 1, epoch=1.0, loss=0.5)
        model_dir = out_dir / "model"
        model_dir.mkdir()
        return {
            "metrics_before": {"accuracy": 0.3},
            "metrics_after": {"accuracy": 0.9},
            "model_dir": str(model_dir),
            "device": "cpu",
        }

    monkeypatch.setattr(worker, "trainer_for", lambda config: fake_trainer)
    monkeypatch.setattr(
        "memorizz.finetune.service.environment",
        lambda: {
            "ready": True,
            "missing": [],
            "unsloth": {"available": False, "reason": "no gpu"},
            "install": "",
        },
    )
    dataset = _seed(ft)
    with pytest.raises(FineTuningError, match="at least 8"):
        ft.start_run(_seed(ft, n=3).dataset_id, detach=False)
    run = ft.start_run(
        dataset.dataset_id, base_model="Qwen/Qwen3-Embedding-0.6B", detach=False
    )
    assert run.status == "succeeded" and run.metrics_after == {"accuracy": 0.9}
    # Everything was marked train: some were held out for scoring.
    assert (
        seen["counts"]["train"] < 14
        and seen["counts"]["test"] + seen["counts"]["validation"] > 0
    )
    assert (
        seen["config"]["backbone"] == "embedding"
        and seen["config"]["family"] == "qwen3-embedding"
    )
    stored = ft.run(run.run_id)
    assert stored["status"] == "succeeded" and stored["progress"]["model_dir"].endswith(
        "model"
    )
    assert [e["type"] for e in ft.events(run.run_id)][-2:] == ["step", "finished"]
    record = ft.activate(run.run_id, "memory_routing")
    assert ft.active()["memory_routing"]["run_id"] == run.run_id == record["run_id"]

    class _Model:
        def __init__(self, path):
            pass

        def decide(self, state, questions=None):
            return {
                "memory_type": {
                    "type": "choice",
                    "choice": "entity_memory",
                    "confidence": 0.8,
                    "probabilities": {},
                }
            }

    monkeypatch.setattr("memorizz.finetune.inference.DecisionModel", _Model)
    assert ft.route("Priya prefers calls")["memory_type"] == "entity_memory"
    with pytest.raises(FineTuningError):
        ft.activate(run.run_id, "embedding")
    assert ft.delete_run(run.run_id) and "memory_routing" not in ft.active()


@pytest.mark.unit
def test_a_failing_trainer_leaves_a_failed_run_with_the_error(ft, monkeypatch):
    from memorizz.finetune import worker

    def broken(config, data, out_dir, report):
        raise RuntimeError(
            "out of memory with key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
        )

    monkeypatch.setattr(worker, "trainer_for", lambda config: broken)
    monkeypatch.setattr(
        "memorizz.finetune.service.environment",
        lambda: {
            "ready": True,
            "missing": [],
            "unsloth": {"available": False, "reason": ""},
            "install": "",
        },
    )
    run = ft.start_run(_seed(ft).dataset_id, detach=False)
    assert (
        run.status == "failed"
        and "out of memory" in run.error
        and "sk-proj" not in run.error
    )


@pytest.mark.unit
def test_training_refuses_without_its_packages(ft, monkeypatch):
    monkeypatch.setattr(
        "memorizz.finetune.service.environment",
        lambda: {
            "ready": False,
            "missing": ["peft"],
            "unsloth": {"available": False, "reason": ""},
            "install": 'pip install "memorizz[finetune]"',
        },
    )
    with pytest.raises(FineTuningError, match="peft"):
        ft.start_run(_seed(ft).dataset_id)


# -- CLI, MCP and UI ------------------------------------------------------------------


@pytest.mark.unit
def test_cli_creates_fills_and_exports_a_dataset(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from memorizz.cli.app import app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MEMORIZZ_BACKEND", "filesystem")
    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(tmp_path / "mem"))
    runner = CliRunner()
    created = runner.invoke(
        app,
        [
            "finetune",
            "create",
            "decision",
            "memory_routing",
            "--name",
            "Routing",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    dataset_id = json.loads(created.stdout)["dataset_id"]
    added = runner.invoke(
        app,
        [
            "finetune",
            "add",
            dataset_id,
            "--state",
            "Priya prefers calls",
            "--label",
            "entity_memory",
            "--json",
        ],
    )
    assert added.exit_code == 0, added.output
    shown = json.loads(
        runner.invoke(app, ["finetune", "show", dataset_id, "--json"]).stdout
    )
    assert shown["counts"]["total"] == 1
    exported = runner.invoke(
        app, ["finetune", "export", dataset_id, "--format", "systemone"]
    )
    assert json.loads(exported.stdout.splitlines()[0])["gold"]
    bad = runner.invoke(
        app, ["finetune", "add", dataset_id, "--state", "x", "--label", "nowhere"]
    )
    assert bad.exit_code == 1 and "not an option" in bad.output
    listed = runner.invoke(app, ["finetune", "tasks"])
    assert (
        "memory_routing" in listed.stdout and "google/embeddinggemma-2" in listed.stdout
    )


@pytest.mark.unit
def test_mcp_tools_scope_and_gate_fine_tuning(tmp_path, monkeypatch):
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server.auth import RequestIdentity
    from memorizz.mcp_server.config import (
        EXECUTE_SCOPE,
        READ_SCOPE,
        WRITE_SCOPE,
        MemorizzMCPServerConfig,
    )
    from memorizz.mcp_server.finetune_tools import FineTuneTools
    from memorizz.mcp_server.runtime import MemorizzRuntime, MemorizzServerError

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    config = MemorizzMCPServerConfig(allow_training=False)
    runtime = MemorizzRuntime(
        config,
        provider=_provider(tmp_path),
        approval_store=SQLiteApprovalStore(tmp_path / "a.sqlite3"),
    )
    tools = FineTuneTools(runtime)
    me = RequestIdentity(
        principal=None,
        scopes=frozenset({READ_SCOPE, WRITE_SCOPE, EXECUTE_SCOPE}),
        authenticated=True,
    )
    reader = RequestIdentity(
        principal=None, scopes=frozenset({READ_SCOPE}), authenticated=True
    )
    dataset = tools.create_dataset(
        me, kind="decision", task="memory_routing", name="MCP"
    )["dataset"]
    added = tools.add_examples(
        dataset["dataset_id"],
        [
            {
                "input": {"state": "Priya prefers calls"},
                "label": {"memory_type": {"choice": "entity_memory"}},
            }
        ],
        me,
    )
    assert added["added"] == 1
    assert (
        tools.get_dataset(dataset["dataset_id"], reader)["dataset"]["counts"]["total"]
        == 1
    )
    with pytest.raises(MemorizzServerError):
        tools.create_dataset(reader, kind="decision", task="memory_routing")
    with pytest.raises(MemorizzServerError, match="disabled"):
        tools.start_training(dataset["dataset_id"], me)
    with pytest.raises(MemorizzServerError, match="not in this dataset"):
        tools.review_examples(dataset["dataset_id"], me, accept=["someone-elses"])
    listing = tools.list_datasets(reader, include_catalog=True)
    assert (
        listing["datasets"][0]["name"] == "MCP" and listing["training_allowed"] is False
    )
    assert {t["task"] for t in listing["tasks"]} >= {
        "memory_routing",
        "relevance",
        "retrieval",
    }


@pytest.mark.unit
def test_mcp_server_registers_the_fine_tuning_tools(tmp_path, monkeypatch):
    import asyncio

    pytest.importorskip("mcp")
    from memorizz.approval import SQLiteApprovalStore
    from memorizz.mcp_server import create_memorizz_mcp_server
    from memorizz.mcp_server.config import MemorizzMCPServerConfig
    from memorizz.mcp_server.runtime import MemorizzRuntime

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    config = MemorizzMCPServerConfig()
    runtime = MemorizzRuntime(
        config,
        provider=_provider(tmp_path),
        approval_store=SQLiteApprovalStore(tmp_path / "a.sqlite3"),
    )
    server = create_memorizz_mcp_server(config, runtime=runtime)
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {
        "memorizz_list_finetune_datasets",
        "memorizz_get_finetune_dataset",
        "memorizz_create_finetune_dataset",
        "memorizz_add_finetune_examples",
        "memorizz_review_finetune_examples",
        "memorizz_generate_finetune_data",
        "memorizz_start_finetune_run",
        "memorizz_get_finetune_run",
        "memorizz_finetune_decide",
    } <= names


@pytest.mark.unit
def test_ui_page_and_api_manage_datasets(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    with patch.dict(
        state._state, {"provider": _provider(tmp_path), "provider_type": "filesystem"}
    ):
        client = TestClient(create_app(), follow_redirects=False)
        page = client.get("/fine-tuning")
        assert page.status_code == 200 and "ft-app" in page.text
        overview = client.get("/api/finetune/overview").json()
        assert {t["task"] for t in overview["tasks"]} >= {"memory_routing", "relevance"}
        dataset = client.post(
            "/api/finetune/datasets",
            json={"kind": "decision", "task": "memory_routing"},
        ).json()
        url = f"/api/finetune/datasets/{dataset['dataset_id']}"
        assert (
            client.post(
                url + "/examples",
                json={
                    "input": {"state": "ok thanks"},
                    "label": {"memory_type": {"choice": "do_not_store"}},
                },
            ).status_code
            == 200
        )
        bad = client.post(
            url + "/examples",
            json={
                "input": {"state": "x"},
                "label": {"memory_type": {"choice": "nope"}},
            },
        )
        assert bad.status_code == 400 and "not an option" in bad.json()["detail"]
        example = client.get(url + "/examples").json()["examples"][0]
        assert (
            client.patch(
                f"/api/finetune/examples/{example['example_id']}",
                json={"split": "test"},
            ).json()["split"]
            == "test"
        )
        preview = client.get(url + "/preview?base_model=Qwen/Qwen3-0.6B").json()
        assert (
            preview["format"] == "chat"
            and preview["rows"][0]["messages"][-1]["role"] == "assistant"
        )
        bundle = client.get(url + "/unsloth-bundle")
        assert (
            bundle.status_code == 200
            and bundle.headers["content-type"] == "application/zip"
        )
        refused = client.post(url + "/runs", json={})
        assert refused.status_code == 400
        assert client.delete(url).json() == {"deleted": True}
