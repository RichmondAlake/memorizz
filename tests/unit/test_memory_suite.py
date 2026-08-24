import json
from pathlib import Path

import pytest

from memorizz.benchmarks.memory_suite import (
    BENCHMARK_CATALOG,
    FusionConfig,
    MemoryBenchmarkCase,
    MemoryDocument,
    MemorySuiteRunner,
    assess_comparability,
    build_query_variants,
)
from memorizz.benchmarks.memory_suite import datasets as dataset_module
from memorizz.benchmarks.memory_suite import (
    derive_semantic_memories,
    fuse_rankings,
    get_protocol_manifest,
    load_benchmark_cases,
    verify_dataset,
)
from memorizz.benchmarks.memory_suite.scoring import (
    answer_f1,
    parse_judge_response,
    reader_response_needs_repair,
    retrieval_metrics,
)
from memorizz.enums import MemoryType
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _locomo_data(root: Path) -> None:
    _write_json(
        root / "locomo10.json",
        [
            {
                "conversation": {
                    "speaker_a": "Ada",
                    "speaker_b": "Ben",
                    "session_1_date_time": "10:00 AM on 01 January, 2025",
                    "session_1": [
                        {"speaker": "Ada", "text": "I moved to Paris yesterday."},
                        {"speaker": "Ben", "text": "I still live in Rome."},
                    ],
                },
                "qa": [
                    {
                        "question": "Where did Ada move?",
                        "answer": "Paris",
                        "evidence": ["D1:1"],
                        "category": 4,
                    }
                ],
            }
        ],
    )
    _write_json(
        root / "locomo_plus.json",
        [
            {
                "cue_dialogue": "A: Please never recommend red shoes.\nB: Understood.",
                "trigger_query": "A: Which red shoes should I buy?",
                "time_gap": "two weeks",
                "relation_type": "preference",
            }
        ],
    )


def test_catalog_contains_requested_paper_suites():
    assert set(BENCHMARK_CATALOG) == {
        "agentmembench",
        "longmemeval-v2",
        "locomo-plus",
        "beam",
        "memoryagentbench",
    }
    assert BENCHMARK_CATALOG["agentmembench"].paper_url.endswith("2608.00009")
    assert BENCHMARK_CATALOG["memoryagentbench"].paper_url.endswith("2507.05257")


def test_loads_agentmembench_and_locomo_plus_official_shapes(tmp_path):
    _locomo_data(tmp_path)
    agent_cases = load_benchmark_cases(
        "agentmembench", tmp_path, variant="locomo", limit=1
    )
    assert agent_cases[0].category == "single-hop"
    assert agent_cases[0].answers == ("Paris",)
    assert agent_cases[0].relevant_source_ids == ("D1:1",)
    assert agent_cases[0].documents[0].metadata["event_time"] == ("2025-01-01T10:00:00")
    assert "yesterday = 2024-12-31" in agent_cases[0].documents[0].content

    cognitive = load_benchmark_cases(
        "locomo-plus", tmp_path, variant="cognitive", limit=1
    )[0]
    assert cognitive.category == "cognitive"
    assert cognitive.scorer == "llm_judge"
    assert cognitive.relevant_source_ids == ("cue:0:1", "cue:0:2")
    assert cognitive.documents[0].metadata["event_type"] == "cognitive_cue"
    assert all("Which red shoes" not in doc.content for doc in cognitive.documents)
    semantic_records = derive_semantic_memories(cognitive.documents)
    grouped = [
        document
        for document in semantic_records
        if document.metadata.get("semantic_memory_type") == "event-summary"
    ]
    assert grouped
    assert grouped[0].linked_source_ids == ("cue:0:1", "cue:0:2")
    assert grouped[0].metadata["event_time"]


def test_reader_evidence_renders_linked_sources_as_distinct_json_ids():
    runner = MemorySuiteRunner.__new__(MemorySuiteRunner)
    runner.max_evidence_chars = 10_000

    evidence = runner._evidence_text(
        [
            {
                "content": "A source-linked lesson.",
                "linked_source_ids": ["cue:0:1", "cue:0:2"],
            }
        ]
    )

    assert 'source_ids=["cue:0:1", "cue:0:2"]' in evidence
    assert "source=cue:0:1,cue:0:2" not in evidence


def test_loads_longmemeval_v2_official_shape(tmp_path):
    (tmp_path / "haystacks").mkdir()
    (tmp_path / "questions.jsonl").write_text(
        json.dumps(
            {
                "id": "q1",
                "domain": "web",
                "question": "What was selected?",
                "question_type": "dynamic_state_tracking",
                "answer": "Blue plan",
                "image": None,
                "eval_function": "llm",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "trajectories.jsonl").write_text(
        json.dumps(
            {
                "id": "t1",
                "domain": "web",
                "goal": "Choose a plan",
                "states": [
                    {
                        "state_index": 0,
                        "step": 1,
                        "action": "click",
                        "observation": "Selected Blue plan",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "haystacks" / "lme_v2_small.json", {"q1": ["t1"]})
    case = load_benchmark_cases(
        "longmemeval-v2", tmp_path, variant="small-web", limit=1
    )[0]
    assert case.category == "dynamic-state-tracking"
    assert case.relevant_source_ids == ("t1",)
    assert case.documents[0].parent_source_id == "t1"


def test_loads_beam_official_shape(tmp_path):
    chat = tmp_path / "chats" / "100K" / "chat_0"
    _write_json(
        chat / "chat.json",
        [
            {
                "batch_number": 1,
                "turns": [
                    [
                        {"id": 7, "role": "user", "content": "My code is cobalt."},
                        {"id": 8, "role": "assistant", "content": "Noted."},
                    ]
                ],
            }
        ],
    )
    _write_json(
        chat / "probing_questions" / "probing_questions.json",
        {
            "information_extraction": [
                {
                    "question": "What is my code?",
                    "ideal_answer": "cobalt",
                    "rubric": ["Must state cobalt."],
                    "source_chat_ids": [7],
                }
            ]
        },
    )
    case = load_benchmark_cases("beam", tmp_path, variant="128k", limit=1)[0]
    assert case.category == "information-extraction"
    assert case.answers == ("cobalt",)
    assert case.relevant_source_ids == ("7",)


def test_loads_memoryagentbench_official_export(tmp_path):
    _write_json(
        tmp_path / "export.json",
        [
            {
                "context": "The launch code is zephyr. " * 20,
                "questions": ["What is the launch code?"],
                "answers": ["zephyr"],
                "metadata": {"source": "event_qa"},
            }
        ],
    )
    case = load_benchmark_cases(
        "memoryagentbench", tmp_path, variant="accurate-retrieval", limit=1
    )[0]
    assert case.scorer == "substring_exact_match"
    assert case.metadata["competency"] == "accurate-retrieval"


def test_scoring_and_retrieval_metrics():
    assert answer_f1("Paris, France", "Paris") == pytest.approx(2 / 3)
    assert answer_f1("2023-05-07", "7 May 2023") == 1.0
    assert answer_f1("May 7, 2023", "7 May 2023") == 1.0
    metrics = retrieval_metrics(
        [
            {"source_id": "noise"},
            {"source_id": "D1:1#part-1", "parent_source_id": "D1:1"},
        ],
        ["D1:1"],
    )
    assert metrics == {
        "recall_at_k": 1.0,
        "mrr": 0.5,
        "ndcg_at_k": pytest.approx(1 / 1.584962500721156),
    }
    linked_metrics = retrieval_metrics(
        [
            {
                "source_id": "derived",
                "parent_source_id": "cue:1",
                "linked_source_ids": ["cue:1", "cue:2"],
            }
        ],
        ["cue:1", "cue:2"],
    )
    assert linked_metrics == {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0}
    assert (
        parse_judge_response('prefix {"score": 0.5, "reason": "partial"}')["score"]
        == 0.5
    )


def test_filesystem_store_many_persists_one_reloadable_batch(tmp_path):
    root = tmp_path / "batch-provider"
    provider = FileSystemProvider(FileSystemConfig(root_path=root, use_faiss=False))
    identifiers = provider.store_many(
        [
            {"_id": "one", "content": "First", "embedding": [1.0, 0.0]},
            {"_id": "two", "content": "Second", "embedding": [0.0, 1.0]},
        ],
        MemoryType.KNOWLEDGE_BASE,
        memory_id="suite",
    )
    provider.close()
    reloaded = FileSystemProvider(FileSystemConfig(root_path=root, use_faiss=False))
    try:
        rows = reloaded.list_all(MemoryType.KNOWLEDGE_BASE)
    finally:
        reloaded.close()
    assert identifiers == ["one", "two"]
    assert {row["_id"] for row in rows} == {"one", "two"}
    assert all(row["memory_id"] == "suite" for row in rows)


class _FakeEmbeddings:
    @staticmethod
    def get_embedding(text: str):
        lowered = text.lower()
        return [1.0, 0.0] if "paris" in lowered or "france" in lowered else [0.0, 1.0]

    def get_embeddings(self, texts):
        return [self.get_embedding(text) for text in texts]


class _CountingEmbeddings(_FakeEmbeddings):
    def __init__(self):
        self.batch_calls = 0

    def get_embeddings(self, texts):
        self.batch_calls += 1
        return super().get_embeddings(texts)


class _FakeModel:
    def __init__(self):
        self.calls = 0

    def generate_text(self, prompt, instructions=None):
        self.calls += 1
        if "grader" in str(instructions).lower():
            return '{"score": 1, "reason": "grounded"}'
        return "Paris"

    def get_last_usage(self):
        return {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}


class _MemAgentFakeModel(_FakeModel):
    model_name = "memory-suite-test"
    provider = "test"

    def generate(self, messages, tools=None):
        self.calls += 1
        return '{"answer":"Paris","source_ids":["fact"],"abstained":false}'


class _RepairModel(_FakeModel):
    def generate_text(self, prompt, instructions=None):
        self.calls += 1
        return '{"answer":"Paris","source_ids":["fact"],"abstained":false}'


def test_runner_ingests_shared_corpus_once_and_reports_zero_cost(tmp_path):
    documents = (
        MemoryDocument(source_id="fact", content="France's capital is Paris."),
        MemoryDocument(source_id="noise", content="Italy's capital is Rome."),
    )
    cases = [
        MemoryBenchmarkCase.create(
            case_id="one",
            benchmark_id="agentmembench",
            corpus_id="shared",
            category="single-hop",
            documents=documents,
            question="What is the capital of France?",
            answers=["Paris"],
            relevant_source_ids=["fact"],
            scorer="answer_f1",
        ),
        MemoryBenchmarkCase.create(
            case_id="two",
            benchmark_id="agentmembench",
            corpus_id="shared",
            category="single-hop",
            documents=documents,
            question="Name France's capital.",
            answers=["Paris"],
            relevant_source_ids=["fact"],
            scorer="llm_judge",
        ),
    ]
    fake_model = _FakeModel()
    runner = MemorySuiteRunner(
        workspace=tmp_path / "workspace",
        model_name="open-test-model",
        embedding_model="open-test-embedding",
        model=fake_model,
        judge_model=fake_model,
        embedding_manager=_FakeEmbeddings(),
        top_k=2,
    )
    try:
        report = runner.run(cases, variant="locomo", dataset_path=tmp_path)
    finally:
        runner.close()
    assert report["overall_score"] == 1.0
    assert report["efficiency"]["corpus_count"] == 1
    assert report["efficiency"]["document_count"] == 2
    assert report["retrieval"]["recall_at_k"] == 1.0
    assert report["answer_quality"]["faithfulness"] == 1.0
    assert report["usage"]["cost_usd"] == 0.0
    assert report["metadata"]["dataset_subset_fingerprint"].startswith("sha256:")
    # Two retrieved readers, two oracle readers, two local-judge calls (the
    # llm-judge case is scored in both lanes), and two faithfulness calls.
    assert fake_model.calls == 8


def test_runner_reports_estimated_gpt_5_5_cost(tmp_path):
    case = MemoryBenchmarkCase.create(
        case_id="hosted",
        benchmark_id="agentmembench",
        corpus_id="shared",
        category="single-hop",
        documents=[
            MemoryDocument(source_id="fact", content="France's capital is Paris.")
        ],
        question="What is the capital of France?",
        answers=["Paris"],
        relevant_source_ids=["fact"],
        scorer="answer_f1",
    )
    fake_model = _FakeModel()
    runner = MemorySuiteRunner(
        workspace=tmp_path / "workspace",
        model_provider="openai",
        model_name="gpt-5.5",
        judge_model_name="gpt-5.5",
        model=fake_model,
        judge_model=fake_model,
        embedding_manager=_FakeEmbeddings(),
        top_k=1,
    )
    try:
        report = runner.run([case], variant="locomo", dataset_path=tmp_path)
    finally:
        runner.close()

    # Retrieved reader, oracle reader, and faithfulness: three calls reporting
    # 5 input + 1 output tokens each.
    assert report["usage"]["total_tokens"] == 18
    assert report["usage"]["cost_usd"] == pytest.approx(0.000165)
    assert report["usage"]["cost_is_estimate"] is True
    assert report["metadata"]["model_provider"] == "openai"


def test_corpus_embeddings_are_reused_across_reader_runs(tmp_path):
    case = MemoryBenchmarkCase.create(
        case_id="cache",
        benchmark_id="agentmembench",
        corpus_id="shared-cache",
        category="single-hop",
        documents=[MemoryDocument(source_id="fact", content="Paris is in France.")],
        question="Where is Paris?",
        answers=["France"],
        relevant_source_ids=["fact"],
    )
    cache_dir = tmp_path / "corpus-cache"
    first_embeddings = _CountingEmbeddings()
    first = MemorySuiteRunner(
        workspace=tmp_path / "reader-a",
        model=_FakeModel(),
        judge_model=_FakeModel(),
        embedding_manager=first_embeddings,
        corpus_cache_dir=cache_dir,
        oracle_reader=False,
    )
    try:
        first_report = first.run([case], variant="locomo", dataset_path=tmp_path)
    finally:
        first.close()

    second_embeddings = _CountingEmbeddings()
    second = MemorySuiteRunner(
        workspace=tmp_path / "reader-b",
        model=_FakeModel(),
        judge_model=_FakeModel(),
        embedding_manager=second_embeddings,
        corpus_cache_dir=cache_dir,
        oracle_reader=False,
    )
    try:
        second_report = second.run([case], variant="locomo", dataset_path=tmp_path)
    finally:
        second.close()

    assert first_embeddings.batch_calls == 1
    assert first_report["efficiency"]["embedding_cache_misses"] == 1
    assert second_embeddings.batch_calls == 0
    assert second_report["efficiency"]["embedding_cache_hits"] == 1


def test_strict_protocol_gate_is_fail_closed_and_evidence_backed():
    manifest = get_protocol_manifest("agentmembench")
    diagnostic = assess_comparability(
        manifest,
        {
            "official_runner": False,
            "official_scorer": False,
            "dataset_verified": True,
            "num_samples": 491,
            "reader_model": "Qwen2.5-7B-Instruct",
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "top_k": 5,
            "decoding": {"quantization": "4-bit", "strategy": "greedy"},
            "dataset_revision": "r1",
            "dataset_fingerprint": "sha256:data",
            "prompt_hash": "sha256:prompt",
            "scorer_hash": "sha256:scorer",
            "dependency_lock_hash": "sha256:lock",
            "hardware": {"machine": "test"},
            "seed": 0,
        },
        profile="paper",
    )
    assert diagnostic["paper_comparable"] is False
    assert any(
        "official runner" in reason
        for reason in diagnostic["non_comparability_reasons"]
    )

    official = assess_comparability(
        manifest,
        {**diagnostic["evidence"], "official_runner": True, "official_scorer": True},
        profile="paper",
    )
    assert official["paper_comparable"] is True
    assert official["comparison_label"] == "Official protocol"


def test_weighted_rrf_can_recover_a_lexical_candidate_without_gold_labels():
    rows = fuse_rankings(
        [
            (
                "semantic:0",
                [
                    {"source_id": "noise", "content": "A general update."},
                    {"source_id": "fact", "content": "Ada moved to Paris."},
                ],
                1.0,
            ),
            (
                "lexical",
                [{"source_id": "fact", "content": "Ada moved to Paris."}],
                1.0,
            ),
        ],
        question="Where did Ada move?",
        config=FusionConfig(top_k=1, candidate_pool_size=8, lexical_weight=1.0),
    )
    assert rows[0]["source_id"] == "fact"
    assert rows[0]["_retrieval"]["lane_ranks"] == {
        "semantic:0": 2,
        "lexical": 1,
    }


def test_fusion_deduplicates_original_and_semantic_copy_by_parent_source():
    rows = fuse_rankings(
        [
            (
                "semantic:0",
                [
                    {
                        "source_id": "turn-1#semantic-1",
                        "parent_source_id": "turn-1",
                        "content": "Constraint: protect time by declining work.",
                    },
                    {
                        "source_id": "turn-1",
                        "parent_source_id": "turn-1",
                        "content": "I am learning to say no because it protects my time.",
                    },
                    {"source_id": "turn-2", "content": "A different fact."},
                ],
                1.0,
            )
        ],
        question="What should I do when overwhelmed?",
        config=FusionConfig(top_k=2, candidate_pool_size=8),
    )
    parent_ids = [row.get("parent_source_id") or row["source_id"] for row in rows]
    assert parent_ids.count("turn-1") == 1
    assert parent_ids == ["turn-1", "turn-2"]


def test_fusion_keeps_broadest_provenance_for_one_parent_rank():
    rows = fuse_rankings(
        [
            (
                "semantic:0",
                [
                    {
                        "source_id": "cue:1#semantic",
                        "parent_source_id": "cue:1",
                        "content": "One statement.",
                    },
                    {
                        "source_id": "cue-group#semantic-summary",
                        "parent_source_id": "cue:1",
                        "linked_source_ids": ["cue:1", "cue:2"],
                        "content": "Statement plus its related advice.",
                    },
                ],
                1.0,
            )
        ],
        question="What prior advice applies?",
        config=FusionConfig(top_k=1, candidate_pool_size=8),
    )
    assert rows[0]["source_id"] == "cue-group#semantic-summary"
    assert rows[0]["linked_source_ids"] == ["cue:1", "cue:2"]
    assert rows[0]["_retrieval"]["lane_ranks"] == {"semantic:0": 1}


def test_query_expansion_links_capacity_questions_to_boundary_memories():
    variants = build_query_variants(
        "What should I do when I feel overwhelmed by commitments?"
    )
    assert variants[0].startswith("What should I do")
    assert any(
        "boundaries" in variant and "protecting time" in variant for variant in variants
    )


def test_semantic_query_variants_do_not_double_count_one_source():
    rows = fuse_rankings(
        [
            (
                "semantic:0",
                [
                    {"source_id": "direct", "content": "direct evidence"},
                    {"source_id": "repeated", "content": "repeated evidence"},
                ],
                1.0,
            ),
            (
                "semantic:1",
                [{"source_id": "repeated", "content": "repeated evidence"}],
                1.0,
            ),
        ],
        question="unrelated query",
        config=FusionConfig(top_k=2, candidate_pool_size=8),
    )
    assert [row["source_id"] for row in rows] == ["direct", "repeated"]
    assert rows[1]["_retrieval"]["lane_ranks"] == {
        "semantic:0": 2,
        "semantic:1": 1,
    }


def test_semantic_memory_derivation_is_source_linked_and_query_independent():
    derived = derive_semantic_memories(
        [
            MemoryDocument(
                source_id="turn-1",
                content="I am learning to say no because it protects my time.",
            )
        ]
    )
    assert derived
    assert derived[0].parent_source_id == "turn-1"
    assert derived[0].metadata["semantic_memory_type"] == "constraint"
    assert "boundaries" in derived[0].metadata["semantic_concepts"]
    assert "Source statement" in derived[0].content
    assert "Reason or consequence: it protects my time" in derived[0].content

    quoted = derive_semantic_memories(
        [
            MemoryDocument(
                source_id="turn-2",
                content="After learning to say 'no', I feel less stressed.",
            )
        ]
    )
    assert quoted[0].metadata["semantic_memory_type"] == "constraint"
    assert {
        "capacity",
        "boundaries",
    }.issubset(set(quoted[0].metadata["semantic_concepts"]))


def test_reader_repair_is_bounded_to_malformed_json_like_output(tmp_path):
    assert reader_response_needs_repair('{"answer": "Paris"') is True
    assert reader_response_needs_repair("Paris") is False
    model = _RepairModel()
    runner = MemorySuiteRunner(
        workspace=tmp_path / "repair",
        model=model,
        judge_model=model,
        embedding_manager=_FakeEmbeddings(),
        oracle_reader=False,
    )
    case = MemoryBenchmarkCase.create(
        case_id="repair",
        benchmark_id="agentmembench",
        corpus_id="repair",
        category="single-hop",
        documents=[],
        question="Where?",
        answers=["Paris"],
    )
    try:
        parsed = runner._parse_reader_with_repair(
            case, '{"answer": "Paris"', lane="generation"
        )
    finally:
        runner.close()
    assert model.calls == 1
    assert parsed["structured"] is True
    assert parsed["repair_attempted"] is True
    assert parsed["repair_succeeded"] is True


def test_full_memagent_mode_runs_automatic_memory_retrieval(tmp_path):
    model = _MemAgentFakeModel()
    template = MemAgentModel(
        agent_id="evaluation-template",
        name="Evaluation template",
        memory_types=None,
        semantic_cache=False,
    )
    case = MemoryBenchmarkCase.create(
        case_id="memagent",
        benchmark_id="agentmembench",
        corpus_id="memagent-corpus",
        category="single-hop",
        documents=[
            MemoryDocument(source_id="fact", content="France's capital is Paris."),
            MemoryDocument(source_id="noise", content="Italy's capital is Rome."),
        ],
        question="What is the capital of France?",
        answers=["Paris"],
        relevant_source_ids=["fact"],
    )
    runner = MemorySuiteRunner(
        workspace=tmp_path / "full-agent",
        model=model,
        judge_model=model,
        embedding_manager=_FakeEmbeddings(),
        top_k=2,
        candidate_pool_size=8,
        evaluation_mode="memagent",
        agent_template=template,
        oracle_reader=False,
    )
    try:
        assert MemoryType.KNOWLEDGE_BASE in runner.agent.active_memory_types
        report = runner.run([case], variant="locomo", dataset_path=tmp_path)
        evidence = runner.agent.last_retrieval_evidence()
    finally:
        runner.close()
    assert report["overall_score"] == 1.0
    assert report["retrieval"]["recall_at_k"] == 1.0
    assert report["retrieval"]["basis"] == "memagent_automatic_retrieval"
    assert report["retrieval"]["fusion"] is None
    assert report["retrieval"]["strategy"].startswith("automatic_provider_search")
    assert report["answer_quality"]["retrieved_evidence_score"] is None
    assert report["answer_quality"]["memagent_end_to_end_score"] == 1.0
    assert report["metadata"]["evaluation_mode"] == "memagent"
    assert report["metadata"]["features_exercised"]["memagent_context_assembly"]
    assert report["metadata"]["features_exercised"]["semantic_cache_decision"] is False
    assert report["metadata"]["agent_evaluation_overrides"]["isolated_runtime_memory"]
    assert evidence["selected_count"] >= 1
    assert evidence["query_variants"][0] == case.question


def test_dataset_verification_records_required_asset_checksums(tmp_path):
    _locomo_data(tmp_path)
    report = verify_dataset("locomo-plus", data_path=tmp_path, variant="cognitive")
    assert report["ready"] is True
    assert report["dataset_fingerprint"].startswith("sha256:")
    assert all(
        row.get("sha256", "").startswith("sha256:") for row in report["artifacts"]
    )


def test_beam_verification_normalizes_remote_and_reports_partial_coverage(
    tmp_path, monkeypatch
):
    chat = tmp_path / "chats" / "100K" / "1"
    _write_json(chat / "chat.json", [])
    _write_json(chat / "probing_questions" / "probing_questions.json", {})
    (tmp_path / ".git").mkdir()
    values = {
        ("rev-parse", "HEAD"): get_protocol_manifest("beam").synchronization_revision,
        ("remote", "get-url", "origin"): (
            "https://github.com/mohammadtavakoli78/BEAM.git"
        ),
    }
    monkeypatch.setattr(
        dataset_module,
        "_git_value",
        lambda _root, *args: values.get(tuple(args)),
    )

    report = verify_dataset("beam", data_path=tmp_path, variant="128k")

    assert report["source"]["path"] == str(tmp_path)
    assert report["source"]["ready"] is True
    assert report["ready_for_diagnostic"] is True
    assert report["coverage_complete"] is False
    assert report["next_actions"]
    assert report["artifacts"][0]["coverage"] == {
        "available_conversations": 1,
        "expected_conversations": 20,
        "complete": False,
    }
