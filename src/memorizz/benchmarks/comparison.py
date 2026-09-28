"""Reproducible reader/reranker experiments shared by the UI and Python API.

Run as a subprocess with ``python -m memorizz.benchmarks.comparison CONFIG``.
Configuration contains model names and settings, never credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .measurement import MeasuredModel, MeasurementLedger, summarize_calls, token_price
from .memory_suite import (
    MemoryBenchmarkCase,
    MemoryDocument,
    MemorySuiteRunner,
    load_benchmark_cases,
)
from .memory_suite.runner import _bootstrap_mean_ci, _percentile
from .rerankers import Reranker


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal[
        "ollama",
        "openai",
        "anthropic",
        "azure",
        "huggingface",
        "mlx",
        "none",
        "heuristic",
        "llm",
        "cross_encoder",
        "cohere",
        "voyage",
        "jev",
    ]
    model: str = Field(default="", max_length=200)
    label: str = Field(default="", max_length=100)
    llm_provider: Literal[
        "ollama", "openai", "anthropic", "azure", "huggingface", "mlx"
    ] = "ollama"
    pricing: dict[str, float] | None = None
    search_unit_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    stream: bool = False
    jev_method: Literal["noul", "score", "choice"] = "noul"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid(self):
        if self.provider not in {"none", "heuristic"} and not self.model.strip():
            raise ValueError("A model name is required")
        allowed = {
            "reasoning_effort",
            "effort",
            "enable_prompt_caching",
            "max_completion_tokens",
            "max_tokens",
            "num_predict",
            "temperature",
            "seed",
            "think",
            "timeout",
            "host",
            "base_url",
            "api_version",
            "endpoint",
        }
        if set(self.options) - allowed:
            raise ValueError(
                "Only inference settings are allowed; credentials belong in environment variables"
            )
        if self.pricing is not None:
            if not {"input", "cached_input", "output"} <= set(self.pricing) or set(
                self.pricing
            ) - {"input", "cached_input", "output", "cache_write", "cache_write_1h"}:
                raise ValueError(
                    "Provide input, cached_input and output prices per million tokens"
                )
            token_price(self.provider, self.model, self.pricing)
        if any(isinstance(v, (dict, list)) for v in self.options.values()):
            raise ValueError("Inference options must be scalar values")
        if any(
            isinstance(v, float) and not math.isfinite(v) for v in self.options.values()
        ):
            raise ValueError("Inference options must be finite")
        for key in ("host", "base_url", "endpoint"):
            if self.options.get(key):
                url = urlsplit(str(self.options[key]))
                if url.username or url.password or url.query or url.fragment:
                    raise ValueError(
                        "Provider URLs cannot contain credentials, query parameters, or fragments"
                    )
        return self


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Model comparison", min_length=1, max_length=120)
    experiment_type: Literal["reader", "reranker", "pipeline"] = "reader"
    dataset: str = "demo"
    data_path: str = ""
    candidate_snapshot_path: str = ""
    variant: str | None = None
    limit: int = Field(default=6, ge=1, le=2000)
    readers: list[ModelSpec] = Field(min_length=1, max_length=8)
    rerankers: list[ModelSpec] = Field(
        default_factory=lambda: [ModelSpec(provider="none")], min_length=1, max_length=8
    )
    judge: ModelSpec = Field(
        default_factory=lambda: ModelSpec(provider="ollama", model="qwen2.5:3b")
    )
    repeats: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=0, ge=0, le=2147483647)
    top_k: int = Field(default=4, ge=1, le=50)
    candidate_pool_size: int = Field(default=20, ge=2, le=100)
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    oracle_reader: bool = False
    max_cost_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_seconds: int = Field(default=1800, ge=10, le=86400)
    local_hourly_usd: float = Field(default=0, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid(self):
        from .measurement import READER_PROVIDERS
        from .rerankers import RERANKERS

        host = urlsplit(self.ollama_host)
        if (
            host.scheme not in {"http", "https"}
            or not host.netloc
            or host.username
            or host.password
            or host.query
            or host.fragment
        ):
            raise ValueError(
                "Ollama host must be an HTTP(S) URL without credentials or query parameters"
            )
        if any(s.provider not in READER_PROVIDERS for s in [*self.readers, self.judge]):
            raise ValueError("Readers and judges need a generative model provider")
        if any(s.provider not in RERANKERS for s in self.rerankers):
            raise ValueError("Select a supported reranker")
        if self.experiment_type == "reader" and (
            len(self.rerankers) != 1 or self.rerankers[0].provider != "none"
        ):
            raise ValueError("Reader comparisons use fixed evidence without a reranker")
        if self.experiment_type == "reranker" and len(self.readers) != 1:
            raise ValueError("Reranker comparisons require one fixed reader")
        if self.top_k > self.candidate_pool_size:
            raise ValueError("Evidence count cannot exceed the candidate pool")
        if len(self.readers) * len(self.rerankers) * self.repeats > 40:
            raise ValueError("An experiment is limited to 40 runs")
        if self.dataset not in {"demo", "memory_checks"} and not self.data_path.strip():
            raise ValueError("Provide a dataset path")
        if self.max_cost_usd:
            for spec in [*self.readers, self.judge, *self.rerankers]:
                provider = (
                    spec.llm_provider if spec.provider == "llm" else spec.provider
                )
                if provider == "cohere":
                    if spec.search_unit_usd is None:
                        raise ValueError(
                            "A spending threshold requires Cohere pricing per search unit"
                        )
                elif token_price(provider, spec.model, spec.pricing) is None:
                    raise ValueError(
                        f"A spending threshold requires prices for {provider}:{spec.model}"
                    )
        return self


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def demo_cases() -> list[MemoryBenchmarkCase]:
    facts = [
        (
            "location",
            "Asha lives in Bristol.",
            "Which city does Asha live in?",
            "Bristol",
        ),
        (
            "preference",
            "Asha prefers tea to coffee.",
            "Which drink does Asha prefer?",
            "tea",
        ),
        (
            "project",
            "Asha's active project is called Meadow.",
            "What is Asha's active project called?",
            "Meadow",
        ),
        (
            "pet",
            "Asha has a cat named Pebble.",
            "What is the name of Asha's cat?",
            "Pebble",
        ),
        (
            "language",
            "Asha is learning Italian.",
            "Which language is Asha learning?",
            "Italian",
        ),
        (
            "schedule",
            "Asha's weekly team meeting is on Tuesday.",
            "On which day is Asha's weekly team meeting?",
            "Tuesday",
        ),
    ]
    docs = tuple(
        MemoryDocument(source_id=f"fact-{i}", content=text)
        for i, (_, text, _, _) in enumerate(facts)
    )
    return [
        MemoryBenchmarkCase.create(
            case_id=f"demo-{i}",
            benchmark_id="locomo-plus",
            corpus_id="demo",
            category=category,
            documents=docs,
            question=question,
            answers=[answer],
            relevant_source_ids=[f"fact-{i}"],
            scorer="substring_exact_match",
            metadata={"synthetic_demo": True},
        )
        for i, (category, _, question, answer) in enumerate(facts)
    ]


def load_cases(config: ComparisonConfig) -> list[MemoryBenchmarkCase]:
    if config.dataset == "demo":
        cases = demo_cases()
    elif config.dataset == "memory_checks":
        cases = memory_check_cases()
    elif config.dataset == "custom":
        path = Path(config.data_path).expanduser()
        if path.stat().st_size > 100_000_000:
            raise ValueError("Custom datasets must be smaller than 100 MB")
        text = path.read_text(encoding="utf-8")
        rows = (
            json.loads(text)
            if text.lstrip().startswith("[")
            else [json.loads(s) for s in text.splitlines() if s.strip()]
        )
        cases = []
        for row in rows:
            documents = [MemoryDocument(**d) for d in row["documents"]]
            cases.append(
                MemoryBenchmarkCase.create(
                    **{**row, "documents": documents, "benchmark_id": "locomo-plus"}
                )
            )
    else:
        cases = load_benchmark_cases(
            config.dataset,
            Path(config.data_path).expanduser(),
            variant=config.variant,
            limit=config.limit,
        )
    if not cases:
        raise ValueError("Dataset has no evaluation cases")
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be unique")
    corpora = {}
    for case in cases:
        material = json.dumps([d.to_dict() for d in case.documents], sort_keys=True)
        if case.corpus_id in corpora and corpora[case.corpus_id] != material:
            raise ValueError("Cases sharing a corpus_id must supply the same documents")
        corpora[case.corpus_id] = material
    shuffled = list(cases)
    random.Random(config.seed).shuffle(shuffled)
    return shuffled[: config.limit]


def memory_check_cases() -> list[MemoryBenchmarkCase]:
    """Six inspectable, strict memory checks; still synthetic, not a benchmark."""
    tasks = [
        (
            "preference",
            [
                "Asha avoids caffeine after lunch and chooses rooibos tea.",
                "Ben prefers espresso after lunch.",
            ],
            "What drink does Asha choose after lunch? Return only the drink name.",
            ["rooibos tea", "rooibos"],
            [0],
        ),
        (
            "update",
            [
                "On 2026-01-10 Asha lived in Bristol.",
                "On 2026-08-03 Asha moved permanently to Leeds.",
                "Asha visited York on 2026-08-10.",
            ],
            "As of 2026-09-01, which city does Asha live in? Return only the city.",
            ["Leeds"],
            [1],
        ),
        (
            "multi-hop",
            [
                "Asha's current project is Meadow.",
                "The Meadow project is led by Priya.",
                "The Cedar project is led by Owen.",
            ],
            "Who leads Asha's current project? Return only the person's first name.",
            ["Priya"],
            [0, 1],
        ),
        (
            "correction",
            [
                "A note mistakenly recorded Asha's cat as Pepper.",
                "Correction: Asha's cat is named Pebble, not Pepper.",
            ],
            "What is Asha's cat actually named? Return only the name.",
            ["Pebble"],
            [1],
        ),
        (
            "entity",
            [
                "Designer Alex Morgan works in Milan.",
                "Engineer Alex Reed works in Oslo.",
            ],
            "In which city does engineer Alex work? Return only the city.",
            ["Oslo"],
            [1],
        ),
        (
            "schedule",
            [
                "The weekly Meadow meeting used to be Tuesday at 10:00.",
                "Effective 2026-09-01, the weekly Meadow meeting moved to Thursday at 14:00.",
            ],
            "After 2026-09-01, on which weekday is the Meadow meeting? Return only the weekday.",
            ["Thursday"],
            [1],
        ),
    ]
    return [
        MemoryBenchmarkCase.create(
            case_id=f"memory-check-{i + 1}",
            benchmark_id="locomo-plus",
            corpus_id=f"memory-check-{i + 1}",
            category=category,
            documents=[
                MemoryDocument(source_id=f"check-{i + 1}-doc-{j + 1}", content=doc)
                for j, doc in enumerate(docs)
            ],
            question=question,
            answers=answers,
            relevant_source_ids=[f"check-{i + 1}-doc-{j + 1}" for j in relevant],
            scorer="exact_match",
            metadata={"synthetic_demo": True, "strict_memory_check": True},
        )
        for i, (category, docs, question, answers, relevant) in enumerate(tasks)
    ]


def accuracy_interval(correct: int, total: int) -> dict | None:
    """Wilson 95% interval stays informative even when every case passes."""
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return {
        "lower": max(0, center - half),
        "upper": min(1, center + half),
        "samples": total,
        "method": "Wilson",
    }


def case_json(case: MemoryBenchmarkCase) -> dict:
    return {**case.__dict__, "documents": [d.to_dict() for d in case.documents]}


class ComparisonRunner(MemorySuiteRunner):
    def __init__(self, *, ledger, reranker, candidates, on_case, **kwargs):
        self.ledger, self.reranker = ledger, reranker
        self.frozen_candidates, self.on_case = candidates, on_case
        super().__init__(**kwargs)

    def _retrieve(self, case, corpus):
        if case.case_id not in self.frozen_candidates:
            old = self.fusion_config
            try:
                self.fusion_config = replace(
                    old, top_k=self.candidate_pool_size, rerank_weight=0
                )
                rows, timing, variants = super()._retrieve(case, corpus)
                self.frozen_candidates[case.case_id] = {
                    "rows": rows,
                    "timing": timing,
                    "variants": variants,
                }
            finally:
                self.fusion_config = old
        snapshot = self.frozen_candidates[case.case_id]
        started = time.perf_counter()
        rows = self.reranker.rank(case.question, snapshot["rows"], self.top_k)
        timing = {
            **snapshot["timing"],
            "reranker_seconds": time.perf_counter() - started,
        }
        return rows, timing, snapshot["variants"]

    def _generate_reader(self, case, rows, *, lane):
        self.ledger.lane = "reader" if lane == "generation" else "oracle"
        started = time.perf_counter()
        result = super()._generate_reader(case, rows, lane=lane)
        result["seconds"] = time.perf_counter() - started
        return result

    def _judge(self, case, prediction):
        self.ledger.lane = "judge"
        return super()._judge(case, prediction)

    def _faithfulness(self, case, prediction, evidence):
        self.ledger.lane = "judge"
        return super()._faithfulness(case, prediction, evidence)

    def _run_case(self, case, index, total):
        self.ledger.case_id = case.case_id
        self.ledger.check()
        start = len(self.ledger.calls)
        result = super()._run_case(case, index, total)
        result["measurements"] = self.ledger.calls[start:]
        # Initial retrieval is measured once and replayed identically in all arms.
        timing = result["retrieval_timing"]
        result["retrieval_seconds"] = sum(
            float(timing[k])
            for k in (
                "semantic_seconds",
                "lexical_seconds",
                "fusion_seconds",
                "reranker_seconds",
            )
        )
        result["candidate_source_ids"] = [
            r.get("source_id") for r in self.frozen_candidates[case.case_id]["rows"]
        ]
        by_id = {
            r.get("source_id"): r for r in self.frozen_candidates[case.case_id]["rows"]
        }
        result["evidence"] = [
            {
                "source_id": trace.get("source_id"),
                "content": by_id.get(trace.get("source_id"), {}).get("content", ""),
            }
            for trace in result.get("retrieval_trace", [])
        ]
        self.on_case(result)
        return result


class ExperimentStopped(RuntimeError):
    pass


def summarize_run(row: dict, local_hourly_usd: float = 0) -> dict:
    cases = row.get("cases", [])
    calls = row.get("calls", [])
    summary = summarize_calls(calls, local_hourly_usd=local_hourly_usd)
    n = len(cases)
    correct = sum(bool(c["correct"]) for c in cases)
    serving_latency = [c["retrieval_seconds"] + c["generation_seconds"] for c in cases]
    summary.update(
        {
            "samples": n,
            "accuracy": correct / n if n else None,
            "correct": correct,
            "accuracy_ci95": accuracy_interval(correct, n),
            "score": statistics.fmean(c["score"] for c in cases) if n else None,
            "latency_p50_seconds": _percentile(serving_latency, 0.5),
            "latency_p95_seconds": _percentile(serving_latency, 0.95),
            "reranker_p95_seconds": _percentile(
                [c["retrieval_timing"].get("reranker_seconds", 0) for c in cases], 0.95
            ),
            "score_ci95": _bootstrap_mean_ci([c["score"] for c in cases], seed=0)
            if n
            else None,
        }
    )
    for key in (
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
        "candidate_recall",
    ):
        values = [
            c["retrieval"][key] for c in cases if c["retrieval"].get(key) is not None
        ]
        summary[key] = statistics.fmean(values) if values else None
    cost = summary["serving_cost_usd"]
    summary["cost_per_question_usd"] = cost / n if cost is not None and n else None
    summary["cost_per_correct_usd"] = (
        cost / correct if cost is not None and correct else None
    )
    return summary


def comparisons(runs: list[dict]) -> list[dict]:
    completed = [r for r in runs if r["status"] == "completed"]
    if not completed:
        return []
    baseline = completed[0]
    source = {c["case_id"]: c for c in baseline["cases"]}
    output = []
    for run in completed[1:]:
        pairs = [
            (source[c["case_id"]], c) for c in run["cases"] if c["case_id"] in source
        ]
        deltas = [b["score"] - a["score"] for a, b in pairs]
        output.append(
            {
                "baseline_id": baseline["id"],
                "run_id": run["id"],
                "paired_cases": len(pairs),
                "score_delta_ci95": _bootstrap_mean_ci(deltas, seed=0),
                "improved": [b["case_id"] for a, b in pairs if b["score"] > a["score"]],
                "regressed": [
                    b["case_id"] for a, b in pairs if b["score"] < a["score"]
                ],
            }
        )
    return output


def run_comparison(
    config: ComparisonConfig, directory: Path, *, runner_factory=ComparisonRunner
) -> dict:
    if config.candidate_snapshot_path:
        from .snapshot_comparison import run_snapshot_comparison

        return run_snapshot_comparison(config, directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "name": config.name,
        "status": "running",
        "config": config.model_dump(),
        "runs": [],
        "started_at": time.time(),
        "warnings": [
            "Diagnostic memory QA; tools and continual learning are not exercised.",
            "First-stage candidates and timings are measured once and replayed across configurations.",
            "Spending stop threshold is checked between calls; an in-flight request can exceed it.",
            "TTFT measures first visible text only when streaming is enabled. Local hardware costs are optional estimates.",
            "Cold model loading is included in call latency. Use repeated runs and a representative held-out dataset before choosing a model.",
            "Local embeddings have no external API charge. Ingestion and embedding token counts are outside the model-call ledger.",
        ],
    }
    output = directory / "result.json"
    candidates = {}

    def save():
        state["comparisons"] = comparisons(state["runs"])
        state["updated_at"] = time.time()
        atomic_json(output, state)

    def check():
        if (directory / "cancel").exists():
            raise ExperimentStopped("cancelled")
        if time.time() - state["started_at"] >= config.max_seconds:
            raise ExperimentStopped("time_limit")
        all_calls = [c for r in state["runs"] for c in r.get("calls", [])]
        if config.max_cost_usd is not None:
            if any(c["cost_usd"] is None for c in all_calls):
                raise ExperimentStopped("unknown_cost")
            if sum(c["cost_usd"] for c in all_calls) >= config.max_cost_usd:
                raise ExperimentStopped("spend_limit")

    save()
    try:
        cases = load_cases(config)
        frozen = [case_json(c) for c in cases]
        state["dataset_fingerprint"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True).encode()
        ).hexdigest()
        state["case_ids"] = [c.case_id for c in cases]
        state["dataset_label"] = (
            "Strict synthetic memory checks — not a production benchmark"
            if config.dataset == "memory_checks"
            else "Synthetic demo — integration check"
            if config.dataset == "demo"
            else config.dataset
        )
        atomic_json(directory / "cases.json", frozen)
        specs = list(itertools.product(config.readers, config.rerankers))
        for repeat in range(config.repeats):
            order = list(enumerate(specs))
            if repeat:
                random.Random(config.seed + repeat).shuffle(order)
            for index, (reader_spec, reranker_spec) in order:
                check()
                row = {
                    "id": f"config-{index + 1}-repeat-{repeat + 1}",
                    "repeat": repeat + 1,
                    "reader": reader_spec.model_dump(),
                    "reranker": reranker_spec.model_dump(),
                    "status": "running",
                    "cases": [],
                    "calls": [],
                }
                state["runs"].append(row)
                ledger = MeasurementLedger(check=check, on_append=save)
                row["calls"] = ledger.calls
                runner = None
                save()
                try:

                    def resolved(spec):
                        value = spec.model_dump()
                        if spec.provider == "ollama" or (
                            spec.provider == "llm" and spec.llm_provider == "ollama"
                        ):
                            value["options"].setdefault("host", config.ollama_host)
                        if spec.provider == "anthropic" and config.repeats > 1:
                            # Repeats resend identical prompts: cache their
                            # bodies so later repeats read at ~0.1x input.
                            value["options"].setdefault(
                                "cache_single_shot_prompts", True
                            )
                        return value

                    reader = MeasuredModel(resolved(reader_spec), ledger)
                    judge = MeasuredModel(resolved(config.judge), ledger)
                    reranker = Reranker(resolved(reranker_spec), ledger)

                    def on_case(case):
                        row["cases"].append(case)
                        row["summary"] = summarize_run(row, config.local_hourly_usd)
                        atomic_json(directory / "candidates.json", candidates)
                        save()

                    runner = runner_factory(
                        workspace=directory / row["id"],
                        model=reader,
                        judge_model=judge,
                        model_provider=reader_spec.provider,
                        model_name=reader_spec.model,
                        judge_model_name=config.judge.model,
                        judge_provider=config.judge.provider,
                        embedding_model=config.embedding_model,
                        ollama_host=config.ollama_host,
                        top_k=config.top_k,
                        candidate_pool_size=config.candidate_pool_size,
                        seed=config.seed,
                        oracle_reader=config.oracle_reader,
                        profile="regression",
                        semantic_memory=False,
                        query_expansion=False,
                        ledger=ledger,
                        reranker=reranker,
                        candidates=candidates,
                        on_case=on_case,
                    )
                    report = runner.run(
                        cases,
                        variant=config.variant or config.dataset,
                        dataset_path=directory / "cases.json",
                    )
                    measured = summarize_calls(ledger.calls)
                    report["usage"]["comparison_measurements"] = measured
                    report["usage"]["cost_usd"] = measured["total_cost_usd"]
                    report["usage"]["lane_cost_usd"] = {
                        label: measured["lanes"].get(lane, {}).get("cost_usd", 0)
                        for label, lane in (
                            ("retrieved_reader", "reader"),
                            ("oracle_reader", "oracle"),
                            ("judge", "judge"),
                            ("reranker", "reranker"),
                        )
                    }
                    report["metadata"]["external_api_cost_usd"] = measured[
                        "total_cost_usd"
                    ]
                    report["metadata"]["comparison_config"] = config.model_dump()
                    atomic_json(directory / f"{row['id']}.json", report)
                    row["status"] = "completed"
                except ExperimentStopped:
                    row["status"] = "stopped"
                    raise
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else f"{type(exc).__name__}: evaluation failed; check provider availability and configuration"
                    )
                finally:
                    row["summary"] = summarize_run(row, config.local_hourly_usd)
                    if runner is not None:
                        runner.close()
                    save()
        state["status"] = (
            "completed"
            if all(r["status"] == "completed" for r in state["runs"])
            else "completed_with_errors"
        )
    except ExperimentStopped as exc:
        state["status"] = str(exc)
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = (
            str(exc)
            if isinstance(exc, ValueError)
            else f"{type(exc).__name__}: could not load or run the evaluation"
        )
    finally:
        state["finished_at"] = time.time()
        save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = ComparisonConfig.model_validate_json(args.config.read_text())
    run_comparison(config, args.config.parent)


if __name__ == "__main__":
    main()
