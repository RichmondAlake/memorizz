"""Native MemoRizz evaluation adapter for the official MemBench data."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from memorizz import CompletionPolicy, ContextPolicy, MemAgent
from memorizz.benchmarks.common import (
    create_benchmark_memory_provider,
    normalize_memory_backend,
)
from memorizz.embeddings import EmbeddingManager, set_global_embedding_manager
from memorizz.enums import MemoryType
from memorizz.llms.openai import OpenAI


@dataclass(frozen=True)
class MemBenchSample:
    sample_id: str
    perspective: str
    memory_kind: str
    category: str
    source_file: str
    messages: Sequence[Mapping[str, Any]]
    question: str
    choices: Mapping[str, str]
    ground_truth: str
    target_source_keys: Sequence[str]


class CountingOpenAI(OpenAI):
    """OpenAI provider with benchmark-level usage and cost accounting."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        self.estimated_cost_usd = 0.0
        self.call_count = 0

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        response = super().generate(*args, **kwargs)
        usage = dict(self.get_last_usage() or {})
        for key in self.total_usage:
            self.total_usage[key] += max(0, int(usage.get(key) or 0))
        prompt = max(0, int(usage.get("prompt_tokens") or 0))
        cached = min(prompt, max(0, int(usage.get("cached_tokens") or 0)))
        output = max(0, int(usage.get("completion_tokens") or 0))
        self.estimated_cost_usd += (
            (prompt - cached) * 0.25 + cached * 0.025 + output * 2.0
        ) / 1_000_000
        self.call_count += 1
        return response


def _flatten_messages(
    message_list: Any, *, perspective: str
) -> tuple[List[Dict[str, Any]], Dict[tuple[int, int], str]]:
    """Normalize first/third-person layouts while preserving official IDs."""
    normalized: List[Dict[str, Any]] = []
    pair_to_key: Dict[tuple[int, int], str] = {}
    if not isinstance(message_list, list):
        return normalized, pair_to_key
    nested = bool(message_list and isinstance(message_list[0], list))
    sessions = message_list if nested else [message_list]
    for session_index, session in enumerate(sessions):
        if not isinstance(session, list):
            continue
        for position, message in enumerate(session):
            if not isinstance(message, Mapping):
                continue
            raw_id = message.get("sid", message.get("mid", position))
            try:
                numeric_id = int(raw_id)
            except (TypeError, ValueError):
                numeric_id = position
            source_key = f"{numeric_id}:{session_index}" if nested else str(numeric_id)
            pair_to_key[(numeric_id, session_index)] = source_key
            if "message" in message:
                text = str(message.get("message") or "")
            else:
                user = message.get("user_message", message.get("user", ""))
                assistant = message.get(
                    "assistant_message", message.get("assistant", "")
                )
                text = f"User: {user}\nAssistant: {assistant}".strip()
            metadata = {
                key: message.get(key)
                for key in ("time", "place", "rel", "attr", "value")
                if message.get(key) is not None
            }
            normalized.append(
                {
                    "source_key": source_key,
                    "source_numeric_id": numeric_id,
                    "session_index": session_index,
                    "position": position,
                    "text": text,
                    "metadata": metadata,
                    "perspective": perspective,
                }
            )
    return normalized, pair_to_key


def _target_keys(
    raw_targets: Any,
    *,
    nested: bool,
    pair_to_key: Mapping[tuple[int, int], str],
) -> List[str]:
    targets = raw_targets if isinstance(raw_targets, list) else []
    output: List[str] = []
    for target in targets:
        if isinstance(target, list) and len(target) >= 2:
            try:
                pair = (int(target[0]), int(target[1]))
            except (TypeError, ValueError):
                continue
            output.append(pair_to_key.get(pair, f"{pair[0]}:{pair[1]}"))
        else:
            try:
                numeric = int(target)
            except (TypeError, ValueError):
                continue
            output.append(str(numeric) if not nested else f"{numeric}:0")
    return output


def load_official_raw_samples(
    data_root: Path, *, samples_per_track: int = 1
) -> List[MemBenchSample]:
    """Load a deterministic 2×2 smoke matrix from official categorical data."""
    root = Path(data_root)
    specifications = [
        ("participation", "factual", root / "FirstAgent" / "simple.json"),
        ("participation", "reflective", root / "FirstAgent" / "highlevel.json"),
        ("observation", "factual", root / "ThirdAgent" / "simple.json"),
        ("observation", "reflective", root / "ThirdAgent" / "highlevel.json"),
    ]
    samples: List[MemBenchSample] = []
    for perspective, memory_kind, path in specifications:
        if not path.exists():
            raise FileNotFoundError(f"Missing official MemBench file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected category object in {path}")
        selected = 0
        for category, records in payload.items():
            if selected >= samples_per_track:
                break
            if not isinstance(records, list):
                continue
            for record in records:
                qa = record.get("QA") if isinstance(record, Mapping) else None
                message_list = (
                    record.get("message_list") if isinstance(record, Mapping) else None
                )
                if not isinstance(qa, Mapping) or not isinstance(message_list, list):
                    continue
                messages, pair_to_key = _flatten_messages(
                    message_list, perspective=perspective
                )
                if not messages:
                    continue
                choices = qa.get("choices")
                ground_truth = str(qa.get("ground_truth") or "").upper()
                if not isinstance(choices, Mapping) or ground_truth not in choices:
                    continue
                nested = bool(message_list and isinstance(message_list[0], list))
                targets = _target_keys(
                    qa.get("target_step_id"),
                    nested=nested,
                    pair_to_key=pair_to_key,
                )
                qid = str(qa.get("qid", selected))
                samples.append(
                    MemBenchSample(
                        sample_id=f"{perspective}:{memory_kind}:{category}:{qid}",
                        perspective=perspective,
                        memory_kind=memory_kind,
                        category=str(category),
                        source_file=str(path),
                        messages=messages,
                        question=str(qa.get("question") or ""),
                        choices={
                            str(key): str(value) for key, value in choices.items()
                        },
                        ground_truth=ground_truth,
                        target_source_keys=targets,
                    )
                )
                selected += 1
                break
    return samples


def _parse_choice(response: str) -> Optional[str]:
    patterns = (
        r'"(?:answer|choice)"\s*:\s*"?([A-D])\b',
        r"\\boxed\{\s*([A-D])\s*\}",
        r"\b(?:answer|choice)(?:\s+is)?\s*[:=-]?\s*([A-D])\b",
        r"^\s*([A-D])\s*[.)]?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, str(response), flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def _message_document(message: Mapping[str, Any]) -> str:
    metadata = json.dumps(
        message.get("metadata") or {}, ensure_ascii=False, sort_keys=True
    )
    return (
        f"MemBench source_key={message.get('source_key')}\n"
        f"Perspective: {message.get('perspective')}\n"
        f"Metadata: {metadata}\n"
        f"Content:\n{message.get('text')}"
    )


def _token_count(texts: Iterable[str]) -> tuple[int, str]:
    joined = "\n".join(texts)
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(joined)), "cl100k_base"
    except ImportError:
        return (len(joined) + 3) // 4, "chars_div_4_estimate"


def evaluate_membench_samples(
    samples: Sequence[MemBenchSample],
    *,
    output_dir: Path,
    model_name: str = "gpt-5-mini",
    top_k: int = 8,
    summary_batch_size: int = 24,
    memory_backend: str = "filesystem",
    learning_control_plane: bool = True,
    embedding_dimensions: int = 256,
) -> Dict[str, Any]:
    """Evaluate a sample set with retrieval, compaction, cache, and audit evidence."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_backend = normalize_memory_backend(memory_backend)
    embedding_manager = EmbeddingManager(
        "openai",
        {
            "api_key": api_key,
            "model": "text-embedding-3-small",
            "dimensions": max(64, int(embedding_dimensions)),
        },
    )
    set_global_embedding_manager(embedding_manager)
    model = CountingOpenAI(
        api_key=api_key,
        model=model_name,
        api_mode="responses",
        reasoning_effort="low",
        max_tokens=2_048,
    )
    records: List[Dict[str, Any]] = []
    evaluation_scope = uuid.uuid5(uuid.NAMESPACE_URL, str(output_dir))
    for sample_index, sample in enumerate(samples):
        sample_started = time.perf_counter()
        sample_root = output_dir / "memory" / f"sample-{sample_index:04d}"
        provider = create_benchmark_memory_provider(
            memory_backend,
            filesystem_root=sample_root,
            embedding_provider=embedding_manager,
        )
        memory_id = (
            f"membench:{evaluation_scope}:"
            f"{uuid.uuid5(uuid.NAMESPACE_URL, sample.sample_id)}"
        )
        user_id = "membench"
        thread_id = sample.sample_id
        documents = [_message_document(message) for message in sample.messages]
        ingest_started = time.perf_counter()
        embeddings = embedding_manager.get_embeddings(documents)
        for message, content, embedding in zip(sample.messages, documents, embeddings):
            base = {
                "content": content,
                "memory_id": memory_id,
                "user_id": user_id,
                "agent_id": "membench-memory",
                "thread_id": thread_id,
                "source_key": message.get("source_key"),
                "source_type": "membench_official_raw",
                # Oracle's relational KB schema preserves portable ingestion
                # provenance in namespace/knowledge_base_id; document stores
                # additionally retain the descriptive source_key fields.
                "namespace": str(message.get("source_key") or ""),
                "knowledge_base_id": sample.sample_id[:64],
                "embedding": embedding,
            }
            provider.store(base, MemoryType.KNOWLEDGE_BASE)
            provider.store({**base, "role": "user"}, MemoryType.CONVERSATION_MEMORY)
        ingest_seconds = time.perf_counter() - ingest_started
        completion_policy = CompletionPolicy(
            enabled=True,
            max_rejections=2,
            fail_closed=True,
            validator=lambda candidate: (
                _parse_choice(candidate.response) is not None,
                "Return exactly one supported choice letter A-D in JSON.",
            ),
            validator_name="membench_choice_parser",
        )
        agent = MemAgent(
            model=model,
            llm_config={"provider": "openai", "model": model_name},
            instruction=(
                "Answer the MemBench multiple-choice question using only the "
                "retrieved and compacted memory evidence. Return JSON exactly as "
                '{"answer":"A"} with one letter A-D and no other text.'
            ),
            memory_provider=provider,
            memory_ids=[memory_id],
            memory_types=[
                MemoryType.CONVERSATION_MEMORY,
                MemoryType.KNOWLEDGE_BASE,
                MemoryType.SUMMARIES,
            ],
            semantic_cache=True,
            semantic_cache_config={
                "similarity_threshold": 1.0,
                "scope": "session",
                "ttl_hours": 24.0,
                "admission_policy": "read_only_deterministic",
                "freshness_by_domain": {"membench": 86_400.0},
            },
            automations_enabled=False,
            context_policy=ContextPolicy(
                progressive_tool_disclosure=True,
                tool_top_k=3,
                max_tool_invocations_per_turn=4,
            ),
            completion_policy=completion_policy,
            max_steps=6,
            name=f"MemoRizz MemBench {sample_index}",
            auto_register=True,
            learning_control_plane=(
                {
                    "enabled": True,
                    "evidence_sources": ["knowledge_base", "summaries"],
                    "evidence_token_budget": 1600,
                    # One compact summary plus three supporting source turns
                    # keeps this multiple-choice reader focused; the wider
                    # candidate pool is still retained for ranking/audit.
                    "evidence_max_items": max(2, min(int(top_k), 4)),
                    "evidence_candidates_per_source": max(2, int(top_k)),
                    "evidence_max_per_source": max(2, min(int(top_k), 4)),
                }
                if learning_control_plane
                else False
            ),
        )
        compaction_started = time.perf_counter()
        summary_ids = agent.generate_summaries(
            days_back=1,
            max_memories_per_summary=max(2, int(summary_batch_size)),
            memory_id=memory_id,
            user_id=user_id,
            thread_id=thread_id,
        )
        compaction_seconds = time.perf_counter() - compaction_started
        choices_text = "\n".join(
            f"{letter}. {value}" for letter, value in sample.choices.items()
        )
        query = f"{sample.question}\n\nChoices:\n{choices_text}"
        retrieval_started = time.perf_counter()
        matches = (
            provider.retrieve_by_query(
                query,
                MemoryType.KNOWLEDGE_BASE,
                limit=max(1, int(top_k)),
                memory_id=memory_id,
                user_id=user_id,
            )
            or []
        )
        retrieval_seconds = time.perf_counter() - retrieval_started
        retrieved_keys = [
            str(row.get("source_key") or row.get("namespace"))
            for row in matches
            if row.get("source_key") is not None or row.get("namespace") is not None
        ]
        targets = set(sample.target_source_keys)
        recall = (
            len(targets.intersection(retrieved_keys)) / len(targets)
            if targets
            else None
        )
        request_context = {
            "benchmark": "MemBench",
            "perspective": sample.perspective,
            "memory_kind": sample.memory_kind,
            "cache_domain": "membench",
            "cache_data_version": sample.sample_id,
        }
        if not learning_control_plane:
            request_context["retrieved_evidence"] = [
                row.get("content") for row in matches
            ]
        answer_started = time.perf_counter()
        response = agent.run(
            query,
            memory_id=memory_id,
            thread_id=f"{thread_id}:qa",
            user_id=user_id,
            context=request_context,
        )
        answer_seconds = time.perf_counter() - answer_started
        # A cache hit is a separate turn and intentionally has no new model
        # completion to validate. Preserve the first turn's host-gate evidence
        # before exercising the exact-repeat semantic-cache probe.
        completion_report = agent.completion_policy_report()
        cache_before_repeat = agent.semantic_cache_stats()
        repeated = agent.run(
            query,
            memory_id=memory_id,
            thread_id=f"{thread_id}:qa",
            user_id=user_id,
            context=request_context,
        )
        cache_after_repeat = agent.semantic_cache_stats()
        predicted = _parse_choice(response)
        outcome_trace = agent.get_trace_context()
        outcome_record = agent.record_task_outcome(
            "success" if predicted == sample.ground_truth else "failure",
            verified=True,
            source="membench_official_answer_key",
            score=1.0 if predicted == sample.ground_truth else 0.0,
            metrics={
                "sample_id": sample.sample_id,
                "prediction": predicted,
                "ground_truth": sample.ground_truth,
            },
            trace_context=outcome_trace,
            external_id=(
                f"membench:{sample.sample_id}:" f"{outcome_trace.get('root_trace_id')}"
            ),
        )
        compiler_report = (
            agent.compile_memory(
                memory_id=memory_id, user_id=user_id, thread_id=f"{thread_id}:qa"
            )
            if learning_control_plane
            else None
        )
        capacity_tokens, token_counter = _token_count(documents)
        try:
            observability = agent.observability_summary(memory_id, user_id)
        except Exception as exc:
            observability = {"error": f"{type(exc).__name__}: {exc}"}
        record = {
            "sample_id": sample.sample_id,
            "perspective": sample.perspective,
            "memory_kind": sample.memory_kind,
            "category": sample.category,
            "source_file": sample.source_file,
            "message_count": len(sample.messages),
            "capacity_tokens": capacity_tokens,
            "token_counter": token_counter,
            "question": sample.question,
            "choices": dict(sample.choices),
            "ground_truth": sample.ground_truth,
            "prediction": predicted,
            "correct": predicted == sample.ground_truth,
            "response": response,
            "repeat_response_match": repeated == response,
            "target_source_keys": list(sample.target_source_keys),
            "retrieved_source_keys": retrieved_keys,
            "retrieval_recall": recall,
            "summary_count": len(summary_ids),
            "summary_ids": summary_ids,
            "timing_seconds": {
                "ingest": ingest_seconds,
                "compaction": compaction_seconds,
                "retrieval": retrieval_seconds,
                "answer": answer_seconds,
                "total": time.perf_counter() - sample_started,
            },
            "completion_policy": completion_report,
            "semantic_cache_before_repeat": cache_before_repeat,
            "semantic_cache_after_repeat": cache_after_repeat,
            "observability": observability,
            "verified_outcome": outcome_record,
            "memory_compiler": compiler_report,
            "learning_control_plane": agent.learning_report(
                memory_id=memory_id, user_id=user_id
            ),
        }
        records.append(record)
        agent.close(close_memory_provider=True)

    def _mean(values: Sequence[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    accuracy_values = [1.0 if record["correct"] else 0.0 for record in records]
    recalls = [
        float(record["retrieval_recall"])
        for record in records
        if record["retrieval_recall"] is not None
    ]
    by_track: Dict[str, Dict[str, Any]] = {}
    for perspective in ("participation", "observation"):
        for memory_kind in ("factual", "reflective"):
            selected = [
                record
                for record in records
                if record["perspective"] == perspective
                and record["memory_kind"] == memory_kind
            ]
            by_track[f"{perspective}_{memory_kind}"] = {
                "count": len(selected),
                "accuracy": _mean(
                    [1.0 if record["correct"] else 0.0 for record in selected]
                ),
                "retrieval_recall": _mean(
                    [
                        float(record["retrieval_recall"])
                        for record in selected
                        if record["retrieval_recall"] is not None
                    ]
                ),
            }
    report = {
        "benchmark": "MemBench",
        "track": "official_raw_categorical_smoke",
        "paper_comparable": False,
        "memory_backend": memory_backend,
        "learning_control_plane_enabled": bool(learning_control_plane),
        "embedding_dimensions": max(64, int(embedding_dimensions)),
        "non_comparability_reasons": [
            "the paper-sampled data2test bundle is not included in the Git repository",
            "bounded raw categorical sample rather than the paper's 0-10k/100k sets",
        ],
        "sample_count": len(records),
        "accuracy": _mean(accuracy_values),
        "retrieval_recall": _mean(recalls),
        "by_track": by_track,
        "total_capacity_tokens": sum(record["capacity_tokens"] for record in records),
        "usage": dict(model.total_usage),
        "model_call_count": model.call_count,
        "estimated_cost_usd": model.estimated_cost_usd,
        "features": {
            "semantic_vector_retrieval": True,
            "semantic_cache_exact_repeat": True,
            "summarization_and_compaction": True,
            "tenant_and_thread_isolation": True,
            "host_completion_gate": True,
            "bounded_evidence_pack": bool(learning_control_plane),
            "immutable_learning_events": bool(learning_control_plane),
            "verified_outcomes": bool(learning_control_plane),
        },
        "records": records,
    }
    (output_dir / "membench-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "MemBenchSample",
    "evaluate_membench_samples",
    "load_official_raw_samples",
]
