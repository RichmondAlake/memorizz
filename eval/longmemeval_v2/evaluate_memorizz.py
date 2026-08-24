#!/usr/bin/env python3
"""Run MemoRizz through the official LongMemEval-V2 evaluation harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memorizz._env_io import load_layered_env  # noqa: E402
from memorizz.benchmarks.longmemeval_v2 import (  # noqa: E402
    trajectory_chunks,
    trajectory_digest,
)
from memorizz.benchmarks.pricing import (  # noqa: E402
    OPENAI_EMBEDDING_PRICING,
    estimate_openai_embedding_cost,
    estimate_openai_text_cost,
    resolve_openai_text_pricing,
)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _official_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _combined_hash(paths: Iterable[Path], *, root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if not path.exists():
            continue
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def verify_official_data(data_root: Path) -> Dict[str, Any]:
    """Verify the three text-tier inputs against the official checksums."""

    checksums_path = data_root / "checksums.sha256"
    if not checksums_path.exists():
        raise FileNotFoundError(f"Missing official checksum manifest: {checksums_path}")
    expected: Dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            expected[parts[1].lstrip("*")] = parts[0].lower()
    required = (
        "questions.jsonl",
        "trajectories.jsonl",
        "haystacks/lme_v2_small.json",
    )
    rows = []
    for relative in required:
        path = data_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing LongMemEval-V2 artifact: {path}")
        actual = _sha256(path)
        expected_hash = expected.get(relative)
        matches = bool(expected_hash and actual == f"sha256:{expected_hash}")
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": (
                    f"sha256:{expected_hash}" if expected_hash else None
                ),
                "matches": matches,
            }
        )
    mismatches = [row["path"] for row in rows if not row["matches"]]
    if mismatches:
        raise RuntimeError(
            "Official LongMemEval-V2 checksum verification failed for: "
            + ", ".join(mismatches)
        )
    return {
        "verified": True,
        "checksum_manifest_sha256": _sha256(checksums_path),
        "artifacts": rows,
    }


def _token_counter():
    try:
        import tiktoken

        encoder = tiktoken.get_encoding("cl100k_base")
        return lambda value: len(encoder.encode(value)), "tiktoken-cl100k_base"
    except (ImportError, OSError):
        return (
            lambda value: max(1, math.ceil(len(value) / 4)),
            "characters-divided-by-4",
        )


def forecast_subset_cost(
    *,
    trajectories_path: Path,
    questions_path: Path,
    model: str,
    evaluator_model: str,
    embedding_model: str,
    chunk_chars: int,
    summary_batch_size: int,
    evidence_token_budget: int,
    enable_summaries: bool,
) -> Dict[str, Any]:
    """Forecast hosted cost before any model or embedding request is made."""

    count_tokens, token_method = _token_counter()
    embedding_tokens = 0
    digest_tokens = 0
    chunk_count = 0
    trajectory_count = 0
    for trajectory in _read_jsonl(trajectories_path):
        chunks = trajectory_chunks(trajectory, max_chars=chunk_chars)
        embedding_tokens += sum(count_tokens(chunk) for chunk in chunks)
        chunk_count += len(chunks)
        digest = trajectory_digest(trajectory)
        digest_count = count_tokens(digest)
        digest_tokens += digest_count
        embedding_tokens += digest_count
        trajectory_count += 1

    questions = list(_read_jsonl(questions_path))
    question_count = len(questions)
    summary_calls = (
        math.ceil(trajectory_count / max(2, summary_batch_size))
        if enable_summaries
        else 0
    )
    llm_judge_calls = sum(
        str(question.get("eval_function") or "").split("|", 1)[0]
        in {"llm_abstention_checker", "llm_gotchas_checker"}
        for question in questions
    )

    # The summary prompts primarily contain trajectory digests. The other
    # allowances cover MemoRizz briefing prompts and the official 50K reader
    # boundary without pretending to know the provider's eventual cache hits.
    model_input_tokens = (
        (digest_tokens + 1_000 * summary_calls if enable_summaries else 0)
        + question_count * (evidence_token_budget + 2_000)
        + question_count * 52_000
    )
    model_output_tokens = 4_096 * (summary_calls + 2 * question_count)
    model_cost = estimate_openai_text_cost(
        model,
        {
            "prompt_tokens": model_input_tokens,
            "completion_tokens": model_output_tokens,
        },
    )
    judge_cost = estimate_openai_text_cost(
        evaluator_model,
        {
            "prompt_tokens": 4_000 * llm_judge_calls,
            "completion_tokens": 2_048 * llm_judge_calls,
        },
    )
    embedding_cost = estimate_openai_embedding_cost(embedding_model, embedding_tokens)
    point = embedding_cost + model_cost + judge_cost
    contingency = point * 1.5
    return {
        "kind": "preflight_cost_forecast",
        "external_calls_made": False,
        "token_estimation_method": token_method,
        "trajectory_count": trajectory_count,
        "question_count": question_count,
        "chunk_count": chunk_count,
        "embedding_tokens_estimated": embedding_tokens,
        "digest_tokens_estimated": digest_tokens,
        "summary_calls_max": summary_calls,
        "reader_and_briefing_calls_max": 2 * question_count,
        "llm_judge_calls_max": llm_judge_calls,
        "model_input_tokens_upper_estimate": model_input_tokens,
        "model_output_tokens_upper_estimate": model_output_tokens,
        "cost_usd": {
            "embedding": round(embedding_cost, 6),
            "memory_and_reader": round(model_cost, 6),
            "judge": round(judge_cost, 6),
            "point_estimate": round(point, 6),
            "with_50_percent_contingency": round(contingency, 6),
        },
        "pricing": {
            "text_model": resolve_openai_text_pricing(model).to_dict(),
            "evaluator_model": resolve_openai_text_pricing(evaluator_model).to_dict(),
            "embedding_model": {
                "model": embedding_model,
                **dict(OPENAI_EMBEDDING_PRICING[embedding_model]),
            },
        },
        "caveat": (
            "This fail-closed preflight is an engineering estimate, not an invoice "
            "cap. It intentionally assumes no prompt-cache discount."
        ),
    }


def prepare_subset(
    *, data_root: Path, output_dir: Path, question_ids: List[str]
) -> tuple[Path, Path, Path, str, int]:
    questions_path = data_root / "questions.jsonl"
    haystack_path = data_root / "haystacks" / "lme_v2_small.json"
    trajectories_path = data_root / "trajectories.jsonl"
    for required in (questions_path, haystack_path, trajectories_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing LongMemEval-V2 artifact: {required}")

    wanted_questions = set(question_ids)
    selected = [
        question
        for question in _read_jsonl(questions_path)
        if str(question.get("id")) in wanted_questions
    ]
    found = {str(question.get("id")) for question in selected}
    missing = sorted(wanted_questions - found)
    if missing:
        raise ValueError(f"Unknown LongMemEval-V2 question IDs: {missing}")
    domains = {str(question.get("domain")) for question in selected}
    if len(domains) != 1:
        raise ValueError("One official harness run may contain only one domain")
    if any(question.get("image") for question in selected):
        raise ValueError(
            "This text-only runner does not download screenshot bundles; select "
            "questions whose image field is null."
        )

    full_haystacks = json.loads(haystack_path.read_text(encoding="utf-8"))
    selected_haystacks = {
        question_id: full_haystacks[question_id] for question_id in question_ids
    }
    trajectory_ids = {
        trajectory_id
        for values in selected_haystacks.values()
        for trajectory_id in values
    }
    subset_dir = output_dir / "official_subset"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_questions = subset_dir / "questions.jsonl"
    subset_haystacks = subset_dir / "haystack.json"
    subset_trajectories = subset_dir / "trajectories.jsonl"
    _write_jsonl(subset_questions, selected)
    _write_json(subset_haystacks, selected_haystacks)

    found_trajectories = set()
    with subset_trajectories.open("w", encoding="utf-8") as target:
        for trajectory in _read_jsonl(trajectories_path):
            trajectory_id = str(trajectory.get("id"))
            if trajectory_id not in trajectory_ids:
                continue
            target.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            found_trajectories.add(trajectory_id)
    missing_trajectories = sorted(trajectory_ids - found_trajectories)
    if missing_trajectories:
        raise RuntimeError(
            f"Official trajectory file is missing {len(missing_trajectories)} "
            "required IDs"
        )
    return (
        subset_questions,
        subset_haystacks,
        subset_trajectories,
        next(iter(domains)),
        len(trajectory_ids),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MemoRizz with the official LongMemEval-V2 harness."
    )
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--question-id", action="append", dest="question_ids", default=[]
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--evaluator-model", default="gpt-5-mini")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--lexical-ratio", type=float, default=0.5)
    parser.add_argument("--summary-top-k", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=12_000)
    parser.add_argument("--summary-batch-size", type=int, default=20)
    parser.add_argument("--embedding-dimensions", type=int, default=256)
    parser.add_argument("--no-summaries", action="store_true")
    parser.add_argument("--no-cache-probe", action="store_true")
    parser.add_argument(
        "--memory-backend", choices=("filesystem", "oracle"), default="filesystem"
    )
    parser.add_argument("--legacy-memory-path", action="store_true")
    parser.add_argument("--evidence-token-budget", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=2.0)
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Write the cost forecast and stop before any hosted request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_layered_env([PROJECT_ROOT / ".env"])
    official_root = args.official_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (official_root / "evaluation" / "harness.py").exists():
        raise FileNotFoundError(
            f"Not an official LongMemEval-V2 checkout: {official_root}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix results into non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    question_ids = args.question_ids or ["01307e07"]
    (
        questions_path,
        haystack_path,
        trajectories_path,
        domain,
        trajectory_count,
    ) = prepare_subset(
        data_root=data_root,
        output_dir=output_dir,
        question_ids=question_ids,
    )

    data_verification = verify_official_data(data_root)
    cost_forecast = forecast_subset_cost(
        trajectories_path=trajectories_path,
        questions_path=questions_path,
        model=args.model,
        evaluator_model=args.evaluator_model,
        embedding_model="text-embedding-3-small",
        chunk_chars=args.chunk_chars,
        summary_batch_size=args.summary_batch_size,
        evidence_token_budget=args.evidence_token_budget,
        enable_summaries=not args.no_summaries,
    )
    _write_json(output_dir / "cost-forecast.json", cost_forecast)
    estimated_with_contingency = float(
        cost_forecast["cost_usd"]["with_50_percent_contingency"]
    )
    if estimated_with_contingency > float(args.max_estimated_cost_usd):
        raise RuntimeError(
            "LongMemEval-V2 preflight forecast exceeds the configured limit: "
            f"${estimated_with_contingency:.4f} > "
            f"${float(args.max_estimated_cost_usd):.2f}"
        )
    if args.forecast_only:
        print(json.dumps(cost_forecast, indent=2, sort_keys=True))
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required after cost preflight")

    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    from memorizz.benchmarks.longmemeval_v2 import (  # noqa: E402
        load_longmemeval_v2_memory_registry,
        register_longmemeval_v2_backend,
    )
    from memorizz.benchmarks.memory_suite import (  # noqa: E402
        assess_comparability,
        get_protocol_manifest,
    )

    load_longmemeval_v2_memory_registry(official_root)
    register_longmemeval_v2_backend()
    from evaluation import harness  # type: ignore  # noqa: E402

    # The official harness otherwise discovers this heavyweight runtime only
    # after all trajectories have been embedded and summarized.
    harness.get_memory_context_processor()

    memory_config = {
        "memory_type": "memorizz",
        "memory_params": {
            "workspace_dir": str(output_dir / "memorizz_memory"),
            "memory_id": (
                f"longmemeval-v2:{question_ids[0]}:"
                f"{uuid.uuid5(uuid.NAMESPACE_URL, str(output_dir)).hex[:12]}"
            ),
            "user_id": "longmemeval-v2",
            "model": args.model,
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": args.embedding_dimensions,
            "top_k": args.top_k,
            "lexical_ratio": args.lexical_ratio,
            "summary_top_k": args.summary_top_k,
            "chunk_chars": args.chunk_chars,
            "embedding_batch_size": 64,
            "summary_batch_size": args.summary_batch_size,
            "enable_summaries": not args.no_summaries,
            "cache_probe": not args.no_cache_probe,
            "api_key_env": "OPENAI_API_KEY",
            "memory_backend": args.memory_backend,
            "learning_control_plane": not args.legacy_memory_path,
            "evidence_token_budget": args.evidence_token_budget,
        },
    }
    memory_config_path = output_dir / "memorizz-memory-config.json"
    _write_json(memory_config_path, memory_config)
    revision = _official_revision(official_root)
    subset_fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {
                    "question_ids": question_ids,
                    "domain": domain,
                    "trajectory_count": trajectory_count,
                    "official_artifacts": data_verification["artifacts"],
                    "subset_files": {
                        "questions": _sha256(questions_path),
                        "haystack": _sha256(haystack_path),
                        "trajectories": _sha256(trajectories_path),
                    },
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    )
    protocol = assess_comparability(
        get_protocol_manifest("longmemeval-v2"),
        {
            "official_runner": True,
            "official_scorer": True,
            "dataset_verified": data_verification["verified"],
            "dataset_revision": data_verification["checksum_manifest_sha256"],
            "dataset_fingerprint": subset_fingerprint,
            "upstream_revision": revision,
            "num_samples": len(question_ids),
            "reader_model": args.model,
            "embedding_model": "text-embedding-3-small",
            "judge_model": args.evaluator_model,
            "top_k": args.top_k,
            "decoding": {
                "strategy": "provider-default",
                "temperature": None,
                "seed": None,
                "quantization": None,
            },
            "prompt_hash": "sha256:"
            + hashlib.sha256(
                (official_root / "evaluation" / "harness.py").read_bytes()
            ).hexdigest(),
            "scorer_hash": _sha256(official_root / "evaluation" / "qa_eval_metrics.py"),
            "dependency_lock_hash": _combined_hash(
                [
                    official_root / "requirements.txt",
                    official_root / "requirements-torch.txt",
                    official_root / "pyproject.toml",
                    PROJECT_ROOT / "pyproject.toml",
                ],
                root=official_root,
            ),
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "seed": args.seed,
        },
        profile="paper" if len(question_ids) == 451 else "smoke",
    )
    manifest_path = output_dir / "memorizz-evaluation-manifest.json"
    _write_json(
        manifest_path,
        {
            "benchmark": "LongMemEval-V2",
            "official_repository": "https://github.com/xiaowu0162/LongMemEval-V2",
            "official_revision": revision,
            "tier": "small",
            "domain": domain,
            "question_ids": question_ids,
            "question_count": len(question_ids),
            "trajectory_count": trajectory_count,
            "reader_model": args.model,
            "evaluator_model": args.evaluator_model,
            "memory_backend": args.memory_backend,
            "learning_control_plane": not args.legacy_memory_path,
            "paper_comparable": protocol["paper_comparable"],
            "comparison_label": protocol["comparison_label"],
            "protocol": protocol,
            "non_comparability_reasons": protocol["non_comparability_reasons"],
            "features": {
                "semantic_vector_retrieval": True,
                "hybrid_lexical_retrieval": args.lexical_ratio > 0,
                "source_diversified_retrieval": args.lexical_ratio > 0,
                "accessibility_tree_boundary_preservation": True,
                "semantic_cache_with_exact_probe": not args.no_cache_probe,
                "summarization_and_compaction": not args.no_summaries,
                "tenant_and_thread_isolation": True,
                "bounded_evidence_pack": not args.legacy_memory_path,
            },
            "dataset_verification": data_verification,
            "cost_forecast": cost_forecast,
            "max_estimated_cost_usd": args.max_estimated_cost_usd,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    original_argv = sys.argv[:]
    sys.argv = [
        str(official_root / "evaluation" / "harness.py"),
        "--domain",
        domain,
        "--questions-path",
        str(questions_path),
        "--haystack-path",
        str(haystack_path),
        "--trajectories-path",
        str(trajectories_path),
        "--memory-config-path",
        str(memory_config_path),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--max-completion-tokens",
        "4096",
        "--memory-context-max-tokens",
        "50000",
        "--prompt-build-max-workers",
        "1",
        "--reader-max-concurrent-requests",
        "1",
        "--reasoning-effort",
        "low",
        "--shuffle-questions-seed",
        str(args.seed),
        "--evaluator-model",
        args.evaluator_model,
        "--evaluator-reasoning-effort",
        "low",
    ]
    try:
        harness.main()
    finally:
        sys.argv = original_argv

    aggregated_path = output_dir / "aggregated_metrics.json"
    per_question_path = output_dir / "per_question.jsonl"
    aggregated = json.loads(aggregated_path.read_text(encoding="utf-8"))
    per_question = list(_read_jsonl(per_question_path))
    reader_usage = dict(aggregated.get("tokens") or {})
    reader_cost = estimate_openai_text_cost(args.model, reader_usage)
    memory_metadata = [
        dict(row.get("memory_post_query_metadata") or {}) for row in per_question
    ]
    memory_model_cost = max(
        (
            float(row.get("memory_model_estimated_cost_usd") or 0.0)
            for row in memory_metadata
        ),
        default=0.0,
    )
    estimated_total = (
        float(cost_forecast["cost_usd"]["embedding"])
        + reader_cost
        + memory_model_cost
        + float(cost_forecast["cost_usd"]["judge"])
    )
    accounting = {
        "embedding_cost_usd_estimated_from_preflight_tokens": float(
            cost_forecast["cost_usd"]["embedding"]
        ),
        "memory_model_cost_usd_from_reported_usage": round(memory_model_cost, 6),
        "official_reader_cost_usd_from_reported_usage": round(reader_cost, 6),
        "llm_judge_cost_usd_estimated": float(cost_forecast["cost_usd"]["judge"]),
        "estimated_total_usd": round(estimated_total, 6),
        "invoice_is_authoritative": True,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "official_metrics": aggregated,
            "actual_cost_accounting": accounting,
        }
    )
    _write_json(manifest_path, manifest)
    _write_json(
        output_dir / "memorizz-run-summary.json",
        {
            "benchmark": "LongMemEval-V2",
            "comparison_label": protocol["comparison_label"],
            "paper_comparable": protocol["paper_comparable"],
            "question_ids": question_ids,
            "metrics": aggregated,
            "cost": accounting,
            "memory": memory_metadata,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "paper_comparable": protocol["paper_comparable"],
                "overall": aggregated.get("overall"),
                "estimated_cost_usd": accounting["estimated_total_usd"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
