#!/usr/bin/env python3
"""Run a paper-backed memory benchmark with local or OpenAI reader models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from memorizz._env_io import load_layered_env  # noqa: E402
from memorizz.benchmarks.memory_suite import (  # noqa: E402
    BENCHMARK_CATALOG,
    get_benchmark_spec,
    run_memory_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MemoRizz on AgentMemBench, LongMemEval-V2, LoCoMo-Plus, "
            "BEAM, or MemoryAgentBench with local Ollama or OpenAI readers."
        )
    )
    parser.add_argument(
        "--list", action="store_true", help="Print benchmark catalog JSON and exit."
    )
    parser.add_argument("--benchmark", choices=tuple(BENCHMARK_CATALOG))
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--variant")
    parser.add_argument(
        "--profile", choices=("smoke", "regression", "paper"), default="smoke"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument(
        "--model-provider", choices=("ollama", "openai"), default="ollama"
    )
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--judge-model")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--lexical-weight", "--lexical-ratio", type=float, default=0.35)
    parser.add_argument("--rerank-weight", type=float, default=0.15)
    parser.add_argument(
        "--query-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic label-blind concept expansion before retrieval.",
    )
    parser.add_argument(
        "--memory-provider", choices=("filesystem", "oracle"), default="filesystem"
    )
    parser.add_argument("--embedding-model-digest")
    parser.add_argument(
        "--evaluation-mode",
        choices=("retrieval", "memagent"),
        default="retrieval",
        help=(
            "retrieval runs the normalized diagnostic reader; memagent runs a "
            "selected agent template through MemAgent.run on isolated memory."
        ),
    )
    parser.add_argument(
        "--agent-template",
        type=Path,
        help="Secret-free MemAgentModel JSON required by --evaluation-mode=memagent.",
    )
    parser.add_argument(
        "--oracle-reader",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also answer from gold evidence to separate retrieval and reader quality.",
    )
    parser.add_argument(
        "--corpus-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse content-addressed corpus embeddings across reader comparisons.",
    )
    parser.add_argument(
        "--semantic-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Derive source-linked constraints, preferences, goals, and state updates.",
    )
    parser.add_argument(
        "--reader-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair malformed JSON-like reader output once.",
    )
    parser.add_argument("--corpus-cache-dir", type=Path)
    parser.add_argument("--strict-paper", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-document-chars", type=int, default=3_500)
    parser.add_argument("--max-evidence-chars", type=int, default=24_000)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default="low",
    )
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _catalog() -> list[dict[str, object]]:
    return [spec.to_dict() for spec in BENCHMARK_CATALOG.values()]


def main() -> int:
    args = parse_args()
    if args.list:
        print(json.dumps(_catalog(), indent=2))
        return 0
    if not args.benchmark:
        raise SystemExit("--benchmark is required unless --list is used")
    agent_template = None
    if args.evaluation_mode == "memagent":
        if args.agent_template is None:
            raise SystemExit("--agent-template is required in memagent evaluation mode")
        try:
            from memorizz.memagent.models import MemAgentModel

            agent_template = MemAgentModel.model_validate_json(
                args.agent_template.expanduser().read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise SystemExit(f"Unable to load MemAgent template: {exc}") from exc
    if args.model_provider == "openai":
        load_layered_env()
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required when --model-provider=openai")
    spec = get_benchmark_spec(args.benchmark)
    configured_data = args.data_path or os.getenv(spec.dataset_env)
    if not configured_data:
        raise SystemExit(
            f"--data-path is required (or set {spec.dataset_env} for {spec.name})"
        )
    output = (
        (
            args.output
            or (PROJECT_ROOT / "eval" / "results" / f"{spec.benchmark_id}.json")
        )
        .expanduser()
        .resolve()
    )
    if output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {output}; pass --force to replace it")
    workspace = (
        (args.workspace or output.parent / f".{output.stem}-workspace")
        .expanduser()
        .resolve()
    )
    report = run_memory_suite(
        spec.benchmark_id,
        configured_data,
        variant=args.variant,
        limit=args.limit,
        profile=args.profile,
        output_path=output,
        workspace=workspace,
        model_provider=args.model_provider,
        model_name=args.model,
        judge_model_name=args.judge_model,
        embedding_model=args.embedding_model,
        ollama_host=args.ollama_host,
        top_k=max(1, args.top_k),
        candidate_pool_size=max(1, args.candidate_pool_size),
        lexical_ratio=args.lexical_weight,
        rerank_weight=max(0.0, args.rerank_weight),
        query_expansion=args.query_expansion,
        max_document_chars=args.max_document_chars,
        max_evidence_chars=args.max_evidence_chars,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=max(64, args.max_output_tokens),
        oracle_reader=args.oracle_reader,
        seed=args.seed,
        corpus_cache_dir=args.corpus_cache_dir,
        corpus_cache_enabled=args.corpus_cache,
        embedding_model_digest=args.embedding_model_digest,
        memory_backend=args.memory_provider,
        semantic_memory=args.semantic_memory,
        reader_repair=args.reader_repair,
        evaluation_mode=args.evaluation_mode,
        agent_template=agent_template,
        strict_paper=args.strict_paper,
        progress=lambda line: print(line, flush=True),
    )
    cost = report["usage"].get("cost_usd")
    cost_label = "unpriced" if cost is None else f"${float(cost):.6f}"
    print(
        f"Wrote {output} | score={report['overall_score']:.3f} | "
        f"accuracy={report['overall_accuracy']:.3f} | "
        f"mode={report['comparison_label']} | cost={cost_label}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
