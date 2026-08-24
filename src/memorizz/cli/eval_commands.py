"""Protocol-aware memory evaluation commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

eval_app = typer.Typer(help="Run and inspect reproducible memory evaluations.")
dataset_app = typer.Typer(help="Synchronize and verify official benchmark assets.")
protocol_app = typer.Typer(help="Inspect versioned paper-reproduction manifests.")
terminal_bench_app = typer.Typer(
    help="Forecast Terminal-Bench 2.1 cost and rank without submitting."
)
eval_app.add_typer(dataset_app, name="dataset")
eval_app.add_typer(protocol_app, name="protocol")
eval_app.add_typer(terminal_bench_app, name="terminal-bench")


def _emit(value, *, compact: bool = False) -> None:
    typer.echo(json.dumps(value, indent=None if compact else 2, sort_keys=True))


@eval_app.command("list")
def list_benchmarks(compact: bool = typer.Option(False, "--compact")) -> None:
    """List benchmark adapters, protocol versions, and local readiness."""

    from ..benchmarks.memory_suite import BENCHMARK_CATALOG, get_protocol_manifest

    rows = []
    for spec in BENCHMARK_CATALOG.values():
        row = spec.to_dict()
        manifest = get_protocol_manifest(spec.benchmark_id)
        row.update(
            {
                "protocol_version": manifest.protocol_version,
                "source_sync_supported": manifest.source_sync_supported,
                "default_comparison_label": "Diagnostic",
            }
        )
        rows.append(row)
    _emit(rows, compact=compact)


@protocol_app.command("show")
def show_protocol(
    benchmark_id: str = typer.Argument(..., help="Benchmark identifier."),
    compact: bool = typer.Option(False, "--compact"),
) -> None:
    """Print the exact manifest used by the comparability gate."""

    from ..benchmarks.memory_suite import get_protocol_manifest

    try:
        manifest = get_protocol_manifest(benchmark_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(manifest.to_dict(), compact=compact)


@dataset_app.command("verify")
def dataset_verify(
    benchmark_id: str = typer.Argument(..., help="Benchmark identifier."),
    data_path: Optional[Path] = typer.Option(None, "--data-path"),
    source_path: Optional[Path] = typer.Option(None, "--source-path"),
    variant: Optional[str] = typer.Option(None, "--variant"),
    deep: bool = typer.Option(False, "--deep", help="Hash directory contents."),
    compact: bool = typer.Option(False, "--compact"),
) -> None:
    """Verify required assets and emit content/revision fingerprints."""

    from ..benchmarks.memory_suite import verify_dataset

    try:
        report = verify_dataset(
            benchmark_id,
            data_path=data_path,
            source_path=source_path,
            variant=variant,
            deep_checksum=deep,
        )
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(report, compact=compact)
    if not report["ready"]:
        raise typer.Exit(1)


@dataset_app.command("sync")
def dataset_sync(
    benchmark_id: str = typer.Argument(..., help="Benchmark identifier."),
    destination: Optional[Path] = typer.Option(None, "--destination"),
    compact: bool = typer.Option(False, "--compact"),
) -> None:
    """Clone/resume a pinned official source checkout without executing it."""

    from ..benchmarks.memory_suite import sync_dataset_source

    try:
        report = sync_dataset_source(benchmark_id, destination=destination)
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(report, compact=compact)


@terminal_bench_app.command("forecast")
def terminal_bench_forecast(
    pilot_json: Optional[Path] = typer.Option(
        None,
        "--pilot-json",
        help="Optional normalized JSON with a trials/results array.",
    ),
    model: str = typer.Option("openai/gpt-5.6-terra", "--model"),
    per_trial_spend_guard_usd: float = typer.Option(
        1.75, "--per-trial-spend-guard-usd"
    ),
    total_budget_usd: float = typer.Option(1_000.0, "--total-budget-usd"),
    reference_full_run_cost_usd: float = typer.Option(
        421.15,
        "--reference-full-run-cost-usd",
        help="Dated external cost reference used only when no pilot costs exist.",
    ),
    output: Optional[Path] = typer.Option(None, "--output"),
    force: bool = typer.Option(False, "--force"),
    compact: bool = typer.Option(False, "--compact"),
) -> None:
    """Create an explicitly unofficial, budget-bounded forecast."""

    from ..benchmarks.terminal_bench_forecast import (
        forecast_terminal_bench,
        load_pilot_trials,
    )

    if per_trial_spend_guard_usd <= 0:
        raise typer.BadParameter("--per-trial-spend-guard-usd must be positive")
    if total_budget_usd <= 0:
        raise typer.BadParameter("--total-budget-usd must be positive")
    try:
        trials = load_pilot_trials(pilot_json) if pilot_json else []
        report = forecast_terminal_bench(
            pilot_trials=trials,
            model=model,
            per_trial_spend_guard_usd=per_trial_spend_guard_usd,
            total_budget_usd=total_budget_usd,
            reference_full_run_cost_usd=reference_full_run_cost_usd,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    if output:
        destination = output.expanduser().resolve()
        if destination.exists() and not force:
            raise typer.BadParameter(
                f"Refusing to overwrite {destination}; pass --force to replace it"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _emit(report, compact=compact)


@eval_app.command("run")
def run_evaluation(
    benchmark_id: str = typer.Argument(..., help="Benchmark identifier."),
    data_path: Optional[Path] = typer.Option(None, "--data-path"),
    source_path: Optional[Path] = typer.Option(None, "--source-path"),
    variant: Optional[str] = typer.Option(None, "--variant"),
    profile: str = typer.Option("smoke", "--profile"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    output: Optional[Path] = typer.Option(None, "--output"),
    workspace: Optional[Path] = typer.Option(None, "--workspace"),
    model_provider: str = typer.Option("ollama", "--model-provider"),
    model: str = typer.Option("qwen2.5:3b", "--model"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
    embedding_model: str = typer.Option("nomic-embed-text", "--embedding-model"),
    embedding_model_digest: Optional[str] = typer.Option(
        None, "--embedding-model-digest"
    ),
    memory_provider: str = typer.Option("filesystem", "--memory-provider"),
    top_k: int = typer.Option(8, "--top-k"),
    candidate_pool_size: int = typer.Option(256, "--candidate-pool-size"),
    lexical_weight: float = typer.Option(0.35, "--lexical-weight"),
    rerank_weight: float = typer.Option(0.15, "--rerank-weight"),
    query_expansion: bool = typer.Option(
        True, "--query-expansion/--no-query-expansion"
    ),
    reader_repair: bool = typer.Option(True, "--reader-repair/--no-reader-repair"),
    evaluation_mode: str = typer.Option("retrieval", "--evaluation-mode"),
    agent_template: Optional[Path] = typer.Option(None, "--agent-template"),
    oracle_reader: bool = typer.Option(True, "--oracle-reader/--no-oracle-reader"),
    semantic_memory: bool = typer.Option(
        True, "--semantic-memory/--no-semantic-memory"
    ),
    corpus_cache: bool = typer.Option(True, "--corpus-cache/--no-corpus-cache"),
    strict_paper: bool = typer.Option(False, "--strict-paper"),
    seed: int = typer.Option(0, "--seed"),
    reasoning_effort: str = typer.Option("low", "--reasoning-effort"),
    max_output_tokens: int = typer.Option(512, "--max-output-tokens"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run a diagnostic or strict official-protocol evaluation profile."""

    from .._env_io import load_layered_env
    from ..benchmarks.memory_suite import (
        default_dataset_root,
        get_benchmark_spec,
        run_memory_suite,
    )

    load_layered_env()
    try:
        spec = get_benchmark_spec(benchmark_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    configured_data = data_path or os.getenv(spec.dataset_env)
    if not configured_data:
        raise typer.BadParameter(f"--data-path is required (or set {spec.dataset_env})")
    provider_name = str(model_provider).strip().lower()
    if provider_name == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter(
            "OPENAI_API_KEY is required for --model-provider=openai"
        )
    effort = str(reasoning_effort or "").strip().lower()
    if effort not in {"none", "low", "medium", "high", "xhigh"}:
        raise typer.BadParameter(
            "--reasoning-effort must be none, low, medium, high, or xhigh"
        )
    if int(max_output_tokens) < 64:
        raise typer.BadParameter("--max-output-tokens must be at least 64")
    selected_evaluation_mode = str(evaluation_mode or "retrieval").strip().lower()
    if selected_evaluation_mode not in {"retrieval", "memagent"}:
        raise typer.BadParameter("--evaluation-mode must be retrieval or memagent")
    template_model = None
    if selected_evaluation_mode == "memagent":
        if agent_template is None:
            raise typer.BadParameter(
                "--agent-template is required for --evaluation-mode=memagent"
            )
        try:
            from ..memagent.models import MemAgentModel

            template_model = MemAgentModel.model_validate_json(
                agent_template.expanduser().read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(f"Unable to load agent template: {exc}") from exc
    destination = (
        output.expanduser().resolve()
        if output
        else default_dataset_root().parent
        / "results"
        / f"{spec.benchmark_id}-{profile}.json"
    )
    if destination.exists() and not force:
        raise typer.BadParameter(
            f"Refusing to overwrite {destination}; pass --force to replace it"
        )
    run_workspace = (
        workspace.expanduser().resolve()
        if workspace
        else destination.parent / f".{destination.stem}-workspace"
    )
    try:
        report = run_memory_suite(
            spec.benchmark_id,
            configured_data,
            variant=variant,
            limit=limit,
            profile=profile,
            output_path=destination,
            workspace=run_workspace,
            model_provider=provider_name,
            model_name=model,
            judge_model_name=judge_model,
            embedding_model=embedding_model,
            embedding_model_digest=embedding_model_digest,
            memory_backend=memory_provider,
            top_k=top_k,
            candidate_pool_size=candidate_pool_size,
            lexical_ratio=lexical_weight,
            rerank_weight=rerank_weight,
            query_expansion=query_expansion,
            reader_repair=reader_repair,
            evaluation_mode=selected_evaluation_mode,
            agent_template=template_model,
            oracle_reader=oracle_reader,
            semantic_memory=semantic_memory,
            corpus_cache_enabled=corpus_cache,
            strict_paper=strict_paper,
            source_path=source_path,
            seed=seed,
            reasoning_effort=effort,
            max_output_tokens=max_output_tokens,
            progress=lambda line: typer.echo(line, err=True),
        )
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _emit(
        {
            "output": str(destination),
            "benchmark": report["benchmark"],
            "profile": report["protocol"]["profile"]["name"],
            "comparison_label": report["comparison_label"],
            "evaluation_mode": selected_evaluation_mode,
            "paper_comparable": report["paper_comparable"],
            "overall_score": report["overall_score"],
            "retrieval_recall_at_k": report["retrieval"]["recall_at_k"],
            "oracle_reader_score": report["answer_quality"][
                "gold_evidence_oracle_score"
            ],
            "cost_usd": report["usage"]["cost_usd"],
        }
    )


__all__ = ["eval_app"]
