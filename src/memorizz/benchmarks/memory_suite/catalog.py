"""Paper-backed benchmark catalog exposed by the CLI and Evalground UI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    name: str
    paper_url: str
    repository_url: str | None
    description: str
    abilities: Tuple[str, ...]
    variants: Tuple[str, ...]
    default_variant: str
    primary_metric: str
    dataset_env: str
    data_hint: str
    license_note: str

    def to_dict(self, *, dataset_path: str | None = None) -> Dict[str, Any]:
        from .protocols import get_protocol_manifest

        configured = dataset_path or os.getenv(self.dataset_env, "")
        exists = bool(configured and Path(configured).expanduser().exists())
        protocol = get_protocol_manifest(self.benchmark_id)
        return {
            "benchmark_id": self.benchmark_id,
            "name": self.name,
            "paper_url": self.paper_url,
            "repository_url": self.repository_url,
            "description": self.description,
            "abilities": list(self.abilities),
            "variants": list(self.variants),
            "default_variant": self.default_variant,
            "primary_metric": self.primary_metric,
            "dataset_env": self.dataset_env,
            "dataset_path": configured,
            "dataset_ready": exists,
            "data_hint": self.data_hint,
            "license_note": self.license_note,
            "protocol_version": protocol.protocol_version,
            "source_sync_supported": protocol.source_sync_supported,
            "pinned_source_revision": protocol.synchronization_revision,
            "comparison_label": "Diagnostic until the strict manifest passes",
        }


BENCHMARK_CATALOG: Mapping[str, BenchmarkSpec] = {
    "agentmembench": BenchmarkSpec(
        benchmark_id="agentmembench",
        name="AgentMemBench",
        paper_url="https://arxiv.org/abs/2608.00009",
        repository_url=None,
        description=(
            "Retrieval quality, answer quality, footprint, and latency across "
            "LoCoMo, MultiDoc2Dial, and MSC."
        ),
        abilities=(
            "long-range recall",
            "answer faithfulness",
            "retrieval efficiency",
            "memory footprint",
        ),
        variants=("locomo", "multidoc2dial", "msc"),
        default_variant="locomo",
        primary_metric="answer_f1",
        dataset_env="MEMORIZZ_AGENTMEMBENCH_DATA",
        data_hint=(
            "Point to a LoCoMo JSON file/directory or a normalized JSON/JSONL "
            "export containing context/history, question, answer, and evidence."
        ),
        license_note="Datasets retain their upstream licenses; no data is bundled.",
    ),
    "longmemeval-v2": BenchmarkSpec(
        benchmark_id="longmemeval-v2",
        name="LongMemEval-V2",
        paper_url="https://arxiv.org/abs/2605.12493",
        repository_url="https://github.com/xiaowu0162/LongMemEval-V2",
        description=(
            "Compact evidence gathering over long web-agent and enterprise "
            "trajectory histories."
        ),
        abilities=(
            "static state recall",
            "dynamic state tracking",
            "workflow knowledge",
            "environment gotchas",
            "premise awareness",
        ),
        variants=(
            "small-web",
            "small-enterprise",
            "medium-web",
            "medium-enterprise",
        ),
        default_variant="small-web",
        primary_metric="llm_judge",
        dataset_env="MEMORIZZ_LONGMEMEVAL_V2_DATA",
        data_hint=(
            "Point to the prepared data root containing questions.jsonl, "
            "trajectories.jsonl, and haystacks/lme_v2_{small,medium}.json."
        ),
        license_note="Official code is Apache-2.0; data remains external.",
    ),
    "locomo-plus": BenchmarkSpec(
        benchmark_id="locomo-plus",
        name="LoCoMo-Plus",
        paper_url="https://arxiv.org/abs/2602.10715",
        repository_url="https://github.com/xjtuleeyf/Locomo-Plus",
        description=(
            "Six-category conversational evaluation including implicit cognitive "
            "constraints under cue-trigger semantic disconnect."
        ),
        abilities=(
            "multi-hop",
            "temporal",
            "common-sense",
            "single-hop",
            "adversarial",
            "cognitive constraint consistency",
        ),
        variants=("cognitive", "original", "all"),
        default_variant="cognitive",
        primary_metric="llm_judge",
        dataset_env="MEMORIZZ_LOCOMO_PLUS_DATA",
        data_hint=(
            "Point to the official data directory containing locomo10.json and "
            "locomo_plus.json."
        ),
        license_note="Use the official repository's dataset terms.",
    ),
    "beam": BenchmarkSpec(
        benchmark_id="beam",
        name="BEAM",
        paper_url="https://arxiv.org/abs/2510.27246",
        repository_url="https://github.com/mohammadtavakoli78/BEAM",
        description=(
            "Long coherent conversations from 128K to 10M tokens with rubric "
            "evaluation across ten memory abilities."
        ),
        abilities=(
            "abstention",
            "contradiction resolution",
            "event ordering",
            "information extraction",
            "instruction following",
            "knowledge update",
            "multi-session reasoning",
            "preference following",
            "summarization",
            "temporal reasoning",
        ),
        variants=("128k", "500k", "1m", "10m"),
        default_variant="128k",
        primary_metric="llm_judge",
        dataset_env="MEMORIZZ_BEAM_DATA",
        data_hint=(
            "Point to the BEAM repository/data root containing chats/<scale>/"
            "<chat>/chat.json and probing_questions/probing_questions.json."
        ),
        license_note="BEAM data is CC BY-SA 4.0; code is MIT.",
    ),
    "memoryagentbench": BenchmarkSpec(
        benchmark_id="memoryagentbench",
        name="MemoryAgentBench",
        paper_url="https://arxiv.org/abs/2507.05257",
        repository_url="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        description=(
            "Incremental inject-once/query-many evaluation across four memory "
            "competencies."
        ),
        abilities=(
            "accurate retrieval",
            "test-time learning",
            "long-range understanding",
            "conflict resolution",
        ),
        variants=(
            "accurate-retrieval",
            "test-time-learning",
            "long-range-understanding",
            "conflict-resolution",
        ),
        default_variant="accurate-retrieval",
        primary_metric="task_specific",
        dataset_env="MEMORIZZ_MEMORYAGENTBENCH_DATA",
        data_hint=(
            "Point to a JSON/JSONL export of ai-hyz/MemoryAgentBench. Each row "
            "must retain context, questions, answers, and metadata.source."
        ),
        license_note="Official benchmark code is MIT; data remains external.",
    ),
}


def get_benchmark_spec(benchmark_id: str) -> BenchmarkSpec:
    normalized = str(benchmark_id or "").strip().lower().replace("_", "-")
    aliases = {
        "agent-mem-bench": "agentmembench",
        "longmemevalv2": "longmemeval-v2",
        "lme-v2": "longmemeval-v2",
        "locomoplus": "locomo-plus",
        "memory-agent-bench": "memoryagentbench",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return BENCHMARK_CATALOG[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown memory benchmark: {benchmark_id}") from exc


__all__ = ["BENCHMARK_CATALOG", "BenchmarkSpec", "get_benchmark_spec"]
