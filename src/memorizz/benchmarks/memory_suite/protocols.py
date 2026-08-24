"""Versioned protocol manifests and strict paper-comparability checks.

The shared MemoRizz runner is useful for diagnostics, but it is not the same
thing as an upstream benchmark implementation.  This module makes that
boundary machine-readable instead of relying on prose in a result file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class EvaluationProfile:
    """One stable evaluation-size policy."""

    name: str
    description: str
    default_limit: Optional[int]
    stratified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "default_limit": self.default_limit,
            "stratified": self.stratified,
        }


EVALUATION_PROFILES: Mapping[str, EvaluationProfile] = {
    "smoke": EvaluationProfile(
        name="smoke",
        description="One bounded case per available category, for integration checks.",
        default_limit=None,
    ),
    "regression": EvaluationProfile(
        name="regression",
        description="A fixed stratified subset suitable for repeatable CI comparisons.",
        default_limit=50,
    ),
    "paper": EvaluationProfile(
        name="paper",
        description="The complete official split through the unmodified upstream evaluator.",
        default_limit=None,
    ),
}


@dataclass(frozen=True)
class ProtocolManifest:
    """Pinned protocol evidence needed to evaluate a benchmark claim."""

    benchmark_id: str
    protocol_version: str
    repository_url: Optional[str]
    synchronization_revision: Optional[str]
    paper_sample_count: Optional[int]
    paper_reader_model: Optional[str]
    paper_embedding_model: Optional[str]
    paper_judge_model: Optional[str]
    paper_retrieval_k: Tuple[int, ...] = ()
    paper_decoding: Mapping[str, Any] = field(default_factory=dict)
    official_scorer_note: str = "Use the evaluator shipped by the official repository."
    official_runner_note: str = "Use the official benchmark lifecycle and input format."
    source_sync_supported: bool = True
    required_runtime_fields: Tuple[str, ...] = (
        "dataset_revision",
        "dataset_fingerprint",
        "prompt_hash",
        "scorer_hash",
        "dependency_lock_hash",
        "hardware",
        "seed",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "protocol_version": self.protocol_version,
            "repository_url": self.repository_url,
            "synchronization_revision": self.synchronization_revision,
            "paper_sample_count": self.paper_sample_count,
            "paper_reader_model": self.paper_reader_model,
            "paper_embedding_model": self.paper_embedding_model,
            "paper_judge_model": self.paper_judge_model,
            "paper_retrieval_k": list(self.paper_retrieval_k),
            "paper_decoding": dict(self.paper_decoding),
            "official_scorer_note": self.official_scorer_note,
            "official_runner_note": self.official_runner_note,
            "source_sync_supported": self.source_sync_supported,
            "required_runtime_fields": list(self.required_runtime_fields),
            "profiles": {
                name: profile.to_dict() for name, profile in EVALUATION_PROFILES.items()
            },
        }


PROTOCOL_MANIFESTS: Mapping[str, ProtocolManifest] = {
    "agentmembench": ProtocolManifest(
        benchmark_id="agentmembench",
        protocol_version="2026-08-22.1",
        repository_url=None,
        synchronization_revision=None,
        paper_sample_count=491,
        paper_reader_model="Qwen2.5-7B-Instruct",
        paper_embedding_model="BAAI/bge-small-en-v1.5",
        paper_judge_model=None,
        paper_retrieval_k=(3, 5, 10),
        paper_decoding={"quantization": "4-bit", "strategy": "greedy"},
        source_sync_supported=False,
        official_scorer_note=(
            "Reproduce each strategy and the paper's retrieval, answer, "
            "faithfulness, footprint, and latency metrics."
        ),
    ),
    "longmemeval-v2": ProtocolManifest(
        benchmark_id="longmemeval-v2",
        protocol_version="2026-08-22.1",
        repository_url="https://github.com/xiaowu0162/LongMemEval-V2",
        synchronization_revision="2cc8c540bdb87fe6761629b585e727e1c4704520",
        paper_sample_count=451,
        paper_reader_model="Qwen3.5-9B",
        paper_embedding_model="Qwen3-Embedding-8B",
        paper_judge_model="GPT-5.2",
        official_scorer_note=(
            "Run evaluation/harness.py and its official evaluator over the "
            "prepared tier/domain artifacts."
        ),
    ),
    "locomo-plus": ProtocolManifest(
        benchmark_id="locomo-plus",
        protocol_version="2026-08-22.1",
        repository_url="https://github.com/xjtuleeyf/Locomo-Plus",
        synchronization_revision="059f4e3d38f7f1f96765e8e2cb7de3097551bffb",
        paper_sample_count=None,
        paper_reader_model=None,
        paper_embedding_model=None,
        paper_judge_model=None,
        official_scorer_note=(
            "Use the repository's unified-input, prediction, and 0/0.5/1 judge scripts."
        ),
    ),
    "beam": ProtocolManifest(
        benchmark_id="beam",
        protocol_version="2026-08-22.1",
        repository_url="https://github.com/mohammadtavakoli78/BEAM",
        synchronization_revision="3e12035532eb85768f1a7cd779832b650c4b2ef9",
        paper_sample_count=2_000,
        paper_reader_model=None,
        paper_embedding_model=None,
        paper_judge_model=None,
        official_scorer_note="Use BEAM's ten-ability rubric evaluator unchanged.",
    ),
    "memoryagentbench": ProtocolManifest(
        benchmark_id="memoryagentbench",
        protocol_version="2026-08-22.1",
        repository_url="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        synchronization_revision="fe1735de8cf8b9908e1e3d3b5612afc815698062",
        paper_sample_count=None,
        paper_reader_model=None,
        paper_embedding_model=None,
        paper_judge_model=None,
        official_scorer_note="Use each task's official metric and aggregation code.",
        official_runner_note=(
            "Preserve incremental inject/query/update/conflict lifecycles rather "
            "than converting every task to a static document corpus."
        ),
    ),
}


def get_evaluation_profile(name: str) -> EvaluationProfile:
    normalized = str(name or "smoke").strip().lower()
    try:
        return EVALUATION_PROFILES[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown evaluation profile: {name}") from exc


def get_protocol_manifest(benchmark_id: str) -> ProtocolManifest:
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
        return PROTOCOL_MANIFESTS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown memory benchmark protocol: {benchmark_id}") from exc


def _matches(actual: Any, expected: Any) -> bool:
    return str(actual or "").strip().lower() == str(expected or "").strip().lower()


def assess_comparability(
    manifest: ProtocolManifest,
    actual: Mapping[str, Any],
    *,
    profile: str,
) -> Dict[str, Any]:
    """Return an evidence-backed comparability decision.

    ``paper_comparable`` is deliberately fail-closed.  A caller cannot obtain
    it merely by selecting the ``paper`` profile; it must also prove that the
    official runner, official scorer, full split, pinned source, models, and
    reproducibility metadata match the manifest.
    """

    selected_profile = get_evaluation_profile(profile)
    reasons = []
    if selected_profile.name != "paper":
        reasons.append(f"profile={selected_profile.name!r}, not the full paper profile")
    if not bool(actual.get("official_runner")):
        reasons.append("MemoRizz diagnostic runner used instead of the official runner")
    if not bool(actual.get("official_scorer")):
        reasons.append("MemoRizz diagnostic scorer used instead of the official scorer")
    if not bool(actual.get("dataset_verified")):
        reasons.append("official dataset assets were not verified")
    if manifest.paper_sample_count is not None and int(
        actual.get("num_samples") or 0
    ) != int(manifest.paper_sample_count):
        reasons.append(
            f"sample count {actual.get('num_samples')!r} does not match "
            f"the paper count {manifest.paper_sample_count}"
        )
    if manifest.synchronization_revision and not _matches(
        actual.get("upstream_revision"), manifest.synchronization_revision
    ):
        reasons.append("upstream source revision does not match the pinned manifest")
    for key, expected in (
        ("reader_model", manifest.paper_reader_model),
        ("embedding_model", manifest.paper_embedding_model),
        ("judge_model", manifest.paper_judge_model),
    ):
        if expected is not None and not _matches(actual.get(key), expected):
            reasons.append(f"{key}={actual.get(key)!r} does not match {expected!r}")
    if manifest.paper_retrieval_k and int(actual.get("top_k") or 0) not in set(
        manifest.paper_retrieval_k
    ):
        reasons.append(
            f"top_k={actual.get('top_k')!r} is outside "
            f"{list(manifest.paper_retrieval_k)!r}"
        )
    actual_decoding = actual.get("decoding") or {}
    for key, expected in manifest.paper_decoding.items():
        if not _matches(actual_decoding.get(key), expected):
            reasons.append(
                f"decoding.{key}={actual_decoding.get(key)!r} does not match {expected!r}"
            )
    for field_name in manifest.required_runtime_fields:
        if actual.get(field_name) in (None, "", {}):
            reasons.append(f"missing reproducibility field: {field_name}")

    comparable = not reasons
    official_adapter = bool(actual.get("official_runner"))
    label = (
        "Official protocol"
        if comparable
        else "Official adapter · diagnostic configuration"
        if official_adapter
        else "Diagnostic"
    )
    return {
        "paper_comparable": comparable,
        "comparison_label": label,
        "profile": selected_profile.to_dict(),
        "protocol_version": manifest.protocol_version,
        "manifest": manifest.to_dict(),
        "evidence": dict(actual),
        "non_comparability_reasons": reasons,
    }


def resolve_profile_limit(profile: str, requested_limit: Optional[int]) -> int:
    """Resolve the requested limit without silently truncating paper runs."""

    selected = get_evaluation_profile(profile)
    if requested_limit is not None:
        if int(requested_limit) < 0:
            raise ValueError("Evaluation limit cannot be negative")
        if selected.name == "paper" and int(requested_limit) > 0:
            raise ValueError("The paper profile cannot be combined with a sample limit")
        return int(requested_limit)
    return int(selected.default_limit or 0)


__all__ = [
    "EVALUATION_PROFILES",
    "PROTOCOL_MANIFESTS",
    "EvaluationProfile",
    "ProtocolManifest",
    "assess_comparability",
    "get_evaluation_profile",
    "get_protocol_manifest",
    "resolve_profile_limit",
]
