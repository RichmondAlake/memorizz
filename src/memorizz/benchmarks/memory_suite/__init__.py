"""Unified, local-first adapters for public long-term-memory benchmarks.

The package keeps benchmark-specific loading and scoring semantics while
sharing the expensive ingestion, retrieval, model invocation, and reporting
plumbing.  Official datasets remain external artifacts and are never vendored
with MemoRizz.
"""

from .catalog import BENCHMARK_CATALOG, BenchmarkSpec, get_benchmark_spec
from .datasets import default_dataset_root, sync_dataset_source, verify_dataset
from .loaders import DatasetLoadError, load_benchmark_cases
from .models import MemoryBenchmarkCase, MemoryDocument
from .protocols import (
    EVALUATION_PROFILES,
    PROTOCOL_MANIFESTS,
    EvaluationProfile,
    ProtocolManifest,
    assess_comparability,
    get_evaluation_profile,
    get_protocol_manifest,
    resolve_profile_limit,
)
from .retrieval import FusionConfig, build_query_variants, fuse_rankings
from .runner import MemorySuiteRunner, run_memory_suite
from .semantic_memory import derive_semantic_memories

__all__ = [
    "BENCHMARK_CATALOG",
    "BenchmarkSpec",
    "DatasetLoadError",
    "EVALUATION_PROFILES",
    "PROTOCOL_MANIFESTS",
    "EvaluationProfile",
    "FusionConfig",
    "MemoryBenchmarkCase",
    "MemoryDocument",
    "MemorySuiteRunner",
    "ProtocolManifest",
    "assess_comparability",
    "build_query_variants",
    "default_dataset_root",
    "derive_semantic_memories",
    "fuse_rankings",
    "get_evaluation_profile",
    "get_benchmark_spec",
    "get_protocol_manifest",
    "load_benchmark_cases",
    "resolve_profile_limit",
    "run_memory_suite",
    "sync_dataset_source",
    "verify_dataset",
]
