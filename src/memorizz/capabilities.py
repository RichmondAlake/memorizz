"""Stable package feature report for deployment/runtime checks."""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict


def _dependency(name: str, import_name: str | None = None) -> Dict[str, Any]:
    module = import_name or name.replace("-", "_")
    installed = importlib.util.find_spec(module) is not None
    resolved = None
    if installed:
        try:
            resolved = version(name)
        except PackageNotFoundError:
            pass
    return {"installed": installed, "version": resolved}


def _browser_python(entry_point: str | None) -> str | None:
    if not entry_point:
        return None
    explicit = str(os.getenv("MEMORIZZ_BROWSER_USE_PYTHON_COMMAND", "")).strip()
    if explicit:
        if os.path.isabs(explicit):
            return (
                explicit
                if os.path.isfile(explicit) and os.access(explicit, os.X_OK)
                else None
            )
        return shutil.which(explicit)
    path = Path(entry_point).resolve()
    try:
        with path.open("r", encoding="utf-8") as entry_file:
            line = entry_file.readline().strip()
    except (OSError, UnicodeError):
        line = ""
    if line.startswith("#!"):
        try:
            parts = shlex.split(line[2:].strip())
        except ValueError:
            parts = []
        if parts and os.path.isabs(parts[0]) and os.access(parts[0], os.X_OK):
            return parts[0]
    for name in ("python", "python3", "python.exe"):
        candidate = path.parent / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def capabilities() -> Dict[str, Any]:
    """Return versioned feature states instead of relying on a version floor."""
    from . import __version__

    configured_graalpy = str(os.getenv("GRAALPY_PATH", "")).strip()
    graalpy_path = (
        configured_graalpy
        if configured_graalpy and os.path.isfile(configured_graalpy)
        else shutil.which("graalpy")
    )
    e2b = _dependency("e2b")
    e2b_code_interpreter = _dependency("e2b-code-interpreter", "e2b_code_interpreter")
    oracle = _dependency("oracledb")
    mcp = _dependency("mcp")
    cryptography = _dependency("cryptography")
    uvicorn = _dependency("uvicorn")
    mcp_client_ready = bool(mcp["installed"] and cryptography["installed"])
    mcp_server_ready = bool(mcp_client_ready and uvicorn["installed"])
    configured_browser_command = str(
        os.getenv("MEMORIZZ_BROWSER_USE_COMMAND", "browser-use")
    ).strip()
    browser_use_path = (
        configured_browser_command
        if os.path.isabs(configured_browser_command)
        and os.path.isfile(configured_browser_command)
        and os.access(configured_browser_command, os.X_OK)
        else shutil.which(configured_browser_command)
    )
    browser_python_path = _browser_python(browser_use_path)
    browser_llm_provider = (
        str(os.getenv("MEMORIZZ_BROWSER_USE_LLM_PROVIDER", "openai")).strip().lower()
    )
    browser_required_key = {
        "browseruse": "BROWSER_USE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(browser_llm_provider)
    browser_llm_key_configured = bool(
        browser_required_key and os.getenv(browser_required_key)
    )

    return {
        "package": "memorizz",
        "version": __version__,
        "capability_schema": 4,
        "features": {
            "mcp_client": {
                "available": mcp_client_ready,
                "transports": ["stdio", "streamable-http", "sse"],
                "install_extra": "mcp",
            },
            "mcp_server": {
                "available": mcp_server_ready,
                "transports": ["stdio", "streamable-http"],
                "install_extra": "mcp",
                "tool_count": 24,
                "strict_input_schemas": True,
                "resources": 4,
                "prompts": 1,
                "local_agent_lifecycle": [
                    "create",
                    "read",
                    "update",
                    "delete",
                    "execute",
                ],
                "tenant_scoped_operations": [
                    "memory",
                    "conversations",
                    "observability",
                    "semantic_cache",
                    "learning",
                    "compaction",
                ],
            },
            "agent_creation": {
                "available": True,
                "sdk": True,
                "cli": True,
                "ui": True,
                "mcp_local_stdio": mcp_server_ready,
                "mcp_remote_http": False,
            },
            "headless_runtime": {
                "available": True,
                "sdk": True,
                "cli": True,
                "mcp_stdio": mcp_server_ready,
                "display_server_required": False,
                "ui_dependency_required": False,
            },
            "meta_harness": {
                "available": True,
                "adapters": ["memagent", "codex", "claude-code", "openhands"],
                "routing": "deterministic_policy_and_outcome",
                "durable_runs": True,
                "durable_cancellation": True,
                "exact_envelope_approval": True,
                "workspace_leases": True,
                "host_verification": True,
                "memory_context_packs": True,
                "ranked_evidence_context": True,
                "workflow_context_snapshot_reuse": True,
                "dependency_context_propagation": True,
                "consolidation_strategies": ["model", "deterministic", "primary"],
                "verified_read_only_delegation_cache": True,
                "verified_continual_learning_evidence": True,
                "coordination_scope": "durable_single_node",
                "sdk": True,
                "cli": True,
                "ui": True,
                "mcp": mcp_server_ready,
            },
            "agent_surface_parity": {
                "available": mcp_server_ready,
                "common_operations": [
                    "create",
                    "read",
                    "update",
                    "delete",
                    "execute",
                    "memory",
                    "conversations",
                    "capabilities",
                    "observability",
                    "semantic_cache",
                    "learning",
                    "compaction",
                    "harness_execution",
                ],
                "trusted_host_only": [
                    "credential_management",
                    "local_path_ingestion",
                    "approval_decisions",
                    "outbound_mcp_configuration",
                ],
            },
            "thread_scoped_summaries": {"available": True},
            "retrieval_policy": {"available": True},
            "personalization_context": {
                "available": True,
                "provider_neutral": True,
                "explicit_cross_thread_recall": True,
                "natural_use_policy": True,
                "surfaces": ["sdk", "mcp", "observability", "ui"],
            },
            "concurrency_safe_run_state": {"available": True},
            "structured_context_provenance": {
                "available": True,
                "raw_request_content_persisted": False,
                "cache_decisions": True,
            },
            "memory_supply_observability": {
                "available": True,
                "stages": ["retrieved", "supplied", "referenced"],
                "raw_memory_content_persisted": False,
            },
            "logical_tool_trace_names": {"available": True},
            "semantic_cache_session_default": {"available": True},
            "durable_approvals": {"available": True, "single_use": True},
            "progressive_tool_disclosure": {"available": True},
            "tool_result_offloading": {"available": True, "size_aware": True},
            "structured_tool_outcomes": {
                "available": True,
                "statuses": [
                    "success",
                    "empty",
                    "degraded",
                    "fallback",
                    "provider_error",
                    "error",
                ],
                "surfaces": ["sdk", "cli", "mcp", "ui", "observability"],
            },
            "semantic_cache_governance": {"available": True},
            "entity_memory": {
                "available": True,
                "strict_tenant_scope": True,
                "bounded_exact_fallback": True,
                "explicit_legacy_migration": True,
                "control_plane_parity": True,
                "canonical_identity_keys": True,
                "dry_run_consolidation": True,
                "soft_supersede_rollback": True,
            },
            "canonical_entity_identity": {
                "available": True,
                "deterministic_ids": True,
                "identity_key_authority": "host_or_application",
                "dry_run_consolidation": True,
                "explicit_cross_name_selection": True,
                "soft_supersede_rollback": True,
            },
            "structured_entity_memory_tools": {
                "available": True,
                "nested_input_schema": True,
                "attribute_alias_normalization": True,
            },
            "host_completion_policy": {
                "available": True,
                "stream_buffered_until_acceptance": True,
                "cache_hits_revalidated": True,
                "runtime_validator_code_persisted": False,
            },
            "answer_streaming": {
                "available": True,
                "event_contract_version": 1,
                "sdk": ["run_stream_events", "arun_stream_events"],
                "default": True,
                "delivery_modes": ["final_stream", "buffered"],
                "gated_default": "buffered",
                "cli_opt_out": "--no-stream",
                "mcp_default": "progress",
                "mcp_answer_extension": "memorizz.events.v1",
                "mcp_extension_opt_in_required": True,
                "mcp_tested_sdk": "2.0.0",
                "resumption": False,
                "bounded_queue": True,
                "cancellation": "cooperative",
                "synchronous_compatibility_api": "run",
            },
            "evaluation_suite": {
                "available": True,
                "adapters": [
                    "terminal-bench-2.1",
                    "agentmembench",
                    "locomo-plus",
                    "longmemeval-v2",
                    "beam",
                    "memoryagentbench",
                    "swe-bench-lite",
                    "membench",
                ],
                "official_graders_preserved": True,
                "protocol_manifests": True,
                "comparability_gate": "fail_closed",
                "profiles": ["smoke", "regression", "paper"],
                "memory_providers": ["filesystem", "oracle"],
                "corpus_embedding_cache": True,
                "retrieval_reader_decomposition": True,
            },
            "learning_control_plane": {
                "available": True,
                "provider_neutral": True,
                "event_sourced": True,
                "bounded_evidence": True,
                "verified_outcomes": True,
                "reversible_forgetting": True,
                "providers": ["filesystem", "mongodb", "oracle"],
            },
            "sandbox_e2b_v2": {"available": True},
            "graalpy_untrusted_wrapper": {
                "available": True,
                "source_shipped": True,
                "prebuilt_jar": False,
                "requires_matching_graalvm_build_and_validation": True,
            },
            "multi_agent_orchestration": {"available": True},
            "oracle_summary_compaction": {"available": True, "atomic": True},
            "oracle_bootstrap_preflight": {"available": True},
            "governed_semantic_layer": {"available": True},
            "browser_control": {
                "available": True,
                "providers": ["browseruse"],
                "durable_approval_required": True,
                "arbitrary_model_code": False,
            },
        },
        "dependencies": {
            "mcp": mcp,
            "cryptography": cryptography,
            "uvicorn": uvicorn,
            "oracledb": oracle,
            "e2b": e2b,
            "e2b_code_interpreter": e2b_code_interpreter,
            "graalpy": {
                "installed": bool(graalpy_path),
                "version": None,
                "path": graalpy_path,
            },
            "browser_use_cli": {
                "installed": bool(browser_use_path),
                "version": None,
                "path": browser_use_path,
                "python_path": browser_python_path,
                "isolated_worker_process": True,
            },
        },
        "providers": {
            "mcp": {
                "ready": mcp_client_ready,
                "server_ready": mcp_server_ready,
                "install_extra": "mcp",
                **mcp,
            },
            "oracle": {"ready": bool(oracle["installed"]), **oracle},
            "e2b": {
                "ready": bool(
                    e2b["installed"]
                    and e2b_code_interpreter["installed"]
                    and os.getenv("E2B_API_KEY")
                ),
                "api_key_configured": bool(os.getenv("E2B_API_KEY")),
                "sdk": e2b,
                "code_interpreter_sdk": e2b_code_interpreter,
            },
            "graalpy": {
                "ready": bool(graalpy_path),
                "path": graalpy_path,
                "subprocess_security_boundary": "execution_provider_only",
            },
            "browseruse": {
                "ready": bool(
                    browser_use_path
                    and browser_python_path
                    and browser_llm_key_configured
                ),
                "path": browser_use_path,
                "python_path": browser_python_path,
                "llm_key_configured": browser_llm_key_configured,
                "llm_provider": browser_llm_provider,
                "required_key_env": browser_required_key,
                "execution_boundary": "isolated_browser_use_python",
                "durable_approval_required": True,
            },
        },
    }


__all__ = ["capabilities"]
