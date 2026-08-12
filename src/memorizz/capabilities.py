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
        "capability_schema": 1,
        "features": {
            "mcp_client": {
                "available": True,
                "transports": ["stdio", "streamable-http", "sse"],
            },
            "mcp_server": {
                "available": True,
                "transports": ["stdio", "streamable-http"],
            },
            "durable_approvals": {"available": True, "single_use": True},
            "progressive_tool_disclosure": {"available": True},
            "tool_result_offloading": {"available": True, "size_aware": True},
            "semantic_cache_governance": {"available": True},
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
            "mcp": {"ready": bool(mcp["installed"]), **mcp},
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
