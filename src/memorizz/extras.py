# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Which optional extras are installed, and how to install a missing one.

The extras and their packages come from ``pyproject.toml`` when MemoRizz runs
from a source checkout, else from the installed package metadata (the same
``optional-dependencies`` table, as pip recorded it). An extra counts as
installed when every package it needs on this platform can be imported; the
check uses ``importlib.util.find_spec`` so nothing heavy (torch, FAISS) is
loaded just to look.

``missing_extra_message`` turns an ``ImportError`` for a driver (oracledb,
pymongo, …) into an instruction: which extra to install and the command.
"""

from __future__ import annotations

import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Development and documentation tooling, and the everything-bundle: not
# features a running MemoRizz uses, so the report leaves them out.
HIDDEN_EXTRAS = frozenset({"all", "dev", "docs", "notebooks"})

# Display names for the extras people install most.
EXTRA_LABELS = {
    "oracle": "Oracle",
    "mongodb": "MongoDB",
    "filesystem": "FileSystem vector search (FAISS)",
    "notion": "Notion",
    "cli": "Command line",
    "ui": "Local UI",
    "mcp": "MCP client and server",
    "desktop": "Desktop control",
    "local": "Local stack (Ollama + FAISS)",
    "anthropic": "Anthropic",
    "ollama": "Ollama",
    "voyageai": "Voyage AI",
    "huggingface": "Hugging Face models",
    "rerank": "Reranker",
    "finetune": "Fine-tuning",
    "mlx": "Apple MLX",
    "ingest-pdf": "PDF ingestion",
    "sandbox": "E2B sandbox",
    "sandbox-e2b": "E2B sandbox",
    "sandbox-daytona": "Daytona sandbox",
    "terminal-bench": "Terminal-Bench",
    "swe-bench": "SWE-bench",
    "membench-eval": "MemBench evaluation",
    "longmemeval-v2": "LongMemEval v2",
}

# A distribution whose import name is not its normalised project name.
_IMPORT_NAMES = {
    "faiss-cpu": "faiss",
    "faiss-gpu": "faiss",
    "python-multipart": "multipart",
    "pyobjc-framework-applicationservices": "ApplicationServices",
    "pillow": "PIL",
    "mkdocs-material": "material",
    "pymdown-extensions": "pymdownx",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
}

# The extra to name when a module is missing, for modules several extras
# share (uvicorn is in both ui and mcp, for example). Others are looked up.
_PREFERRED_EXTRA = {
    "oracledb": "oracle",
    "pymongo": "mongodb",
    "bson": "mongodb",
    "faiss": "filesystem",
    "fastapi": "ui",
    "uvicorn": "ui",
    "jinja2": "ui",
    "multipart": "ui",
    "python_multipart": "ui",
    "starlette": "ui",
    "mcp": "mcp",
    "cryptography": "mcp",
    "anthropic": "anthropic",
    "voyageai": "voyageai",
    "sentence_transformers": "huggingface",
    "transformers": "huggingface",
    "huggingface_hub": "huggingface",
    "torch": "finetune",
    "pypdf": "ingest-pdf",
    "e2b": "sandbox-e2b",
    "e2b_code_interpreter": "sandbox-e2b",
    "daytona": "sandbox-daytona",
}

# Used only when neither pyproject.toml nor package metadata is readable.
_FALLBACK_EXTRAS: Dict[str, List[str]] = {
    "oracle": ["oracledb"],
    "mongodb": ["pymongo"],
    "filesystem": ["faiss-cpu"],
    "notion": [],
    "ui": ["fastapi", "uvicorn", "jinja2", "python-multipart"],
    "mcp": ["mcp", "uvicorn", "cryptography"],
    "anthropic": ["anthropic"],
    "voyageai": ["voyageai"],
    "huggingface": ["sentence-transformers", "transformers", "huggingface-hub"],
}


def import_name(distribution: str) -> str:
    """The module a distribution installs: ``faiss-cpu`` gives ``faiss``."""
    key = re.sub(r"[-_.]+", "-", str(distribution)).lower()
    return _IMPORT_NAMES.get(key, key.replace("-", "_"))


def _pyproject_extras() -> Optional[Dict[str, List[str]]]:
    """``[project.optional-dependencies]`` from the source checkout, if any."""
    candidate = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not candidate.is_file():
        return None
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    project = data.get("project") or {}
    if project.get("name") != "memorizz":
        return None
    table = project.get("optional-dependencies")
    if not isinstance(table, dict):
        return None
    return {str(name): [str(item) for item in items] for name, items in table.items()}


def _metadata_extras() -> Optional[Dict[str, List[str]]]:
    """The same table from the installed package's metadata."""
    try:
        from importlib.metadata import PackageNotFoundError, metadata, requires
    except ImportError:  # pragma: no cover
        return None
    try:
        names = metadata("memorizz").get_all("Provides-Extra") or []
        requirements = requires("memorizz") or []
    except PackageNotFoundError:
        return None
    except Exception:
        return None
    extras: Dict[str, List[str]] = {str(name): [] for name in names}
    for line in requirements:
        match = re.search(r"""extra\s*==\s*['"]([^'"]+)['"]""", line)
        if match and match.group(1) in extras:
            extras[match.group(1)].append(line)
    return extras


@lru_cache(maxsize=1)
def declared_extras() -> Dict[str, List[str]]:
    """Every extra with its requirement strings, in declaration order."""
    return _pyproject_extras() or _metadata_extras() or dict(_FALLBACK_EXTRAS)


def _requirement(text: str, extra: str) -> Tuple[str, frozenset, bool]:
    """``(name, extras, applies_here)`` for one requirement of ``extra``."""
    try:
        from packaging.requirements import Requirement

        requirement = Requirement(text)
        # Metadata scopes each line with ``extra == "…"``; the platform part
        # of the marker decides whether it applies on this computer.
        applies = requirement.marker is None or requirement.marker.evaluate(
            {"extra": extra}
        )
        return requirement.name, frozenset(requirement.extras), bool(applies)
    except Exception:
        name = re.split(r"[\s\[<>=!~;]", text.strip(), maxsplit=1)[0]
        return name, frozenset(), True


def _modules_for(extra: str, seen: Optional[set] = None) -> Tuple[List[str], bool]:
    """The modules an extra needs here, and whether any line was skipped
    because it does not apply to this platform."""
    seen = set() if seen is None else seen
    if extra in seen:
        return [], False
    seen.add(extra)
    modules: List[str] = []
    skipped = False
    for line in declared_extras().get(extra, []):
        name, nested, applies = _requirement(line, extra)
        if not applies:
            skipped = True
            continue
        if name.lower() == "memorizz":
            for inner in sorted(nested):
                inner_modules, inner_skipped = _modules_for(inner, seen)
                modules.extend(inner_modules)
                skipped = skipped or inner_skipped
            continue
        modules.append(import_name(name))
    return list(dict.fromkeys(modules)), skipped


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def install_command(extras: Iterable[str]) -> str:
    names = ",".join(dict.fromkeys(extras))
    return f"pip install 'memorizz[{names}]'"


def uv_command(extras: Iterable[str]) -> str:
    names = ",".join(dict.fromkeys(extras))
    return f"uv tool install --force 'memorizz[{names}]'"


def extras_report(*, include_hidden: bool = False) -> List[Dict[str, Any]]:
    """One row per extra: whether it is installed and what is missing.

    ``status`` is ``installed``, ``missing``, ``partial`` (some packages
    present), ``built-in`` (nothing beyond the base install) or
    ``unavailable`` (none of its packages apply to this platform).
    """
    rows: List[Dict[str, Any]] = []
    for extra in declared_extras():
        if extra in HIDDEN_EXTRAS and not include_hidden:
            continue
        modules, skipped = _modules_for(extra)
        missing = [module for module in modules if not _importable(module)]
        if not modules:
            status = "unavailable" if skipped else "built-in"
        elif not missing:
            status = "installed"
        elif len(missing) < len(modules):
            status = "partial"
        else:
            status = "missing"
        rows.append(
            {
                "extra": extra,
                "label": EXTRA_LABELS.get(extra, extra),
                "status": status,
                "installed": status in {"installed", "built-in"},
                "modules": modules,
                "missing": missing,
                "install": install_command([extra]),
            }
        )
    return rows


def extra_for_module(module: Optional[str]) -> Optional[str]:
    """The smallest extra that provides ``module`` (``oracledb`` gives oracle)."""
    if not module:
        return None
    top = str(module).split(".", 1)[0]
    if top in _PREFERRED_EXTRA:
        return _PREFERRED_EXTRA[top]
    candidates = []
    for extra in declared_extras():
        if extra in HIDDEN_EXTRAS:
            continue
        modules, _ = _modules_for(extra)
        if top in modules:
            candidates.append((len(modules), extra))
    return min(candidates)[1] if candidates else None


_NO_MODULE = re.compile(r"No module named ['\"]?([A-Za-z0-9_.]+)")
# Providers that re-raise a driver ImportError with their own wording.
_PACKAGE_WORDING = re.compile(r"\b(oracledb|pymongo|faiss|fastapi|uvicorn|mcp)\b")


def missing_module(exc: BaseException) -> Optional[str]:
    """The module an ImportError (or anything it caused) could not import."""
    seen = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImportError):
            if getattr(current, "name", None):
                return str(current.name)
            text = str(current)
            match = _NO_MODULE.search(text) or _PACKAGE_WORDING.search(text)
            if match:
                return match.group(1)
        current = current.__cause__ or current.__context__
    return None


def missing_extra_message(exc: BaseException, *, with_ui: bool = True) -> Optional[str]:
    """“Oracle needs the oracle extra: pip install 'memorizz[oracle]' (or uv
    tool install --force 'memorizz[ui,oracle]')”, or None when ``exc`` is
    not a missing optional dependency."""
    module = missing_module(exc)
    extra = extra_for_module(module)
    if not extra:
        return None
    label = EXTRA_LABELS.get(extra, extra)
    uv_extras = ["ui", extra] if with_ui and extra != "ui" else [extra]
    return (
        f"{label} needs the {extra} extra: {install_command([extra])} "
        f"(or {uv_command(uv_extras)})"
    )


__all__ = [
    "EXTRA_LABELS",
    "declared_extras",
    "extra_for_module",
    "extras_report",
    "import_name",
    "install_command",
    "missing_extra_message",
    "missing_module",
    "uv_command",
]
