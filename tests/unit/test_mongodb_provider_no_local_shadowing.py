"""Regression test for the v0.0.41 UnboundLocalError bug.

In v0.0.41 (and earlier), MongoDBProvider.store() and a few other methods
contained ``from ...enums.memory_type import MemoryType`` *inside conditional
branches*. Python's scoping rules then treated ``MemoryType`` as a function-
local for the entire function body — every reference earlier in the same
function (e.g. ``MemoryType.MEMAGENT`` at the dispatch point) raised
``UnboundLocalError: local variable 'MemoryType' referenced before assignment``
whenever execution didn't pass through the local-import branch first.

The fix in v0.0.42 removes the redundant local imports — the module-level
``from ...enums.memory_type import MemoryType`` at the top of provider.py is
sufficient. This test fails if anyone reintroduces the shadowing pattern.

This regression is hard to catch with the filesystem provider (which is used
in most unit tests), so we lint the source directly.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

PROVIDER_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "src"
    / "memorizz"
    / "memory_provider"
)


def _find_local_memorytype_imports(path: pathlib.Path):
    """Yield (function_name, lineno) for any function-body import of MemoryType."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if inner is node:
                continue
            if (
                isinstance(inner, ast.ImportFrom)
                and inner.module
                and "memory_type" in inner.module
            ):
                for alias in inner.names:
                    if alias.name == "MemoryType":
                        yield (node.name, inner.lineno)


@pytest.mark.parametrize(
    "provider_file",
    sorted(PROVIDER_DIR.rglob("provider.py")),
    ids=lambda p: str(p.relative_to(PROVIDER_DIR)),
)
def test_no_function_local_memorytype_import(provider_file):
    """No function in a provider module may locally re-import MemoryType.

    The module-level import at the top of each provider already binds
    ``MemoryType``; a local re-import in a conditional branch shadows it
    and breaks any earlier reference inside the same function.
    """
    offenders = list(_find_local_memorytype_imports(provider_file))
    assert not offenders, (
        f"{provider_file}: local `from ...memory_type import MemoryType` "
        f"shadows the module-level import. Offending sites: {offenders}. "
        f"This is what caused the v0.0.41 UnboundLocalError regression on "
        f"MongoDBProvider.store()."
    )
