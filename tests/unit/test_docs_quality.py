"""Static quality gates for the published developer documentation."""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
EXCLUDED_PAGES = {
    Path("README.md"),
    Path("observability-roadmap.md"),
}


def _markdown_files() -> list[Path]:
    return sorted(DOCS_ROOT.rglob("*.md"))


def _developer_documentation_files() -> list[Path]:
    return [REPO_ROOT / "README.md", *_markdown_files()]


def _is_published(path: Path) -> bool:
    relative = path.relative_to(DOCS_ROOT)
    return relative not in EXCLUDED_PAGES and relative.parts[0] != "issues"


def test_published_pages_are_present_in_navigation() -> None:
    config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    nav_pages = {
        Path(value)
        for value in re.findall(r"^\s*-?\s*[^:\n]+:\s*([^#\s]+\.md)\s*$", config, re.M)
    }
    published = {
        path.relative_to(DOCS_ROOT) for path in _markdown_files() if _is_published(path)
    }

    assert published == nav_pages, (
        f"Missing from nav: {sorted(published - nav_pages)}; "
        f"missing files: {sorted(nav_pages - published)}"
    )


def test_python_code_fences_are_syntactically_valid() -> None:
    failures: list[str] = []
    pattern = re.compile(r"```python(?:[^\n]*)\n(.*?)```", re.S)

    for path in _developer_documentation_files():
        text = path.read_text(encoding="utf-8")
        for block_number, match in enumerate(pattern.finditer(text), start=1):
            try:
                ast.parse(textwrap.dedent(match.group(1)))
            except SyntaxError as exc:
                failures.append(
                    f"{path.relative_to(REPO_ROOT)} block {block_number}: "
                    f"line {exc.lineno}: {exc.msg}"
                )

    assert not failures, "\n".join(failures)


def test_local_markdown_links_resolve() -> None:
    failures: list[str] = []
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

    for path in _developer_documentation_files():
        for raw_target in pattern.findall(path.read_text(encoding="utf-8")):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            elif " " in target:
                target = target.split(" ", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            clean_target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            resolved = (path.parent / clean_target).resolve()
            if resolved.is_dir():
                resolved = resolved / "index.md"
            if not resolved.exists():
                failures.append(f"{path.relative_to(REPO_ROOT)} -> {raw_target}")

    assert not failures, "Broken local links:\n" + "\n".join(failures)


def test_developer_entry_points_do_not_regress_to_stale_guidance() -> None:
    entry_points = [
        REPO_ROOT / "README.md",
        DOCS_ROOT / "index.md",
        DOCS_ROOT / "getting-started" / "overview.md",
        DOCS_ROOT / "getting-started" / "python-sdk-quickstart.md",
        DOCS_ROOT / "getting-started" / "local-ui.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in entry_points)

    assert "Python 3.7" not in combined
    assert "No built-in authentication is enabled" not in combined
    assert "agent.memory.conversation_memory" not in combined
    assert "agent.memory.semantic_cache" not in combined


def test_metaharness_comparison_notebooks_are_safe_and_auditable() -> None:
    methodology_path = (
        REPO_ROOT
        / "examples"
        / "metaharness"
        / "05_single_vs_multi_harness_evaluation.ipynb"
    )
    topology_path = (
        REPO_ROOT
        / "examples"
        / "metaharness"
        / "06_fair_harness_comparison_results.ipynb"
    )
    artifact_path = (
        REPO_ROOT
        / "eval"
        / "results"
        / "2026-08-22-metaharness-provider-comparison-optimized.json"
    )
    factorial_artifact_path = (
        REPO_ROOT / "eval" / "results" / "2026-08-22-metaharness-factorial-paid.json"
    )
    methodology = json.loads(methodology_path.read_text(encoding="utf-8"))
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    factorial_artifact = json.loads(factorial_artifact_path.read_text(encoding="utf-8"))
    methodology_source = "\n".join(
        "".join(cell.get("source", [])) for cell in methodology.get("cells", [])
    )
    topology_source = "\n".join(
        "".join(cell.get("source", [])) for cell in topology.get("cells", [])
    )

    assert methodology.get("nbformat") == 4
    assert topology.get("nbformat") == 4
    assert all(not cell.get("outputs") for cell in methodology.get("cells", []))
    assert any(cell.get("outputs") for cell in topology.get("cells", []))
    serialized_topology = json.dumps(topology)
    assert "/Users/" not in serialized_topology
    assert "/private/" not in serialized_topology
    assert "sk-proj-" not in serialized_topology
    assert "sk-ant-" not in serialized_topology
    assert artifact["status"] == "valid"
    assert artifact["paper_comparable"] is False
    assert factorial_artifact["status"] == "valid_with_judge_limitation"
    assert factorial_artifact["valid"] is True
    assert factorial_artifact["quality_validity"]["primary_metric_valid"] is True
    assert (
        factorial_artifact["quality_validity"]["llm_judge_scalar_valid_for_comparison"]
        is False
    )
    assert "MEMORIZZ_RUN_HARNESS_EVALUATION" in methodology_source
    assert "if RUN_LIVE:" in methodology_source
    assert "CodexHarness" in topology_source
    assert "ClaudeCodeHarness" in topology_source
    assert "factorial_comparison" in topology_source
    assert "len(manifest['arms']) == 12" in topology_source
    assert ".with_execution_harness(" in topology_source
    assert "full_panel.model is None" in topology_source
    assert "adaptive_panel.model is None" in topology_source
    assert "consolidation_strategy='structured'" in topology_source
    assert "adaptive_escalation" in topology_source
    assert "winner_allowed" in topology_source
    assert "if RUN_LIVE:" in topology_source
    panel_rows = [
        row
        for row in artifact["results"]
        if str(row["arm"]).startswith("memorizz_panel")
    ]
    assert panel_rows
    assert all(row["harness_runs_mean"] == 1.0 for row in panel_rows)
    combined = methodology_source + topology_source
    assert "sk-proj-" not in combined
    assert "sk-ant-" not in combined
