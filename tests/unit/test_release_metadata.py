"""Keep public release identities and private-artifact policy synchronized."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_release_versions_and_curated_changelog_agree():
    project = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE).group(1)
    module = ast.parse((ROOT / "src/memorizz/__init__.py").read_text())
    exported = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    )
    npm = json.loads((ROOT / "packaging/npm/package.json").read_text())
    assert exported == version == npm["version"]
    assert re.search(
        rf"^## {re.escape(version)} — \d{{4}}-\d{{2}}-\d{{2}}$",
        (ROOT / "CHANGELOG.md").read_text(),
        re.MULTILINE,
    )


def test_local_audit_reports_are_excluded_from_public_sources():
    assert "/reports/" in (ROOT / ".gitignore").read_text().splitlines()
    project = (ROOT / "pyproject.toml").read_text()
    sdist = project.split("[tool.hatch.build.targets.sdist]", 1)[1]
    assert '"reports/**"' in sdist
    assert '"docs/releases/**"' in sdist
    assert '"docs/**/*report*.md"' in sdist
    assert not (ROOT / "docs/releases").exists()
    assert not list((ROOT / "docs").rglob("*report*.md"))
    navigation = (ROOT / "mkdocs.yml").read_text()
    assert "Observability Incident Report:" not in navigation
    assert "Release Verification:" not in navigation
