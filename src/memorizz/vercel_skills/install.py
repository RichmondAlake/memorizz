# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Save marketplace skills locally so they can be attached to an agent.

An attached skill is a SKILL.md under ``$MEMORIZZ_HOME/skills`` whose path is
in the agent's ``skill_paths``. The agent lists it with ``list_skills`` and
reads it with ``read_skill``; nothing in the skill is executed on install.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .._env_io import memorizz_home
from .provider import VercelSkillsProvider
from .skill_md import parse_skill_md

SOURCE_FILE = ".memorizz-source.json"
MAX_SKILL_BYTES = 512_000


def skills_root() -> Path:
    return memorizz_home() / "skills"


def _safe_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-.")
    return part[:100] or "skill"


def install_skill(
    repo: str,
    skill_name: Optional[str] = None,
    *,
    branch: str = "main",
    provider: Optional[VercelSkillsProvider] = None,
    root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fetch a skill and write it to ``<root>/<owner>/<repo>/<skill>/SKILL.md``."""
    provider = provider or VercelSkillsProvider()
    result = provider.fetch_skill(
        repo=repo, skill_name=skill_name or None, branch=branch
    )
    if not result.get("ok"):
        return result
    if result.get("multi_skill_repo"):
        error = (
            f"'{skill_name}' was not found in {result.get('repo') or repo}. "
            "Choose one of its skills."
            if skill_name
            else "This repository holds several skills. Choose one to attach."
        )
        return {
            "ok": False,
            "error": error,
            "available_skills": result.get("available_skills") or [],
        }
    content = str(result.get("content") or "")
    if not content.strip():
        return {"ok": False, "error": "The skill file is empty."}
    if len(content.encode("utf-8")) > MAX_SKILL_BYTES:
        return {"ok": False, "error": "The skill file is larger than 500 KB."}

    owner, _, repo_name = str(result["repo"]).partition("/")
    source_path = str(result.get("path") or "SKILL.md")
    folder_name = Path(source_path).parent.name or repo_name
    base = (root or skills_root()).resolve()
    folder = (
        base / _safe_part(owner) / _safe_part(repo_name) / _safe_part(folder_name)
    ).resolve()
    if base not in folder.parents:
        return {"ok": False, "error": "Refusing to write outside the skills folder."}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    source = {
        "repo": result["repo"],
        "path": source_path,
        "url": f"https://github.com/{result['repo']}/blob/{branch}/{source_path}",
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }
    (folder / SOURCE_FILE).write_text(json.dumps(source, indent=2), encoding="utf-8")
    return {"ok": True, **describe_skill_file(path)}


def describe_skill_file(path: Any) -> Dict[str, Any]:
    """Name, description and source of a skill file an agent points at."""
    file_path = Path(str(path)).expanduser()
    row: Dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.is_file(),
        "name": file_path.parent.name
        if file_path.name == "SKILL.md"
        else file_path.stem,
        "description": "",
        "source": None,
    }
    if not row["exists"]:
        return row
    try:
        parsed = parse_skill_md(file_path.read_text(encoding="utf-8"))
        row["name"] = parsed["name"] or row["name"]
        row["description"] = parsed["description"]
    except (OSError, UnicodeDecodeError):
        pass
    source_file = file_path.parent / SOURCE_FILE
    if source_file.is_file():
        try:
            row["source"] = json.loads(source_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return row


__all__ = ["describe_skill_file", "install_skill", "skills_root"]
