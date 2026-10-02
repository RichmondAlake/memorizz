# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Read the name and description of a SKILL.md file.

Agent Skills put them in YAML frontmatter; older ``.skills.md`` files use a
``# Heading`` and a first paragraph instead.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)(.*)$", re.DOTALL)
_HEADING = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)


def parse_skill_md(content: str) -> Dict[str, Any]:
    """``name``, ``description``, ``body`` and other frontmatter ``metadata``."""
    text = str(content or "")
    metadata: Dict[str, str] = {}
    name = description = ""
    body = text
    match = _FRONTMATTER.match(text)
    if match:
        body = match.group(2).strip()
        lines = match.group(1).splitlines()
        for index, line in enumerate(lines):
            key_match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
            if not key_match:
                continue
            key, value = key_match.group(1), key_match.group(2).strip().strip("\"'")
            if value in {">", ">-", "|", "|-", ""}:
                # A block scalar: the value is the indented lines that follow.
                block = []
                for follow in lines[index + 1 :]:
                    if follow and not follow[0].isspace():
                        break
                    block.append(follow.strip())
                value = " ".join(part for part in block if part)
            if key == "name":
                name = value
            elif key == "description":
                description = value
            elif value:
                metadata[key] = value
    if not name:
        heading = _HEADING.search(body)
        name = heading.group(1).strip() if heading else ""
    if not description:
        description = next(
            (
                line.strip()
                for line in body.splitlines()
                if line.strip() and not line.lstrip().startswith(("#", "---"))
            ),
            "",
        )
    return {
        "name": name,
        "description": description[:240],
        "body": body,
        "metadata": metadata,
    }


__all__ = ["parse_skill_md"]
