# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.

"""Vercel Agent Skills provider.

Searches skills.sh for available skills and fetches SKILL.md instruction files
from GitHub repositories so the agent can use them to complete tasks.
"""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard paths where skills CLI looks for SKILL.md files in a repo
_SKILL_SEARCH_PATHS = [
    "",  # root
    "skills/",
    "skills/.curated/",
    "skills/.experimental/",
    "skills/.system/",
    ".claude/skills/",
    ".agents/skills/",
    ".cursor/skills/",
    ".continue/skills/",
]

_GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
_GITHUB_API_BASE = "https://api.github.com"
_SKILLS_SH_BASE = "https://skills.sh"

MISSING_GITHUB_TOKEN_MESSAGE = (
    "Vercel Agent Skills is enabled but GITHUB_TOKEN is not set. "
    "Skill search calls GitHub's code search API, which requires authentication "
    "and will fail with HTTP 401 without a token. "
    "Set GITHUB_TOKEN in your environment (or pass github_token in the provider "
    "config). Create a token with the 'public_repo' scope at "
    "https://github.com/settings/tokens — see "
    "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens"
)


class VercelSkillsProvider:
    """Provider for discovering and fetching Vercel Agent Skills.

    Supports two operations:
    1. Search skills.sh directory for skills by keyword
    2. Fetch a SKILL.md from a specific GitHub repo (owner/repo format)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config else {}
        self._github_token = self._config.get("github_token", "")
        if not self._github_token:
            logger.warning(MISSING_GITHUB_TOKEN_MESSAGE)

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Search skills.sh directory for skills matching a query.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            Dict with ``ok``, ``skills`` list, and metadata.
        """
        query = (query or "").strip()
        if not query:
            return {"ok": False, "error": "Search query is required."}

        safe_limit = max(1, min(int(limit or 20), 100))

        # skills.sh doesn't have a public API, so we use the npx skills find
        # equivalent by scraping the directory. Fall back to GitHub search.
        results = self._search_github_skills(query, safe_limit)
        return results

    def fetch_skill(
        self,
        repo: str,
        skill_name: Optional[str] = None,
        branch: str = "main",
    ) -> Dict[str, Any]:
        """Fetch a SKILL.md from a GitHub repository.

        Args:
            repo: GitHub repo in ``owner/repo`` format. Also accepts full
                GitHub URLs which will be parsed.
            skill_name: Optional specific skill name within the repo.
                If provided, looks for ``skills/<skill_name>/SKILL.md``.
            branch: Git branch to fetch from (default: ``main``).

        Returns:
            Dict with ``ok``, ``name``, ``description``, ``instructions``,
            and ``metadata``.
        """
        repo = (repo or "").strip()
        if not repo:
            return {"ok": False, "error": "Repository is required (owner/repo format)."}

        # Parse GitHub URLs into owner/repo
        owner_repo = self._parse_repo(repo)
        if not owner_repo:
            return {
                "ok": False,
                "error": f"Invalid repository format: '{repo}'. Use owner/repo or a GitHub URL.",
            }

        # If a specific skill name is given, look in targeted paths first
        if skill_name:
            skill_name = skill_name.strip()
            targeted_paths = [
                f"skills/{skill_name}/SKILL.md",
                f"skills/.curated/{skill_name}/SKILL.md",
                f"skills/.experimental/{skill_name}/SKILL.md",
                f"{skill_name}/SKILL.md",
            ]
            for path in targeted_paths:
                result = self._fetch_raw_file(owner_repo, path, branch)
                if result:
                    parsed = self._parse_skill_md(result, owner_repo, path)
                    return {"ok": True, **parsed}

        # Try standard discovery paths
        for prefix in _SKILL_SEARCH_PATHS:
            path = f"{prefix}SKILL.md"
            result = self._fetch_raw_file(owner_repo, path, branch)
            if result:
                parsed = self._parse_skill_md(result, owner_repo, path)
                return {"ok": True, **parsed}

        # Try listing the skills/ directory via GitHub API for repos with
        # multiple skills
        multi_skills = self._discover_multi_skills(owner_repo, branch)
        if multi_skills:
            return {
                "ok": True,
                "multi_skill_repo": True,
                "repo": owner_repo,
                "available_skills": multi_skills,
                "instructions": (
                    f"This repository contains {len(multi_skills)} skills. "
                    "Use the skill_name parameter to fetch a specific one: "
                    + ", ".join(s["name"] for s in multi_skills[:10])
                ),
            }

        # Also try master branch if main failed
        if branch == "main":
            return self.fetch_skill(repo, skill_name=skill_name, branch="master")

        return {
            "ok": False,
            "error": (
                f"No SKILL.md found in {owner_repo}. "
                "The repository may not contain a Vercel Agent Skill."
            ),
        }

    def _parse_repo(self, value: str) -> Optional[str]:
        """Parse owner/repo from various input formats."""
        value = value.strip().rstrip("/")

        # Already in owner/repo format
        if re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", value):
            return value

        # Full GitHub URL
        match = re.match(
            r"https?://github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)", value
        )
        if match:
            return match.group(1)

        return None

    def _fetch_raw_file(self, owner_repo: str, path: str, branch: str) -> Optional[str]:
        """Fetch a raw file from GitHub."""
        url = f"{_GITHUB_RAW_BASE}/{owner_repo}/{branch}/{path}"
        headers = {
            "Accept": "text/plain",
            "User-Agent": "Memorizz-VercelSkills/1.0",
        }
        if self._github_token:
            headers["Authorization"] = f"token {self._github_token}"

        request = urllib.request.Request(url=url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError:
            return None
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)
            return None

    def _parse_skill_md(self, content: str, repo: str, path: str) -> Dict[str, Any]:
        """Parse a SKILL.md file into structured data."""
        result: Dict[str, Any] = {
            "repo": repo,
            "path": path,
            "name": "",
            "description": "",
            "instructions": content,
            "metadata": {},
        }

        # Extract YAML frontmatter
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
        )
        if frontmatter_match:
            frontmatter_text = frontmatter_match.group(1)
            body = frontmatter_match.group(2).strip()
            result["instructions"] = body

            # Simple YAML parsing for name/description
            for line in frontmatter_text.split("\n"):
                line = line.strip()
                if line.startswith("name:"):
                    result["name"] = line[5:].strip().strip("\"'")
                elif line.startswith("description:"):
                    result["description"] = line[12:].strip().strip("\"'")
                else:
                    key_match = re.match(r"^([a-zA-Z_]+):\s*(.+)$", line)
                    if key_match:
                        result["metadata"][key_match.group(1)] = (
                            key_match.group(2).strip().strip("\"'")
                        )

        # Fall back to repo name if no name in frontmatter
        if not result["name"]:
            result["name"] = repo.split("/")[-1]

        return result

    def _discover_multi_skills(
        self, owner_repo: str, branch: str
    ) -> List[Dict[str, str]]:
        """List skills in a repo's skills/ directory via GitHub API."""
        url = f"{_GITHUB_API_BASE}/repos/{owner_repo}/contents/skills?ref={branch}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Memorizz-VercelSkills/1.0",
        }
        if self._github_token:
            headers["Authorization"] = f"token {self._github_token}"

        request = urllib.request.Request(url=url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:
            return []

        if not isinstance(data, list):
            return []

        skills = []
        for entry in data:
            if isinstance(entry, dict) and entry.get("type") == "dir":
                skills.append(
                    {
                        "name": entry["name"],
                        "path": f"skills/{entry['name']}/SKILL.md",
                    }
                )
        return skills

    def _search_github_skills(self, query: str, limit: int) -> Dict[str, Any]:
        """Search GitHub for Vercel Agent Skills via code search.

        Searches for repositories containing SKILL.md files that match
        the query, focusing on the skills ecosystem.
        """
        # Search for SKILL.md files with the query terms
        search_query = f"{query} filename:SKILL.md"
        encoded_q = urllib.parse.quote(search_query, safe="")
        url = f"{_GITHUB_API_BASE}/search/code" f"?q={encoded_q}&per_page={limit}"
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Memorizz-VercelSkills/1.0",
        }
        if self._github_token:
            headers["Authorization"] = f"token {self._github_token}"

        request = urllib.request.Request(url=url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 401 and not self._github_token:
                hint = MISSING_GITHUB_TOKEN_MESSAGE
            else:
                hint = (
                    "Set GITHUB_TOKEN in environment or vercel skills config "
                    "for better rate limits. See "
                    "https://github.com/settings/tokens"
                )
            return {
                "ok": False,
                "error": f"GitHub search failed (HTTP {exc.code}). {error_body[:200]}",
                "hint": hint,
            }
        except Exception as exc:
            return {"ok": False, "error": f"Search request failed: {exc}"}

        if not isinstance(data, dict):
            return {"ok": False, "error": "Unexpected GitHub response format."}

        items = data.get("items", [])
        skills: List[Dict[str, Any]] = []
        seen_repos: set = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            repo_info = item.get("repository", {})
            full_name = repo_info.get("full_name", "")
            if not full_name or full_name in seen_repos:
                continue
            seen_repos.add(full_name)

            skills.append(
                {
                    "repo": full_name,
                    "name": item.get("name", "SKILL.md"),
                    "path": item.get("path", ""),
                    "description": repo_info.get("description", ""),
                    "html_url": item.get("html_url", ""),
                    "repo_url": f"https://github.com/{full_name}",
                    "stars": repo_info.get("stargazers_count", 0),
                }
            )

        return {
            "ok": True,
            "query": query,
            "skills": skills,
            "count": len(skills),
            "total_count": data.get("total_count", len(skills)),
        }
