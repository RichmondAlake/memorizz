"""Secret-free local configuration for meta-harness adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .._env_io import ensure_home, memorizz_home


def harness_config_path() -> Path:
    return memorizz_home() / "harnesses.json"


def default_harness_config() -> Dict[str, Any]:
    return {
        "version": 1,
        "allowlist": ["memagent", "codex", "claude-code", "openhands"],
        "preference": ["memagent", "codex", "claude-code", "openhands"],
        "adapters": {
            "codex": {"command": "codex", "model": None, "enabled": True},
            "claude-code": {"command": "claude", "model": None, "enabled": True},
            "openhands": {
                "command": "openhands",
                "model": None,
                "enabled": True,
                "external_isolation": False,
            },
        },
        "allowed_workspace_roots": [],
        "context_max_chars": 24_000,
    }


def load_harness_config(path: Path | None = None) -> Dict[str, Any]:
    target = path or harness_config_path()
    defaults = default_harness_config()
    if not target.exists():
        return defaults
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid harness configuration at {target}: {exc}") from exc
    if not isinstance(value, dict) or int(value.get("version") or 0) != 1:
        raise ValueError("Harness configuration must be a version 1 JSON object")
    result = {**defaults, **value}
    result["adapters"] = {**defaults["adapters"], **dict(value.get("adapters") or {})}
    return result


def save_harness_config(value: Dict[str, Any], path: Path | None = None) -> Path:
    ensure_home()
    target = path or harness_config_path()
    payload = dict(value or {})
    payload["version"] = 1
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return target


__all__ = [
    "default_harness_config",
    "harness_config_path",
    "load_harness_config",
    "save_harness_config",
]
