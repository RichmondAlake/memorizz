# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Self-awareness manager for host codebase inspection and guarded operations."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class SelfAwarenessManager:
    """Provide bounded host file and command access for self-aware agents."""

    POLICY_VERSION = "v1"

    READ_COMMANDS = {
        "pwd",
        "ls",
        "find",
        "rg",
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "git",
    }
    WRITE_COMMANDS = {"mkdir", "touch", "cp", "mv", "rm", "rmdir"}
    DELETE_COMMANDS = {"rm", "rmdir"}
    COMMANDS_WITH_PATH_ARGS = {
        "ls",
        "find",
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "mkdir",
        "touch",
        "cp",
        "mv",
        "rm",
        "rmdir",
    }
    FORBIDDEN_SHELL_TOKENS = (";", "&&", "||", "|", ">", "<", "`", "$(")

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, cwd: Optional[str] = None
    ):
        self.cwd = Path(cwd or os.getcwd()).expanduser().resolve()
        self._config = self._normalize_config(config or {})

    # --- Configuration ---

    def get_config(self) -> Dict[str, Any]:
        """Return current normalized configuration."""
        return dict(self._config)

    def configure(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply and return a normalized self-awareness configuration."""
        self._config = self._normalize_config(config or {})
        return self.get_config()

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        roots = self._normalize_root_paths(config.get("root_paths"))
        allow_writes = self._coerce_bool(config.get("allow_writes"), default=False)
        allow_deletes = self._coerce_bool(config.get("allow_deletes"), default=False)
        if not allow_writes:
            allow_deletes = False

        return {
            "root_paths": roots,
            "allow_writes": allow_writes,
            "allow_deletes": allow_deletes,
            "policy_version": str(config.get("policy_version") or self.POLICY_VERSION),
            "timeout_seconds": self._coerce_int(
                config.get("timeout_seconds"), default=30, min_value=1, max_value=300
            ),
            "max_output_chars": self._coerce_int(
                config.get("max_output_chars"),
                default=50000,
                min_value=1024,
                max_value=1000000,
            ),
            "max_file_read_bytes": self._coerce_int(
                config.get("max_file_read_bytes"),
                default=250000,
                min_value=1024,
                max_value=5000000,
            ),
            "max_file_write_bytes": self._coerce_int(
                config.get("max_file_write_bytes"),
                default=250000,
                min_value=1,
                max_value=5000000,
            ),
        }

    def _normalize_root_paths(self, raw_paths: Any) -> List[str]:
        values: List[Any]
        if isinstance(raw_paths, str):
            values = [raw_paths]
        elif isinstance(raw_paths, list):
            values = raw_paths
        else:
            values = []

        normalized: List[str] = []
        seen = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = self.cwd / candidate
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)

        if not normalized:
            normalized = [str(self.cwd)]
        return normalized

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "on", "yes"}

    @staticmethod
    def _coerce_int(
        value: Any,
        default: int,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        if min_value is not None:
            result = max(min_value, result)
        if max_value is not None:
            result = min(max_value, result)
        return result

    # --- Paths ---

    def list_roots(self) -> Dict[str, Any]:
        """Return configured root paths and active policy flags."""
        return {
            "root_paths": list(self._config.get("root_paths", [])),
            "allow_writes": bool(self._config.get("allow_writes", False)),
            "allow_deletes": bool(self._config.get("allow_deletes", False)),
            "policy_version": self._config.get("policy_version", self.POLICY_VERSION),
        }

    def _is_within_roots(self, candidate: Path) -> bool:
        for root_value in self._config.get("root_paths", []):
            root = Path(root_value).resolve()
            try:
                candidate.relative_to(root)
                return True
            except Exception:
                continue
        return False

    def _resolve_path(self, raw_path: str, allow_missing: bool = False) -> Path:
        text = str(raw_path or ".").strip() or "."
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        resolved = path.resolve()
        if not self._is_within_roots(resolved):
            raise ValueError(f"Path is outside allowed roots: {text}")
        if not allow_missing and not resolved.exists():
            raise FileNotFoundError(f"Path not found: {text}")
        return resolved

    def _display_path(self, resolved: Path) -> str:
        try:
            return str(resolved.relative_to(self.cwd))
        except Exception:
            return str(resolved)

    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        include_hidden: bool = False,
        max_entries: int = 500,
    ) -> Dict[str, Any]:
        """List files under a constrained path."""
        target = self._resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        if not target.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        cap = max(1, min(int(max_entries or 500), 5000))
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries: List[Dict[str, Any]] = []
        truncated = False

        for entry in iterator:
            if not include_hidden and any(part.startswith(".") for part in entry.parts):
                continue
            if len(entries) >= cap:
                truncated = True
                break
            entry_type = "directory" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else None
            entries.append(
                {
                    "path": self._display_path(entry),
                    "type": entry_type,
                    "size": size,
                }
            )

        return {
            "path": self._display_path(target),
            "entries": entries,
            "entry_count": len(entries),
            "truncated": truncated,
        }

    def read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 400,
        max_chars: int = 30000,
    ) -> Dict[str, Any]:
        """Read a file with line and size bounds."""
        target = self._resolve_path(path)
        if not target.is_file():
            raise ValueError(f"Path is not a file: {path}")

        max_bytes = int(self._config["max_file_read_bytes"])
        with target.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
        bytes_truncated = len(payload) > max_bytes
        if bytes_truncated:
            payload = payload[:max_bytes]
        text = payload.decode("utf-8", errors="replace")

        lines = text.splitlines()
        start = max(1, int(start_line or 1))
        stop = max(start, int(end_line or start))
        selected = lines[start - 1 : stop]
        content = "\n".join(selected)

        limit_chars = max(256, min(int(max_chars or 30000), 200000))
        content_truncated = len(content) > limit_chars
        if content_truncated:
            content = content[:limit_chars]

        return {
            "path": self._display_path(target),
            "start_line": start,
            "end_line": stop,
            "content": content,
            "bytes_truncated": bytes_truncated,
            "content_truncated": content_truncated,
        }

    def search_files(
        self,
        pattern: str,
        path: str = ".",
        glob: str = "",
        max_results: int = 200,
    ) -> Dict[str, Any]:
        """Search for a pattern in files under a constrained root."""
        if not str(pattern or "").strip():
            raise ValueError("Pattern is required.")

        target = self._resolve_path(path)
        cap = max(1, min(int(max_results or 200), 5000))

        if shutil.which("rg"):
            cmd = [
                "rg",
                "-n",
                "--no-heading",
                "--color",
                "never",
                str(pattern),
                str(target),
            ]
            if str(glob or "").strip():
                cmd.extend(["-g", str(glob).strip()])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(self._config["timeout_seconds"]),
                shell=False,
            )
            lines = [
                line for line in (result.stdout or "").splitlines() if line.strip()
            ]
            matches = lines[:cap]
            return {
                "path": self._display_path(target),
                "pattern": pattern,
                "matches": matches,
                "match_count": len(matches),
                "truncated": len(lines) > cap,
                "exit_code": result.returncode,
            }

        # Fallback: simple text scan
        matches: List[str] = []
        for file_path in target.rglob("*"):
            if not file_path.is_file():
                continue
            if len(matches) >= cap:
                break
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for idx, line in enumerate(handle, start=1):
                        if pattern in line:
                            matches.append(
                                f"{self._display_path(file_path)}:{idx}:{line.rstrip()}"
                            )
                            if len(matches) >= cap:
                                break
            except Exception:
                continue

        return {
            "path": self._display_path(target),
            "pattern": pattern,
            "matches": matches,
            "match_count": len(matches),
            "truncated": False,
            "exit_code": 0,
        }

    def write_file(
        self, path: str, content: str, mode: str = "overwrite"
    ) -> Dict[str, Any]:
        """Write a file if write policy is enabled."""
        if not self._config.get("allow_writes", False):
            raise PermissionError("Self-aware writes are disabled.")

        mode_value = str(mode or "overwrite").strip().lower()
        if mode_value not in {"overwrite", "append", "create"}:
            raise ValueError("mode must be one of: overwrite, append, create")

        target = self._resolve_path(path, allow_missing=True)
        payload = str(content or "")
        payload_size = len(payload.encode("utf-8"))
        if payload_size > int(self._config["max_file_write_bytes"]):
            raise ValueError("content exceeds max_file_write_bytes limit.")

        target.parent.mkdir(parents=True, exist_ok=True)
        file_mode = (
            "w" if mode_value == "overwrite" else "a" if mode_value == "append" else "x"
        )
        with target.open(file_mode, encoding="utf-8") as handle:
            handle.write(payload)

        return {
            "path": self._display_path(target),
            "mode": mode_value,
            "bytes_written": payload_size,
        }

    def delete_path(
        self, path: str, recursive: bool = False, force: bool = False
    ) -> Dict[str, Any]:
        """Delete a file or directory if delete policy is enabled."""
        if not self._config.get("allow_writes", False):
            raise PermissionError("Self-aware writes are disabled.")
        if not self._config.get("allow_deletes", False):
            raise PermissionError("Self-aware deletes are disabled.")

        raw = str(path or "").strip()
        if not raw:
            raise ValueError("path is required.")
        if any(token in raw for token in ("*", "?", "[", "]")):
            raise ValueError("wildcard/glob deletes are not allowed.")

        target = self._resolve_path(raw, allow_missing=True)
        for root_value in self._config.get("root_paths", []):
            if target == Path(root_value).resolve():
                raise ValueError("Cannot delete a configured root path.")

        if not target.exists():
            if force:
                return {
                    "path": self._display_path(target),
                    "deleted": False,
                    "reason": "not_found",
                }
            raise FileNotFoundError(f"Path not found: {raw}")

        if target.is_dir():
            if recursive and not force:
                raise ValueError("Recursive deletes require force=true.")
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()

        return {"path": self._display_path(target), "deleted": True}

    # --- Command execution ---

    def run_command(self, command: str, cwd: str = ".") -> Dict[str, Any]:
        """Execute a guarded command under configured policy."""
        command_text = str(command or "").strip()
        if not command_text:
            raise ValueError("command is required.")
        self._validate_command_string(command_text)

        args = shlex.split(command_text, posix=True)
        if not args:
            raise ValueError("command is required.")

        cmd_name = Path(args[0]).name
        self._validate_command_name(cmd_name)

        working_dir = self._resolve_path(cwd)
        self._validate_command_args(cmd_name, args[1:], working_dir)

        timeout = int(self._config["timeout_seconds"])
        max_output_chars = int(self._config["max_output_chars"])
        result = subprocess.run(
            args,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        truncated = False
        if len(stdout) > max_output_chars:
            stdout = stdout[:max_output_chars]
            truncated = True
        if len(stderr) > max_output_chars:
            stderr = stderr[:max_output_chars]
            truncated = True

        return {
            "command": command_text,
            "args": args,
            "cwd": self._display_path(working_dir),
            "exit_code": int(result.returncode),
            "stdout": stdout,
            "stderr": stderr,
            "output_truncated": truncated,
        }

    def _validate_command_string(self, command: str) -> None:
        for token in self.FORBIDDEN_SHELL_TOKENS:
            if token in command:
                raise ValueError(f"Forbidden shell token in command: {token}")

    def _validate_command_name(self, cmd_name: str) -> None:
        allowed = set(self.READ_COMMANDS)
        if self._config.get("allow_writes", False):
            allowed.update(self.WRITE_COMMANDS)
        if cmd_name not in allowed:
            raise PermissionError(f"Command '{cmd_name}' is not allowed.")
        if cmd_name in self.DELETE_COMMANDS and not self._config.get(
            "allow_deletes", False
        ):
            raise PermissionError(f"Delete command '{cmd_name}' is disabled.")

    def _validate_command_args(self, cmd_name: str, args: List[str], cwd: Path) -> None:
        if cmd_name not in self.COMMANDS_WITH_PATH_ARGS:
            return

        for arg in args:
            text = str(arg or "").strip()
            if not text:
                continue
            if text.startswith("-"):
                continue
            if cmd_name in self.DELETE_COMMANDS and any(
                c in text for c in ("*", "?", "[", "]")
            ):
                raise ValueError("Wildcard/glob delete arguments are not allowed.")

            candidate = Path(text).expanduser()
            if not candidate.is_absolute():
                candidate = cwd / candidate
            resolved = candidate.resolve()
            if not self._is_within_roots(resolved):
                raise ValueError(f"Command argument path escapes allowed roots: {text}")
            if cmd_name in self.DELETE_COMMANDS:
                for root_value in self._config.get("root_paths", []):
                    if resolved == Path(root_value).resolve():
                        raise ValueError("Cannot delete a configured root path.")
