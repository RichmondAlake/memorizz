"""Workspace, environment, redaction, and fingerprint guards for harness runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

_SENSITIVE_KEYS = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|cookie|credential)", re.I
)
_TOKEN_TELEMETRY_KEYS = re.compile(
    r"^(?:[a-z0-9]+_)*(?:input|output|total|prompt|completion|reasoning|"
    r"cached|candidate|context|memory|evidence)_tokens(?:_details)?"
    r"(?:_(?:mean|median|min|max|sum|stdev|sample_stdev|p(?:50|90|95|99)|"
    r"mean_delta|paired_deltas))?$|"
    r"^(?:tokens_(?:used|saved|avoided)|token_estimates?|token_count)$|"
    r"^[a-z0-9_]+_(?:token_count|token_budget)$|^per_million_tokens$",
    re.I,
)
_SENSITIVE_VALUES = re.compile(
    r"(?:(?<![A-Za-z0-9])(?:sk|pk|tvly|e2b|npm|pypi)[-_]"
    r"[A-Za-z0-9._-]{12,}(?![A-Za-z0-9])|"
    r"Bearer\s+[A-Za-z0-9._~+/-]{12,})",
    re.I,
)
_BASE_ENV = {
    # Preserve the caller's home directory so vendor CLIs can use their normal
    # authenticated credential stores. Adapters still disable user project
    # configuration where the CLI supports it, and no value is rewritten.
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}


class HarnessSecurityError(ValueError):
    """Raised before a harness process starts when its envelope is unsafe."""


def _is_safe_token_telemetry(key: str, value: Any) -> bool:
    """Distinguish safe token telemetry from similarly named credentials."""
    normalized_key = key.replace("-", "_").lower()
    if normalized_key == "token_reporting":
        return isinstance(value, bool) or value is None
    if not _TOKEN_TELEMETRY_KEYS.fullmatch(normalized_key):
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, Mapping, list, tuple)) or value is None:
        return True
    return isinstance(value, str) and value.strip().isdigit()


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _SENSITIVE_KEYS.search(str(key))
                and not _is_safe_token_telemetry(str(key), item)
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUES.sub("[REDACTED]", value)
    return value


def redact_paths(value: Any, replacements: Mapping[str, str]) -> Any:
    """Replace known local path prefixes in a serializable report.

    This is deliberately explicit: arbitrary slash-delimited text can be valid
    evidence, while repository, home, and run-directory prefixes are private
    environment details that should not appear in publishable artifacts.
    """
    normalized = sorted(
        (
            (str(Path(source).expanduser()), str(label))
            for source, label in replacements.items()
            if str(source).strip()
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        if isinstance(item, str):
            result = item
            for source, label in normalized:
                result = result.replace(source, label)
            return result
        return item

    return visit(value)


def build_child_environment(
    *,
    allowed_names: Iterable[str] = (),
    overrides: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    names = set(_BASE_ENV)
    names.update(str(item) for item in allowed_names if str(item).strip())
    result = {name: os.environ[name] for name in names if name in os.environ}
    for key, value in dict(overrides or {}).items():
        if _SENSITIVE_KEYS.search(str(key)) and key not in names:
            raise HarnessSecurityError(
                f"Secret environment variable {key!r} was not explicitly allowlisted"
            )
        result[str(key)] = str(value)
    return result


def resolve_workspace(workspace: str, allowed_roots: Iterable[str] = ()) -> Path:
    path = Path(workspace).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise HarnessSecurityError("Harness workspace must be an existing directory")
    roots = [Path(root).expanduser().resolve(strict=True) for root in allowed_roots]
    if roots and not any(path == root or root in path.parents for root in roots):
        raise HarnessSecurityError(
            f"Workspace {path} is outside the configured allowed roots"
        )
    return path


def _run_git(workspace: Path, *args: str, timeout: int = 20) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace), *args],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=build_child_environment(),
        )
        return completed.returncode, completed.stdout, completed.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


def _untracked_content_digest(workspace: Path) -> Optional[str]:
    code, output, _ = _run_git(
        workspace, "ls-files", "--others", "--exclude-standard", "-z", timeout=60
    )
    if code != 0:
        return None
    digest = hashlib.sha256()
    for relative in sorted(item for item in output.split("\x00") if item):
        raw_candidate = workspace / relative
        candidate = raw_candidate.resolve()
        if candidate != workspace and workspace not in candidate.parents:
            raise HarnessSecurityError("Untracked workspace path escaped its root")
        digest.update(relative.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\x00")
        if raw_candidate.is_symlink():
            digest.update(
                os.readlink(raw_candidate).encode("utf-8", errors="surrogatepass")
            )
            continue
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _workspace_tree_digest(
    workspace: Path,
    *,
    max_files: int = 50_000,
    max_bytes: int = 1_073_741_824,
) -> str:
    """Fingerprint a non-Git write workspace or fail before approval."""
    digest = hashlib.sha256()
    files_seen = 0
    bytes_seen = 0
    for root, directories, files in os.walk(workspace, followlinks=False):
        directories[:] = sorted(item for item in directories if item != ".git")
        for name in directories:
            candidate = Path(root) / name
            relative = candidate.relative_to(workspace).as_posix()
            digest.update(b"directory\x00")
            digest.update(relative.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\x00")
            if candidate.is_symlink():
                resolved = candidate.resolve()
                if resolved != workspace and workspace not in resolved.parents:
                    raise HarnessSecurityError(
                        "Write workspace contains a symlink that escapes its root"
                    )
                digest.update(
                    os.readlink(candidate).encode("utf-8", errors="surrogatepass")
                )
        for name in sorted(files):
            files_seen += 1
            if files_seen > max_files:
                raise HarnessSecurityError(
                    "Non-Git workspace is too large to fingerprint safely; initialize "
                    "Git or reduce it below 50,000 files"
                )
            candidate = Path(root) / name
            relative = candidate.relative_to(workspace).as_posix()
            digest.update(relative.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\x00")
            if candidate.is_symlink():
                resolved = candidate.resolve()
                if resolved != workspace and workspace not in resolved.parents:
                    raise HarnessSecurityError(
                        "Write workspace contains a symlink that escapes its root"
                    )
                digest.update(
                    os.readlink(candidate).encode("utf-8", errors="surrogatepass")
                )
                continue
            if not candidate.is_file():
                continue
            size = candidate.stat().st_size
            bytes_seen += size
            if bytes_seen > max_bytes:
                raise HarnessSecurityError(
                    "Non-Git workspace is too large to fingerprint safely; initialize "
                    "Git or reduce it below 1 GiB"
                )
            with candidate.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def workspace_snapshot(
    workspace: Path, *, include_untracked_contents: bool = False
) -> Dict[str, Any]:
    head_code, head, _ = _run_git(workspace, "rev-parse", "HEAD")
    status_code, status, _ = _run_git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all"
    )
    diff_code, diff, _ = _run_git(workspace, "diff", "--binary", "HEAD")
    git_repo = head_code == 0 and status_code == 0
    status_digest = hashlib.sha256(status.encode("utf-8")).hexdigest()
    diff_digest = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    untracked_digest = (
        _untracked_content_digest(workspace) if include_untracked_contents else None
    )
    tree_digest = (
        _workspace_tree_digest(workspace)
        if include_untracked_contents and not git_repo
        else None
    )
    payload = {
        "workspace": str(workspace),
        "git_repository": git_repo,
        "head": head.strip() if head_code == 0 else None,
        "dirty": bool(status.strip()) if status_code == 0 else None,
        "status": status[:100_000],
        "diff": diff[:2_000_000] if diff_code == 0 else "",
        "status_digest": status_digest,
        "diff_digest": diff_digest,
        "untracked_content_digest": untracked_digest,
        "tree_digest": tree_digest,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "workspace": payload["workspace"],
                "git_repository": payload["git_repository"],
                "head": payload["head"],
                "dirty": payload["dirty"],
                "status_digest": status_digest,
                "diff_digest": diff_digest,
                "untracked_content_digest": untracked_digest,
                "tree_digest": tree_digest,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def workspace_diff(workspace: Path) -> str:
    code, diff, error = _run_git(workspace, "diff", "--binary", "HEAD", timeout=60)
    if code != 0:
        return (
            f"[workspace diff unavailable: {error.strip() or 'not a git repository'}]"
        )
    status_code, status, _ = _run_git(
        workspace, "status", "--porcelain=v1", "--untracked-files=all", timeout=60
    )
    evidence = ""
    if status_code == 0 and status.strip():
        evidence = "# git status --porcelain\n" + status
    if diff:
        evidence += "\n# git diff --binary HEAD\n" + diff
    return evidence[:2_000_000]


__all__ = [
    "HarnessSecurityError",
    "build_child_environment",
    "redact",
    "resolve_workspace",
    "workspace_diff",
    "workspace_snapshot",
]
