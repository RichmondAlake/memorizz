"""Official-source synchronization and dataset readiness checks.

Synchronization only checks out a pinned, allowlisted official repository. It
never executes downloaded code. Some benchmarks distribute data separately;
the resulting report makes that distinction explicit and provides the next
operator action.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .catalog import get_benchmark_spec
from .protocols import get_protocol_manifest


def default_dataset_root() -> Path:
    configured = os.getenv("MEMORIZZ_EVAL_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    home = Path(os.getenv("MEMORIZZ_HOME", Path.home() / ".memorizz"))
    return (home / "eval" / "datasets").expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _manifest_digest(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _git_value(root: Path, *args: str) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _normalized_repository_url(value: str | None) -> str | None:
    """Normalize harmless Git remote spelling differences for pin checks."""

    if not value:
        return None
    normalized = str(value).strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.split(":", 1)[1]
    return normalized.lower()


def _artifact_groups(
    benchmark_id: str, root: Path, variant: str
) -> Sequence[tuple[str, List[Path]]]:
    """Return alternative path groups; one existing path satisfies each group."""

    if benchmark_id == "agentmembench":
        if variant == "locomo":
            return (
                (
                    "LoCoMo conversations",
                    [root / "locomo10.json", root / "data" / "locomo10.json"],
                ),
            )
        return (
            (
                f"{variant} normalized export",
                [
                    root / f"{variant}.json",
                    root / f"{variant}.jsonl",
                    root / "data" / f"{variant}.json",
                    root / "data" / f"{variant}.jsonl",
                ],
            ),
        )
    if benchmark_id == "longmemeval-v2":
        tier = variant.split("-", 1)[0]
        return (
            ("questions", [root / "questions.jsonl"]),
            ("trajectories", [root / "trajectories.jsonl"]),
            ("haystack", [root / "haystacks" / f"lme_v2_{tier}.json"]),
        )
    if benchmark_id == "locomo-plus":
        groups: List[tuple[str, List[Path]]] = []
        if variant in {"original", "all", "cognitive"}:
            groups.append(
                (
                    "LoCoMo conversations",
                    [root / "locomo10.json", root / "data" / "locomo10.json"],
                )
            )
        if variant in {"cognitive", "all"}:
            groups.append(
                (
                    "LoCoMo-Plus cues",
                    [root / "locomo_plus.json", root / "data" / "locomo_plus.json"],
                )
            )
        return tuple(groups)
    if benchmark_id == "beam":
        scale_aliases = {
            "128k": ("128K", "100K", "128k"),
            "500k": ("500K", "500k"),
            "1m": ("1M", "1m"),
            "10m": ("10M", "10m"),
        }
        return (
            (
                f"BEAM {variant} chats",
                [
                    root / "chats" / value
                    for value in scale_aliases.get(variant, (variant,))
                ],
            ),
        )
    if benchmark_id == "memoryagentbench":
        if root.is_file():
            return (("MemoryAgentBench export", [root]),)
        candidates = sorted([*root.glob("*.json"), *root.glob("*.jsonl")])
        return (("MemoryAgentBench JSON/JSONL export", candidates),)
    return ()


def verify_dataset(
    benchmark_id: str,
    *,
    data_path: str | Path | None = None,
    source_path: str | Path | None = None,
    variant: str | None = None,
    deep_checksum: bool = False,
) -> Dict[str, Any]:
    """Verify source revision, expected assets, licenses, and fingerprints."""

    spec = get_benchmark_spec(benchmark_id)
    manifest = get_protocol_manifest(spec.benchmark_id)
    selected_variant = str(variant or spec.default_variant)
    configured_data = data_path or os.getenv(spec.dataset_env) or ""
    resolved_data = (
        Path(configured_data).expanduser().resolve() if configured_data else None
    )
    if source_path:
        resolved_source = Path(source_path).expanduser().resolve()
    elif resolved_data and (resolved_data / ".git").exists():
        # BEAM and some smaller benchmark distributions keep the data inside
        # the pinned source checkout. Recognize that layout automatically.
        resolved_source = resolved_data
    else:
        resolved_source = default_dataset_root() / spec.benchmark_id / "source"

    revision = _git_value(resolved_source, "rev-parse", "HEAD")
    remote = _git_value(resolved_source, "remote", "get-url", "origin")
    source_ready = bool(
        manifest.repository_url
        and revision
        and revision == manifest.synchronization_revision
        and _normalized_repository_url(remote)
        == _normalized_repository_url(manifest.repository_url)
    )
    artifact_rows: List[Dict[str, Any]] = []
    if resolved_data and resolved_data.exists():
        for label, candidates in _artifact_groups(
            spec.benchmark_id, resolved_data, selected_variant
        ):
            selected = next(
                (candidate for candidate in candidates if candidate.exists()), None
            )
            row: Dict[str, Any] = {
                "name": label,
                "ready": selected is not None,
                "path": str(selected) if selected else None,
                "alternatives": [str(candidate) for candidate in candidates],
            }
            if selected and selected.is_file():
                row["bytes"] = selected.stat().st_size
                row["sha256"] = _sha256(selected)
            elif selected and selected.is_dir():
                files = [item for item in selected.rglob("*") if item.is_file()]
                row["file_count"] = len(files)
                row["bytes"] = sum(item.stat().st_size for item in files)
                row["manifest_sha256"] = _manifest_digest(files, selected)
                if spec.benchmark_id == "beam":
                    expected = {"128k": 20, "500k": 35, "1m": 35, "10m": 10}.get(
                        selected_variant
                    )
                    available = sum(
                        1
                        for child in selected.iterdir()
                        if child.is_dir()
                        and (child / "chat.json").exists()
                        and (
                            child / "probing_questions" / "probing_questions.json"
                        ).exists()
                    )
                    row["coverage"] = {
                        "available_conversations": available,
                        "expected_conversations": expected,
                        "complete": bool(
                            expected is not None and available == expected
                        ),
                    }
                if deep_checksum:
                    row["content_sha256"] = {
                        str(item.relative_to(selected)): _sha256(item) for item in files
                    }
            artifact_rows.append(row)

    dataset_ready = bool(artifact_rows and all(row["ready"] for row in artifact_rows))
    coverage_states = [
        bool(row["coverage"]["complete"])
        for row in artifact_rows
        if isinstance(row.get("coverage"), dict)
    ]
    coverage_complete = all(coverage_states) if coverage_states else None
    fingerprint_payload = {
        "benchmark": spec.benchmark_id,
        "variant": selected_variant,
        "artifacts": [
            {
                "name": row["name"],
                "sha256": row.get("sha256"),
                "manifest_sha256": row.get("manifest_sha256"),
                "content_sha256": row.get("content_sha256"),
            }
            for row in artifact_rows
        ],
    }
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
    )
    license_files = []
    if resolved_source.exists():
        license_files = [
            str(path)
            for name in ("LICENSE", "LICENSE.md", "COPYING", "README.md")
            if (path := resolved_source / name).exists()
        ]

    next_actions: List[str] = []
    if manifest.source_sync_supported and not source_ready:
        next_actions.append(f"memorizz eval dataset sync {spec.benchmark_id}")
    if not dataset_ready:
        next_actions.append(
            f"Prepare the official data, then set {spec.dataset_env} or pass --data-path."
        )
    elif coverage_complete is False:
        next_actions.append(
            "The available data is sufficient for a diagnostic subset, but the "
            "selected benchmark variant is incomplete. Download the remaining "
            "official conversations before a full-profile run."
        )
    return {
        "benchmark_id": spec.benchmark_id,
        "variant": selected_variant,
        "ready": dataset_ready,
        "ready_for_diagnostic": dataset_ready,
        "coverage_complete": coverage_complete,
        "data_path": str(resolved_data) if resolved_data else None,
        "dataset_fingerprint": fingerprint if artifact_rows else None,
        "artifacts": artifact_rows,
        "source": {
            "sync_supported": manifest.source_sync_supported,
            "path": str(resolved_source),
            "repository_url": manifest.repository_url,
            "expected_revision": manifest.synchronization_revision,
            "revision": revision,
            "remote": remote,
            "ready": source_ready,
            "license_files": license_files,
        },
        "license_note": spec.license_note,
        "deep_checksum": bool(deep_checksum),
        "next_actions": next_actions,
    }


def _run_git(command: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=600,
    )


def sync_dataset_source(
    benchmark_id: str,
    *,
    destination: str | Path | None = None,
) -> Dict[str, Any]:
    """Clone or resume a pinned official checkout without executing it."""

    spec = get_benchmark_spec(benchmark_id)
    manifest = get_protocol_manifest(spec.benchmark_id)
    if not (
        manifest.source_sync_supported
        and manifest.repository_url
        and manifest.synchronization_revision
    ):
        raise ValueError(
            f"{spec.name} has no allowlisted source synchronization target; "
            "follow the paper's dataset instructions."
        )
    target = (
        Path(destination).expanduser().resolve()
        if destination
        else default_dataset_root() / spec.benchmark_id / "source"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not (target / ".git").exists():
        if any(target.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-git directory: {target}")
        target.rmdir()
    if not target.exists():
        _run_git(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                manifest.repository_url,
                str(target),
            ]
        )
    else:
        dirty = _git_value(target, "status", "--porcelain")
        if dirty:
            raise RuntimeError(
                f"Refusing to change a dirty official checkout: {target}"
            )
        remote = _git_value(target, "remote", "get-url", "origin")
        if remote != manifest.repository_url:
            raise RuntimeError(
                f"Checkout origin {remote!r} does not match {manifest.repository_url!r}"
            )
    _run_git(
        ["git", "fetch", "--depth", "1", "origin", manifest.synchronization_revision],
        cwd=target,
    )
    _run_git(
        ["git", "checkout", "--detach", manifest.synchronization_revision], cwd=target
    )
    report = verify_dataset(
        spec.benchmark_id,
        source_path=target,
        variant=spec.default_variant,
    )
    report["executed_downloaded_code"] = False
    return report


__all__ = [
    "default_dataset_root",
    "sync_dataset_source",
    "verify_dataset",
]
