# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""``memorizz doctor``: which optional extras this install has.

Each extra in pyproject.toml's ``optional-dependencies`` counts as installed
when every package it needs on this platform can be imported. The memory
backend the environment names is checked against its extra, so a missing
driver shows up here rather than as an ImportError later.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .. import extras as extras_module

# The extra each memory backend needs. FileSystem works without FAISS (search
# falls back to brute force), so its extra is a speed-up, not a requirement.
BACKEND_EXTRAS = {"oracle": "oracle", "mongodb": "mongodb", "filesystem": "filesystem"}
OTHER_CHECKS = (
    "memorizz oracle preflight",
    "memorizz harness doctor",
    "memorizz finetune doctor",
    "memorizz desktop doctor",
)


def doctor_report() -> Dict[str, Any]:
    rows = extras_module.extras_report()
    by_name = {row["extra"]: row for row in rows}
    backend = str(os.environ.get("MEMORIZZ_BACKEND", "") or "").strip().lower()
    needed = BACKEND_EXTRAS.get(backend or "filesystem")
    row = by_name.get(needed) if needed else None
    backend_check = {
        "backend": backend or "filesystem",
        "from_environment": bool(backend),
        "extra": needed,
        "installed": bool(row["installed"]) if row else True,
        "required": backend in {"oracle", "mongodb"},
    }
    if row and not row["installed"]:
        backend_check["install"] = (
            f"{extras_module.install_command([needed])} "
            f"(or {extras_module.uv_command(['ui', needed])})"
        )
    return {"extras": rows, "backend": backend_check, "other_checks": OTHER_CHECKS}


def print_doctor(*, raw_json: bool = False) -> int:
    """Print the report; exit 1 when the configured backend's driver is missing."""
    from . import config as cfg

    cfg.load_layered_env()
    report = doctor_report()
    backend = report["backend"]
    failed = backend["required"] and not backend["installed"]
    if raw_json:
        print(json.dumps(report, indent=2, sort_keys=True, default=list))
        return 1 if failed else 0

    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    console = Console()
    table = Table(title="Optional extras", title_justify="left", show_lines=False)
    table.add_column("Extra", style="bold")
    table.add_column("For")
    table.add_column("Status")
    table.add_column("To install", overflow="fold")
    styles = {
        "installed": "[green]installed[/green]",
        "built-in": "[green]built in[/green]",
        "partial": "[yellow]partly installed[/yellow]",
        "missing": "[dim]not installed[/dim]",
        "unavailable": "[dim]not for this platform[/dim]",
    }
    for row in report["extras"]:
        status = styles.get(row["status"], row["status"])
        if row["status"] == "partial":
            status += f" [dim](missing {', '.join(row['missing'])})[/dim]"
        install = (
            ""
            if row["installed"] or row["status"] == "unavailable"
            else escape(row["install"])
        )
        table.add_row(row["extra"], escape(row["label"]), status, install)
    console.print(table)

    name = backend["backend"]
    source = "MEMORIZZ_BACKEND" if backend["from_environment"] else "default"
    if backend["installed"]:
        console.print(f"\nMemory backend: {name} ({source}). Its driver is installed.")
    elif backend["required"]:
        console.print(
            f"\n[red]Memory backend: {name} ({source}), but the {backend['extra']} "
            f"extra is not installed.[/red] Install it: {escape(backend['install'])}"
        )
    else:
        console.print(
            f"\nMemory backend: {name} ({source}). FAISS is not installed, so "
            f"vector search is slower: {escape(backend.get('install', ''))}"
        )
    console.print("More checks: " + ", ".join(report["other_checks"]))
    return 1 if failed else 0


__all__ = ["doctor_report", "print_doctor"]
