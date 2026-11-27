# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Exa's Websets and Monitors as agent tools, and the tables they return.

A *webset* is a list Exa builds: it searches for things matching a
description, checks each one against criteria and fills in extra columns.
It takes minutes and costs more than a search, so it is for building a
dataset, not for answering a question. A *monitor* is a search Exa runs
again on a schedule.

Both come back as the same structure everywhere (a tool result, the SDK,
the CLI and the console): :func:`webset_table` and :func:`monitor_view`.
An agent keeps a note of the websets and monitors it made, under
``<MEMORIZZ_HOME>/exa/<agent>.json``, so it can only read, save and stop
its own.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._env_io import memorizz_home
from ..tool_outcomes import ToolOutcome, ToolOutcomeStatus, ToolResult
from ..tooling import governed_tool

DEFAULT_WEBSET_MAX = 25
MAX_WAIT_SECONDS = 90
RUN_WAIT_SECONDS = 60
RUN_POLL_SECONDS = 3
ROWS_FOR_MODEL = 25
WEBHOOK_ENV = "EXA_MONITOR_WEBHOOK_URL"

_LOCK = threading.RLock()

# The fields worth a column for each kind of thing Exa finds.
_ENTITY_FIELDS = {
    "company": (
        ("industry", "Industry"),
        ("location", "Location"),
        ("employees", "Employees"),
        ("foundedYear", "Founded"),
    ),
    "person": (
        ("position", "Position"),
        ("company", "Company"),
        ("location", "Location"),
    ),
    "article": (("author", "Author"), ("publishedAt", "Published")),
    "researchPaper": (("author", "Author"), ("publishedAt", "Published")),
    "custom": (("author", "Author"), ("publishedAt", "Published")),
}
_ENTITY_KEYS = {
    "company": "company",
    "person": "person",
    "article": "article",
    "research_paper": "researchPaper",
    "researchPaper": "researchPaper",
    "custom": "custom",
}


# --- what an agent made ---------------------------------------------------


class ExaRecords:
    """The websets and monitors one agent created, kept on this machine."""

    def __init__(self, agent_id: str, folder: Optional[Path] = None):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(agent_id or "agent"))
        self.path = (folder or memorizz_home() / "exa") / f"{safe}.json"

    def _read(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        for kind in ("websets", "monitors"):
            if not isinstance(data.get(kind), dict):
                data[kind] = {}
        return data

    def _write(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        draft = self.path.with_suffix(".tmp")
        draft.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(draft, self.path)

    def add(self, kind: str, item_id: str, **about: Any) -> Dict[str, Any]:
        with _LOCK:
            data = self._read()
            entry = {**data[kind].get(item_id, {}), **about}
            entry.setdefault("created_at", time.time())
            data[kind][item_id] = entry
            self._write(data)
            return dict(entry, id=item_id)

    def get(self, kind: str, item_id: str) -> Optional[Dict[str, Any]]:
        entry = self._read()[kind].get(str(item_id))
        return dict(entry, id=str(item_id)) if isinstance(entry, dict) else None

    def all(self, kind: str) -> List[Dict[str, Any]]:
        rows = [dict(entry, id=key) for key, entry in self._read()[kind].items()]
        return sorted(rows, key=lambda row: row.get("created_at") or 0, reverse=True)

    def remove(self, kind: str, item_id: str) -> None:
        with _LOCK:
            data = self._read()
            if data[kind].pop(str(item_id), None) is not None:
                self._write(data)


# --- one structure for a webset and a monitor -------------------------------


def _plain(value: Any) -> str:
    """A cell: a short string for whatever Exa returned."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("name") or value.get("title") or "").strip()
    if isinstance(value, (list, tuple)):
        return ", ".join(part for part in (_plain(item) for item in value) if part)
    return " ".join(str(value).split())


def webset_table(webset: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A webset as a table: ``columns`` and ``rows``, with its progress.

    Each row has ``name``, ``url``, ``description`` and ``cells`` (one value
    per column key). A criterion's cell is ``"yes"``, ``"no"`` or
    ``"unclear"``; ``why`` holds Exa's reasoning for criteria and filled-in
    columns, by column key.
    """
    searches = [s for s in webset.get("searches") or [] if isinstance(s, dict)]
    search = searches[-1] if searches else {}
    entity = (search.get("entity") or {}).get("type") or ""
    entity_key = _ENTITY_KEYS.get(entity, "")
    progress = search.get("progress") or {}

    columns: List[Dict[str, str]] = [
        {"key": key, "label": label, "kind": "field"}
        for key, label in _ENTITY_FIELDS.get(entity_key, ())
    ]
    criteria = [
        str(item.get("description") or "").strip()
        for item in search.get("criteria") or []
        if isinstance(item, dict) and item.get("description")
    ]
    for index, text in enumerate(criteria):
        columns.append(
            {"key": f"criterion_{index}", "label": text, "kind": "criterion"}
        )
    enrichments = [e for e in webset.get("enrichments") or [] if isinstance(e, dict)]
    for item in enrichments:
        columns.append(
            {
                "key": f"enrichment_{item.get('id')}",
                "label": str(item.get("title") or item.get("description") or "Column"),
                "kind": "enrichment",
                "format": str(item.get("format") or "text"),
            }
        )

    rows = []
    for item in items:
        properties = item.get("properties") or {}
        detail = next(
            (
                properties[key]
                for key in _ENTITY_FIELDS
                if isinstance(properties.get(key), dict)
            ),
            {},
        )
        cells: Dict[str, str] = {}
        why: Dict[str, str] = {}
        for column in columns:
            if column["kind"] == "field":
                cells[column["key"]] = _plain(detail.get(column["key"]))
        judged = {
            str(entry.get("criterion") or "").strip(): entry
            for entry in item.get("evaluations") or []
            if isinstance(entry, dict)
        }
        for index, text in enumerate(criteria):
            entry = judged.get(text) or {}
            met = entry.get("satisfied")
            cells[f"criterion_{index}"] = (
                (
                    "yes"
                    if met in (True, "yes")
                    else "no"
                    if met in (False, "no")
                    else "unclear"
                )
                if entry
                else ""
            )
            if entry.get("reasoning"):
                why[f"criterion_{index}"] = _plain(entry["reasoning"])[:600]
        for entry in item.get("enrichments") or []:
            if not isinstance(entry, dict):
                continue
            key = f"enrichment_{entry.get('enrichmentId')}"
            cells[key] = (
                _plain(entry.get("result"))
                if entry.get("status") == "completed"
                else ""
            )
            if entry.get("status") == "pending":
                cells[key] = "…"
            if entry.get("reasoning"):
                why[key] = _plain(entry["reasoning"])[:600]
        rows.append(
            {
                "id": str(item.get("id") or ""),
                "name": _plain(detail.get("name") or detail.get("title"))
                or _plain(properties.get("description"))[:80]
                or str(properties.get("url") or ""),
                "url": str(properties.get("url") or ""),
                "description": _plain(properties.get("description"))[:400],
                "picture": str(detail.get("logoUrl") or detail.get("pictureUrl") or ""),
                "cells": cells,
                "why": why,
            }
        )

    return {
        "kind": "exa_webset",
        "id": str(webset.get("id") or ""),
        "title": str(webset.get("title") or search.get("query") or "Webset"),
        "query": str(search.get("query") or ""),
        "entity": entity,
        "status": "done" if webset.get("status") == "idle" else "running",
        "wanted": int(search.get("count") or 0),
        "found": len(rows),
        "analyzed": int(progress.get("analyzed") or 0),
        "percent": float(progress.get("completion") or 0),
        "url": str(webset.get("dashboardUrl") or ""),
        "columns": columns,
        "rows": rows,
    }


def _run_pages(run: Dict[str, Any]) -> List[Dict[str, str]]:
    output = run.get("output") or {}
    pages = []
    for item in output.get("results") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        highlights = item.get("highlights")
        snippet = (
            " ".join(str(h) for h in highlights)
            if isinstance(highlights, list)
            else item.get("summary") or item.get("text") or ""
        )
        pages.append(
            {
                "title": _plain(item.get("title") or item["url"])[:200],
                "url": str(item["url"]),
                "published": str(item.get("publishedDate") or "")[:32],
                "snippet": _plain(snippet).lstrip("# ")[:320],
            }
        )
    return pages


def monitor_view(monitor: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A monitor with its latest runs, newest first, as plain data."""
    search = monitor.get("search") or {}
    trigger = monitor.get("trigger") or {}
    ordered = sorted(
        runs, key=lambda run: str(run.get("createdAt") or ""), reverse=True
    )
    return {
        "kind": "exa_monitor",
        "id": str(monitor.get("id") or ""),
        "name": str(monitor.get("name") or search.get("query") or "Monitor"),
        "query": str(search.get("query") or ""),
        "every": str(trigger.get("period") or ""),
        "status": str(monitor.get("status") or ""),
        "next_run_at": str(monitor.get("nextRunAt") or ""),
        # "temporary" for a webhook.site address made for it, else "yours".
        "reports_to": (
            "temporary"
            if "webhook.site/" in str((monitor.get("webhook") or {}).get("url") or "")
            else "yours"
        ),
        "webhook_url": str((monitor.get("webhook") or {}).get("url") or ""),
        "created_at": str(monitor.get("createdAt") or ""),
        "runs": [
            {
                "id": str(run.get("id") or ""),
                "status": str(run.get("status") or ""),
                "at": str(run.get("completedAt") or run.get("createdAt") or ""),
                "failed_because": str(run.get("failReason") or ""),
                "summary": _plain((run.get("output") or {}).get("content"))[:1200],
                "results": _run_pages(run),
            }
            for run in ordered
        ],
    }


def table_text(
    table: Dict[str, Any], limit: int = ROWS_FOR_MODEL
) -> List[Dict[str, str]]:
    """The rows as flat mappings of column label to value, for a model, a
    notebook or a CSV writer."""
    rows = []
    for row in (table.get("rows") or [])[:limit]:
        flat = {"name": row.get("name", ""), "url": row.get("url", "")}
        for column in table.get("columns") or []:
            value = (row.get("cells") or {}).get(column["key"], "")
            if value and value != "…":
                flat[column["label"]] = value
        rows.append(flat)
    return rows


def row_note(table: Dict[str, Any], row: Dict[str, Any]) -> str:
    """One row as a self-contained note for semantic memory."""
    parts = [f"{row.get('name') or row.get('url')}"]
    if row.get("url"):
        parts[0] += f" ({row['url']})"
    if row.get("description"):
        parts.append(row["description"])
    for column in table.get("columns") or []:
        value = (row.get("cells") or {}).get(column["key"], "")
        if not value or value == "…":
            continue
        if column["kind"] == "criterion":
            if value == "yes":
                parts.append(f"Meets: {column['label']}")
            elif value == "no":
                parts.append(f"Does not meet: {column['label']}")
        else:
            parts.append(f"{column['label']}: {value}")
    parts.append(f"From the Exa webset \"{table.get('title')}\".")
    return ". ".join(part.rstrip(". ") for part in parts if part) + "."


# --- the tools ----------------------------------------------------------------


def _as_list(value: Any) -> List[str]:
    """Models pass a list, or one string with the items on lines."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[\n;]+", value) if part.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _ok(
    value: Dict[str, Any],
    display: Optional[Dict[str, Any]],
    count: Optional[int] = None,
) -> ToolResult:
    return ToolResult(
        value={"ok": True, **value},
        outcome=ToolOutcome(
            status=ToolOutcomeStatus.SUCCESS, provider="exa", result_count=count
        ),
        display=display,
    )


def _failed(message: str, code: str = "provider_error") -> ToolResult:
    return ToolResult(
        value={"ok": False, "error": message, "error_code": code},
        outcome=ToolOutcome(
            status=ToolOutcomeStatus.PROVIDER_ERROR
            if code == "provider_error"
            else ToolOutcomeStatus.ERROR,
            reason_code=code,
            provider="exa",
        ),
    )


def temporary_webhook(provider: Any, folder: Optional[Path] = None) -> str:
    """One temporary webhook.site address for this machine's monitors.

    It is asked for once and reused until a day before it expires, so
    making several monitors does not make several addresses."""
    path = (folder or memorizz_home() / "exa") / "webhook.json"
    with _LOCK:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        if (
            isinstance(saved, dict)
            and saved.get("url")
            and float(saved.get("expires_at") or 0) > time.time() + 24 * 3600
        ):
            return str(saved["url"])
        url = provider.temporary_webhook()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "url": url,
                    "expires_at": time.time()
                    + getattr(provider, "TEMPORARY_WEBHOOK_SECONDS", 7 * 24 * 3600),
                }
            ),
            encoding="utf-8",
        )
        return url


def keep_rows(
    agent: Any, title: str, notes: List[str], metadata: Dict[str, Any]
) -> str:
    """Store ``notes`` as one knowledge-base entry, a chunk per note, and
    attach it to ``agent`` so every conversation of the agent can recall
    them. Returns the entry's ``knowledge_base_id``."""
    from ..long_term.semantic.knowledge_base import KnowledgeBase

    knowledge = KnowledgeBase(agent.memory_provider)
    user_id = getattr(agent, "_current_user_id", None)
    knowledge_base_id = knowledge.ingest_knowledge(
        "\n\n".join(notes),
        f"webset: {title}"[:120],
        # One row, one chunk: a row is recalled whole or not at all.
        chunking_strategy=lambda _text, **_options: list(notes),
        user_id=user_id,
        metadata=metadata,
        embeddings="optional",
    )
    if user_id is None:
        knowledge.attach_to_agent(agent, knowledge_base_id)
    else:
        knowledge.attach_to_agent(agent, knowledge_base_id, user_id=user_id)
    return knowledge_base_id


def save_table_to_memory(
    table: Dict[str, Any],
    records: ExaRecords,
    keep: Any,
    *,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """File a webset's rows in semantic memory, each row once.

    ``keep(title, notes, metadata)`` stores the notes and returns the id of
    what it stored (:func:`keep_rows` with the agent filled in). Rows saved
    from this webset before are skipped, so a webset that is still filling
    can be saved again later for the new rows.
    """
    record = records.get("websets", table["id"]) or {}
    already = set(record.get("saved_items") or [])
    fresh = [
        row
        for row in table.get("rows") or []
        if row.get("id") and row["id"] not in already
    ]
    stored = list(record.get("knowledge_base_ids") or [])
    if fresh:
        stored.append(
            keep(
                str(table.get("title") or "Webset"),
                [row_note(table, row) for row in fresh],
                {
                    "source": "exa_webset",
                    "webset_id": table["id"],
                    "webset_title": table.get("title"),
                    "webset_url": table.get("url"),
                    "agent_id": agent_id,
                },
            )
        )
        already.update(row["id"] for row in fresh)
        records.add(
            "websets",
            table["id"],
            saved_items=sorted(already),
            knowledge_base_ids=stored,
        )
    return {
        "saved": len(fresh),
        "in_memory": len(already),
        "knowledge_base_ids": stored,
    }


def build_exa_tools(agent: Any, provider: Any) -> list:
    """The Websets and Monitors tools for ``agent``, backed by ``provider``."""
    from .providers.exa import ExaError

    def records() -> ExaRecords:
        return ExaRecords(getattr(agent, "agent_id", None) or "agent")

    def mine(kind: str, item_id: str) -> Optional[Dict[str, Any]]:
        return records().get(kind, str(item_id or "").strip())

    def not_mine(kind: str) -> ToolResult:
        return _failed(
            f"That is not a {kind} this agent made. Use the id a create call returned.",
            "not_found",
        )

    def webhook() -> Optional[str]:
        """Where a new monitor reports: your address, or one temporary
        address shared by this machine's monitors until it nears expiry."""
        own = (
            getattr(provider, "monitor_webhook_url", "")
            or os.environ.get(WEBHOOK_ENV, "")
        ).strip()
        if own or not getattr(provider, "temporary_webhooks", True):
            return own or None
        return temporary_webhook(provider)

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=True,
        domains=("internet",),
    )
    def exa_create_webset(
        query: str,
        count: int = 10,
        entity: str = "",
        criteria: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        title: str = "",
    ) -> Any:
        """Build a dataset from the web: a list of companies, people, articles or papers that match a description, each one checked.

        Use it when the user wants a LIST or a TABLE of many things ("20 AI
        labs in Europe with their funding", "people who lead data teams at
        banks"). It takes a few minutes and costs far more than a search, so
        do not use it to answer a question or look up one thing; use
        internet_search for that. It returns at once with a webset_id; read
        the rows with exa_get_webset.

        Parameters:
            query: what to find, in plain words.
            count: how many to find (keep it small; 10 is a good start).
            entity: "company", "person", "article" or "research_paper".
            criteria: up to 5 conditions every item must meet.
            columns: extra facts to look up for each item, one per column,
                such as "Year founded" or "Name of the CEO".
            title: a short name for the dataset.
        """
        limit = int(getattr(provider, "webset_max_count", DEFAULT_WEBSET_MAX))
        wanted = max(1, min(int(count or 10), limit))
        try:
            webset = provider.create_webset(
                query,
                count=wanted,
                entity=entity or None,
                criteria=_as_list(criteria),
                enrichments=_as_list(columns),
                title=title or None,
            )
        except ExaError as exc:
            return _failed(str(exc))
        webset_id = str(webset.get("id") or "")
        records().add(
            "websets", webset_id, title=title or query, query=query, count=wanted
        )
        return _ok(
            {
                "webset_id": webset_id,
                "status": "running",
                "count": wanted,
                "next": "It fills in over a few minutes. Call exa_get_webset "
                "with this webset_id to read the rows. The user can already "
                "see the table filling in.",
            },
            {
                "kind": "exa_webset",
                "id": webset_id,
                "title": title or query,
                "query": query,
                "status": "running",
            },
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=False,
        domains=("internet",),
    )
    def exa_get_webset(webset_id: str, wait_seconds: int = 45) -> Any:
        """Read a webset you made: its rows so far and whether it has finished.

        It waits up to wait_seconds for the webset to finish, then returns
        what is there. If status is "running", tell the user it is still
        filling in rather than calling this again and again.
        """
        if not mine("websets", webset_id):
            return not_mine("webset")
        try:
            webset = provider.wait_for_webset(
                webset_id, seconds=max(0, min(int(wait_seconds or 0), MAX_WAIT_SECONDS))
            )
            table = webset_table(webset, provider.list_webset_items(webset_id))
        except ExaError as exc:
            return _failed(str(exc))
        record = mine("websets", webset_id) or {}
        return _ok(
            {
                "webset_id": table["id"],
                "title": table["title"],
                "status": table["status"],
                "found": table["found"],
                "wanted": table["wanted"],
                "rows": table_text(table),
                "in_memory": len(record.get("saved_items") or []),
            },
            {key: table[key] for key in ("kind", "id", "title", "query", "status")},
            count=table["found"],
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=True,
        domains=("internet", "memory"),
    )
    def exa_save_webset_to_memory(webset_id: str) -> Any:
        """Keep a webset's rows in your knowledge base, one entry per row, so you can recall them in any later conversation (with knowledge_base_lookup) without searching again.

        Use it once the webset has rows the user will want again. Rows already
        saved are skipped, so it is safe to call again after more arrive.
        """
        if not mine("websets", webset_id):
            return not_mine("webset")
        try:
            table = webset_table(
                provider.get_webset(webset_id), provider.list_webset_items(webset_id)
            )
        except ExaError as exc:
            return _failed(str(exc))
        if not table["rows"]:
            return _failed(
                "The webset has no rows yet. Try again in a minute.", "empty"
            )
        saved = save_table_to_memory(
            table,
            records(),
            lambda title, notes, metadata: keep_rows(agent, title, notes, metadata),
            agent_id=getattr(agent, "agent_id", None),
        )
        return _ok(
            {"webset_id": table["id"], **saved},
            {key: table[key] for key in ("kind", "id", "title", "query", "status")},
            count=saved["saved"],
        )

    @governed_tool(
        deterministic=False,
        side_effects=True,
        requires_approval=True,
        approval_reason="A monitor runs on a schedule and is billed for every run.",
        domains=("internet",),
    )
    def exa_create_monitor(
        query: str, every: str = "1d", name: str = "", results_per_run: int = 10
    ) -> Any:
        """Have a search run again on a schedule, to watch for news on a topic ("tell me when a competitor launches something").

        Use it only when the user asks to keep watching something over time.
        For what is true now, use internet_search. Read what a monitor found
        with exa_monitor_results.

        Parameters:
            query: what to watch for, in plain words.
            every: how often, as one number and unit: "1h", "6h", "1d", "7d".
            name: a short name for the monitor.
            results_per_run: pages to keep from each run.
        """
        if not re.fullmatch(r"\d{1,3}[hdw]", str(every or "").strip()):
            return _failed(
                '"every" is one number and a unit, such as "6h" or "1d".',
                "invalid_arguments",
            )
        try:
            monitor = provider.create_monitor(
                query,
                webhook_url=webhook(),
                period=str(every).strip(),
                name=name or None,
                num_results=results_per_run,
                metadata={"made_by": "memorizz"},
            )
        except ExaError as exc:
            return _failed(str(exc))
        monitor_id = str(monitor.get("id") or "")
        records().add(
            "monitors", monitor_id, name=name or query, query=query, every=every
        )
        return _ok(
            {
                "monitor_id": monitor_id,
                "status": monitor.get("status"),
                "every": every,
                "next_run_at": monitor.get("nextRunAt"),
                **(
                    {
                        "note": "It reports to a temporary public webhook.site "
                        "address made for it. The user is shown this and how "
                        "to use their own; mention it in one line and carry "
                        "on. Do not pause or delete the monitor because of it."
                    }
                    if "webhook.site" in str((monitor.get("webhook") or {}).get("url"))
                    else {}
                ),
            },
            {
                "kind": "exa_monitor",
                "id": monitor_id,
                "name": name or query,
                "query": query,
                "every": every,
                "status": str(monitor.get("status") or "active"),
            },
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=False,
        domains=("internet",),
    )
    def exa_list_monitors() -> Any:
        """List the monitors you have set up, with what each watches and how often."""
        made = records().all("monitors")
        return _ok(
            {
                "monitors": [
                    {
                        "monitor_id": row["id"],
                        "name": row.get("name"),
                        "query": row.get("query"),
                        "every": row.get("every"),
                    }
                    for row in made
                ]
            },
            None,
            count=len(made),
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=True,
        domains=("internet",),
    )
    def exa_monitor_results(monitor_id: str, run_now: bool = False) -> Any:
        """Read what a monitor you made has found in its latest runs. Set run_now to start a run now instead of waiting for the schedule; the call waits up to a minute for it. If "running" is true the run has not finished: say so, do not call again in this turn."""
        record = mine("monitors", monitor_id)
        if not record:
            return not_mine("monitor")
        try:
            started = bool(
                run_now and provider.trigger_monitor(monitor_id).get("triggered")
            )
            # A run takes a moment; wait for it so one call returns results.
            deadline = time.monotonic() + (RUN_WAIT_SECONDS if started else 0)
            while True:
                runs = provider.list_monitor_runs(monitor_id)
                working = any(
                    run.get("status") in ("pending", "running") for run in runs
                )
                if (runs and not working) or time.monotonic() >= deadline:
                    break
                time.sleep(RUN_POLL_SECONDS)
            view = monitor_view(provider.get_monitor(monitor_id), runs)
        except ExaError as exc:
            return _failed(str(exc))
        latest = next(
            (run for run in view["runs"] if run["status"] == "completed"), None
        )
        return _ok(
            {
                "monitor_id": view["id"],
                "status": view["status"],
                "every": view["every"],
                "next_run_at": view["next_run_at"],
                "run_started": started,
                "running": any(
                    run["status"] in ("pending", "running") for run in view["runs"]
                ),
                "latest": (
                    {
                        "at": latest["at"],
                        "summary": latest["summary"],
                        "results": latest["results"][:10],
                    }
                    if latest
                    else None
                ),
            },
            {
                key: view[key]
                for key in ("kind", "id", "name", "query", "every", "status")
            },
            count=len(latest["results"]) if latest else 0,
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=True,
        domains=("internet",),
    )
    def exa_stop_monitor(monitor_id: str, delete: bool = False) -> Any:
        """Pause a monitor you made so it stops running on its schedule. Set delete to remove it for good."""
        if not mine("monitors", monitor_id):
            return not_mine("monitor")
        try:
            if delete:
                provider.delete_monitor(monitor_id)
                records().remove("monitors", monitor_id)
                return _ok({"monitor_id": monitor_id, "status": "deleted"}, None)
            monitor = provider.update_monitor(monitor_id, status="paused")
        except ExaError as exc:
            return _failed(str(exc))
        return _ok(
            {"monitor_id": monitor_id, "status": monitor.get("status") or "paused"},
            {
                "kind": "exa_monitor",
                "id": monitor_id,
                "status": str(monitor.get("status") or "paused"),
            },
        )

    @governed_tool(
        deterministic=False,
        requires_approval=False,
        side_effects=True,
        domains=("internet",),
    )
    def exa_resume_monitor(monitor_id: str) -> Any:
        """Set a paused monitor you made running on its schedule again."""
        if not mine("monitors", monitor_id):
            return not_mine("monitor")
        try:
            monitor = provider.update_monitor(monitor_id, status="active")
        except ExaError as exc:
            return _failed(str(exc))
        return _ok(
            {"monitor_id": monitor_id, "status": monitor.get("status") or "active"},
            {
                "kind": "exa_monitor",
                "id": monitor_id,
                "status": str(monitor.get("status") or "active"),
            },
        )

    return [
        exa_create_webset,
        exa_get_webset,
        exa_save_webset_to_memory,
        exa_create_monitor,
        exa_list_monitors,
        exa_monitor_results,
        exa_stop_monitor,
        exa_resume_monitor,
    ]
