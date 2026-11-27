"""What the playground shows for Exa: a webset's table and a monitor's runs.

Read live from Exa with the key in the environment. An agent's page can
only reach the websets and monitors that agent made
(:class:`memorizz.internet_access.exa_tools.ExaRecords`).
"""

import asyncio
import csv
import io
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from ...internet_access.exa_tools import (
    ExaRecords,
    keep_rows,
    monitor_view,
    save_table_to_memory,
    table_text,
    webset_table,
)
from ..helpers import _load_agent
from ..state import _state

router = APIRouter(tags=["exa"])


def _exa():
    from ...internet_access.providers.exa import ExaProvider

    try:
        return ExaProvider()
    except ValueError as exc:
        raise HTTPException(400, "EXA_API_KEY is not set where MemoRizz runs") from exc


def _made(agent_id: str, kind: str, item_id: str) -> Dict[str, Any]:
    _load_agent(agent_id)
    record = ExaRecords(agent_id).get(kind, item_id)
    if not record:
        raise HTTPException(404, "This agent did not make that")
    return record


async def _call(work):
    from ...internet_access.providers.exa import ExaError

    try:
        return await asyncio.to_thread(work)
    except ExaError as exc:
        raise HTTPException(502, str(exc)) from exc


def _table(webset_id: str) -> Dict[str, Any]:
    exa = _exa()
    return webset_table(exa.get_webset(webset_id), exa.list_webset_items(webset_id))


@router.get("/api/agents/{agent_id}/exa")
async def exa_made(agent_id: str):
    """The websets and monitors this agent has made, newest first."""
    _load_agent(agent_id)
    records = ExaRecords(agent_id)
    return {"websets": records.all("websets"), "monitors": records.all("monitors")}


@router.get("/api/agents/{agent_id}/exa/websets/{webset_id}")
async def exa_webset(agent_id: str, webset_id: str):
    record = _made(agent_id, "websets", webset_id)
    table = await _call(lambda: _table(webset_id))
    table["in_memory"] = len(record.get("saved_items") or [])
    return table


@router.get("/api/agents/{agent_id}/exa/websets/{webset_id}.csv")
async def exa_webset_csv(agent_id: str, webset_id: str):
    _made(agent_id, "websets", webset_id)
    table = await _call(lambda: _table(webset_id))
    rows = table_text(table, limit=10_000)
    names = ["name", "url"] + [c["label"] for c in table["columns"]]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(names)))
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        out.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{webset_id}.csv"'},
    )


@router.post("/api/agents/{agent_id}/exa/websets/{webset_id}/save")
async def exa_webset_save(agent_id: str, webset_id: str):
    """Keep the rows in the agent's knowledge base, one entry per row."""
    if _state.get("read_only"):
        raise HTTPException(403, "This console is read-only")
    _made(agent_id, "websets", webset_id)
    store = _state.get("provider")

    def work():
        from ...memagent import MemAgent

        agent = MemAgent.load(agent_id, memory_provider=store)
        return save_table_to_memory(
            _table(webset_id),
            ExaRecords(agent_id),
            lambda title, notes, metadata: keep_rows(agent, title, notes, metadata),
            agent_id=agent_id,
        )

    return {"ok": True, **await _call(work)}


@router.get("/api/agents/{agent_id}/exa/monitors/{monitor_id}")
async def exa_monitor(agent_id: str, monitor_id: str):
    _made(agent_id, "monitors", monitor_id)

    def work():
        exa = _exa()
        return monitor_view(
            exa.get_monitor(monitor_id), exa.list_monitor_runs(monitor_id)
        )

    return await _call(work)


@router.post("/api/agents/{agent_id}/exa/monitors/{monitor_id}/run")
async def exa_monitor_run(agent_id: str, monitor_id: str):
    if _state.get("read_only"):
        raise HTTPException(403, "This console is read-only")
    _made(agent_id, "monitors", monitor_id)
    return {"ok": True, **await _call(lambda: _exa().trigger_monitor(monitor_id))}
