"""The timers the playground shows above the chat box.

The same timers the CLI shows: see :mod:`memorizz.timers`. They belong to the
person at this machine, so accounts limited to a scope don't see them.
"""

from fastapi import APIRouter, HTTPException

from ... import timers
from ..trace_access import current_principal

router = APIRouter(tags=["timers"])


def _unrestricted() -> None:
    if current_principal.get().restricted:
        raise HTTPException(403, "Timers require an unrestricted account")


@router.get("/api/timers")
async def list_timers():
    _unrestricted()
    return {"timers": timers.listed()}


@router.post("/api/timers/{timer_id}/cancel")
async def cancel_timer(timer_id: str):
    _unrestricted()
    if timers.cancel(timer_id) is None:
        raise HTTPException(404, "That timer has already gone")
    return {"ok": True}


@router.post("/api/timers/{timer_id}/dismiss")
async def dismiss_timer(timer_id: str):
    _unrestricted()
    if not timers.dismiss(timer_id):
        raise HTTPException(404, "That timer has already gone")
    return {"ok": True}
