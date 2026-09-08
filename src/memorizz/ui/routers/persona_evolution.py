"""Host-owned account personas; never mutate a shared agent configuration."""
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from ..security import audit_trace_view, ui_read_only
from ..state import templates
from ..trace_access import current_principal

router = APIRouter(tags=["persona-evolution"])


def _admin():
    if current_principal.get().restricted:
        raise HTTPException(
            403, "Persona evolution requires an unrestricted operator account"
        )


def _account(user_id):
    if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 128:
        raise HTTPException(400, "A valid account ID is required")
    return user_id


def _audit(request, action, user_id):
    if not audit_trace_view(
        request, action, content_mode="persona_profile", target_user_id=user_id
    ):
        raise HTTPException(503, "Audited persona access is unavailable")


@router.get("/persona-evolution", response_class=HTMLResponse)
async def page(request: Request, user_id: str = ""):
    _admin()
    adapter = request.app.state.persona_evolution
    state, error = None, None
    if user_id and adapter:
        _account(user_id)
        _audit(request, "persona_profile_viewed", user_id)
        try:
            state = await run_in_threadpool(adapter.view, user_id)
        except Exception:
            error = "The account's learning profile could not be loaded. No changes were made."
    response = templates.TemplateResponse(
        "persona_evolution.html",
        {
            "request": request,
            "active_page": "persona-evolution",
            "profile": state,
            "user_id": user_id,
            "connected": adapter is not None,
            "error": error,
            "writable": bool(adapter and request.app.state.persona_evolution_actions),
            "trace_read_only": ui_read_only(),
        },
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@router.post("/persona-evolution/action")
async def action(request: Request):
    _admin()
    adapter = request.app.state.persona_evolution
    if not adapter or not request.app.state.persona_evolution_actions:
        raise HTTPException(403, "Persona changes are not enabled for this UI")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 4096:
            raise HTTPException(400, "Invalid persona action")
        body.extend(chunk)
    try:
        data = json.loads(body)
    except ValueError:
        raise HTTPException(400, "Invalid persona action") from None
    if not isinstance(data, dict) or set(data) - {
        "user_id",
        "action",
        "revision",
        "proposal_id",
        "paused",
        "daily_enabled",
    }:
        raise HTTPException(400, "Invalid persona action")
    if not isinstance(data.get("action"), str):
        raise HTTPException(400, "Invalid persona action")
    if data["action"] in {"approve", "dismiss", "pause", "schedule", "undo"}:
        if type(data.get("revision")) is not int or data["revision"] < 0:
            raise HTTPException(400, "A non-negative integer revision is required")
    if data["action"] in {"approve", "dismiss"}:
        proposal = data.get("proposal_id")
        if not isinstance(proposal, str) or not proposal.strip() or len(proposal) > 240:
            raise HTTPException(400, "A valid proposal ID is required")
    for operation, flag in (("pause", "paused"), ("schedule", "daily_enabled")):
        if data["action"] == operation and type(data.get(flag)) is not bool:
            raise HTTPException(400, "A boolean setting is required")
    user_id = _account(data.get("user_id"))
    actor_id = "operator:" + current_principal.get().principal_id
    if data.get("action") == "reflect":
        operation, kwargs = adapter.reflect, {"actor_id": actor_id}
    elif data.get("action") in {"approve", "dismiss"}:
        operation, kwargs = adapter.decide, {
            "proposal_id": data.get("proposal_id"),
            "revision": data.get("revision"),
            "accept": data["action"] == "approve",
            "actor_id": actor_id,
        }
    elif data.get("action") == "pause":
        operation, kwargs = adapter.pause, {
            "revision": data.get("revision"),
            "paused": data.get("paused"),
        }
    elif data.get("action") == "schedule":
        if not callable(getattr(adapter, "schedule", None)):
            raise HTTPException(400, "Daily review is not supported by this host")
        operation, kwargs = adapter.schedule, {
            "revision": data.get("revision"),
            "daily_enabled": data.get("daily_enabled"),
        }
    elif data.get("action") == "undo":
        operation, kwargs = adapter.undo, {
            "revision": data.get("revision"),
            "actor_id": actor_id,
        }
    else:
        raise HTTPException(400, "Unknown persona action")
    _audit(request, "persona_" + data["action"] + "_requested", user_id)
    try:
        result = await run_in_threadpool(operation, user_id, **kwargs)
    except ValueError:
        raise HTTPException(
            409,
            "This action could not be applied. Reload the profile; it may have changed or be paused.",
        ) from None
    except Exception:
        raise HTTPException(
            503,
            "Persona service unavailable. Reload to check the saved state before retrying.",
        ) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
