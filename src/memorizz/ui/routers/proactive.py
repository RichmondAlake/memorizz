"""Proactive agents in the playground: the suggestion card above the chat box.

The page asks while it is open and idle (the heartbeat a browser can give);
the engine's own admission rules decide whether a model is called. A step the
user accepts is sent through the chat like any message, so it streams where
they can watch it. A step the agent takes on its own (``medium`` and ``high``)
runs here, behind the same lock a chat turn holds.
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request

from ..helpers import _json_object, _load_agent, _save_agent_fields
from ..state import _state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["proactive"])

#: How long a loaded agent is reused between polls of one page.
AGENT_TTL = 120.0
_agents: Dict[str, Tuple[float, Any, Any]] = {}


def _policy_of(record: Any):
    from ...proactive import ProactivePolicy

    value = (
        record.get("proactive")
        if isinstance(record, dict)
        else getattr(record, "proactive", None)
    )
    try:
        return ProactivePolicy.from_value(value)
    except ValueError:
        return ProactivePolicy.off()


def _live_agent(agent_id: str, record: Any) -> Any:
    """The agent as a MemAgent, reused for a couple of minutes: a page polls
    every few seconds and loading an agent is not free."""
    from ...memagent import MemAgent

    provider = _state.get("provider")
    settings = _policy_of(record).to_dict()
    cached = _agents.get(agent_id)
    now = time.monotonic()
    if (
        cached
        and now - cached[0] < AGENT_TTL
        and cached[2] == settings
        and getattr(cached[1], "memory_provider", None) is provider
    ):
        return cached[1]
    agent = MemAgent.load(agent_id, memory_provider=provider)
    _agents[agent_id] = (now, agent, settings)
    return agent


def _forget_agent(agent_id: str) -> None:
    _agents.pop(agent_id, None)


def _latest_thread(agent_id: str, memory_id: Optional[str]) -> Optional[str]:
    if not memory_id:
        return None
    try:
        from ...cli.conversations import latest_thread_id

        return latest_thread_id(
            _state["provider"], memory_id, user_id=None, agent_id=agent_id
        )
    except Exception:
        return None


@contextmanager
def _turn_lock(agent_id: str):
    """The lock a playground chat turn holds, so a step the agent takes on
    its own never interleaves with one."""
    from ...streaming import execution_lock

    lock = execution_lock(("ui", agent_id))
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _parse_seconds(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(
            status_code=400, detail="seconds must be a non-negative integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="seconds must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise HTTPException(
            status_code=400, detail="seconds must be a non-negative integer"
        )
    return parsed


def _state_payload(
    agent_id: str, record: Any, *, with_usage: bool = True
) -> Dict[str, Any]:
    policy = _policy_of(record)
    payload: Dict[str, Any] = {
        "enabled": policy.enabled,
        "level": policy.level if policy.enabled else "off",
        "label": policy.label,
        "predict": bool(policy.enabled and policy.predict),
        "scan": bool(policy.scan_tools),
        "allow_external_actions": bool(policy.allow_external_actions),
    }
    if policy.enabled and with_usage:
        from ...proactive import ProactiveStore, reconcile_store

        store = ProactiveStore(_state.get("provider"), agent_id)
        reconcile_store(store)
        state = store.load()
        today = time.strftime("%Y-%m-%d", time.gmtime())
        same_day = state["state"].get("day") == today
        payload.update(
            evaluations_today=int(state["state"].get("daily_runs") or 0)
            if same_day
            else 0,
            evaluations_allowed=policy.daily_evaluations,
            actions_today=int(state["state"].get("daily_actions") or 0)
            if same_day
            else 0,
            actions_allowed=policy.daily_actions,
            snoozed_until=(
                state["state"].get("snoozed_until")
                if float(state["state"].get("snoozed_until") or 0) > time.time()
                else None
            ),
            usage=dict(state["state"].get("usage") or {}),
            recent=[
                {
                    key: event.get(key)
                    for key in (
                        "id",
                        "title",
                        "message",
                        "status",
                        "decision",
                        "by",
                        "origin",
                        "reason",
                        "result",
                        "error",
                        "proposal_id",
                        "memory_id",
                        "evidence",
                        "created_at",
                    )
                }
                for event in reversed(state["events"][-8:])
            ],
        )
    return payload


@router.get("/api/agents/{agent_id}/proactive")
async def proactive_state(agent_id: str):
    """Whether this agent is proactive, its level and its recent suggestions."""
    record = _load_agent(agent_id)
    return await asyncio.to_thread(_state_payload, agent_id, record)


@router.post("/api/agents/{agent_id}/proactive")
async def proactive_configure(request: Request, agent_id: str):
    """Set the level (``off``, ``low``, ``medium``, ``high``) and the
    experimental prediction and scanning switches; saved on the agent."""
    from ...proactive import ProactivePolicy, ProactiveStore

    record = _load_agent(agent_id)
    body = await _json_object(request)
    current = _policy_of(record)
    level = str(body.get("level") or current.label).strip().lower()
    try:
        if level in ("off", "none", "false"):
            policy = ProactivePolicy.off()
        else:
            settings = current.to_dict() if current.enabled else {}
            settings.update(enabled=True, level="high" if level == "ultra" else level)
            if level == "ultra":
                settings["predict"] = True
            if "predict" in body:
                settings["predict"] = bool(body.get("predict"))
            if "scan" in body:
                settings["scan_tools"] = ("*",) if body.get("scan") else ()
            if "allow_external_actions" in body:
                settings["allow_external_actions"] = bool(
                    body.get("allow_external_actions")
                )
            policy = ProactivePolicy.from_value(settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await asyncio.to_thread(
        _save_agent_fields,
        record,
        proactive=policy.to_dict() if policy.enabled else None,
    )
    if policy.enabled and "level" in body:
        # Changing the level wakes a snoozed agent, matching the playground note.
        def _wake() -> None:
            store = ProactiveStore(_state.get("provider"), agent_id)
            if float(store.load()["state"].get("snoozed_until") or 0) > time.time():
                store.update(lambda data: data["state"].update(snoozed_until=0))

        await asyncio.to_thread(_wake)
    _forget_agent(agent_id)
    return await asyncio.to_thread(_state_payload, agent_id, _load_agent(agent_id))


@router.post("/api/agents/{agent_id}/proactive/suggestions")
async def proactive_suggestions(request: Request, agent_id: str):
    """Ask the agent whether it has a next step for this conversation."""
    record = _load_agent(agent_id)
    if not _policy_of(record).enabled:
        return {"status": "disabled", "suggestions": []}
    body = await _json_object(request, required=False)
    memory_id = str(body.get("memory_id") or "").strip() or None
    trigger = str(body.get("trigger") or "view")
    activity = body.get("activity")
    if activity is not None and not isinstance(activity, (str, dict)):
        raise HTTPException(status_code=400, detail="activity is text or an object")

    def evaluate() -> Dict[str, Any]:
        agent = _live_agent(agent_id, record)
        engine = agent.proactive_engine
        return engine.evaluate(
            activity,
            memory_id=memory_id,
            thread_id=_latest_thread(agent_id, memory_id),
            trigger=trigger,
            force=bool(body.get("force")),
            acting_lock=lambda: _turn_lock(agent_id),
        )

    try:
        result = await asyncio.to_thread(evaluate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Proactive evaluation failed: %s", type(exc).__name__)
        return {"status": "unavailable", "suggestions": []}
    return {
        "status": result["status"],
        "suggestions": [item.to_dict() for item in result["suggestions"]],
    }


@router.post("/api/agents/{agent_id}/proactive/suggestions/{suggestion_id}/{action}")
async def proactive_answer(
    request: Request, agent_id: str, suggestion_id: str, action: str
):
    """Answer a suggestion: ``accept`` (the page then sends the step through
    the chat; ``{"run": true}`` runs it here instead), ``dismiss``, or
    ``snooze``."""
    record = _load_agent(agent_id)
    if action not in ("accept", "dismiss", "snooze"):
        raise HTTPException(status_code=404, detail="Unknown action")
    body = await _json_object(request, required=False)

    seconds = _parse_seconds(body.get("seconds")) if action == "snooze" else None

    def answer() -> Dict[str, Any]:
        engine = _live_agent(agent_id, record).proactive_engine
        if action == "accept":
            run = bool(body.get("run"))
            if run:
                with _turn_lock(agent_id):
                    return engine.accept(suggestion_id, run=True).to_dict()
            return engine.accept(suggestion_id, run=False).to_dict()
        if action == "dismiss":
            return engine.dismiss(suggestion_id).to_dict()
        engine.get(suggestion_id)  # 404 if unknown, matching accept/dismiss
        engine.snooze(seconds)
        return {"id": suggestion_id, "status": "snoozed"}

    try:
        return {"ok": True, "suggestion": await asyncio.to_thread(answer)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Suggestion not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/agents/{agent_id}/proactive/snooze")
async def proactive_snooze(request: Request, agent_id: str):
    """Not now for ``seconds`` (an hour by default); 0 ends a snooze."""
    record = _load_agent(agent_id)
    body = await _json_object(request, required=False)
    seconds = _parse_seconds(body.get("seconds"))

    def snooze() -> None:
        _live_agent(agent_id, record).proactive_engine.snooze(seconds)

    await asyncio.to_thread(snooze)
    return await asyncio.to_thread(_state_payload, agent_id, record)
