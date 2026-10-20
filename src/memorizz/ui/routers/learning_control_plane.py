"""Small operator surface for learning events, compilation, and forgetting."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...learning import LearningControlPlane
from ...learning.compiler import MemoryCompiler
from ..control_plane_monitor import build_control_plane_view
from ..dashboard import template_helpers
from ..helpers import (
    _agent_field,
    _build_agent_nav_items,
    _extract_agent_identifier,
    _extract_agent_persona_name,
    _list_agents,
)
from ..security import ui_read_only
from ..state import _state, templates

logger = logging.getLogger(__name__)
router = APIRouter(tags=["learning-control-plane"])
_last_actions: Dict[str, Dict[str, Any]] = {}


def _plane(agent_id: str) -> LearningControlPlane:
    return LearningControlPlane(
        _state["provider"],
        agent_id=agent_id,
        config={"enabled": True, "compile_async": False},
    )


def _retention_overrides(agent_id: str) -> Optional[Dict[str, Any]]:
    """The agent's per-agent ``retrieval_policy["retention"]`` mapping, if any."""
    try:
        agent = _state["provider"].retrieve_memagent(agent_id)
    except Exception:
        return None
    policy = getattr(agent, "retrieval_policy", None)
    if hasattr(policy, "to_dict"):
        policy = policy.to_dict()
    if isinstance(policy, dict) and isinstance(policy.get("retention"), dict):
        return dict(policy["retention"])
    return None


def _saved_agent(agent_id: str) -> Any:
    try:
        return _state["provider"].retrieve_memagent(agent_id)
    except Exception:
        return None


def _consolidator(agent_id: str, *, with_model: bool = True) -> Any:
    """A consolidator for a saved agent, on its background (else chat) model."""
    from ...learning.consolidation import (
        ConsolidationConfig,
        MemoryConsolidator,
        model_for_agent,
    )

    record = _saved_agent(agent_id)
    model = None
    if with_model and record is not None:
        try:
            model = model_for_agent(record)
        except Exception as exc:
            logger.warning("No consolidation model for %s: %s", agent_id, exc)
    try:
        config = ConsolidationConfig.from_value(
            _agent_field(record, "memory_consolidation") or None
        )
    except (TypeError, ValueError):
        config = ConsolidationConfig()
    return MemoryConsolidator(
        _state["provider"],
        agent_id=agent_id,
        model=model,
        config=config,
        knowledge_base_ids=_agent_field(record, "knowledge_base_ids") or (),
    )


def _entity_scope(agent_id: str) -> str:
    policy = _agent_field(_saved_agent(agent_id), "retrieval_policy")
    if hasattr(policy, "to_dict"):
        policy = policy.to_dict()
    return str((policy or {}).get("entity_scope") or "memory")


def _pending_consolidations(agent_id: str) -> Dict[str, Any]:
    """Plans the agent's background schedule left for review in this process."""
    from ...memagent.core import _shared_upkeep

    return _shared_upkeep(_state["provider"], agent_id).plans


def _applied_consolidations(agent_id: str, limit: int = 8) -> list:
    """Recently applied consolidation plans, from the memory history journal."""
    from ...memory_history import MemoryHistory

    try:
        events = MemoryHistory(_state["provider"]).timeline(
            agent_id=agent_id, limit=1000
        )["events"]
    except Exception as exc:
        logger.debug("Consolidation history unavailable: %s", exc)
        return []
    plans: Dict[str, Dict[str, Any]] = {}
    for event in events:
        source = str(event.get("source") or "")
        if not source.startswith("consolidation:"):
            continue
        parts = source.split(":")
        plan = plans.setdefault(
            parts[1],
            {
                "plan_id": parts[1],
                "changes": 0,
                "undone": False,
                "approver": event.get("actor"),
                "at": event.get("timestamp"),
                "memory_id": event.get("memory_id"),
                "user_id": event.get("user_id"),
            },
        )
        if source.endswith(":restore"):
            plan["undone"] = True
        else:
            plan["changes"] += 1
            plan["at"] = max(str(plan["at"] or ""), str(event.get("timestamp") or ""))
            plan["memory_id"] = plan["memory_id"] or event.get("memory_id")
    return sorted(plans.values(), key=lambda item: str(item["at"]), reverse=True)[
        :limit
    ]


def _redirect(agent_id: str, **scope: Any) -> RedirectResponse:
    query = {"agent_id": agent_id}
    query.update(
        {key: value for key, value in scope.items() if value not in (None, "")}
    )
    return RedirectResponse(
        url="/learning-control-plane?" + urlencode(query), status_code=303
    )


def _capture_enabled(agent: Any) -> Optional[bool]:
    """Whether the saved agent records learning events (None: not a saved agent).

    Agents persist the effective setting, including the control plane that
    continual learning turns on implicitly.
    """
    if agent is None:
        return None
    return bool(_agent_field(agent, "learning_control_plane"))


@router.get("/learning-control-plane", response_class=HTMLResponse)
async def learning_control_plane_page(
    request: Request,
    agent_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        agents = _list_agents() or []
    except Exception:
        agents = []
    by_id = {
        identifier: agent
        for agent in agents
        for identifier in [_extract_agent_identifier(agent)]
        if identifier
    }
    known_ids = list(by_id)
    selected = (
        agent_id if agent_id in known_ids else (known_ids[0] if known_ids else agent_id)
    )
    selected_agent = by_id.get(selected or "")
    # The scope form always submits every field; a blank field means "all".
    memory_id, user_id, thread_id = (
        (value or "").strip() for value in (memory_id, user_id, thread_id)
    )
    view = None
    error = None
    if selected:
        plane = _plane(selected)
        try:
            records = plane.store.list_records(
                memory_id=memory_id or None,
                user_id=user_id or ...,
                thread_id=thread_id or None,
                limit=50_000,
            )
            saved = _agent_field(selected_agent, "learning_control_plane_config")
            budget = (
                saved.get("evidence_token_budget") if isinstance(saved, dict) else None
            )
            view = build_control_plane_view(
                records,
                # The stream the compile and dry-run actions work on.
                scope_stream_id=MemoryCompiler.stream_id(
                    selected,
                    memory_id=memory_id or None,
                    user_id=user_id or None,
                    thread_id=thread_id or None,
                ),
                evidence_budget=budget or plane.config.evidence_token_budget,
            )
        except Exception as exc:
            logger.error("Learning control-plane page failed: %s", exc)
            error = str(exc)
        finally:
            plane.close()
    suppressed = []
    if selected:
        review_plane = _plane(selected)
        try:
            suppressed = review_plane.suppressed_memories(
                memory_id=memory_id or None, user_id=user_id if user_id else ...
            )
        except Exception as exc:
            logger.debug("Suppressed-memory listing failed: %s", exc)
        finally:
            review_plane.close()
    consolidation: Dict[str, Any] = {}
    if selected:
        try:
            record = selected_agent or _saved_agent(selected)
            schedule = _agent_field(record, "memory_consolidation") or None
            background = _agent_field(record, "background_llm_config") or {}
            chat = _agent_field(record, "llm_config") or {}
            model = background or chat
            consolidation = {
                "memory_ids": list(_agent_field(record, "memory_ids") or []),
                "schedule": schedule,
                "model": (
                    f"{model.get('provider')}/{model.get('model')}"
                    if model.get("model")
                    else None
                ),
                "model_is_background": bool(background),
                "pending": list(_pending_consolidations(selected).values()),
                "applied": _applied_consolidations(selected),
                "entity_scope": _entity_scope(selected),
            }
        except Exception as exc:
            logger.debug("Consolidation panel data failed: %s", exc)
    return templates.TemplateResponse(
        "learning_control_plane.html",
        {
            "suppressed": suppressed,
            "consolidation": consolidation,
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agents_nav": _build_agent_nav_items(
                active_agent_id=selected, agents=agents
            ),
            "agents": agents,
            "known_agent_ids": known_ids,
            "agent_names": {
                identifier: _extract_agent_persona_name(agent)
                for identifier, agent in by_id.items()
            },
            "active_agent_id": selected,
            "selected_agent_id": selected,
            "capture_enabled": _capture_enabled(selected_agent),
            "memory_id": memory_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "view": view,
            "action": _last_actions.get(selected or ""),
            "active_page": "learning-control-plane",
            "ui_read_only": ui_read_only(),
            "error": error,
            "now": datetime.now(timezone.utc),
            **template_helpers(),
        },
    )


@router.post("/learning-control-plane/compile")
async def compile_memory(
    agent_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        report = plane.compile(
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        ).to_dict()
        _last_actions[agent_id] = {
            "kind": "compile",
            "message": f"Compiled {report['compiled_events']} event(s)",
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/forget-plan")
async def plan_forgetting(
    agent_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        report = plane.plan_forgetting(
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        ).to_dict()
        _last_actions[agent_id] = {
            "kind": "forget-plan",
            "message": f"Dry run found {report['candidate_count']} candidate(s)",
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/forget-apply")
async def apply_forgetting(
    agent_id: str = Form(...),
    plan_id: str = Form(...),
    approved_by: str = Form(...),
    reason: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    plane = _plane(agent_id)
    try:
        plan = plane.get_forgetting_plan(
            plan_id,
            memory_id=memory_id or None,
            user_id=user_id or None,
            thread_id=thread_id or None,
        )
        scope = {
            "memory_id": memory_id or None,
            "user_id": user_id or None,
            "thread_id": thread_id or None,
        }
        if plane.is_retention_plan(plan):
            report = plane.apply_retention(
                plan,
                approved_by=approved_by,
                reason=reason,
                scope=scope,
                config=_retention_overrides(agent_id),
            ).to_dict()
            message = (
                f"Suppressed {report['tombstoned']} memory record(s); "
                "each can be restored below"
            )
            report["plan_kind"] = "retention"
        else:
            report = plane.apply_forgetting(
                plan, approved_by=approved_by, reason=reason, scope=scope
            ).to_dict()
            message = f"Tombstoned {report['tombstoned']} artifact(s)"
        _last_actions[agent_id] = {
            "kind": "forget-apply",
            "message": message,
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/retention-plan")
async def plan_retention(
    agent_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Dry run over primary memories (conversation, knowledge, entities, ...)."""
    plane = _plane(agent_id)
    try:
        report = plane.plan_retention(
            memory_id=memory_id or None,
            user_id=user_id if user_id else ...,
            thread_id=thread_id or None,
            config=_retention_overrides(agent_id),
        ).to_dict()
        report["plan_kind"] = "retention"
        _last_actions[agent_id] = {
            "kind": "forget-plan",
            "message": (
                f"Dry run found {report['candidate_count']} memory record(s) "
                f"to suppress; {report['retained']} retained"
            ),
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/unsuppress")
async def unsuppress_memory(
    agent_id: str = Form(...),
    record_id: str = Form(...),
    memory_type: str = Form(...),
    approved_by: str = Form(...),
    reason: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Reverse a suppression; the record returns to retrieval immediately."""
    plane = _plane(agent_id)
    try:
        restored = plane.unsuppress_memory(
            record_id, memory_type, approved_by=approved_by, reason=reason
        )
        _last_actions[agent_id] = {
            "kind": "unsuppress",
            "message": (
                f"Restored {record_id[:12]} to retrieval"
                if restored
                else f"{record_id[:12]} was not suppressed"
            ),
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    finally:
        plane.close()
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/consolidation-plan")
async def plan_consolidation(
    agent_id: str = Form(...),
    consolidation_memory_id: str = Form(...),
    consolidation_user_id: Optional[str] = Form(None),
    consolidation_thread_ids: Optional[str] = Form(None),
    knowledge_base: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Dry-run consolidation of one memory (the model calls take a while)."""
    from starlette.concurrency import run_in_threadpool

    target = consolidation_memory_id.strip()
    person = (consolidation_user_id or "").strip() or None
    threads = [
        item.strip()
        for item in (consolidation_thread_ids or "").replace("\n", ",").split(",")
        if item.strip()
    ]

    def _plan() -> Dict[str, Any]:
        consolidator = _consolidator(agent_id)
        wanted = threads or consolidator.recent_threads(target, person)
        plan = consolidator.plan(
            memory_id=target,
            user_id=person,
            thread_ids=wanted,
            knowledge_base=knowledge_base in ("on", "true", "1"),
            entities=_entity_scope(agent_id) == "memory",
        )
        return {**plan.to_dict(), "thread_ids": wanted}

    try:
        report = await run_in_threadpool(_plan)
        total = len(report["ops"])
        _last_actions[agent_id] = {
            "kind": "consolidation-plan",
            "message": (
                f"Consolidation dry run for {target}: {total} change(s) to review"
                if total
                else f"Consolidation dry run for {target}: memory is tidy"
            ),
            "report": report,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/consolidation-review")
async def review_consolidation(
    agent_id: str = Form(...),
    plan_id: str = Form(...),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Open a plan the background schedule left for review."""
    plan = _pending_consolidations(agent_id).get(plan_id)
    if plan is None:
        _last_actions[agent_id] = {
            "kind": "error",
            "message": f"Consolidation plan {plan_id} is no longer waiting",
        }
    else:
        _last_actions[agent_id] = {
            "kind": "consolidation-plan",
            "message": f"Scheduled consolidation plan for {plan.memory_id}",
            "report": plan.to_dict(),
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/consolidation-apply")
async def apply_consolidation(
    agent_id: str = Form(...),
    plan_id: str = Form(...),
    approved_by: str = Form(...),
    reason: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Apply the reviewed plan; every change can be undone."""
    from ...learning.consolidation import ConsolidationPlan

    action = _last_actions.get(agent_id) or {}
    report = (
        action.get("report") if action.get("kind") == "consolidation-plan" else None
    )
    try:
        if not report or report.get("plan_id") != plan_id:
            raise ValueError(
                "This consolidation plan is no longer the one under review; plan again"
            )
        plan = ConsolidationPlan.from_dict(report)
        result = _consolidator(agent_id, with_model=False).apply(
            plan, approved_by=approved_by, reason=reason
        )
        _pending_consolidations(agent_id).pop(plan_id, None)
        applied = sum(result.applied.values())
        _last_actions[agent_id] = {
            "kind": "consolidation-applied",
            "message": (
                f"Applied {applied} consolidation change(s)"
                + (f"; {len(result.errors)} failed" if result.errors else "")
            ),
            "report": {
                **result.to_dict(),
                "memory_id": plan.memory_id,
                "user_id": plan.user_id,
            },
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


@router.post("/learning-control-plane/consolidation-restore")
async def restore_consolidation(
    agent_id: str = Form(...),
    plan_id: str = Form(...),
    approved_by: str = Form(...),
    consolidation_memory_id: str = Form(...),
    consolidation_user_id: Optional[str] = Form(None),
    memory_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    thread_id: Optional[str] = Form(None),
):
    """Undo an applied consolidation plan."""
    try:
        counts = _consolidator(agent_id, with_model=False).restore(
            plan_id,
            memory_id=consolidation_memory_id.strip(),
            user_id=(consolidation_user_id or "").strip() or None,
            approved_by=approved_by,
        )
        _last_actions[agent_id] = {
            "kind": "consolidation-restored",
            "message": (
                f"Undid consolidation {plan_id[:20]}: {counts['entities']} merge(s), "
                f"{counts['attributes']} retired fact(s) and {counts['rows']} hidden "
                f"row(s) restored, {counts['facts_removed']} added fact(s) removed"
            ),
            "at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        _last_actions[agent_id] = {"kind": "error", "message": str(exc)}
    return _redirect(
        agent_id, memory_id=memory_id, user_id=user_id, thread_id=thread_id
    )


__all__ = ["router"]
