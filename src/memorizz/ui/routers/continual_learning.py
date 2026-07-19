# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Continual-learning pages: trajectory classes and skill lifecycle.

Replaces the generic memory-list rendering for two slugs (registered
BEFORE the generic ``/memory/{memory_type}`` route so the literal paths
win):

- ``GET /memory/workflows`` — workflow runs grouped into trajectory
  classes by ``canonical_hash``, with per-class promotion-gate status, a
  per-class "Distill now" action for eligible classes, and a
  "Run promotion cycle" action per agent.
- ``GET /memory/skills`` — learned skills with lifecycle status, stats,
  the distilled SKILL.md, and Activate (shadow → active) / Demote
  (active → demoted) actions.

Promotion here is human-*triggered*, never human-*exempted*: the
"Distill now" action still runs the full eligibility gates and the
distillation validation gate.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from ...enums.memory_type import MemoryType
from ...llms.llm_factory import create_llm_provider
from ...long_term.procedural.skillbox import PromotionConfig, SkillStatus
from ...long_term.procedural.workflow.canonicalization import (
    TrajectoryStats,
    aggregate_trajectory_stats,
)
from ..state import _state, templates

logger = logging.getLogger(__name__)

router = APIRouter(tags=["continual-learning"])

# Last promotion/distillation report per agent, rendered as a panel on the
# workflows page after a redirect (server-rendered flash-message pattern).
_last_reports: Dict[str, Dict[str, Any]] = {}


class _DocsProvider:
    """Adapter feeding pre-fetched docs to aggregate_trajectory_stats,
    so the page does one provider scan instead of one per agent."""

    def __init__(self, docs: List[Dict[str, Any]]):
        self._docs = docs

    def list_all(self, memory_store_type=None):
        return self._docs


def _build_manager(agent_id: Optional[str]):
    """Construct a ContinualLearningManager for UI-triggered actions.

    Deliberately does NOT load a full MemAgent (tool registration can make
    LLM calls); the manager only needs the provider, the agent's stored
    llm_config for distillation, and its continual_learning_config.
    """
    from ...memagent.managers.continual_learning_manager import ContinualLearningManager

    provider = _state["provider"]
    model = None
    config = None
    if agent_id:
        try:
            agent_model = provider.retrieve_memagent(agent_id)
        except Exception:
            agent_model = None
        if agent_model is not None:
            config = getattr(agent_model, "continual_learning_config", None)
            llm_config = getattr(agent_model, "llm_config", None)
            if llm_config:
                try:
                    model = create_llm_provider(dict(llm_config))
                except Exception as exc:
                    logger.warning(
                        "Could not build LLM for agent %s: %s", agent_id, exc
                    )
    return ContinualLearningManager(
        provider, llm_provider=model, agent_id=agent_id, config=config
    )


def _gate_breakdown(
    stat: TrajectoryStats, config: PromotionConfig
) -> List[Dict[str, Any]]:
    """Per-gate pass/fail for display. The authoritative decision (which
    also handles re-promotion windows and live-skill coverage) comes from
    the engine; this is the explainer next to it."""
    age_days = (datetime.now() - stat.last_seen).days if stat.last_seen else None
    return [
        {
            "name": "executions",
            "passed": stat.executions >= config.min_executions,
            "detail": f"{stat.executions} / {config.min_executions} runs",
        },
        {
            "name": "success rate",
            "passed": stat.success_rate >= config.min_success_rate,
            "detail": f"{stat.success_rate:.0%} (need {config.min_success_rate:.0%})",
        },
        {
            "name": "query diversity",
            "passed": stat.distinct_query_count >= config.min_distinct_queries,
            "detail": (
                f"{stat.distinct_query_count} distinct "
                f"(need {config.min_distinct_queries})"
            ),
        },
        {
            "name": "recency",
            "passed": (
                age_days is not None and age_days <= config.max_days_since_last_seen
            ),
            "detail": (
                f"last seen {age_days}d ago"
                if age_days is not None
                else "no timestamps"
            ),
        },
    ]


def _tool_sequence(provider, stat: TrajectoryStats) -> str:
    """Human-readable tool chain for a class, from one sample run."""
    for run in stat.runs:
        if not run.record_id:
            continue
        try:
            doc = provider.retrieve_by_id(
                run.record_id, memory_store_type=MemoryType.WORKFLOW_MEMORY
            )
        except Exception:
            doc = None
        if not doc:
            continue
        signature = doc.get("canonical_signature")
        if signature:
            return " → ".join(str(unit.get("tool")) for unit in signature)
        steps = doc.get("steps") or {}
        if steps:
            from ...long_term.procedural.workflow.canonicalization import (
                canonical_signature,
            )

            return " → ".join(
                str(unit.get("tool")) for unit in canonical_signature(steps)
            )
    return "(no steps recorded)"


def _serialize_run(run) -> Dict[str, Any]:
    return {
        "workflow_id": run.workflow_id,
        "outcome": run.outcome,
        "created_at": run.created_at.strftime("%Y-%m-%d %H:%M")
        if run.created_at
        else "",
        "user_query": run.user_query or "",
        "promoted": bool(run.promoted_skill_id),
    }


def _report_panel(report) -> Dict[str, Any]:
    return {
        "promoted": list(report.promoted),
        "rejected": [
            {"hash": item[0][:12], "reasons": item[1]} for item in report.rejected
        ],
        "skipped": [
            {"hash": item[0][:12], "reason": item[1]} for item in report.skipped
        ],
        "review_flags": list(report.review_flags),
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@router.get("/memory/workflows", response_class=HTMLResponse)
async def workflow_classes_page(request: Request):
    """Workflow memory grouped into trajectory classes with gate status."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    provider = _state["provider"]

    documents: List[Dict[str, Any]] = []
    try:
        documents = (
            provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY) or []
        )
    except Exception as exc:
        logger.error("Failed to list workflow memory: %s", exc)

    # Partition once, aggregate per agent through the shared canonical logic.
    docs_by_agent: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for doc in documents:
        if isinstance(doc, dict):
            docs_by_agent.setdefault(doc.get("agent_id"), []).append(doc)

    skill_names: Dict[str, str] = {}
    try:
        for skill_doc in provider.list_all(memory_store_type=MemoryType.SKILLBOX) or []:
            if isinstance(skill_doc, dict) and skill_doc.get("skill_id"):
                skill_names[str(skill_doc["skill_id"])] = skill_doc.get("name", "")
    except Exception:
        pass

    config = PromotionConfig()
    agent_sections = []
    for agent_id, docs in sorted(
        docs_by_agent.items(), key=lambda kv: (kv[0] is None, str(kv[0]))
    ):
        stats = aggregate_trajectory_stats(_DocsProvider(docs), agent_id=agent_id)
        agent_config = config
        if agent_id:
            try:
                agent_model = provider.retrieve_memagent(agent_id)
                agent_config = PromotionConfig.from_dict(
                    getattr(agent_model, "continual_learning_config", None)
                    if agent_model
                    else None
                )
            except Exception:
                agent_config = config

        classes = []
        for stat in stats:
            gates = _gate_breakdown(stat, agent_config)
            eligible = all(gate["passed"] for gate in gates)
            covering_skill = (
                skill_names.get(str(stat.promoted_skill_id))
                if stat.promoted_skill_id
                else None
            )
            classes.append(
                {
                    "hash": stat.canonical_hash,
                    "hash_short": stat.canonical_hash[:12],
                    "tools": _tool_sequence(provider, stat),
                    "executions": stat.executions,
                    "success_rate": f"{stat.success_rate:.0%}",
                    "distinct_queries": stat.distinct_query_count,
                    "last_seen": stat.last_seen.strftime("%Y-%m-%d %H:%M")
                    if stat.last_seen
                    else "—",
                    "gates": gates,
                    "eligible": eligible and not stat.already_promoted,
                    "promoted": stat.already_promoted,
                    "covering_skill_id": stat.promoted_skill_id,
                    "covering_skill_name": covering_skill,
                    "runs": [_serialize_run(run) for run in stat.runs[:10]],
                    "run_count": len(stat.runs),
                }
            )

        agent_sections.append(
            {
                "agent_id": agent_id,
                "label": agent_id or "(no agent id)",
                "classes": classes,
                "total_runs": len(docs),
                "report": _last_reports.get(agent_id or ""),
                "can_act": agent_id is not None,
            }
        )

    return templates.TemplateResponse(
        "workflow_classes.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "agent_sections": agent_sections,
            "total_workflows": len(documents),
            "active_page": "workflows",
        },
    )


@router.get("/memory/skills", response_class=HTMLResponse)
async def skills_page(request: Request):
    """Learned skills with lifecycle status and actions."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    provider = _state["provider"]

    documents: List[Dict[str, Any]] = []
    try:
        documents = provider.list_all(memory_store_type=MemoryType.SKILLBOX) or []
    except Exception as exc:
        logger.error("Failed to list skillbox: %s", exc)

    skills = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        stats = doc.get("stats") or {}
        status = str(doc.get("status") or "candidate")
        injection_role = str(doc.get("injection_role") or "user").strip().lower()
        if injection_role not in {"user", "developer"}:
            injection_role = "user"
        skills.append(
            {
                "skill_id": doc.get("skill_id"),
                "agent_id": doc.get("agent_id"),
                "name": doc.get("name") or "(unnamed skill)",
                "description": doc.get("description") or "",
                "content": doc.get("content") or "",
                "status": status,
                "injection_role": injection_role,
                "version": doc.get("version", 1),
                "preconditions": doc.get("preconditions") or [],
                "tools_used": doc.get("tools_used") or [],
                "source_hash_short": str(doc.get("source_canonical_hash") or "")[:12],
                "activations": stats.get("activations", 0),
                "successes": stats.get("successes", 0),
                "failures": stats.get("failures", 0),
                "deviations": stats.get("deviations", 0),
                "baseline": doc.get("baseline") or {},
                "promoted_at": doc.get("promoted_at") or "",
                "demoted_at": doc.get("demoted_at") or "",
                "demotion_reason": doc.get("demotion_reason") or "",
                "can_activate": status
                in (
                    SkillStatus.SHADOW.value,
                    SkillStatus.CANDIDATE.value,
                ),
                "can_demote": status == SkillStatus.ACTIVE.value,
            }
        )
    order = {"active": 0, "shadow": 1, "candidate": 2, "deprecated": 3, "demoted": 4}
    skills.sort(key=lambda s: (order.get(s["status"], 9), s["name"]))

    return templates.TemplateResponse(
        "skills_list.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "skills": skills,
            "active_page": "skills",
        },
    )


@router.post("/continual-learning/{agent_id}/cycle")
async def run_promotion_cycle(agent_id: str):
    """Run a full gated promotion cycle for one agent."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        manager = _build_manager(agent_id)
        report = manager.run_promotion_cycle()
        _last_reports[agent_id] = _report_panel(report)
    except Exception as exc:
        logger.error("Promotion cycle failed for %s: %s", agent_id, exc)
        _last_reports[agent_id] = {
            "promoted": [],
            "rejected": [],
            "skipped": [],
            "review_flags": [f"cycle failed: {exc}"],
            "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    return RedirectResponse(url="/memory/workflows", status_code=303)


@router.post("/continual-learning/{agent_id}/distill/{canonical_hash}")
async def distill_class(agent_id: str, canonical_hash: str):
    """Gated single-class promotion ("Distill now")."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        manager = _build_manager(agent_id)
        report = manager.promote_class(canonical_hash)
        _last_reports[agent_id] = _report_panel(report)
    except Exception as exc:
        logger.error("Distillation failed for %s/%s: %s", agent_id, canonical_hash, exc)
        _last_reports[agent_id] = {
            "promoted": [],
            "rejected": [{"hash": canonical_hash[:12], "reasons": [str(exc)]}],
            "skipped": [],
            "review_flags": [],
            "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    return RedirectResponse(url="/memory/workflows", status_code=303)


@router.post("/continual-learning/skills/{skill_id}/activate")
async def activate_skill(skill_id: str):
    """SHADOW/CANDIDATE → ACTIVE, with workflow stamp backfill."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        agent_id = _skill_agent_id(skill_id)
        manager = _build_manager(agent_id)
        manager.activate_skill(skill_id)
    except Exception as exc:
        logger.error("Skill activation failed for %s: %s", skill_id, exc)
    return RedirectResponse(url="/memory/skills", status_code=303)


@router.post("/continual-learning/skills/{skill_id}/demote")
async def demote_skill(skill_id: str):
    """Manual demotion: stops retrieval, releases suppressed workflows."""
    if not _state["provider"]:
        return RedirectResponse(url="/connect", status_code=302)
    try:
        agent_id = _skill_agent_id(skill_id)
        manager = _build_manager(agent_id)
        manager.demote_skill(skill_id, reason="manually demoted from UI")
    except Exception as exc:
        logger.error("Skill demotion failed for %s: %s", skill_id, exc)
    return RedirectResponse(url="/memory/skills", status_code=303)


def _skill_agent_id(skill_id: str) -> Optional[str]:
    try:
        for doc in (
            _state["provider"].list_all(memory_store_type=MemoryType.SKILLBOX) or []
        ):
            if isinstance(doc, dict) and str(doc.get("skill_id")) == str(skill_id):
                return doc.get("agent_id")
    except Exception:
        pass
    return None
