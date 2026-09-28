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
from ...long_term.procedural.skillbox import PromotionConfig
from ...long_term.procedural.workflow.canonicalization import (
    TrajectoryStats,
    aggregate_trajectory_stats,
)
from ..learning_view import (
    LEARNING_TABS,
    build_skill_monitor,
    build_workflow_monitor,
    shape_class,
    shape_skill,
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
    now = datetime.now()
    agent_sections = []
    for agent_id, docs in sorted(
        docs_by_agent.items(), key=lambda kv: (kv[0] is None, str(kv[0]))
    ):
        stats = aggregate_trajectory_stats(_DocsProvider(docs), agent_id=agent_id)
        agent_config = config
        agent_name = ""
        if agent_id:
            try:
                agent_model = provider.retrieve_memagent(agent_id)
                agent_config = PromotionConfig.from_dict(
                    getattr(agent_model, "continual_learning_config", None)
                    if agent_model
                    else None
                )
                agent_name = str(getattr(agent_model, "name", "") or "")
            except Exception:
                agent_config = config

        classes = [
            shape_class(
                stat,
                agent_id=agent_id,
                tools=_tool_sequence(provider, stat),
                config=agent_config,
                now=now,
                skill_name=(
                    skill_names.get(str(stat.promoted_skill_id))
                    if stat.promoted_skill_id
                    else None
                ),
            )
            for stat in stats
        ]
        agent_sections.append(
            {
                "agent_id": agent_id,
                "name": agent_name,
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
            "view": build_workflow_monitor(
                agent_sections, total_runs=len(documents), now=now
            ),
            "total_workflows": len(documents),
            "learning_tabs": LEARNING_TABS,
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

    agents: Dict[Optional[str], tuple] = {}

    def _agent_info(agent_id: Optional[str]) -> tuple:
        """(promotion config, display name) per agent, read once."""
        if agent_id in agents:
            return agents[agent_id]
        config, name = PromotionConfig(), ""
        if agent_id:
            try:
                model = provider.retrieve_memagent(agent_id)
                config = PromotionConfig.from_dict(
                    getattr(model, "continual_learning_config", None) if model else None
                )
                name = str(getattr(model, "name", "") or "") if model else ""
            except Exception:
                pass
        agents[agent_id] = (config, name)
        return agents[agent_id]

    now = datetime.now()
    skills = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        config, name = _agent_info(doc.get("agent_id"))
        skills.append(shape_skill(doc, config, now, agent_name=name))

    return templates.TemplateResponse(
        "skills_list.html",
        {
            "request": request,
            "provider_type": _state["provider_type"],
            "connection_info": _state["connection_info"],
            "view": build_skill_monitor(skills),
            "skills": skills,
            "learning_tabs": LEARNING_TABS,
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
