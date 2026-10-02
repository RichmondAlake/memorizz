"""Shape continual-learning data for the workflow and skill monitor pages.

``/memory/workflows`` lists trajectory classes (one tool path per agent) with
their promotion criteria; ``/memory/skills`` lists the skills distilled from
them. Both pages render one grid row per item plus a summary tape, so this
module turns engine objects into flat, sortable rows and totals. It does no
I/O: the routes fetch documents and pass them in.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..long_term.procedural.skillbox import (
    PromotionConfig,
    SkillStatus,
    calculate_shadow_readiness,
)

# Below this success rate a class or skill is shown as failing (red).
LOW_SUCCESS = 0.5

CRITERIA_MEASURES = {
    "executions": "How many times the agent has run this exact tool path.",
    "success rate": "Share of those runs that ended in success.",
    "query diversity": (
        "Distinct user requests that led to this path, so a skill is not "
        "tied to one phrasing."
    ),
    "recency": "Days since the last run. Stale procedures are not distilled.",
}

CLASS_STATUS = {
    "ready": ("Ready to distill", "healthy"),
    "failing": ("Failing", "failing"),
    "evidence": ("Needs more evidence", "idle"),
    "promoted": ("Promoted", "promoted"),
}

SKILL_STATUS_ORDER = ["active", "shadow", "candidate", "deprecated", "demoted"]
SKILL_HEALTH = {
    "active": "healthy",
    "shadow": "degraded",
    "candidate": "idle",
    "deprecated": "idle",
    "demoted": "failing",
}
SKILL_PILL = {"active": "success", "shadow": "warning", "demoted": "danger"}


# ---------- small helpers ----------


def _local(value: Optional[datetime]) -> Optional[datetime]:
    """Naive local time, so stored aware and naive timestamps compare."""
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone().replace(tzinfo=None)
    return value


def _parse(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _local(value)
    if not value:
        return None
    try:
        return _local(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except (TypeError, ValueError):
        return None


def ago(value: Optional[datetime], now: datetime) -> str:
    """Compact age such as ``12m ago`` or ``3d ago``; ``—`` when unknown."""
    value, now = _local(value), _local(now)
    if value is None or now is None:
        return "—"
    seconds = (now - value).total_seconds()
    if seconds < 0:
        return "just now"
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def _stamp(value: Optional[datetime]) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if value else ""


def tone(rate: Optional[float], good: float = 0.8) -> str:
    """Signal colour class for a 0-1 success rate."""
    if rate is None:
        return ""
    if rate >= good:
        return "is-good"
    if rate >= LOW_SUCCESS:
        return "is-warn"
    return "is-bad"


def safe_key(value: Any) -> str:
    """Characters safe in element ids, data-keys and URL fragments."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-") or "none"


def class_key(agent_id: Optional[str], canonical_hash: str) -> str:
    """Stable row key for one agent's trajectory class (also a URL fragment)."""
    return f"{str(canonical_hash or '')[:16]}-{safe_key(agent_id)}"


def agent_key(agent_id: Optional[str]) -> str:
    return safe_key(agent_id) if agent_id else "none"


# ---------- workflow trajectories ----------


def promotion_criteria(
    stat: Any, config: PromotionConfig, now: datetime
) -> List[Dict[str, Any]]:
    """Per-criterion pass/fail for one trajectory class.

    The authoritative decision (re-promotion windows, live-skill coverage)
    stays with the promotion engine; this explains it next to the class.
    """
    last_seen = _local(getattr(stat, "last_seen", None))
    age_days = (_local(now) - last_seen).days if last_seen else None
    success_rate = float(stat.success_rate or 0.0)
    rows = [
        {
            "name": "executions",
            "passed": stat.executions >= config.min_executions,
            "detail": f"{stat.executions} / {config.min_executions} runs",
        },
        {
            "name": "success rate",
            "passed": success_rate >= config.min_success_rate,
            "detail": f"{success_rate:.0%} (need {config.min_success_rate:.0%})",
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
                f"last seen {age_days}d ago "
                f"(within {config.max_days_since_last_seen}d)"
                if age_days is not None
                else "no timestamps"
            ),
        },
    ]
    for row in rows:
        row["measures"] = CRITERIA_MEASURES[row["name"]]
    return rows


def _serialize_run(run: Any) -> Dict[str, Any]:
    created = _local(getattr(run, "created_at", None))
    return {
        "workflow_id": run.workflow_id,
        "outcome": run.outcome,
        "created_at": _stamp(created),
        "created_dt": created,
        "user_query": run.user_query or "",
        "promoted": bool(run.promoted_skill_id),
    }


def shape_class(
    stat: Any,
    *,
    agent_id: Optional[str],
    tools: str,
    config: PromotionConfig,
    now: datetime,
    skill_name: Optional[str] = None,
    run_limit: int = 10,
) -> Dict[str, Any]:
    """One grid row for a trajectory class (a ``TrajectoryStats``)."""
    criteria = promotion_criteria(stat, config, now)
    met = sum(1 for item in criteria if item["passed"])
    promoted = bool(stat.already_promoted)
    eligible = met == len(criteria) and not promoted
    rate = float(stat.success_rate or 0.0) if stat.executions else None
    if promoted:
        status = "promoted"
    elif eligible:
        status = "ready"
    elif rate is not None and rate < LOW_SUCCESS:
        status = "failing"
    else:
        status = "evidence"
    tags = [status] if status in ("ready", "promoted") else ["evidence"]
    if rate is not None and rate < LOW_SUCCESS:
        tags.append("failing")
    last_seen = _local(stat.last_seen)
    first_seen = _local(stat.first_seen)
    runs = [_serialize_run(run) for run in stat.runs[:run_limit]]
    return {
        "key": class_key(agent_id, stat.canonical_hash),
        "agent_id": agent_id,
        "agent_key": agent_key(agent_id),
        "hash": stat.canonical_hash,
        "hash_short": stat.canonical_hash[:12],
        "tools": tools,
        "step_count": len([part for part in tools.split(" → ") if part]),
        "executions": stat.executions,
        "successes": stat.successes,
        "failed": stat.executions - stat.successes,
        "success_ratio": rate,
        "success_rate": f"{rate:.0%}" if rate is not None else "—",
        "success_tone": tone(rate, config.min_success_rate),
        "distinct_queries": stat.distinct_query_count,
        "first_seen": _stamp(first_seen) or "—",
        "first_seen_ago": ago(first_seen, now),
        "last_seen": _stamp(last_seen) or "—",
        "last_seen_at": last_seen,
        "last_seen_ago": ago(last_seen, now),
        "last_seen_ts": last_seen.timestamp() if last_seen else 0,
        "gates": criteria,
        "criteria_met": met,
        "criteria_total": len(criteria),
        "eligible": eligible,
        "promoted": promoted,
        "status": status,
        "status_label": CLASS_STATUS[status][0],
        "status_health": CLASS_STATUS[status][1],
        "tags": tags,
        "covering_skill_id": stat.promoted_skill_id,
        "covering_skill_name": skill_name,
        "covering_skill_key": (
            safe_key(stat.promoted_skill_id) if stat.promoted_skill_id else ""
        ),
        "runs": runs,
        "run_count": len(stat.runs),
    }


def _agent_label(agent_id: Optional[str], name: str) -> str:
    if name:
        return name
    return str(agent_id)[:8] if agent_id else "(no agent id)"


def build_workflow_monitor(
    sections: Iterable[Dict[str, Any]], *, total_runs: int, now: datetime
) -> Dict[str, Any]:
    """Rows, per-agent roll-up and tape totals for the workflows page.

    ``sections`` holds one dict per agent: ``agent_id``, ``name``,
    ``classes`` (from :func:`shape_class`), ``total_runs``, ``report`` and
    ``can_act``.
    """
    rows: List[Dict[str, Any]] = []
    agents: List[Dict[str, Any]] = []
    for section in sections:
        agent_id = section.get("agent_id")
        label = _agent_label(agent_id, section.get("name") or "")
        classes = section.get("classes") or []
        can_act = bool(section.get("can_act"))
        for cls in classes:
            search = " ".join(
                [label, str(agent_id or ""), cls["tools"], cls["hash"]]
                + [cls["status_label"]]
                + [run["user_query"] for run in cls["runs"]]
            )
            rows.append(
                {
                    **cls,
                    "agent_label": label,
                    "agent_short": str(agent_id)[:8] if agent_id else "",
                    "agent_named": bool(section.get("name")),
                    "can_act": can_act,
                    "search": search.lower(),
                }
            )
        executions = sum(cls["executions"] for cls in classes)
        successes = sum(cls["successes"] for cls in classes)
        seen = [cls["last_seen_at"] for cls in classes if cls["last_seen_at"]]
        newest = max(seen) if seen else None
        rate = successes / executions if executions else None
        agents.append(
            {
                "agent_id": agent_id,
                "key": agent_key(agent_id),
                "label": label,
                "short": str(agent_id)[:8] if agent_id else "",
                "classes": len(classes),
                "runs": section.get("total_runs", executions),
                "ready": sum(1 for cls in classes if cls["status"] == "ready"),
                "promoted": sum(1 for cls in classes if cls["promoted"]),
                "success_rate": f"{rate:.0%}" if rate is not None else "—",
                "success_tone": tone(rate),
                "newest": _stamp(newest) or "—",
                "newest_ago": ago(newest, now),
                "report": section.get("report"),
                "can_act": can_act,
            }
        )

    # Actionable first: ready to distill, then most evidence, newest first.
    order = {"ready": 0, "failing": 1, "evidence": 2, "promoted": 3}
    rows.sort(
        key=lambda row: (
            order[row["status"]],
            -row["executions"],
            -row["last_seen_ts"],
        )
    )

    executions = sum(row["executions"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    rate = successes / executions if executions else None
    seen = [row["last_seen_at"] for row in rows if row["last_seen_at"]]
    newest = max(seen) if seen else None
    counts = {
        tag: sum(1 for row in rows if tag in row["tags"])
        for tag in ("ready", "evidence", "failing", "promoted")
    }
    return {
        "rows": rows,
        "agents": agents,
        "reports": [agent for agent in agents if agent["report"]],
        "counts": counts,
        "summary": {
            "runs": total_runs,
            "classes": len(rows),
            "agents": len(agents),
            "ready": counts["ready"],
            "promoted": counts["promoted"],
            "failing": counts["failing"],
            "success_rate": f"{rate:.0%}" if rate is not None else "—",
            "success_tone": tone(rate),
            "newest": _stamp(newest),
            "newest_ago": ago(newest, now),
        },
    }


# ---------- learned skills ----------


def shape_skill(
    doc: Dict[str, Any],
    config: PromotionConfig,
    now: datetime,
    agent_name: str = "",
) -> Dict[str, Any]:
    """One grid row for a skillbox document."""
    stats = doc.get("stats") or {}
    status = str(doc.get("status") or "candidate")
    agent_id = doc.get("agent_id")
    shadow_stats = stats.get("shadow") or {}
    readiness = calculate_shadow_readiness(
        shadow_stats,
        min_observations=config.shadow_readiness_min_observations,
        min_trajectory_match_rate=config.shadow_readiness_min_trajectory_match_rate,
        min_matched_success_rate=config.shadow_readiness_min_matched_success_rate,
    )
    injection_role = str(doc.get("injection_role") or "user").strip().lower()
    if injection_role not in {"user", "developer"}:
        injection_role = "user"
    successes = int(stats.get("successes", 0) or 0)
    failures = int(stats.get("failures", 0) or 0)
    outcomes = successes + failures
    rate = successes / outcomes if outcomes else None
    source_hash = str(doc.get("source_canonical_hash") or "")
    created = _parse(doc.get("created_at"))
    promoted_at = _parse(doc.get("promoted_at"))
    listed_at = promoted_at or created
    tags = [status]
    if status == SkillStatus.SHADOW.value and readiness["ready"]:
        tags.append("ready")
    skill_id = doc.get("skill_id")
    return {
        "key": safe_key(skill_id),
        "skill_id": skill_id,
        "agent_id": agent_id,
        "agent_key": agent_key(agent_id),
        "agent_short": str(agent_id)[:8] if agent_id else "",
        "agent_label": _agent_label(agent_id, agent_name),
        "agent_named": bool(agent_name),
        "name": doc.get("name") or "(unnamed skill)",
        "description": doc.get("description") or "",
        "content": doc.get("content") or "",
        "status": status,
        "status_label": status.capitalize(),
        "status_health": SKILL_HEALTH.get(status, "idle"),
        "status_pill": SKILL_PILL.get(status, ""),
        "status_rank": (
            SKILL_STATUS_ORDER.index(status) if status in SKILL_STATUS_ORDER else 9
        ),
        "tags": tags,
        "injection_role": injection_role,
        "version": doc.get("version", 1),
        "preconditions": doc.get("preconditions") or [],
        "tools_used": doc.get("tools_used") or [],
        "source_hash": source_hash,
        "source_hash_short": source_hash[:12],
        "source_key": class_key(agent_id, source_hash) if source_hash else "",
        "activations": int(stats.get("activations", 0) or 0),
        "successes": successes,
        "failures": failures,
        "deviations": int(stats.get("deviations", 0) or 0),
        "success_ratio": rate,
        "success_rate": f"{rate:.0%}" if rate is not None else "—",
        "success_tone": tone(rate),
        "baseline": doc.get("baseline") or {},
        "created_at": _stamp(created),
        "promoted_at": doc.get("promoted_at") or "",
        "listed_at": _stamp(listed_at) or "—",
        "listed_ago": ago(listed_at, now),
        "listed_ts": listed_at.timestamp() if listed_at else 0,
        "demoted_at": doc.get("demoted_at") or "",
        "demotion_reason": doc.get("demotion_reason") or "",
        "shadow_observations": readiness["observations"],
        "shadow_trajectory_match_rate": readiness["trajectory_match_rate"],
        "shadow_matched_success_rate": readiness["matched_success_rate"],
        "shadow_last_evaluated_at": shadow_stats.get("last_evaluated_at") or "",
        "shadow_ready": readiness["ready"],
        "shadow_readiness_reasons": readiness["reasons"],
        "shadow_evaluation_enabled": config.shadow_evaluation_enabled,
        "can_activate": status
        in (SkillStatus.SHADOW.value, SkillStatus.CANDIDATE.value),
        "can_demote": status == SkillStatus.ACTIVE.value,
        "search": " ".join(
            [
                str(doc.get("name") or ""),
                str(doc.get("description") or ""),
                str(skill_id or ""),
                str(agent_id or ""),
                agent_name,
                status,
                " ".join(str(tool) for tool in doc.get("tools_used") or []),
                source_hash,
            ]
        ).lower(),
    }


def build_skill_monitor(skills: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Rows (status order, then name) and tape totals for the skills page."""
    rows = sorted(skills, key=lambda row: (row["status_rank"], row["name"]))
    by_status = {status: 0 for status in SKILL_STATUS_ORDER}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    successes = sum(row["successes"] for row in rows)
    outcomes = successes + sum(row["failures"] for row in rows)
    rate = successes / outcomes if outcomes else None
    listed = [row for row in rows if row["listed_ts"]]
    newest = max(listed, key=lambda row: row["listed_ts"]) if listed else None
    agents = sorted(
        {row["agent_key"]: row["agent_label"] for row in rows}.items(),
        key=lambda item: item[1].lower(),
    )
    return {
        "rows": rows,
        # Chips only for statuses that occur, in lifecycle order.
        "statuses": [(status, count) for status, count in by_status.items() if count],
        "counts": by_status,
        "ready": sum(1 for row in rows if "ready" in row["tags"]),
        "agents": [{"key": key, "label": label} for key, label in agents],
        "summary": {
            "skills": len(rows),
            "activations": sum(row["activations"] for row in rows),
            "deviations": sum(row["deviations"] for row in rows),
            "success_rate": f"{rate:.0%}" if rate is not None else "—",
            "success_tone": tone(rate),
            "newest": newest["listed_at"] if newest else "",
            "newest_ago": newest["listed_ago"] if newest else "—",
        },
    }
