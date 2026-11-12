# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Skill document model for the Skillbox.

A learned skill is a SKILL.md document that lives in the database with a
lifecycle and reviewed instruction authority attached. It is distilled from
repeated successful workflow trajectories and injected as user context by
default, or as an application/developer instruction after explicit review.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ...._time import to_utc, utc_now
from ....embeddings import get_embedding

# Distinguishes "no embedding supplied — generate one" (fresh skill) from
# "document loaded without its embedding" (provider projections exclude the
# vector by default). Loading must NEVER re-bill the embedding API.
_UNSET = object()


class SkillStatus(Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DEMOTED = "demoted"


class SkillInjectionRole(str, Enum):
    """Authority assigned to a learned skill when it is rendered."""

    USER = "user"
    DEVELOPER = "developer"


def normalize_skill_injection_role(value: Any) -> SkillInjectionRole:
    """Normalize persisted/configured authority values or reject them."""
    if isinstance(value, SkillInjectionRole):
        return value
    normalized = str(value or SkillInjectionRole.USER.value).strip().lower()
    try:
        return SkillInjectionRole(normalized)
    except ValueError as exc:
        allowed = ", ".join(role.value for role in SkillInjectionRole)
        raise ValueError(
            f"skill injection role must be one of: {allowed}; got {value!r}"
        ) from exc


def _default_stats() -> Dict[str, Any]:
    return {
        "activations": 0,
        "successes": 0,
        "failures": 0,
        # Activated but the run took a different canonical_hash — the agent
        # had the skill in context and went another way. Signal, not noise.
        "deviations": 0,
        "last_activated_at": None,
        # Ring buffer of the most recent activation outcomes (True=success),
        # capped by the monitor's drift window. Avoids re-querying workflow
        # memory to compute the rolling success rate.
        "recent_outcomes": [],
        # The holdout: outcomes of matching runs with the skill injected and
        # with it withheld. ``verified_*`` count only runs whose outcome a
        # host or an evaluator confirmed.
        "holdout": {
            "with": _default_holdout_side(),
            "without": _default_holdout_side(),
        },
        # Passive SHADOW evaluation is deliberately separate from active
        # attribution.  These counters are reconciled from the auditable
        # ``Workflow.shadow_evaluations`` records, never from
        # ``skills_activated``.
        "shadow": {
            "observations": 0,
            "trajectory_matches": 0,
            "trajectory_mismatches": 0,
            "matched_successes": 0,
            "matched_failures": 0,
            "last_evaluated_at": None,
            "recent_evaluations": [],
        },
    }


def _default_holdout_side() -> Dict[str, int]:
    return {"runs": 0, "successes": 0, "verified_runs": 0, "verified_successes": 0}


def holdout_summary(
    stats: Optional[Dict[str, Any]], min_runs: int = 20, flag_z: float = 1.0
) -> Dict[str, Any]:
    """The with-and-without figure for one skill.

    Verified outcomes are used when both sides have some; otherwise the
    run's own outcome, labelled unverified. ``flag`` is set when each side
    has at least ``min_runs`` runs and the skill's side is lower by more
    than chance explains: ``flag_z`` standard errors of the difference
    (0 flags any negative difference; higher flags fewer skills that make
    no difference and misses more that do harm). A
    person then decides, nothing is demoted on this figure.
    """
    holdout = (stats or {}).get("holdout") or {}
    sides = {
        name: {**_default_holdout_side(), **(holdout.get(name) or {})}
        for name in ("with", "without")
    }
    verified = all(side["verified_runs"] > 0 for side in sides.values())
    summary: Dict[str, Any] = {"verified": verified}
    for name, side in sides.items():
        runs = side["verified_runs"] if verified else side["runs"]
        successes = side["verified_successes"] if verified else side["successes"]
        summary[name] = {
            "runs": int(runs),
            "successes": int(successes),
            "rate": (successes / runs) if runs else None,
        }
    with_side, without_side = summary["with"], summary["without"]
    summary["enough"] = (
        with_side["runs"] >= min_runs and without_side["runs"] >= min_runs
    )
    summary["difference"] = (
        with_side["rate"] - without_side["rate"]
        if with_side["rate"] is not None and without_side["rate"] is not None
        else None
    )
    margin = 0.0
    if summary["difference"] is not None and flag_z > 0:
        # Standard error of the difference between two success rates, with
        # one pseudo-run on each side so 0% and 100% still carry some doubt.
        variance = 0.0
        for side in (with_side, without_side):
            rate = (side["successes"] + 0.5) / (side["runs"] + 1)
            variance += rate * (1 - rate) / side["runs"]
        margin = float(flag_z) * variance**0.5
    summary["margin"] = margin
    summary["flag"] = bool(
        summary["enough"]
        and summary["difference"] is not None
        and summary["difference"] < -margin
    )
    return summary


def _merge_stats(stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge persisted stats with defaults, including the nested channel.

    Older skill documents have no ``shadow`` key.  A shallow merge would
    also lose newly added shadow defaults when a partial nested document is
    loaded, so keep this compatibility normalization in one place.
    """
    supplied = dict(stats or {})
    defaults = _default_stats()
    supplied_shadow = supplied.pop("shadow", None)
    supplied_holdout = supplied.pop("holdout", None)
    defaults.update(supplied)
    if isinstance(supplied_holdout, dict):
        defaults["holdout"] = {
            name: {
                **_default_holdout_side(),
                **(
                    supplied_holdout.get(name)
                    if isinstance(supplied_holdout.get(name), dict)
                    else {}
                ),
            }
            for name in ("with", "without")
        }
    defaults["shadow"] = {
        **_default_stats()["shadow"],
        **(supplied_shadow if isinstance(supplied_shadow, dict) else {}),
    }
    return defaults


class Skill:
    """A promoted skill document (``MemoryType.SKILLBOX``)."""

    def __init__(
        self,
        name: str,
        description: str,
        content: str,
        preconditions: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        skill_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        source_canonical_hash: Optional[str] = None,
        source_workflow_ids: Optional[List[str]] = None,
        exemplar_workflow_id: Optional[str] = None,
        status: SkillStatus = SkillStatus.CANDIDATE,
        version: int = 1,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        promoted_at: Optional[datetime] = None,
        demoted_at: Optional[datetime] = None,
        demotion_reason: Optional[str] = None,
        injection_role: Any = SkillInjectionRole.USER,
        baseline: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        embedding: Any = _UNSET,
    ):
        self.name = name
        self.description = description
        self.content = content
        self.preconditions = list(preconditions or [])
        self.tools_used = list(tools_used or [])
        self.queries = list(queries or [])
        self.skill_id = skill_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.user_id = user_id
        self.source_canonical_hash = source_canonical_hash
        self.source_workflow_ids = list(source_workflow_ids or [])
        self.exemplar_workflow_id = exemplar_workflow_id
        self.status = (
            status if isinstance(status, SkillStatus) else SkillStatus(str(status))
        )
        self.version = int(version)
        self.created_at = created_at or utc_now()
        self.updated_at = updated_at or utc_now()
        self.promoted_at = promoted_at
        self.demoted_at = demoted_at
        self.demotion_reason = demotion_reason
        self.injection_role = normalize_skill_injection_role(injection_role)
        self.baseline = dict(baseline) if baseline else {}
        self.stats = _merge_stats(stats)
        # Reuse the stored embedding on round-trip; only embed fresh docs.
        # ``None`` (doc loaded under an embedding-excluding projection) is
        # preserved as None — regenerating here would bill the embedding API
        # on every list/get.
        self.embedding = (
            self._generate_embedding() if embedding is _UNSET else embedding
        )

    @property
    def stated_by_user(self) -> bool:
        """The user stated this procedure; it was not distilled from runs."""
        return bool((self.baseline or {}).get("stated_by_user"))

    @property
    def holdout_exempt(self) -> bool:
        """Never withheld by the holdout. A skill the user stated is exempt
        unless someone turned that off, because the user asked for it.

        Kept in ``baseline`` so it is stored by every provider without a
        schema change."""
        value = (self.baseline or {}).get("holdout_exempt")
        return self.stated_by_user if value is None else bool(value)

    @holdout_exempt.setter
    def holdout_exempt(self, value: bool) -> None:
        self.baseline = {**(self.baseline or {}), "holdout_exempt": bool(value)}

    def _generate_embedding(self) -> List[float]:
        """Embed APPLICABILITY semantics only: name + description +
        preconditions + queries.

        Deliberately excludes ``content``, procedure steps, and the tool
        sequence. Workflow embeddings encode *how it was done*; skill
        retrieval must match *when to do it*. Including mechanism here would
        retrieve skills for queries that mention similar tools rather than
        similar intents — the most likely silent-quality bug in the system,
        so a unit test asserts on :meth:`embedding_input`.
        """
        return get_embedding(self.embedding_input())

    def embedding_input(self) -> str:
        """The exact text the skill embedding is generated from."""
        return " ".join(
            [
                self.name,
                self.description,
                " ".join(self.preconditions),
                " ".join(self.queries),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "preconditions": self.preconditions,
            "tools_used": self.tools_used,
            "queries": self.queries,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "source_canonical_hash": self.source_canonical_hash,
            "source_workflow_ids": self.source_workflow_ids,
            "exemplar_workflow_id": self.exemplar_workflow_id,
            "status": self.status.value,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "promoted_at": (self.promoted_at.isoformat() if self.promoted_at else None),
            "demoted_at": self.demoted_at.isoformat() if self.demoted_at else None,
            "demotion_reason": self.demotion_reason,
            "injection_role": self.injection_role.value,
            "baseline": self.baseline,
            "stats": self.stats,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            content=data.get("content", ""),
            preconditions=data.get("preconditions"),
            tools_used=data.get("tools_used"),
            queries=data.get("queries"),
            skill_id=data.get("skill_id"),
            agent_id=data.get("agent_id"),
            user_id=data.get("user_id"),
            source_canonical_hash=data.get("source_canonical_hash"),
            source_workflow_ids=data.get("source_workflow_ids"),
            exemplar_workflow_id=data.get("exemplar_workflow_id"),
            status=SkillStatus(data.get("status", SkillStatus.CANDIDATE.value)),
            version=data.get("version", 1),
            # Stamps without an offset were written with datetime.now().
            created_at=to_utc(data.get("created_at"), naive="local"),
            updated_at=to_utc(data.get("updated_at"), naive="local"),
            promoted_at=to_utc(data.get("promoted_at"), naive="local"),
            demoted_at=to_utc(data.get("demoted_at"), naive="local"),
            demotion_reason=data.get("demotion_reason"),
            # Legacy skills pre-date trust-aware injection and must retain the
            # original user-context authority after upgrade.
            injection_role=data.get("injection_role", SkillInjectionRole.USER.value),
            baseline=data.get("baseline"),
            stats=data.get("stats"),
            embedding=data.get("embedding"),
        )
