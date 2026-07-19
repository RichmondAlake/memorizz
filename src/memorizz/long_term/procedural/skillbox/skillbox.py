# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Skillbox — the store manager for learned skills (Toolbox-shaped)."""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....llms.llm_provider import LLMProvider
from ....memory_provider import MemoryProvider
from .skill import Skill, SkillStatus

logger = logging.getLogger(__name__)
_SCOPE_UNSET = object()


@dataclass
class ScoredSkill:
    """A retrieved skill with its vector-search similarity score."""

    skill: Skill
    similarity: float


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Skillbox:
    """Manage learned-skill documents in a memory provider.

    Mirrors the :class:`Toolbox` constructor shape so the two procedural
    stores stay structurally interchangeable.
    """

    def __init__(
        self,
        memory_provider: MemoryProvider,
        llm_provider: Optional[LLMProvider] = None,
        agent_id: Optional[str] = None,
    ):
        self.memory_provider = memory_provider
        if llm_provider is None:
            # Lazy default, matching Toolbox.get_openai_default() — only
            # needed by the distiller, so failure here must not be fatal.
            self.llm_provider = None
        else:
            self.llm_provider = llm_provider
        self.agent_id = agent_id
        # hash → skill_id map for write-time stamping; invalidated on any
        # mutation (add/update/set_status).
        self._active_hash_cache: Optional[Dict[Tuple[str, Optional[str]], str]] = None

    # ------------------------------------------------------------------ CRUD

    def add_skill(self, skill: Skill) -> str:
        if skill.agent_id is None and self.agent_id is not None:
            skill.agent_id = self.agent_id
        self._active_hash_cache = None
        return self.memory_provider.store(
            skill.to_dict(), memory_store_type=MemoryType.SKILLBOX
        )

    def get_skill_by_id(self, skill_id: str) -> Optional[Skill]:
        for doc in self._list_docs():
            if str(doc.get("skill_id")) == str(skill_id):
                return Skill.from_dict(doc)
        return None

    def get_skill_by_name(self, name: str) -> Optional[Skill]:
        for doc in self._list_docs():
            if doc.get("name") == name:
                return Skill.from_dict(doc)
        return None

    def get_active_skill_for_hash(
        self,
        canonical_hash: str,
        user_id: Any = _SCOPE_UNSET,
    ) -> Optional[Skill]:
        """Return the ACTIVE skill covering a trajectory class and user.

        ``user_id`` is exact when supplied, including ``None`` for anonymous
        workflows. Omitting it preserves the legacy unscoped lookup.
        """
        if not canonical_hash:
            return None
        if self._active_hash_cache is None:
            self._active_hash_cache = {
                (
                    str(doc.get("source_canonical_hash")),
                    doc.get("user_id"),
                ): str(doc.get("skill_id"))
                for doc in self._list_docs()
                if doc.get("status") == SkillStatus.ACTIVE.value
                and doc.get("source_canonical_hash")
            }
        if user_id is _SCOPE_UNSET:
            skill_id = next(
                (
                    skill_id
                    for (stored_hash, _stored_user_id), skill_id in (
                        self._active_hash_cache.items()
                    )
                    if stored_hash == str(canonical_hash)
                ),
                None,
            )
        else:
            skill_id = self._active_hash_cache.get((str(canonical_hash), user_id))
        if not skill_id:
            return None
        return self.get_skill_by_id(skill_id)

    def list_skills(
        self, statuses: Optional[Sequence[SkillStatus]] = None
    ) -> List[Skill]:
        wanted = {status.value for status in statuses} if statuses else None
        skills = []
        for doc in self._list_docs():
            if wanted is not None and doc.get("status") not in wanted:
                continue
            skills.append(Skill.from_dict(doc))
        return skills

    def update_skill(self, skill: Skill) -> bool:
        """Patch a skill's mutable lifecycle/stat fields.

        Deliberately partial: providers exclude ``embedding`` from reads by
        default, so a loaded Skill often has ``embedding=None`` — writing
        the full document back would destroy the stored vector (and with it
        retrieval). Identity/content fields never change in place; a
        re-promotion creates a new versioned skill instead.
        """
        skill.updated_at = datetime.now()
        self._active_hash_cache = None
        doc = self._find_doc_by_skill_id(skill.skill_id)
        if not doc:
            return False
        record_id = doc.get("_id") or doc.get("skill_id")
        patch = {
            "status": skill.status.value,
            "injection_role": skill.injection_role.value,
            "version": skill.version,
            "updated_at": skill.updated_at.isoformat(),
            "promoted_at": (
                skill.promoted_at.isoformat() if skill.promoted_at else None
            ),
            "demoted_at": skill.demoted_at.isoformat() if skill.demoted_at else None,
            "demotion_reason": skill.demotion_reason,
            "baseline": skill.baseline,
            "stats": skill.stats,
        }
        return bool(
            self.memory_provider.update_by_id(
                str(record_id), patch, memory_store_type=MemoryType.SKILLBOX
            )
        )

    def update_shadow_stats(self, skill_id: str, shadow_stats: Dict[str, Any]) -> bool:
        """Patch only one skill's passive-evaluation aggregate.

        This intentionally avoids :meth:`update_skill`: activation can race
        a background evaluation, and writing a previously loaded lifecycle
        status back with the stats could undo an explicit activation.
        """
        doc = self._find_doc_by_skill_id(skill_id)
        if not doc:
            return False
        skill = Skill.from_dict(doc)
        stats = dict(skill.stats)
        stats["shadow"] = dict(shadow_stats)
        record_id = doc.get("_id") or doc.get("skill_id")
        return bool(
            self.memory_provider.update_by_id(
                str(record_id),
                {"stats": stats},
                memory_store_type=MemoryType.SKILLBOX,
            )
        )

    def set_status(
        self,
        skill_id: str,
        status: SkillStatus,
        reason: Optional[str] = None,
    ) -> bool:
        """Transition a skill's lifecycle status.

        DEMOTED / DEPRECATED transitions stamp ``demoted_at`` and
        ``demotion_reason`` so every exit from ACTIVE is auditable.
        """
        skill = self.get_skill_by_id(skill_id)
        if not skill:
            return False
        skill.status = status
        if status in (SkillStatus.DEMOTED, SkillStatus.DEPRECATED):
            skill.demoted_at = datetime.now()
            skill.demotion_reason = reason
        elif status == SkillStatus.ACTIVE:
            skill.promoted_at = skill.promoted_at or datetime.now()
        return self.update_skill(skill)

    # ------------------------------------------------------------- retrieval

    def retrieve_skills_by_query(
        self,
        query: str,
        limit: int = 2,
        min_similarity: float = 0.70,
        statuses: Sequence[SkillStatus] = (SkillStatus.ACTIVE,),
        user_id: Any = _SCOPE_UNSET,
    ) -> List[ScoredSkill]:
        """Vector-search skills by WHEN-to-apply semantics, with scores.

        ``min_similarity`` defaults deliberately stricter than any other
        memory retrieval in the codebase: skills carry instruction
        authority, so a false-positive retrieval costs more than a miss
        (negative transfer on partial matches). Configurable — never
        silently lowered.
        """
        try:
            docs = (
                self.memory_provider.retrieve_by_query(
                    query,
                    memory_store_type=MemoryType.SKILLBOX,
                    limit=max(limit * 3, 6),
                )
                or []
            )
        except Exception as exc:
            logger.warning("Skill retrieval failed: %s", exc)
            return []

        return self._score_documents(
            query,
            docs,
            limit=limit,
            min_similarity=min_similarity,
            statuses=statuses,
            user_id=user_id,
            exact_agent_scope=False,
        )

    def retrieve_shadow_skills_by_query(
        self,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.70,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[ScoredSkill]:
        """Retrieve only SHADOW skills under exact agent/user isolation.

        First-party provider hooks apply all filters before final top-k.
        Providers without the hook retain a deliberately large
        over-fetch-and-filter fallback.
        """
        provider_hook = getattr(
            self.memory_provider, "retrieve_skillbox_candidates", None
        )
        try:
            if callable(provider_hook):
                docs = provider_hook(
                    query,
                    limit=limit,
                    statuses=[SkillStatus.SHADOW.value],
                    agent_id=agent_id,
                    user_id=user_id,
                )
            else:
                docs = self.memory_provider.retrieve_by_query(
                    query,
                    memory_store_type=MemoryType.SKILLBOX,
                    limit=max(limit * 10, 30),
                )
        except Exception as exc:
            logger.warning(
                "Shadow-skill retrieval failed for agent scope %s (%s)",
                agent_id or "(none)",
                type(exc).__name__,
            )
            return []

        return self._score_documents(
            query,
            docs or [],
            limit=limit,
            min_similarity=min_similarity,
            statuses=(SkillStatus.SHADOW,),
            user_id=user_id,
            agent_id=agent_id,
            exact_agent_scope=True,
        )

    def _score_documents(
        self,
        query: str,
        docs: Sequence[Dict[str, Any]],
        *,
        limit: int,
        min_similarity: float,
        statuses: Sequence[SkillStatus],
        user_id: Any = _SCOPE_UNSET,
        agent_id: Any = _SCOPE_UNSET,
        exact_agent_scope: bool,
    ) -> List[ScoredSkill]:
        """Apply provider-independent score and isolation checks."""
        wanted = {
            status.value if isinstance(status, SkillStatus) else str(status)
            for status in statuses
        }
        query_embedding = None
        scored: List[ScoredSkill] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            if doc.get("status") not in wanted:
                continue
            expected_agent = self.agent_id if agent_id is _SCOPE_UNSET else agent_id
            if exact_agent_scope:
                if doc.get("agent_id") != expected_agent:
                    continue
            elif expected_agent is not None and doc.get("agent_id") not in (
                None,
                expected_agent,
            ):
                continue
            if user_id is not _SCOPE_UNSET and doc.get("user_id") != user_id:
                continue
            similarity = doc.get("score")
            if similarity is None:
                similarity = doc.get("similarity")
            if similarity is None:
                if query_embedding is None:
                    try:
                        query_embedding = get_embedding(query)
                    except Exception:
                        query_embedding = []
                similarity = _cosine(query_embedding, doc.get("embedding") or [])
            similarity = float(similarity)
            if similarity < min_similarity:
                continue
            scored.append(
                ScoredSkill(skill=Skill.from_dict(doc), similarity=similarity)
            )

        scored.sort(key=lambda item: item.similarity, reverse=True)
        return scored[:limit]

    # -------------------------------------------------------------- internal

    def _list_docs(self) -> List[Dict[str, Any]]:
        try:
            docs = self.memory_provider.list_all(memory_store_type=MemoryType.SKILLBOX)
        except Exception as exc:
            logger.warning("Skillbox list failed: %s", exc)
            return []
        result = []
        for doc in docs or []:
            if not isinstance(doc, dict):
                continue
            if self.agent_id is not None and doc.get("agent_id") not in (
                None,
                self.agent_id,
            ):
                continue
            result.append(doc)
        return result

    def _find_doc_by_skill_id(self, skill_id: str) -> Optional[Dict[str, Any]]:
        for doc in self._list_docs():
            if str(doc.get("skill_id")) == str(skill_id):
                return doc
        return None
