# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Before-and-after checks of the learning-loop changes, without a model.

Each check runs the library's own code twice on the same generated scenario:
once configured as MemoRizz 0.17.24 behaved, once with the change on. No
model is called, so the numbers say what the mechanisms do, not what an
agent's answers gain from them. That needs the round-based suite described
in the README next to this file.

    python eval/learning_loop/mechanisms.py [--seed 7] [--json out.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from memorizz._time import utc_now
from memorizz.long_term.procedural.skillbox import PromotionConfig, Skill, SkillStatus
from memorizz.long_term.procedural.skillbox.distiller import (
    ParsedSkillMd,
    SkillDistiller,
)
from memorizz.long_term.procedural.skillbox.monitor import SkillMonitor
from memorizz.long_term.procedural.skillbox.promotion import PromotionEngine
from memorizz.long_term.procedural.skillbox.skill import holdout_summary
from memorizz.long_term.procedural.workflow.canonicalization import (
    RunRef,
    TrajectoryStats,
)
from memorizz.long_term.semantic.entity_memory import EntityMemory, attribute_timeline
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.long_term.semantic.persona import GuardrailWriteRefused, Persona
from memorizz.memagent.managers.continual_learning_manager import (
    ContinualLearningManager,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

DIMENSIONS = 256


def embed(text: str) -> List[float]:
    """A deterministic bag-of-words vector, so no embedding API is called."""
    vector = [0.0] * DIMENSIONS
    for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
        if len(word) < 3:
            continue
        digest = hashlib.sha256(word.encode()).digest()
        vector[int.from_bytes(digest[:2], "big") % DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return embed(text)


def _provider(root: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=root,
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


def _patch_embeddings() -> None:
    import memorizz.embeddings as embeddings
    from memorizz.long_term.procedural.skillbox import skill as skill_module
    from memorizz.long_term.semantic.persona import persona as persona_module

    embeddings.get_embedding = embed
    entity_module.get_embedding = embed
    skill_module.get_embedding = embed
    persona_module.get_embedding = embed


# ---------------------------------------------------------------- C1 and C2

# (first writer, later writer, whose value is right afterwards)
FACT_SCENARIOS = {
    "host value, then a tool result mentions another": (3, 1, "first"),
    "host value, then a user remark disagrees": (3, 2, "first"),
    "user states it, then the agent infers another": (2, 1, "first"),
    "user states it, then states a new value": (2, 2, "later"),
    "agent infers it, then the user states it": (1, 2, "later"),
    "agent infers it, then the host sets it": (1, 3, "later"),
}


def entity_check(rng: random.Random, root: Path, per_scenario: int = 60) -> Dict:
    """Is the value served after two writes the right one?

    Before: every write replaces the stored value (last write wins; the
    0.17.24 rule for unpinned values). After: the source rule.
    """
    results: Dict[str, Any] = {"scenarios": {}}
    for arm in ("before", "after"):
        entities = EntityMemory(_provider(root / f"entities-{arm}"))
        scope = {"memory_id": "m", "user_id": "u"}
        correct_total = stale_total = count = 0
        for label, (first, later, truth) in FACT_SCENARIOS.items():
            correct = 0
            for index in range(per_scenario):
                name = f"{label[:10]}-{index}"
                right, wrong = f"value-{rng.randrange(10**6)}", f"other-{index}"
                values = (right, wrong) if truth == "first" else (wrong, right)
                for value, level in zip(values, (first, later)):
                    entities.upsert_entity(
                        name=name,
                        attributes=[{"name": "fact", "value": value}],
                        # Before the rule every source ranked the same.
                        source_level=level if arm == "after" else 2,
                        **scope,
                    )
                stored = entities.get_entity_by_name(name, **scope)["attributes"][0]
                correct += stored["value"] == right
                stale_total += stored["value"] != right
            results["scenarios"].setdefault(label, {})[arm] = correct / per_scenario
            correct_total += correct
            count += per_scenario
        results[arm] = {
            "current_value_correct": correct_total / count,
            "wrong_value_served": stale_total / count,
        }
        entities.memory_provider.close()
    return results


def history_check(root: Path, changes: int = 12) -> Dict:
    """After ``changes`` replacements, how many past values can be dated?"""
    out = {}
    for arm, limit in (("before", 5), ("after", 20)):
        entities = EntityMemory(
            _provider(root / f"history-{arm}"), attribute_history_limit=limit
        )
        scope = {"memory_id": "m", "user_id": "u"}
        for index in range(changes + 1):
            entities.upsert_entity(
                name="limit",
                attributes=[{"name": "fact", "value": f"v{index}"}],
                source_level=2,
                **scope,
            )
        attribute = entities.get_entity_by_name("limit", **scope)["attributes"][0]
        rows = attribute_timeline(attribute)[1:]
        dated = [row for row in rows if row["valid_from"] and row["valid_to"]]
        out[arm] = {
            "past_values_kept": len(rows) / changes,
            # 0.17.24 kept one updated_at per replaced value: no interval.
            "past_values_with_both_dates": (len(dated) / changes)
            if arm == "after"
            else 0.0,
        }
        entities.memory_provider.close()
    return out


# ------------------------------------------------------------------------ A1


def gate_check(rng: random.Random, classes: int = 200) -> Dict:
    """Of the classes the gates let through, how many really worked?

    Every class has six runs that did not raise. In half of them the task
    did not actually succeed, and the host says so (it verifies none).
    """
    now = utc_now()
    stats: List[Tuple[TrajectoryStats, bool]] = []
    for index in range(classes):
        really_works = rng.random() < 0.5
        verified = rng.randint(2, 6) if really_works else 0
        runs = [
            RunRef(
                workflow_id=f"c{index}-r{n}",
                outcome="success",
                created_at=now - timedelta(hours=n),
                user_query=f"request {n}",
                outcome_verified=n < verified,
            )
            for n in range(6)
        ]
        stats.append(
            (
                TrajectoryStats(
                    canonical_hash=f"class-{index}",
                    executions=6,
                    successes=6,
                    success_rate=1.0,
                    distinct_query_count=6,
                    first_seen=now - timedelta(hours=6),
                    last_seen=now,
                    runs=runs,
                    verified_successes=verified,
                ),
                really_works,
            )
        )
    out = {}
    for arm, needed in (("before", 0), ("after", 2)):
        engine = PromotionEngine(
            memory_provider=None,
            skillbox=SimpleNamespace(list_skills=lambda **_: []),
            config=PromotionConfig(min_verified_outcomes=needed),
        )
        passed = [works for stat, works in stats if engine._eligibility(stat, {})[0]]
        out[arm] = {
            "classes_eligible": len(passed),
            "eligible_that_really_work": sum(passed) / len(passed) if passed else None,
            "working_classes_still_eligible": sum(passed)
            / sum(1 for _, works in stats if works),
        }
    return out


# ------------------------------------------------------------------------ A2


def holdout_check(rng: random.Random, skills: int = 60, threads: int = 600) -> Dict:
    """Does a skill that makes things worse get noticed?

    Each skill changes the success rate of matching requests by -15, 0 or
    +15 points from a 75% base. Before: every matching thread gets the
    skill, so there is nothing to compare with. After: one thread in ten
    runs without it.
    """
    out = {}
    for arm, rate in (("before", 0.0), ("after", 0.1)):
        manager = ContinualLearningManager.__new__(ContinualLearningManager)
        manager.config = PromotionConfig(holdout_rate=rate)
        flagged = {"harmful": 0, "neutral": 0, "helpful": 0}
        flagged_any = {"harmful": 0, "neutral": 0, "helpful": 0}
        totals = {"harmful": 0, "neutral": 0, "helpful": 0}
        withheld = total_runs = 0
        for index in range(skills):
            kind = ("harmful", "neutral", "helpful")[index % 3]
            effect = {"harmful": -0.15, "neutral": 0.0, "helpful": 0.15}[kind]
            skill = Skill(
                name=f"skill-{index}",
                description="d",
                content="c",
                skill_id=f"skill-{arm}-{index}",
                status=SkillStatus.ACTIVE,
                embedding=[0.0],
            )
            for thread in range(threads):
                held = manager.is_held_out(skill, f"thread-{thread}")
                success = rng.random() < 0.75 + (0.0 if held else effect)
                SkillMonitor._credit_holdout(
                    skill, "without" if held else "with", success, verified=True
                )
                withheld += held
                total_runs += 1
            totals[kind] += 1
            flagged[kind] += holdout_summary(skill.stats, 20)["flag"]
            flagged_any[kind] += holdout_summary(skill.stats, 20, flag_z=0)["flag"]
        out[arm] = {
            "threads_run_without_the_skill": withheld / total_runs,
            "harmful_skills_flagged": flagged["harmful"] / totals["harmful"],
            "helpful_skills_wrongly_flagged": flagged["helpful"] / totals["helpful"],
            "neutral_skills_flagged": flagged["neutral"] / totals["neutral"],
            "without_chance_margin": {
                "harmful_skills_flagged": flagged_any["harmful"] / totals["harmful"],
                "neutral_skills_flagged": flagged_any["neutral"] / totals["neutral"],
            },
        }
    return out


# ------------------------------------------------------------------------ A3

LIVE = [
    (
        "refund-status",
        "When a customer asks where their refund is",
        "where is my refund",
    ),
    ("order-tracking", "When asked to track a parcel or delivery", "track my parcel"),
    (
        "address-change",
        "When a customer wants to change a delivery address",
        "change my address",
    ),
    ("invoice-copy", "When asked for a copy of an invoice", "send me the invoice"),
]
CANDIDATES = [
    # (name, description, sample query, duplicates a live skill)
    (
        "refund-lookup",
        "When a customer asks where their refund is now",
        "where is my refund",
        True,
    ),
    (
        "parcel-tracking",
        "When asked to track a parcel or a delivery",
        "track my parcel please",
        True,
    ),
    (
        "delivery-address-update",
        "When a customer wants to change their delivery address",
        "change my address",
        True,
    ),
    (
        "invoice-resend",
        "When asked for a copy of an invoice again",
        "send me the invoice",
        True,
    ),
    (
        "warranty-claim",
        "When a customer reports a faulty product under warranty",
        "my kettle broke",
        False,
    ),
    (
        "loyalty-points",
        "When asked how many loyalty points an account has",
        "how many points do I have",
        False,
    ),
    (
        "store-hours",
        "When asked for the opening hours of a shop",
        "when do you open",
        False,
    ),
    (
        "gift-card-balance",
        "When asked for the balance left on a gift card",
        "gift card balance",
        False,
    ),
]


def duplicate_check() -> Dict:
    live = [
        Skill(
            name=name,
            description=description,
            content="c",
            queries=[query],
            status=SkillStatus.ACTIVE,
        )
        for name, description, query in LIVE
    ]
    out = {}
    for arm, threshold in (("before", 0.0), ("after", 0.85)):
        rejected_duplicates = rejected_new = 0
        for name, description, query, is_duplicate in CANDIDATES:
            parsed = ParsedSkillMd(
                name=name,
                description=description,
                preconditions=[],
                tools=[],
                body="",
                content="",
            )
            hit = SkillDistiller.duplicate_of(parsed, live, [query], threshold)
            rejected_duplicates += bool(hit) and is_duplicate
            rejected_new += bool(hit) and not is_duplicate
        duplicates = sum(1 for item in CANDIDATES if item[3])
        out[arm] = {
            "duplicate_candidates_rejected": rejected_duplicates / duplicates,
            "new_candidates_wrongly_rejected": rejected_new
            / (len(CANDIDATES) - duplicates),
        }
    return out


# ------------------------------------------------------------------------ B1


def guardrail_check(root: Path) -> Dict:
    provider = _provider(root / "persona")
    persona = Persona(name="Ada", guardrails=["Never disclose Q9."])
    persona.store_persona(provider)
    attempts = refused = 0
    for source in ("tool_call", "persona_learning", "refine_cycle"):
        attempts += 1
        try:
            if source == "tool_call":
                persona.update(
                    {"guardrails": []}, {"reason": "x", "source_type": source}, provider
                )
            else:
                persona.set_guardrails([], {"reason": "x", "source_type": source})
        except GuardrailWriteRefused:
            refused += 1
    provider.close()
    return {
        "before": {"learning_paths_refused": None, "note": "no guardrails existed"},
        "after": {
            "learning_paths_refused": refused / attempts,
            "guardrail_intact": persona.guardrails == ["Never disclose Q9."],
            "rendered_first": persona.generate_system_prompt_input().startswith(
                "Guardrails."
            ),
        },
    }


# ------------------------------------------------------------------------ D2


def restart_check(root: Path) -> Dict:
    """Does a plan left for review survive the process that made it?"""
    import threading

    from memorizz.learning.consolidation import (
        ConsolidationConfig,
        ConsolidationOp,
        ConsolidationPlan,
    )
    from memorizz.learning.refine import RefineCycle

    provider = _provider(root / "refine")
    plan = ConsolidationPlan(
        plan_id="plan-1",
        memory_id="m",
        user_id=None,
        ops=(ConsolidationOp("merge_entities", "entity_memory", "e1", "same"),),
    )
    in_process = {"plan-1": plan}  # where 0.17.24 kept a waiting plan
    agent = SimpleNamespace(
        memory_provider=provider,
        agent_id="agent-1",
        memory_ids=["m"],
        multi_tenant=False,
        memory_consolidation=ConsolidationConfig(enabled=True, auto_apply=False),
        persona_learning=SimpleNamespace(enabled=False),
        persona_manager=None,
        continual_learning_manager=None,
        background_model=SimpleNamespace(),
        _upkeep_lock=threading.Lock(),
        _consolidation_state={},
        plan_memory_consolidation=lambda memory, **_: plan,
        _drop_consolidation_plan=lambda plan_id: None,
        plan_memory_retention=lambda memory, user_id=None: {"candidate_count": 0},
    )
    RefineCycle(agent).run()
    in_process.clear()  # the restart
    later = RefineCycle(SimpleNamespace(memory_provider=provider, agent_id="agent-1"))
    kept = [item for item in later.pending_plans() if item["kind"] == "consolidation"]
    provider.close()
    return {
        "before": {"waiting_plans_after_restart": len(in_process)},
        "after": {"waiting_plans_after_restart": len(kept)},
    }


def run(seed: int = 7) -> Dict[str, Any]:
    _patch_embeddings()
    rng = random.Random(seed)
    root = Path(tempfile.mkdtemp(prefix="memorizz-learning-loop-"))
    try:
        return {
            "seed": seed,
            "C2 source rule: value served after two writes": entity_check(rng, root),
            "C1 dated history: after 12 replacements": history_check(root),
            "A1 verified-outcome gate": gate_check(rng),
            "A2 holdout": holdout_check(rng),
            "A3 duplicate check": duplicate_check(),
            "B1 guardrails": guardrail_check(root),
            "D2 plans waiting for review, after a restart": restart_check(root),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _show(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.0%}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", dest="json_path")
    args = parser.parse_args()
    results = run(args.seed)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(results, indent=2, default=str))
    for title, result in results.items():
        if not isinstance(result, dict):
            continue
        print(f"\n{title}")
        keys = [key for key in result.get("after", {}) if key != "note"]
        for key in keys:
            before = result.get("before", {}).get(key)
            print(
                f"  {key.replace('_', ' '):<38} before {_show(before):>5}   "
                f"after {_show(result['after'][key]):>5}"
            )
        for label, arms in (result.get("scenarios") or {}).items():
            print(
                f"    {label:<52} before {_show(arms['before']):>5}   "
                f"after {_show(arms['after']):>5}"
            )


if __name__ == "__main__":
    main()
