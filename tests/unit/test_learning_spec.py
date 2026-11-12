# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The nine changes of the workflow-to-skills, persona, entity and refine-cycle
spec: the verified-outcome gate, the holdout, the duplicate check, guardrails,
dated history, the source rule, aliases, the ordered cycle and its report."""

from __future__ import annotations

import shutil
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

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
from memorizz.long_term.semantic.entity_memory import (
    EntityMemory,
    attribute_timeline,
    entity_aliases,
)
from memorizz.long_term.semantic.entity_memory import entity_memory as entity_module
from memorizz.long_term.semantic.entity_memory import source_level
from memorizz.long_term.semantic.persona import GuardrailWriteRefused, Persona
from memorizz.memagent.managers.continual_learning_manager import (
    ContinualLearningManager,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

VECTOR = [0.6, 0.8, 0.0, 0.0]


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-spec-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def embeds(monkeypatch):
    def embed(text):
        return list(VECTOR)

    monkeypatch.setattr("memorizz.embeddings.get_embedding", embed)
    monkeypatch.setattr(entity_module, "get_embedding", embed)
    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding", embed
    )
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_dimensions", lambda model=None: len(VECTOR)
    )


class _Embedder:
    def get_embedding(self, text: str) -> List[float]:
        return list(VECTOR)


def _provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


def _stat(verified: int, runs: int = 6) -> TrajectoryStats:
    now = utc_now()
    refs = [
        RunRef(
            workflow_id=f"w{i}",
            outcome="success",
            created_at=now - timedelta(hours=i),
            user_query=f"query {i}",
            outcome_verified=i < verified,
        )
        for i in range(runs)
    ]
    return TrajectoryStats(
        canonical_hash="h1",
        executions=runs,
        successes=runs,
        success_rate=1.0,
        distinct_query_count=runs,
        first_seen=now - timedelta(hours=runs),
        last_seen=now,
        runs=refs,
        verified_successes=verified,
    )


# -- A1: the verified-outcome gate ---------------------------------------------


@pytest.mark.unit
def test_gate_is_off_by_default_and_blocks_unverified_classes_when_set():
    def engine(needed):
        return PromotionEngine(
            memory_provider=None,
            skillbox=SimpleNamespace(list_skills=lambda **_: []),
            config=PromotionConfig(min_verified_outcomes=needed),
        )

    # Off (0): five clean runs are enough, as before.
    assert engine(0)._eligibility(_stat(verified=0), {}) == (True, None)
    # Set to 2: the agent's own opinion of its runs is no longer enough.
    eligible, reason = engine(2)._eligibility(_stat(verified=1), {})
    assert not eligible and reason == "verified_outcomes 1 < 2"
    assert engine(2)._eligibility(_stat(verified=2), {}) == (True, None)


@pytest.mark.unit
def test_an_outcome_evaluator_marks_the_run_verified():
    from memorizz.long_term.procedural.workflow import Workflow, WorkflowOutcome
    from memorizz.memagent.core import MemAgent

    agent = SimpleNamespace(workflow_outcome_evaluator=lambda workflow: True)
    workflow = Workflow(name="run", generate_embedding=False)
    MemAgent._apply_workflow_outcome_evaluator(agent, workflow)
    assert workflow.outcome == WorkflowOutcome.SUCCESS and workflow.outcome_verified
    assert workflow.to_dict()["outcome_verified"] is True
    # Without an evaluator a run that merely did not raise is not verified.
    plain = Workflow(name="run", generate_embedding=False)
    MemAgent._apply_workflow_outcome_evaluator(
        SimpleNamespace(workflow_outcome_evaluator=None), plain
    )
    assert plain.outcome_verified is False


# -- A2: the holdout -----------------------------------------------------------


def _manager(**config) -> ContinualLearningManager:
    manager = ContinualLearningManager.__new__(ContinualLearningManager)
    manager.config = PromotionConfig(**config)
    return manager


def _skill(skill_id="s1", **kwargs) -> Skill:
    return Skill(
        name="lookup",
        description="When asked for an order",
        content="steps",
        skill_id=skill_id,
        status=SkillStatus.ACTIVE,
        embedding=list(VECTOR),
        **kwargs,
    )


@pytest.mark.unit
def test_one_thread_in_ten_runs_without_the_skill_and_stays_on_its_side():
    manager, skill = _manager(holdout_rate=0.1), _skill()
    sides = [manager.is_held_out(skill, f"thread-{n}") for n in range(4000)]
    assert 0.07 < sum(sides) / len(sides) < 0.13
    assert sides == [manager.is_held_out(skill, f"thread-{n}") for n in range(4000)]
    # Off at 0, never without a thread, never for an exempt skill.
    assert not any(
        _manager(holdout_rate=0.0).is_held_out(skill, f"thread-{n}") for n in range(200)
    )
    assert manager.is_held_out(skill, None) is False
    exempt = _skill(baseline={"holdout_exempt": True})
    assert not any(manager.is_held_out(exempt, f"thread-{n}") for n in range(500))


@pytest.mark.unit
def test_a_skill_the_user_stated_is_exempt_unless_someone_says_otherwise():
    stated = _skill(baseline={"stated_by_user": True})
    assert stated.holdout_exempt is True
    stated.holdout_exempt = False
    assert stated.holdout_exempt is False and stated.stated_by_user
    assert _skill().holdout_exempt is False


@pytest.mark.unit
def test_outcomes_are_credited_to_with_or_without_and_never_auto_demote():
    skills = {"s1": _skill("s1", baseline={"success_rate": 0.9})}
    box = SimpleNamespace(get_skill_by_id=skills.get, update_skill=lambda skill: True)
    monitor = SkillMonitor(box, memory_provider=None, config=PromotionConfig())

    def run(with_skill: bool, ok: bool, verified: bool = True):
        monitor.record_run_outcome(
            SimpleNamespace(
                skills_activated=["s1"] if with_skill else [],
                skills_held_out=[] if with_skill else ["s1"],
                outcome=SimpleNamespace(value="success" if ok else "failure"),
                outcome_verified=verified,
                canonical_hash=None,
            )
        )

    for index in range(20):
        run(True, ok=index < 13)  # 65% with the skill
        run(False, ok=index < 18)  # 90% without it
    summary = holdout_summary(skills["s1"].stats, min_runs=20)
    assert summary["verified"] and summary["enough"]
    assert summary["with"] == {"runs": 20, "successes": 13, "rate": 0.65}
    assert summary["without"] == {"runs": 20, "successes": 18, "rate": 0.9}
    # The skill's side is lower by more than chance explains: a review
    # flag, and nothing else happens.
    assert summary["flag"] is True
    # A gap inside the chance margin is reported but not flagged; with the
    # margin off, any lower figure is.
    close = {
        "holdout": {
            "with": {"verified_runs": 20, "verified_successes": 17},
            "without": {"verified_runs": 20, "verified_successes": 18},
        }
    }
    assert holdout_summary(close, min_runs=20)["flag"] is False
    assert holdout_summary(close, min_runs=20, flag_z=0)["flag"] is True
    assert skills["s1"].status == SkillStatus.ACTIVE
    # Held-out runs never count as activations or feed the drift monitor.
    assert skills["s1"].stats["activations"] == 20


@pytest.mark.unit
def test_unverified_runs_are_used_and_labelled_when_the_host_reports_nothing():
    stats = {
        "holdout": {
            "with": {"runs": 30, "successes": 27},
            "without": {"runs": 25, "successes": 15},
        }
    }
    summary = holdout_summary(stats, min_runs=20)
    assert summary["verified"] is False and summary["flag"] is False
    assert round(summary["difference"], 2) == 0.30
    # Below the run count on a side the figure is reported, never flagged.
    stats["holdout"]["without"] = {"runs": 5, "successes": 5}
    assert holdout_summary(stats, min_runs=20)["flag"] is False


@pytest.mark.unit
def test_the_holdout_share_is_set_per_agent_and_survives_a_reload(tmp_path):
    from memorizz.memagent import MemAgent

    provider = _provider(tmp_path)
    agent = MemAgent(
        instruction="Help.",
        memory_provider=provider,
        continual_learning=True,
        agent_id="agent-holdout",
    )
    agent.save()
    assert agent.continual_learning_manager.config.holdout_rate == 0.1

    settings = agent.set_holdout(0.25, min_runs=30, flag_z=0)
    assert settings == {
        "holdout_rate": 0.25,
        "holdout_min_runs": 30,
        "holdout_flag_z": 0.0,
    }
    # In force at once on this agent, and after a reload.
    assert agent.continual_learning_manager.config.holdout_rate == 0.25
    assert agent.continual_learning_manager.engine.config.holdout_min_runs == 30
    reloaded = MemAgent.load("agent-holdout", memory_provider=provider)
    assert reloaded.continual_learning_manager.config.holdout_rate == 0.25
    assert reloaded.continual_learning_manager.config.holdout_flag_z == 0.0

    # None keeps a value; a share outside 0 to 1 is refused and nothing moves.
    assert agent.set_holdout(min_runs=40)["holdout_rate"] == 0.25
    with pytest.raises(ValueError):
        agent.set_holdout(1.5)
    assert agent.continual_learning_manager.config.holdout_rate == 0.25
    assert agent.set_holdout(0)["holdout_rate"] == 0.0


# -- A3: the duplicate check ---------------------------------------------------


@pytest.mark.unit
def test_validation_rejects_a_near_duplicate_and_names_the_live_skill():
    parsed = ParsedSkillMd(
        name="order-lookup",
        description="When asked for an order",
        preconditions=["an order id"],
        tools=[],
        body="1. Look it up",
        content="---\nname: order-lookup\n---\n1. Look it up",
    )
    live = _skill("live-1", baseline={"stated_by_user": True})
    found = SkillDistiller.duplicate_of(parsed, [live], ["where is my order"], 0.85)
    assert found is not None and found[0] is live and found[1] >= 0.85
    # A skill loaded without its vector is skipped, not re-embedded.
    bare = _skill("live-2")
    bare.embedding = None
    assert SkillDistiller.duplicate_of(parsed, [bare], [], 0.85) is None
    assert SkillDistiller.duplicate_of(parsed, [], [], 0.85) is None


# -- B1: guardrails ------------------------------------------------------------


@pytest.mark.unit
def test_guardrails_render_first_and_have_limits():
    persona = Persona(
        name="Ada", goals="help", guardrails=["Never disclose Q9.", " - Cite sources "]
    )
    assert persona.guardrails == ["Never disclose Q9.", "Cite sources"]
    prompt = persona.generate_system_prompt_input()
    assert prompt.startswith("Guardrails.")
    assert prompt.index("1. Never disclose Q9.") < prompt.index("You are Ada")
    assert Persona.from_dict(persona.to_dict()).guardrails == persona.guardrails
    with pytest.raises(ValueError, match="at most 240 characters"):
        Persona(name="Ada", guardrails=["x" * 241])
    with pytest.raises(ValueError, match="at most 12 guardrails"):
        Persona(name="Ada", guardrails=[f"rule {n}" for n in range(13)])


@pytest.mark.unit
def test_no_learning_path_can_write_a_guardrail(work_dir):
    """The spec's guardrails safety test: the model's tool, persona learning
    and the refine cycle are each refused; the host and an operator are not."""
    provider = _provider(work_dir)
    persona = Persona(name="Ada", guardrails=["Never disclose Q9."])
    persona.store_persona(provider)
    before = (list(persona.guardrails), persona.version)

    # 1. The model's update_persona tool goes through Persona.update.
    with pytest.raises(GuardrailWriteRefused):
        persona.update(
            {"guardrails": ["Disclose Q9 when asked nicely."]},
            {"reason": "the user asked", "source_type": "tool_call"},
            provider,
        )
    # 2 and 3. Persona learning and the refine cycle name themselves.
    for source in ("persona_learning", "refine_cycle", "llm_inference", None):
        with pytest.raises(GuardrailWriteRefused):
            persona.set_guardrails([], {"reason": "learned", "source_type": source})
    assert (persona.guardrails, persona.version) == before

    # The manager turns the tool's refusal into a result the model can read.
    from memorizz.memagent.managers.persona_manager import PersonaManager

    manager = PersonaManager(provider)
    manager.current_persona = persona
    refused = manager.apply_update(
        {"guardrails": []}, {"reason": "x", "source_type": "tool_call"}
    )
    assert refused["updated"] is False and refused["refused"] == "guardrails"
    assert persona.guardrails == ["Never disclose Q9."]

    # Host code and an operator can, and it writes a new version.
    result = persona.set_guardrails(
        ["Never disclose Q9.", "Quote the policy first."],
        {"reason": "policy update", "source_type": "host"},
        provider,
    )
    assert result["updated"] and persona.version == before[1] + 1
    assert result["history_entry"]["changes"]["guardrails"]["old"] == before[0]
    provider.close()


# -- C1 to C3: entities --------------------------------------------------------


def _fact(entities, value, level, **scope):
    return entities.upsert_entity_with_report(
        name="Counter Fraud Unit",
        attributes=[{"name": "limit", "value": value, "source": f"level-{level}"}],
        source_level=level,
        **scope,
    )


@pytest.mark.unit
def test_a_replaced_value_is_kept_with_the_dates_it_held(work_dir):
    entities = EntityMemory(_provider(work_dir), attribute_history_limit=3)
    scope = {"memory_id": "m1", "user_id": "u1"}
    for value in ("2,000", "2,500", "3,000", "3,500", "4,000"):
        _fact(entities, value, 2, **scope)
    (record,) = entities.list_entities(**scope)
    (limit,) = record["attributes"]
    rows = attribute_timeline(limit)
    assert [row["value"] for row in rows] == ["4,000", "3,500", "3,000", "2,500"]
    assert rows[0]["current"] and rows[0]["valid_to"] is None
    # Each earlier value has both ends, and they join up.
    for newer, older in zip(rows, rows[1:]):
        assert older["valid_from"] and older["valid_to"]
        assert older["valid_to"] == newer["valid_from"]
    # The limit is a setting (3 here; 20 by default).
    assert len(limit["history"]) == 3
    assert EntityMemory(entities.memory_provider).attribute_history_limit == 20


@pytest.mark.unit
def test_a_record_written_before_the_upgrade_needs_no_migration():
    legacy = {
        "name": "limit",
        "value": "3,000",
        "updated_at": "2026-10-02T00:00:00+00:00",
        "history": [{"value": "2,500", "updated_at": "2026-09-01T00:00:00+00:00"}],
    }
    current, earlier = attribute_timeline(legacy)
    assert current["valid_from"] == "2026-10-02T00:00:00+00:00"
    assert earlier["valid_from"] == "2026-09-01T00:00:00+00:00" and earlier["applied"]
    # No source said: the lowest level, so it stays as replaceable as before.
    assert source_level(legacy) == 1
    assert source_level({"source": "turn"}) == 2
    assert source_level({"source": "memory_routing:turn"}) == 2
    assert source_level({"source": "tool"}) == 1
    assert source_level({"source": "tool", "pinned": True}) == 3


@pytest.mark.unit
def test_a_weaker_write_is_kept_but_does_not_replace_the_value(work_dir):
    entities = EntityMemory(_provider(work_dir))
    scope = {"memory_id": "m1", "user_id": "u1"}
    _fact(entities, "2,500", 3, **scope)  # the host, from the policy system
    stated = _fact(entities, "3,000", 2, **scope)  # a handler says otherwise
    inferred = _fact(entities, "2,000", 1, **scope)  # a tool result mentions it
    assert stated["kept_stronger"] == inferred["kept_stronger"] == ["limit"]
    (limit,) = entities.list_entities(**scope)[0]["attributes"]
    assert limit["value"] == "2,500"
    rows = attribute_timeline(limit)
    assert [(row["value"], row["applied"]) for row in rows[1:]] == [
        ("2,000", False),
        ("3,000", False),
    ]
    # Offering the same weaker value again adds nothing.
    again = _fact(entities, "2,000", 1, **scope)
    assert again["unchanged"] is True
    # The same level or higher replaces it, and the old value keeps its dates.
    _fact(entities, "2,750", 3, **scope)
    (limit,) = entities.list_entities(**scope)[0]["attributes"]
    assert limit["value"] == "2,750"
    replaced = [row for row in attribute_timeline(limit) if row["value"] == "2,500"]
    assert replaced[0]["applied"] and replaced[0]["valid_to"]


@pytest.mark.unit
def test_forgetting_a_thread_never_restores_a_value_that_was_not_applied(work_dir):
    entities = EntityMemory(_provider(work_dir))
    scope = {"memory_id": "m1", "user_id": "u1"}
    entities.upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Porto", "thread_id": "t1"}],
        source_level=2,
        **scope,
    )
    entities.upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Lisbon", "thread_id": "t2"}],
        source_level=3,
        **scope,
    )
    entities.upsert_entity(
        name="Ines",
        attributes=[{"name": "city", "value": "Faro", "thread_id": "t3"}],
        source_level=1,
        **scope,
    )
    entities.forget_thread("t2", **scope)
    (city,) = entities.list_entities(**scope)[0]["attributes"]
    assert city["value"] == "Porto"  # not "Faro", which was never current


@pytest.mark.unit
def test_a_merge_adds_aliases_that_resolve_and_undo_takes_them_back(work_dir):
    entities = EntityMemory(_provider(work_dir))
    scope = {"memory_id": "m1", "user_id": "u1"}
    kept = entities.upsert_entity(
        name="Counter Fraud Unit",
        entity_type="team",
        attributes=[{"name": "lead", "value": "Priya"}],
        **scope,
    )
    merged_away = entities.upsert_entity(
        name="CFU",
        entity_type="team",
        attributes=[{"name": "office", "value": "Leeds"}],
        **scope,
    )
    entities.consolidate_duplicate_entities(
        entity_ids=[kept, merged_away], apply=True, provenance="plan-1", **scope
    )
    survivor = next(
        row
        for row in entities.list_entities(**scope)
        if entity_aliases(row) or row["entity_id"] in (kept, merged_away)
    )
    other_name = (
        "CFU" if survivor["name"] == "Counter Fraud Unit" else "Counter Fraud Unit"
    )
    assert entity_aliases(survivor) == [other_name]
    # A later mention of the old name lands on the survivor, without renaming it.
    again = entities.upsert_entity(
        name=other_name, attributes=[{"name": "floor", "value": "3"}], **scope
    )
    assert again == survivor["entity_id"]
    assert entities.get_entity(again, **scope)["name"] == survivor["name"]
    # An operator can edit the list; undoing the merge removes what it added.
    assert entities.set_aliases(again, [other_name, "Fraud team"], **scope) == [
        other_name,
        "Fraud team",
    ]
    entities.restore_consolidation("plan-1", **scope)
    assert entity_aliases(entities.get_entity(again, **scope)) == []
    assert len(entities.list_entities(**scope)) == 2


# -- D1 and D2: the cycle and its report ---------------------------------------


class _Agent:
    """The parts of a MemAgent a refine cycle touches."""

    def __init__(self, provider, auto_apply=True):
        import threading

        from memorizz.learning.consolidation import (
            ConsolidationConfig,
            ConsolidationOp,
            ConsolidationPlan,
        )

        self.memory_provider, self.agent_id = provider, "agent-1"
        self.memory_ids, self.multi_tenant = ["m1"], False
        self.memory_consolidation = ConsolidationConfig(
            enabled=True, auto_apply=auto_apply, refine_cycle=True
        )
        self.persona_learning = SimpleNamespace(enabled=False, auto_apply=True)
        self.persona_manager = None
        self.continual_learning_manager = None
        self.background_model = SimpleNamespace(generate_text=lambda prompt: "ok")
        self._upkeep_lock = threading.Lock()
        self._consolidation_state = {}
        self.calls: List[str] = []
        self.applied: List[str] = []
        self._plan = ConsolidationPlan(
            plan_id="plan-9",
            memory_id="m1",
            user_id=None,
            ops=(
                ConsolidationOp("merge_entities", "entity_memory", "e1", "same"),
                ConsolidationOp("add_fact", "entity_memory", "e1", "stated"),
            ),
        )

    def plan_memory_consolidation(self, memory, **_):
        self.calls.append("consolidation")
        self.background_model.generate_text("review these entities")
        return self._plan

    def apply_memory_consolidation(self, plan, approved_by):
        self.applied.append(approved_by)
        return SimpleNamespace(applied={"merge_entities": 1, "add_fact": 1}, errors=[])

    def _drop_consolidation_plan(self, plan_id):
        pass

    def plan_memory_retention(self, memory, user_id=None):
        self.calls.append("retention")
        return {"candidate_count": 3, "plan_id": "ret-1"}


@pytest.mark.unit
def test_a_cycle_runs_the_jobs_in_order_and_writes_one_report(work_dir):
    from memorizz.learning.refine import JOBS, RefineCycle

    agent = _Agent(_provider(work_dir))
    report = RefineCycle(agent).run()
    assert [job["name"] for job in report["jobs"]] == [name for name, _ in JOBS]
    assert agent.calls == ["consolidation", "retention"]  # entities before retention
    by_name = {job["name"]: job for job in report["jobs"]}
    assert by_name["entities"]["changes"] == {"merge_entities": 1}
    assert by_name["stated_facts"]["changes"] == {"add_fact": 1}
    # Each change is also written out, so a person can read what the cycle did.
    (merge,) = by_name["entities"]["items"]
    assert merge["kind"] == "merge_entities" and merge["applied"] is True
    assert merge["detail"] == "same"
    (fact,) = by_name["stated_facts"]["items"]
    assert fact["kind"] == "add_fact" and fact["applied"] is True
    (planned,) = by_name["retention"]["items"]
    assert planned["applied"] is False and "3 memory record" in planned["text"]
    assert by_name["entities"]["tokens"] > 0 and by_name["entities"]["tokens_estimated"]
    assert by_name["promotion"]["status"] == "skipped"
    # Retention never applies itself: it waits, and the wait is in the report.
    assert by_name["retention"]["changes"] == {"planned_suppressions": 3}
    assert report["status"] == "done" and report["changed"] == 5
    assert [item["kind"] for item in report["pending"]] == ["retention"]
    assert agent.applied == ["memorizz:refine_cycle"]
    # One stored record per cycle, readable without the agent.
    stored = RefineCycle(
        SimpleNamespace(memory_provider=agent.memory_provider, agent_id="agent-1")
    ).list_runs()
    assert [run["run_id"] for run in stored] == [report["run_id"]]
    # The model's methods are left as they were.
    assert agent.background_model.generate_text("x") == "ok"


@pytest.mark.unit
def test_a_plan_waiting_for_review_is_stored_with_the_run(work_dir):
    """Before, a consolidation plan left for review lived in process memory
    and a restart lost it."""
    from memorizz.learning.consolidation import ConsolidationPlan
    from memorizz.learning.refine import RefineCycle

    provider = _provider(work_dir)
    agent = _Agent(provider, auto_apply=False)
    report = RefineCycle(agent).run()
    assert agent.applied == []
    assert report["waiting"] == 2  # the consolidation plan and retention

    # A new process: nothing but the provider and the agent's ID.
    later = RefineCycle(SimpleNamespace(memory_provider=provider, agent_id="agent-1"))
    waiting = [
        item for item in later.pending_plans() if item["kind"] == "consolidation"
    ]
    assert len(waiting) == 1 and waiting[0]["run_id"] == report["run_id"]
    plan = ConsolidationPlan.from_dict(waiting[0]["plan"])
    assert plan.plan_id == "plan-9" and len(plan.ops) == 2
    # Once reviewed it leaves the list.
    assert later.resolve_pending(report["run_id"], "plan-9") is True
    assert [item["kind"] for item in later.pending_plans()] == ["retention"]


@pytest.mark.unit
def test_a_dry_run_applies_nothing_and_a_quiet_agent_is_skipped(work_dir):
    from memorizz.learning.refine import RefineCycle

    agent = _Agent(_provider(work_dir))
    dry = RefineCycle(agent).run(dry_run=True)
    assert dry["status"] == "dry_run" and agent.applied == []
    by_name = {job["name"]: job for job in dry["jobs"]}
    assert by_name["entities"]["changes"] == {"planned_merge_entities": 1}
    # The scheduler skips a cycle when no conversation is newer than the last.
    cycle = RefineCycle(agent)
    cycle.has_new_activity = lambda since: False
    skipped = cycle.run(trigger="schedule", skip_if_idle=True)
    assert skipped["status"] == "skipped" and "No new conversation" in skipped["reason"]


@pytest.mark.unit
def test_refineing_turns_the_per_turn_counters_off():
    from memorizz.learning.consolidation import ConsolidationConfig

    config = ConsolidationConfig.from_value({"enabled": True, "refine_cycle": True})
    assert config.refine_cycle and config.refine_interval_hours == 24.0
    assert ConsolidationConfig().refine_cycle is False
    saved = ConsolidationConfig.from_value(config.to_dict())
    assert saved.refine_cycle and saved.enabled


@pytest.mark.unit
def test_the_refine_cycle_is_an_automation_action():
    from memorizz.automation import runner

    job = SimpleNamespace(
        action_type="refine_cycle", agent_id="agent-1", action_config={}
    )
    seen = {}

    def fake(job_arg, *, memory_provider):
        seen["job"] = job_arg
        return {"response": "Refine cycle done"}

    original = runner._run_refine_cycle
    runner._run_refine_cycle = fake
    try:
        result = runner.execute_job_action(
            job, scheduled_for_utc=utc_now(), memory_provider=object()
        )
    finally:
        runner._run_refine_cycle = original
    assert seen["job"] is job and result["response"] == "Refine cycle done"


@pytest.mark.unit
def test_the_persona_page_lists_what_each_agent_persona_learned(work_dir):
    from unittest.mock import patch

    from memorizz.ui import state
    from memorizz.ui.routers.persona_evolution import agent_persona_rows

    learned = {
        "version": 2,
        "timestamp": "2026-11-12T01:16:43+00:00",
        "changes": {"goals": {"old": "Help.", "new": "Help. Draft decline letters."}},
        "change_trigger": {
            "source_type": "persona_learning",
            "reason": "A pattern across 3 conversations",
            "addition": "Draft decline letters.",
            "approved_by": "memorizz:refine_cycle",
            "evidence": [{"thread_id": "t1", "quote": "draft a decline letter"}],
        },
    }
    agents = [
        {
            "agent_id": "a-1",
            "name": "Claims desk",
            "persona": {
                "name": "Ada",
                "version": 2,
                "guardrails": ["Never disclose Q9."],
                "evolution_history": [learned],
            },
            "persona_learning": {"enabled": True, "auto_apply": False},
            "persona_learning_state": {
                "pending": {
                    "field": "background",
                    "addition": "Knows subsidence claims.",
                    "evidence": [{"quote": "a subsidence claim"}],
                }
            },
        },
        {"agent_id": "a-2", "name": "No persona"},
    ]
    with patch.dict(state._state, {"provider": object()}), patch(
        "memorizz.ui.helpers._list_agents", return_value=agents
    ):
        (row,) = agent_persona_rows()
    assert (row["agent_name"], row["version"], row["guardrails"]) == (
        "Claims desk",
        2,
        1,
    )
    assert row["learning"] and not row["auto_apply"]
    assert row["pending"]["addition"] == "Knows subsidence claims."
    (change,) = row["changes"]
    assert change["addition"] == "Draft decline letters." and change["learned"]
    assert change["evidence"] == ["draft a decline letter"]
    assert change["by"] == "memorizz:refine_cycle"
