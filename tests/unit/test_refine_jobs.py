"""The refine cycle's verify, propagation, toolbox and compact jobs, and how
standing rules, confirmations and stale skill references reach its report."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest

from memorizz.enums import MemoryType
from memorizz.learning.compaction import Compactor
from memorizz.learning.consolidation import (
    ConsolidationConfig,
    ConsolidationOp,
    ConsolidationPlan,
)
from memorizz.learning.persona_learning import RuleProposal
from memorizz.learning.refine import JOBS, RefineCycle, describe_op
from memorizz.learning.tool_learning import ToolReview, ToolRevision
from memorizz.learning.verification import (
    SOURCE_MISSING,
    UNSUPPORTED,
    NoteVerifier,
    record_fact_evidence,
)
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]
ORDER = [
    "verify",
    "entities",
    "stated_facts",
    "propagation",
    "toolbox",
    "skill_staleness",
    "promotion",
    "skill_review",
    "persona_learning",
    "compact",
    "retention",
]


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr("memorizz.embeddings.get_embedding", lambda text: list(VECTOR))
    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding",
        lambda text: list(VECTOR),
    )
    monkeypatch.setattr(
        "memorizz.embeddings.get_embedding_dimensions", lambda model=None: len(VECTOR)
    )
    monkeypatch.setenv("MEMORIZZ_IMPORTANCE_RATER", "heuristic")


class _Model:
    """Answers the consolidation and verification prompts with fixed JSON."""

    model = "background"

    def __init__(self, facts=None, verdict=None):
        self.facts = facts or []
        self.verdict = verdict or {"supported": False, "quote": ""}
        self.prompts: List[str] = []

    def generate_text(self, prompt, instructions=None):
        self.prompts.append(prompt)
        if "durable facts" in prompt:
            return json.dumps({"facts": self.facts})
        if "same_fact" in prompt:
            return json.dumps({"same_fact": [], "same_entity": []})
        if "Does the conversation say this?" in prompt:
            return json.dumps(self.verdict)
        return "ok"

    def generate(self, messages, tools=None, **kwargs):
        return "ok"

    def get_config(self):
        return {"provider": "openai", "model": "background"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return None

    def asked_to_verify(self) -> int:
        return sum(
            "Does the conversation say this?" in prompt for prompt in self.prompts
        )


class _Embedder:
    def get_embedding(self, text):
        return list(VECTOR)


@pytest.fixture()
def work_dir():
    path = Path(tempfile.mkdtemp(prefix="memorizz-refine-jobs-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _provider(work_dir: Path) -> FileSystemProvider:
    return FileSystemProvider(
        FileSystemConfig(
            root_path=work_dir / "memory",
            embedding_provider=_Embedder(),
            use_faiss=False,
            lazy_vector_indexes=True,
        )
    )


def _agent(work_dir: Path, model: _Model, **kwargs) -> MemAgent:
    return MemAgent(
        model=_Model(),
        background_model=model,
        memory_provider=_provider(work_dir),
        memory_ids=["memory-1"],
        application_mode="assistant",
        auto_register=False,
        agent_id="agent-1",
        **kwargs,
    )


def _forget_conversation(provider, thread_id: str) -> None:
    for row in provider.list_all(MemoryType.CONVERSATION_MEMORY):
        if str(row.get("thread_id") or row.get("conversation_id")) == thread_id:
            provider.delete_by_id(
                str(row.get("_id") or row.get("id")), MemoryType.CONVERSATION_MEMORY
            )


def _job(report, name):
    return next(job for job in report["jobs"] if job["name"] == name)


# -- the order, the settings, old runs ---------------------------------------


def test_the_jobs_run_in_the_documented_order():
    assert [name for name, _ in JOBS] == ORDER


def test_the_cycle_settings_default_on_and_old_agents_load():
    config = ConsolidationConfig()
    assert config.refine_verify and config.refine_compact
    assert config.refine_verify_model_checks == 3
    assert config.refine_tool_review == "propose"
    assert config.refine_drop_suppressed_vectors is False
    # An agent saved before the settings existed gets the defaults.
    old = ConsolidationConfig.from_value({"enabled": True, "refine_cycle": True})
    assert old.refine_tool_review == "propose" and old.refine_verify
    assert ConsolidationConfig(refine_tool_review=False).refine_tool_review == "off"
    assert ConsolidationConfig(refine_tool_review="AUTO").refine_tool_review == "auto"
    assert ConsolidationConfig.from_value(config.to_dict()) == config


def test_a_run_stored_with_seven_jobs_still_prints():
    from memorizz.cli.learning_commands import _refine_lines

    old = {
        "run_id": "refine-old",
        "status": "done",
        "trigger": "manual",
        "started_at": "2026-11-01T00:00:00+00:00",
        "jobs": [
            {"name": "entities", "label": "Entities", "status": "done", "tokens": 0},
            {"name": "retention", "label": "Retention", "status": "done", "tokens": 0},
        ],
        "pending": [],
    }
    lines = _refine_lines(old)
    assert any("Entities" in line for line in lines)


def test_a_confirmation_is_a_stated_fact_and_reads_as_one():
    op = ConsolidationOp(
        "reinforce_fact",
        "entity_memory",
        "e1",
        "said again",
        {
            "entity": "user",
            "attribute": "city",
            "value": "Porto",
            "evidence": "still in Porto",
        },
    )
    text, detail = describe_op(op)
    assert text == "Confirmed user.city = Porto" and "still in Porto" in detail


# -- verify --------------------------------------------------------------------


def test_a_stated_fact_is_checked_against_the_words_it_quoted(work_dir):
    model = _Model(
        facts=[
            {
                "entity": "user",
                "attribute": "city",
                "value": "Lisbon",
                "evidence": "I live in Lisbon",
            }
        ]
    )
    agent = _agent(work_dir, model)
    agent._record_interaction("I live in Lisbon.", "Noted.", "memory-1", "t1")
    plan = agent.plan_memory_consolidation(thread_ids=["t1"])
    agent.apply_memory_consolidation(plan, approved_by="operator-7")

    found = agent.verify_notes("memory-1")
    assert found["checked"] == 1 and found["supported"] == 1
    assert found["findings"] == [] and model.asked_to_verify() == 0

    # The conversation is forgotten: the fact is flagged, and stays.
    _forget_conversation(agent.memory_provider, "t1")
    dry = agent.verify_notes("memory-1", dry_run=True)
    assert [item["kind"] for item in dry["findings"]] == [SOURCE_MISSING]
    assert agent.unverified_notes() == []  # a dry run stores no flag
    real = agent.verify_notes("memory-1")
    (flag,) = real["opened"]
    assert flag["target_type"] == "stated_fact" and flag["value"] == "Lisbon"
    (record,) = EntityMemory(agent.memory_provider).list_entities(memory_id="memory-1")
    assert [a["value"] for a in record["attributes"]] == ["Lisbon"]
    # Flagged once: the next pass does not open it again.
    assert agent.verify_notes("memory-1")["opened"] == []
    with pytest.raises(ValueError):
        agent.dismiss_unverified_note(flag["flag_id"], by="")
    agent.dismiss_unverified_note(flag["flag_id"], by="dev")
    assert agent.unverified_notes() == []
    assert agent.verify_notes("memory-1")["opened"] == []
    agent.close()


def test_a_quote_that_is_not_in_the_conversation_is_unsupported(work_dir):
    agent = _agent(work_dir, _Model())
    agent._record_interaction("I work Tuesdays to Fridays.", "Noted.", "memory-1", "t1")
    entity_id = EntityMemory(agent.memory_provider).upsert_entity(
        name="user",
        attributes=[{"name": "work_days", "value": "Mondays to Thursdays"}],
        memory_id="memory-1",
        source_level=1,
    )
    record_fact_evidence(
        agent.memory_provider,
        "agent-1",
        entity_id=entity_id,
        entity="user",
        attribute="work_days",
        value="Mondays to Thursdays",
        evidence="I work Mondays to Thursdays",
        thread_id="t1",
        memory_id="memory-1",
        user_id=None,
    )
    (finding,) = agent.verify_notes("memory-1", dry_run=True)["findings"]
    assert finding["kind"] == UNSUPPORTED
    assert "no longer in the conversation" in finding["reason"]
    agent.close()


def test_a_captured_fact_goes_to_the_model_only_when_code_cannot_tell(work_dir):
    model = _Model(verdict={"supported": False, "quote": ""})
    agent = _agent(work_dir, model)
    agent._record_interaction("I moved to Porto last year.", "Noted.", "memory-1", "t1")
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(
        name="user",
        attributes=[
            {"name": "city", "value": "Porto", "thread_id": "t1", "source": "turn"},
            {"name": "grade", "value": "Senior handler", "thread_id": "t1"},
        ],
        memory_id="memory-1",
        source_level=2,
    )
    # No budget: the one code could not settle is left, not flagged.
    left = agent.verify_notes("memory-1", dry_run=True, model_checks=0)
    assert left["supported"] == 1 and left["left"] == 1 and left["findings"] == []
    assert model.asked_to_verify() == 0
    # With a budget the model is asked once, about that one only.
    found = agent.verify_notes("memory-1", dry_run=True, model_checks=3)
    assert model.asked_to_verify() == 1 and found["model_checked"] == 1
    (finding,) = found["findings"]
    assert finding["kind"] == UNSUPPORTED and finding["attribute"] == "grade"
    agent.close()


def test_the_models_yes_needs_the_users_words_and_is_remembered(work_dir):
    model = _Model(verdict={"supported": True, "quote": "words nobody wrote"})
    agent = _agent(work_dir, model)
    agent._record_interaction("I run the home desk.", "Noted.", "memory-1", "t1")
    EntityMemory(agent.memory_provider).upsert_entity(
        name="user",
        attributes=[{"name": "role", "value": "Desk Lead", "thread_id": "t1"}],
        memory_id="memory-1",
        source_level=2,
    )
    # A yes with a quote the user never wrote is a no.
    assert len(agent.verify_notes("memory-1", dry_run=True)["findings"]) == 1
    model.verdict = {"supported": True, "quote": "I run the home desk"}
    assert agent.verify_notes("memory-1")["findings"] == []
    asked = model.asked_to_verify()
    # Confirmed once: it is not asked about again.
    assert agent.verify_notes("memory-1")["supported"] == 1
    assert model.asked_to_verify() == asked
    agent.close()


# -- the cycle, on a real agent ------------------------------------------------


def _note(provider, note_id, content, **extra):
    provider.store(
        data={"_id": note_id, "memory_id": "memory-1", "content": content, **extra},
        memory_store_type=MemoryType.KNOWLEDGE_BASE,
    )


def test_a_cycle_verifies_first_then_flags_what_still_says_the_old_value(work_dir):
    agent = _agent(
        work_dir,
        _Model(),
        memory_consolidation={"enabled": True, "refine_cycle": True},
    )
    provider = agent.memory_provider
    agent._record_interaction("Morning.", "Morning.", "memory-1", "t1")
    entities = EntityMemory(provider)
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Leeds"}],
        memory_id="memory-1",
        source_level=2,
    )
    agent.propagate_supersessions = False  # so the cycle, not the write, finds it
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Manchester"}],
        memory_id="memory-1",
        source_level=2,
    )
    agent.propagate_supersessions = True
    _note(provider, "note-1", "The Counter Fraud Unit sits in Leeds.", embedding=VECTOR)
    # A captured fact whose conversation never existed.
    entities.upsert_entity(
        name="user",
        attributes=[{"name": "city", "value": "Porto", "thread_id": "gone"}],
        memory_id="memory-1",
        source_level=2,
    )

    dry = agent.refine(dry_run=True, memory_id="memory-1")
    assert [job["name"] for job in dry["jobs"]] == ORDER
    assert _job(dry, "verify")["changes"] == {"would_flag_source_missing": 1}
    assert _job(dry, "propagation")["changes"] == {"would_flag_note": 1}
    assert agent.unverified_notes() == [] and agent.stale_references() == []

    report = agent.refine(memory_id="memory-1")
    verify, propagation = _job(report, "verify"), _job(report, "propagation")
    assert verify["changes"] == {"flagged_source_missing": 1}
    assert verify["items"][0]["applied"] is False
    assert "note(s) checked" in verify["notes"][0]
    assert propagation["changes"] == {"flagged_note": 1}
    assert "still says Leeds" in propagation["items"][0]["text"]
    assert len(agent.unverified_notes()) == 1 and len(agent.stale_references()) == 1
    # Flags are for a person: nothing waits in the cycle's review list for them.
    assert all(item["kind"] == "retention" for item in report["pending"])

    # Someone fixes the note; the next cycle closes its flag and opens none.
    _note(provider, "note-1", "The Counter Fraud Unit sits in Manchester.")
    again = agent.refine(memory_id="memory-1")
    assert _job(again, "propagation")["changes"] == {"closed": 1}
    assert _job(again, "verify")["changes"] == {}
    assert agent.stale_references() == []
    agent.close()


def test_a_skipped_cycle_skips_every_job(work_dir):
    agent = _agent(
        work_dir, _Model(), memory_consolidation={"enabled": True, "refine_cycle": True}
    )
    agent._record_interaction("Morning.", "Morning.", "memory-1", "t1")
    agent.refine(memory_id="memory-1")
    cycle = RefineCycle(agent)
    cycle.has_new_activity = lambda since: False
    skipped = cycle.run(trigger="schedule", skip_if_idle=True)
    assert skipped["status"] == "skipped"
    assert [job["status"] for job in skipped["jobs"]] == ["skipped"] * len(ORDER)
    agent.close()


def test_the_new_jobs_have_a_switch_each(work_dir):
    agent = _agent(work_dir, _Model())
    agent.with_refine_cycle(True, verify=False, compact=False, tool_review="off")
    agent._record_interaction("Morning.", "Morning.", "memory-1", "t1")
    report = agent.refine(memory_id="memory-1")
    for name in ("verify", "compact", "toolbox"):
        assert _job(report, name)["status"] == "skipped", name
    agent.close()


# -- compact -------------------------------------------------------------------


def test_compaction_embeds_again_what_lost_its_vector(work_dir):
    agent = _agent(
        work_dir, _Model(), memory_consolidation={"enabled": True, "refine_cycle": True}
    )
    provider = agent.memory_provider
    agent._record_interaction("Morning.", "Morning.", "memory-1", "t1")
    entities = EntityMemory(provider)
    for name in ("Counter Fraud Unit", "Large Loss Unit"):
        entities.upsert_entity(
            name=name,
            attributes=[{"name": "office", "value": "Leeds"}],
            memory_id="memory-1",
        )
    (lost, _kept) = entities.list_entities(memory_id="memory-1")
    provider.store(
        data={**lost, "embedding": None}, memory_store_type=MemoryType.ENTITY_MEMORY
    )
    _note(provider, "note-1", "Beacon holds the claims.", embedding=VECTOR)
    _note(provider, "note-2", "Tally releases payments.", embedding=None)
    _note(
        provider,
        "note-3",
        "An old duplicate.",
        embedding=VECTOR,
        retention_state="suppressed",
    )

    dry = agent.refine(dry_run=True, memory_id="memory-1")
    assert _job(dry, "compact")["changes"] == {
        "would_re_embed_entities": 1,
        "would_re_embed_notes": 1,
    }
    stored = {r["name"]: r for r in entities.list_entities(memory_id="memory-1")}
    assert not stored[lost["name"]].get("embedding")  # a dry run embeds nothing

    report = agent.refine(memory_id="memory-1")
    compact = _job(report, "compact")
    assert compact["changes"] == {"re_embed_entities": 1, "re_embed_notes": 1}
    assert all(item["applied"] for item in compact["items"])
    stored = {r["name"]: r for r in entities.list_entities(memory_id="memory-1")}
    assert stored[lost["name"]]["embedding"] == VECTOR
    notes = {r["_id"]: r for r in provider.list_all(MemoryType.KNOWLEDGE_BASE)}
    assert notes["note-2"]["embedding"] == VECTOR
    # Text is never touched, and a suppressed row keeps its vector by default.
    assert notes["note-2"]["content"] == "Tally releases payments."
    assert notes["note-3"]["embedding"] == VECTOR

    done = Compactor(agent).run("memory-1", drop_suppressed_vectors=True)
    assert done["dropped"] == 1
    notes = {r["_id"]: r for r in provider.list_all(MemoryType.KNOWLEDGE_BASE)}
    assert not notes["note-3"].get("embedding")
    assert notes["note-3"]["content"] == "An old duplicate."
    agent.close()


def test_an_entity_changed_in_the_cycle_is_embedded_again(work_dir):
    agent = _agent(work_dir, _Model())
    entities = EntityMemory(agent.memory_provider)
    entities.upsert_entity(
        name="Counter Fraud Unit",
        attributes=[{"name": "office", "value": "Leeds"}],
        memory_id="memory-1",
    )
    untouched = Compactor(agent).run("memory-1", since="2999-01-01T00:00:00+00:00")
    assert untouched["entities"] == 0
    changed = Compactor(agent).run("memory-1", since="2000-01-01T00:00:00+00:00")
    assert changed["entities"] == 1
    assert "changed in this cycle" in changed["items"][0][2]
    agent.close()


# -- toolbox and standing rules, on a stub ------------------------------------


class _Stub:
    """The parts of a MemAgent a refine cycle touches."""

    def __init__(self, provider, **settings):
        self.memory_provider, self.agent_id = provider, "agent-1"
        self.memory_ids, self.multi_tenant = ["m1"], False
        self.memory_consolidation = ConsolidationConfig(
            enabled=True, refine_cycle=True, **settings
        )
        self.persona_learning = SimpleNamespace(enabled=False, auto_apply=True)
        self.persona_manager = None
        self.continual_learning_manager = None
        self.background_model = SimpleNamespace(generate_text=lambda prompt: "ok")
        self._upkeep_lock = threading.Lock()
        self._consolidation_state = {}
        self.tool_reviews: List[dict] = []
        self.persona_applied: List[str] = []

    def plan_memory_consolidation(self, memory, **_):
        return ConsolidationPlan(plan_id="p", memory_id="m1", user_id=None, ops=())

    def _drop_consolidation_plan(self, plan_id):
        pass

    def plan_memory_retention(self, memory, user_id=None):
        return {"candidate_count": 0}


def _tool_review(status="proposed"):
    return ToolReview(
        revisions=[
            ToolRevision(
                revision_id="rev-1",
                agent_id="agent-1",
                tool="beacon_lookup_claim",
                status=status,
                notes_added=["claim_ref is CW- followed by six digits."],
                avoid_added=["Not for large losses; use refer_llu."],
            )
        ],
        skipped={"refer_llu": "too_few_calls", "set_hsr": "too_few_calls"},
        flags=[
            {
                "tool": "request_cat_export",
                "flag": "retrieved_never_called",
                "offered_turns": 6,
                "calls": 0,
                "turns_observed": 24,
            }
        ],
    )


def test_tool_revisions_wait_for_a_person_unless_told_otherwise(work_dir):
    provider = _provider(work_dir)
    agent = _Stub(provider)
    # No toolbox: the job costs nothing.
    assert _job(RefineCycle(agent).run(), "toolbox")["status"] == "skipped"

    agent.toolbox = object()

    def review_tools(**kwargs):
        agent.tool_reviews.append(kwargs)
        return _tool_review("applied" if kwargs["auto_apply"] else "proposed")

    agent.review_tools = review_tools
    report = RefineCycle(agent).run()
    job = _job(report, "toolbox")
    assert job["changes"] == {"planned_tool_revisions": 1, "usage_flags": 1}
    assert job["waiting"] == 1 and agent.tool_reviews[-1]["auto_apply"] is False
    revision, flag = job["items"]
    assert "1 usage note(s)" in revision["text"] and "“not for”" in revision["text"]
    assert revision["applied"] is False
    assert "retrieved never called" in flag["text"] and "6 turn(s)" in flag["detail"]
    assert "2 too few calls" in job["notes"][0]
    (waiting,) = report["pending"]
    assert waiting["kind"] == "tool_revision" and waiting["plan_id"] == "rev-1"

    dry = RefineCycle(agent).run(dry_run=True)
    assert agent.tool_reviews[-1]["dry_run"] is True and dry["pending"] == []

    auto = _Stub(provider, refine_tool_review="auto")
    auto.toolbox, auto.review_tools = object(), review_tools
    report = RefineCycle(auto).run()
    assert agent.tool_reviews[-1]["auto_apply"] is True
    assert agent.tool_reviews[-1]["approved_by"] == "memorizz:refine_cycle"
    assert _job(report, "toolbox")["changes"]["tool_revisions"] == 1
    assert report["pending"] == []


def test_a_standing_rule_is_reported_and_never_applied_by_the_cycle(work_dir):
    agent = _Stub(_provider(work_dir))
    agent.persona_learning = SimpleNamespace(
        enabled=True, auto_apply=True, across_users=True
    )
    agent.multi_tenant = True  # a desk sharing one assistant, read as a team
    agent.persona_manager = SimpleNamespace(configuration_persona=object())
    rule = RuleProposal(
        proposal_id="rule-1",
        rule="In customer updates, say a Q9 claim is under review.",
        evidence=(
            {"quote": "no mention of the CFU", "user_id": "priya.shah"},
            {"quote": "say it's under review", "user_id": "ellie.burns"},
        ),
        base_version=1,
        memory_id="m1",
        users=("priya.shah", "ellie.burns"),
    )
    agent.review_persona = lambda memory, user_id=None, dry_run=False: SimpleNamespace(
        proposal=None, outcome="no_change", code="no_pattern", rule_proposals=(rule,)
    )
    agent.apply_persona_proposal = lambda *a, **k: agent.persona_applied.append("x")

    report = RefineCycle(agent).run()
    job = _job(report, "persona_learning")
    assert job["status"] == "done"
    assert job["changes"] == {"planned_standing_rules": 1} and job["waiting"] == 1
    (item,) = job["items"]
    assert item["kind"] == "standing_rule" and item["applied"] is False
    assert "priya.shah, ellie.burns" in item["detail"]
    (waiting,) = report["pending"]
    assert waiting["kind"] == "persona_rule" and waiting["plan_id"] == "rule-1"
    assert agent.persona_applied == []  # auto_apply does not reach a rule

    # One user among many, and not reading the team: still skipped.
    agent.persona_learning.across_users = False
    assert _job(RefineCycle(agent).run(), "persona_learning")["status"] == "skipped"


def test_the_learning_page_template_still_compiles():
    from jinja2 import Environment, FileSystemLoader

    import memorizz.ui as ui

    templates = Path(ui.__file__).parent / "templates"
    Environment(loader=FileSystemLoader(str(templates))).get_template(
        "learning_control_plane.html"
    )
