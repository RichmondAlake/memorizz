"""The refine cycle of a memory harnesses share: raw answers kept, findings
distilled, each quoted from the answer it came from."""

import json

import pytest

from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.metaharness.context import HarnessContextBuilder
from memorizz.metaharness.handoff import handoff_memory_record
from memorizz.metaharness.memory_refine import (
    HarnessMemoryRefine,
    RefineSettings,
    parse_saved,
)

RUN_A, RUN_B, RUN_C = "aaaaaaaa-1111", "bbbbbbbb-2222", "cccccccc-3333"


class Model:
    """Answers with whatever findings the test gives it."""

    def __init__(self, *findings):
        self.replies = [list(findings)]
        self.prompts = []

    def then(self, *findings):
        self.replies.append(list(findings))
        return self

    def generate_text(self, prompt):
        self.prompts.append(prompt)
        # One reply per call, the last one repeated.
        reply = self.replies[min(len(self.prompts), len(self.replies)) - 1]
        return "Here you go:\n" + json.dumps({"findings": reply})


def add(text, run, quote):
    return {"action": "add", "text": text, "run": run, "quote": quote}


@pytest.fixture
def store(tmp_path):
    return FileSystemProvider(FileSystemConfig(root_path=tmp_path / "memory"))


def save(store, harness, run, text, memory="notes"):
    record = handoff_memory_record(
        {"harness": harness, "status": "succeeded", "run_id": run, "response": text},
        workflow={},
        memory_id=memory,
        thread_id=f"harness-run-{run}",
        agent_id="metaharness",
        user_id=None,
    )
    store.store(
        record, memory_store_type=MemoryType.CONVERSATION_MEMORY, memory_id=memory
    )


def cycle(store, tmp_path, model, **options):
    return HarnessMemoryRefine(
        store,
        settings=RefineSettings(tmp_path / "settings.json"),
        model_factory=lambda settings: model,
        **options,
    )


@pytest.mark.unit
def test_answers_become_findings_and_every_raw_answer_is_kept(store, tmp_path):
    save(
        store,
        "droid",
        RUN_A,
        "The release codeword is PELICAN-4471, and the freeze starts on Thursday.",
    )
    save(
        store,
        "codex",
        RUN_B,
        "The release codeword is **PELICAN-4471**. I searched memory to find it.",
    )
    save(store, "pi", RUN_C, "An answer in another memory.", memory="elsewhere")
    model = Model(
        add(
            "The release codeword is PELICAN-4471.",
            RUN_A,
            "The release codeword is PELICAN-4471",
        ),
        add(
            "The release codeword is PELICAN-4471.",
            RUN_B,
            "release codeword is PELICAN-4471",
        ),
        add("The freeze starts on Thursday.", RUN_A, "the freeze starts on Thursday"),
        add("The deadline moved to March.", RUN_A, "the deadline moved to March"),
        add("Codex searched memory.", "not-a-run", "I searched memory"),
    )
    refine = cycle(store, tmp_path, model, task_of=lambda run: "What is the codeword?")

    report = refine.run("notes")

    assert report["status"] == "done" and report["memory_id"] == "notes"
    findings, raw = report["jobs"]
    assert findings["changes"] == {"added": 2, "not_taken": 2}
    assert raw["changes"] == {"distilled": 2}
    reasons = [
        item["detail"] for item in findings["items"] if item["kind"] == "not_taken"
    ]
    assert reasons == [
        "its quote is not in that run's answer",
        "it names no answer read in this cycle",
    ]
    # The model saw the request beside each answer, and only this memory.
    assert "Asked: What is the codeword?" in model.prompts[0]
    assert "another memory" not in model.prompts[0]

    known = refine.findings("notes")
    codeword = next(item for item in known if "codeword" in item["text"])
    # Two harnesses said it in the same cycle: one finding, confirmed twice.
    assert codeword["confirmations"] == 2
    assert codeword["runs"] == [RUN_A, RUN_B]
    assert codeword["harnesses"] == ["droid", "codex"]
    assert codeword["quote"] == "The release codeword is PELICAN-4471"
    assert len(known) == 2

    # Raw answers are still in conversation memory, marked as distilled.
    kept = refine.answers("notes")
    assert [item["harness"] for item in kept] == ["droid", "codex"]
    assert all(item["distilled"]["refine_run_id"] == report["run_id"] for item in kept)
    assert refine.answers("notes", only_new=True) == []
    assert refine.list_runs("notes")[0]["run_id"] == report["run_id"]
    assert refine.list_runs("elsewhere") == []


@pytest.mark.unit
def test_later_runs_are_handed_findings_not_the_distilled_answers(store, tmp_path):
    save(store, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    builder = HarnessContextBuilder(store)

    def handed():
        pack = builder.build(
            "release codeword", memory_id="notes", user_id=None, thread_id=None
        )
        return sorted(record["memory_type"] for record in pack.records)

    assert handed() == ["conversation_memory"]
    model = Model(
        add(
            "The release codeword is PELICAN-4471.",
            RUN_A,
            "release codeword is PELICAN-4471",
        )
    )
    cycle(store, tmp_path, model).run("notes")
    assert handed() == ["knowledge_base"]


@pytest.mark.unit
def test_a_later_run_confirms_or_replaces_a_finding(store, tmp_path):
    save(store, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    model = Model(
        add("The release codeword is PELICAN-4471.", RUN_A, "codeword is PELICAN-4471")
    )
    refine = cycle(store, tmp_path, model)
    refine.run("notes")
    (held,) = refine.findings("notes")

    # A model credits one run; any other answer holding the quote agrees.
    assert held["confirmations"] == 1

    # A model that adds what is held is corrected to a confirmation.
    save(store, "codex", RUN_B, "Confirmed: the release codeword is PELICAN-4471.")
    model.then(
        add(
            "The release codeword is PELICAN-4471",
            RUN_B,
            "the release codeword is PELICAN-4471",
        )
    )
    second = refine.run("notes")
    assert second["jobs"][0]["changes"] == {"confirmed": 1}
    (held,) = refine.findings("notes")
    assert held["confirmations"] == 2 and held["harnesses"] == ["droid", "codex"]
    assert f"[{held['id']}]" in model.prompts[1]

    save(store, "pi", RUN_C, "The codeword changed: it is now HERON-9.")
    model.then(
        {"action": "replace", "note": held["id"], "text": "The release codeword is HERON-9.",
         "run": RUN_C, "quote": "it is now HERON-9", "why": "the codeword changed"}
    )  # fmt: skip
    third = refine.run("notes")
    assert third["jobs"][0]["changes"] == {"replaced": 1}
    (current,) = refine.findings("notes")
    assert current["text"] == "The release codeword is HERON-9."
    assert current["replaces"] == held["id"]
    # The corrected note is kept, hidden, pointing at its successor.
    old = next(
        i
        for i in refine.findings("notes", include_replaced=True)
        if i["id"] == held["id"]
    )
    assert (
        old["replaced_by"] == current["id"]
        and old["why_replaced"] == "the codeword changed"
    )


@pytest.mark.unit
def test_a_dry_run_changes_nothing_and_nothing_new_is_skipped(store, tmp_path):
    save(store, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    model = Model(
        add("The release codeword is PELICAN-4471.", RUN_A, "codeword is PELICAN-4471")
    )
    refine = cycle(store, tmp_path, model)

    preview = refine.run("notes", dry_run=True)
    assert preview["status"] == "dry_run"
    assert preview["jobs"][0]["items"][0]["applied"] is False
    assert preview["jobs"][1]["changes"] == {"would_distil": 1}
    assert refine.findings("notes") == [] and refine.list_runs("notes") == []
    assert len(refine.answers("notes", only_new=True)) == 1

    refine.run("notes")
    again = refine.run("notes")
    assert again["status"] == "skipped" and len(model.prompts) == 2


@pytest.mark.unit
def test_without_a_model_a_cycle_says_so_and_keeps_the_answers_waiting(store, tmp_path):
    save(store, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    refine = cycle(store, tmp_path, None)

    report = refine.run("notes")
    assert report["status"] == "failed"
    assert "No model is set" in report["reason"]
    assert len(refine.answers("notes", only_new=True)) == 1

    class Broken:
        def generate_text(self, prompt):
            return "I would rather not."

    assert cycle(store, tmp_path, Broken()).run("notes")["status"] == "failed"
    with pytest.raises(ValueError, match="memory_id is required"):
        refine.run("")


@pytest.mark.unit
def test_settings_and_starting_by_itself(store, tmp_path):
    settings = RefineSettings(tmp_path / "settings.json")
    assert settings.get() == {
        "provider": "",
        "model": "",
        "auto": False,
        "after_answers": 3,
    }
    # Automatic cycles need a model: they spend tokens.
    assert settings.set({"auto": True})["auto"] is False
    saved = settings.set(
        {"provider": "OpenAI", "model": "gpt-6.2", "auto": True, "after_answers": 2}
    )
    assert saved == {
        "provider": "openai",
        "model": "gpt-6.2",
        "auto": True,
        "after_answers": 2,
    }
    with pytest.raises(ValueError):
        settings.set({"after_answers": "many"})

    model = Model(
        add("The release codeword is PELICAN-4471.", RUN_A, "codeword is PELICAN-4471")
    )
    refine = HarnessMemoryRefine(
        store, settings=settings, model_factory=lambda s: model
    )
    save(store, "droid", RUN_A, "The release codeword is PELICAN-4471.")
    assert refine.maybe_start("notes") is False  # one answer, two needed
    save(store, "codex", RUN_B, "Nothing new here.")
    assert refine.maybe_start("notes") is True
    import time

    for _ in range(100):
        if refine.list_runs("notes"):
            break
        time.sleep(0.05)
    assert refine.list_runs("notes")[0]["trigger"] == "automatic"
    assert (
        refine.findings("notes")[0]["text"] == "The release codeword is PELICAN-4471."
    )


@pytest.mark.unit
def test_a_saved_answer_is_taken_apart():
    saved = parse_saved(
        "[Harness run: claude-code · succeeded]\nAll good.\nHarness run: 1234abcd-ef"
    )
    assert saved == {
        "harness": "claude-code",
        "status": "succeeded",
        "stage": "",
        "run_id": "1234abcd-ef",
        "text": "All good.",
    }
    stage = parse_saved(
        "[Harness plan stage 1/4: Diagnose · codex · succeeded]\nTwo bugs.\n"
        "Harness run: 3ea79903-6d2b"
    )
    assert (stage["harness"], stage["stage"], stage["status"]) == (
        "codex",
        "Diagnose",
        "succeeded",
    )
    assert stage["text"] == "Two bugs." and stage["run_id"] == "3ea79903-6d2b"
    compared = parse_saved(
        "[Harness comparison run 2/3: pi · failed]\nNo.\nHarness run: 99999999-aaaa"
    )
    assert (compared["harness"], compared["stage"], compared["status"]) == (
        "pi",
        "",
        "failed",
    )
    assert parse_saved("A person's message")["harness"] == ""


@pytest.mark.unit
def test_a_cycle_takes_its_model_as_a_config_like_an_agents(
    store, tmp_path, monkeypatch
):
    """``refine_memory(memory_id, model={"provider": ..., "model": ...})``."""
    save(store, "codex", RUN_A, "The release codeword is PELICAN-4471.")
    model = Model(
        add("The codeword is PELICAN-4471.", RUN_A, "codeword is PELICAN-4471")
    )
    asked = []

    def create(config):
        asked.append(config)
        return model

    monkeypatch.setattr("memorizz.llms.llm_factory.create_llm_provider", create)
    refine = HarnessMemoryRefine(
        store,
        settings=RefineSettings(tmp_path / "settings.json"),
        model_factory=lambda settings: None,  # no model set for refining
    )

    report = refine.run("notes", model={"provider": "openai", "model": "small"})

    assert report["status"] == "done"
    assert asked == [{"provider": "openai", "model": "small"}]
    assert [f["text"] for f in refine.findings("notes")] == [
        "The codeword is PELICAN-4471."
    ]


@pytest.mark.unit
def test_the_model_is_told_to_keep_what_a_run_changed_and_whether_it_passed(
    store, tmp_path
):
    """An edit and the result of its check are the facts later runs, and the
    agents sharing the memory, most need."""
    record = handoff_memory_record(
        {
            "harness": "codex",
            "status": "succeeded",
            "run_id": RUN_A,
            "response": "Implemented the sale.",
            "files_changed": ["pricing.py"],
            "verification": {"verified": True, "command": "python check.py"},
        },
        workflow={},
        memory_id="notes",
        thread_id=f"harness-run-{RUN_A}",
        agent_id="metaharness",
        user_id=None,
    )
    store.store(
        record, memory_store_type=MemoryType.CONVERSATION_MEMORY, memory_id="notes"
    )
    model = Model()

    cycle(store, tmp_path, model).run("notes")

    prompt = model.prompts[0]
    assert "A change a run made to the project's files is a result" in prompt
    assert "Files changed: pricing.py" in prompt
    assert "Verification passed: python check.py" in prompt


@pytest.mark.unit
def test_a_threaded_run_is_still_handed_the_memorys_records(tmp_path):
    """Only conversations are kept by thread. A run given a thread of its own
    must still be handed the entity record the whole memory shares."""

    class Store:
        def __init__(self):
            self.asked = []

        def retrieve_by_query(
            self, query, memory_store_type=None, memory_id=None, limit=4, **scope
        ):
            self.asked.append((memory_store_type, scope.get("thread_id")))
            if memory_store_type is MemoryType.ENTITY_MEMORY:
                if scope.get("thread_id"):
                    return []  # a store asked for one thread leaves it out
                return [{"_id": "sale", "name": "Spring Sale", "memory_id": "team"}]
            return []

    provider = Store()

    pack = HarnessContextBuilder(provider).build(
        "the sale", memory_id="team", user_id=None, thread_id="pricing-brief"
    )

    assert [record["name"] for record in pack.records] == ["Spring Sale"]
    asked = dict(provider.asked)
    assert asked[MemoryType.CONVERSATION_MEMORY] == "pricing-brief"
    assert asked[MemoryType.ENTITY_MEMORY] is None


@pytest.mark.unit
def test_a_report_says_what_the_cycle_spent_and_on_which_model(
    store, tmp_path, monkeypatch
):
    """The model passed to a cycle is the one its report names, and the
    tokens of its call are counted, once."""

    class Chatty(Model):
        model = "small"

        def get_config(self):
            return {"provider": "openai", "model": "small"}

        def get_last_usage(self):
            return {"prompt_tokens": 900, "completion_tokens": 100}

        def generate(self, messages, tools=None):
            return self.reply(messages[0]["content"])

        def generate_text(self, prompt):
            # As a gateway model does: the text call is a chat call.
            return self.generate([{"role": "user", "content": prompt}])

        def reply(self, prompt):
            return Model.generate_text(self, prompt)

    finding = add("The codeword is PELICAN-4471.", RUN_A, "codeword is PELICAN-4471")
    save(store, "codex", RUN_A, "The release codeword is PELICAN-4471.")
    refine = HarnessMemoryRefine(
        store,
        settings=RefineSettings(tmp_path / "settings.json"),
        model_factory=lambda settings: None,
    )

    report = refine.run("notes", model=Chatty(finding))

    assert (report["status"], report["model"]) == ("done", "openai · small")
    assert (report["tokens"], report["tokens_estimated"]) == (1000, False)

    # A config is named the same way, and a cycle with no model names none.
    save(store, "pi", RUN_B, "The freeze starts on Thursday.")
    monkeypatch.setattr(
        "memorizz.llms.llm_factory.create_llm_provider",
        lambda config: Chatty(add("The freeze is Thursday.", RUN_B, "freeze starts")),
    )
    again = refine.run("notes", model={"provider": "openai", "model": "small"})
    assert (again["model"], again["tokens"]) == ("openai · small", 1000)
    save(store, "pi", RUN_C, "Another answer.")
    assert refine.run("notes")["model"] == ""


@pytest.mark.unit
def test_what_the_host_saw_a_run_do_is_a_finding_at_once(store, tmp_path):
    """Files changed and the check's result are observed, not claimed: they
    are kept when the run ends, with no model and no cycle."""
    refine = cycle(store, tmp_path, Model())
    built = {
        "harness": "codex",
        "run_id": RUN_A,
        "status": "succeeded",
        "response": "Implemented the sale.",
        "files_changed": ["pricing.py"],
        "verification": {"command": "python check.py", "verified": True},
    }

    refine.record_outcome("notes", None, built, "Apply the Spring Sale to pricing.py")

    (finding,) = refine.findings("notes")
    assert finding["text"] == (
        "codex changed pricing.py and its check passed. "
        "Asked: Apply the Spring Sale to pricing.py"
    )
    assert finding["harnesses"] == ["codex"] and finding["runs"] == [RUN_A]
    assert finding["quote"] == "Files changed: pricing.py"

    # A failed check is as much a fact as a passed one.
    broke = {
        **built,
        "run_id": RUN_B,
        "verification": {**built["verification"], "verified": False},
    }
    refine.record_outcome("notes", None, broke)
    assert "codex changed pricing.py and its check failed." in [
        f["text"] for f in refine.findings("notes")
    ]
    # A run that changed nothing and had no check leaves no outcome.
    read = {"harness": "pi", "run_id": RUN_C, "status": "succeeded", "response": "x"}
    assert refine.record_outcome("notes", None, read, "Read it") is None
    assert len(refine.findings("notes")) == 2
