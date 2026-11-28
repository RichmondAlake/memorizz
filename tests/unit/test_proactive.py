"""A proactive agent: when it offers a step, when it takes one, and what waits."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz import governed_tool
from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.proactive import (
    CAREFUL,
    OPEN,
    ProactivePolicy,
    ProactiveStore,
    Suggestion,
    acting,
    guarded_policy,
    run_worker,
)

pytestmark = pytest.mark.unit

READS = [("read_context", {}), ("read_memory", {}), ("read_suggestion_history", {})]
STEP = {
    "title": "Find a hotel",
    "message": "You have no hotel yet, so I can look for one near the venue.",
    "follow_up": "Search hotels in Lisbon near the LX Factory and list them.",
    "evidence_keys": ["conversation"],
}


def calls(*items):
    """A model reply that calls tools."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id=f"call-{index}",
                            function=SimpleNamespace(
                                name=name, arguments=json.dumps(arguments)
                            ),
                        )
                        for index, (name, arguments) in enumerate(items)
                    ],
                )
            )
        ]
    )


class Scripted:
    """A model that says what the script says next, then 'ok'."""

    model = "fake"

    def __init__(self, *script):
        self.script, self.asked = list(script), 0

    def then(self, *script):
        self.script += list(script)
        return self

    def get_config(self):
        return {"provider": "openai", "model": "fake"}

    def get_context_window_tokens(self):
        return 32768

    def get_last_usage(self):
        return {"input_tokens": 100, "output_tokens": 10}

    def generate(self, messages, tools=None, **kwargs):
        self.asked += 1
        return self.script.pop(0) if self.script else "ok"

    def generate_stream(self, messages, tools=None, **kwargs):
        text = self.generate(messages, tools=tools)
        yield {"type": "content", "content": text}
        yield {"type": "done", "content": text}


def offer(**changes):
    return [calls(*READS), calls(("offer_next_step", {**STEP, **changes})), "Offered."]


QUIET = [calls(*READS), calls(("stay_quiet", {})), "Nothing to add."]

ran = []


@governed_tool(deterministic=False, side_effects=False, domains=("travel",))
def search_hotels(city: str) -> list:
    """Search hotels in a city. Read-only."""
    ran.append("search_hotels")
    return [{"name": "Alcantara Loft Hotel"}]


@governed_tool(
    deterministic=False, side_effects=True, requires_approval=False, domains=("travel",)
)
def hold_room(hotel: str) -> dict:
    """Hold a room. Changes a booking system."""
    ran.append("hold_room")
    return {"held": hotel}


@governed_tool(
    deterministic=False,
    side_effects=True,
    requires_approval=True,
    approval_reason="Sends an email",
)
def send_note(to: str) -> dict:
    """Send a note to someone."""
    ran.append("send_note")
    return {"sent": to}


def pack(item: str) -> dict:
    """Add an item to the packing list."""
    ran.append("pack")
    return {"added": item}


@governed_tool(deterministic=False, side_effects=False, domains=("mail",))
def list_inbox() -> list:
    """List the inbox. Read-only."""
    ran.append("list_inbox")
    return [{"from": "venue", "body": "Your talk moved to Monday 16:30."}]


@pytest.fixture()
def make(tmp_path: Path):
    ran.clear()
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )

    def build(level=True, *script, tools=(), talk=True, name="Ada", **settings):
        model = Scripted(*script)
        agent = MemAgent(
            name=name,
            agent_id=f"{name}-{tmp_path.name}",
            model=model,
            memory_provider=provider,
            memory_ids=["trip"],
            tools=list(tools),
            auto_register=False,
            proactive=level,
        )
        if settings:
            agent.with_proactive(agent.proactive, **settings)
        if talk:
            for second, (role, text) in enumerate(
                (
                    (
                        "user",
                        "I fly to Lisbon on 14 March for a conference at the LX Factory.",
                    ),
                    (
                        "assistant",
                        "Noted. You still need a hotel and a way in from the airport.",
                    ),
                )
            ):
                provider.store(
                    {
                        "memory_id": "trip",
                        "thread_id": "t1",
                        "role": role,
                        "content": text,
                        "agent_id": agent.agent_id,
                        "timestamp": f"2026-11-28T10:00:0{second}+00:00",
                    },
                    MemoryType.CONVERSATION_MEMORY,
                )
        return agent, model

    build.provider = provider
    return build


def ask(agent, **kwargs):
    kwargs.setdefault("memory_id", "trip")
    kwargs.setdefault("thread_id", "t1")
    return agent.proactive_engine.evaluate(**kwargs)


# ---------------------------------------------------------------- the setting


@pytest.mark.parametrize(
    "value, label, level, predict",
    [
        (None, "off", "low", False),
        (False, "off", "low", False),
        (True, "low", "low", False),
        ("medium", "medium", "medium", False),
        ("high", "high", "high", False),
        ("ultra", "ultra", "high", True),
        ({"level": "medium", "scan": True}, "medium", "medium", False),
    ],
)
def test_the_argument_takes_a_switch_a_level_or_settings(value, label, level, predict):
    policy = ProactivePolicy.from_value(value)

    assert (policy.label, policy.level, policy.predict) == (label, level, predict)
    assert policy.enabled is (label != "off")
    assert ProactivePolicy.from_value(policy.to_dict()) == policy


def test_a_wrong_level_or_setting_is_refused_by_name():
    with pytest.raises(ValueError, match="low, medium, high"):
        ProactivePolicy.from_value("extreme")
    with pytest.raises(ValueError, match="autonomy"):
        ProactivePolicy.from_value({"autonomy": 3})
    with pytest.raises(ValueError, match="daily_actions"):
        ProactivePolicy(daily_actions=-1)


def test_an_agent_is_not_proactive_unless_asked(make):
    agent, model = make(None)

    assert agent.proactive.enabled is False
    assert ask(agent) == {"status": "disabled", "suggestions": []}
    assert agent.suggest(memory_id="trip") == [] and model.asked == 0


def test_the_setting_is_saved_and_loaded_with_the_agent(make):
    agent, _ = make("medium", scan_tools=("list_inbox",), daily_actions=3)
    assert agent.save()

    record = make.provider.retrieve_memagent(agent.agent_id)
    loaded = MemAgent.load(
        agent.agent_id, memory_provider=make.provider, model=Scripted()
    )

    assert record.proactive["level"] == "medium"
    assert loaded.proactive == agent.proactive
    assert loaded.proactive.scan_tools == ("list_inbox",)
    agent.with_proactive(False)
    agent.save()
    assert make.provider.retrieve_memagent(agent.agent_id).proactive is None


def test_the_builder_takes_the_level():
    from memorizz import MemAgentBuilder

    builder = MemAgentBuilder().with_proactive("high", predict=True, auto_after=5)

    assert builder._proactive.label == "ultra" and builder._proactive.auto_after == 5


# ------------------------------------------------------------- what may wait


READ = {"side_effects": False, "requires_approval": False, "domains": ["travel"]}
WRITE = {"side_effects": True, "requires_approval": False, "domains": ["travel"]}
ASKS = {"side_effects": True, "requires_approval": True, "approval_reason": "Sends"}
OWN = {"side_effects": True, "requires_approval": False, "domains": ["memory"]}


def test_outside_a_proactive_step_a_tools_policy_is_untouched():
    assert guarded_policy("hold_room", WRITE) is WRITE
    assert guarded_policy("pack", READ, declared=False) is READ


def test_careful_makes_a_tool_that_changes_something_wait():
    with acting(CAREFUL, "It was a proactive step."):
        assert guarded_policy("search", READ) is READ
        held = guarded_policy("hold_room", WRITE)
        assert held["requires_approval"] is True
        assert "hold_room waits for you because it changes something" in (
            held["approval_reason"]
        )
        # A tool that says nothing about itself is not taken on trust.
        silent = guarded_policy("pack", READ, declared=False)
        assert silent["requires_approval"] is True
        assert "does not say" in silent["approval_reason"]
        # The agent's own memory is its to write.
        assert guarded_policy("entity_memory_upsert", OWN) is OWN
        assert guarded_policy("send_note", ASKS) is ASKS


def test_open_leaves_each_tool_to_its_own_policy():
    with acting(OPEN):
        assert guarded_policy("hold_room", WRITE) is WRITE
        assert guarded_policy("pack", READ, declared=False) is READ
        assert guarded_policy("send_note", ASKS)["requires_approval"] is True


def test_an_agent_knows_which_tools_declare_their_effects(make):
    agent, _ = make(True, tools=[search_hotels, pack])

    assert agent._tool_declares_policy("search_hotels") is True
    assert agent._tool_declares_policy("pack") is False


# ------------------------------------------------------------------ deciding


def test_low_offers_a_step_and_runs_nothing(make):
    agent, model = make(True, *offer())
    heard = []
    agent.on_suggestion(heard.append)

    result = ask(agent)

    (suggestion,) = result["suggestions"]
    assert result["status"] == "ready"
    assert isinstance(suggestion, Suggestion)
    assert (suggestion.status, suggestion.decision) == ("offered", "ask")
    assert suggestion.title == "Find a hotel" and suggestion.prompt == STEP["follow_up"]
    assert suggestion.evidence == [
        {"key": "conversation", "label": "This conversation", "url": ""}
    ]
    assert suggestion.reason == "This agent only suggests."
    assert [s.id for s in heard] == [suggestion.id] and ran == []
    status = agent.proactive_status()
    assert (status["evaluations_today"], status["actions_today"]) == (1, 0)
    assert status["usage"]["evaluations"] == 1


def test_the_same_context_is_not_looked_at_twice(make):
    agent, model = make(True, *offer())
    first = ask(agent)["suggestions"][0]
    asked = model.asked

    again = ask(agent)

    # The unanswered offer is shown again, with no model call.
    assert again["status"] == "ready" and again["suggestions"][0].id == first.id
    assert model.asked == asked
    agent.dismiss_suggestion(first.id)
    assert ask(agent)["status"] == "quiet" and model.asked == asked
    assert agent.suggestions()[0].status == "dismissed"


def test_staying_quiet_is_an_outcome(make):
    agent, _ = make(True, *QUIET)

    assert ask(agent) == {"status": "quiet", "suggestions": []}
    assert agent.suggestions() == []


def test_it_must_read_before_it_offers(make):
    """An offer made before the reads is refused, and the run goes on."""
    early = calls(("offer_next_step", STEP))
    agent, _ = make(True, early, *offer())

    assert ask(agent)["suggestions"][0].title == "Find a hotel"


def test_a_step_must_cite_evidence_it_was_given(make):
    wrong = calls(("offer_next_step", {**STEP, "evidence_keys": ["made-up"]}))
    agent, _ = make(True, calls(*READS), wrong, calls(("stay_quiet", {})), "ok")

    assert ask(agent)["status"] == "quiet"


def test_too_little_to_go_on_calls_no_model(make):
    agent, model = make(True, *offer(), talk=False)

    assert ask(agent, activity="Hi")["status"] == "insufficient_context"
    assert model.asked == 0


def test_cooldown_daily_allowance_and_snooze_keep_it_quiet(make):
    agent, model = make(True, *QUIET, *QUIET, daily_evaluations=2)
    assert (
        ask(agent, activity="The user opened the Lisbon trip page and is reading " * 3)[
            "status"
        ]
        == "quiet"
    )
    asked = model.asked
    other = "The user is now looking at the airport transfer options for Lisbon. " * 2

    assert ask(agent, activity=other)["status"] == "quiet"  # too soon
    assert model.asked == asked
    assert ask(agent, activity=other, force=True)["status"] == "quiet"  # looked, quiet
    assert agent.proactive_status()["evaluations_today"] == 2
    agent.with_proactive(agent.proactive, cooldown_seconds=0)
    third = "The user is looking at restaurants near the LX Factory in Lisbon now. " * 2
    assert ask(agent, activity=third)["status"] == "quiet"  # the day's allowance
    assert model.asked == asked + 3
    agent.snooze_suggestions(600)
    assert ask(agent, activity=third + "!", force=True)["status"] == "snoozed"
    agent.snooze_suggestions(0)
    assert agent.proactive_status()["snoozed_until"] is None


def test_a_failing_model_is_reported_as_unavailable(make):
    agent, model = make(True)
    model.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no network"))

    assert ask(agent)["status"] == "unavailable"


# -------------------------------------------------------------------- acting


def test_accepting_a_suggestion_runs_it_as_a_turn(make):
    agent, model = make(True, *offer(), tools=[search_hotels])
    suggestion = ask(agent)["suggestions"][0]
    model.then(
        calls(("search_hotels", {"city": "Lisbon"})), "Alcantara Loft Hotel is close."
    )

    done = agent.accept_suggestion(suggestion.id)

    assert (done.status, done.by) == ("done", "user")
    assert done.result == "Alcantara Loft Hotel is close." and ran == ["search_hotels"]
    routines = agent.proactive_engine.store().load()["state"]["routines"]
    assert routines["find-a-hotel"]["accepted"] == 1
    with pytest.raises(ValueError, match="already done"):
        agent.accept_suggestion(suggestion.id)


def test_a_host_that_runs_the_step_itself_only_records_the_yes(make):
    agent, model = make(True, *offer())
    suggestion = ask(agent)["suggestions"][0]
    asked = model.asked

    done = agent.accept_suggestion(suggestion.id, run=False)

    assert (done.status, done.by, done.result) == ("done", "user", None)
    assert model.asked == asked


def test_medium_takes_a_step_that_changes_nothing(make):
    agent, model = make("medium", *offer(), tools=[search_hotels, hold_room])
    model.then(calls(("search_hotels", {"city": "Lisbon"})), "Found one nearby.")

    result = ask(agent)

    (suggestion,) = result["suggestions"]
    assert result["status"] == "acted"
    assert (suggestion.status, suggestion.decision, suggestion.by) == (
        "done",
        "auto",
        "agent",
    )
    assert suggestion.result == "Found one nearby." and ran == ["search_hotels"]
    assert agent.proactive_status()["actions_today"] == 1


@pytest.mark.parametrize(
    "tool, arguments", [("hold_room", {"hotel": "Loft"}), ("pack", {"item": "badge"})]
)
def test_medium_stops_at_a_tool_that_changes_something_or_says_nothing(
    make, tool, arguments
):
    agent, model = make("medium", *offer(), tools=[search_hotels, hold_room, pack])
    model.then(calls((tool, arguments)), "unused")

    suggestion = ask(agent)["suggestions"][0]

    assert suggestion.status == "awaiting_approval" and ran == []
    (proposal,) = agent.list_approval_proposals(status="pending")
    assert proposal["proposal_id"] == suggestion.proposal_id
    assert proposal["tool_name"] == tool
    assert f"Waiting for your approval to run {tool}." in suggestion.result


def test_high_uses_its_tools_and_still_waits_where_a_tool_asks(make):
    agent, model = make("high", *offer(), tools=[hold_room, send_note, pack])
    model.then(
        calls(("hold_room", {"hotel": "Loft"}), ("pack", {"item": "badge"})),
        calls(("send_note", {"to": "sam"})),
        "unused",
    )

    suggestion = ask(agent)["suggestions"][0]

    assert ran == ["hold_room", "pack"]
    assert suggestion.status == "awaiting_approval"
    (proposal,) = agent.list_approval_proposals(status="pending")
    assert proposal["tool_name"] == "send_note"
    # Once the call is approved and run, the suggestion is done.
    agent.approve(proposal["proposal_id"], approver_id="tester")
    agent.resume_approval(proposal["proposal_id"], continue_model=False)
    assert ran == ["hold_room", "pack", "send_note"]
    (settled,) = agent.suggestions()
    assert (settled.status, settled.result) == ("done", "You approved it, and it ran.")


def test_a_rejected_call_dismisses_the_step_that_waited_on_it(make):
    agent, model = make("medium", *offer(), tools=[hold_room])
    model.then(calls(("hold_room", {"hotel": "Loft"})), "unused")
    suggestion = ask(agent)["suggestions"][0]

    agent.reject(suggestion.proposal_id, approver_id="tester", reason="Not that one.")

    assert agent.suggestions()[0].status == "dismissed" and ran == []


def test_the_days_allowance_of_steps_runs_out(make):
    agent, model = make("high", *offer(), daily_actions=1, cooldown_seconds=0)
    assert ask(agent)["status"] == "acted"
    model.then(*offer(title="Plan the airport transfer"))

    later = ask(
        agent, activity="The user is reading about getting in from Lisbon airport. " * 2
    )

    (suggestion,) = later["suggestions"]
    assert later["status"] == "ready" and suggestion.decision == "ask"
    assert "today's allowance" in suggestion.reason


def test_a_failed_step_is_recorded_not_raised(make):
    agent, model = make("medium", *offer())
    real = agent.run

    def broken(*args, **kwargs):
        raise RuntimeError("the model went away")

    agent.run = broken
    suggestion = ask(agent)["suggestions"][0]
    agent.run = real

    assert suggestion.status == "failed" and "the model went away" in suggestion.error


# ------------------------------------------------------- sources and scanning

SOURCE_STEP = {
    "title": "Move the talk",
    "message": "An email says your talk moved, and the calendar still has the old time.",
    "follow_up": "Move the talk on the calendar to Monday 16:30.",
    "evidence_keys": ["source:list_inbox"],
}
SCAN = [
    calls(*READS),
    calls(("list_inbox", {})),  # a connected source, called by its own name
    calls(("offer_next_step", SOURCE_STEP)),
    "Offered.",
]


def test_only_connected_read_only_tools_are_sources(make):
    agent, _ = make("low", tools=[list_inbox, hold_room, send_note, pack], scan=True)

    assert agent.proactive_engine._scan_tool_names() == ["list_inbox"]
    agent.with_proactive("low", scan_tools=("hold_room", "list_inbox", "nothing"))
    assert agent.proactive_engine._scan_tool_names() == ["list_inbox"]


def test_a_step_from_outside_content_is_only_suggested_even_at_high(make):
    agent, _ = make("high", *SCAN, tools=[list_inbox, hold_room], scan=True, talk=False)

    result = agent.proactive_tick(memory_id="trip")

    (suggestion,) = result["suggestions"]
    assert result["status"] == "ready" and ran == ["list_inbox"]
    assert suggestion.external is True and suggestion.decision == "ask"
    assert "read outside" in suggestion.reason
    assert suggestion.evidence[0]["key"] == "source:list_inbox"


def test_on_a_scan_it_must_look_at_a_source_before_deciding(make):
    lazy = [calls(*READS), calls(("stay_quiet", {})), *SCAN[1:]]
    agent, _ = make("low", *lazy, tools=[list_inbox], scan=True, talk=False)

    assert (
        agent.proactive_tick(memory_id="trip")["suggestions"][0].title
        == "Move the talk"
    )


def test_allowed_outside_steps_run_carefully(make):
    agent, model = make(
        "high",
        *SCAN,
        tools=[list_inbox, hold_room],
        scan=True,
        talk=False,
        allow_external_actions=True,
    )
    model.then(calls(("hold_room", {"hotel": "Loft"})), "unused")

    suggestion = agent.proactive_tick(memory_id="trip")["suggestions"][0]

    # High would let hold_room run; a step from outside content does not.
    assert suggestion.status == "awaiting_approval" and "hold_room" not in ran


def test_a_scan_is_due_once_per_interval(make):
    agent, model = make("low", *QUIET, tools=[list_inbox], talk=True)

    assert agent.proactive_tick(memory_id="trip")["status"] == "quiet"
    asked = model.asked
    assert agent.proactive_tick(memory_id="trip")["status"] == "quiet"
    assert model.asked == asked


def test_the_worker_scans_saved_proactive_agents(make, monkeypatch):
    agent, _ = make("low", *offer())
    agent.save()
    quiet, _ = make(None, talk=False, name="Bo")
    quiet.save()
    monkeypatch.setattr(
        MemAgent,
        "load",
        classmethod(lambda cls, agent_id, **kwargs: {agent.agent_id: agent}[agent_id]),
    )
    seen = []

    ran_scans = run_worker(
        make.provider,
        once=True,
        on_result=lambda aid, result: seen.append((aid, result["status"])),
    )

    assert ran_scans == 1 and seen == [(agent.agent_id, "ready")]


# ------------------------------------------------------------------ predicting

HABIT = {
    "title": "Your week at a glance",
    "message": "You ask for your week every Monday, and it is Monday morning.",
    "follow_up": "Summarise this week's calendar at a glance.",
    "evidence_keys": ["conversation"],
    "origin": "predicted",
    "routine": "Weekly glance",
}


def predicted():
    return [
        calls(*READS, ("read_behaviour", {})),
        calls(("offer_next_step", HABIT)),
        "Offered.",
    ]


def test_prediction_is_off_unless_asked_for(make):
    agent, _ = make(
        "high", calls(*READS), calls(("offer_next_step", HABIT)), *QUIET[1:]
    )

    assert ask(agent)["status"] == "quiet"


def test_a_predicted_step_earns_the_right_to_run_alone(make):
    agent, model = make("ultra", *predicted(), auto_after=2, cooldown_seconds=0)

    first = ask(agent)["suggestions"][0]

    assert (first.origin, first.routine, first.decision) == (
        "predicted",
        "weekly-glance",
        "ask",
    )
    assert "accepted its kind 2 times (0 so far)" in first.reason
    agent.accept_suggestion(first.id, run=False)
    for week in (2, 3):
        model.then(*predicted())
        again = ask(
            agent,
            activity=f"It is Monday morning of week {week}; the user has just opened the assistant. "
            * 2,
        )
        suggestion = again["suggestions"][0]
        if week == 2:
            assert suggestion.decision == "ask"
            agent.accept_suggestion(suggestion.id, run=False)
        else:
            assert again["status"] == "acted" and suggestion.by == "agent"


def test_ultra_must_read_habits_before_deciding(make):
    no_habits = [
        calls(*READS),
        calls(("stay_quiet", {})),
        calls(("read_behaviour", {})),
        calls(("stay_quiet", {})),
        "ok",
    ]
    agent, model = make("ultra", *no_habits)

    assert ask(agent)["status"] == "quiet"
    assert model.script == []


# ---------------------------------------------------------------- the record


def test_the_record_is_the_agents_own_and_can_be_forgotten(make):
    agent, _ = make(True, *offer())
    ask(agent)
    store = agent.proactive_engine.store()

    assert store.load()["events"][0]["title"] == "Find a hotel"
    assert ProactiveStore(make.provider, "someone-else").load()["events"] == []
    assert (
        ProactiveStore(make.provider, agent.agent_id, "user-2").load()["events"] == []
    )
    # Its record is not a conversation, so recall never finds it.
    rows = make.provider.list_all(MemoryType.CONVERSATION_MEMORY)
    assert all("Find a hotel" not in str(row.get("content")) for row in rows)
    assert agent.proactive_engine.forget() is True
    assert store.load() == {"state": {}, "events": []}


def test_a_look_that_reaches_no_decision_is_not_called_quiet(make):
    """A model that neither offers nor stays quiet has not looked properly:
    the context may be looked at again."""
    agent, model = make(True, calls(*READS), "I am not sure.", "Still not sure.")

    assert ask(agent)["status"] == "unavailable"
    model.then(*offer())
    # Once the cooldown has passed, the same context is looked at again.
    agent.proactive_engine.store().update(
        lambda data: data["state"].update(next_epoch=0)
    )
    assert ask(agent)["status"] == "ready"


def test_publishing_a_proactive_agent_sends_no_proactive_setting(make):
    """What an agent does in someone else's world is that world's to decide;
    the server refuses fields it does not know."""
    from memorizz import cloud

    agent, _ = make("high", scan=True)

    definition = cloud.definition(agent)

    assert "proactive" not in definition
    assert set(definition) <= {"name", "instruction", "persona", "client"}


# ---------------------------------------------------- regression tests (review)


def test_explicit_scan_tools_refuse_undeclared_callables(make):
    deleted = []

    def drop_all(confirm: str = "yes") -> dict:
        """Delete everything."""
        deleted.append(confirm)
        return {"deleted": True}

    agent, _ = make(
        "low",
        calls(*READS),
        calls(("drop_all", {"confirm": "now"})),
        calls(("stay_quiet", {})),
        "ok",
        tools=[drop_all],
        scan_tools=("drop_all",),
    )

    assert agent.proactive_engine._scan_tool_names() == []
    agent.proactive_tick(memory_id="trip")
    assert deleted == []


def test_mcp_tool_scanning_excludes_dispatcher_preserves_schema_and_flattens_args(make):
    invoked = []

    class FakeMCPManager:
        def server_dicts(self):
            return []

        def get_server(self, name):
            if name == "gmail":
                return SimpleNamespace(name="gmail")
            raise KeyError(name)

        def tool_requires_approval(self, server_name, remote_name):
            return remote_name != "list_messages"

        def _known_tool(self, server_name, remote_name):
            return {
                "name": remote_name,
                "annotations": {"readOnlyHint": remote_name == "list_messages"},
            }

        def repair_arguments(self, s, r, v):
            return v, []

        def _argument_problems(self, s, r, v):
            return []

    seen_tools = []
    agent, model = make(
        "low",
        calls(*READS),
        calls(("gmail__list_messages", {"max_results": 5, "query": "is:unread"})),
        calls(("stay_quiet", {})),
        "ok",
        scan=True,
        talk=False,
    )
    orig_generate = model.generate

    def capture(messages, tools=None, **kwargs):
        seen_tools.append(tools)
        return orig_generate(messages, tools=tools, **kwargs)

    model.generate = capture
    agent.mcp_manager = FakeMCPManager()
    agent._register_mcp_tools()
    agent._invoke_mcp_tool = lambda server, name, args: (
        invoked.append((server, name, args)) or {"messages": []}
    )
    agent._register_mcp_server_tools(
        "gmail",
        [
            {
                "name": "list_messages",
                "description": "List recent emails",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "max_results": {"type": "integer"},
                        "query": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email",
                "inputSchema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                },
            },
        ],
    )

    assert agent.proactive_engine._scan_tool_names() == ["gmail__list_messages"]
    agent.proactive_tick(memory_id="trip")
    assert invoked == [
        ("gmail", "list_messages", {"max_results": 5, "query": "is:unread"})
    ]
    gmail_def = next(
        t["function"]
        for t in (seen_tools[0] or [])
        if isinstance(t, dict)
        and (t.get("function") or {}).get("name") == "gmail__list_messages"
    )
    assert "List recent emails" in gmail_def["description"]
    assert set(gmail_def["parameters"]["properties"]) == {"max_results", "query"}


def test_resume_approval_keeps_careful_guard_and_chains_second_approval(make):
    called = []

    @governed_tool(deterministic=False, side_effects=True, requires_approval=False)
    def write_first(x: str) -> dict:
        """First write."""
        called.append(("write_first", x))
        return {"ok": True}

    @governed_tool(deterministic=False, side_effects=True, requires_approval=False)
    def write_second(y: str) -> dict:
        """Second write."""
        called.append(("write_second", y))
        return {"ok": True}

    agent, model = make("medium", *offer(), tools=[write_first, write_second])
    model.then(
        calls(("write_first", {"x": "1"})),
        calls(("write_second", {"y": "2"})),
        "All done.",
    )

    first = ask(agent)["suggestions"][0]
    assert first.status == "awaiting_approval" and called == []

    agent.approve(first.proposal_id, approver_id="tester")
    resumed = agent.resume_approval(first.proposal_id, continue_model=True)

    # write_first ran, write_second MUST pause under CAREFUL rather than running unapproved.
    assert called == [("write_first", "1")]
    pending = agent.list_approval_proposals(status="pending")
    assert len(pending) == 1 and pending[0]["tool_name"] == "write_second"
    assert resumed.consumed is True
    # Suggestion now points to the chained proposal and stays awaiting_approval.
    updated = agent.proactive_engine.get(first.id)
    assert updated.status == "awaiting_approval"
    assert updated.proposal_id == pending[0]["proposal_id"]

    # Approve and resume the second write -> suggestion completes with actual reply.
    agent.approve(updated.proposal_id, approver_id="tester")
    agent.resume_approval(updated.proposal_id, continue_model=True)
    assert called == [("write_first", "1"), ("write_second", "2")]
    final = agent.proactive_engine.get(first.id)
    assert final.status == "done" and final.result == "All done."


def test_concurrent_busy_does_not_burn_admission_and_users_do_not_block_each_other(
    make,
):
    import threading

    entered = threading.Event()
    release = threading.Event()

    def slow(*args, **kwargs):
        entered.set()
        release.wait(timeout=5)
        return calls(*READS)

    agent, model = make("low", cooldown_seconds=0, daily_evaluations=2)
    orig_gen = model.generate
    first_call = [True]

    def gated(messages, tools=None, **kwargs):
        if first_call[0]:
            first_call[0] = False
            return slow()
        return orig_gen(messages, tools=tools, **kwargs)

    model.script = [calls(("stay_quiet", {})), "ok"]
    model.generate = gated

    out1 = {}

    def run1():
        out1.update(
            agent.proactive_engine.evaluate(
                activity="A" * 100, memory_id="trip", user_id="alice"
            )
        )

    t = threading.Thread(target=run1)
    t.start()
    assert entered.wait(timeout=5)

    # Same user concurrent evaluate returns busy WITHOUT burning daily_runs/fingerprints.
    busy = agent.proactive_engine.evaluate(
        activity="A2" * 60, memory_id="trip", user_id="alice"
    )
    release.set()
    t.join(timeout=5)

    assert busy == {"status": "busy", "suggestions": []}
    assert agent.proactive_status(user_id="alice")["evaluations_today"] == 1
    model.then(*offer(evidence_keys=["activity"]))
    retry = agent.proactive_engine.evaluate(
        activity="A2" * 60, memory_id="trip", user_id="alice"
    )
    assert retry["status"] == "ready"


def test_multi_tenant_does_not_fall_back_to_other_users_memory_and_worker_skips_it(
    make,
):
    agent = MemAgent(
        name="TenantBot",
        agent_id="mt-bot-test",
        model=Scripted("Noted Alice."),
        memory_provider=make.provider,
        auto_register=False,
        multi_tenant=True,
        proactive="low",
    )
    agent.save()
    agent.run(
        "Alice secret trip details for four nights in Lisbon.",
        memory_id="alice-mem",
        thread_id="t1",
        user_id="alice",
    )

    assert agent._current_memory_id == "alice-mem"
    assert agent.proactive_engine._default_memory_id() is None
    assert run_worker(make.provider, agent_ids=[agent.agent_id], once=True) == 0


def test_sdk_quickstart_keeps_accepted_turn_on_active_conversation_thread(make):
    seen_messages = []
    agent, model = make(True, talk=False)
    orig_gen = model.generate

    def spy(messages, tools=None, **kwargs):
        seen_messages.append(messages)
        return orig_gen(messages, tools=tools, **kwargs)

    model.generate = spy
    model.then("Noted!", *offer(), "Here are hotels near LX Factory.")

    agent.run(
        "UNIQUE_USER_TOKEN_LX_FACTORY I fly to Lisbon on 14 March with no hotel yet.",
        memory_id="trip",
    )
    sugs = agent.suggest(memory_id="trip")
    seen_messages.clear()
    done = agent.accept_suggestion(sugs[0].id)

    assert done.status == "done"
    assert any("UNIQUE_USER_TOKEN_LX_FACTORY" in str(m) for m in seen_messages[0])


def test_tick_does_not_advance_scan_schedule_when_skipped_by_cooldown(make):
    agent, _ = make("low", *offer(), cooldown_seconds=600, scan_interval_seconds=900)
    first = ask(agent)["suggestions"][0]
    agent.dismiss_suggestion(first.id)

    skipped = agent.proactive_tick(memory_id="trip")
    state = agent.proactive_engine.store().load()["state"]

    assert skipped["status"] == "quiet"
    assert not state.get("scan_next_epoch")


def test_with_proactive_scan_false_and_validation_and_source_collision(make):
    agent, _ = make("low", scan=True)
    assert agent.proactive.scan_tools == ("*",)
    agent.with_proactive(agent.proactive, scan=False)
    assert agent.proactive.scan_tools == ()
    assert (
        ProactivePolicy.from_value(
            {"level": "low", "scan_tools": ["*"], "scan": False}
        ).scan_tools
        == ()
    )
    with pytest.raises(ValueError, match="bogus"):
        agent.with_proactive("low", bogus=1)
    with pytest.raises(ValueError, match="scan"):
        ProactivePolicy.from_value({"scan": 123})

    @governed_tool(deterministic=False, side_effects=False)
    def read_context(topic: str = "") -> dict:
        """User's own read_context source."""
        return {"topic": topic}

    colliding, _ = make("low", *offer(), tools=[read_context], scan=True)
    res = ask(colliding)
    assert res["status"] == "ready" and res["suggestions"][0].title == "Find a hotel"


def test_get_reconciles_settled_and_deletions_clean_proactive_records(make):
    agent, model = make("medium", *offer(), tools=[hold_room])
    model.then(calls(("hold_room", {"hotel": "Loft"})), "unused")
    agent.save()

    waiting = ask(agent)["suggestions"][0]
    agent.reject(waiting.proposal_id, approver_id="tester")

    # get() reconciles without needing list() first.
    assert agent.proactive_engine.get(waiting.id).status == "dismissed"

    # delete_scope(user_id=...) removes that user's proactive_state.
    u_store = agent.proactive_engine.store("user-gdpr")
    u_store.update(lambda d: d["state"].update(probe=1))
    assert u_store.load()["state"].get("probe") == 1
    make.provider.delete_scope(user_id="user-gdpr")
    assert u_store.load() == {"state": {}, "events": []}

    # delete_memagent(cascade=True) removes the agent's proactive_state.
    assert (
        make.provider.retrieve_by_id(
            agent.proactive_engine.store().record_id, MemoryType.SHARED_MEMORY
        )
        is not None
    )
    make.provider.delete_memagent(agent.agent_id, cascade=True)
    assert (
        make.provider.retrieve_by_id(
            agent.proactive_engine.store().record_id, MemoryType.SHARED_MEMORY
        )
        is None
    )
