"""Prompt-cache reuse across turns, tool loops, harness runs and the UI.

Provider behaviour pinned here was verified against the live APIs:
GPT-5.6/6 read cache only from explicit breakpoints on input content (not
assistant output) and find an earlier entry by lookback from a later
breakpoint; Anthropic reads at ``cache_control`` breakpoints.
"""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from jinja2 import Environment, FileSystemLoader

from memorizz.benchmarks.measurement import summarize_calls
from memorizz.llms.anthropic import Anthropic
from memorizz.llms.openai import OpenAI
from memorizz.metaharness.adapters import NativeMemAgentHarness
from memorizz.metaharness.service import MetaHarness
from memorizz.observability.prompt_cache import add_call_usage

BREAKPOINT = {"mode": "explicit"}
TEMPLATES = Path(__file__).parents[2] / "src/memorizz/ui/templates"


def _openai(model):
    provider = OpenAI.__new__(OpenAI)
    provider.model = model
    provider.base_url = None
    provider._prompt_cache_key, provider._prompt_cache_retention = "thread", None
    return provider


def _conversation(turns, final="ctx V\nnext question"):
    messages = [{"role": "system", "content": "stable instructions"}]
    for index in range(turns):
        messages += [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index}"},
        ]
    return messages + [{"role": "user", "content": final}]


def _marked(items):
    return [
        index
        for index, item in enumerate(items)
        if isinstance(item.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("prompt_cache_breakpoint")
            for block in item["content"]
        )
    ]


@pytest.mark.parametrize("field", ["messages", "input"])
def test_gpt56_history_anchor_sits_on_last_prior_user_message(field):
    provider = _openai("gpt-5.6-luna")
    messages = _conversation(turns=2)
    before = deepcopy(messages)
    kwargs = {field: messages}

    provider._apply_cache_options(kwargs)

    assert messages == before
    # Instructions (0) and the last user message before the volatile one (3).
    assert _marked(kwargs[field]) == [0, 3]
    text_type = "input_text" if field == "input" else "text"
    assert kwargs[field][3]["content"] == [
        {
            "type": text_type,
            "text": "question 1",
            "prompt_cache_breakpoint": BREAKPOINT,
        }
    ]
    # Assistant output is never a read point on these models.
    assert all(kwargs[field][i]["role"] != "assistant" for i in _marked(kwargs[field]))
    assert provider._last_prompt_cache_metadata["prompt_cache_enabled"] is True


def test_gpt56_first_turn_marks_instructions_only():
    provider = _openai("gpt-6-sol")
    kwargs = {"messages": _conversation(turns=0)}
    provider._apply_cache_options(kwargs)
    assert _marked(kwargs["messages"]) == [0]


def test_gpt56_anchor_is_stable_through_a_tool_loop_and_advances_per_turn():
    provider = _openai("gpt-5.6-sol")
    turn = _conversation(turns=2)
    first = {"messages": turn}
    provider._apply_cache_options(first)
    loop = turn + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ]
    iteration = {"messages": loop}
    provider._apply_cache_options(iteration)
    # Same anchor on every iteration, so the request prefix never shifts.
    assert _marked(iteration["messages"]) == _marked(first["messages"]) == [0, 3]
    assert iteration["messages"][: len(turn)] == first["messages"]

    next_turn = {"messages": _conversation(turns=3)}
    provider._apply_cache_options(next_turn)
    # Two items after the previous anchor: within the provider's lookback.
    assert _marked(next_turn["messages"]) == [0, 5]


@pytest.mark.parametrize("model", ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.5"])
def test_automatic_prefix_models_get_no_explicit_breakpoints(model):
    provider = _openai(model)
    messages = _conversation(turns=2)
    kwargs = {"messages": messages}
    provider._apply_cache_options(kwargs)
    assert kwargs["messages"] is messages
    assert _marked(messages) == []


def _anthropic():
    provider = Anthropic.__new__(Anthropic)
    provider.model = "claude-sonnet-5"
    provider.context_window_tokens = 200_000
    provider._last_usage = None
    provider._max_tokens = 256
    provider._enable_prompt_caching = True
    provider._request_options = {}
    provider.client = MagicMock()
    return provider


def _with_skill(skill):
    messages = _conversation(turns=1)
    return messages[:-1] + [{"role": "developer", "content": skill}, messages[-1]]


def test_anthropic_skill_changes_do_not_move_the_stable_system_breakpoint():
    provider = _anthropic()
    first = provider._build_request_kwargs(_with_skill("skill A"))
    second = provider._build_request_kwargs(_with_skill("a different skill B"))

    assert (
        first["system"][0]
        == second["system"][0]
        == {
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral"},
        }
    )
    assert first["system"][1] == {"type": "text", "text": "skill A"}
    assert provider._count_cache_control_breakpoints(first) == 3


def test_anthropic_without_trailing_instructions_keeps_single_system_block():
    request = _anthropic()._build_request_kwargs(_conversation(turns=1))
    assert request["system"] == [
        {
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_anthropic_stable_boundary_respects_a_full_caller_budget():
    marker = {"type": "ephemeral"}
    messages = _with_skill("skill")
    for index in (1, 2, 4):
        messages[index] = {
            **messages[index],
            "content": [
                {
                    "type": "text",
                    "text": messages[index]["content"],
                    "cache_control": marker,
                }
            ],
        }
    tool = {
        "type": "function",
        "function": {"name": "f", "parameters": {"type": "object"}},
        "cache_control": marker,
    }
    request = _anthropic()._build_request_kwargs(messages, tools=[tool])
    assert _anthropic()._count_cache_control_breakpoints(request) == 4
    assert "cache_control" not in request["system"][0]


def test_anthropic_single_shot_text_caches_instructions_not_the_prompt():
    provider = _anthropic()
    provider.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], usage=None
    )
    provider.generate_text("unique case evidence", instructions="rubric")
    request = provider.client.messages.create.call_args.kwargs
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["messages"] == [{"role": "user", "content": "unique case evidence"}]


def test_run_usage_counts_unreported_cache_as_unknown_not_a_miss():
    totals = {}
    add_call_usage(totals, {"input_tokens": 1000, "output_tokens": 5})
    add_call_usage(
        totals,
        {
            "input_tokens": 1000,
            "output_tokens": 5,
            "cached_tokens": 900,
            "cache_write_tokens": 50,
        },
    )
    add_call_usage(
        totals, {"input_tokens": 1000, "output_tokens": 5, "cached_tokens": 0}
    )
    assert totals["calls"] == 3
    assert totals["input_tokens"] == 3000
    assert totals["cache_reported_calls"] == 2
    assert totals["cache_hit_calls"] == 1
    assert totals["measured_input_tokens"] == 2000
    assert totals["read_percent"] == 45.0
    assert totals["cache_write_tokens"] == 50


def test_memagent_run_usage_sums_every_call_and_resets_per_run(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.model.get_last_response_metadata = lambda: {}
    current = {}
    agent.model.get_last_usage = lambda: current
    agent._begin_trace_turn("alice", emit_start=False)
    for iteration, usage in enumerate(
        [
            {"prompt_tokens": 1000, "completion_tokens": 10, "cached_tokens": 0},
            {"prompt_tokens": 1200, "completion_tokens": 20, "cached_tokens": 1000},
        ],
        start=1,
    ):
        span, started = agent._start_model_trace(iteration=iteration, stage="loop")
        current.clear()
        current.update(usage)
        agent._finish_model_trace(span, started, iteration=iteration, stage="loop")
    totals = agent.get_last_run_usage()
    assert totals["calls"] == 2
    assert totals["input_tokens"] == 2200
    assert totals["output_tokens"] == 30
    assert totals["cache_hit_calls"] == 1
    assert totals["read_percent"] == round(100 * 1000 / 2200, 1)

    agent._begin_trace_turn("alice", emit_start=False)
    span, started = agent._start_model_trace(iteration=1, stage="loop")
    agent._finish_model_trace(span, started, iteration=1, stage="loop")
    assert agent.get_last_run_usage()["calls"] == 1


def test_streamed_run_done_reports_turn_cache_usage(memagent_with_mocks, monkeypatch):
    agent = memagent_with_mocks
    monkeypatch.setattr(agent, "_build_llm_tools", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(agent, "_record_interaction", lambda *a, **k: None)

    def provider(*a, **k):
        yield {"type": "content", "content": "hello"}
        yield {"type": "done", "content": "hello"}

    agent.model.generate_stream = provider
    agent.model.get_last_response_metadata = lambda: {
        "input_tokens": 4000,
        "output_tokens": 12,
        "cached_tokens": 3600,
        "cache_write_tokens": 200,
    }
    events = list(agent.run_stream_events("hello"))
    usage = events[-1]["usage"]
    assert events[-1]["type"] == "run.done"
    assert usage["calls"] == 1
    assert usage["cache_hit_calls"] == 1
    assert usage["read_percent"] == 90.0
    assert usage["cache_write_tokens"] == 200


def test_failed_tool_events_carry_a_safe_reason_code_but_no_prose():
    import queue

    from memorizz.streaming import _Session

    sink = queue.Queue()
    session = _Session(None, sink, "final_stream", {"run_id": "r1"})
    base = {"type": "trace", "tool_name": "lookup_order", "span_id": "s1"}
    session.callback({**base, "trace_kind": "tool_call"})
    session.callback(
        {
            **base,
            "trace_kind": "tool_result",
            "status": "error",
            "outcome_reason_code": "invalid_tool_invocation",
            "error": "Tool 'lookup_order' has no trusted callable binding",
        }
    )
    session.callback(
        {**base, "trace_kind": "tool_result", "status": "success", "span_id": "s2"}
    )
    started, failed, succeeded = (sink.get_nowait().to_dict() for _ in range(3))
    assert started["type"] == "tool.started" and "reason_code" not in started
    assert failed["type"] == "tool.completed"
    assert failed["reason_code"] == "invalid_tool_invocation"
    assert "trusted callable" not in json.dumps(failed)
    assert "reason_code" not in succeeded


def test_native_harness_reports_whole_run_usage_so_budgets_apply():
    agent = SimpleNamespace(
        agent_id="agent",
        model=SimpleNamespace(get_last_usage=lambda: {"prompt_tokens": 10}),
        run=lambda *a, **k: "done",
        get_last_run_usage=lambda: {
            "calls": 3,
            "input_tokens": 9000,
            "output_tokens": 300,
            "cached_tokens": 6000,
        },
    )
    task = SimpleNamespace(
        run_id="run",
        task="task",
        memory_id=None,
        thread_id=None,
        user_id=None,
        context={},
        budget=SimpleNamespace(
            max_input_tokens=5000, max_output_tokens=None, max_cost_usd=None
        ),
    )
    outcome = NativeMemAgentHarness(agent).run(
        task,
        workspace=Path("."),
        context_pack=SimpleNamespace(rendered="", source_ids=[]),
        emit=lambda event: None,
        cancel_event=SimpleNamespace(is_set=lambda: False),
    )
    assert outcome.usage["input_tokens"] == 9000
    assert outcome.usage["cached_tokens"] == 6000
    MetaHarness._apply_reported_budget(task, outcome)
    assert outcome.error_code == "input_token_budget_exceeded"


def test_comparison_lanes_report_cache_hit_calls():
    call = dict(
        case_id="c",
        lane="reader",
        provider="openai",
        model="m",
        seconds=1.0,
        status="completed",
        pricing=None,
        cost_usd=0.0,
    )
    summary = summarize_calls(
        [
            {
                **call,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "cached_tokens": 64,
                },
            },
            {
                **call,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                },
            },
            {**call, "usage": {"prompt_tokens": 100, "completion_tokens": 1}},
        ]
    )
    assert summary["lanes"]["reader"]["cache_reported_calls"] == 2
    assert summary["lanes"]["reader"]["cache_hit_calls"] == 1


def _render(name, **context):
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    return env.get_template(name).render(**context)


def test_evalground_shows_cache_rates_and_legacy_runs_render():
    usage = {
        "generation_prompt_tokens": 8000,
        "generation_cached_tokens": 6000,
        "generation_cache_write_tokens": 1500,
        "generation_calls": 4,
        "generation_cache_reported_calls": 4,
        "generation_cache_hit_calls": 3,
        "judge_prompt_tokens": 2000,
        "judge_cached_tokens": 0,
        "judge_cache_write_tokens": 0,
        "judge_calls": 4,
        "judge_cache_reported_calls": 4,
        "judge_cache_hit_calls": 0,
        "lane_cost_usd": {},
    }
    html = _render("_eval_usage.html", eval_results={"usage": usage})
    assert "Provider prompt cache" in html
    assert "60.0%" in html  # 6000 of 10000 input tokens
    assert "37.5%" in html  # 3 of 8 calls
    assert "75.0% <small>(3/4)</small>" in html

    legacy = _render(
        "_eval_usage.html",
        eval_results={
            "usage": {"generation_prompt_tokens": 100, "generation_cached_tokens": 0}
        },
    )
    assert "predates per-call cache tracking" in legacy


def _router(sticky_limit):
    from memorizz.tooling import SemanticToolRouter

    names = ["book_flight", "cancel_order", "lookup_order", "weather", "hotel"]
    manager = SimpleNamespace(
        get_tool_metadata=lambda: [
            {"name": name, "description": name.replace("_", " "), "parameters": {}}
            for name in names
        ]
    )
    router = SemanticToolRouter(manager, top_k=1, sticky_limit=sticky_limit)
    router.preview = lambda query, **_: [query]  # deterministic relevance
    return router


def _turn(router, query, scope):
    router.begin_turn(user_id="alice")
    schemas = router.schemas_for_turn(query, user_id="alice", cache_scope=scope)
    return [schema["function"]["name"] for schema in schemas]


def test_sticky_tools_keep_the_cached_tool_prefix_stable():
    import uuid

    scope = uuid.uuid4().hex
    router = _router(sticky_limit=3)
    first = _turn(router, "lookup_order", scope)
    second = _turn(router, "weather", scope)
    third = _turn(router, "lookup_order", scope)
    fourth = _turn(router, "weather", scope)
    # Growth changes the prefix once; returning to known tools does not.
    assert "lookup_order" in second and "weather" in second
    assert third == fourth == second
    assert first != second
    # Sticky tools are visible, so the dispatch gate allows them this turn.
    assert {"lookup_order", "weather"} <= set(router.checkpoint_state()["selected"])
    # A new router instance (hosts rebuild agents per request) stays warm.
    assert _turn(_router(sticky_limit=3), "weather", scope) == second
    # A different scope (another agent) keeps its own disclosure.
    assert "weather" not in _turn(router, "lookup_order", uuid.uuid4().hex)


def test_sticky_disclosure_is_bounded_and_can_be_disabled():
    import uuid

    scope = uuid.uuid4().hex
    router = _router(sticky_limit=2)
    for query in ("lookup_order", "weather"):
        _turn(router, query, scope)
    # A third tool exceeds the bound: evict the least recently selected.
    visible = _turn(router, "hotel", scope)
    assert "lookup_order" not in visible and "weather" in visible
    plain = _router(sticky_limit=0)
    _turn(plain, "lookup_order", scope)
    assert "lookup_order" not in _turn(plain, "weather", scope)


def test_openai_single_shot_calls_get_a_stable_instruction_routing_key():
    provider = _openai("gpt-4.1-mini")
    provider._prompt_cache_key = None
    first = {"input": "evidence A? question 1", "instructions": "Return JSON only."}
    second = {"input": "evidence B? question 2", "instructions": "Return JSON only."}
    provider._apply_cache_options(first)
    provider._apply_cache_options(second)
    # Live: Responses requests without a key did not reuse a shared prefix.
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["prompt_cache_key"].startswith("memorizz:")
    other = {"input": "q", "instructions": "Different rubric."}
    provider._apply_cache_options(other)
    assert other["prompt_cache_key"] != first["prompt_cache_key"]
    # Nothing reusable to route by, an explicit key, or a local server: unchanged.
    bare = {"input": "q"}
    provider._apply_cache_options(bare)
    assert "prompt_cache_key" not in bare
    provider.set_prompt_cache_key("memorizz:agent:memory")
    pinned = {"input": "q", "instructions": "Return JSON only."}
    provider._apply_cache_options(pinned)
    assert pinned["prompt_cache_key"] == "memorizz:agent:memory"
    provider.base_url = "http://127.0.0.1:8080/v1"
    local = {"messages": [{"role": "system", "content": "s"}]}
    provider._apply_cache_options(local)
    assert "prompt_cache_key" not in local


def test_anthropic_calls_are_priced_including_cache_reads_and_writes():
    from memorizz.observability.pricing import DEFAULT_PRICING

    quote = DEFAULT_PRICING.quote(
        {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "input_tokens": 20_000,
            "cached_tokens": 15_000,
            "cache_write_tokens": 4_000,
            "output_tokens": 100,
        }
    )
    # 1k uncached x $2 + 15k read x $0.20 + 4k written x $2.50 + 100 out x $10
    assert quote["cost_status"] == "calculated"
    assert quote["cost_usd"] == "0.016"
    dated = DEFAULT_PRICING.quote(
        {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "input_tokens": 1_000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 10,
        }
    )
    assert dated["cost_status"] == "calculated"
    long_context = DEFAULT_PRICING.quote(
        {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "input_tokens": 250_000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 10,
        }
    )
    assert long_context["cost_status"] == "unknown"


@pytest.mark.parametrize(
    "provider,model,accepted",
    [
        ("openai", "gpt-4.1-mini", True),
        ("openai", "gpt-5.6-luna", False),
        ("openai", "o4-mini", False),
        ("anthropic", "claude-haiku-4-5", True),
        ("anthropic", "claude-opus-4-6", True),
        ("anthropic", "claude-sonnet-5", False),
        ("anthropic", "claude-opus-5-5", False),
        ("anthropic", "claude-fable-5-1", False),
        ("ollama", "qwen2.5:3b", True),
    ],
)
def test_eval_defaults_omit_temperature_for_models_that_reject_it(
    provider, model, accepted
):
    from memorizz.benchmarks.measurement import accepts_temperature

    assert accepts_temperature(provider, model) is accepted


def test_repeated_single_shot_prompts_can_opt_into_body_caching():
    provider = _anthropic()
    provider._cache_single_shot_prompts = True
    provider.client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")], usage=None
    )
    provider.generate_text("identical evaluation prompt", instructions="rubric")
    request = provider.client.messages.create.call_args.kwargs
    assert request["messages"][0]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


def test_repeated_cache_drops_are_grouped_with_a_count():
    from memorizz.observability.prompt_cache import _group_warnings

    warning = {"model": "anthropic / m", "code": "prefix_changed", "message": "x"}
    grouped = _group_warnings([dict(warning, event_id=str(i)) for i in range(5)])
    assert len(grouped) == 1 and grouped[0]["count"] == 5
    assert grouped[0]["event_id"] == "0"


def test_streamed_turns_without_thread_id_continue_the_same_thread(
    memagent_with_mocks, monkeypatch
):
    agent = memagent_with_mocks
    monkeypatch.setattr(agent, "_build_llm_tools", lambda *a, **k: [])
    monkeypatch.setattr(agent, "_build_context", lambda *a, **k: {})
    monkeypatch.setattr(agent, "_record_interaction", lambda *a, **k: None)

    def provider(*a, **k):
        yield {"type": "content", "content": "ok"}
        yield {"type": "done", "content": "ok"}

    agent.model.generate_stream = provider
    first = list(agent.run_stream_events("one", memory_id="memory-1"))
    second = list(agent.run_stream_events("two", memory_id="memory-1"))
    thread = first[0]["thread_id"]
    assert second[0]["thread_id"] == thread
    assert agent._known_thread_id("memory-1") == thread
    other = list(agent.run_stream_events("three", memory_id="memory-2"))
    assert other[0]["thread_id"] != thread
    agent.reset_thread_state()
    assert agent._known_thread_id("memory-1") is None
