"""Requested aliases and provider-reported IDs survive the full trace path."""

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from memorizz import MemAgent
from memorizz.enums import MemoryType
from memorizz.llms.response_metadata import response_metadata
from memorizz.llms.streaming import ProviderStreamError
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability import (
    ObservabilityStore,
    TraceSnapshot,
    compare_trace_windows,
    normalize_trace_snapshot,
)
from memorizz.observability.lineage import build_lineage_inspectors
from memorizz.observability.models import TraceEventV3
from memorizz.observability.sql_index import SQLiteSpanIndex

REQUESTED = "fixture-latest"
RETURNED = "fixture-2026-09-17"
KINDS = ["openai", "responses", "azure", "anthropic", "ollama"]


def provider_fixture(kind):
    from memorizz.llms.anthropic import Anthropic
    from memorizz.llms.azure import AzureOpenAI
    from memorizz.llms.ollama import OllamaLLM
    from memorizz.llms.openai import OpenAI

    cls = {
        "openai": OpenAI,
        "responses": OpenAI,
        "azure": AzureOpenAI,
        "anthropic": Anthropic,
        "ollama": OllamaLLM,
    }[kind]
    provider = cls.__new__(cls)
    provider.model = REQUESTED
    provider.context_window_tokens = 128000
    provider._request_options = {}
    provider._prompt_cache_key = provider._prompt_cache_retention = None
    provider.base_url = None
    provider.api_mode = "responses" if kind == "responses" else "chat_completions"
    provider.reasoning_effort = None
    provider._max_tokens = 128
    provider._enable_prompt_caching = False
    provider._chat_kwargs = lambda **kwargs: {"model": provider.model, **kwargs}
    provider.get_config = lambda: {"provider": kind, "model": provider.model}
    wire = Mock()
    provider.client = NS(
        responses=NS(create=wire),
        messages=NS(create=wire, stream=wire),
        chat=Mock(completions=NS(create=wire), side_effect=wire),
    )
    return provider, wire


def regular_response(kind, model):
    fields = {"id": "request-fixture", "model": model, "usage": None}
    if kind == "responses":
        return NS(**fields, status="completed", output=[], output_text="answer")
    if kind == "anthropic":
        return NS(
            **fields, stop_reason="end_turn", content=[NS(type="text", text="answer")]
        )
    if kind == "ollama":
        return NS(**fields, message=NS(content="answer", tool_calls=None))
    return NS(
        **fields,
        choices=[
            NS(finish_reason="stop", message=NS(content="answer", tool_calls=None))
        ],
    )


def stream_response(kind, model, *, complete=True):
    # Only the initial event includes a model: later chunks must preserve it.
    if kind == "responses":
        events = [
            NS(type="response.created", response=NS(model=model, id="request-fixture")),
            NS(type="response.output_text.delta", delta="answer"),
        ]
        if complete:
            events.append(
                NS(
                    type="response.completed",
                    response=NS(status="completed", usage=None),
                )
            )
    elif kind == "anthropic":
        events = [
            NS(type="message_start", message=NS(model=model, id="request-fixture")),
            NS(type="content_block_delta", delta=NS(type="text_delta", text="answer")),
        ]
        if complete:
            events.extend(
                [
                    NS(type="message_delta", delta=NS(stop_reason="end_turn")),
                    NS(type="message_stop"),
                ]
            )
    elif kind == "ollama":
        events = [{"model": model, "message": {"content": "answer"}}]
        if complete:
            events.append({"message": {}, "done": True, "done_reason": "stop"})
    else:
        events = [
            NS(
                model=model,
                choices=[
                    NS(delta=NS(content="answer", tool_calls=None), finish_reason=None)
                ],
            )
        ]
        if complete:
            events.append(
                NS(
                    choices=[
                        NS(
                            delta=NS(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ]
                )
            )
    return nullcontext(iter(events)) if kind == "anthropic" else iter(events)


def call(agent, streaming):
    kwargs = {"tools": None, "iteration": 0, "stage": "test"}
    if streaming:
        list(agent._generate_stream_with_trace([], **kwargs))
    else:
        assert agent._generate_with_trace([], **kwargs) == "answer"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("streaming", [False, True])
def test_returned_model_is_current_and_persists_independently_of_alias(
    tmp_path, kind, streaming
):
    provider, wire = provider_fixture(kind)
    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    agent = MemAgent(model=provider, memory_provider=memory)
    agent._begin_trace_turn(None)
    # Include an unresolved alias and a missing model after two real versions.
    versions = ["fixture-2026-09-10", RETURNED, None, REQUESTED]
    expected = {}
    for version in versions:
        wire.return_value = (
            stream_response(kind, version)
            if streaming
            else regular_response(kind, version)
        )
        call(agent, streaming)
        result = agent._stream_trace_events[-1]
        assert result.get("response_model") == version
        expected[result["span_id"]] = version

    agent._record_stream_trace_bundle("memory-fixture", "thread-fixture")
    rows = memory.list_all(MemoryType.SHARED_MEMORY)
    events = normalize_trace_snapshot(TraceSnapshot(bundle_rows=rows)).events
    calls = [e for e in events if e["kind"] == "model_call"]
    results = [e for e in events if e["kind"] == "model_result"]
    assert len(calls) == len(results) == len(versions)
    assert all(e["model"] == REQUESTED for e in calls + results)
    assert all("response_model" not in e for e in calls)
    assert {e["span_id"]: e.get("response_model") for e in results} == expected
    assert all(e["status"] == "success" for e in results)
    assert all(c.kwargs["model"] == REQUESTED for c in wire.call_args_list)

    # A failed subsequent request must not borrow the previous model ID.
    wire.side_effect = TimeoutError("fixture unavailable")
    with pytest.raises(TimeoutError):
        call(agent, streaming)
    failed = agent._stream_trace_events[-1]
    assert failed["trace_kind"] == "model_result"
    assert failed["status"] == "error"
    assert "response_model" not in failed


@pytest.mark.parametrize("kind", KINDS)
def test_incomplete_stream_keeps_observed_model_but_empty_stream_does_not(kind):
    provider, wire = provider_fixture(kind)
    agent = MemAgent(model=provider)
    agent._begin_trace_turn(None)
    wire.return_value = stream_response(kind, RETURNED, complete=False)
    with pytest.raises(ProviderStreamError, match="incomplete"):
        call(agent, True)
    result = agent._stream_trace_events[-1]
    assert result["response_model"] == RETURNED
    assert result["status"] == "error"

    wire.return_value = nullcontext(iter([])) if kind == "anthropic" else iter([])
    with pytest.raises(ProviderStreamError, match="incomplete"):
        call(agent, True)
    assert "response_model" not in agent._stream_trace_events[-1]


@pytest.mark.parametrize("kind", KINDS)
def test_final_reported_model_overrides_stream_start_alias(kind):
    provider, wire = provider_fixture(kind)
    stream = stream_response(kind, REQUESTED)
    with stream if kind == "anthropic" else nullcontext(stream) as chunks:
        events = list(chunks)
    if kind == "responses":
        events[-1].response.model = RETURNED
    elif kind == "anthropic":
        events[-2].delta.model = RETURNED
    elif kind == "ollama":
        events[-1]["model"] = RETURNED
    else:
        events[-1].model = RETURNED
    wire.return_value = (
        nullcontext(iter(events)) if kind == "anthropic" else iter(events)
    )
    agent = MemAgent(model=provider)
    agent._begin_trace_turn(None)
    call(agent, True)
    assert agent._stream_trace_events[-1]["model"] == REQUESTED
    assert agent._stream_trace_events[-1]["response_model"] == RETURNED


@pytest.mark.parametrize("model", [None, "", " ", 123, {}, [RETURNED], "x" * 241])
def test_invalid_response_model_is_unknown_without_losing_other_metadata(model):
    for response in ({"model": model, "id": "request"}, NS(model=model, id="request")):
        assert response_metadata(response) == {"request_id": "request"}


def test_response_model_survives_v3_index_export_inspectors_and_comparison(tmp_path):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    identity = dict(
        agent_id="agent-fixture",
        thread_id="thread-fixture",
        run_id="run-fixture",
        turn_id="turn-fixture",
        root_trace_id="root-fixture",
        memory_id="memory-fixture",
    )
    event = TraceEventV3(
        **identity,
        event_id="result-fixture",
        span_id="model-fixture",
        event_kind="model_call",
        operation="generate",
        phase="result",
        status="success",
        attributes={
            "model": REQUESTED,
            "response_model": RETURNED,
            "provider": "openai",
        },
    )
    memory = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    ObservabilityStore(memory).record_trace_event(event)
    rows = memory.list_all(MemoryType.SHARED_MEMORY)
    normalized = normalize_trace_snapshot(TraceSnapshot(bundle_rows=rows)).events
    assert normalized[0]["response_model"] == RETURNED

    index = SQLiteSpanIndex(tmp_path / "index")
    index.initialize()
    from memorizz.observability.normalization import read_payload

    for row in rows:
        index.write_bundle(read_payload(row))
    assert (
        index.query(agent_ids=[identity["agent_id"]])["items"][0]["response_model"]
        == RETURNED
    )

    contract = {
        **identity,
        "kind": "output_contract",
        "contract_name": "answer",
        "timestamp": "2099-01-01T00:00:00Z",
    }
    inspectors = build_lineage_inspectors([*normalized, contract])
    assert inspectors["contracts"][0]["model_evidence"]["response_model"] == RETURNED
    previous = [{**normalized[0], "response_model": "fixture-2026-09-10"}]
    comparison = compare_trace_windows(previous, normalized)
    assert (
        comparison["changes"]["operations"]["added"][0]["value"]["response_model"]
        == RETURNED
    )

    agent_row = {
        "agent_id": identity["agent_id"],
        "memory_ids": [identity["memory_id"]],
    }
    with patch.dict(state._state, {"provider": memory}), patch.object(
        memory, "list_memagents", return_value=[agent_row]
    ), patch.object(memory, "retrieve_memagent", return_value=agent_row):
        client = TestClient(create_app())
        suffix = "agent_id=agent-fixture&thread_id=thread-fixture"
        html = client.get("/traces?" + suffix)
        exported = client.get("/traces/events.json?" + suffix)
    assert html.status_code == exported.status_code == 200
    assert f"Requested: {REQUESTED}" in html.text
    assert f"Returned: {RETURNED}" in html.text
    assert exported.json()["items"][0]["response_model"] == RETURNED
