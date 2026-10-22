# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Run labels (trace_tags) on a turn's trace, the Traces tag filter, and
MemAgent.trace_url()."""

from __future__ import annotations

import json
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.observability.normalization import TraceSnapshot, normalize_trace_snapshot
from memorizz.observability.tags import (
    normalize_trace_tags,
    parse_tag_filter,
    select_tagged_turns,
)
from memorizz.ui import state
from memorizz.ui.app import create_app


class _Model:
    def get_config(self):
        return {"model": "gpt-4o-mini", "provider": "openai"}

    def get_last_usage(self):
        return {"prompt_tokens": 100, "completion_tokens": 10}

    def generate(self, messages, tools=None):
        return "answer"


@pytest.fixture()
def provider(tmp_path):
    instance = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield instance
    instance.close()


def _agent(provider, name="tagged"):
    agent = MemAgent(
        model=_Model(),
        memory_provider=provider,
        memory_types=[MemoryType.CONVERSATION_MEMORY],
        name=name,
        context_policy={"progressive_tool_disclosure": False},
    )
    agent._schedule_conversation_embedding_backfill = lambda rows: None
    return agent


def _bundles(provider):
    return [
        row
        for row in provider.list_all(MemoryType.SHARED_MEMORY)
        if row.get("record_type") == "observability_trace_bundle"
    ]


# --- validation -------------------------------------------------------------


@pytest.mark.unit
def test_tags_are_validated_and_sorted():
    assert normalize_trace_tags(None) == {}
    assert normalize_trace_tags({"question": "q17", "experiment": "lme v2"}) == {
        "experiment": "lme v2",
        "question": "q17",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "tags, error",
    [
        (["experiment"], TypeError),
        ({"experiment": 2}, TypeError),
        ({"": "x"}, ValueError),
        ({"-lead": "x"}, ValueError),
        ({"k" * 65: "x"}, ValueError),
        ({"experiment": ""}, ValueError),
        ({"experiment": "v" * 129}, ValueError),
        ({"experiment": "a=b"}, ValueError),
        ({"experiment": "a,b"}, ValueError),
        ({"experiment": "two  spaces"}, ValueError),
        ({"experiment": "line\nbreak"}, ValueError),
        ({"owner": "someone@example.com"}, ValueError),
        ({"source": "https://example.com/x"}, ValueError),
        ({f"k{i}": "v" for i in range(17)}, ValueError),
    ],
)
def test_invalid_tags_are_rejected(tags, error):
    with pytest.raises(error):
        normalize_trace_tags(tags)


@pytest.mark.unit
def test_tag_filters_parse_repeated_and_comma_separated_pairs():
    assert parse_tag_filter(["experiment=lme-v2", "question=q17"]) == {
        "experiment": "lme-v2",
        "question": "q17",
    }
    assert parse_tag_filter(["experiment=lme-v2, question=q17"]) == {
        "experiment": "lme-v2",
        "question": "q17",
    }
    assert parse_tag_filter([]) == {}
    with pytest.raises(ValueError):
        parse_tag_filter(["experiment"])
    with pytest.raises(ValueError):
        parse_tag_filter(["experiment=a", "experiment=b"])


@pytest.mark.unit
def test_run_rejects_invalid_tags_before_the_turn(provider):
    agent = _agent(provider)
    with pytest.raises(ValueError):
        agent.run("question", memory_id="m1", trace_tags={"bad key": "x"})
    with pytest.raises(TypeError):
        agent.run_stream_events("question", memory_id="m1", trace_tags=["x"])
    assert _bundles(provider) == []
    assert agent.trace_url() is None


# --- recording --------------------------------------------------------------


@pytest.mark.unit
def test_run_records_tags_on_the_turn_and_its_bundle(provider):
    agent = _agent(provider)
    tags = {"experiment": "lme-v2", "question": "q17"}
    assert agent.run("question", memory_id="m1", thread_id="t1", trace_tags=tags) == (
        "answer"
    )

    (bundle,) = _bundles(provider)
    payload = json.loads(bundle["content"])
    assert payload["trace_tags"] == tags
    assert bundle["trace_tags"] == tags  # promoted for native queries
    by_kind = {event["trace_kind"]: event for event in payload["events"]}
    assert by_kind["turn_start"]["trace_tags"] == tags
    assert by_kind["turn_result"]["trace_tags"] == tags
    assert "trace_tags" not in by_kind["model_call"]
    # Tags are trace metadata only: never in the trace context carrier.
    assert "trace_tags" not in agent.get_trace_context()

    # Normalization copies the bundle's tags onto every event of the turn.
    window = normalize_trace_snapshot(TraceSnapshot(bundle_rows=[bundle]))
    assert window.metadata["normalization_errors"] == 0
    assert all(event["trace_tags"] == tags for event in window.events)

    # The next turn without tags is untagged.
    agent.run("again", memory_id="m1", thread_id="t1")
    untagged = [
        row
        for row in _bundles(provider)
        if not json.loads(row["content"]).get("trace_tags")
    ]
    assert len(untagged) == 1


@pytest.mark.unit
def test_streamed_turns_record_tags_and_trace_url_links_them(provider):
    agent = _agent(provider)
    events = list(
        agent.run_stream_events(
            "question",
            memory_id="m1",
            thread_id="t-stream",
            trace_tags={"experiment": "lme-v2"},
        )
    )
    done = events[-1]
    assert done["type"] == "run.done" and done["status"] == "completed", done
    (bundle,) = _bundles(provider)
    assert json.loads(bundle["content"])["trace_tags"] == {"experiment": "lme-v2"}

    url = agent.trace_url("http://localhost:9000/")
    parts = urlsplit(url)
    assert parts.netloc == "localhost:9000" and parts.path == "/traces"
    assert parts.fragment == "trace-timeline"
    query = {key: values[0] for key, values in parse_qs(parts.query).items()}
    assert query == {
        "agent_id": agent.agent_id,
        "thread_id": "t-stream",
        "root_trace_id": done["root_trace_id"],
        "turn_id": done["turn_id"],
    }


@pytest.mark.unit
def test_trace_url_is_none_before_a_turn_and_links_a_given_turn(provider):
    agent = _agent(provider)
    assert agent.trace_url() is None
    agent.run("first", memory_id="m1", thread_id="t1")
    first = agent.get_trace_context()
    agent.run("second", memory_id="m1", thread_id="t2")
    latest = parse_qs(urlsplit(agent.trace_url()).query)
    assert latest["thread_id"] == ["t2"]
    default_base = agent.trace_url(turn_id=first["turn_id"])
    assert default_base.startswith("http://127.0.0.1:8765/traces?")
    assert parse_qs(urlsplit(default_base).query)["turn_id"] == [first["turn_id"]]


# --- selection ---------------------------------------------------------------


@pytest.mark.unit
def test_tag_selection_keeps_tagged_turns_and_their_messages(provider):
    agent = _agent(provider)
    agent.run("tagged question", memory_id="m1", thread_id="t1", trace_tags={"q": "17"})
    agent.run("other question", memory_id="m1", thread_id="t2", trace_tags={"q": "18"})
    snapshot = TraceSnapshot(
        conversation_rows=provider.list_all(MemoryType.CONVERSATION_MEMORY),
        bundle_rows=_bundles(provider),
    )
    events = normalize_trace_snapshot(snapshot).events
    selected = select_tagged_turns(events, {"q": "17"})
    assert selected
    assert {event.get("thread_id") for event in selected} == {"t1"}
    roles = {event.get("role") for event in selected if event["kind"] == "conversation"}
    assert roles == {"user", "assistant"}, "the turn's messages stay with it"
    assert selected.coverage["tag_filter"] == "q=17"
    assert selected.coverage["tagged_turns"] == 1
    assert select_tagged_turns(events, {"q": "19"}) == []


# --- Traces page --------------------------------------------------------------


@pytest.fixture()
def ui(provider, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "")
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_ACCOUNTS", "{}")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)


@pytest.mark.unit
def test_traces_page_filters_by_tag_and_shows_tags(provider, ui):
    tagged = _agent(provider, "tagged-agent")
    tagged.run(
        "q", memory_id="m1", thread_id="t-17", trace_tags={"experiment": "lme-v2"}
    )
    other = _agent(provider, "other-agent")
    other.run("q", memory_id="m2", thread_id="t-other", trace_tags={"experiment": "v1"})

    page = ui.get("/traces", params={"tag": "experiment=lme-v2"})
    assert page.status_code == 200, page.text
    html = page.text
    assert 'data-agent-id="%s"' % tagged.agent_id in html
    assert 'data-agent-id="%s"' % other.agent_id not in html
    assert 'value="experiment=lme-v2"' in html
    assert "Clear tag" in html

    timeline = ui.get(
        "/traces",
        params={
            "agent_id": tagged.agent_id,
            "thread_id": "t-17",
            "tag": "experiment=lme-v2",
        },
    )
    assert timeline.status_code == 200, timeline.text
    assert ">Turn Start<" in timeline.text
    assert "experiment=lme-v2</a>" in timeline.text
    assert "Run tags" in timeline.text

    empty = ui.get(
        "/traces",
        params={"agent_id": tagged.agent_id, "tag": "experiment=nope"},
    )
    assert empty.status_code == 200
    assert ">Turn Start<" not in empty.text

    bad = ui.get("/traces", params={"tag": "experiment"})
    assert bad.status_code == 400


@pytest.mark.unit
def test_events_json_filters_trace_events_by_tag(provider, ui):
    agent = _agent(provider)
    agent.run("q", memory_id="m1", thread_id="t1", trace_tags={"q": "17"})
    agent.run("q", memory_id="m1", thread_id="t2", trace_tags={"q": "18"})
    page = ui.get(
        "/traces/events.json",
        params={"store": "trace", "agent_id": agent.agent_id, "tag": "q=17"},
    )
    assert page.status_code == 200, page.text
    items = page.json()["items"]
    assert items and {item["thread_id"] for item in items} == {"t1"}
    combined = ui.get(
        "/traces/events.json",
        params={"store": "all", "agent_id": agent.agent_id, "tag": "q=18"},
    )
    assert combined.status_code == 200, combined.text
    assert {item["thread_id"] for item in combined.json()["items"]} == {"t2"}


@pytest.mark.unit
def test_tag_filters_read_bundles_on_the_index_read_path(provider, ui, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_DUAL_WRITE", "true")
    monkeypatch.setenv("MEMORIZZ_OBSERVABILITY_READ_PATH", "index")
    provider.get_observability_index().initialize()
    agent = _agent(provider)
    agent.run("q", memory_id="m1", thread_id="t-indexed", trace_tags={"q": "17"})

    page = ui.get("/traces", params={"tag": "q=17"})
    assert page.status_code == 200, page.text
    assert 'data-agent-id="%s"' % agent.agent_id in page.text
    events = ui.get(
        "/traces/events.json",
        params={"store": "trace", "agent_id": agent.agent_id, "tag": "q=17"},
    )
    assert events.status_code == 200, events.text
    assert {item["thread_id"] for item in events.json()["items"]} == {"t-indexed"}
