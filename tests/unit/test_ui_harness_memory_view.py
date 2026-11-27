"""What a run did with memory, and one memory as harnesses share it."""

import json

import pytest

from memorizz.ui.harness_memory_view import (
    known_memories,
    lookups,
    memory_overview,
    run_memory,
)

SAVED = "[Harness run: droid · succeeded]\nThe codeword is PELICAN.\nHarness run: 24e95384-cc7e"


def _run(
    run_id, harness, memory_id="", answer="Done.", at="2026-11-23T10:00:00", **task
):
    return {
        "run_id": run_id,
        "harness": harness,
        "status": "succeeded",
        "created_at": at,
        "task": {"task": "Find the codeword", "memory_id": memory_id, **task},
        "result": {"final_response": answer, "context_pack": {"records": []}},
    }


def _call(sequence, tool_id, name, arguments):
    return {"sequence": sequence, "type": "tool_call",
            "data": {"id": tool_id, "name": name, "input": arguments}}  # fmt: skip


def _result(sequence, tool_id, content, is_error=False):
    return {"sequence": sequence, "type": "tool_result", "data": {"message": {"content": [
        {"tool_use_id": tool_id, "content": content, "is_error": is_error}]}}}  # fmt: skip


@pytest.mark.unit
def test_a_runs_memory_tab_shows_what_it_was_given_looked_up_and_was_refused():
    run = _run(
        "r2",
        "claude-code",
        "other",
        permissions={"mcp_read_memory_ids": ["notes", "other"]},
    )
    run["result"]["context_pack"]["records"] = [
        {"source_id": "m1", "memory_type": "conversation_memory", "memory_id": "notes",
         "content": SAVED, "score": 0.41, "timestamp": "2026-11-23T09:00:00"},
    ]  # fmt: skip
    events = [
        _call(1, "a", "mcp__memorizz__memorizz_search_memories", {"query": "codeword"}),
        _result(2, "a", json.dumps({"result": {"ok": True, "count": 2, "memories": [1, 2]}})),
        _call(3, "b", "mcp__memorizz__memorizz_search_memories", {"query": "x", "memory_id": "secret"}),
        _result(4, "b", "memory_out_of_scope: This server is limited to the memory other; it cannot reach 'secret'", True),
        _call(5, "c", "memorizz_list_memories", '{"limit": 5}'),
        _result(6, "c", json.dumps({"ok": True, "memories": []})),
        _call(7, "d", "mcp__memorizz__memorizz_remember", {"content": "A note"}),
        _result(8, "d", "provider_error: could not write", True),
        _call(9, "e", "Read", {"file_path": "README.md"}),
    ]  # fmt: skip

    view = run_memory(run, events)

    assert view["scope"] == {
        "memory_id": "other", "private": False, "open": False, "read_only": ["notes"],
        "access": "read_only", "can_write": False, "tools": True,
    }  # fmt: skip
    (given,) = view["given"]
    assert given["written_by"] == "droid" and given["run_id"] == "24e95384-cc7e"
    assert given["text"] == "The codeword is PELICAN." and given["read_only"] is True
    assert [(c["tool"], c["outcome"], c["found"]) for c in view["lookups"]] == [
        ("search_memories", "ok", 2),
        ("search_memories", "refused", None),
        ("list_memories", "nothing", 0),
        ("remember", "failed", None),
    ]
    assert "cannot reach 'secret'" in view["lookups"][1]["message"]
    assert view["lookups"][2]["arguments"] == {"limit": "5"}
    assert view["lookups"][3]["writes"] is True
    assert view["refused"] == 1 and view["saved"] is True


@pytest.mark.unit
def test_a_run_with_no_memory_is_private_and_codex_calls_are_read():
    view = run_memory(_run("r1", "codex"), [
        {"sequence": 1, "type": "tool_call", "data": {
            "id": "t", "server": "memorizz", "tool": "search_memories",
            "arguments": {"query": "x"}, "status": "completed",
            "result": {"count": 0, "memories": []}}},
    ])  # fmt: skip
    assert view["scope"]["private"] is True and view["saved"] is False
    assert [(c["tool"], c["outcome"]) for c in view["lookups"]] == [
        ("search_memories", "nothing")
    ]
    opened = _run(
        "r1", "codex", permissions={"mcp_memory_scope": "all", "mcp_access": "none"}
    )
    assert run_memory(opened, [])["scope"]["open"] is True
    assert run_memory(opened, [])["scope"]["tools"] is False
    assert lookups([]) == []


@pytest.mark.unit
def test_a_shared_memory_lists_who_wrote_and_who_only_read():
    runs = [
        _run("r3", "pi", "notes", at="2026-11-23T12:00:00"),
        _run("r1", "droid", "notes", at="2026-11-23T10:00:00"),
        _run("r2", "codex", "notes", answer="", at="2026-11-23T11:00:00"),
        _run("r4", "claude-code", "other", at="2026-11-23T13:00:00",
             permissions={"mcp_read_memory_ids": ["notes"]}),
        _run("r5", "codex", at="2026-11-23T14:00:00"),
    ]  # fmt: skip

    page = memory_overview("notes", runs)
    assert [row["run_id"] for row in page["runs"]] == ["r1", "r2", "r3"]
    assert [row["saved"] for row in page["runs"]] == [True, False, True]
    assert [row["run_id"] for row in page["readers"]] == ["r4"]
    assert page["saved"] == 2
    assert page["harnesses"] == [
        {"harness": "codex", "runs": 1, "saved": 0},
        {"harness": "droid", "runs": 1, "saved": 1},
        {"harness": "pi", "runs": 1, "saved": 1},
    ]
    # A failed run's error text is not an answer; it is not counted as saved.
    failed = dict(_run("r9", "hermes", "notes", answer="HTTP 402"), status="failed")
    assert memory_overview("notes", [failed])["runs"][0]["saved"] is False
    assert memory_overview("nobody", runs)["runs"] == []
    known = known_memories(runs)
    assert [entry["memory_id"] for entry in known] == ["other", "notes"]
    assert known[1]["runs"] == 3 and known[1]["harnesses"] == ["pi", "droid", "codex"]
