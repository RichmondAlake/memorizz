"""The text a harness streams while it writes reaches a watching terminal,
never the ledger."""

from __future__ import annotations

import time

import pytest

from memorizz.cli.harness_feed import HarnessFeed
from memorizz.metaharness import ClaudeCodeHarness, HermesHarness, PiHarness, live
from memorizz.metaharness.models import HarnessTask
from memorizz.streaming import HarnessProgress, harness_event_listener

KEY = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"


def _watch(run_id):
    seen = []
    return seen, live.subscribe(run_id, seen.append)


def _pieces(seen):
    return [(e["data"]["kind"], e["data"]["id"], e["data"]["delta"]) for e in seen]


@pytest.mark.unit
def test_live_text_reaches_listeners_only_while_they_listen():
    live.publish("r0", "message", "nobody is watching")
    live.new_block("r0")
    assert "r0" not in live._blocks  # nothing kept for an unwatched run

    seen, stop = _watch("r1")
    live.new_block("r1")
    live.publish("r1", "message", "Hel")
    live.publish("r1", "message", "")  # empty pieces are dropped
    live.publish("r1", "message", "lo")
    live.new_block("r1")
    live.publish("r1", "reasoning", "hmm")
    live.publish("r2", "message", "another run")
    stop()
    live.publish("r1", "message", "after")
    assert _pieces(seen) == [
        ("message", "1", "Hel"),
        ("message", "1", "lo"),
        ("reasoning", "2", "hmm"),
    ]
    assert "r1" not in live._listeners and "r1" not in live._blocks


@pytest.mark.unit
def test_claude_code_streams_partial_messages_and_stores_none(tmp_path):
    harness = ClaudeCodeHarness(command="claude-for-test")
    task = HarnessTask(task="T", workspace=str(tmp_path), harness="claude-code")
    harness._cli_flags = frozenset({"--include-partial-messages"})
    assert "--include-partial-messages" in harness.build_command(
        task, workspace=tmp_path, prompt="P"
    )
    harness._cli_flags = frozenset()  # an older build without the flag
    assert "--include-partial-messages" not in harness.build_command(
        task, workspace=tmp_path, prompt="P"
    )

    def stream(event, **extra):
        return {"type": "stream_event", "event": event, "session_id": "s", **extra}

    seen, stop = _watch("run-c")
    try:
        records = [
            stream({"type": "message_start", "message": {"id": "msg_1"}}),
            stream({"type": "content_block_start", "index": 0}),
            stream(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Plan it"},
                }
            ),
            stream({"type": "content_block_start", "index": 1}),
            stream(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Agent memory "},
                }
            ),
            stream(
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "is…"},
                }
            ),
            stream(
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": "{"},
                }
            ),
            stream(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "subagent text"},
                },
                parent_tool_use_id="toolu_1",
            ),
        ]
        for record in records:
            assert harness.parse_event("run-c", record) == ([], {})
    finally:
        stop()
    assert _pieces(seen) == [
        ("reasoning", "1", "Plan it"),
        ("message", "2", "Agent memory "),
        ("message", "2", "is…"),
    ]


@pytest.mark.unit
def test_pi_and_hermes_stream_their_text_as_it_is_written():
    pi = PiHarness(command="pi-for-test", provider="ollama")

    def update(kind, **values):
        return {
            "type": "message_update",
            "assistantMessageEvent": {"type": kind, "contentIndex": 0, **values},
        }

    hermes = HermesHarness(command="hermes-for-test", provider="anthropic")
    seen, stop = _watch("run-p")
    try:
        for record in [
            update("thinking_start"),
            update("thinking_delta", delta="Look first."),
            update("text_start"),
            update("text_delta", delta="The directory "),
            update("text_delta", delta="is empty."),
            update("toolcall_delta", delta='{"path"'),
        ]:
            assert pi.parse_event("run-p", record) == ([], {})
    finally:
        stop()
    assert _pieces(seen) == [
        ("reasoning", "1", "Look first."),
        ("message", "2", "The directory "),
        ("message", "2", "is empty."),
    ]

    seen, stop = _watch("run-h")
    try:
        hermes.parse_event("run-h", {"type": "text", "text": "Reading "})
        hermes.parse_event("run-h", {"type": "text", "text": "the file."})
        events, _ = hermes.parse_event(
            "run-h", {"type": "tool_use", "name": "read_file", "input": {}}
        )
        assert events[0].data["text"] == "Reading the file."  # stored whole
        hermes.parse_event("run-h", {"type": "text", "text": "Done."})
    finally:
        stop()
    assert _pieces(seen) == [
        ("message", "1", "Reading "),
        ("message", "1", "the file."),
        ("message", "2", "Done."),
    ]


def _partial(feed, run_id, kind, block, delta):
    feed.listener(
        run_id,
        "claude-code",
        {"type": "partial", "data": {"kind": kind, "id": block, "delta": delta}},
    )


def _preview(feed, width=80, rows=10):
    feed.drain()
    return [line.plain for line in feed.preview(width, rows)]


@pytest.mark.unit
def test_feed_previews_the_latest_message_and_the_text_being_written():
    feed = HarnessFeed()
    put = lambda kind, data: feed.listener(
        "r1", "claude-code", {"type": kind, "data": data}
    )
    # A message that arrives whole (Codex) shows at once, before anything
    # follows it.
    put("message", {"role": "assistant", "text": "I'll look at the folder."})
    assert [line.plain for line in feed.drain()] == ["● Claude Code"]
    assert _preview(feed) == ["  I'll look at the folder."]

    _partial(feed, "r1", "reasoning", "1", "Checking what ")
    _partial(feed, "r1", "reasoning", "1", "is there.")
    assert _preview(feed) == [
        "  I'll look at the folder.",
        "  ∴ Checking what is there.",
    ]
    put("reasoning", {"text": ""})  # Claude Code withholds the text
    _partial(feed, "r1", "message", "2", f"Agent memory keeps {KEY} ")
    _partial(feed, "r1", "message", "2", "what an agent learns.\n\nIt has kinds")
    assert _preview(feed) == [
        "  I'll look at the folder.",
        "  Agent memory keeps [REDACTED] what an agent learns.",
        "  ",
        "  It has kinds",
    ]
    # The whole message replaces its pieces; the earlier one is printed.
    answer = f"Agent memory keeps {KEY} what an agent learns.\n\nIt has kinds."
    put("message", {"role": "assistant", "text": answer})
    assert [line.plain for line in feed.drain()] == ["  I'll look at the folder."]
    assert _preview(feed) == [
        "  Agent memory keeps [REDACTED] what an agent learns.",
        "  ",
        "  It has kinds.",
    ]
    # Only the last rows fit, wrapped to the terminal's width.
    _partial(feed, "r1", "message", "3", "word " * 60)
    rows = _preview(feed, width=40, rows=4)
    assert len(rows) == 4 and all(len(row) <= 40 for row in rows)
    # The answer is the chat's to print; the live area empties.
    assert feed.finish(answer) == []
    assert feed.preview(80) == []


@pytest.mark.unit
def test_a_long_command_output_shows_its_first_and_last_lines():
    feed = HarnessFeed()
    output = "\n".join(f"line {n}" for n in range(1, 21))
    feed.listener(
        "r1",
        "codex",
        {
            "type": "command",
            "data": {"id": "c", "command": "seq", "exit_code": 0, "output": output},
        },
    )
    assert [line.plain for line in feed.drain()][2:] == [
        "    line 1",
        "    line 2",
        "    line 3",
        "    line 4",
        "    … 12 more lines",
        "    line 17",
        "    line 18",
        "    line 19",
        "    line 20",
    ]


@pytest.mark.unit
def test_output_of_parallel_commands_follows_its_own_command():
    feed = HarnessFeed()
    put = lambda data: feed.listener("r1", "codex", {"type": "command", "data": data})
    put({"id": "a", "command": "seq 1 3", "status": "in_progress"})
    put({"id": "b", "command": "cat README.md", "status": "in_progress"})
    put({"id": "b", "command": "cat README.md", "exit_code": 0, "output": "# Hi"})
    put({"id": "a", "command": "seq 1 3", "exit_code": 0, "output": "1\n2\n3"})
    put({"id": "c", "command": "true", "status": "in_progress"})
    put({"id": "c", "command": "true", "exit_code": 0, "output": "ok"})
    assert [line.plain for line in feed.drain()][1:] == [
        "  $ seq 1 3",
        "  $ cat README.md",
        "    # Hi",
        "  $ seq 1 3",
        "    1",
        "    2",
        "    3",
        "  $ true",
        "    ok",  # nothing in between: no repeat
    ]


@pytest.mark.unit
def test_a_claude_code_tool_result_shows_the_tools_output_not_its_record():
    feed = HarnessFeed()

    def result(content, **block):
        feed.listener(
            "r1",
            "claude-code",
            {
                "type": "tool_result",
                "data": {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t",
                                "content": content,
                                **block,
                            }
                        ],
                    },
                    "tool_use_result": {"type": "text", "file": {"filePath": "/x"}},
                },
            },
        )

    result("     1\u2192# Scratch workspace\n     2\u2192The code word")
    result([{"type": "text", "text": "3 matches"}])
    result("File does not exist.", is_error=True)
    lines = feed.drain()
    assert [line.plain for line in lines][1:] == [
        "    ↳ 1→# Scratch workspace",
        "    ↳ 3 matches",
        "    ↳ File does not exist.",
    ]
    assert "red" in str(lines[-1].spans[-1].style)


class _Quiet:
    agent = None

    def emit(self, *args, **kwargs):
        pass


class _NoEvents:
    def events(self, run_id, *, after=0, limit=1000):
        return []


@pytest.mark.unit
def test_progress_hands_streamed_text_to_the_listener_until_the_run_stops():
    got = []
    token = harness_event_listener.set(lambda *args: got.append(args))
    try:
        progress = HarnessProgress(_Quiet(), _NoEvents(), "pi", poll_seconds=0.01)
    finally:
        harness_event_listener.reset(token)
    progress.start("run-x")
    live.publish("run-x", "message", "Hi")
    progress.stop()
    live.publish("run-x", "message", "late")
    time.sleep(0.02)
    kinds = [(run_id, harness, event["type"]) for run_id, harness, event in got]
    assert kinds == [
        ("run-x", "pi", "run.started"),
        ("run-x", "pi", "partial"),
        ("run-x", "pi", "run.finished"),
    ]
    assert got[1][2]["data"]["delta"] == "Hi"
    assert "run-x" not in live._listeners


@pytest.mark.unit
def test_progress_without_a_listener_never_subscribes():
    progress = HarnessProgress(_Quiet(), _NoEvents(), "pi", poll_seconds=0.01)
    progress.start("run-y")
    try:
        assert "run-y" not in live._listeners
    finally:
        progress.stop()
