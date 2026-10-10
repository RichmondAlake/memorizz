"""CLI projection of the SDK's public event contract. Stdout is data only."""

import json
import os
import sys

from ..llms.streaming import provider_error_message
from . import config as cfg


def save_stream_session(session):
    try:
        cfg.save_state({"memory_id": session.memory_id, "thread_id": session.thread_id})
        if getattr(session.agent, "memory_provider", None) is not None:
            if session.agent.save() is False:
                return "failed"
        return "written"
    except Exception:
        return "failed"


def consume_stream(session, prompt, *, output="text", stdout=None, stderr=None):
    if output not in {"text", "jsonl"}:
        raise ValueError("output must be text or jsonl")
    from ..streaming import harness_event_listener
    from .harness_feed import HarnessFeed, plain

    stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
    # In text mode a harness's own output goes to stderr as it happens.
    feed = HarnessFeed() if output == "text" else None
    listening = harness_event_listener.set(feed.listener if feed else None)
    try:
        stream = session.agent.run_stream_events(
            prompt,
            memory_id=session.memory_id,
            thread_id=session.thread_id,
            user_id=session.user_id,
        )
    except BaseException:
        harness_event_listener.reset(listening)
        raise
    status, state_saved, failure_message = "error", "unknown", None
    answer = ""
    try:
        for event in stream:
            kind = event["type"]
            if feed is not None:
                if kind == "answer.delta":
                    answer += event["delta"]
                lines = (
                    feed.finish(answer)
                    if kind in {"answer.done", "run.done"}
                    else feed.drain()
                )
                if lines:
                    stderr.write(plain(lines))
                    stderr.flush()
            if kind in {"run.started", "run.done"}:
                session.memory_id = event.get("memory_id") or session.memory_id
                session.thread_id = event.get("thread_id") or session.thread_id
            if kind == "run.done":
                status = event["status"]
                failure_message = provider_error_message(event.get("error_code"))
                state_saved = save_stream_session(session)
                event = {
                    **event,
                    "persistence": {
                        **event.get("persistence", {}),
                        "cli_state": state_saved,
                    },
                }
            if output == "jsonl":
                stdout.write(
                    json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stdout.flush()
            elif kind == "answer.delta":
                stdout.write(event["delta"])
                stdout.flush()
            elif kind == "status":
                stderr.write(
                    (
                        event.get("message")
                        or event.get("stage", "working").replace("_", " ")
                    )
                    + "\n"
                )
                stderr.flush()
        if output == "text":
            stdout.write("\n")
            stdout.flush()
        if status != "completed" or state_saved == "failed":
            stderr.write(f"Run {status}; CLI state {state_saved}.\n")
            if output == "text" and failure_message:
                stderr.write(failure_message + "\n")
        return (
            130
            if status == "cancelled"
            else 3
            if status == "approval_required"
            else 0
            if status == "completed" and state_saved != "failed"
            else 1
        )
    except KeyboardInterrupt:
        stderr.write("Interrupted; partial answer retained.\n")
        return 130
    except BrokenPipeError:
        # Prevent Python's final stdout flush from replacing our exit code with 120.
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), stdout.fileno())
        except (AttributeError, OSError, ValueError):
            pass
        return 141
    finally:
        stream.close()
        harness_event_listener.reset(listening)
