"""MemAgent(raise_errors=True) and run(raise_errors=...): a failure is raised
instead of returned as the answer text."""

from __future__ import annotations

import threading

import pytest

from memorizz.memagent import MemAgent
from tests.mocks.mock_providers import MockMemoryProvider
from tests.unit.test_memagent_core_fixes import _StubModel

pytestmark = pytest.mark.unit


def _agent(**options) -> MemAgent:
    return MemAgent(
        model=_StubModel(error=RuntimeError("429 Too Many Requests")),
        memory_provider=MockMemoryProvider(),
        memory_ids=["memory-1"],
        auto_register=False,
        **options,
    )


def test_by_default_a_failure_is_the_answer_text():
    agent = _agent()
    try:
        answer = agent.run("hello", memory_id="memory-1", thread_id="t1")
        assert "I encountered an error" in answer and "429" in answer
    finally:
        agent.close()


def test_raise_errors_raises_the_failure():
    agent = _agent(raise_errors=True)
    try:
        with pytest.raises(RuntimeError, match="429 Too Many Requests"):
            agent.run("hello", memory_id="memory-1", thread_id="t1")
    finally:
        agent.close()


def test_one_call_can_raise_and_the_next_answers():
    agent = _agent()
    try:
        with pytest.raises(RuntimeError, match="429"):
            agent.run("hello", memory_id="memory-1", thread_id="t1", raise_errors=True)
        answer = agent.run("hello", memory_id="memory-1", thread_id="t1")
        assert "I encountered an error" in answer
        # And the other way round on an agent that raises.
        agent.raise_errors = True
        answer = agent.run(
            "hello", memory_id="memory-1", thread_id="t1", raise_errors=False
        )
        assert "I encountered an error" in answer
    finally:
        agent.close()


def test_a_per_call_choice_stays_with_its_own_call():
    agent = _agent()
    outcomes = {}

    def call(name, raise_errors):
        try:
            outcomes[name] = agent.run(
                "hello", memory_id="memory-1", thread_id=name, raise_errors=raise_errors
            )
        except RuntimeError as exc:
            outcomes[name] = exc

    try:
        threads = [
            threading.Thread(target=call, args=("raises", True)),
            threading.Thread(target=call, args=("answers", None)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert isinstance(outcomes["raises"], RuntimeError)
        assert "I encountered an error" in outcomes["answers"]
    finally:
        agent.close()
