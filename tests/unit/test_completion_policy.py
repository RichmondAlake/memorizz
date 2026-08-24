from __future__ import annotations

from unittest.mock import Mock

import pytest

from memorizz import (
    CompletionCandidate,
    CompletionPolicy,
    CompletionRejectedError,
    MemAgentBuilder,
)


def test_completion_policy_retries_in_same_model_loop(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        max_rejections=1,
        validator=lambda candidate: (
            candidate.response == "verified",
            "Host evidence is not verified yet.",
        ),
        validator_name="test_verifier",
    )
    agent.model.generate.side_effect = ["not done", "verified"]

    response = agent.run("finish the task", memory_id="completion-test")

    assert response == "verified"
    assert agent.model.generate.call_count == 2
    second_messages = agent.model.generate.call_args_list[1].args[0]
    assert second_messages[-2] == {"role": "assistant", "content": "not done"}
    assert second_messages[-1]["role"] == "developer"
    assert "Host evidence" in second_messages[-1]["content"]
    report = agent.completion_policy_report()
    assert report["accepted"] is True
    assert report["rejection_count"] == 1
    assert all(item["response_sha256"] for item in report["decisions"])


def test_completion_policy_fails_closed_after_bounded_retries(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        max_rejections=0,
        validator=lambda _candidate: (False, "Acceptance test failed."),
        validator_name="always_fail",
    )
    agent.model.generate.return_value = "looks done"

    with pytest.raises(CompletionRejectedError) as raised:
        agent.run("finish the task", memory_id="completion-failure")

    assert raised.value.decision.code == "validator_rejected"
    assert "Acceptance test failed" in str(raised.value)


def test_persisted_required_validator_fails_closed_until_rebound():
    configured = CompletionPolicy(
        enabled=True,
        validator=lambda _candidate: True,
        validator_name="trusted_host_check",
    )

    restored = CompletionPolicy.from_value(configured.to_dict())
    decision = restored.evaluate(
        CompletionCandidate(query="q", response="done", iteration=1)
    )

    assert decision.accepted is False
    assert decision.code == "validator_not_bound"
    assert "trusted_host_check" in decision.reason


def test_builder_exposes_completion_policy():
    policy = CompletionPolicy(enabled=True, forbidden_response_patterns=("unfinished",))

    agent = (
        MemAgentBuilder()
        .with_model(Mock(generate=Mock(return_value="done")))
        .with_memory_provider(False)
        .with_completion_policy(policy)
        .build(validate=False)
    )

    assert agent.completion_policy is policy


def test_streaming_gate_does_not_leak_rejected_tokens(memagent_with_mocks):
    agent = memagent_with_mocks
    attempts = iter(
        [
            [
                {"type": "content", "content": "unfinished"},
                {"type": "done", "content": "unfinished"},
            ],
            [
                {"type": "content", "content": "verified"},
                {"type": "done", "content": "verified"},
            ],
        ]
    )
    agent.model.generate_stream = lambda *_args, **_kwargs: iter(next(attempts))
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        max_rejections=1,
        validator=lambda candidate: candidate.response == "verified",
        validator_name="stream_verifier",
    )

    chunks = list(agent.run_stream("finish", memory_id="completion-stream"))

    assert chunks == ["verified"]
    assert agent.completion_policy_report()["rejection_count"] == 1


def test_cached_completion_is_revalidated_by_current_host_policy(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value="cached answer")
    agent.model.generate.reset_mock()
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        validator=lambda candidate: candidate.response == "cached answer",
        validator_name="current_host_policy",
    )

    response = agent.run("question", memory_id="completion-cache")

    assert response == "cached answer"
    agent.model.generate.assert_not_called()
    report = agent.completion_policy_report()
    assert report["accepted"] is True
    assert report["decisions"][0]["iteration"] == 0


def test_rejected_cached_completion_falls_through_to_fresh_model_turn(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    agent.cache_manager.enabled = True
    agent.cache_manager.get_cached_response = Mock(return_value="stale answer")
    agent.cache_manager.cache_response = Mock(return_value=True)
    agent.model.generate.reset_mock()
    agent.model.generate.return_value = "fresh answer"
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        validator=lambda candidate: candidate.response == "fresh answer",
        validator_name="freshness_check",
    )

    response = agent.run("question", memory_id="completion-cache-refresh")

    assert response == "fresh answer"
    agent.model.generate.assert_called_once()
    report = agent.completion_policy_report()
    assert [item["accepted"] for item in report["decisions"]] == [False, True]


def test_tool_evidence_policy_bypasses_semantic_cache_lookup(memagent_with_mocks):
    agent = memagent_with_mocks
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        require_tool_calls=True,
    )

    assert agent._semantic_cache_preflight_bypass("finish") == (
        "completion_policy_requires_fresh_tool_evidence"
    )


def test_completion_policy_is_part_of_semantic_cache_fingerprint(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    original = agent._semantic_cache_metadata()["fingerprints"]["completion_policy"]
    agent.completion_policy = CompletionPolicy(
        enabled=True,
        validator_name="new_policy",
        validator_required=True,
    )

    changed = agent._semantic_cache_metadata()["fingerprints"]["completion_policy"]

    assert changed != original
