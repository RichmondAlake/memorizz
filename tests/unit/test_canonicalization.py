# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Unit tests for workflow trajectory canonicalization.

Trajectory identity is the foundation of continual learning: promotion is
counting, and counting requires identity. These tests pin down the rules
R1–R4 documented on ``canonical_signature``.
"""

from datetime import datetime, timedelta

from memorizz.long_term.procedural.workflow.canonicalization import (
    aggregate_trajectory_stats,
    canonical_hash,
    canonical_signature,
    normalize_query,
    toolset_hash,
)


def _step(tool_args, result="ok", ts=None, error=None):
    return {
        "arguments": tool_args,
        "result": result,
        "timestamp": ts,
        "error": error,
    }


def _steps(*entries):
    """entries: (tool_name, arguments, error) tuples -> steps dict."""
    steps = {}
    base = datetime(2026, 7, 1, 10, 0, 0)
    for index, (tool, args, error) in enumerate(entries):
        steps[f"Step {index + 1}: {tool}"] = _step(
            args, ts=(base + timedelta(seconds=index)).isoformat(), error=error
        )
    return steps


class TestCanonicalSignature:
    def test_same_tools_different_argument_values_hash_identically(self):
        run_a = _steps(
            ("lookup_order", {"order_id": "A-1"}, None),
            ("issue_refund", {"order_id": "A-1", "amount": 10}, None),
        )
        run_b = _steps(
            ("lookup_order", {"order_id": "Z-999"}, None),
            ("issue_refund", {"amount": 5000, "order_id": "Z-999"}, None),
        )
        assert canonical_hash(canonical_signature(run_a)) == canonical_hash(
            canonical_signature(run_b)
        )

    def test_different_tool_order_hashes_differently(self):
        run_a = _steps(
            ("lookup_order", {"order_id": "A"}, None),
            ("issue_refund", {"order_id": "A"}, None),
        )
        run_b = _steps(
            ("issue_refund", {"order_id": "A"}, None),
            ("lookup_order", {"order_id": "A"}, None),
        )
        assert canonical_hash(canonical_signature(run_a)) != canonical_hash(
            canonical_signature(run_b)
        )

    def test_retry_collapse_fail_fail_success(self):
        run = _steps(
            ("lookup_order", {"order_id": "A"}, "timeout"),
            ("lookup_order", {"order_id": "A"}, "timeout"),
            ("lookup_order", {"order_id": "A"}, None),
        )
        signature = canonical_signature(run)
        assert len(signature) == 1
        assert signature[0]["retry_count"] == 3
        assert signature[0]["errored"] is False

    def test_retry_count_excluded_from_hash(self):
        clean = _steps(("lookup_order", {"order_id": "A"}, None))
        retried = _steps(
            ("lookup_order", {"order_id": "B"}, "boom"),
            ("lookup_order", {"order_id": "B"}, None),
        )
        assert canonical_hash(canonical_signature(clean)) == canonical_hash(
            canonical_signature(retried)
        )

    def test_consecutive_successes_do_not_collapse(self):
        run = _steps(
            ("send_email", {"to": "a"}, None),
            ("send_email", {"to": "b"}, None),
        )
        assert len(canonical_signature(run)) == 2

    def test_terminal_error_is_part_of_identity(self):
        succeeded = _steps(("lookup_order", {"order_id": "A"}, None))
        failed = _steps(("lookup_order", {"order_id": "A"}, "boom"))
        assert canonical_hash(canonical_signature(succeeded)) != canonical_hash(
            canonical_signature(failed)
        )

    def test_argument_shape_is_identity(self):
        with_amount = _steps(("issue_refund", {"order_id": "A", "amount": 1}, None))
        without_amount = _steps(("issue_refund", {"order_id": "A"}, None))
        assert canonical_hash(canonical_signature(with_amount)) != canonical_hash(
            canonical_signature(without_amount)
        )

    def test_missing_timestamps_fall_back_to_insertion_order(self):
        steps = {
            "Step 1: b_tool": _step({"x": 1}),
            "Step 2: a_tool": _step({"x": 1}),
        }
        signature = canonical_signature(steps)
        assert [unit["tool"] for unit in signature] == ["b_tool", "a_tool"]

    def test_timestamps_override_insertion_order(self):
        steps = {
            "Step 1: second": _step({"x": 1}, ts="2026-07-01T10:00:05"),
            "Step 2: first": _step({"x": 1}, ts="2026-07-01T10:00:01"),
        }
        signature = canonical_signature(steps)
        assert [unit["tool"] for unit in signature] == ["first", "second"]

    def test_empty_steps_have_no_hash(self):
        assert canonical_signature({}) == []
        assert canonical_hash([]) is None

    def test_determinism_fixed_vector(self):
        run = _steps(
            ("lookup_order", {"order_id": "A"}, None),
            ("issue_refund", {"amount": 1, "order_id": "A"}, None),
        )
        assert canonical_hash(canonical_signature(run)) == (
            "5e6b70e86ec21732b51dd93e2ddd9627b12c43186d3d048e7a8d356b41c37e12"
        )

    def test_toolset_hash_is_order_insensitive(self):
        run_a = _steps(("a", {}, None), ("b", {}, None))
        run_b = _steps(("b", {}, None), ("a", {}, None))
        assert toolset_hash(canonical_signature(run_a)) == toolset_hash(
            canonical_signature(run_b)
        )


class TestNormalizeQuery:
    def test_case_and_whitespace_collapse(self):
        assert normalize_query("  Refund   Order A1 ") == "refund order a1"
        assert normalize_query(None) == ""


class _ListProvider:
    """Minimal provider stub exposing list_all only."""

    def __init__(self, docs):
        self._docs = docs

    def list_all(self, memory_store_type=None):
        return self._docs


def _wf_doc(idx, chash, outcome="success", agent_id="agent-1", query="q", **extra):
    doc = {
        "_id": f"rec-{idx}",
        "workflow_id": f"wf-{idx}",
        "agent_id": agent_id,
        "canonical_hash": chash,
        "outcome": outcome,
        "user_query": query,
        "created_at": (datetime(2026, 7, 1) + timedelta(hours=idx)).isoformat(),
        "steps": {},
    }
    doc.update(extra)
    return doc


class TestAggregateTrajectoryStats:
    def test_groups_by_hash_and_computes_rates(self):
        docs = [
            _wf_doc(1, "h1", query="refund order a"),
            _wf_doc(2, "h1", query="refund order b"),
            _wf_doc(3, "h1", outcome="failure", query="refund order c"),
            _wf_doc(4, "h2", query="something else"),
        ]
        stats = {
            s.canonical_hash: s
            for s in aggregate_trajectory_stats(_ListProvider(docs), "agent-1")
        }
        assert stats["h1"].executions == 3
        assert stats["h1"].successes == 2
        assert abs(stats["h1"].success_rate - 2 / 3) < 1e-9
        assert stats["h1"].distinct_query_count == 3
        assert stats["h2"].executions == 1

    def test_agent_scoping(self):
        docs = [_wf_doc(1, "h1"), _wf_doc(2, "h1", agent_id="other")]
        stats = aggregate_trajectory_stats(_ListProvider(docs), "agent-1")
        assert stats[0].executions == 1

    def test_unhashed_docs_are_hashed_on_the_fly(self):
        steps = {
            "Step 1: lookup_order": _step({"order_id": "A"}, ts="2026-07-01T10:00:00")
        }
        expected = canonical_hash(canonical_signature(steps))
        docs = [_wf_doc(1, None, canonical_hash=None, steps=steps)]
        docs[0].pop("canonical_hash")
        docs[0]["steps"] = steps
        stats = aggregate_trajectory_stats(_ListProvider(docs), "agent-1")
        assert stats[0].canonical_hash == expected

    def test_promoted_stamp_surfaces(self):
        docs = [
            _wf_doc(1, "h1", promoted_skill_id="skill-9"),
            _wf_doc(2, "h1"),
        ]
        stats = aggregate_trajectory_stats(_ListProvider(docs), "agent-1")
        assert stats[0].already_promoted is True
        assert stats[0].promoted_skill_id == "skill-9"

    def test_record_ids_prefer_provider_id(self):
        docs = [_wf_doc(1, "h1")]
        stats = aggregate_trajectory_stats(_ListProvider(docs), "agent-1")
        assert stats[0].runs[0].record_id == "rec-1"
