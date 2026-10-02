# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Data shaping for the continual-learning monitor pages.

``memorizz.ui.learning_view`` turns trajectory stats and skillbox documents
into grid rows, chip counts and tape totals for ``/memory/workflows`` and
``/memory/skills``. These tests pin the classification (ready, needs more
evidence, failing, promoted), the criteria explainer, and the totals.
"""

import re
from datetime import datetime, timedelta, timezone

import pytest

from memorizz.long_term.procedural.skillbox import PromotionConfig
from memorizz.long_term.procedural.workflow.canonicalization import (
    RunRef,
    TrajectoryStats,
)
from memorizz.ui.learning_view import (
    ago,
    build_skill_monitor,
    build_workflow_monitor,
    class_key,
    promotion_criteria,
    safe_key,
    shape_class,
    shape_skill,
    tone,
)

NOW = datetime(2026, 9, 28, 12, 0, 0)
CONFIG = PromotionConfig()  # 5 runs, 80% success, 2 queries, 30 days


def _stat(
    executions=5,
    successes=None,
    queries=3,
    last_seen=NOW - timedelta(hours=2),
    promoted_skill_id=None,
    canonical_hash="a" * 64,
):
    successes = executions if successes is None else successes
    runs = [
        RunRef(
            workflow_id=f"wf-{i}",
            outcome="success" if i < successes else "failure",
            created_at=(last_seen - timedelta(minutes=i)) if last_seen else None,
            user_query=f"refund order {i}",
            promoted_skill_id=promoted_skill_id,
        )
        for i in range(executions)
    ]
    return TrajectoryStats(
        canonical_hash=canonical_hash,
        executions=executions,
        successes=successes,
        success_rate=successes / executions if executions else 0.0,
        distinct_query_count=queries,
        first_seen=(last_seen - timedelta(days=1)) if last_seen else None,
        last_seen=last_seen,
        already_promoted=promoted_skill_id is not None,
        promoted_skill_id=promoted_skill_id,
        runs=runs,
    )


def _shape(stat, agent_id="agent-1", tools="lookup_order → issue_refund", **kw):
    return shape_class(
        stat, agent_id=agent_id, tools=tools, config=CONFIG, now=NOW, **kw
    )


class TestHelpers:
    @pytest.mark.unit
    def test_ago_is_compact_and_tolerates_aware_times(self):
        assert ago(NOW - timedelta(minutes=5), NOW) == "5m ago"
        assert ago(NOW - timedelta(hours=3), NOW) == "3h ago"
        assert ago(NOW - timedelta(days=2, hours=1), NOW) == "2d ago"
        assert ago(None, NOW) == "—"
        aware = datetime.now(timezone.utc) - timedelta(hours=1)
        assert ago(aware, datetime.now()) == "1h ago"

    @pytest.mark.unit
    def test_tone_uses_signal_colours(self):
        assert tone(0.95) == "is-good"
        assert tone(0.6) == "is-warn"
        assert tone(0.2) == "is-bad"
        assert tone(None) == ""
        assert tone(0.7, good=0.6) == "is-good"

    @pytest.mark.unit
    def test_keys_are_safe_for_ids_and_fragments(self):
        assert safe_key("skill 1/α") == "skill-1"
        assert safe_key(None) == "none"
        assert class_key(None, "b" * 64) == "b" * 16 + "-none"
        assert class_key("agent/1", "c" * 64) == "c" * 16 + "-agent-1"

    @pytest.mark.unit
    def test_learning_tabs_link_the_four_sibling_pages(self):
        from memorizz.ui.state import templates

        macro = templates.env.get_template("_learning_tabs.html").module
        html = str(macro.learning_tabs("skills"))
        links = re.findall(r'href="([^"]+)"', html)
        assert links == [
            "/memory/workflows",
            "/memory/skills",
            "/learning-control-plane",
            "/persona-evolution",
        ]
        for label in (
            "Workflow trajectories",
            "Learned skills",
            "Control plane",
            "Persona evolution",
        ):
            assert label in html


class TestPromotionCriteria:
    @pytest.mark.unit
    def test_all_four_criteria_with_detail_and_explanation(self):
        rows = promotion_criteria(_stat(), CONFIG, NOW)
        assert [row["name"] for row in rows] == [
            "executions",
            "success rate",
            "query diversity",
            "recency",
        ]
        assert all(row["passed"] for row in rows)
        assert rows[0]["detail"] == "5 / 5 runs"
        assert rows[1]["detail"] == "100% (need 80%)"
        assert rows[2]["detail"] == "3 distinct (need 2)"
        assert rows[3]["detail"].startswith("last seen 0d ago")
        assert all(row["measures"] for row in rows)

    @pytest.mark.unit
    def test_unmet_criteria(self):
        stale = _stat(
            executions=2, successes=1, queries=1, last_seen=NOW - timedelta(days=45)
        )
        assert [row["passed"] for row in promotion_criteria(stale, CONFIG, NOW)] == [
            False,
            False,
            False,
            False,
        ]
        no_time = _stat(last_seen=None)
        recency = promotion_criteria(no_time, CONFIG, NOW)[3]
        assert recency == {**recency, "passed": False, "detail": "no timestamps"}

    @pytest.mark.unit
    def test_aware_last_seen_does_not_break_recency(self):
        aware = datetime.now(timezone.utc) - timedelta(days=1)
        rows = promotion_criteria(_stat(last_seen=aware), CONFIG, datetime.now())
        assert rows[3]["passed"] is True


class TestShapeClass:
    @pytest.mark.unit
    def test_eligible_class_is_ready_to_distill(self):
        row = _shape(_stat())
        assert row["status"] == "ready"
        assert row["tags"] == ["ready"]
        assert row["eligible"] is True
        assert row["criteria_met"] == 4 and row["criteria_total"] == 4
        assert row["success_rate"] == "100%" and row["success_tone"] == "is-good"
        assert row["step_count"] == 2
        assert row["key"] == class_key("agent-1", "a" * 64)
        assert row["last_seen_ago"] == "2h ago"
        assert row["run_count"] == 5 and len(row["runs"]) == 5

    @pytest.mark.unit
    def test_short_history_needs_more_evidence(self):
        row = _shape(_stat(executions=2))
        assert row["status"] == "evidence"
        assert row["tags"] == ["evidence"]
        assert row["eligible"] is False
        assert row["criteria_met"] == 3

    @pytest.mark.unit
    def test_low_success_is_failing_and_needs_evidence(self):
        row = _shape(_stat(executions=4, successes=1))
        assert row["status"] == "failing"
        assert row["tags"] == ["evidence", "failing"]
        assert row["success_rate"] == "25%" and row["success_tone"] == "is-bad"
        assert row["failed"] == 3

    @pytest.mark.unit
    def test_promoted_class_is_not_offered_for_distillation(self):
        row = _shape(_stat(promoted_skill_id="skill/1"), skill_name="Refunds")
        assert row["status"] == "promoted"
        assert row["tags"] == ["promoted"]
        assert row["eligible"] is False
        assert row["covering_skill_name"] == "Refunds"
        assert row["covering_skill_key"] == "skill-1"
        assert all(run["promoted"] for run in row["runs"])

    @pytest.mark.unit
    def test_run_list_is_bounded(self):
        row = _shape(_stat(executions=14), run_limit=10)
        assert row["run_count"] == 14
        assert len(row["runs"]) == 10
        assert row["runs"][0]["created_at"] == "2026-09-28 10:00"


class TestWorkflowMonitor:
    def _sections(self):
        return [
            {
                "agent_id": "agent-1",
                "name": "Support bot",
                "classes": [
                    _shape(_stat(executions=2), tools="discover_tools"),
                    _shape(_stat(canonical_hash="b" * 64)),
                ],
                "total_runs": 7,
                "report": None,
                "can_act": True,
            },
            {
                "agent_id": None,
                "name": "",
                "classes": [
                    _shape(
                        _stat(executions=4, successes=0, canonical_hash="c" * 64),
                        agent_id=None,
                    )
                ],
                "total_runs": 4,
                "report": {"promoted": ["s1"], "at": "2026-09-28 11:00:00"},
                "can_act": False,
            },
        ]

    @pytest.mark.unit
    def test_rows_put_ready_first_and_carry_agent_labels(self):
        view = build_workflow_monitor(self._sections(), total_runs=11, now=NOW)
        assert [row["status"] for row in view["rows"]] == [
            "ready",
            "failing",
            "evidence",
        ]
        first = view["rows"][0]
        assert first["agent_label"] == "Support bot"
        assert first["agent_short"] == "agent-1"
        assert first["can_act"] is True
        assert "support bot" in first["search"]
        assert "refund order 0" in first["search"]
        unowned = view["rows"][1]
        assert unowned["agent_label"] == "(no agent id)"
        assert unowned["agent_key"] == "none"
        assert unowned["can_act"] is False

    @pytest.mark.unit
    def test_tape_counts_and_overall_success(self):
        view = build_workflow_monitor(self._sections(), total_runs=11, now=NOW)
        summary = view["summary"]
        assert summary["runs"] == 11
        assert summary["classes"] == 3
        assert summary["agents"] == 2
        assert summary["ready"] == 1
        assert summary["failing"] == 1
        # 2 + 5 + 0 successes out of 2 + 5 + 4 runs.
        assert summary["success_rate"] == "64%"
        assert summary["success_tone"] == "is-warn"
        assert summary["newest"] == "2026-09-28 10:00"
        assert summary["newest_ago"] == "2h ago"
        assert view["counts"] == {
            "ready": 1,
            "evidence": 2,
            "failing": 1,
            "promoted": 0,
        }

    @pytest.mark.unit
    def test_agent_roll_up_and_reports(self):
        view = build_workflow_monitor(self._sections(), total_runs=11, now=NOW)
        support, unowned = view["agents"]
        assert support["label"] == "Support bot"
        assert support["classes"] == 2
        assert support["runs"] == 7
        assert support["ready"] == 1
        assert support["success_rate"] == "100%"
        assert unowned["key"] == "none" and unowned["can_act"] is False
        assert [agent["label"] for agent in view["reports"]] == ["(no agent id)"]

    @pytest.mark.unit
    def test_empty_sections(self):
        view = build_workflow_monitor([], total_runs=0, now=NOW)
        assert view["rows"] == [] and view["agents"] == []
        assert view["summary"]["success_rate"] == "—"
        assert view["summary"]["newest_ago"] == "—"


def _skill_doc(status="shadow", **overrides):
    doc = {
        "skill_id": "skill-1",
        "agent_id": "agent-1",
        "name": "Refund lookup",
        "description": "Use when the user asks for a refund.",
        "content": "# Refund lookup",
        "status": status,
        "injection_role": "developer",
        "version": 2,
        "tools_used": ["lookup_order", "issue_refund"],
        "source_canonical_hash": "d" * 64,
        "stats": {
            "activations": 6,
            "successes": 4,
            "failures": 2,
            "deviations": 1,
            "shadow": {
                "observations": 12,
                "trajectory_matches": 10,
                "matched_successes": 9,
                "matched_failures": 1,
                "last_evaluated_at": "2026-09-28T09:00:00+00:00",
            },
        },
        "created_at": (NOW - timedelta(days=3)).isoformat(),
        "promoted_at": (NOW - timedelta(days=1)).isoformat(),
    }
    doc.update(overrides)
    return doc


class TestSkills:
    @pytest.mark.unit
    def test_shadow_skill_row(self):
        row = shape_skill(_skill_doc(), CONFIG, NOW, agent_name="Support bot")
        assert row["key"] == "skill-1"
        assert row["status_label"] == "Shadow"
        assert row["status_health"] == "degraded"
        assert row["tags"] == ["shadow", "ready"]
        assert row["can_activate"] is True and row["can_demote"] is False
        assert row["injection_role"] == "developer"
        assert row["success_rate"] == "67%" and row["success_tone"] == "is-warn"
        assert row["shadow_observations"] == 12
        assert round(row["shadow_trajectory_match_rate"], 2) == 0.83
        assert row["shadow_ready"] is True
        assert row["source_key"] == class_key("agent-1", "d" * 64)
        assert row["listed_ago"] == "1d ago"
        assert row["agent_label"] == "Support bot"
        assert "lookup_order" in row["search"] and "support bot" in row["search"]

    @pytest.mark.unit
    def test_active_and_unknown_role(self):
        row = shape_skill(
            _skill_doc("active", injection_role="system", stats={}), CONFIG, NOW
        )
        assert row["injection_role"] == "user"
        assert row["can_demote"] is True and row["can_activate"] is False
        assert row["tags"] == ["active"]
        assert row["success_rate"] == "—"
        assert row["agent_label"] == "agent-1"

    @pytest.mark.unit
    def test_monitor_orders_by_lifecycle_and_counts_present_statuses(self):
        skills = [
            shape_skill(
                _skill_doc("demoted", skill_id="d", name="B", agent_id="agent-2"),
                CONFIG,
                NOW,
            ),
            shape_skill(_skill_doc("shadow", skill_id="s", name="C"), CONFIG, NOW),
            shape_skill(_skill_doc("active", skill_id="a", name="A"), CONFIG, NOW),
        ]
        view = build_skill_monitor(skills)
        assert [row["status"] for row in view["rows"]] == [
            "active",
            "shadow",
            "demoted",
        ]
        assert view["statuses"] == [("active", 1), ("shadow", 1), ("demoted", 1)]
        assert view["counts"]["candidate"] == 0
        assert view["ready"] == 1
        assert view["summary"]["skills"] == 3
        assert view["summary"]["activations"] == 18
        assert view["summary"]["deviations"] == 3
        assert view["summary"]["success_rate"] == "67%"
        assert [agent["key"] for agent in view["agents"]] == ["agent-1", "agent-2"]

    @pytest.mark.unit
    def test_empty_skill_monitor(self):
        view = build_skill_monitor([])
        assert view["rows"] == [] and view["statuses"] == []
        assert view["summary"]["newest_ago"] == "—"


class TestPages:
    """The routes render the monitor layout with the shaped data."""

    @pytest.fixture()
    def client(self, tmp_path):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from pathlib import Path
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from memorizz.enums.memory_type import MemoryType
        from memorizz.long_term.procedural.workflow.canonicalization import (
            canonical_hash,
            canonical_signature,
        )
        from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
        from memorizz.ui import state
        from memorizz.ui.app import create_app

        provider = FileSystemProvider(
            FileSystemConfig(root_path=Path(tmp_path) / "lw", lazy_vector_indexes=True)
        )
        steps = {
            "Step 1: lookup_order": {
                "arguments": {"order_id": "A"},
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }
        }
        signature = canonical_signature(steps)
        self.hash = canonical_hash(signature)
        for index in range(5):
            provider.store(
                {
                    "workflow_id": f"wf-{index}",
                    "agent_id": "agent-lw",
                    "user_query": f"where is order {index}",
                    "outcome": "success",
                    "created_at": datetime.now().isoformat(),
                    "steps": steps,
                    "canonical_hash": self.hash,
                    "canonical_signature": signature,
                    "embedding": [0.0],
                },
                memory_store_type=MemoryType.WORKFLOW_MEMORY,
            )
        provider.store(
            _skill_doc("active", agent_id="agent-lw", embedding=[0.0]),
            memory_store_type=MemoryType.SKILLBOX,
        )
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
    def test_workflows_page_is_a_monitor(self, client):
        page = client.get("/memory/workflows").text
        assert 'data-sidebar-scope="learning"' in page
        assert 'data-sidebar-default="collapsed"' in page
        assert "/static/css/pages/learning.css" in page
        assert 'aria-label="Continual learning views"' in page
        assert 'href="/persona-evolution"' in page
        assert 'id="lw-table"' in page
        assert 'data-monitor-filter-for="lw-table"' in page
        assert 'data-monitor-tag="ready"' in page
        assert f'data-hash="{self.hash}"' in page
        assert f'action="/continual-learning/agent-lw/distill/{self.hash}"' in page
        assert 'action="/continual-learning/agent-lw/cycle"' in page
        assert "Ready to distill" in page

    @pytest.mark.unit
    def test_skills_page_is_a_monitor(self, client):
        page = client.get("/memory/skills").text
        assert 'data-sidebar-scope="learning"' in page
        assert 'id="lw-skill-table"' in page
        assert 'data-monitor-tag="active"' in page
        assert 'action="/continual-learning/skills/skill-1/demote"' in page
        assert 'id="lw-skill-md-skill-1"' in page
        assert 'href="/memory/workflows"' in page
        assert "authority: developer" in page
