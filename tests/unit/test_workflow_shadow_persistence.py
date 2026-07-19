"""Workflow shadow-evaluation serialization across document providers."""

from datetime import datetime
from pathlib import Path

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.workflow import Workflow
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

EVALUATION = {
    "idempotency_key": "record-1:skill-1:v1",
    "skill_id": "skill-1",
    "skill_version": 1,
    "similarity": 0.87,
    "expected_canonical_hash": "hash-a",
    "observed_canonical_hash": "hash-a",
    "trajectory_match": True,
    "applicability_mismatch": False,
    "matched_trajectory_succeeded": True,
    "observed_outcome": "success",
    "evaluated_at": "2026-07-19T12:00:00+00:00",
    "evaluator_version": "v1",
}


def _workflow():
    return Workflow(
        name="run",
        workflow_id="workflow-shadow-1",
        created_at=datetime(2026, 7, 19, 12, 0, 0),
        steps={},
        shadow_evaluations=[EVALUATION],
        embedding=[],
    )


def test_workflow_model_round_trips_shadow_evaluations():
    workflow = _workflow()
    clone = Workflow.from_dict(workflow.to_dict())
    assert clone.shadow_evaluations == [EVALUATION]
    assert clone.shadow_evaluations is not workflow.shadow_evaluations


def test_filesystem_round_trips_shadow_evaluations(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "workflow-shadow")
    )
    workflow = _workflow()
    record_id = workflow.store_workflow(provider)
    stored = provider.retrieve_by_id(
        record_id, memory_store_type=MemoryType.WORKFLOW_MEMORY
    )
    assert stored["shadow_evaluations"] == [EVALUATION]
    assert Workflow.from_dict(stored).shadow_evaluations == [EVALUATION]


def test_mongodb_round_trips_shadow_evaluations():
    mongomock = pytest.importorskip("mongomock")
    pytest.importorskip("pymongo")
    from memorizz.memory_provider.mongodb.provider import MongoDBProvider

    provider = MongoDBProvider.__new__(MongoDBProvider)
    provider.db = mongomock.MongoClient()["workflow-shadow"]
    record_id = provider.store(
        _workflow().to_dict(),
        memory_store_type=MemoryType.WORKFLOW_MEMORY,
    )
    stored = provider.retrieve_by_id(record_id, MemoryType.WORKFLOW_MEMORY)
    assert stored["shadow_evaluations"] == [EVALUATION]
