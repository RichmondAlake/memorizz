"""Entity memory: confirmations of a stored value, and its two clocks."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List

import pytest

from memorizz.long_term.semantic.entity_memory import (
    EntityMemory,
    attribute_timeline,
    entity_attribute_timeline,
    value_as_of,
)
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

VECTOR = [1.0, 0.0, 0.0]
SCOPE = {"memory_id": "memory-1", "user_id": "priya"}


@pytest.fixture()
def entities(monkeypatch):
    calls: List[str] = []

    def embed(text):
        calls.append(text)
        return list(VECTOR)

    monkeypatch.setattr(
        "memorizz.long_term.semantic.entity_memory.entity_memory.get_embedding", embed
    )
    path = Path(tempfile.mkdtemp(prefix="memorizz-clocks-"))

    class _Embedder:
        def get_embedding(self, text):
            return list(VECTOR)

    memory = EntityMemory(
        FileSystemProvider(
            FileSystemConfig(
                root_path=path / "memory",
                embedding_provider=_Embedder(),
                use_faiss=False,
                lazy_vector_indexes=True,
            )
        )
    )
    memory.embedding_calls = calls
    try:
        yield memory
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _attribute(entities, name, attribute):
    record = entities.get_entity_by_name(name, **SCOPE)
    return next(a for a in record["attributes"] if a["name"] == attribute)


# ---------------------------------------------------------------- reinforce


def test_a_weaker_source_saying_the_same_thing_confirms_it(entities):
    entities.upsert_entity(
        name="CFU",
        attributes=[{"name": "office", "value": "Leeds", "valid_from": "2026-09-01"}],
        **SCOPE,
    )
    before = _attribute(entities, "CFU", "office")
    report = entities.upsert_entity_with_report(
        name="CFU",
        attributes=[
            {"name": "office", "value": "leeds", "source": "turn", "thread_id": "t2"}
        ],
        source_level=2,
        **SCOPE,
    )
    after = _attribute(entities, "CFU", "office")
    assert report["unchanged"] is False and report["superseded"] == []
    assert after["confirmations"] == 1 and after["last_seen"]
    assert after["confirmed_by"][0]["thread_id"] == "t2"
    assert after["confirmed_by"][0]["level"] == 2
    # The value, its level, its valid-from and when it was recorded all stay.
    for field in ("value", "level", "valid_from", "created_at"):
        assert after[field] == before[field]
    assert not after.get("history")
    row = attribute_timeline(after)[0]
    assert row["confirmations"] == 1 and row["last_seen"] == after["last_seen"]


def test_the_same_place_repeating_itself_is_counted_once(entities):
    fact = [{"name": "office", "value": "Leeds", "source": "turn", "thread_id": "t1"}]
    entities.upsert_entity(name="CFU", attributes=fact, source_level=2, **SCOPE)
    again = entities.upsert_entity_with_report(
        name="CFU", attributes=fact, source_level=2, **SCOPE
    )
    assert again["unchanged"] is True
    assert "confirmations" not in _attribute(entities, "CFU", "office")
    # Another conversation saying it is a confirmation, once.
    other = [{"name": "office", "value": "Leeds", "source": "turn", "thread_id": "t2"}]
    for _ in range(2):
        entities.upsert_entity(name="CFU", attributes=other, source_level=2, **SCOPE)
    assert _attribute(entities, "CFU", "office")["confirmations"] == 1


def test_reinforce_attribute_counts_once_per_source_and_can_be_withdrawn(entities):
    entities.upsert_entity(
        name="CFU", attributes=[{"name": "office", "value": "Leeds"}], **SCOPE
    )
    assert entities.reinforce_attribute(
        "office", name="CFU", source="consolidation:p1", thread_id="t9", **SCOPE
    )
    # The same source does not count twice, and a stale value is not counted.
    assert not entities.reinforce_attribute(
        "office", name="CFU", source="consolidation:p1", **SCOPE
    )
    assert not entities.reinforce_attribute(
        "office", name="CFU", value="Manchester", source="x", **SCOPE
    )
    assert _attribute(entities, "CFU", "office")["confirmations"] == 1
    assert entities.withdraw_confirmations("consolidation:p1", **SCOPE) == 1
    cleared = _attribute(entities, "CFU", "office")
    assert "confirmations" not in cleared and "confirmed_by" not in cleared


def test_a_new_value_starts_with_no_confirmations_and_forgetting_takes_them(entities):
    entities.upsert_entity(
        name="CFU", attributes=[{"name": "office", "value": "Leeds"}], **SCOPE
    )
    entities.upsert_entity(
        name="CFU",
        attributes=[
            {"name": "office", "value": "Leeds", "source": "turn", "thread_id": "t2"}
        ],
        source_level=2,
        **SCOPE,
    )
    entities.forget_thread("t2", **SCOPE)
    assert "confirmations" not in _attribute(entities, "CFU", "office")
    entities.reinforce_attribute("office", name="CFU", source="a", **SCOPE)
    entities.upsert_entity(
        name="CFU", attributes=[{"name": "office", "value": "Manchester"}], **SCOPE
    )
    moved = _attribute(entities, "CFU", "office")
    assert "confirmations" not in moved
    assert moved["history"][0]["confirmations"] == 1  # the old value keeps its count


# -------------------------------------------------------------- two clocks


def test_a_change_ends_the_old_value_when_the_new_one_began(entities):
    entities.upsert_entity(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£2,500", "valid_from": "2026-09-01"}],
        **SCOPE,
    )
    report = entities.upsert_entity_with_report(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£3,000", "valid_from": "2027-01-01"}],
        **SCOPE,
    )
    assert report["superseded"][0]["old_value"] == "£2,500"
    current, earlier = attribute_timeline(_attribute(entities, "TPA", "grade_1"))
    assert current["valid_from"] == "2027-01-01" and current["recorded_at"]
    assert earlier["valid_from"] == "2026-09-01" and earlier["valid_to"] == "2027-01-01"
    assert earlier["corrected"] is False
    # Recorded time is when the system learned and stopped believing it.
    assert earlier["recorded_at"] != earlier["valid_from"]
    assert earlier["superseded_at"] == current["recorded_at"]

    def ask(when=None, **kwargs):
        row = entities.value_as_of("grade_1", when, name="TPA", **kwargs, **SCOPE)
        return row and row["value"]

    assert ask("2026-10-15") == "£2,500"
    assert ask("2027-02-01") == "£3,000"
    assert ask("2026-08-01") is None  # before anything was true
    assert ask() == "£3,000"


def test_a_correction_marks_the_old_value_as_never_true(entities):
    entities.upsert_entity(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£2,000", "valid_from": "2026-09-01"}],
        **SCOPE,
    )
    learned = _attribute(entities, "TPA", "grade_1")["created_at"]
    entities.upsert_entity(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£2,500", "correction": True}],
        **SCOPE,
    )
    attribute = _attribute(entities, "TPA", "grade_1")
    assert "correction" not in attribute
    current, wrong = attribute_timeline(attribute)
    # The right value covers the period the wrong one claimed.
    assert current["valid_from"] == "2026-09-01"
    assert wrong["corrected"] is True and wrong["valid_to"] is None
    assert wrong["corrected_at"] == wrong["superseded_at"] == current["recorded_at"]
    rows = attribute_timeline(attribute)
    # By valid time the wrong value was never true...
    assert value_as_of(rows, "2026-10-01")["value"] == "£2,500"
    # ...but it is what the system believed before the correction.
    assert value_as_of(rows, "2026-10-01", known_at=learned)["value"] == "£2,000"
    assert value_as_of(rows, known_at=learned)["value"] == "£2,000"
    assert value_as_of(rows, known_at="2020-01-01") is None


def test_the_same_valid_from_is_a_correction_and_an_earlier_one_is_backfilled(entities):
    entities.upsert_entity(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£3,000", "valid_from": "2027-01-01"}],
        **SCOPE,
    )
    report = entities.upsert_entity_with_report(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£2,500", "valid_from": "2026-09-01"}],
        **SCOPE,
    )
    assert report["superseded"] == []  # the current value stays
    attribute = _attribute(entities, "TPA", "grade_1")
    assert attribute["value"] == "£3,000"
    rows = attribute_timeline(attribute)
    assert rows[1]["backfilled"] and rows[1]["valid_to"] == "2027-01-01"
    assert value_as_of(rows, "2026-10-01")["value"] == "£2,500"
    assert value_as_of(rows, "2027-03-01")["value"] == "£3,000"
    entities.upsert_entity(
        name="TPA",
        attributes=[{"name": "grade_1", "value": "£3,250", "valid_from": "2027-01-01"}],
        **SCOPE,
    )
    rows = attribute_timeline(_attribute(entities, "TPA", "grade_1"))
    assert rows[0]["value"] == "£3,250" and rows[1]["corrected"] is True
    assert value_as_of(rows, "2027-03-01")["value"] == "£3,250"


def test_records_written_before_the_clocks_still_read(entities):
    legacy = {
        "name": "office",
        "value": "Manchester",
        "updated_at": "2026-03-01T00:00:00+00:00",
        "history": [
            {
                "value": "Leeds",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "valid_from": "2026-01-01T00:00:00+00:00",
                "valid_to": "2026-03-01T00:00:00+00:00",
            },
            {
                "value": "York",
                "updated_at": "2026-02-01T00:00:00+00:00",
                "applied": False,
            },
        ],
    }
    rows = attribute_timeline(legacy)
    assert [row["confirmations"] for row in rows] == [0, 0, 0]
    assert rows[1]["recorded_at"] == "2026-01-01T00:00:00+00:00"
    assert rows[1]["superseded_at"] == "2026-03-01T00:00:00+00:00"
    assert value_as_of(rows, "2026-02-01")["value"] == "Leeds"
    assert value_as_of(rows, "2026-04-01")["value"] == "Manchester"
    assert value_as_of(rows, known_at="2026-02-15")["value"] == "Leeds"


def test_a_retired_value_is_part_of_the_entity_timeline(entities):
    entity_id = entities.upsert_entity(
        name="CFU", attributes=[{"name": "office", "value": "Leeds"}], **SCOPE
    )
    entities.retire_attributes(
        entity_id,
        retirements=[{"name": "office", "reason": "replaced"}],
        retired_by="plan-1",
        **SCOPE,
    )
    entities.upsert_entity(
        name="CFU",
        attributes=[{"name": "office", "value": "Manchester"}],
        source_level=1,
        **SCOPE,
    )
    record = entities.get_entity_by_name("CFU", **SCOPE)
    rows = entity_attribute_timeline(record, "office")
    assert [row["value"] for row in rows] == ["Manchester", "Leeds"]
    assert rows[1]["retired"] and rows[1]["valid_to"] and not rows[1]["current"]
    assert entity_attribute_timeline(record, "missing") == []
