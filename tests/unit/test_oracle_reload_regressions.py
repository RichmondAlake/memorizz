# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Two Oracle bugs a fake database couldn't show.

A persona's history came back from the archive as a raw JSON string, which
``Persona.from_dict`` dropped, so the next save erased it. Automation times
were stored with the region name ``'UTC'``, which the thin driver can't fetch,
so reading any job failed with DPY-3022. Both only show on a real database:
``tests/integration/test_oracle_live.py`` runs them there.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.automation.store import oracle as automation_store
from memorizz.enums import MemoryType
from memorizz.long_term.semantic.persona.persona import Persona
from memorizz.memory_provider.oracle import archive, timestamps

pytestmark = pytest.mark.unit

HISTORY = [{"version": 2, "changes": {"goals": {"old": "a", "new": "b"}}}]


class _Cursor:
    def __init__(self, names, rows):
        self.description = [(name.upper(),) for name in names]
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


def test_the_archive_gives_a_persona_its_history_as_a_list():
    cursor = _Cursor(
        ["persona_id", "version", "evolution_history"],
        [("p1", 2, json.dumps(HISTORY))],
    )
    provider = SimpleNamespace(_read_lob_value=lambda value: value)
    [record] = archive._decode(provider, MemoryType.PERSONAS, cursor)
    assert record["evolution_history"] == HISTORY


def test_a_persona_read_back_keeps_its_history():
    persona = Persona.from_dict(
        {"name": "Zed", "version": 2, "evolution_history": json.dumps(HISTORY)}
    )
    assert persona.evolution_history == HISTORY


def test_a_history_that_cant_be_read_is_no_crash():
    persona = Persona.from_dict({"name": "Zed", "evolution_history": "not json"})
    assert persona.evolution_history == []


def test_times_are_written_with_an_offset_the_thin_driver_can_read():
    assert "'+00:00'" in timestamps.utc_tstz_sql("next_run_at")
    assert "'UTC'" not in timestamps.utc_tstz_sql("next_run_at")


def test_every_automation_time_is_read_as_utc_so_old_rows_still_load():
    select = automation_store.OracleAutomationStore._select_job_columns(None)
    for column in ("start_at", "next_run_at", "last_run_at", "lock_expires_at"):
        assert f"SYS_EXTRACT_UTC({column}) AS {column}" in select
    for column in ("scheduled_for", "started_at", "finished_at"):
        assert f"SYS_EXTRACT_UTC({column}) AS {column}" in automation_store._RUN_TIMES


def test_no_time_column_is_selected_bare():
    source = Path(automation_store.__file__).read_text(encoding="utf-8")
    assert "scheduled_for, started_at, finished_at," not in source


def test_the_migration_writes_offsets_too():
    migration = (
        Path(timestamps.__file__).parent / "migrations" / "013_utc_timestamps.sql"
    )
    assert "''UTC''" not in migration.read_text(encoding="utf-8")
