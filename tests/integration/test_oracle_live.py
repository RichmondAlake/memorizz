# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""What only a real Oracle shows: run these after changing the Oracle provider.

    MEMORIZZ_TEST_ORACLE_USER=... MEMORIZZ_TEST_ORACLE_PASSWORD=... \\
    MEMORIZZ_TEST_ORACLE_DSN=127.0.0.1:1529/FREEPDB1 \\
    pytest tests/integration/test_oracle_live.py -m integration

They skip without those variables. Rows are tagged ``memorizz-live-test`` and
removed. Embeddings are stubbed to MEMORIZZ_TEST_ORACLE_DIMENSIONS (768).
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration

ENV = ("USER", "PASSWORD", "DSN")
settings = {name: os.environ.get(f"MEMORIZZ_TEST_ORACLE_{name}") for name in ENV}
if not all(settings.values()):
    pytest.skip(
        "Needs MEMORIZZ_TEST_ORACLE_USER, _PASSWORD and _DSN", allow_module_level=True
    )

oracledb = pytest.importorskip("oracledb")

TAG = "memorizz-live-test"
DIMENSIONS = int(os.environ.get("MEMORIZZ_TEST_ORACLE_DIMENSIONS", "768"))


def _provider():
    from memorizz.memory_provider.oracle.provider import OracleConfig, OracleProvider

    return OracleProvider(
        OracleConfig(
            user=settings["USER"],
            password=settings["PASSWORD"],
            dsn=settings["DSN"],
            lazy_vector_indexes=True,
            in_database_embedding=False,
        )
    )


@pytest.fixture()
def provider(monkeypatch):
    monkeypatch.setattr(
        "memorizz.long_term.semantic.persona.persona.get_embedding",
        lambda text, **kwargs: [0.1] * DIMENSIONS,
    )
    store = _provider()
    yield store
    with oracledb.connect(
        user=settings["USER"], password=settings["PASSWORD"], dsn=settings["DSN"]
    ) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM personas WHERE name = :tag", {"tag": TAG})
        cursor.execute("DELETE FROM automation_jobs WHERE name = :tag", {"tag": TAG})
        connection.commit()


def _agent(provider):
    from memorizz.long_term.semantic.persona.persona import Persona
    from memorizz.memagent import MemAgent

    agent = MemAgent(
        name=TAG,
        llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
        memory_provider=provider,
        memory_ids=[TAG],
        persona=Persona(name=TAG, role="Harbour pilot", goals="Help."),
        persona_learning={"every_n_turns": 7},
    )
    agent.save()
    return agent


def test_a_persona_keeps_its_history_through_a_reload_and_a_save(provider):
    from memorizz.memagent import MemAgent

    agent = _agent(provider)
    try:
        persona = provider.retrieve_memagent(agent.agent_id).persona
        persona.update({"goals": "Help ships."}, {"reason": "test"}, provider)

        reloaded = MemAgent.load(agent.agent_id, memory_provider=_provider())
        reloaded.save()

        stored = _provider().retrieve_memagent(agent.agent_id)
        assert stored.persona.version == 2
        assert len(stored.persona.evolution_history) == 1
        assert stored.persona_learning["every_n_turns"] == 7
    finally:
        provider.delete_memagent(agent.agent_id)


def test_a_replaced_persona_doesnt_come_back(provider):
    from memorizz.long_term.semantic.persona.persona import Persona
    from memorizz.memagent import MemAgent

    agent = _agent(provider)
    try:
        first = provider.retrieve_memagent(agent.agent_id).persona.persona_id
        agent.set_persona(Persona(name=TAG, goals="Replacement."), save=False)
        agent.save()

        loaded = MemAgent.load(agent.agent_id, memory_provider=_provider())
        current = loaded.persona_manager.current_persona
        assert current.persona_id != first
        assert "Replacement." in current.goals
        with oracledb.connect(
            user=settings["USER"], password=settings["PASSWORD"], dsn=settings["DSN"]
        ) as connection:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT agent_id FROM personas WHERE persona_id = :id", {"id": first}
            )
            assert cursor.fetchone()[0] is None
    finally:
        provider.delete_memagent(agent.agent_id)


def test_a_save_doesnt_undo_a_change_made_elsewhere(provider):
    from memorizz.memagent import MemAgent, persistence

    running = _agent(provider)
    try:
        other = _provider()
        persona = other.retrieve_memagent(running.agent_id).persona
        persona.update({"goals": "Changed elsewhere."}, {"reason": "ui"}, other)
        persistence.update_agent_record(other, running.agent_id, persona=persona)

        running.save()

        stored = MemAgent.load(running.agent_id, memory_provider=_provider())
        current = stored.persona_manager.current_persona
        assert current.goals == "Changed elsewhere."
        assert current.version == 2
    finally:
        provider.delete_memagent(running.agent_id)


def test_persona_learning_state_is_kept_on_the_record(provider):
    from memorizz.memagent import persistence

    agent = _agent(provider)
    try:
        persistence.update_agent_record(
            provider,
            agent.agent_id,
            persona_learning_state={"reviews": [{"code": "x" * 500}] * 100},
        )
        agent.save()
        state = _provider().retrieve_memagent(agent.agent_id).persona_learning_state
        assert len(state["reviews"]) == 100
    finally:
        provider.delete_memagent(agent.agent_id)


def test_a_job_can_be_read_whichever_way_its_time_was_stored(provider):
    from memorizz.automation.models import AutomationJob
    from memorizz.automation.schedule import compute_next_run_at, utcnow
    from memorizz.automation.store.oracle import OracleAutomationStore

    store = OracleAutomationStore(provider)
    now = utcnow()
    new = store.create_job(
        AutomationJob(
            job_id=str(uuid.uuid4()),
            agent_id=TAG,
            name=TAG,
            schedule_type="one_shot",
            interval_seconds=1500,
            timezone="UTC",
            next_run_at=compute_next_run_at(
                schedule_type="one_shot",
                cron_expr=None,
                interval_seconds=1500,
                tz_name="UTC",
                after_utc=now,
            ),
            action_type="agent_query",
            action_config={"query_template": "hello"},
        )
    )
    assert 1490 < (new.next_run_at - now).total_seconds() < 1510

    # What the code wrote before it used an offset: thin mode can't fetch it.
    old_id = str(uuid.uuid4())
    with oracledb.connect(
        user=settings["USER"], password=settings["PASSWORD"], dsn=settings["DSN"]
    ) as connection:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO automation_jobs (job_id, agent_id, name, enabled, "
            "schedule_type, timezone, next_run_at, misfire_policy, "
            "max_run_seconds, retry_max_attempts, retry_backoff_seconds, action_type) "
            "VALUES (:job, :tag, :tag, 1, 'one_shot', 'UTC', "
            "FROM_TZ(TIMESTAMP '2026-01-01 08:30:00', 'UTC'), 'skip', 900, 1, 60, "
            "'agent_query')",
            {"job": old_id, "tag": TAG},
        )
        connection.commit()
    old = store.get_job(old_id)
    assert old.next_run_at.isoformat() == "2026-01-01T08:30:00+00:00"
    claimed = store.claim_due_jobs("worker", utcnow(), 5, 60)
    assert old_id in [job.job_id for job in claimed]
