"""Host-owned personas must not become shared singleton configuration."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier

import pytest

from memorizz.memagent.managers.persona_manager import PersonaManager


def test_snapshot_isolated_between_concurrent_accounts_and_stream_workers():
    manager = PersonaManager()
    manager.set_persona({"name": "Default"}, "shared", save=False)
    barrier = Barrier(2)

    def run(name):
        with manager.use_snapshot({"name": name, "version": 4}):
            barrier.wait(timeout=5)
            with ThreadPoolExecutor(1) as workers:
                copied = workers.submit(
                    copy_context().run, manager.export_persona
                ).result()
            assert copied["name"] == name
            return manager.current_persona.name

    with ThreadPoolExecutor(2) as workers:
        assert list(workers.map(run, ["Alice", "Bob"])) == ["Alice", "Bob"]
    assert manager.current_persona.name == "Default"


def test_snapshot_read_only_and_resets_after_error():
    manager = PersonaManager()
    original = {"name": "Alice", "evolution_history": [{"version": 2}]}
    with pytest.raises(ValueError):
        with manager.use_snapshot(original):
            assert not manager.apply_update({"name": "Other"}, {"reason": "test"})[
                "updated"
            ]
            with pytest.raises(RuntimeError):
                manager.set_persona({"name": "Other"}, "shared")
            with pytest.raises(RuntimeError):
                manager.delete_persona("shared")
            manager.current_persona.evolution_history.clear()
            raise ValueError("cancelled")
    assert original["evolution_history"] == [{"version": 2}]
    assert manager.current_persona is None


def test_nested_none_snapshot_restores_outer_account():
    manager = PersonaManager()
    with manager.use_snapshot({"name": "Alice"}):
        with manager.use_snapshot(None):
            assert manager.current_persona is None
        assert manager.current_persona.name == "Alice"


def test_saving_agent_inside_snapshot_never_persists_account_persona(tmp_path):
    from memorizz import MemAgent
    from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

    class Model:
        def get_config(self):
            return {"provider": "openai", "model": "synthetic"}

    provider = FileSystemProvider(
        FileSystemConfig(root_path=str(tmp_path), use_faiss=False)
    )
    agent = MemAgent(
        model=Model(),
        memory_provider=provider,
        memory_types=[],
        auto_register=False,
        automations_enabled=False,
        context_policy={"progressive_tool_disclosure": False},
    )
    with agent.persona_manager.use_snapshot(
        {"name": "Private account persona", "version": 7}
    ):
        assert agent.persona_manager.current_persona.name == "Private account persona"
        agent.save()
    saved = provider.retrieve_memagent(agent.agent_id)
    assert saved.persona is None


def test_trace_pins_the_persona_version_used_by_the_turn(memagent_with_mocks):
    agent = memagent_with_mocks
    events = []
    agent.set_stream_event_callback(events.append)
    with agent.persona_manager.use_snapshot(
        {"persona_id": "account-persona", "version": 7}
    ):
        agent._emit_trace_turn_start()
    from memorizz.observability.references import persona_reference

    assert (
        events[-1]["persona_id"]
        == persona_reference({"persona_id": "account-persona", "version": 7})["ref"]
    )
    assert events[-1]["persona_version"] == 7
    from memorizz.observability.normalization import LEGACY_FIELDS

    assert {"persona_id", "persona_version"} <= LEGACY_FIELDS


def test_persona_identity_survives_persisted_trace_compaction(memagent_with_mocks):
    from memorizz.observability.references import persona_reference

    agent = memagent_with_mocks
    snapshot = {"persona_id": "account-persona", "version": 7}
    with agent.persona_manager.use_snapshot(snapshot):
        agent._begin_trace_turn("account")
        rows = agent._build_trace_bundle_events(agent._stream_trace_events)
    start = next(row for row in rows if row["trace_kind"] == "turn_start")
    assert start.get("persona_id") == persona_reference(snapshot)["ref"]
    assert start.get("persona_version") == 7


def test_cache_identity_includes_actual_goals_even_after_version_reset(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    with agent.persona_manager.use_snapshot(
        {"persona_id": "account", "version": 2, "goals": "Use examples."}
    ):
        before = agent._semantic_cache_metadata()["fingerprints"]["prompt"]
    with agent.persona_manager.use_snapshot(
        {"persona_id": "account", "version": 2, "goals": "Use short definitions."}
    ):
        after = agent._semantic_cache_metadata()["fingerprints"]["prompt"]
    assert before != after


def test_persona_is_versioned_supplied_style_not_claimed_behavior(memagent_with_mocks):
    import json

    from memorizz.observability.references import persona_reference

    agent, events = memagent_with_mocks, []
    agent.set_stream_event_callback(events.append)
    snapshot = {
        "persona_id": "private-account@example.invalid",
        "version": 9,
        "goals": "Private preference: teach using diagrams",
    }
    with agent.persona_manager.use_snapshot(snapshot):
        agent._begin_trace_turn("account")
        agent._emit_memory_context_trace({})
        agent._emit_memory_reference_trace("Here is a definition.")
    supply = next(e for e in events if e.get("trace_kind") == "memory_context")
    assert supply["input_refs"] == [persona_reference(snapshot)]
    assert supply["input_refs"][0]["version"] == "9"
    assert supply["memory_supplied_count"] == 1
    payload = json.loads(supply["content"])
    assert payload["source_counts"]["persona_snapshots"] == 1
    assert payload["retrieved_candidate_count"] == 0  # It was bound, not searched.
    assert "private-account" not in json.dumps(supply)
    assert "teach using diagrams" not in json.dumps(supply)
    reference = next(e for e in events if e.get("trace_kind") == "memory_reference")
    assert json.loads(reference["content"])["behavioral_sources_not_measured"] == 1
    assert reference["memory_referenced_count"] == 0
    with agent.persona_manager.use_snapshot(None):
        agent._emit_memory_context_trace({})
        assert events[-1]["memory_supplied_count"] == 0


def test_persona_ref_matches_host_dictionary_and_native_snapshot():
    from memorizz.long_term.semantic.persona import Persona
    from memorizz.observability.references import persona_reference

    snapshot = {"persona_id": "account", "version": 4, "goals": "Use examples"}
    assert persona_reference(snapshot) == persona_reference(Persona.from_dict(snapshot))


def test_actual_prompt_assembly_preserves_persona_usage_measurement(
    memagent_with_mocks,
):
    agent = memagent_with_mocks
    with agent.persona_manager.use_snapshot(
        {"persona_id": "account", "goals": "Always use a concrete example."}
    ):
        system = agent._build_system_prompt()
        agent._build_prompt_messages(system, "Explain overfitting", {})
        assert agent._memory_usage_chars["persona"] == agent._rendered_persona_chars > 0
    with agent.persona_manager.use_snapshot(None):
        system = agent._build_system_prompt()
        agent._build_prompt_messages(system, "Explain overfitting", {})
        assert "persona" not in agent._memory_usage_chars
