"""Portable graph restore, identity remapping, isolation and integrity."""

import copy
import json
import os
import uuid

import pytest

from memorizz import MemoryArchive, MemoryArchiveError, MemoryHistory, MemoryType
from memorizz.memory_archive import TAXONOMY, read_archive, seal_archive, write_archive
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def provider(path):
    return FileSystemProvider(FileSystemConfig(root_path=path, use_faiss=False))


def populate(store):
    from memorizz.memagent.models import MemAgentModel

    root, child = str(uuid.uuid4()), str(uuid.uuid4())
    store.store_memagent(
        MemAgentModel(
            agent_id=child,
            name="Researcher",
            memory_ids=["private"],
            llm_config={"provider": "ollama", "model": "test"},
        )
    )
    store.store_memagent(
        MemAgentModel(
            agent_id=root,
            name="Coordinator",
            memory_ids=["team"],
            delegates=[child],
            llm_config={
                "provider": "ollama",
                "model": "test",
                "api_key": "DO-NOT-EXPORT",
            },
        )
    )
    ids = {}
    for kind in MemoryType:
        if kind == MemoryType.MEMAGENT:
            continue
        identifier = str(uuid.uuid4())
        data = {
            "_id": identifier,
            "id": identifier,
            "memory_id": "team",
            "agent_id": root,
            "content": f"{kind.value} facts",
            "name": "Memory example",
            "user_id": None,
            "created_at": "2026-10-07T10:00:00+00:00",
            "embedding": [1.0, 0.0],
        }
        from memorizz.memory_archive import ID_FIELDS

        if kind in ID_FIELDS:
            data[ID_FIELDS[kind]] = identifier
        data.update(
            {
                MemoryType.PERSONAS: {
                    "background": "Research",
                    "goals": ["Explain provenance"],
                },
                MemoryType.TOOLBOX: {
                    "parameters": {"type": "object", "properties": {}},
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                },
                MemoryType.CONVERSATION_MEMORY: {
                    "role": "user",
                    "timestamp": "2026-10-07T10:00:00+00:00",
                    "thread_id": "turn-thread",
                },
                MemoryType.SHARED_MEMORY: {
                    "content": json.dumps(
                        {
                            "root_agent_id": root,
                            "delegate_agent_ids": [child],
                            "blackboard": [
                                {"agent_id": child, "content": "Shared evidence"}
                            ],
                        }
                    ),
                    "owner_agent_id": root,
                    "scope": "private",
                },
                MemoryType.SUMMARIES: {
                    "source_message_ids": [ids.get(MemoryType.CONVERSATION_MEMORY)]
                },
                MemoryType.SEMANTIC_CACHE: {
                    "cache_key": identifier,
                    "query_text": "Explain",
                    "response": "Evidence",
                    "scope": "local",
                },
                MemoryType.TOOL_LOG: {
                    "tool_name": "lookup",
                    "arguments": {"q": "demo"},
                    "result": {"found": True},
                },
            }.get(kind, {})
        )
        ids[kind] = store.store(data, memory_store_type=kind)
    private = store.store(
        {
            "content": "Private researcher note",
            "agent_id": child,
            "memory_id": "private",
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    unrelated = store.store(
        {"content": "Other agent", "agent_id": "outside", "memory_id": "team"},
        MemoryType.KNOWLEDGE_BASE,
    )
    return root, child, ids, private, unrelated


def test_all_taxonomy_graph_roundtrip(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    root, child, ids, private, unrelated = populate(source)
    archive = MemoryArchive(source).export(agent_id=root)
    assert set(archive["stores"]) == set(TAXONOMY)
    assert all(archive["manifest"]["counts"].values())
    serialized = json.dumps(archive)
    assert "DO-NOT-EXPORT" not in serialized and '"embedding"' not in serialized
    assert "Private researcher note" in serialized and "Other agent" not in serialized
    service = MemoryArchive(target)
    preview = service.import_archive(archive)
    assert preview["ok"] and preview["imported"] == 0
    assert not target.list_all(MemoryType.KNOWLEDGE_BASE)
    result = service.import_archive(archive, dry_run=False)
    assert result["ok"] and result["imported"] == archive["manifest"]["record_count"]
    agent = target.retrieve_memagent(root)
    assert agent.delegates == [child] and not agent.automations_enabled
    assert agent.llm_config["model"] == "test"
    assert (
        target.retrieve_by_id(private, MemoryType.KNOWLEDGE_BASE)["content"]
        == "Private researcher note"
    )
    assert (
        target.retrieve_by_id(ids[MemoryType.TOOLBOX], MemoryType.TOOLBOX)["function"][
            "name"
        ]
        == "lookup"
    )
    restored = MemoryArchive(target).export(agent_id=root)
    assert restored["manifest"]["counts"] == archive["manifest"]["counts"]


def test_clone_remaps_private_shared_and_summary_links(tmp_path):
    source = provider(tmp_path / "memory")
    root, child, ids, *_ = populate(source)
    archive = MemoryArchive(source).export(agent_id=root)
    result = MemoryArchive(source).import_archive(
        archive, dry_run=False, id_strategy="new"
    )
    assert result["ok"]
    mapping = result["id_map"]
    agent = source.retrieve_memagent(mapping[root])
    delegate = source.retrieve_memagent(mapping[child])
    assert agent.delegates == [mapping[child]]
    assert agent.memory_ids == [mapping["team"]] and delegate.memory_ids == [
        mapping["private"]
    ]
    assert agent.memory_ids != delegate.memory_ids
    summary = source.retrieve_by_id(
        mapping[ids[MemoryType.SUMMARIES]], MemoryType.SUMMARIES
    )
    assert summary["source_message_ids"] == [
        mapping[ids[MemoryType.CONVERSATION_MEMORY]]
    ]
    shared = source.retrieve_by_id(
        mapping[ids[MemoryType.SHARED_MEMORY]], MemoryType.SHARED_MEMORY
    )
    assert shared["content"]["root_agent_id"] == mapping[root]
    assert shared["content"]["blackboard"][0]["agent_id"] == mapping[child]


def test_restore_creates_agents_before_personas_then_binds_configuration(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    root, _child, ids, *_ = populate(source)
    agent = source.retrieve_memagent(root)
    agent.persona = {
        "persona_id": ids[MemoryType.PERSONAS],
        "background": "Research",
        "goals": ["Explain provenance"],
    }
    source.store_memagent(agent)
    save = target.store_archive_record
    bindings = []

    def relational_save(kind, identifier, data, **options):
        if kind == MemoryType.PERSONAS:
            assert target.retrieve_memagent(data["agent_id"]) is not None
        if kind == MemoryType.MEMAGENT and data.get("persona"):
            persona_id = data["persona"]["persona_id"]
            assert target.retrieve_by_id(persona_id, MemoryType.PERSONAS) is not None
            bindings.append(persona_id)
        return save(kind, identifier, data, **options)

    target.store_archive_record = relational_save
    report = MemoryArchive(target).import_archive(
        MemoryArchive(source).export(agent_id=root), dry_run=False, id_strategy="new"
    )
    assert report["ok"]
    restored = target.retrieve_memagent(report["id_map"][root])
    assert bindings == [report["id_map"][ids[MemoryType.PERSONAS]]]
    assert restored.persona.persona_id == bindings[0]
    assert restored.persona.goals == ["Explain provenance"]


def test_history_scoped_export_and_restored_timeline(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    history = MemoryHistory(source)
    with history.recording(actor="writer", user_id="alice", memory_id="project"):
        identifier = source.store(
            {"content": "Old", "memory_id": "project", "user_id": "alice"},
            MemoryType.KNOWLEDGE_BASE,
        )
        source.update_by_id(identifier, {"content": "New"}, MemoryType.KNOWLEDGE_BASE)
    archive = MemoryArchive(source).export(memory_id="project", user_id="alice")
    assert archive["manifest"]["counts"]["shared_memory"] == 2
    assert not any(
        r["data"]["content"]["record_type"] == "memory_history_head"
        for r in archive["stores"]["shared_memory"]
    )
    report = MemoryArchive(target).import_archive(
        archive, dry_run=False, user_id="alice"
    )
    assert report["ok"]
    events = MemoryHistory(target).timeline(memory_id="project", user_id="alice")[
        "events"
    ]
    assert len(events) == 2 and events[1]["previous_event_id"] == events[0]["record_id"]
    assert events[1]["actor"] == "writer"
    assert "Old" not in json.dumps(archive)


def test_conflicts_are_preflighted_and_skip_replace_are_explicit(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    identifier = source.store({"content": "New"}, MemoryType.KNOWLEDGE_BASE)
    target.store({"_id": identifier, "content": "Existing"}, MemoryType.KNOWLEDGE_BASE)
    archive = MemoryArchive(source).export()
    service = MemoryArchive(target)
    assert not service.import_archive(archive)["ok"]
    with pytest.raises(MemoryArchiveError, match="already exist"):
        service.import_archive(archive, dry_run=False)
    assert (
        service.import_archive(archive, dry_run=False, conflict="skip")["imported"] == 0
    )
    assert (
        target.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"]
        == "Existing"
    )
    assert service.import_archive(archive, dry_run=False, conflict="replace")["ok"]
    assert (
        target.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"] == "New"
    )


def test_tenant_collision_and_unknown_version_never_write(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    identifier = source.store(
        {"content": "Alice", "user_id": "alice"}, MemoryType.KNOWLEDGE_BASE
    )
    target.store(
        {"_id": identifier, "content": "Bob", "user_id": "bob"},
        MemoryType.KNOWLEDGE_BASE,
    )
    archive = MemoryArchive(source).export(user_id="alice")
    with pytest.raises(MemoryArchiveError, match="another scope"):
        MemoryArchive(target).import_archive(
            archive, dry_run=False, conflict="replace", user_id="alice"
        )
    with pytest.raises(MemoryArchiveError, match="authorized user"):
        MemoryArchive(target).import_archive(archive, user_id="bob")
    archive["version"] = 999
    with pytest.raises(MemoryArchiveError, match="version 1"):
        MemoryArchive(target).import_archive(seal_archive(archive), dry_run=False)
    assert (
        target.retrieve_by_id(identifier, MemoryType.KNOWLEDGE_BASE)["content"] == "Bob"
    )


def test_integrity_duplicates_limits_and_private_file(tmp_path):
    source = provider(tmp_path / "memory")
    source.store({"content": "Hello"}, MemoryType.KNOWLEDGE_BASE)
    archive = MemoryArchive(source).export()
    path = tmp_path / "backup.memorizz.json"
    write_archive(archive, path)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert read_archive(path) == archive
    with pytest.raises(FileExistsError):
        write_archive(archive, path)
    archive["stores"]["knowledge_base"][0]["data"]["content"] = "Modified"
    with pytest.raises(MemoryArchiveError, match="checksum"):
        MemoryArchive(source).import_archive(archive)
    archive = seal_archive(archive)
    archive["stores"]["knowledge_base"].append(
        copy.deepcopy(archive["stores"]["knowledge_base"][0])
    )
    with pytest.raises(MemoryArchiveError, match="duplicate"):
        MemoryArchive(source).import_archive(seal_archive(archive))
    with pytest.raises(MemoryArchiveError, match="record limit"):
        MemoryArchive(source).export(max_records=0)


def test_destination_failure_reports_partial_progress(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    for text in ("First", "Second", "Third"):
        source.store({"content": text}, MemoryType.KNOWLEDGE_BASE)
    save = target.store_archive_record
    calls = []

    def fail(kind, identifier, data, **kwargs):
        calls.append(identifier)
        if len(calls) == 2:
            raise RuntimeError("Sensitive provider message must not escape")
        return save(kind, identifier, data, **kwargs)

    target.store_archive_record = fail
    report = MemoryArchive(target).import_archive(
        MemoryArchive(source).export(), dry_run=False
    )
    assert not report["ok"] and report["imported"] == 1 and not report["atomic"]
    assert "Sensitive" not in json.dumps(report)


def test_new_ids_do_not_rewrite_memory_prose(tmp_path):
    source = provider(tmp_path / "source")
    identifier = source.store(
        {
            "_id": "id-123",
            "memory_id": "source-project",
            "content": "Literal id-123 source-project",
            "supersedes": "deleted",
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    archive = MemoryArchive(source).export(memory_id="source-project")
    report = MemoryArchive(source).import_archive(
        archive, dry_run=False, id_strategy="new", target_memory_id="target-project"
    )
    row = source.retrieve_by_id(report["id_map"][identifier], MemoryType.KNOWLEDGE_BASE)
    assert row["memory_id"] == "target-project"
    assert (
        row["content"] == "Literal id-123 source-project"
        and row["supersedes"] == "deleted"
    )


def test_import_records_who_restored_final_delegate_configuration(tmp_path):
    source, target = provider(tmp_path / "source"), provider(tmp_path / "target")
    root, child, *_ = populate(source)
    with MemoryHistory(target).recording(actor="restore-operator"):
        report = MemoryArchive(target).import_archive(
            MemoryArchive(source).export(agent_id=root), dry_run=False
        )
    assert report["ok"]
    changes = MemoryHistory(target).timeline(agent_id=root, memory_type="agents")[
        "events"
    ]
    assert len(changes) == 1
    assert (
        changes[0]["source"] == "archive_import"
        and changes[0]["actor"] == "restore-operator"
    )
    assert target.retrieve_memagent(root).delegates == [child]


def test_checksum_survives_browser_number_serialization(tmp_path):
    from memorizz.memory_archive import _canonical, validate_archive

    source = provider(tmp_path / "source")
    source.store(
        {
            "content": "Numeric facts",
            "importance": 1.0,
            "latency": 0.000001,
            "tiny": 1e-7,
            "large": 1e20,
            "zero": -0.0,
        },
        MemoryType.KNOWLEDGE_BASE,
    )
    archive = MemoryArchive(source).export()
    uploaded = copy.deepcopy(archive)
    record = uploaded["stores"]["knowledge_base"][0]["data"]
    record.update(
        importance=1, latency=0.000001, tiny=1e-7, large=100000000000000000000, zero=0
    )
    assert validate_archive(uploaded)
    assert (
        _canonical([1.0, -0.0, 1e-6, 1e-7, 1e20, 1e21])
        == b"[1,0,0.000001,1e-7,100000000000000000000,1e+21]"
    )
