"""Export/restore all taxonomy types and a delegate graph, optionally to Oracle.

No model is called and no embeddings are generated. Each run uses new identities.
Run from the repository with its installed/editable Python environment.
"""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from memorizz import MemoryArchive, MemoryHistory, MemoryType
from memorizz.memagent.models import MemAgentModel
from memorizz.memory_archive import ID_FIELDS
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


def seed(provider):
    root, delegate = str(uuid.uuid4()), str(uuid.uuid4())
    namespace, private = (
        "archive-harbor-" + uuid.uuid4().hex[:10],
        "archive-private-" + uuid.uuid4().hex[:10],
    )
    ids = {
        kind: str(uuid.uuid4()) for kind in MemoryType if kind != MemoryType.MEMAGENT
    }
    now = datetime.now(timezone.utc).isoformat()
    history = MemoryHistory(provider)
    with history.recording(actor="archive-demo", source="sdk_example"):
        provider.store_memagent(
            MemAgentModel(
                agent_id=delegate,
                name="Harbor archive · Researcher",
                memory_ids=[private],
                llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
            )
        )
        provider.store_memagent(
            MemAgentModel(
                agent_id=root,
                name="Harbor archive · Coordinator",
                memory_ids=[namespace],
                delegates=[delegate],
                llm_config={"provider": "ollama", "model": "qwen2.5:7b"},
            )
        )
        for kind, identifier in ids.items():
            fields = {
                MemoryType.PERSONAS: {
                    "name": "Harbor researcher",
                    "background": "Check launch evidence",
                    "goals": ["Preserve provenance"],
                },
                MemoryType.TOOLBOX: {
                    "name": "lookup_launch",
                    "description": "Look up the launch brief",
                    "tool_type": "function",
                    "parameters": {"type": "object", "properties": {}},
                    "input_schema": {"type": "object", "properties": {}},
                },
                MemoryType.ENTITY_MEMORY: {
                    "name": "Harbor",
                    "entity_type": "project",
                    "attributes": {"owner": "Mina"},
                    "relations": [],
                    "metadata": {"source_record_ids": [ids[MemoryType.KNOWLEDGE_BASE]]},
                },
                MemoryType.SHORT_TERM_MEMORY: {
                    "content": "Check the release owner's latest correction.",
                    "ttl": 3600,
                },
                MemoryType.KNOWLEDGE_BASE: {
                    "content": "Harbor launches after 18:00 UTC; Mina owns the release.",
                    "importance": 1.0,
                    "metadata": {"source": "launch brief"},
                },
                MemoryType.CONVERSATION_MEMORY: {
                    "role": "assistant",
                    "content": "I will verify the launch owner before preparing the checklist.",
                    "thread_id": "archive-turn-" + uuid.uuid4().hex[:8],
                    "timestamp": now,
                },
                MemoryType.WORKFLOW_MEMORY: {
                    "name": "Launch verification",
                    "description": "Verify owner and release time",
                    "steps": [{"tool": "lookup_launch", "status": "complete"}],
                    "status": "completed",
                    "outcome": {"owner_verified": True},
                },
                MemoryType.SKILLBOX: {
                    "name": "Check release ownership",
                    "description": "Follow explicit source evidence",
                    "content": "Read the latest brief and confirm the owner.",
                    "source_workflow_ids": [ids[MemoryType.WORKFLOW_MEMORY]],
                    "status": "active",
                    "injection_role": "user",
                    "version": 1,
                },
                MemoryType.SHARED_MEMORY: {
                    "content": {
                        "root_agent_id": root,
                        "delegate_agent_ids": [delegate],
                        "sub_agent_ids": [],
                        "blackboard": [
                            {
                                "agent_id": delegate,
                                "content": "The current launch brief is available.",
                                "entry_type": "finding",
                                "created_at": now,
                            }
                        ],
                        "status": "active",
                        "user_id": None,
                    },
                    "scope": "private",
                    "owner_agent_id": root,
                    "access_list": [root, delegate],
                },
                MemoryType.SUMMARIES: {
                    "content": "Verify Harbor's launch owner before writing a checklist.",
                    "source_message_ids": [ids[MemoryType.CONVERSATION_MEMORY]],
                    "summary_type": "conversation",
                    "thread_id": "summary-thread",
                    "memory_units_count": 1,
                },
                MemoryType.SEMANTIC_CACHE: {
                    "cache_key": identifier,
                    "query_text": "Who owns Harbor?",
                    "response": "Check the current brief.",
                    "scope": "agent",
                    "similarity_threshold": 0.85,
                    "hit_count": 0,
                },
                MemoryType.TOOL_LOG: {
                    "tool_name": "lookup_launch",
                    "arguments": {"project": "Harbor"},
                    "result": {"brief": "available"},
                    "success": True,
                    "outcome": "success",
                    "timestamp": now,
                },
            }[kind]
            data = {
                "_id": identifier,
                "id": identifier,
                "memory_id": namespace,
                "agent_id": root,
                "user_id": None,
                "name": fields.get("name", kind.value.replace("_", " ")),
                "content": fields.get("content", "Harbor release evidence"),
                "created_at": now,
                **fields,
            }
            if kind in ID_FIELDS:
                data[ID_FIELDS[kind]] = identifier
            provider.store(data, memory_store_type=kind)
        provider.store(
            {
                "content": "Researcher's private note: confirm the corrected owner.",
                "agent_id": delegate,
                "memory_id": private,
                "user_id": None,
            },
            MemoryType.KNOWLEDGE_BASE,
        )
        with history.recording(
            actor="release-reviewer",
            source="sdk_example",
            agent_id=root,
            memory_id=namespace,
        ):
            provider.update_by_id(
                ids[MemoryType.KNOWLEDGE_BASE],
                {"content": "Harbor launches after 18:00 UTC; Mira owns the release."},
                MemoryType.KNOWLEDGE_BASE,
            )
    return {
        "agent_id": root,
        "delegate_id": delegate,
        "memory_id": namespace,
        "private_memory_id": private,
        "records": {kind.value: identifier for kind, identifier in ids.items()},
    }


def verify(provider, archive, result, source):
    assert result["ok"], result["errors"]
    mapping = result["id_map"]
    root = mapping.get(source["agent_id"], source["agent_id"])
    child = mapping.get(source["delegate_id"], source["delegate_id"])
    agent, delegate = provider.retrieve_memagent(root), provider.retrieve_memagent(
        child
    )
    assert agent.delegates == [child]
    assert agent.memory_ids != delegate.memory_ids
    assert not agent.automations_enabled
    restored = MemoryArchive(provider).export(agent_id=root)
    assert all(restored["manifest"]["counts"].values())
    brief = next(
        r["data"]
        for r in restored["stores"]["knowledge_base"]
        if "Mira owns" in r["data"].get("content", "")
    )
    assert brief["agent_id"] == root
    summary = restored["stores"]["summaries"][0]["data"]
    message_id = mapping.get(
        source["records"]["conversation_memory"],
        source["records"]["conversation_memory"],
    )
    assert summary["source_message_ids"] == [message_id]
    shared = next(
        r["data"]
        for r in restored["stores"]["shared_memory"]
        if isinstance(r["data"].get("content"), dict)
        and "blackboard" in r["data"]["content"]
    )
    assert shared["content"]["delegate_agent_ids"] == [child]
    assert shared["content"]["blackboard"][0]["agent_id"] == child
    events = MemoryHistory(provider).timeline(agent_id=root)["events"]
    assert any(e["actor"] == "release-reviewer" for e in events)
    return {
        "agent_id": root,
        "delegate_id": child,
        "memory_id": agent.memory_ids[0],
        "private_memory_id": delegate.memory_ids[0],
        "counts": restored["manifest"]["counts"],
        "history_events": len(events),
        "relationships_verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument(
        "--oracle-config",
        type=Path,
        help="Private JSON containing user/password/dsn (never copied into evidence)",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root or args.root / "source-memory"
    target_root = args.target_root or args.root / "restored-memory"
    source = FileSystemProvider(
        FileSystemConfig(root_path=source_root, use_faiss=False)
    )
    source_evidence = seed(source)
    archive = MemoryArchive(source).export(agent_id=source_evidence["agent_id"])
    filename = args.root / (source_evidence["memory_id"] + ".memorizz.json")
    from memorizz.memory_archive import write_archive

    write_archive(archive, filename)
    target = FileSystemProvider(
        FileSystemConfig(root_path=target_root, use_faiss=False)
    )
    try:
        service = MemoryArchive(target)
        assert service.import_archive(archive, id_strategy="new")["ok"]
        result = service.import_archive(archive, dry_run=False, id_strategy="new")
        filesystem = verify(target, archive, result, source_evidence)
        evidence = {
            "archive": str(filename.resolve()),
            "format": archive["format"],
            "version": archive["version"],
            "source_root": str(source_root.resolve()),
            "target_root": str(target_root.resolve()),
            "source": source_evidence,
            "filesystem": filesystem,
            "filesystem_report": result,
        }
        if args.oracle_config:
            from memorizz.memory_provider.oracle import OracleConfig, OracleProvider

            connection = json.loads(args.oracle_config.read_text())
            oracle = OracleProvider(
                OracleConfig(
                    **connection, in_database_embedding=False, index_policy="none"
                )
            )
            try:
                service = MemoryArchive(oracle)
                assert service.import_archive(archive, id_strategy="new")["ok"]
                result = service.import_archive(
                    archive, dry_run=False, id_strategy="new"
                )
                evidence["oracle"] = verify(oracle, archive, result, source_evidence)
                evidence["oracle_report"] = result
                # Verify an Oracle-produced archive can return to filesystem.
                oracle_archive = service.export(agent_id=evidence["oracle"]["agent_id"])
                roundtrip = FileSystemProvider(
                    FileSystemConfig(
                        root_path=args.root
                        / "oracle-roundtrip"
                        / source_evidence["memory_id"],
                        use_faiss=False,
                    )
                )
                try:
                    report = MemoryArchive(roundtrip).import_archive(
                        oracle_archive, dry_run=False
                    )
                    assert report["ok"], report["errors"]
                    assert (
                        MemoryArchive(roundtrip).export()["manifest"]["counts"]
                        == oracle_archive["manifest"]["counts"]
                    )
                    evidence["oracle_to_filesystem_verified"] = True
                finally:
                    roundtrip.close()
            finally:
                oracle.close()
        (args.root / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "archive": str(filename),
                    "evidence": str(args.root / "evidence.json"),
                    "filesystem": filesystem,
                    "oracle": evidence.get("oracle"),
                },
                indent=2,
            )
        )
    finally:
        source.close()
        target.close()


if __name__ == "__main__":
    main()
