"""Create a real, isolated local-agent memory evolution example.

Uses Ollama for generation/embeddings and a fresh namespace on filesystem or
Oracle. Oracle credentials come only from the environment. Never clears stores.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlencode

from memorizz import MemAgent, MemoryHistory, MemoryType
from memorizz.embeddings import configure_embeddings
from memorizz.llms.ollama import OllamaLLM


def create_provider(args):
    """Use the same local embedding setup for both evolution examples."""
    configure_embeddings("ollama", {"model": "nomic-embed-text"})
    if args.backend == "oracle":
        from memorizz.memory_provider.oracle import OracleConfig, OracleProvider

        provider = OracleProvider(
            OracleConfig(
                user=os.environ["ORACLE_USER"],
                password=os.environ["ORACLE_PASSWORD"],
                dsn=os.environ["ORACLE_DSN"],
                in_database_embedding=False,
                embedding_provider="ollama",
                embedding_config={"model": "nomic-embed-text"},
            )
        )
    else:
        from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

        provider = FileSystemProvider(
            FileSystemConfig(root_path=Path(args.store), use_faiss=False)
        )
    return provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["filesystem", "oracle"], default="filesystem"
    )
    parser.add_argument("--store", default="/tmp/memorizz-evolution/memory")
    parser.add_argument("--portal", default="http://127.0.0.1:8766")
    parser.add_argument("--output", default="/tmp/memorizz-evolution/evidence.json")
    args = parser.parse_args()
    provider = create_provider(args)
    agent = MemAgent(
        agent_id=str(uuid.uuid4()),
        name="Harbor · Memory evolution · " + args.backend,
        model=OllamaLLM(
            model="qwen2.5:7b", context_window_tokens=16384, num_predict=1024
        ),
        memory_provider=provider,
        instruction="You track a fictional release. Reply briefly using conversation history and explicit corrections. Do not call tools unless asked.",
        memory_types=[
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.SUMMARIES,
            MemoryType.KNOWLEDGE_BASE,
        ],
        capture_memory_history=True,
        capture_context_snapshots=True,
        auto_register=False,
    )
    history = MemoryHistory(provider, record_changes=True)
    memory_id = "harbor-evolution-" + uuid.uuid4().hex[:10]
    agent.memory_ids = [memory_id]
    agent.save()
    scope = {"agent_id": agent.agent_id, "memory_id": memory_id}
    turns = []
    try:
        with history.recording(
            actor="demo-author", source="example:launch-brief", **scope
        ):
            brief = provider.store(
                {
                    **scope,
                    "content": "Owner Mina; launch after 18:00 UTC.",
                    "name": "Original launch brief",
                    "source_path": "fictional-launch-brief.md",
                    "metadata": {
                        "name": "Original launch brief",
                        "source_path": "fictional-launch-brief.md",
                    },
                },
                MemoryType.KNOWLEDGE_BASE,
            )
        turns.append(
            agent.run(
                "The release owner is Mina and launch must be after 18:00 UTC. Remember this and acknowledge briefly.",
                memory_id=memory_id,
            )
        )
        with history.recording(
            actor="release-owner", source="example:correction", **scope
        ):
            provider.update_by_id(
                brief,
                {"content": "Owner Mira; launch after 18:00 UTC."},
                MemoryType.KNOWLEDGE_BASE,
            )
        turns.append(
            agent.run(
                "Correction: the release owner is now Mira. Who is the current owner?",
                memory_id=memory_id,
            )
        )
        with history.recording(
            actor="release-reviewer", source="example:derived-brief", **scope
        ):
            revised = provider.store(
                {
                    **scope,
                    "content": "Owner Mira; launch after 19:00 UTC.",
                    "name": "Revised launch brief",
                    "source_path": "fictional-release-review.md",
                    "supersedes": brief,
                    "source_ids": [brief],
                    "metadata": {
                        "name": "Revised launch brief",
                        "source_path": "fictional-release-review.md",
                        "supersedes": brief,
                        "source_ids": [brief],
                    },
                },
                MemoryType.KNOWLEDGE_BASE,
            )
        turns.append(
            agent.run(
                "The launch time was revised to after 19:00 UTC. Give the current owner and earliest launch time.",
                memory_id=memory_id,
            )
        )
        with history.recording(
            actor="demo-compactor", source="example:compaction", **scope
        ):
            summaries = agent.generate_summaries(
                memory_id=memory_id,
                days_back=36500,
                keep_recent=2,
                max_memories_per_summary=200,
                summary_type="compaction",
            )
        with history.recording(
            actor="demo-cleanup", source="example:expired-note", **scope
        ):
            note = provider.store(
                {
                    **scope,
                    "content": "Temporary release checklist",
                    "name": "Temporary working note",
                },
                MemoryType.SHORT_TERM_MEMORY,
            )
            provider.delete_by_id(note, MemoryType.SHORT_TERM_MEMORY)
        agent.save()
        events = history.timeline(agent_id=agent.agent_id)["events"]
        assert all(
            e["memory_type"] in {kind.value for kind in MemoryType} for e in events
        )
        assert any(e["action"] == "updated" for e in events)
        assert any(e["action"] == "deleted" for e in events)
        assert any(e["parents"] for e in events)
        assert agent.get_context_history(memory_id)
        evidence = {
            "backend": args.backend,
            **scope,
            "brief_record_id": brief,
            "derived_record_id": revised,
            "summary_ids": summaries,
            "responses": turns,
            "changes": len(events),
            "playground": f"{args.portal}/agents/{agent.agent_id}/playground",
            "timeline": args.portal + "/traces/memory-evolution?" + urlencode(scope),
            "observability": args.portal
            + "/traces?"
            + urlencode({"agent_id": agent.agent_id}),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    finally:
        agent.close()
        provider.close()


if __name__ == "__main__":
    main()
