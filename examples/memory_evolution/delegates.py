"""Run two local delegates and publish their findings as linked team memories.

The coordinator owns the shared blackboard; each contribution retains its
author. Delegate-private conversations remain on their individual timelines.
Only fictional launch data is used, in a fresh namespace on either backend.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from urllib.parse import urlencode

from demo import create_provider

from memorizz import MemAgent, MemoryHistory, MemoryType
from memorizz.llms.ollama import OllamaLLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["filesystem", "oracle"], default="filesystem"
    )
    parser.add_argument("--store", default="/tmp/memorizz-evolution/memory")
    parser.add_argument("--portal", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--output", default="/tmp/memorizz-evolution/team-evidence.json"
    )
    args = parser.parse_args()
    provider = create_provider(args)
    history = MemoryHistory(provider, record_changes=True)
    namespace = "harbor-team-" + uuid.uuid4().hex[:10]

    def participant(name, **settings):
        return MemAgent(
            agent_id=str(uuid.uuid4()),
            name=name,
            model=OllamaLLM(
                model="qwen2.5:7b", context_window_tokens=16384, num_predict=256
            ),
            memory_provider=provider,
            instruction="You track a fictional release. Follow explicit corrections. Reply in one sentence and do not call tools.",
            memory_types=[MemoryType.CONVERSATION_MEMORY, MemoryType.KNOWLEDGE_BASE],
            capture_memory_history=True,
            capture_context_snapshots=True,
            auto_register=False,
            **settings,
        )

    researcher = participant("Harbor researcher")
    reviewer = participant("Harbor reviewer")
    coordinator = participant(
        "Harbor team coordinator",
        delegates=[researcher, reviewer],
        delegation={
            "enabled": True,
            "mode": "deterministic",
            "persist_participants": True,
            "consolidation_strategy": "primary",
            "primary_task_id": "review",
            "max_workers": 1,
        },
    )
    agents = [researcher, reviewer, coordinator]
    for agent in agents:
        agent.memory_ids = [namespace]
        with history.recording(actor="example-setup", source="example:setup"):
            agent.save()
    try:
        report = coordinator.delegate(
            "Check the launch brief and its correction.",
            memory_id=namespace,
            return_report=True,
            plan=[
                {
                    "task_id": "research",
                    "assigned_agent_id": researcher.agent_id,
                    "description": "Original launch brief: owner Mina, launch after 18:00 UTC. Repeat these two facts.",
                },
                {
                    "task_id": "review",
                    "assigned_agent_id": reviewer.agent_id,
                    "description": "The original brief is superseded. The corrected owner is Mira and launch is after 19:00 UTC. State ONLY the corrected facts.",
                    "dependencies": ["research"],
                },
            ],
        )
        results = {item["task_id"]: item for item in report["tasks"]}
        assert all(
            item["status"] == "completed" for item in results.values()
        ), report.get("error")
        assert "Mira" in str(results["review"]["result"])
        scope = {"agent_id": coordinator.agent_id, "memory_id": namespace}
        # Explicit SDK publication of real delegate outputs, with explicit sources.
        with history.recording(
            actor=researcher.agent_id,
            initiator=coordinator.agent_id,
            source="example:publish-finding",
            **scope,
        ):
            original = provider.store(
                {
                    **scope,
                    "name": "Initial launch finding",
                    "content": str(results["research"]["result"]),
                    "metadata": {"name": "Initial launch finding"},
                },
                MemoryType.KNOWLEDGE_BASE,
            )
        with history.recording(
            actor=reviewer.agent_id,
            initiator=coordinator.agent_id,
            source="example:publish-review",
            **scope,
        ):
            revised = provider.store(
                {
                    **scope,
                    "name": "Reviewed launch finding",
                    "content": str(results["review"]["result"]),
                    "supersedes": original,
                    "metadata": {
                        "name": "Reviewed launch finding",
                        "supersedes": original,
                    },
                },
                MemoryType.KNOWLEDGE_BASE,
            )
        with history.recording(
            actor=coordinator.agent_id, source="example:publish-team-result", **scope
        ):
            final = provider.store(
                {
                    **scope,
                    "name": "Team launch decision",
                    "content": str(report["response"]),
                    "source_ids": [revised],
                    "metadata": {
                        "name": "Team launch decision",
                        "source_ids": [revised],
                    },
                },
                MemoryType.KNOWLEDGE_BASE,
            )
        events = history.timeline(agent_id=coordinator.agent_id, limit=200)["events"]
        shared = [event for event in events if event["memory_type"] == "shared_memory"]
        assert any(
            event["operation"] == "compare_and_swap_shared_memory" for event in shared
        )
        assert {researcher.agent_id, reviewer.agent_id} <= {
            event["actor"] for event in shared
        }
        assert all(event["actor"] != "unknown" for event in events)
        coordinator_messages = [
            event for event in events if event["memory_type"] == "conversation_memory"
        ]
        assert coordinator_messages and all(
            event["actor"] == coordinator.agent_id for event in coordinator_messages
        )
        delegate_writes = [
            event
            for event in shared
            if event["actor"] in {researcher.agent_id, reviewer.agent_id}
        ]
        assert all(
            event.get("initiator") == coordinator.agent_id for event in delegate_writes
        )
        query = urlencode({"agent_id": coordinator.agent_id})
        evidence = {
            "backend": args.backend,
            "agent_id": coordinator.agent_id,
            "memory_id": namespace,
            "delegates": [
                {"agent_id": agent.agent_id, "name": agent.name}
                for agent in [researcher, reviewer]
            ],
            "shared_memory_id": report["shared_memory_id"],
            "record_ids": [original, revised, final],
            "changes": len(events),
            "shared_changes": len(shared),
            "unknown_writers": sum(event["actor"] == "unknown" for event in events),
            "delegate_writes_with_initiator": len(delegate_writes),
            "results": results,
            "final_response": report["response"],
            "timeline": args.portal
            + "/traces/memory-evolution?"
            + query
            + "&group=writer",
            "playground": f"{args.portal}/agents/{coordinator.agent_id}/playground",
            "observability": args.portal + "/traces?" + query,
            "publication": "Delegate outputs published by this SDK example, not autonomous memory tool calls.",
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    finally:
        for agent in reversed(agents):
            agent.close()
        provider.close()


if __name__ == "__main__":
    main()
