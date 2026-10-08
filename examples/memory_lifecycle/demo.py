"""Run a real, local memory lifecycle in the connected MemoRizz playground.

    python examples/memory_lifecycle/demo.py --store /tmp/memorizz-memory-demo/memory \
        --portal http://127.0.0.1:8766 --output /tmp/memorizz-memory-demo/evidence

Creates one isolated fictional Harbor launch agent. Uses Ollama for both
generation and embeddings; never changes other agents or their memories.
The portal must already be connected to the same filesystem store.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from memorizz.embeddings import configure_embeddings
from memorizz.enums import MemoryType
from memorizz.long_term.semantic.entity_memory import EntityMemory
from memorizz.long_term.semantic.knowledge_base import KnowledgeBase
from memorizz.long_term.semantic.persona import Persona
from memorizz.memagent import MemAgent
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


def stream_turn(portal: str, agent_id: str, memory_id: str, query: str) -> dict:
    request = Request(
        f"{portal}/agents/{agent_id}/playground/stream",
        data=urlencode({"query": query, "memory_id": memory_id}).encode(),
        headers={"Origin": portal, "Content-Type": "application/x-www-form-urlencoded"},
    )
    events = []
    started = time.monotonic()
    event_name = "message"
    with urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                events.append({"event": event_name, "data": payload})
                if event_name in {"error", "run.done"}:
                    print(
                        json.dumps({"event": event_name, "data": payload}), flush=True
                    )
    failures = [
        event
        for event in events
        if event["event"] == "error"
        or (event["event"] == "run.done" and event["data"].get("status") != "completed")
    ]
    if failures:
        raise RuntimeError(f"Playground run failed: {failures}")
    if not any(event["event"] == "run.done" for event in events):
        raise RuntimeError("Playground stream closed without a terminal event")
    return {
        "query": query,
        "seconds": round(time.monotonic() - started, 2),
        "events": events,
    }


def refresh_connection(portal: str, store: Path) -> None:
    """Refresh the portal's filesystem catalog after external SDK writes."""
    request = Request(
        f"{portal}/connect",
        data=urlencode(
            {"provider_type": "filesystem", "filesystem_path": str(store.expanduser())}
        ).encode(),
        headers={"Origin": portal, "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request, timeout=30) as response:
        response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--portal", default="http://127.0.0.1:8766")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--resume", action="store_true", help="Continue the agent in evidence.json"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_embeddings("ollama", {"model": "nomic-embed-text"})
    if args.resume:
        manifest = json.loads((args.output / "evidence.json").read_text())
        run_conversations(args, manifest)
        return
    provider = FileSystemProvider(FileSystemConfig(root_path=args.store.expanduser()))
    memory_id = f"harbor-memory-demo-{uuid.uuid4().hex[:8]}"
    agent = MemAgent(
        agent_id=str(uuid.uuid4()),
        name="Harbor · Memory lifecycle demo",
        llm_config={
            "provider": "ollama",
            "model": args.model,
            "temperature": 0,
            "context_window_tokens": 16384,
            "num_predict": 2048,
        },
        embedding_provider="ollama",
        embedding_config={"model": "nomic-embed-text"},
        memory_provider=provider,
        memory_ids=[memory_id],
        memory_types=[
            MemoryType.PERSONAS,
            MemoryType.TOOLBOX,
            MemoryType.KNOWLEDGE_BASE,
            MemoryType.ENTITY_MEMORY,
            MemoryType.CONVERSATION_MEMORY,
            MemoryType.WORKFLOW_MEMORY,
            MemoryType.SKILLBOX,
            MemoryType.SUMMARIES,
            MemoryType.SEMANTIC_CACHE,
            MemoryType.TOOL_LOG,
        ],
        persona=Persona(
            name="Harbor release assistant",
            role="release coordinator",
            goals="Make evidence-based launch decisions and show budget arithmetic.",
            background="A fictional demonstration; no actual launch is performed.",
        ),
        instruction=(
            "Help plan the fictional Harbor launch. Use attached knowledge and known "
            "entities when relevant. Follow the launch-readiness skill. Distinguish "
            "confirmed facts from assumptions. Never claim to deploy or notify anyone. "
            "For questions that only recall our conversation, answer directly from "
            "history or its summary without calling tools. Keep answers concise."
        ),
        authored_skills=[
            {
                "name": "Harbor launch readiness",
                "description": "Decide Harbor launch readiness, budget headroom and release timing.",
                "content": "Before recommending launch: calculate budget minus spend; "
                "confirm rollback verification; name the release owner; respect the "
                "user's release-time constraint. If rollback is unverified, HOLD.",
                "queries": ["Harbor launch readiness budget rollback"],
            }
        ],
        skill_retrieval=True,
        skill_retrieval_config={"min_similarity": 0.15, "top_k": 1},
        semantic_cache=True,
        semantic_cache_config={
            "similarity_threshold": 0.99,
            "embedding_provider": "ollama",
            "embedding_config": {"model": "nomic-embed-text"},
        },
        tool_result_policy={
            "offload_above_chars": 1200,
            "digest_chars": 700,
            "persist_all_results": True,
        },
        automations_enabled=False,
        max_steps=10,
        streaming=True,
    )
    agent.save()
    provider.store(
        {
            **agent.persona_manager.configuration_persona.to_dict(),
            "agent_id": agent.agent_id,
        },
        MemoryType.PERSONAS,
    )
    knowledge = KnowledgeBase(provider)
    kb_id = knowledge.ingest_knowledge(
        "HARBOR DEMO LAUNCH BRIEF. Budget: USD 1,200. Committed spend: USD 940. "
        "Rollback verification: NOT YET VERIFIED. Rollback verification is required "
        "before launch. Do not invent a completed rollback check. This is fictional data.",
        namespace="Harbor memory lifecycle demo",
        chunking_strategy="none",
        metadata={"demo": "harbor-memory-lifecycle", "agent_id": agent.agent_id},
    )
    if not knowledge.attach_to_agent(agent, kb_id):
        raise RuntimeError("Could not attach demo knowledge base")
    entity_id = EntityMemory(provider).upsert_entity(
        name="Harbor",
        entity_type="project",
        memory_id=memory_id,
        attributes=[{"name": "release_owner", "value": "Mina"}],
        metadata={"demo": "harbor-memory-lifecycle"},
    )
    # Native tools rebuild on load; persist their metadata for the explorer.
    for metadata in agent.tool_manager.get_tool_metadata():
        provider.store(
            {**metadata, "agent_id": agent.agent_id, "demo": "harbor-memory-lifecycle"},
            MemoryType.TOOLBOX,
        )
    manifest = {
        "agent_id": agent.agent_id,
        "memory_id": memory_id,
        "knowledge_base_id": kb_id,
        "entity_id": entity_id,
        "model": args.model,
        "embedding_model": "nomic-embed-text",
        "playground": f"{args.portal}/agents/{agent.agent_id}/playground",
        "observability": f"{args.portal}/traces?agent_id={agent.agent_id}",
        "seeded": [
            "persona",
            "knowledge_base",
            "entity",
            "authored_skill",
            "tool_metadata",
        ],
        "turns": [],
    }
    manifest_path = args.output / "evidence.json"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    save()
    print(
        json.dumps(
            {key: manifest[key] for key in ["agent_id", "memory_id", "playground"]}
        ),
        flush=True,
    )
    refresh_connection(args.portal, args.store)
    run_conversations(args, manifest)


def run_conversations(args: argparse.Namespace, manifest: dict) -> None:
    agent_id, memory_id = manifest["agent_id"], manifest["memory_id"]
    refresh_connection(args.portal, args.store)
    manifest_path = args.output / "evidence.json"

    def save() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    prompts = [
        "For this Harbor demo conversation, my scheduling constraint is: launch only "
        "after 18:00 UTC. Acknowledge this in one sentence; do not use tools or "
        "save it as an entity. This is temporary conversation context.",
        "Assess Harbor launch readiness. Call knowledge_base_lookup for the budget, "
        "spend and rollback status, and entity_memory_lookup for the release owner. "
        "Use the Harbor launch readiness skill and my earlier scheduling constraint. "
        "State budget headroom, owner, earliest launch time and GO/HOLD with a reason.",
    ]
    for query in prompts[len(manifest["turns"]) :]:
        manifest["turns"].append(stream_turn(args.portal, agent_id, memory_id, query))
        save()
    # Compact only this demo's conversation, using the actual local model.
    provider = FileSystemProvider(FileSystemConfig(root_path=args.store.expanduser()))
    loaded = MemAgent.load(agent_id, memory_provider=provider)
    if not manifest.get("summary_ids"):
        manifest["summary_ids"] = loaded.generate_summaries(
            memory_id=memory_id,
            keep_recent=0,
            summary_type="manual_demo",
        )
    if not manifest["summary_ids"]:
        raise RuntimeError("The real model did not create a conversation summary")
    save()
    refresh_connection(args.portal, args.store)
    recall = "What earliest launch time did I specify earlier for Harbor? Answer in one sentence from our conversation memory, without tools."
    for _ in range(max(0, 4 - len(manifest["turns"]))):
        manifest["turns"].append(stream_turn(args.portal, agent_id, memory_id, recall))
        save()
    print(f"Evidence saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
