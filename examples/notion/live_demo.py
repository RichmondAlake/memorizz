"""Create and verify a persistent Notion walkthrough with no paid LLM calls.

Run from a source checkout with PYTHONPATH=src. The Notion token is loaded from
the environment or uncommitted .env. Only a NEW child area is provisioned.
The demo stays in Notion; local vectors and its manifest stay in --output-dir.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class DemoEmbeddings:
    """Deterministic demonstration vectors, not a production embedding model."""

    def get_embedding(self, text):
        text = text.lower()
        return [
            1.0 + sum(text.count(word) for word in ("coffee", "cafe", "espresso")),
            1.0 + sum(text.count(word) for word in ("tea", "matcha", "teapot")),
            0.1,
        ]

    def get_dimensions(self):
        return 3

    def get_provider_info(self):
        return {"provider": "demo", "model": "drink-word-counts-v1", "dimensions": 3}


class DemoModel:
    """Synthetic responses/usage; never represents a provider invoice."""

    model = "notion-walkthrough-v1"

    def __init__(self):
        self.messages = []

    def get_config(self):
        return {"provider": "deterministic-demo", "model": self.model}

    def generate(self, messages, tools=None, **kwargs):
        self.messages = messages
        prompt = json.dumps(messages, default=str)
        answer = (
            "Tea with lemon." if "tea with lemon" in prompt else "Coffee with oat milk."
        )
        return {"role": "assistant", "content": answer}

    def generate_stream(self, messages, tools=None, **kwargs):
        answer = self.generate(messages, tools, **kwargs)["content"]
        for word in answer.split():
            yield {"type": "content", "content": word + " "}
        yield {"type": "done", "content": answer}

    def get_last_usage(self):
        return {
            "prompt_tokens": 24,
            "completion_tokens": 6,
            "total_tokens": 30,
            **self.get_config(),
        }

    def get_context_window_tokens(self):
        return 8192


def exercise(workspace, output_dir, *, client, token, progress=print):
    """Exercise real SDK persistence; caller explicitly supplies the Notion client."""
    from memorizz import MemAgent
    from memorizz.enums import MemoryType
    from memorizz.memory_provider import (
        FileSystemConfig,
        FileSystemProvider,
        NotionConfig,
        NotionProvider,
    )
    from memorizz.memory_provider.notion.provider import rich_text
    from memorizz.observability.usage_query import query_usage

    output_dir = Path(output_dir)
    source = uuid.UUID(workspace["data_source_id"])
    user_id, memory_id, thread_id = (
        "demo-alice",
        "notion-demo-memory",
        "notion-demo-thread",
    )
    agent_id = str(uuid.uuid5(source, "demo-agent"))
    config = NotionConfig(
        str(source), token=token, state_path=output_dir / "repair.sqlite3"
    )
    vector_config = FileSystemConfig(
        output_dir / "vectors", embedding_provider=DemoEmbeddings(), use_faiss=False
    )
    vectors = FileSystemProvider(vector_config)
    memory = NotionProvider(config, semantic_provider=vectors, client=client)
    agent = None
    knowledge_id = str(uuid.uuid5(source, MemoryType.KNOWLEDGE_BASE.value))
    samples = {
        MemoryType.PERSONAS: (
            "Helpful concierge",
            {"description": "Be concise and use saved preferences."},
        ),
        MemoryType.TOOLBOX: (
            "Preference lookup (illustrative)",
            {
                "description": "Find a scoped saved preference; this demo does not execute this tool."
            },
        ),
        MemoryType.ENTITY_MEMORY: (
            "Alice (synthetic person)",
            {
                "entity_type": "person",
                "description": "Alice is the synthetic user in this walkthrough.",
            },
        ),
        MemoryType.SHORT_TERM_MEMORY: (
            "Current task",
            {"content": "Recommend a drink using Alice's saved preference."},
        ),
        MemoryType.KNOWLEDGE_BASE: (
            "Alice's drink preference — edit Content",
            {"content": "Alice prefers coffee with oat milk."},
        ),
        MemoryType.WORKFLOW_MEMORY: (
            "Preference-aware recommendation",
            {
                "description": "Recall the preference, recommend a drink, persist the conversation."
            },
        ),
        MemoryType.SKILLBOX: (
            "Recommend a saved drink",
            {
                "when_to_use": "When the user asks for a drink recommendation",
                "description": "Use the user's current saved preference.",
            },
        ),
        MemoryType.SUMMARIES: (
            "Walkthrough summary",
            {
                "summary": "Two synthetic turns demonstrate retrieval before and after a Notion edit."
            },
        ),
        MemoryType.SEMANTIC_CACHE: (
            "Illustrative cached question",
            {
                "query_text": "Where is the demo cafe?",
                "response": "This is an illustrative cache record, not a real place.",
            },
        ),
        MemoryType.TOOL_LOG: (
            "Illustrative preference lookup — synthetic",
            {
                "tool_name": "demo_preference_lookup",
                "success": True,
                "result": "Synthetic example; no external tool was invoked.",
                "content": "Illustrative tool activity, not a real tool invocation.",
            },
        ),
    }
    try:
        memory.repair_index()
        for memory_type, (name, fields) in samples.items():
            memory.store(
                {
                    "id": str(uuid.uuid5(source, memory_type.value)),
                    "name": name,
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "thread_id": thread_id,
                    "agent_id": agent_id,
                    "demo": True,
                    **fields,
                },
                memory_type,
            )
        progress("Stored representative memory types in Notion.")
        # Verify deletion without removing the preference that remains visible.
        probe = memory.store(
            {"content": "Disposable CRUD probe", "user_id": "demo-disposable"},
            MemoryType.KNOWLEDGE_BASE,
        )
        assert memory.update_by_id(
            probe, {"content": "Updated probe"}, MemoryType.KNOWLEDGE_BASE
        )
        assert (
            memory.retrieve_by_id(probe, MemoryType.KNOWLEDGE_BASE)["content"]
            == "Updated probe"
        )
        assert memory.delete_by_id(probe, MemoryType.KNOWLEDGE_BASE)
        assert memory.retrieve_by_id(probe, MemoryType.KNOWLEDGE_BASE) is None

        model = DemoModel()
        agent = MemAgent(
            model=model,
            memory_provider=memory,
            agent_id=agent_id,
            name="Notion concierge — synthetic demo",
            instruction="Use the saved drink preference. This is a synthetic walkthrough.",
            memory_types=[MemoryType.KNOWLEDGE_BASE, MemoryType.CONVERSATION_MEMORY],
            automations_enabled=False,
            verbose=False,
        )
        scope = {"memory_id": memory_id, "thread_id": thread_id, "user_id": user_id}
        first = "".join(
            agent.run_stream("What espresso should I order?", **scope)
        ).strip()
        assert "Coffee with oat milk." in first
        assert "Alice prefers coffee with oat milk." in json.dumps(
            model.messages, default=str
        )
        assert agent._embedding_backfill_executor is None
        progress("First streamed MemAgent turn persisted with a trace.")

        hit = memory.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE, **scope)[
            0
        ]
        page_id = hit["notion"]["page_id"]
        client.request(
            "PATCH",
            "/pages/" + page_id,
            body={
                "properties": {
                    memory._properties["content"]: {
                        "rich_text": rich_text("Alice prefers tea with lemon.")
                    }
                }
            },
        )
        # This direct Notion edit follows the same path as editing Content in UI.
        assert (
            memory.retrieve_by_query("espresso", MemoryType.KNOWLEDGE_BASE, **scope)
            == []
        )
        assert memory.sync_page(page_id)
        second = "".join(
            agent.run_stream("What matcha should I order?", **scope)
        ).strip()
        assert "Tea with lemon." in second
        assert "Alice prefers tea with lemon." in json.dumps(
            model.messages, default=str
        )
        assert not memory.retrieve_by_query(
            "matcha", MemoryType.KNOWLEDGE_BASE, user_id="demo-bob"
        )
        progress(
            "Direct Notion edit synchronized; second turn used the new preference."
        )
    finally:
        if agent is not None:
            agent.close(close_memory_provider=False)
        memory.close()
        vectors.close()

    vectors = FileSystemProvider(vector_config)
    memory = NotionProvider(config, semantic_provider=vectors, client=client)
    try:
        assert (
            memory.retrieve_by_id(knowledge_id, MemoryType.KNOWLEDGE_BASE)["content"]
            == "Alice prefers tea with lemon."
        )
        assert (
            memory.retrieve_by_query(
                "matcha", MemoryType.KNOWLEDGE_BASE, user_id=user_id
            )[0]["id"]
            == knowledge_id
        )
        history = memory.retrieve_conversation_history_ordered_by_timestamp(
            memory_id, user_id=user_id, thread_id=thread_id
        )
        assert len(history) >= 4
        assert memory.retrieve_memagent(agent_id) is not None
        traces = memory.query_observability_records(
            MemoryType.SHARED_MEMORY,
            record_type="observability_trace_bundle",
            user_id=user_id,
        )
        assert len(traces["items"]) >= 2
        counts = {kind.value: len(memory.list_all(kind)) for kind in MemoryType}
        assert all(counts.values()), counts
        usage = query_usage(memory, filters={"user_id": user_id})
        assert usage["coverage"]["read_complete"] is True
        assert usage["totals"]["total_tokens"] >= 60
        status = memory.index_status()
        assert status["pending"] == 0
        # The secondary store contains no duplicate memory text.
        vector_records = vectors.list_all(MemoryType.KNOWLEDGE_BASE)
        assert "Alice prefers" not in json.dumps(vector_records)
        progress(
            "Restart, all 13 memory types, conversation history, traces and index verified."
        )
        return {
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "synthetic_demo": True,
            "model": model.get_config(),
            "memory_counts": counts,
            "conversation_rows": len(history),
            "trace_bundles": len(traces["items"]),
            "synthetic_total_tokens": usage["totals"]["total_tokens"],
            "index_pending": status["pending"],
            "agent_id": agent_id,
            "preference_page_id": page_id,
            "answers": [first, second],
            "checks": [
                "CRUD",
                "semantic retrieval",
                "stale-read protection",
                "Notion edit and sync",
                "tenant isolation",
                "restart persistence",
                "13 memory types",
                "streamed conversations",
                "persisted traces",
                "content-free vector store",
            ],
        }
    finally:
        memory.close()
        vectors.close()


def create_overview(workspace, result, client):
    """Create a readable snapshot; linked database views remain the live UI."""
    from memorizz.memory_provider.notion.provider import rich_text

    def text_block(kind, value):
        return {"object": "block", "type": kind, kind: {"rich_text": rich_text(value)}}

    def link(label, identifier):
        return {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": label,
                            "link": {
                                "url": "https://www.notion.so/"
                                + identifier.replace("-", "")
                            },
                        },
                    }
                ]
            },
        }

    rows = [("Memory type", "Saved records"), *result["memory_counts"].items()]
    blocks = [
        text_block("heading_1", "Memorizz + Notion: verified walkthrough"),
        text_block(
            "paragraph",
            "Real Notion storage and local filesystem vectors; a deterministic demo model supplies "
            "synthetic responses and token counts. No paid LLM was called. This is a verification "
            "snapshot, not a billing statement or an interactive Notion chat client.",
        ),
        link(
            "Open the memory library — 13 memory-type views", workspace["database_id"]
        ),
        link(
            "Open the agent workspace — agents, conversations, traces and tool activity",
            workspace["interface_page_id"],
        ),
        link("Inspect Alice's current saved preference", result["preference_page_id"]),
        text_block("heading_2", "What the agent remembered"),
        text_block(
            "paragraph",
            "Turn 1: Alice's memory said coffee with oat milk. The streaming MemAgent replied: "
            + result["answers"][0],
        ),
        text_block(
            "paragraph",
            "We edited Content directly in Notion to tea with lemon, verified the stale vector "
            "could not serve old content, and synchronized the index.",
        ),
        text_block(
            "paragraph",
            "Turn 2: The streaming MemAgent used the edited preference and replied: "
            + result["answers"][1],
        ),
        text_block("heading_2", "Memory inventory"),
        {
            "object": "block",
            "type": "table",
            "table": {
                "table_width": 2,
                "has_column_header": True,
                "has_row_header": False,
                "children": [
                    {
                        "object": "block",
                        "type": "table_row",
                        "table_row": {"cells": [rich_text(label), rich_text(count)]},
                    }
                    for label, count in rows
                ],
            },
        },
        text_block("heading_2", "Verification"),
        text_block(
            "paragraph",
            f"Checked {result['verified_at']}. Conversation records: {result['conversation_rows']}. "
            f"Trace bundles: {result['trace_bundles']}. Pending vector repairs: {result['index_pending']}. "
            f"Synthetic demo tokens: {result['synthetic_total_tokens']}.",
        ),
        *[
            text_block("bulleted_list_item", "Passed: " + check)
            for check in result["checks"]
        ],
        text_block(
            "paragraph",
            "The tool-activity row is explicitly illustrative; no external tool was executed. "
            "Trace bundles come from the real MemAgent persistence path. Edit only Content/Name "
            "on managed records, then sync before semantic retrieval. Scope, identity and "
            "timestamp columns are managed by Memorizz.",
        ),
    ]
    return client.request(
        "POST",
        "/pages",
        body={
            "parent": {"page_id": workspace["root_page_id"]},
            "properties": {"title": {"title": rich_text("Start here — verified demo")}},
            "children": blocks,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-page-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse this directory's existing demo; never provision another root.",
    )
    args = parser.parse_args()
    parent = str(uuid.UUID(args.parent_page_id))
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MEMORIZZ_HOME"] = str(directory / "runtime")
    os.environ["MEMORIZZ_UI_AUDIT_LOG"] = str(directory / "trace-audit.jsonl")
    from dotenv import load_dotenv

    from memorizz.memory_provider.notion import NotionClient, provision_notion_workspace

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        parser.error("Set NOTION_TOKEN privately in the environment or .env")
    manifest_path = directory / "workspace.json"
    if manifest_path.exists():
        if not args.resume:
            parser.error(
                "This output directory already has a demo manifest; use --resume."
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest["parent_page_id"] != parent:
            parser.error("The existing demo belongs to a different parent page.")
    else:
        if args.resume:
            parser.error("No existing demo manifest to resume.")
        manifest = {"parent_page_id": parent, "phase": "provisioning", "workspace": {}}

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    client = NotionClient(token)
    try:
        client.request("GET", "/pages/" + parent)
        if not args.resume:
            save()  # An interrupted provision is never blindly recreated.
            try:
                manifest["workspace"] = provision_notion_workspace(
                    parent,
                    title="Memorizz live demo — "
                    + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    client=client,
                )
            except BaseException as exc:
                manifest["workspace"] = getattr(exc, "notion_workspace", {})
                save()
                raise
            manifest["phase"] = "provisioned"
            save()
            print("Created demo: " + manifest["workspace"]["url"], flush=True)
        if manifest["phase"] == "provisioning":
            parser.error(
                "Provisioning was incomplete. Inspect the saved resource IDs before continuing; "
                "no second area was created."
            )
        if manifest["phase"] == "provisioned":
            manifest["result"] = exercise(
                manifest["workspace"],
                directory,
                client=client,
                token=token,
                progress=lambda value: print(value, flush=True),
            )
            manifest["phase"] = "verified"
            save()
        if manifest["phase"] == "verified":
            manifest["phase"] = "overview_creating"
            save()
            overview = create_overview(
                manifest["workspace"], manifest["result"], client
            )
            manifest["overview_page_id"] = overview["id"]
            manifest["overview_url"] = overview.get(
                "url"
            ) or "https://www.notion.so/" + overview["id"].replace("-", "")
            manifest["phase"] = "complete"
            save()
        if manifest["phase"] != "complete":
            parser.error(
                "Overview creation may have committed; inspect the demo root before retrying. "
                "No duplicate overview was created."
            )
        client.request("GET", "/pages/" + manifest["overview_page_id"])
        print(json.dumps(manifest, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
