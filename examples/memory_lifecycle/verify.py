"""Check the real demo's answers, persisted memories and portal trace evidence.

Run after demo.py with the same --store, --portal and --output arguments.
No generated or mocked agent responses are used by this verifier.
"""

import argparse
import json
from collections import Counter
from urllib.parse import urlencode
from urllib.request import urlopen

from memorizz.enums import MemoryType
from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider


def verify(store, portal, output):
    from pathlib import Path

    output = Path(output)
    manifest = json.loads((output / "evidence.json").read_text())
    agent_id, memory_id = manifest["agent_id"], manifest["memory_id"]
    provider = FileSystemProvider(FileSystemConfig(root_path=store))

    def get(route):
        with urlopen(portal + route, timeout=60) as response:
            assert response.status == 200, (route, response.status)
            return json.load(response)

    thread = get(
        f"/agents/{agent_id}/playground/thread?" + urlencode({"memory_id": memory_id})
    )
    export = get(
        "/traces/events.json?" + urlencode({"agent_id": agent_id, "limit": 500})
    )
    assert not export["errors"], export["errors"]
    assert not export["normalization_errors"], export["normalization_errors"]
    events = export["items"]
    messages = [
        row for row in thread["messages"] if row.get("message_type") != "trace_bundle"
    ]
    answers = [
        row.get("content", "") for row in messages if row.get("role") == "assistant"
    ]
    readiness = next((answer for answer in answers if "260" in answer), "")
    assert readiness, "The actual agent never calculated the expected $260 headroom"
    for required in ["mina", "18:00", "hold", "rollback"]:
        assert required in readiness.lower(), (required, readiness)
    assert len(answers) >= 4 and all(
        "18:00" in answer for answer in answers[-2:]
    ), answers

    panels = {
        key: len(thread[key])
        for key in [
            "toolbox_memory",
            "workflow_memory",
            "skill_memory",
            "entity_memory",
            "summary_memory",
            "tool_log_memory",
        ]
    }
    assert all(panels.values()), panels
    counts = {kind.value: len(provider.list_all(kind)) for kind in MemoryType}
    exercised = [
        "personas",
        "toolbox",
        "entity_memory",
        "knowledge_base",
        "conversation_memory",
        "workflow_memory",
        "skillbox",
        "summaries",
        "semantic_cache",
        "tool_log",
    ]
    assert all(counts[kind] for kind in exercised), counts
    calls = [
        e for e in events if e.get("kind") == "tool_result" and e.get("success") is True
    ]
    assert any(
        e.get("tool_name") == "knowledge_base_lookup"
        and "1,200" in e.get("content", "")
        for e in calls
    )
    assert any(
        e.get("tool_name") == "entity_memory_lookup" and "Mina" in e.get("content", "")
        for e in calls
    )
    summaries = [
        e
        for e in events
        if e.get("kind") == "memory_supply"
        and e.get("memory_type") == "summaries"
        and e.get("memory_chars", 0) > 0
    ]
    assert summaries, "No trace proves the generated summary reached the model"
    supplied_types = {
        e.get("memory_type")
        for e in events
        if e.get("kind") == "memory_supply" and e.get("memory_chars", 0) > 0
    }
    assert {
        "persona",
        "skills",
        "history",
        "entity",
        "episodic",
        "summaries",
        "tool_log",
    } <= supplied_types, supplied_types
    assert any(
        e.get("kind") == "cache_decision" and '"hit"' in e.get("content", "")
        for e in events
    ), "No actual semantic-cache hit"
    workflows = provider.list_all(MemoryType.WORKFLOW_MEMORY)
    assert any(
        row.get("skills_activated") for row in workflows
    ), "No actual skill activation recorded"
    report = {
        "passed": True,
        "agent_id": agent_id,
        "memory_id": memory_id,
        "playground": manifest["playground"],
        "observability": manifest["observability"],
        "exercised_memory_types": exercised,
        "not_exercised": ["short_term_memory", "shared_memory (agent coordination)"],
        "seeded_inputs": manifest["seeded"],
        "memory_record_counts": counts,
        "playground_panel_counts": panels,
        "trace_event_counts": dict(Counter(e.get("kind") for e in events)),
        "memory_supplied_to_model": sorted(supplied_types),
        "earlier_failed_attempts": [
            {"run_id": e.get("run_id"), "error_code": e.get("error_code")}
            for e in events
            if e.get("kind") == "turn_result" and e.get("status") == "error"
        ],
        "readiness_answer": readiness,
        "recall_answers": answers[-2:],
        "summary_ids": manifest["summary_ids"],
        "turn_seconds": [t["seconds"] for t in manifest["turns"]],
        "model": manifest["model"],
        "embedding_model": manifest["embedding_model"],
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    (output / "trace-events.json").write_text(json.dumps(export, indent=2))
    (output / "playground-thread.json").write_text(json.dumps(thread, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--portal", default="http://127.0.0.1:8766")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(args.store, args.portal, args.output)
