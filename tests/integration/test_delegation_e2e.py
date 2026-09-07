"""Public delegate -> real agent loop -> tools -> persisted, verified artifacts.

No paid model or host application: a deterministic model drives real tool calls.
Document/slides fixtures are JSON, not claims of production office rendering.
"""

import copy
import hashlib
import json
import socket
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from memorizz import CompletionPolicy, MemAgent, get_tool_context
from memorizz.coordination.shared_memory.shared_memory import SharedMemory
from memorizz.enums import MemoryType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from tests.mocks.mock_providers import MockLLMProvider


class ArtifactModel(MockLLMProvider):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = prompts

    def generate(self, messages, tools=None, tool_choice="auto"):
        ctx = get_tool_context()
        identity = (ctx["user_id"], ctx["delegated_task_id"])
        self.prompts.append((identity, copy.deepcopy(messages)))
        responses = [message for message in messages if message["role"] == "tool"]
        if not responses:
            assert "save_artifact" in {tool["function"]["name"] for tool in tools}
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    id="fixture-call",
                                    type="function",
                                    function=SimpleNamespace(
                                        name="save_artifact", arguments="{}"
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        # Reject one answer, then correct only the answer; never call save twice.
        if not any(message["role"] == "developer" for message in messages):
            return "unverified draft"
        return responses[-1]["content"]


@pytest.mark.parametrize("invalid", [None, "owner", "source", "task", "empty"])
def test_two_tenants_share_cached_agents_without_sharing_artifacts_or_histories(
    tmp_path, monkeypatch, invalid
):
    def deny(*args, **kwargs):
        raise AssertionError("Artifact E2E must not call any external service")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", use_faiss=False)
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    receipts = threading.local()
    tool_calls, prompts, events = [], [], []
    overlap = threading.Barrier(4)
    manifests = {
        user: {
            "owner": user,
            "source_version": hashlib.sha256(user.encode()).hexdigest(),
            "text": f"PRIVATE-{user}-source",
        }
        for user in ("alice", "bob")
    }

    def save_artifact() -> dict:
        """Save one fixture document or slide artifact from host-bound source."""
        ctx = get_tool_context()
        user, task_id = ctx["user_id"], ctx["delegated_task_id"]
        source = ctx["source_manifest"]
        assert source == manifests[user]
        overlap.wait(timeout=5)
        artifact = {
            "owner": user,
            "source_version": source["source_version"],
            "task_id": task_id,
            "content": f"{task_id}: {source['text']}",
        }
        if invalid == "owner":
            artifact["owner"] = "stranger"
        if invalid == "source":
            artifact["source_version"] = "obsolete"
        if invalid == "task":
            artifact["task_id"] = "unrelated"
        if invalid == "empty":
            artifact["content"] = ""
        path = artifacts / f"{user}-{task_id}.json"
        # Exclusive create makes a duplicate paid/save action fail the test.
        with path.open("x") as handle:
            json.dump(artifact, handle)
        receipt = {"artifact_id": path.name, "thread_id": ctx["thread_id"]}
        receipts.value = receipt
        tool_calls.append((user, task_id))
        return receipt

    def verify(candidate):
        ctx = get_tool_context()
        try:
            receipt = json.loads(candidate.response)
            if receipt != receipts.value:
                return False
            path = artifacts / receipt["artifact_id"]
            stored = json.loads(path.read_text())
        except (ValueError, KeyError, AttributeError, FileNotFoundError):
            return False
        return (
            stored["owner"] == ctx["user_id"]
            and stored["source_version"] == manifests[ctx["user_id"]]["source_version"]
            and stored["task_id"] == ctx["delegated_task_id"]
            and bool(stored["content"].strip())
        )

    def collect(event):
        return {
            "receipt": dict(receipts.value),
            "verified": event["status"] == "completed",
        }

    workers = [
        MemAgent(
            agent_id=name,
            model=ArtifactModel(prompts),
            tools=[save_artifact],
            memory_provider=provider,
            semantic_cache=False,
            continual_learning=False,
            auto_register=False,
            memory_types=[MemoryType.CONVERSATION_MEMORY],
            context_policy={"progressive_tool_disclosure": False},
            completion_policy=CompletionPolicy(
                enabled=True,
                max_rejections=1,
                require_tool_calls=True,
                validator=verify,
            ),
        )
        for name in ("document", "slides")
    ]
    root = MemAgent(
        agent_id="coordinator",
        model=MockLLMProvider(),
        delegates=workers,
        memory_provider=provider,
        semantic_cache=False,
        continual_learning=False,
        auto_register=False,
        delegation={
            "mode": "deterministic",
            "evidence_context": False,
            "consolidation_strategy": "deterministic",
        },
    )
    plan = [
        {"task_id": name, "assigned_agent_id": name, "description": f"Create {name}"}
        for name in ("document", "slides")
    ]

    # Distractor histories must not appear even when the incoming IDs match.
    for user in manifests:
        provider.store(
            {
                "memory_id": "shared-memory",
                "thread_id": "parent",
                "user_id": user,
                "role": "assistant",
                "content": f"OLD-{user}-PARENT-SECRET",
            },
            MemoryType.CONVERSATION_MEMORY,
        )

    def dispatch(user):
        return root.delegate(
            "Create both artifacts",
            memory_id="shared-memory",
            thread_id="parent",
            user_id=user,
            plan=plan,
            return_report=True,
            context={"source_manifest": manifests[user]},
            tool_context={"workflow_id": "same-id", "source_manifest": manifests[user]},
            on_task_result=collect,
            on_task_event=events.append,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = dict(zip(manifests, pool.map(dispatch, manifests)))
    assert sorted(tool_calls) == [
        ("alice", "document"),
        ("alice", "slides"),
        ("bob", "document"),
        ("bob", "slides"),
    ]
    assert all(not agent.memory_ids for agent in [root, *workers])
    child_threads = set()
    for user, report in reports.items():
        assert report["ok"] is (invalid is None), report
        assert not report["recording_warnings"]
        assert report["counts"]["completed" if invalid is None else "failed"] == 2
        for row in report["tasks"]:
            child_threads.add(row["thread_id"])
            assert (
                row["metadata"]["receipt"]["artifact_id"]
                == f"{user}-{row['task_id']}.json"
            )
            assert row["metadata"]["verified"] is (invalid is None)
            history = provider.retrieve_conversation_history_ordered_by_timestamp(
                "shared-memory",
                MemoryType.CONVERSATION_MEMORY,
                user_id=user,
                thread_id=row["thread_id"],
            )
            assert bool(history) is (invalid is None)
        session = SharedMemory._decode_payload(
            provider.retrieve_by_id(
                report["shared_memory_id"], MemoryType.SHARED_MEMORY
            )
        )
        assert (
            session["counts"] == report["counts"]
            and session["outcome"] == report["outcome"]
        )
    assert len(child_threads) == 4
    assert Counter(identity for identity, _ in prompts) == {
        (user, task): 3 for user in manifests for task in ("document", "slides")
    }
    assert reports["alice"]["shared_memory_id"] != reports["bob"]["shared_memory_id"]
    for (user, task_id), messages in prompts:
        text = json.dumps(messages)
        other = "bob" if user == "alice" else "alice"
        assert f"PRIVATE-{other}-source" not in text
        assert "PARENT-SECRET" not in text
    assert not hasattr(receipts, "value")
    assert not get_tool_context()
