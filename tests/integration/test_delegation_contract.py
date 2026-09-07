"""Public delegation contracts: real scheduling and memory, deterministic workers."""
import copy
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from memorizz.approval import ApprovalRequired
from memorizz.completion import (
    CompletionDecision,
    CompletionPolicy,
    CompletionRejectedError,
)
from memorizz.coordination.shared_memory.shared_memory import SharedMemory
from memorizz.enums import MemoryType
from memorizz.memagent import MemAgent
from memorizz.multi_agent_orchestrator import MultiAgentOrchestrator
from memorizz.streaming import (
    CancellationToken,
    StreamCancelled,
    current_cancellation,
    current_stream,
)
from memorizz.task_decomposition import SubTask
from tests.mocks.mock_providers import MockLLMProvider, MockMemoryProvider

SOURCE = {
    "user_id": "fixture-user",
    "analysis_id": "security-video",
    "content_version": "sha256:fixture-transcript",
    "status": "ready",
}
PARENT_THREAD = "fixture-parent-thread"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("This diagnostic must not contact any external service")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


class Worker(MemAgent):
    def __init__(self, provider, name, body):
        super().__init__(
            model=MockLLMProvider(),
            memory_provider=provider,
            agent_id=name,
            instruction=f"Fixture {name} worker",
            semantic_cache=False,
            continual_learning=False,
        )
        self.body = body

    def run(self, query, memory_id=None, thread_id=None, **kwargs):
        return self.body(query, memory_id, thread_id, **kwargs)


class FixtureMemoryProvider(MockMemoryProvider):
    """Add the shared-session update/list surface absent in the general mock."""

    def __init__(self):
        super().__init__()
        self.shared_rows = {}
        self.shared_lock = threading.Lock()

    def store(self, data=None, memory_store_type=None, **kwargs):
        if memory_store_type == MemoryType.SHARED_MEMORY:
            with self.shared_lock:
                key = data["memory_id"]
                self.shared_rows[key] = {**copy.deepcopy(data), "_id": key}
                return key
        return super().store(data, memory_store_type, **kwargs)

    def retrieve_by_id(self, unit_id, memory_store_type=None):
        if memory_store_type == MemoryType.SHARED_MEMORY:
            with self.shared_lock:
                return copy.deepcopy(self.shared_rows.get(unit_id))
        return super().retrieve_by_id(unit_id, memory_store_type)

    def update_by_id(self, unit_id, values, memory_store_type):
        assert memory_store_type == MemoryType.SHARED_MEMORY
        with self.shared_lock:
            if unit_id not in self.shared_rows:
                return False
            self.shared_rows[unit_id].update(copy.deepcopy(values))
            return True

    def compare_and_swap_shared_memory(self, memory_id, expected_content, content):
        with self.shared_lock:
            row = self.shared_rows.get(memory_id)
            if row is None or row.get("content") != expected_content:
                return False
            row["content"] = content
            return True

    def list_all(self, memory_store_type, *args, **kwargs):
        assert memory_store_type == MemoryType.SHARED_MEMORY
        with self.shared_lock:
            return copy.deepcopy(list(self.shared_rows.values()))


def coordinator(bodies):
    provider = FixtureMemoryProvider()
    workers = [Worker(provider, name, body) for name, body in bodies.items()]
    root = MemAgent(
        model=MockLLMProvider(),
        memory_provider=provider,
        agent_id="fixture-coordinator",
        delegates=workers,
        semantic_cache=False,
        continual_learning=False,
        delegation={
            "enabled": True,
            "mode": "deterministic",
            "consolidation_strategy": "deterministic",
            "evidence_context": False,
        },
    )
    # A fallback is observable without granting it any real action authority.
    root.run = Mock(return_value="Unexpected coordinator fallback")
    return root


def task(name, dependencies=(), worker=None):
    return {
        "task_id": name,
        "description": f"Create {name} from the bound source",
        "assigned_agent_id": worker or name,
        "dependencies": list(dependencies),
    }


def dispatch(root, plan, **options):
    return root.delegate(
        "Ingest the security video, then create a document and slides from it",
        memory_id="fixture-memory",
        thread_id=PARENT_THREAD,
        user_id=SOURCE["user_id"],
        trace_id="fixture-parent-trace",
        context={"source_manifest": copy.deepcopy(SOURCE)},
        tool_context={
            "workflow_id": "fixture-plan",
            "source_manifest": copy.deepcopy(SOURCE),
        },
        plan=plan,
        return_report=True,
        **options,
    )


def test_public_delegate_runs_ready_branches_in_parallel_with_the_same_source():
    ready = threading.Event()
    overlap = threading.Barrier(2)
    spans = []

    def ingest(*args, **kwargs):
        ready.set()
        return json.dumps(SOURCE)

    def artifact(*args, **kwargs):
        assert ready.is_set()
        assert kwargs["user_id"] == SOURCE["user_id"]
        assert kwargs["context"]["source_manifest"] == SOURCE
        assert json.loads(kwargs["context"]["dependency_results"]["ingest"]) == SOURCE
        started = time.monotonic()
        overlap.wait(timeout=3)
        spans.append((started, time.monotonic()))
        return json.dumps(
            {"ok": True, "artifact_id": kwargs["tool_context"]["delegated_task_id"]}
        )

    root = coordinator({"ingest": ingest, "doc": artifact, "slides": artifact})
    result = dispatch(
        root, [task("ingest"), task("doc", ["ingest"]), task("slides", ["ingest"])]
    )
    assert result["ok"] is True
    assert [row["status"] for row in result["tasks"]] == ["completed"] * 3
    assert max(start for start, _ in spans) <= min(end for _, end in spans)
    assert result["consolidation"]["model_used"] is False
    assert root.model.call_count == 0  # validated plan: no second LLM planner
    root.run.assert_not_called()


def test_slides_from_document_are_sequential():
    order = []

    def doc(*args, **kwargs):
        order.append("doc-saved")
        return json.dumps({"artifact_id": "verified-doc"})

    def slides(*args, **kwargs):
        assert order == ["doc-saved"]
        assert (
            json.loads(kwargs["context"]["dependency_results"]["doc"])["artifact_id"]
            == "verified-doc"
        )
        order.append("slides-saved")
        return "verified-slides"

    result = dispatch(
        coordinator({"doc": doc, "slides": slides}),
        [task("doc"), task("slides", ["doc"])],
    )
    assert result["ok"] and order == ["doc-saved", "slides-saved"]


def test_failed_ingestion_blocks_both_artifacts():
    def fail(*args, **kwargs):
        raise RuntimeError("Fixture transcript is unavailable")

    artifacts = Mock(return_value="must not run")
    result = dispatch(
        coordinator({"ingest": fail, "doc": artifacts, "slides": artifacts}),
        [task("ingest"), task("doc", ["ingest"]), task("slides", ["ingest"])],
    )
    assert result["ok"] is False
    assert [row["status"] for row in result["tasks"]] == [
        "failed",
        "blocked",
        "blocked",
    ]
    artifacts.assert_not_called()


def test_failed_artifact_envelope_must_not_be_completed():
    root = coordinator({"doc": lambda *a, **k: '{"ok":false,"error":"not saved"}'})
    policy = CompletionPolicy(
        enabled=True, validator=lambda candidate: json.loads(candidate.response)["ok"]
    )
    result = dispatch(root, [task("doc")], task_completion_policy=policy)
    assert result["ok"] is False
    assert result["tasks"][0]["status"] == "failed"


def root_with_runtime_error(error):
    root = coordinator({"doc": lambda *a, **k: "unused"})
    worker = MemAgent(
        model=MockLLMProvider(),
        memory_provider=root.memory_provider,
        agent_id="doc",
        semantic_cache=False,
        continual_learning=False,
    )
    worker._build_context = Mock(return_value={})
    worker._build_system_prompt = Mock(return_value="Fixture instruction")
    worker._execute_llm_interaction = Mock(side_effect=error)
    root.delegates = [worker]
    return root, worker


def test_real_memagent_runtime_exception_does_not_complete_the_task():
    root, worker = root_with_runtime_error(RuntimeError("Fixture provider interrupted"))
    result = dispatch(root, [task("doc")])
    worker._execute_llm_interaction.assert_called_once()
    assert result["ok"] is False
    assert result["tasks"][0]["status"] == "failed"


def test_existing_completion_rejection_fails_a_real_delegate():
    rejection = CompletionRejectedError(
        CompletionDecision(
            accepted=False,
            code="artifact_not_verified",
            reason="Fixture has no saved artifact",
        ),
        1,
    )
    root, worker = root_with_runtime_error(rejection)
    result = dispatch(root, [task("doc")])
    worker._execute_llm_interaction.assert_called_once()
    assert result["ok"] is False
    assert result["tasks"][0]["status"] == "failed"
    root.run.assert_not_called()


def test_unknown_delegate_must_fail_closed_without_root_execution():
    root = coordinator({"doc": lambda *a, **k: "saved"})
    result = dispatch(root, [task("doc", worker="unregistered")])
    root.run.assert_not_called()
    assert result["ok"] is False


def test_parallel_children_have_independent_conversation_threads():
    threads = []

    def capture(query, memory_id, thread_id, **kwargs):
        threads.append(thread_id)
        return "saved"

    result = dispatch(
        coordinator({"doc": capture, "slides": capture}), [task("doc"), task("slides")]
    )
    assert result["ok"]
    assert len(set(threads)) == 2 and PARENT_THREAD not in threads


def test_worker_receives_host_progress_context_and_cancellation_token():
    trace = ContextVar("fixture_progress_context", default=None)
    trace_token = trace.set("fixture-progress-sink")
    cancellation = CancellationToken()
    cancel_token = current_cancellation.set(cancellation)
    captured = []
    try:

        def capture(*args, **kwargs):
            captured.append((trace.get(), current_cancellation.get()))
            return "saved"

        dispatch(coordinator({"doc": capture}), [task("doc")])
        assert captured == [("fixture-progress-sink", cancellation)]
    finally:
        trace.reset(trace_token)
        current_cancellation.reset(cancel_token)


def test_post_completion_hook_failure_must_not_rerun_the_original_request(monkeypatch):
    def fail_hook(*args):
        raise RuntimeError("Fixture progress store unavailable")

    monkeypatch.setattr(MultiAgentOrchestrator, "_after_task_completion", fail_hook)
    saved = Mock(return_value="already-saved-artifact")
    root = coordinator({"doc": saved})
    result = dispatch(root, [task("doc")])
    saved.assert_called_once()
    root.run.assert_not_called()
    assert result["tasks"][0]["result"] == "already-saved-artifact"


def test_parallel_blackboard_appends_do_not_lose_successful_entries():
    class ConditionalProvider:
        def __init__(self):
            self.row = {"content": json.dumps({"blackboard": []})}
            self.read_barrier = threading.Barrier(2)
            self.lock = threading.Lock()
            self.reads = 0

        def retrieve_by_id(self, *args):
            with self.lock:
                snapshot = copy.deepcopy(self.row)
                self.reads += 1
                first_snapshot = self.reads <= 2
            if first_snapshot:
                self.read_barrier.wait(timeout=3)
            return snapshot

        def compare_and_swap_shared_memory(self, memory_id, expected_content, content):
            with self.lock:
                if self.row["content"] != expected_content:
                    return False
                self.row["content"] = content
                return True

        def update_by_id(self, memory_id, values, memory_type):
            self.row.update(copy.deepcopy(values))
            return True

    provider = ConditionalProvider()
    shared = SharedMemory(provider)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                shared.add_blackboard_entry,
                "board",
                name,
                {"task_id": name},
                "task_start",
            )
            for name in ("doc", "slides")
        ]
        assert [future.result() for future in futures] == [True, True]
    entries = json.loads(provider.row["content"])["blackboard"]
    assert {entry["agent_id"] for entry in entries} == {"doc", "slides"}


def test_shared_session_status_agrees_with_failed_workflow_report():
    def fail(*args, **kwargs):
        raise RuntimeError("Fixture save failed")

    root = coordinator({"doc": fail})
    result = dispatch(root, [task("doc")])
    assert result["ok"] is False
    session = root.memory_provider.retrieve_by_id(
        result["shared_memory_id"], MemoryType.SHARED_MEMORY
    )
    assert SharedMemory._decode_payload(session)["status"] == "failed"


@pytest.mark.parametrize(
    "plan",
    [
        [],
        {},
        "not a graph",
        [None],
        [{"description": "missing id", "assigned_agent_id": "doc"}],
        [{"task_id": "doc", "description": "missing assignment"}],
        [task("doc", worker="unknown")],
        [task("doc"), task("doc")],
        [task("doc", ["missing"]), task("slides")],
        [task("doc", ["slides"]), task("slides", ["doc"])],
        [task("doc", ["doc"])],
        [{**task("doc"), "status": "completed", "result": "saved"}],
        [{**task("doc"), "status": "in_progress"}],
        [{**task("doc"), "status": "skipped"}],
        [{**task("doc"), "status": "invalid"}],
        [{**task("doc"), "dependencies": "slides"}],
    ],
)
def test_invalid_plan_executes_no_actions_even_with_fallback_enabled(plan):
    worker = Mock(return_value="must not run")
    root = coordinator({"doc": worker, "slides": worker})
    report = dispatch(root, plan, allow_root_fallback=True)
    assert not report["ok"] and not report["fallback"]
    assert not report["partial"] and report["outcome"] == "failed"
    assert report["tasks"] == [] and report["error_code"]
    assert root.memory_provider.shared_rows == {}
    root.run.assert_not_called()
    worker.assert_not_called()


def test_callable_plan_is_invoked_once_and_empty_result_is_invalid():
    root = coordinator({"doc": lambda *a, **k: "saved"})
    plan = Mock(return_value=[])
    report = dispatch(root, plan, allow_root_fallback=True)
    assert not report["ok"]
    plan.assert_called_once()
    root.run.assert_not_called()


def test_automatic_empty_decomposition_requires_explicit_fallback_opt_in(monkeypatch):
    from memorizz.task_decomposition import TaskDecomposer

    monkeypatch.setattr(TaskDecomposer, "decompose_task", lambda *a, **k: [])
    root = coordinator({"doc": lambda *a, **k: "unused"})
    assert dispatch(root, None)["ok"] is False
    root.run.assert_not_called()
    assert dispatch(root, None, allow_root_fallback=True)["fallback"] is True
    root.run.assert_called_once()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("provider", "RuntimeError"),
        ("unexpected", "unexpected_response_format"),
        ("budget", "iteration_limit"),
        ("no_model", "model_unavailable"),
    ],
)
def test_real_agent_incomplete_exits_fail_and_block_consumers(mode, code):
    root = coordinator(
        {"doc": lambda *a, **k: "unused", "slides": Mock(return_value="unused")}
    )
    worker = MemAgent(
        model=MockLLMProvider(),
        memory_provider=root.memory_provider,
        agent_id="doc",
        semantic_cache=False,
        continual_learning=False,
    )
    worker._build_context = Mock(return_value={})
    worker._build_system_prompt = Mock(return_value="Bound fixture")
    if mode == "no_model":
        worker.model = None
    elif mode == "budget":
        worker._get_tool_iteration_limit = Mock(return_value=0)
    elif mode == "provider":
        worker._generate_with_trace = Mock(
            side_effect=RuntimeError("fixture provider failed")
        )
    else:
        worker._generate_with_trace = Mock(return_value=object())
    root.delegates[0] = worker
    report = dispatch(root, [task("doc"), task("slides", ["doc"])])
    assert [row["status"] for row in report["tasks"]] == ["failed", "blocked"]
    assert report["tasks"][0]["result"]["error_code"] == code
    assert not report["partial"] and report["outcome"] == "failed"
    root.delegates[1].body.assert_not_called()


def test_ordinary_conversation_retains_friendly_error_compatibility():
    root, worker = root_with_runtime_error(RuntimeError("fixture conversation failed"))
    response = worker.run("hello")
    assert "fixture conversation failed" in response


def test_approval_required_is_waiting_not_a_completed_artifact():
    proposal = SimpleNamespace(
        tool_name="fixture_save",
        proposal_id="proposal-1",
        to_dict=lambda **kwargs: {"proposal_id": "proposal-1"},
    )
    root, worker = root_with_runtime_error(ApprovalRequired(proposal))
    downstream = Worker(root.memory_provider, "slides", Mock(return_value="unused"))
    root.delegates.append(downstream)
    report = dispatch(root, [task("doc"), task("slides", ["doc"])])
    assert report["outcome"] == "waiting" and report["ok"] is False
    assert report["counts"]["waiting"] == 1
    assert [row["status"] for row in report["tasks"]] == ["waiting", "blocked"]
    downstream.body.assert_not_called()
    session = root.memory_provider.retrieve_by_id(
        report["shared_memory_id"], MemoryType.SHARED_MEMORY
    )
    assert SharedMemory._decode_payload(session)["outcome"] == "waiting"


def test_each_worker_gets_its_own_copied_context_but_not_parent_answer_stream():
    ambient = ContextVar("delegation_fixture_scope", default=None)
    ambient_token = ambient.set("parent")
    cancellation = CancellationToken()
    cancel_token = current_cancellation.set(cancellation)
    parent_stream = SimpleNamespace(agent=None)
    stream_token = current_stream.set(parent_stream)
    barrier = threading.Barrier(2)
    try:

        def body(*args, **kwargs):
            assert ambient.get() == "parent"
            assert current_cancellation.get() is cancellation
            assert current_stream.get() is None
            identity = kwargs["tool_context"]["delegated_task_id"]
            ambient.set(identity)
            barrier.wait(timeout=3)
            assert ambient.get() == identity
            return identity

        report = dispatch(
            coordinator({"doc": body, "slides": body}), [task("doc"), task("slides")]
        )
        assert report["ok"]
        assert ambient.get() == "parent" and current_stream.get() is parent_stream
    finally:
        ambient.reset(ambient_token)
        current_cancellation.reset(cancel_token)
        current_stream.reset(stream_token)


def test_stop_before_start_executes_no_worker():
    cancellation = CancellationToken()
    cancellation.cancel()
    body = Mock(return_value="unused")
    root = coordinator({"doc": body, "slides": body})
    report = dispatch(root, [task("doc"), task("slides")], cancellation=cancellation)
    body.assert_not_called()
    assert report["outcome"] == "cancelled"
    assert report["counts"]["cancelled"] == 2 and not report["partial"]


def test_stop_from_worker_prevents_new_independent_and_dependent_actions():
    cancellation = CancellationToken()

    def stop(*args, **kwargs):
        cancellation.cancel()
        raise StreamCancelled()

    unused = Mock(return_value="unused")
    root = coordinator({"doc": stop, "slides": unused, "summary": unused})
    report = dispatch(
        root,
        [task("doc"), task("slides"), task("summary", ["doc"])],
        cancellation=cancellation,
        max_workers=1,
    )
    unused.assert_not_called()
    assert report["outcome"] == "cancelled"
    assert report["counts"]["completed"] == 0


def test_deadline_reports_blocking_worker_honestly_and_keeps_it_supervised():
    from memorizz.multi_agent_orchestrator import _supervised_lock, _supervised_workers

    started, release, settled = threading.Event(), threading.Event(), threading.Event()

    def blocking(*args, **kwargs):
        started.set()
        assert release.wait(timeout=3)
        return "already-saved-artifact"

    def collector(event):
        if event["task_id"] == "doc":
            settled.set()
        return {"receipt": event["result"]}

    unused = Mock(return_value="unused")
    root = coordinator({"doc": blocking, "slides": unused})
    try:
        before = time.monotonic()
        report = dispatch(
            root,
            [task("doc"), task("slides", ["doc"])],
            timeout=0.15,
            on_task_result=collector,
        )
        assert started.is_set() and time.monotonic() - before < 1.5
        assert report["outcome"] == "cancelled"
        assert report["execution"]["cancellation_reason"] == "deadline_exceeded"
        assert report["execution"]["workers_still_running"] == ["doc"]
        assert report["tasks"][0]["status"] == "cancelling"
        assert report["reconciliation_required"]
        unused.assert_not_called()
        with _supervised_lock:
            assert _supervised_workers
    finally:
        release.set()
    assert settled.wait(timeout=3)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with _supervised_lock:
            if not _supervised_workers:
                break
        time.sleep(0.01)
    with _supervised_lock:
        assert not _supervised_workers
    entries = SharedMemory(root.memory_provider).get_blackboard_entries(
        report["shared_memory_id"], entry_type="task_settled"
    )
    assert entries[-1]["content"]["metadata"]["receipt"] == "already-saved-artifact"
    assert report["tasks"][0]["status"] == "cancelling"  # immutable snapshot


@pytest.mark.parametrize("callback", ["on_task_event", "on_task_result"])
def test_throwing_runtime_callback_preserves_result_and_exposes_warning(callback):
    saved = Mock(return_value="saved-once")
    root = coordinator({"doc": saved})
    report = dispatch(
        root,
        [task("doc")],
        **{callback: Mock(side_effect=RuntimeError("recording unavailable"))},
    )
    assert report["ok"] and report["tasks"][0]["result"] == "saved-once"
    assert report["recording_warnings"] and report["reconciliation_required"]
    saved.assert_called_once()
    root.run.assert_not_called()


def test_receipt_collector_executes_inside_worker_and_reaches_coordinator():
    receipt = threading.local()
    coordinator_thread = threading.get_ident()
    events = []

    def save(*args, **kwargs):
        receipt.value = kwargs["tool_context"]["delegated_task_id"]
        return "saved"

    def collector(event):
        assert threading.get_ident() != coordinator_thread
        assert event["execution_context"] == "worker"
        assert receipt.value == event["task_id"]
        return {"artifact_receipt": receipt.value}

    root = coordinator({"doc": save, "slides": save})
    report = dispatch(
        root,
        [task("doc"), task("slides")],
        on_task_result=collector,
        on_task_event=events.append,
    )
    assert report["ok"] and not report["recording_warnings"]
    assert [row["metadata"]["artifact_receipt"] for row in report["tasks"]] == [
        "doc",
        "slides",
    ]
    assert all(event["workflow_id"] == "fixture-plan" for event in events)
    assert events[-1]["outcome"] == report["outcome"]
    assert not hasattr(receipt, "value")


def test_shared_cached_agents_and_caller_plan_are_not_mutated():
    root = coordinator(
        {"doc": lambda *a, **k: "saved", "slides": lambda *a, **k: "saved"}
    )
    agents = [root, *root.delegates]
    original_ids = [copy.deepcopy(agent.memory_ids) for agent in agents]
    plan = [
        SubTask("doc", "Create doc", "doc"),
        SubTask("slides", "Create slides", "slides"),
    ]
    first = dispatch(root, plan)
    second = dispatch(root, plan)
    assert first["ok"] and second["ok"]
    assert [row["thread_id"] for row in first["tasks"]] == [
        row["thread_id"] for row in second["tasks"]
    ]
    assert all(item.status == "pending" and item.result is None for item in plan)
    assert [agent.memory_ids for agent in agents] == original_ids
    legacy = dispatch(root, plan, thread_strategy="shared")
    assert all(row["thread_id"] == PARENT_THREAD for row in legacy["tasks"])


@pytest.mark.parametrize(
    "bodies,outcome,partial",
    [
        ({"doc": True, "slides": True}, "succeeded", False),
        ({"doc": True, "slides": False}, "partial", True),
        ({"doc": False, "slides": False}, "failed", False),
    ],
)
def test_report_and_persisted_outcome_counts_agree(bodies, outcome, partial):
    def succeed(*args, **kwargs):
        return "saved"

    def fail(*args, **kwargs):
        raise RuntimeError("not saved")

    root = coordinator({name: succeed if ok else fail for name, ok in bodies.items()})
    report = dispatch(root, [task("doc"), task("slides")])
    payload = SharedMemory._decode_payload(
        root.memory_provider.retrieve_by_id(
            report["shared_memory_id"], MemoryType.SHARED_MEMORY
        )
    )
    assert report["outcome"] == payload["outcome"] == outcome
    assert report["counts"] == payload["counts"]
    assert report["partial"] is partial
    assert report["status"] == payload["status"]


def test_recording_failure_is_not_an_execution_failure_or_a_silent_success():
    root = coordinator({"doc": Mock(return_value="saved")})
    root.memory_provider.compare_and_swap_shared_memory = Mock(
        side_effect=RuntimeError("storage unavailable")
    )
    report = dispatch(root, [task("doc")])
    assert report["ok"] and report["tasks"][0]["result"] == "saved"
    assert report["reconciliation_required"] and report["recording_warnings"]
    root.delegates[0].body.assert_called_once()
    root.run.assert_not_called()


def test_unsupported_provider_rejects_unsafe_append_without_legacy_update():
    provider = SimpleNamespace(update_by_id=Mock(), retrieve_by_id=Mock())
    assert (
        SharedMemory(provider).add_blackboard_entry("board", "doc", "saved", "result")
        is False
    )
    provider.update_by_id.assert_not_called()
    provider.retrieve_by_id.assert_not_called()


def test_atomic_entry_retries_are_idempotent_and_id_collision_is_rejected():
    provider = FixtureMemoryProvider()
    shared = SharedMemory(provider)
    session = shared.create_shared_session("root")
    assert shared.add_blackboard_entry(
        session, "doc", "saved", "result", entry_id="receipt-1"
    )
    assert shared.add_blackboard_entry(
        session, "doc", "saved", "result", entry_id="receipt-1"
    )
    assert not shared.add_blackboard_entry(
        session, "doc", "different", "result", entry_id="receipt-1"
    )
    assert len(shared.get_blackboard_entries(session)) == 1


def test_anonymous_lookup_never_joins_a_tenant_session():
    shared = SharedMemory(FixtureMemoryProvider())
    shared.create_shared_session("root", workflow_id="same", user_id="alice")
    assert (
        shared.find_active_session_for_agent("root", workflow_id="same", user_id=None)
        is None
    )


@pytest.mark.parametrize(
    "key", ["on_task_event", "on_task_result", "task_completion_policy", "cancellation"]
)
def test_runtime_callbacks_and_policies_cannot_enter_saved_configuration(key):
    with pytest.raises(TypeError, match="runtime-only"):
        MemAgent(
            model=MockLLMProvider(),
            memory_provider=FixtureMemoryProvider(),
            delegation={key: lambda event: None},
        )


@pytest.mark.parametrize(
    "outcome,expected",
    [("success", "completed"), ("error", "failed"), ("provider_error", "failed")],
)
def test_explicit_typed_worker_outcomes(outcome, expected):
    from memorizz.tool_outcomes import ToolOutcome, ToolResult

    result = ToolResult("payload", ToolOutcome(status=outcome))
    report = dispatch(coordinator({"doc": lambda *a, **k: result}), [task("doc")])
    assert report["tasks"][0]["status"] == expected
    if expected == "completed":
        assert report["tasks"][0]["result"] == "payload"


@pytest.mark.parametrize(
    "status,expected",
    [
        ("pending_approval", "waiting"),
        ("canceled", "cancelled"),
        ("interrupted", "cancelled"),
        ("failed", "failed"),
        ("succeeded", "completed"),
    ],
)
def test_runtime_harness_states_are_not_all_completed_or_failed(status, expected):
    from memorizz.metaharness.models import HarnessResult

    root = coordinator({"doc": lambda *a, **k: "unused"})
    worker = root.delegates[0]
    worker.meta_harness = SimpleNamespace()
    worker.meta_harness_mode = "runtime"
    worker.run_on_harness = Mock(
        return_value=HarnessResult(
            "harness-run", "fixture", status, final_response="saved"
        )
    )
    report = dispatch(root, [task("doc")])
    assert report["tasks"][0]["status"] == expected
    assert report["ok"] == (status == "succeeded")


def test_shared_session_creation_failure_executes_zero_actions():
    root = coordinator({"doc": Mock(return_value="must not run")})
    root.memory_provider.store = Mock(return_value=None)
    report = dispatch(root, [task("doc")])
    assert not report["ok"]
    root.run.assert_not_called()
    root.delegates[0].body.assert_not_called()


def test_parent_recording_exception_preserves_completed_child():
    root = coordinator({"doc": Mock(return_value="saved")})
    root._record_interaction = Mock(
        side_effect=RuntimeError("conversation unavailable")
    )
    report = dispatch(root, [task("doc")])
    assert report["ok"] and report["tasks"][0]["result"] == "saved"
    assert report["reconciliation_required"]
    assert "parent_interaction" in {
        row["operation"] for row in report["recording_warnings"]
    }
    root.run.assert_not_called()
    root.delegates[0].body.assert_called_once()


def test_missing_conversation_write_ids_are_recording_warnings_not_execution_failure():
    root, worker = root_with_runtime_error(None)
    worker._execute_llm_interaction.side_effect = None
    worker._execute_llm_interaction.return_value = "saved"
    worker.memory_manager.save_memory_unit = Mock(return_value=None)
    report = dispatch(root, [task("doc")])
    assert report["ok"] and report["tasks"][0]["result"] == "saved"
    assert "conversation_persistence" in {
        row["operation"] for row in report["recording_warnings"]
    }


def test_task_policy_and_receipt_callback_receive_authoritative_worker_tool_context():
    from memorizz import get_tool_context

    seen = []

    def capture(candidate):
        scope = get_tool_context()
        assert scope["user_id"] == SOURCE["user_id"]
        assert scope["delegated_task_id"] == "doc"
        assert scope["thread_id"] != PARENT_THREAD
        seen.append(scope["thread_id"])
        return True

    def collect(event):
        assert get_tool_context()["thread_id"] == seen[-1]

    report = dispatch(
        coordinator({"doc": lambda *a, **k: "saved"}),
        [task("doc")],
        task_completion_policy=CompletionPolicy(enabled=True, validator=capture),
        on_task_result=collect,
    )
    assert report["ok"] and not report["recording_warnings"]
    assert seen == [report["tasks"][0]["thread_id"]]
    assert get_tool_context() == {}


def test_stop_between_two_tools_in_one_model_response_prevents_second_effect():
    from memorizz import get_tool_context

    token = CancellationToken()
    second = Mock()

    def stop_action():
        token.cancel()
        return "first action finished"

    def second_action():
        second()
        return "must not run"

    root = coordinator({"doc": lambda *a, **k: "unused"})
    calls = [
        SimpleNamespace(
            id=name,
            type="function",
            function=SimpleNamespace(name=name, arguments="{}"),
        )
        for name in ("stop_action", "second_action")
    ]
    model = MockLLMProvider(
        responses=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=None, tool_calls=calls)
                    )
                ]
            )
        ]
    )
    root.delegates = [
        MemAgent(
            model=model,
            memory_provider=root.memory_provider,
            agent_id="doc",
            tools=[stop_action, second_action],
            context_policy={"progressive_tool_disclosure": False},
            semantic_cache=False,
            continual_learning=False,
        )
    ]
    report = dispatch(root, [task("doc")], cancellation=token)
    assert report["outcome"] == "cancelled" and not report["ok"]
    second.assert_not_called()
    assert get_tool_context() == {}


@pytest.mark.parametrize(
    "response",
    [
        "",
        "   ",
        SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[]))
            ]
        ),
    ],
)
def test_empty_native_model_answer_is_not_a_completed_delegate(response):
    root = coordinator({"doc": lambda *a, **k: "unused"})
    root.delegates = [
        MemAgent(
            agent_id="doc",
            model=MockLLMProvider(responses=[response]),
            memory_provider=root.memory_provider,
            semantic_cache=False,
            continual_learning=False,
        )
    ]
    report = dispatch(root, [task("doc")])
    assert not report["ok"]
    assert report["tasks"][0]["result"]["error_code"] == "empty_response"


def test_exhausted_global_worker_capacity_never_starts_an_action():
    from memorizz.multi_agent_orchestrator import _worker_slots

    acquired = 0
    try:
        while _worker_slots.acquire(blocking=False):
            acquired += 1
        body = Mock(return_value="must not run")
        report = dispatch(coordinator({"doc": body}), [task("doc")])
        assert not report["ok"] and report["tasks"][0]["status"] == "blocked"
        body.assert_not_called()
    finally:
        for _ in range(acquired):
            _worker_slots.release()


def test_pre_cancelled_automatic_plan_does_not_call_the_planner():
    token = CancellationToken()
    token.cancel()
    root = coordinator({"doc": Mock()})
    report = dispatch(root, None, cancellation=token)
    assert report["outcome"] == "cancelled"
    assert root.model.call_count == 0
    root.run.assert_not_called()


def test_conflict_retry_exhaustion_is_explicit_and_does_not_fall_back_to_replace():
    provider = FixtureMemoryProvider()
    shared = SharedMemory(provider)
    session = shared.create_shared_session("root")
    provider.compare_and_swap_shared_memory = Mock(return_value=False)
    provider.update_by_id = Mock()
    assert shared.add_blackboard_entry(session, "doc", "saved", "result") is False
    assert provider.compare_and_swap_shared_memory.call_count == 16
    provider.update_by_id.assert_not_called()


@pytest.mark.parametrize("fails", [True, False])
def test_nested_delegation_cannot_hide_child_failure_or_finish_parent_session_early(
    fails,
):
    root = coordinator({"slides": Mock(return_value="slides saved")})

    def leaf_body(*args, **kwargs):
        if fails:
            raise RuntimeError("nested save failed")
        return "nested saved"

    leaf = Worker(root.memory_provider, "leaf", leaf_body)
    middle = MemAgent(
        agent_id="middle",
        model=MockLLMProvider(),
        delegates=[leaf],
        memory_provider=root.memory_provider,
        semantic_cache=False,
        continual_learning=False,
        delegation={
            "mode": "deterministic",
            "plan": [task("leaf")],
            "consolidation_strategy": "deterministic",
            "evidence_context": False,
        },
    )
    root.delegates.insert(0, middle)
    report = dispatch(root, [task("doc", worker="middle"), task("slides", ["doc"])])
    assert report["ok"] is (not fails)
    assert not report["recording_warnings"], report["recording_warnings"]
    if fails:
        assert [row["status"] for row in report["tasks"]] == ["failed", "blocked"]
        nested = report["tasks"][0]["result"]["workflow_report"]
        assert not nested["ok"] and nested["counts"]["failed"] == 1
        root.delegates[1].body.assert_not_called()
    row = root.memory_provider.retrieve_by_id(
        report["shared_memory_id"], MemoryType.SHARED_MEMORY
    )
    payload = SharedMemory._decode_payload(row)
    assert payload["counts"] == report["counts"]
    assert payload["sub_agent_ids"] == ["leaf"]
    # Only the owning root may commit the shared session's overall outcome.
    assert payload["revision"] == len(payload["blackboard"]) + 1


@pytest.mark.parametrize(
    "change",
    [
        {"task_id": 42},
        {"description": {"text": "doc"}},
        {"assigned_agent_id": ["doc"]},
        {"dependencies": {}},
        {"dependencies": [False]},
        {"status": False},
    ],
)
def test_malformed_plan_fields_are_not_silently_coerced_into_executable_work(change):
    body = Mock(return_value="must not run")
    root = coordinator({"doc": body})
    report = dispatch(root, [{**task("doc"), **change}])
    assert not report["ok"]
    assert not root.memory_provider.shared_rows
    body.assert_not_called()


def test_configured_plan_must_not_synthesize_missing_task_ids():
    with pytest.raises(ValueError, match="task_id"):
        MemAgent(
            model=MockLLMProvider(),
            memory_provider=FixtureMemoryProvider(),
            delegation={"plan": [{"description": "save", "assigned_agent_id": "doc"}]},
        )


def test_invalid_plan_cannot_be_admitted_by_a_cache_that_ignores_bypass_reason():
    root = coordinator({"doc": Mock(return_value="must not run")})
    root.cache_manager.enabled = True
    root.cache_manager.get_cached_response = Mock(return_value="stale completion")
    report = dispatch(root, [task("unknown")])
    assert not report["ok"] and report["tasks"] == []
    root.run.assert_not_called()
    root.delegates[0].body.assert_not_called()


def test_nonparticipant_cannot_register_children_on_a_shared_session():
    shared = SharedMemory(FixtureMemoryProvider())
    session = shared.create_shared_session("root", ["doc"])
    assert not shared.register_sub_agents(session, "stranger", ["child"])
    assert shared.get_agent_hierarchy(session)["sub_agents"] == []
