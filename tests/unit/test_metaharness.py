"""Deterministic contract tests for the memory-first meta-harness."""

from __future__ import annotations

import json
import os
import runpy
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from memorizz.approval import ApprovalStateError, SQLiteApprovalStore
from memorizz.enums.memory_type import MemoryType
from memorizz.memory_provider.filesystem.provider import (
    FileSystemConfig,
    FileSystemProvider,
)
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessContextPack,
    HarnessEvent,
    HarnessEventType,
    HarnessPermissions,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
    VerificationSpec,
)
from memorizz.metaharness.adapters import (
    ClaudeCodeHarness,
    CodexHarness,
    NativeMemAgentHarness,
    OpenHandsHarness,
    PersistedMemAgentHarness,
)
from memorizz.metaharness.base import AdapterOutcome, SubprocessHarness
from memorizz.metaharness.security import redact, redact_paths


class FakeHarness(AgentHarness):
    name = "fake"

    def __init__(self, *, available: bool = True, block: bool = False) -> None:
        self.available = available
        self.block = block

    def probe(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            name=self.name,
            available=self.available,
            command="fake",
            mcp=True,
            usage_reporting=True,
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        emit(
            HarnessEvent(
                task.run_id,
                HarnessEventType.MESSAGE,
                {"text": "token=sk-test-this-must-not-survive"},
            )
        )
        if self.block:
            deadline = time.monotonic() + 5
            while not cancel_event.wait(0.02) and time.monotonic() < deadline:
                pass
            if cancel_event.is_set():
                return AdapterOutcome(
                    error_code="canceled", error="Harness run was canceled"
                )
        if task.writes_workspace:
            (workspace / "result.txt").write_text("complete\n", encoding="utf-8")
        return AdapterOutcome(
            final_response="done",
            usage={"input_tokens": 10, "output_tokens": 2},
            cost_usd=0.01,
            exit_code=0,
        )


class _ContextCaptureHarness(FakeHarness):
    name = "context-capture"

    def __init__(self) -> None:
        super().__init__()
        self.context_packs = []

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        self.context_packs.append(context_pack)
        return super().run(
            task,
            workspace=workspace,
            context_pack=context_pack,
            emit=emit,
            cancel_event=cancel_event,
        )


class _CountingContextProvider:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve_by_query(self, query, memory_store_type, **kwargs):
        self.calls += 1
        if memory_store_type != MemoryType.KNOWLEDGE_BASE:
            return []
        return [
            {
                "_id": "source-1",
                "memory_id": "memory-1",
                "user_id": "user-1",
                "thread_id": "thread-1",
                "content": "One shared policy requirement.",
            }
        ]


class _PinnedCommandHarness(SubprocessHarness):
    name = "pinned"

    def build_command(self, task, *, workspace, prompt):
        return ["./runner"]

    def parse_event(self, run_id, payload):
        return (
            [HarnessEvent(run_id, HarnessEventType.MESSAGE, payload)],
            {"final_response": str(payload.get("message") or "")},
        )


class _AuthenticationFailingHarness(SubprocessHarness):
    name = "auth-failing"
    authentication_remediation = (
        "Set TEST_HARNESS_API_KEY, then rerun `memorizz harness doctor auth-failing`."
    )

    def build_command(self, task, *, workspace, prompt):
        return [self.command]

    def parse_event(self, run_id, payload):
        return [], {}


class _LifecycleCommandHarness(SubprocessHarness):
    """Emit two vendor lifecycle records for one logical command."""

    name = "lifecycle-command"

    def build_command(self, task, *, workspace, prompt):
        script = (
            "import json; "
            "print(json.dumps({'id':'command-1','status':'in_progress'})); "
            "print(json.dumps({'id':'command-1','status':'completed'}))"
        )
        return [self.command, "-c", script]

    def parse_event(self, run_id, payload):
        return [HarnessEvent(run_id, HarnessEventType.COMMAND, payload)], {}


class _AuthRequiredHarness(AgentHarness):
    name = "auth-required"

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            command="auth-required",
            error_code="authentication_required",
            error="The harness API key is not configured.",
            remediation="Set TEST_HARNESS_API_KEY before starting this harness.",
            metadata={"authentication_configured": False},
        )

    def run(self, task, *, workspace, context_pack, emit, cancel_event):
        raise AssertionError("An unauthenticated harness must not start")


def _service(tmp_path: Path, adapter: AgentHarness | None = None) -> MetaHarness:
    return MetaHarness(
        adapters=[adapter or FakeHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )


def _git_workspace(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@memorizz.dev"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "MemoRizz Tests"],
        check=True,
    )
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
    return path


def test_step_budget_counts_one_vendor_action_lifecycle_once(tmp_path: Path) -> None:
    harness = _LifecycleCommandHarness(command=sys.executable)
    task = HarnessTask(
        task="Inspect",
        workspace=str(tmp_path),
        harness=harness.name,
        budget={"max_steps": 1},
    )
    emitted = []

    outcome = harness.run(
        task,
        workspace=tmp_path,
        context_pack=HarnessContextPack(query=task.task),
        emit=emitted.append,
        cancel_event=threading.Event(),
    )

    assert outcome.error_code is None
    assert [event.data["status"] for event in emitted] == [
        "in_progress",
        "completed",
    ]


def test_redaction_preserves_numeric_token_usage_but_not_credentials() -> None:
    value = redact(
        {
            "input_tokens": 120,
            "input_tokens_mean": 118,
            "output_tokens_sample_stdev": 2.5,
            "total_tokens_mean_delta": -320,
            "input_tokens_paired_deltas": [-300, -340],
            "output_tokens": "8",
            "input_tokens_details": {"cached_tokens": 40},
            "cache_write_input_tokens": 12,
            "reasoning_output_tokens": 4,
            "candidate_tokens": 72,
            "context_tokens": 48,
            "tokens_used": 40,
            "tokens_saved": 32,
            "token_reporting": True,
            "max_output_tokens": 1_000,
            "evidence_token_budget": 900,
            "per_million_tokens": {"input": 0.2, "output": 1.2},
            "token_estimate": 24,
            "token_estimates": [24, 36],
            "api_token": "opaque-credential-value",
            "token": "another-opaque-credential",
        }
    )

    assert value["input_tokens"] == 120
    assert value["input_tokens_mean"] == 118
    assert value["output_tokens_sample_stdev"] == 2.5
    assert value["total_tokens_mean_delta"] == -320
    assert value["input_tokens_paired_deltas"] == [-300, -340]
    assert value["output_tokens"] == "8"
    assert value["input_tokens_details"]["cached_tokens"] == 40
    assert value["cache_write_input_tokens"] == 12
    assert value["reasoning_output_tokens"] == 4
    assert value["candidate_tokens"] == 72
    assert value["context_tokens"] == 48
    assert value["tokens_used"] == 40
    assert value["tokens_saved"] == 32
    assert value["token_reporting"] is True
    assert value["max_output_tokens"] == 1_000
    assert value["evidence_token_budget"] == 900
    assert value["per_million_tokens"] == {"input": 0.2, "output": 1.2}
    assert value["token_estimate"] == 24
    assert value["token_estimates"] == [24, 36]
    assert value["api_token"] == "[REDACTED]"
    assert value["token"] == "[REDACTED]"
    assert redact("opus5_task_native_small_suite_v1") == (
        "opus5_task_native_small_suite_v1"
    )
    assert redact("risk_router") == "risk_router"
    assert redact("credential=sk-test-abcdefghijklmnop") == "credential=[REDACTED]"


def test_report_path_redaction_only_replaces_known_prefixes(tmp_path: Path) -> None:
    value = redact_paths(
        {
            "command": f"{tmp_path}/.venv/bin/python -B verify.py",
            "evidence": "access_policy.py:5 and /unrelated/evidence",
        },
        {str(tmp_path): "[REPOSITORY]"},
    )

    assert value["command"] == "[REPOSITORY]/.venv/bin/python -B verify.py"
    assert value["evidence"] == "access_policy.py:5 and /unrelated/evidence"


def test_provider_comparison_uses_reversed_order_pairs_and_current_cost_math() -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "provider_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="provider_comparison_test")

    orders = namespace["counterbalanced_orders"](2, 20260822)

    assert orders[1] == list(reversed(orders[0]))
    assert {
        arm: sum(order.index(arm) + 1 for order in orders) / len(orders)
        for arm in namespace["ARM_SPECS"]
    } == {arm: 2.5 for arm in namespace["ARM_SPECS"]}
    assert (
        namespace["estimate_openai_cost"](
            {
                "input_tokens": 10_000,
                "cached_input_tokens": 4_000,
                "output_tokens": 1_000,
            },
            "gpt-5.6-luna",
        )
        == 0.00248
    )


def test_factorial_comparison_separates_wrapper_coordination_and_provider_effects() -> (
    None
):
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_comparison_test")

    specs = namespace["arm_specs"](["filesystem", "oracle"])
    orders = namespace["counterbalanced_orders"](list(specs), 2, 20260822)
    manifest = namespace["protocol_manifest"](
        repo_root=Path(__file__).parents[2],
        profile="factorial",
        providers=["filesystem", "oracle"],
        repeats=2,
        seed=20260822,
        codex_model="codex-model",
        claude_model="claude-model",
        judge_model="judge-model",
        maximum_observed_cost_usd=5.0,
    )
    repeated_manifest = namespace["protocol_manifest"](
        repo_root=Path(__file__).parents[2],
        profile="factorial",
        providers=["filesystem", "oracle"],
        repeats=2,
        seed=20260822,
        codex_model="codex-model",
        claude_model="claude-model",
        judge_model="judge-model",
        maximum_observed_cost_usd=5.0,
    )

    assert len(specs) == 12
    assert orders[1] == list(reversed(orders[0]))
    assert {
        arm: sum(order.index(arm) + 1 for order in orders) / len(orders)
        for arm in specs
    } == {arm: 6.5 for arm in specs}
    assert {value["kind"] for value in specs.values()} == {"direct", "wrapper", "panel"}
    assert manifest["controls"]["equal_total_compute"] is False
    assert manifest["controls"]["coordinator_model_calls"] == 0
    assert manifest["controls"]["independent_candidate_judging"] is True
    assert manifest["inference"]["independent_task_count"] == 1
    assert manifest["inference"]["inferential_statistics_allowed"] is False
    assert manifest["paper_comparable"] is False
    assert manifest["protocol_fingerprint"]
    assert manifest["protocol_fingerprint"] == repeated_manifest["protocol_fingerprint"]
    assert (
        manifest["environment_fingerprint"]
        == repeated_manifest["environment_fingerprint"]
    )
    assert set(manifest["fixture"]["file_sha256"]) == {
        "access_policy.py",
        "verify.py",
    }

    assert (
        namespace["_required_criterion_recall"](
            {
                "structured_coverage": {
                    "required_ids": ["one", "two"],
                    "covered_ids": ["one"],
                }
            }
        )
        == 50.0
    )


def test_factorial_validation_enforces_strategy_call_policy() -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_validation_test")
    specs = namespace["arm_specs"](["filesystem"])

    def run(harness: str) -> dict:
        return {
            "harness": harness,
            "status": "succeeded",
            "verified": True,
            "grounded": True,
            "context_content_fingerprint": "content-fingerprint",
            "usage_totals": {"input_tokens": 100, "output_tokens": 50},
        }

    base = {
        "ok": True,
        "structured_coverage": {
            "coverage_complete": True,
            "missing_ids": [],
            "parse_errors": {},
        },
        "workflow_failures": [],
        "source_id": "source-1",
    }
    direct = {**base, "runs": [run("codex")]}
    full = {
        **base,
        "runs": [run("codex"), run("claude-code")],
        "execution": {"strategy": "parallel"},
        "consolidation": {
            "strategy": "structured",
            "model_used": False,
            "memory_context": {"source_ids": ["source-1"]},
        },
    }
    adaptive = {
        **base,
        "runs": [run("codex")],
        "execution": {
            "strategy": "adaptive_coverage",
            "primary_task_ids": ["policy-review"],
            "escalation_task_ids": ["verification-review"],
        },
        "consolidation": {
            "strategy": "structured",
            "model_used": False,
            "memory_context": {"source_ids": ["source-1"]},
        },
    }

    assert namespace["validate_record"](direct, specs["direct_codex__filesystem"]) == []
    assert namespace["validate_record"](full, specs["panel_full__filesystem"]) == []
    assert (
        namespace["validate_record"](adaptive, specs["panel_adaptive__filesystem"])
        == []
    )
    failures = namespace["validate_record"](adaptive, specs["panel_full__filesystem"])
    assert any("full_panel_requires_codex_and_claude" in item for item in failures)


def test_provider_comparison_can_build_filesystem_only_runtime(tmp_path: Path) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "provider_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="provider_runtime_test")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    namespace["_write_fixture"](workspace)
    for name, content in namespace["FIXTURE_FILES"].items():
        assert (workspace / name).read_text(encoding="utf-8") == content
    runtime = namespace["create_runtime"](
        tmp_path, workspace, provider_names=["filesystem"]
    )
    try:
        assert set(runtime["providers"]) == {"filesystem"}
        assert runtime["preflight"]["filesystem"]["ok"] is True
        assert set(runtime["probes"]) == {"codex", "claude-code"}
    finally:
        namespace["cleanup_runtime"](runtime)


def test_factorial_paid_path_without_openai_auth_writes_structured_error(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    output = tmp_path / "missing-auth.json"
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["MEMORIZZ_EVAL_DISABLE_DOTENV"] = "1"
    env["MEMORIZZ_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MEMORIZZ_HOME"] = str(tmp_path / "memorizz-home")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--execute",
            "--profile",
            "smoke",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 2
    assert artifact["status"] == "error"
    assert artifact["records"] == []
    assert artifact["error"]["code"] == "openai_authentication_required"


def test_factorial_memagent_wrapper_executes_exactly_one_harness_run(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_wrapper_test")
    legacy = namespace["legacy"]

    class StructuredCodexHarness(FakeHarness):
        name = "codex"

        def probe(self):
            capabilities = super().probe()
            capabilities.metadata["output_schema"] = True
            return capabilities

        def run(self, task, *, workspace, context_pack, emit, cancel_event):
            findings = [
                {
                    "criterion_id": criterion_id,
                    "title": criterion_id,
                    "finding": f"Finding for {criterion_id}",
                    "severity": "medium",
                    "evidence": ["access_policy.py:1"],
                    "missing_tests": ["Add the requirement-derived case"],
                    "source_ids": [context_pack.source_ids[0]],
                }
                for criterion_id in legacy.REQUIRED_FINDING_IDS
            ]
            return AdapterOutcome(
                final_response=json.dumps({"findings": findings}),
                usage={"input_tokens": 100, "output_tokens": 50},
                cost_usd=0.01,
                exit_code=0,
            )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    legacy._write_fixture(workspace)
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    source_id = provider.store(
        {
            "title": "Export policy requirements",
            "content": f"Applicable review requirements:\n{legacy.REQUIREMENTS}",
            "user_id": "eval-user",
            "thread_id": "repeat-1",
        },
        MemoryType.KNOWLEDGE_BASE,
        memory_id="wrapper-memory",
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[StructuredCodexHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(workspace)],
    )
    runtime = {
        "providers": {"filesystem": provider},
        "services": {"filesystem": service},
        "agent_ids": {"filesystem": set()},
    }
    try:
        record = namespace["run_wrapper"](
            runtime,
            arm="memagent_codex__filesystem",
            display_name="MemAgent → Codex (Filesystem)",
            provider_name="filesystem",
            harness="codex",
            model="gpt-5.6-luna",
            scope={
                "memory_id": "wrapper-memory",
                "source_id": str(source_id),
                "seed_latency_ms": 0,
            },
            repeat=1,
            workspace=workspace,
            permissions=HarnessPermissions(
                workspace_mode="read_only",
                allowed_roots=[str(workspace)],
                network="none",
                mcp_access="none",
            ),
            verification=VerificationSpec(command=f'"{sys.executable}" -B verify.py'),
        )

        assert record["ok"] is True, record
        assert record["interface"] == "memagent_runtime"
        assert len(record["runs"]) == 1
        assert record["runs"][0]["harness"] == "codex"
        assert record["runs"][0]["verified"] is True
        assert record["runs"][0]["grounded"] is True
        assert record["structured_coverage"]["coverage_complete"] is True
        assert len(service.list_runs(limit=10)) == 1
    finally:
        service.close()
        provider.close()


def test_factorial_full_panel_executes_codex_and_claude_via_memagents(
    tmp_path: Path,
) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_full_panel_test")
    legacy = namespace["legacy"]

    class StructuredHarness(FakeHarness):
        def __init__(self) -> None:
            super().__init__()
            self.tasks = []

        def probe(self):
            capabilities = super().probe()
            capabilities.metadata["output_schema"] = True
            return capabilities

        def run(self, task, *, workspace, context_pack, emit, cancel_event):
            self.tasks.append(task.task)
            findings = [
                {
                    "criterion_id": criterion_id,
                    "title": criterion_id,
                    "finding": f"{self.name} finding for {criterion_id}",
                    "severity": "medium",
                    "evidence": ["access_policy.py:1"],
                    "missing_tests": ["Add the requirement-derived case"],
                    "source_ids": [context_pack.source_ids[0]],
                }
                for criterion_id in legacy.REQUIRED_FINDING_IDS
            ]
            return AdapterOutcome(
                final_response=json.dumps({"findings": findings}),
                usage={"input_tokens": 100, "output_tokens": 50},
                cost_usd=0.01,
                exit_code=0,
            )

    class StructuredCodexHarness(StructuredHarness):
        name = "codex"

    class StructuredClaudeHarness(StructuredHarness):
        name = "claude-code"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    legacy._write_fixture(workspace)
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    source_id = provider.store(
        {
            "title": "Export policy requirements",
            "content": f"Applicable review requirements:\n{legacy.REQUIREMENTS}",
            "user_id": "eval-user",
            "thread_id": "repeat-1",
        },
        MemoryType.KNOWLEDGE_BASE,
        memory_id="panel-memory",
    )
    codex_harness = StructuredCodexHarness()
    claude_harness = StructuredClaudeHarness()
    service = MetaHarness(
        memory_provider=provider,
        adapters=[codex_harness, claude_harness],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(workspace)],
    )
    runtime = {
        "providers": {"filesystem": provider},
        "services": {"filesystem": service},
        "scope_ids": {"filesystem": set()},
        "agent_ids": {"filesystem": set()},
    }
    try:
        record = legacy.run_panel(
            runtime,
            run_scope="full-panel-test",
            arm="panel_full__filesystem",
            provider_name="filesystem",
            scope={
                "memory_id": "panel-memory",
                "source_id": str(source_id),
                "seed_latency_ms": 0,
            },
            repeat=1,
            workspace=workspace,
            permissions=HarnessPermissions(
                workspace_mode="read_only",
                allowed_roots=[str(workspace)],
                network="none",
                mcp_access="none",
            ),
            verification=VerificationSpec(command=f'"{sys.executable}" -B verify.py'),
            codex_model="gpt-5.6-luna",
            claude_model="sonnet",
            adaptive=False,
            display_name="MemAgent Full Panel (Filesystem)",
        )

        assert record["ok"] is True, record
        assert record["execution"]["strategy"] == "parallel"
        assert {run["harness"] for run in record["runs"]} == {
            "codex",
            "claude-code",
        }
        assert len(record["runs"]) == 2
        assert record["consolidation"]["model_used"] is False
        assert record["structured_coverage"]["coverage_complete"] is True
        assert codex_harness.tasks == [legacy.COMMON_TASK]
        assert claude_harness.tasks == [legacy.COMMON_TASK]
        assert (
            namespace["validate_record"](
                record,
                namespace["arm_specs"](["filesystem"])["panel_full__filesystem"],
            )
            == []
        )
    finally:
        service.close()
        provider.close()


def test_factorial_judges_each_candidate_in_an_independent_blinded_call(
    monkeypatch,
) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_judge_test")

    class FakeJudge:
        def __init__(self) -> None:
            self.payloads = []
            self.closed = False

        def generate(self, messages, tools=None):
            assert tools is None
            payload = json.loads(messages[1]["content"])
            self.payloads.append(payload)
            candidate_id = payload["candidate"]["candidate_id"]
            return json.dumps(
                {
                    "score": {
                        "candidate_id": candidate_id,
                        "correctness_0_35": 30,
                        "gold_coverage_0_30": 25,
                        "evidence_0_15": 12,
                        "actionability_0_10": 8,
                        "grounding_0_10": 10,
                        "covered_gold_ids": ["G1"],
                        "unsupported_claims": [],
                        "rationale": "Grounded and mostly correct.",
                    }
                }
            )

        def get_last_usage(self):
            return {"input_tokens": 100, "output_tokens": 20}

        def close(self):
            self.closed = True

    judge = FakeJudge()
    monkeypatch.setattr(
        namespace["legacy"], "create_llm_provider", lambda config: judge
    )
    records = [
        {"arm": "first", "repeat": 1, "response": "First response"},
        {"arm": "second", "repeat": 1, "response": "Second response"},
    ]

    reports = namespace["judge_records_independently"](
        records, repeats=1, seed=20260822, judge_model="gpt-4.1"
    )

    assert len(judge.payloads) == 2
    assert all("candidate" in payload for payload in judge.payloads)
    assert all("candidates" not in payload for payload in judge.payloads)
    assert reports[0]["mode"] == "independent_candidate_scoring"
    assert reports[0]["judge_calls"] == 2
    assert {record["judge"]["score_0_100"] for record in records} == {85.0}
    assert judge.closed is True


def test_factorial_judge_normalizes_fractional_point_scale() -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_scale_test")

    score = namespace["_validated_judge_score"](
        {
            "candidate_id": "candidate_0101",
            "correctness_0_35": 0.35,
            "gold_coverage_0_30": 0.30,
            "evidence_0_15": 0.15,
            "actionability_0_10": 0.10,
            "grounding_0_10": 0.10,
            "covered_gold_ids": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "unsupported_claims": [],
            "rationale": "All six findings are grounded.",
        },
        "candidate_0101",
    )

    assert score["score_0_100"] == 100.0
    assert score["correctness_0_35"] == 35.0
    assert score["scale"] == "points"
    assert score["scale_normalization"] == "fractional_x100"


def test_factorial_judge_failure_retains_paid_attempt_accounting(monkeypatch) -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_failed_judge_test")

    class InvalidJudge:
        def generate(self, messages, tools=None):
            candidate_id = json.loads(messages[1]["content"])["candidate"][
                "candidate_id"
            ]
            return json.dumps(
                {
                    "score": {
                        "candidate_id": candidate_id,
                        "correctness_0_35": 34,
                        "gold_coverage_0_30": 30,
                        "evidence_0_15": 15,
                        "actionability_0_10": 10,
                        "grounding_0_10": 10,
                        "covered_gold_ids": ["G1"],
                        "unsupported_claims": [],
                        "rationale": "A deduction was applied for verbosity.",
                    }
                }
            )

        def get_last_usage(self):
            return {"input_tokens": 100, "output_tokens": 20}

        def close(self):
            return None

    monkeypatch.setattr(
        namespace["legacy"], "create_llm_provider", lambda config: InvalidJudge()
    )
    with pytest.raises(namespace["JudgeValidationError"]) as captured:
        namespace["judge_records_independently"](
            [{"arm": "one", "repeat": 1, "response": "review"}],
            repeats=1,
            seed=20260822,
            judge_model="gpt-4.1",
        )

    assert len(captured.value.calls) == 2
    assert all(call["status"] == "rejected" for call in captured.value.calls)
    assert all(call["cost_usd"] is not None for call in captured.value.calls)


def test_factorial_judge_rejects_prohibited_verbosity_scoring() -> None:
    script = (
        Path(__file__).parents[2] / "eval" / "metaharness" / "factorial_comparison.py"
    )
    namespace = runpy.run_path(str(script), run_name="factorial_rationale_test")

    with pytest.raises(RuntimeError, match="prohibited style/verbosity"):
        namespace["_validated_judge_score"](
            {
                "candidate_id": "candidate_0101",
                "correctness_0_35": 33,
                "gold_coverage_0_30": 29,
                "evidence_0_15": 14,
                "actionability_0_10": 9,
                "grounding_0_10": 9,
                "covered_gold_ids": ["G1", "G2", "G3", "G4", "G5", "G6"],
                "unsupported_claims": [],
                "rationale": "Minor deductions were applied for verbosity.",
            },
            "candidate_0101",
        )

    for rationale in (
        "The only minor deduction is for slight redundancy.",
        "Minor deductions were made for not explicitly stating no files were modified.",
        "A point was deducted for not quoting the code verbatim.",
    ):
        payload = {
            "candidate_id": "candidate_0101",
            "correctness_0_35": 34,
            "gold_coverage_0_30": 30,
            "evidence_0_15": 15,
            "actionability_0_10": 10,
            "grounding_0_10": 10,
            "covered_gold_ids": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "unsupported_claims": [],
            "rationale": rationale,
        }
        with pytest.raises(RuntimeError, match="prohibited style/verbosity"):
            namespace["_validated_judge_score"](payload, "candidate_0101")

    accepted = namespace["_validated_judge_score"](
        {
            "candidate_id": "candidate_0101",
            "correctness_0_35": 35,
            "gold_coverage_0_30": 30,
            "evidence_0_15": 15,
            "actionability_0_10": 10,
            "grounding_0_10": 10,
            "covered_gold_ids": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "unsupported_claims": [],
            "rationale": "No deductions were applied for verbosity or style.",
        },
        "candidate_0101",
    )
    assert accepted["score_0_100"] == 100.0


def test_read_only_run_is_durable_verified_and_redacted(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Inspect the project",
                workspace=str(workspace),
                harness="fake",
                verification=VerificationSpec(command="test -d ."),
            )
        )

        assert result.status == HarnessStatus.SUCCEEDED
        assert result.verified is True
        assert {
            "context",
            "mcp_setup",
            "adapter",
            "verification",
            "workspace_finalize",
            "total",
        }.issubset(result.phase_timings_ms)
        assert result.phase_timings_ms["total"] == result.latency_ms
        persisted = service.get_run(result.run_id)
        assert persisted is not None
        assert persisted["status"] == "succeeded"
        events = service.events(result.run_id)
        assert any(event["type"] == "verification" for event in events)
        assert "sk-test" not in str(events)
        assert "[REDACTED]" in str(events)
    finally:
        service.close()


def test_related_panel_stages_reuse_one_scoped_context_snapshot(
    tmp_path: Path,
) -> None:
    provider = _CountingContextProvider()
    adapter = _ContextCaptureHarness()
    service = MetaHarness(
        memory_provider=provider,
        adapters=[adapter],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    common = {
        "memory_query": "Review the policy implementation",
        "shared_context_key": "workflow-1",
    }
    try:
        first = service.run(
            HarnessTask(
                task="Review authorization",
                workspace=str(workspace),
                harness=adapter.name,
                memory_id="memory-1",
                user_id="user-1",
                thread_id="thread-1",
                context=common,
            )
        )
        calls_after_first = provider.calls
        second = service.run(
            HarnessTask(
                task="Review test coverage",
                workspace=str(workspace),
                harness=adapter.name,
                memory_id="memory-1",
                user_id="user-1",
                thread_id="thread-1",
                context=common,
            )
        )

        assert first.ok and second.ok
        assert calls_after_first > 0
        assert provider.calls == calls_after_first
        assert len(adapter.context_packs) == 2
        assert (
            adapter.context_packs[0].fingerprint == adapter.context_packs[1].fingerprint
        )
        assert adapter.context_packs[0].source_ids == ["source-1"]
        assert (
            adapter.context_packs[0].content_fingerprint
            == adapter.context_packs[1].content_fingerprint
        )
        assert adapter.context_packs[0].metadata["context_cache_hit"] is False
        assert adapter.context_packs[1].metadata["shared_context_reused"] is True
        assert adapter.context_packs[1].metadata["context_cache_hit"] is True
        assert adapter.context_packs[1].metadata["context_access_latency_ms"] == 0
    finally:
        service.close()


def test_context_content_fingerprint_ignores_provider_specific_source_ids() -> None:
    first = HarnessContextPack(
        query="Review",
        rendered=(
            "[memory:knowledge_base:"
            "11111111-1111-1111-1111-111111111111] policy requirement"
        ),
        source_ids=["11111111-1111-1111-1111-111111111111"],
    )
    second = HarnessContextPack(
        query="Review",
        rendered=(
            "[memory:knowledge_base:"
            "22222222-2222-2222-2222-222222222222] policy requirement"
        ),
        source_ids=["22222222-2222-2222-2222-222222222222"],
    )

    assert first.fingerprint != second.fingerprint
    assert first.content_fingerprint == second.content_fingerprint


def test_context_cache_does_not_cross_agent_ownership_without_explicit_scope(
    tmp_path: Path,
) -> None:
    provider = _CountingContextProvider()
    service = MetaHarness(
        memory_provider=provider,
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
    )
    common = {
        "memory_query": "Review the policy implementation",
        "shared_context_key": "workflow-1",
    }
    try:
        first = HarnessTask(
            task="First review",
            workspace=str(tmp_path),
            agent_id="agent-a",
            memory_id="memory-1",
            user_id="user-1",
            thread_id="thread-1",
            context=common,
        )
        second = HarnessTask.from_dict(
            {**first.to_dict(), "task": "Second review", "agent_id": "agent-b"}
        )
        service._context_for_task(first)
        calls_after_first = provider.calls
        service._context_for_task(second)

        assert provider.calls > calls_after_first
    finally:
        service.close()


def test_subprocess_prompt_renders_bounded_stage_evidence_after_memory() -> None:
    task = HarnessTask(
        task="Audit only unresolved gaps",
        workspace="/tmp/workspace",
        context={
            "model_context": {"primary": "Finding with evidence"},
            "private_host_value": "must-not-be-rendered",
        },
    )
    pack = HarnessContextPack(
        query="Review",
        rendered="[memory:knowledge_base:source-1] policy",
        source_ids=["source-1"],
    )

    prompt = SubprocessHarness._prompt(task, pack)

    assert prompt.index("Host execution contract") < prompt.index(
        "MemoRizz memory context"
    )
    assert prompt.index("MemoRizz memory context") < prompt.index(
        "MemoRizz stage context"
    )
    assert prompt.index("MemoRizz stage context") < prompt.index("--- Task ---")
    assert "Finding with evidence" in prompt
    assert "must-not-be-rendered" not in prompt


def test_write_run_uses_exact_single_use_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    try:
        pending = service.run(
            HarnessTask(
                task="Create the result",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
                verification=VerificationSpec(command="test -f result.txt"),
            )
        )
        assert pending.status == HarnessStatus.PENDING_APPROVAL
        proposal_id = pending.checkpoint["proposal_id"]

        proposal = service.approve(proposal_id, approver_id="host-user")
        assert proposal["approver_id"] == "host-user"
        assert proposal["arguments"]["context_fingerprint"]
        assert proposal["arguments"]["metadata_fingerprint"]
        result = service.resume_approval(proposal_id)
        assert result.ok is True
        assert result.workspace_diff and "result.txt" in result.workspace_diff
        with pytest.raises(ApprovalStateError):
            service.resume_approval(proposal_id)
    finally:
        service.close()


def test_model_supplied_host_verification_requires_exact_approval(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "must-not-run-before-approval"
    try:
        result = service.run(
            HarnessTask(
                task="Inspect only",
                workspace=str(workspace),
                harness="fake",
                verification=VerificationSpec(
                    command="touch must-not-run-before-approval"
                ),
                metadata={"model_initiated": True},
            )
        )
        assert result.status == HarnessStatus.PENDING_APPROVAL
        assert not marker.exists()
        proposal = service.approval_store.get(result.checkpoint["proposal_id"])
        assert proposal is not None
        assert (
            proposal.arguments["verification"]["command"]
            == "touch must-not-run-before-approval"
        )
    finally:
        service.close()


def test_workspace_change_invalidates_approved_envelope(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    try:
        pending = service.run(
            HarnessTask(
                task="Create the result",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
            )
        )
        proposal_id = pending.checkpoint["proposal_id"]
        service.approve(proposal_id, approver_id="host-user")
        (workspace / "README.md").write_text("changed\n", encoding="utf-8")

        result = service.resume_approval(proposal_id)
        assert result.error_code == "workspace_changed_after_approval"
        persisted = service.get_run(result.run_id)
        assert persisted is not None
        assert persisted["status"] == "failed"
    finally:
        service.close()


def test_untracked_content_change_invalidates_dirty_workspace_approval(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    untracked = workspace / "draft.txt"
    untracked.write_text("before\n", encoding="utf-8")
    try:
        pending = service.run(
            HarnessTask(
                task="Use the reviewed dirty workspace",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(
                    workspace_mode="direct", allow_dirty_workspace=True
                ),
            )
        )
        proposal_id = pending.checkpoint["proposal_id"]
        service.approve(proposal_id, approver_id="host-user")
        untracked.write_text("after\n", encoding="utf-8")

        result = service.resume_approval(proposal_id)
        assert result.error_code == "workspace_changed_after_approval"
        assert not (workspace / "result.txt").exists()
    finally:
        service.close()


def test_non_git_workspace_change_invalidates_approval(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("before\n", encoding="utf-8")
    try:
        pending = service.run(
            HarnessTask(
                task="Use the reviewed non-Git workspace",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
            )
        )
        proposal_id = pending.checkpoint["proposal_id"]
        service.approve(proposal_id, approver_id="host-user")
        source.write_text("after\n", encoding="utf-8")

        result = service.resume_approval(proposal_id)
        assert result.error_code == "workspace_changed_after_approval"
        assert not (workspace / "result.txt").exists()
    finally:
        service.close()


def test_cancel_pending_approval_prevents_later_execution(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    try:
        pending = service.run(
            HarnessTask(
                task="Create the result",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
            )
        )
        proposal_id = pending.checkpoint["proposal_id"]
        canceled = service.cancel(pending.run_id)
        assert canceled["status"] == "canceled"
        assert service.approval_store.get(proposal_id).status.value == "rejected"
        with pytest.raises(ApprovalStateError):
            service.resume_approval(proposal_id)
        assert not (workspace / "result.txt").exists()
    finally:
        service.close()


def test_cancel_after_approval_consumes_without_executing(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    try:
        pending = service.run(
            HarnessTask(
                task="Create the result",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
            )
        )
        proposal_id = pending.checkpoint["proposal_id"]
        service.approve(proposal_id, approver_id="host-user")
        service.cancel(pending.run_id)
        result = service.resume_approval(proposal_id)
        assert result.status == HarnessStatus.CANCELED
        assert result.error_code == "canceled_before_resume"
        assert service.approval_store.get(proposal_id).status.value == "consumed"
        assert not (workspace / "result.txt").exists()
    finally:
        service.close()


def test_cancellation_is_visible_across_service_process_boundaries(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "runs.sqlite3"
    approval_path = tmp_path / "approvals.sqlite3"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    worker = MetaHarness(
        adapters=[FakeHarness(block=True)],
        run_store=SQLiteHarnessRunStore(store_path),
        approval_store=SQLiteApprovalStore(approval_path),
        allowed_workspace_roots=[str(tmp_path)],
    )
    controller = MetaHarness(
        adapters=[FakeHarness()],
        run_store=SQLiteHarnessRunStore(store_path),
        approval_store=SQLiteApprovalStore(approval_path),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        run = worker.start(
            HarnessTask(
                task="Wait until canceled",
                workspace=str(workspace),
                harness="fake",
            )
        )
        deadline = time.monotonic() + 3
        while worker.run_store.get(run.run_id).status != HarnessStatus.RUNNING:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        canceled = controller.cancel(run.run_id)
        assert canceled["ok"] is True

        while True:
            current = worker.run_store.get(run.run_id)
            assert current is not None
            if current.status.terminal:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert current.status == HarnessStatus.CANCELED
        assert current.cancel_requested is True
    finally:
        controller.close()
        worker.close()


def test_cancellation_stops_host_verification(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / "verification-started"
    command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "from pathlib import Path; import time; "
        "Path('verification-started').write_text('yes'); time.sleep(20)"
    )
    try:
        run = service.start(
            HarnessTask(
                task="Finish before a long host check",
                workspace=str(workspace),
                harness="fake",
                verification=VerificationSpec(command=command),
            )
        )
        deadline = time.monotonic() + 5
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        service.cancel(run.run_id)
        while True:
            current = service.run_store.get(run.run_id)
            assert current is not None
            if current.status.terminal:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert current.status == HarnessStatus.CANCELED
        assert current.result["verification"]["error"] == "verification_canceled"
    finally:
        service.close()


def test_dirty_write_workspace_is_rejected_before_adapter_execution(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    workspace = _git_workspace(tmp_path / "workspace")
    (workspace / "README.md").write_text("dirty\n", encoding="utf-8")
    try:
        result = service.run(
            HarnessTask(
                task="Change files",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(workspace_mode="direct"),
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "HarnessSecurityError"
    finally:
        service.close()


def test_router_rejects_unavailable_explicit_adapter(tmp_path: Path) -> None:
    service = _service(tmp_path, FakeHarness(available=False))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(task="Inspect", workspace=str(workspace), harness="fake")
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "harness_unavailable"
        assert "doctor fake" in str(result.error)
        assert result.routing["rejection"]["code"] == "harness_unavailable"
    finally:
        service.close()


def test_task_roots_cannot_widen_operator_workspace_boundary(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    service = MetaHarness(
        adapters=[FakeHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(allowed)],
    )
    try:
        result = service.run(
            HarnessTask(
                task="Attempt to widen roots",
                workspace=str(outside),
                harness="fake",
                permissions=HarnessPermissions(allowed_roots=[str(outside)]),
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "HarnessSecurityError"
        assert "outside the configured allowed roots" in str(result.error)
    finally:
        service.close()


def test_native_adapter_cannot_route_back_into_the_same_memagent(
    tmp_path: Path,
) -> None:
    class SameAgent:
        agent_id = "same-agent"

        def run(self, *args, **kwargs):  # pragma: no cover - must not execute
            raise AssertionError("native recursion guard failed")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, NativeMemAgentHarness(SameAgent()))
    try:
        result = service.run(
            HarnessTask(
                task="Do not recurse",
                workspace=str(workspace),
                harness="native",
                agent_id="same-agent",
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert "native_recursion_guard" in str(result.error)
    finally:
        service.close()


def test_native_adapter_reports_memagent_model_usage(tmp_path: Path) -> None:
    class UsageModel:
        def get_last_usage(self):
            return {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            }

    class WorkerAgent:
        agent_id = "native-worker"
        model = UsageModel()

        def run(self, *args, **kwargs):
            return "native response"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = NativeMemAgentHarness(WorkerAgent())
    service = _service(tmp_path, adapter)
    try:
        capability = service.probe("memagent")
        assert capability["usage_reporting"] is True
        assert capability["metadata"]["token_reporting"] is True
        result = service.run(
            HarnessTask(
                task="Run the native worker",
                workspace=str(workspace),
                harness="memagent",
                agent_id="comparison-controller",
            )
        )
        assert result.status == HarnessStatus.SUCCEEDED
        assert result.final_response == "native response"
        assert result.usage == {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
        }
    finally:
        service.close()


def test_provider_backed_service_exposes_explicit_native_agent_adapter(
    tmp_path: Path,
) -> None:
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    service = MetaHarness.from_env(
        memory_provider=provider,
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        assert "memagent" in {
            row["name"] for row in service.list_harnesses(probe=False)
        }
        native = service.probe("native")
        assert native["name"] == "memagent"
        assert native["ready"] is False
        assert "No saved MemAgent" in native["error"]
        assert native["metadata"]["requires_agent_id"] is True
    finally:
        service.close()
        provider.close()


def test_persisted_native_adapter_runs_the_selected_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memorizz.memagent.builders import MemAgentBuilder
    from tests.mocks.mock_providers import MockLLMProvider

    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    agent = (
        MemAgentBuilder()
        .with_name("Persisted worker")
        .with_model(MockLLMProvider(["saved worker response"]))
        .with_memory_provider(provider)
        .with_memory_ids("native-memory")
        .build_and_save()
    )
    agent.close(close_memory_provider=False)
    monkeypatch.setattr(
        "memorizz.memagent.core.create_llm_provider",
        lambda _config: MockLLMProvider(["saved worker response"]),
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[PersistedMemAgentHarness(provider)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Use the selected saved worker",
                workspace=str(workspace),
                harness="native",
                agent_id=agent.agent_id,
                memory_id="native-memory",
                user_id="native-user",
                thread_id="native-thread",
                permissions=HarnessPermissions(mcp_access="none"),
            )
        )
        assert result.status == HarnessStatus.SUCCEEDED
        assert result.final_response == "saved worker response"
    finally:
        service.close()
        provider.close()


def test_memagent_runtime_mode_executes_full_turn_on_selected_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memorizz.memagent.builders import MemAgentBuilder

    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(tmp_path / "memory"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path)
    agent = (
        MemAgentBuilder()
        .with_instruction("Use the governed runtime")
        .with_execution_harness(
            "fake",
            meta_harness=service,
            config={
                "workspace": str(workspace),
                "model": "pinned-evaluation-model",
                "permissions": {"mcp_access": "none"},
            },
        )
        .build()
    )
    try:
        response = agent.run(
            "Inspect this workspace",
            memory_id="memory-1",
            thread_id="thread-1",
            user_id="user-1",
        )
        assert response == "done"
        run = service.list_runs(limit=1)[0]
        assert run["harness"] == "fake"
        assert run["task"]["agent_id"] == agent.agent_id
        assert run["task"]["mode"] == "runtime"
        assert run["task"]["model"] == "pinned-evaluation-model"
    finally:
        agent.close(close_memory_provider=True)
        service.close()


def test_memagent_delegate_mode_registers_bounded_harness_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memorizz.memagent.builders import MemAgentBuilder

    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(tmp_path / "memory"))
    service = _service(tmp_path)
    agent = (
        MemAgentBuilder()
        .with_instruction("Delegate specialist workspace work")
        .with_meta_harness(service, mode="delegate", default_harness="fake")
        .build()
    )
    try:
        assert {
            "run_harness_task",
            "get_harness_run",
            "list_agent_harnesses",
        }.issubset(agent.tool_manager.tools)
    finally:
        agent.close(close_memory_provider=True)
        service.close()


def test_memagent_configuration_allows_registered_unready_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Portable agent configuration must not depend on the builder host's CLIs."""
    from memorizz.memagent.builders import MemAgentBuilder

    monkeypatch.setenv("MEMORIZZ_MEMORY_ROOT", str(tmp_path / "memory"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = MetaHarness(
        adapters=[CodexHarness(command=str(tmp_path / "missing-codex"))],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(workspace)],
    )
    agent = (
        MemAgentBuilder()
        .with_execution_harness(
            "codex",
            meta_harness=service,
            config={"workspace": str(workspace)},
        )
        .build(validate=True)
    )
    try:
        assert agent.validate_configuration()["ok"] is True
        capability = service.probe("codex")
        assert capability["ready"] is False
        assert capability["error_code"] == "harness_unavailable"
    finally:
        agent.close(close_memory_provider=True)
        service.close()


def test_reported_cost_budget_is_enforced_for_custom_adapters(tmp_path: Path) -> None:
    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Inspect within a strict budget",
                workspace=str(workspace),
                harness="fake",
                budget={"max_cost_usd": 0.001},
            )
        )
        assert result.status == HarnessStatus.BUDGET_EXCEEDED
        assert result.error_code == "cost_budget_exceeded"
    finally:
        service.close()


def test_verified_run_becomes_continual_learning_evidence(tmp_path: Path) -> None:
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=tmp_path / "memory",
            embedding_provider=None,
            lazy_vector_indexes=True,
            use_faiss=False,
        )
    )
    service = MetaHarness(
        memory_provider=provider,
        adapters=[FakeHarness()],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Inspect and verify",
                workspace=str(workspace),
                harness="fake",
                agent_id="agent-learning",
                memory_id="memory-learning",
                user_id="user-learning",
                thread_id="thread-learning",
                verification=VerificationSpec(command="test -d ."),
            )
        )
        assert result.ok is True
        rows = provider.list_all(MemoryType.SHARED_MEMORY)
        payloads = [json.loads(row["content"]) for row in rows if row.get("content")]
        event_types = {
            payload.get("event_type")
            for payload in payloads
            if payload.get("record_type") == "learning_event"
        }
        assert "workflow_recorded" in event_types
        assert "outcome_recorded" in event_types
        outcome = next(
            payload
            for payload in payloads
            if payload.get("event_type") == "outcome_recorded"
        )
        assert outcome["payload"]["verified"] is True
        assert outcome["user_id"] == "user-learning"
    finally:
        service.close()
        provider.close()


def test_vendor_commands_use_noninteractive_bounded_security_flags(
    tmp_path: Path,
) -> None:
    output_schema = {
        "type": "object",
        "properties": {"finding": {"type": "string"}},
        "required": ["finding"],
        "additionalProperties": False,
    }
    read_task = HarnessTask(
        task="Inspect",
        workspace=str(tmp_path),
        harness="codex",
        permissions=HarnessPermissions(mcp_access="none"),
        output_schema=output_schema,
        metadata={
            "codex_config": ['web_search="live"', "hooks.pre=[]"],
            "_memorizz_output_schema_path": str(tmp_path / "output-schema.json"),
        },
    )
    codex = CodexHarness().build_command(read_task, workspace=tmp_path, prompt="PROMPT")
    assert codex[:3] == ["codex", "exec", "--json"]
    assert codex.count("--ephemeral") == 1
    assert "--ignore-user-config" in codex
    assert "--ignore-rules" in codex
    assert "--skip-git-repo-check" in codex
    assert codex[codex.index("--sandbox") + 1] == "read-only"
    codex_configs = [
        codex[index + 1]
        for index, value in enumerate(codex[:-1])
        if value == "--config"
    ]
    assert codex_configs.index('web_search="live"') < codex_configs.index(
        'web_search="disabled"'
    )
    assert "hooks={}" in codex_configs
    assert "mcp_servers={}" in codex_configs
    assert "sandbox_workspace_write.network_access=false" in codex_configs
    assert "sandbox_workspace_write.writable_roots=[]" in codex_configs
    assert codex[codex.index("--output-schema") + 1] == str(
        tmp_path / "output-schema.json"
    )
    assert codex[-2:] == ["--", "PROMPT"]

    write_task = HarnessTask(
        task="Edit",
        workspace=str(tmp_path),
        harness="claude-code",
        permissions=HarnessPermissions(
            workspace_mode="direct",
            network="none",
            mcp_access="read_only",
            denied_tools=["NotebookEdit"],
        ),
        output_schema=output_schema,
        metadata={"claude_effort": "medium"},
    )
    claude = ClaudeCodeHarness().build_command(
        write_task, workspace=tmp_path, prompt="PROMPT"
    )
    assert "--bare" in claude
    assert "--no-session-persistence" in claude
    assert claude[claude.index("--max-turns") + 1] == str(write_task.budget.max_steps)
    allowed = claude[claude.index("--allowedTools") + 1]
    denied = claude[claude.index("--disallowedTools") + 1]
    assert "Edit" in allowed and "Write" in allowed
    assert "mcp__memorizz__*" in allowed
    assert "Bash" in denied and "WebFetch" in denied and "WebSearch" in denied
    assert "NotebookEdit" in denied
    assert claude[claude.index("--effort") + 1] == "medium"
    assert json.loads(claude[claude.index("--json-schema") + 1]) == output_schema
    assert claude[-2:] == ["--", "PROMPT"]


def test_output_schema_fails_closed_for_unsupported_adapter(tmp_path: Path) -> None:
    service = _service(tmp_path)
    try:
        result = service.run(
            HarnessTask(
                task="Return a finding",
                workspace=str(tmp_path),
                harness="fake",
                output_schema={
                    "type": "object",
                    "properties": {"finding": {"type": "string"}},
                },
            )
        )

        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "output_schema_unsupported"
        assert "output_schema_unsupported" in str(result.error)
    finally:
        service.close()


def test_claude_probe_reports_missing_authentication_without_secret_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "claude"
    executable.write_text(
        "#!/bin/sh\necho 'Claude Code 1.0'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    for name in ClaudeCodeHarness.auth_environment:
        monkeypatch.delenv(name, raising=False)

    capability = ClaudeCodeHarness(command=str(executable)).probe().to_dict()

    assert capability["available"] is True
    assert capability["ready"] is False
    assert capability["error_code"] == "authentication_required"
    assert "ANTHROPIC_API_KEY" in capability["error"]
    assert "doctor claude-code" in capability["remediation"]
    assert capability["metadata"]["authentication_configured"] is False


def test_codex_probe_accepts_cli_login_and_reports_when_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "codex"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'codex 1.0\'; exit 0; fi\n'
        'if [ "$1" = "login" ] && [ "$2" = "status" ]; then '
        "echo 'Not logged in'; exit 1; fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_API_KEY", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)

    capability = CodexHarness(command=str(executable)).probe().to_dict()

    assert capability["available"] is True
    assert capability["ready"] is False
    assert capability["error_code"] == "authentication_required"
    assert capability["metadata"]["authentication_status"] == "missing"
    assert "codex login" in capability["remediation"]


def test_codex_restores_only_generated_memorizz_mcp_after_policy_reset(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    task = HarnessTask(
        task="Recall scoped memory",
        workspace=str(tmp_path),
        harness="codex",
        user_id="mcp-user",
        permissions=HarnessPermissions(mcp_access="read_only"),
    )
    try:
        service._configure_mcp(task, tmp_path)
        command = CodexHarness().build_command(
            task, workspace=tmp_path, prompt="PROMPT"
        )
        configs = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--config"
        ]
        reset = configs.index("mcp_servers={}")
        generated = [
            index
            for index, value in enumerate(configs)
            if value.startswith("mcp_servers.memorizz.")
        ]
        assert generated and all(index > reset for index in generated)
        server_config = json.loads((tmp_path / "mcp.json").read_text())
        server_env = server_config["mcpServers"]["memorizz"]["env"]
        assert server_env["MEMORIZZ_MCP_SERVER_ALLOW_WRITES"] == "false"
        assert server_env["MEMORIZZ_MCP_SERVER_LOCAL_PRINCIPAL"] == "mcp-user"
    finally:
        service.close()


def test_capability_ready_and_task_tool_policy_are_fail_closed(tmp_path: Path) -> None:
    unavailable = HarnessCapabilities(
        name="auth-required",
        available=True,
        error_code="authentication_required",
        error="authentication missing",
        remediation="Set TEST_HARNESS_API_KEY.",
        metadata={"authentication_configured": False},
    )
    capability = unavailable.to_dict()
    assert capability["ready"] is False
    assert capability["error_code"] == "authentication_required"
    assert capability["remediation"] == "Set TEST_HARNESS_API_KEY."

    service = _service(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        result = service.run(
            HarnessTask(
                task="Request an adapter-specific tool",
                workspace=str(workspace),
                harness="fake",
                permissions=HarnessPermissions(allowed_tools=["custom-tool"]),
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert "task_tool_policy_unsupported" in str(result.error)
    finally:
        service.close()


def test_sdk_returns_typed_missing_authentication_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(tmp_path, _AuthRequiredHarness())
    try:
        result = service.run(
            HarnessTask(
                task="Inspect",
                workspace=str(workspace),
                harness="auth-required",
                permissions=HarnessPermissions(mcp_access="none"),
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "authentication_required"
        assert "API key" in str(result.error)
        assert "TEST_HARNESS_API_KEY" in str(result.remediation)
        assert result.routing["rejection"]["harness"] == "auth-required"
    finally:
        service.close()


def test_subprocess_execution_is_pinned_to_the_probed_executable(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted-runner"
    trusted.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"message\":\"trusted\"}'\n",
        encoding="utf-8",
    )
    trusted.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    malicious = workspace / "runner"
    malicious.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    malicious.chmod(0o700)
    service = _service(tmp_path, _PinnedCommandHarness(command=str(trusted)))
    try:
        result = service.run(
            HarnessTask(
                task="Use the pinned adapter",
                workspace=str(workspace),
                harness="pinned",
                permissions=HarnessPermissions(mcp_access="none"),
            )
        )
        assert result.status == HarnessStatus.SUCCEEDED
        assert result.final_response == "trusted"
    finally:
        service.close()


def test_openhands_requires_operator_attested_external_isolation(
    tmp_path: Path,
) -> None:
    service = MetaHarness(
        adapters=[OpenHandsHarness(command="/bin/echo", external_isolation=False)],
        run_store=SQLiteHarnessRunStore(tmp_path / "runs.sqlite3"),
        approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
        allowed_workspace_roots=[str(tmp_path)],
    )
    try:
        result = service.run(
            HarnessTask(
                task="Edit in OpenHands",
                workspace=str(tmp_path),
                harness="openhands",
                metadata={"execution_backend": "docker"},
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "external_isolation_required"
        assert "MEMORIZZ_OPENHANDS_EXTERNAL_ISOLATION=true" in str(result.remediation)
    finally:
        service.close()


def test_subprocess_runtime_authentication_failure_is_normalized(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "auth-runner"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'auth-runner 1.0\'; exit 0; fi\n'
        "echo '401 Unauthorized: invalid API key' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = _service(
        tmp_path,
        _AuthenticationFailingHarness(command=str(executable)),
    )
    try:
        result = service.run(
            HarnessTask(
                task="Inspect",
                workspace=str(workspace),
                harness="auth-failing",
                permissions=HarnessPermissions(mcp_access="none"),
            )
        )
        assert result.status == HarnessStatus.FAILED
        assert result.error_code == "authentication_required"
        assert result.error == "auth-failing could not authenticate."
        assert "TEST_HARNESS_API_KEY" in str(result.remediation)
        assert "Unauthorized" not in str(result.error)
    finally:
        service.close()
