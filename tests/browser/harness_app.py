"""Shared, isolated portal fixture for harness judging and approval navigation.

The release suite uses synthetic judgments. Set MEMORIZZ_BROWSER_LOCAL_JUDGE=1
to repeat judge acceptance against real local Ollama models instead.
"""

import json
import os
import time
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import uvicorn

from memorizz.approval import SQLiteApprovalStore
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.metaharness import (
    AgentHarness,
    HarnessCapabilities,
    HarnessPermissions,
    HarnessRun,
    HarnessStatus,
    HarnessTask,
    MetaHarness,
    SQLiteHarnessRunStore,
)
from memorizz.metaharness.base import AdapterOutcome
from memorizz.metaharness.judging import HarnessJudge
from memorizz.ui import state
from memorizz.ui.app import create_app


class FixtureHarness(AgentHarness):
    def __init__(self, name, answer):
        self.name, self.answer = name, answer

    def probe(self):
        return HarnessCapabilities(
            name=self.name,
            available=True,
            mcp=True,
            metadata={"network_modes": ["none", "full"], "network_policy": "fixture"},
        )

    def run(self, task, **kwargs):
        return AdapterOutcome(final_response=self.answer, cost_usd=0.01, exit_code=0)


class FixtureJudge:
    def __init__(self, config):
        self.ledger = SimpleNamespace(
            calls=[{"response_metadata": {"response_model": config["model"]}}]
        )

    def generate_text(self, prompt, instructions=None):
        correct = json.loads(prompt)["candidate_answer"] == "Paris"
        return json.dumps(
            {"score": 100 if correct else 0, "rationale": "Synthetic answer key."}
        )


def seed_judgments(service, root):
    for run_id, harness, answer, cost, latency in [
        ("judge-fixture-correct", "codex", "Paris", 0.03, 1800),
        ("judge-fixture-wrong", "claude-code", "London", 0.01, 1000),
    ]:
        service.run_store.create(
            HarnessRun(
                run_id=run_id,
                harness=harness,
                status=HarnessStatus.SUCCEEDED,
                task={"task": "What is the capital of France?", "workspace": root},
                result={
                    "final_response": answer,
                    "cost_usd": cost,
                    "latency_ms": latency,
                },
            )
        )
    if not os.getenv("MEMORIZZ_BROWSER_LOCAL_JUDGE"):
        from memorizz.llms import model_lists

        model_lists.latest_models = lambda *args, **kwargs: ["qwen2.5:3b", "qwen2.5:7b"]
        service._judge = HarnessJudge(service.run_store, model_factory=FixtureJudge)


def seed_approvals(service, root):
    # An unrelated pending run must not redirect a new comparison.
    if not os.getenv("MEMORIZZ_BROWSER_EMPTY_LEDGER"):
        service.start(
            HarnessTask(
                task="Unrelated earlier approval",
                workspace=root,
                harness="codex",
                permissions=HarnessPermissions(network="full", mcp_access="none"),
            )
        )
    drive = service._drive_compare

    def delayed(*args, **kwargs):
        time.sleep(0.8)
        return drive(*args, **kwargs)

    service._drive_compare = delayed


def run():
    with TemporaryDirectory(prefix="memorizz-harness-browser-") as root:
        os.environ.update(
            MEMORIZZ_HOME=root,
            MEMORIZZ_UI_AUTH_TOKEN="memorizz-browser-fixture-token",
            MEMORIZZ_UI_AUTH_ACCOUNTS="{}",
            MEMORIZZ_UI_READ_ONLY="false",
            MEMORIZZ_UI_AUDIT_LOG=root + "/audit.jsonl",
        )
        provider = FileSystemProvider(
            FileSystemConfig(root_path=root + "/memory", lazy_vector_indexes=True)
        )
        service = MetaHarness(
            memory_provider=provider,
            adapters=[
                FixtureHarness("codex", "Paris"),
                FixtureHarness("claude-code", "London"),
            ],
            run_store=SQLiteHarnessRunStore(root + "/runs.sqlite3"),
            approval_store=SQLiteApprovalStore(root + "/approvals.sqlite3"),
            scratch_root=root + "/scratch",
        )
        suite = os.getenv("MEMORIZZ_HARNESS_BROWSER_SUITE", "harness_judge")
        if suite == "harness_judge":
            seed_judgments(service, root)
        else:
            seed_approvals(service, root)
        state._state.update(
            provider=provider,
            provider_type="filesystem",
            connection_info={},
            meta_harness=service,
            meta_harness_provider=provider,
        )
        try:
            uvicorn.run(
                create_app(),
                host="127.0.0.1",
                port=int(os.getenv("MEMORIZZ_BROWSER_TEST_PORT", "8792")),
                lifespan="off",
            )
        finally:
            service.close()
            provider.close()


if __name__ == "__main__":
    run()
