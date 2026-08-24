"""Regression coverage for no-display MemoRizz operation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_cli_and_sdk_run_without_display_or_ui(tmp_path):
    project_root = Path(__file__).parents[2]
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    environment["PYTHONPATH"] = str(project_root / "src")
    environment["MEMORIZZ_HOME"] = str(tmp_path / "cli")
    environment["OPENAI_API_KEY"] = ""
    environment["ANTHROPIC_API_KEY"] = ""
    environment["AZURE_OPENAI_API_KEY"] = ""
    environment["MEMORIZZ_DEFAULT_EMBEDDING_PROVIDER"] = ""

    created = subprocess.run(
        [
            sys.executable,
            "-m",
            "memorizz.cli",
            "agents",
            "create",
            "--name",
            "Headless CLI Agent",
            "--instruction",
            "Run without a display server.",
            "--no-llm",
            "--json",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert created.returncode == 0, created.stderr
    assert json.loads(created.stdout)["agent"]["name"] == "Headless CLI Agent"

    environment["MEMORIZZ_HOME"] = str(tmp_path / "sdk")
    sdk = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from memorizz import MemAgent, MemAgentBuilder, MetaHarness, capabilities; "
                "from tests.mocks.mock_providers import MockLLMProvider; "
                "agent=(MemAgentBuilder().with_name('Headless SDK Agent')"
                ".with_model(MockLLMProvider(['headless-ok']))"
                ".with_memory_ids('headless-memory').build_and_save()); "
                "answer=agent.run('confirm', memory_id='headless-memory', "
                "user_id='headless-user', thread_id='headless-thread'); "
                "loaded=MemAgent.load(agent.agent_id, memory_provider=agent.memory_provider, "
                "model=MockLLMProvider(['reload-ok'])); "
                "assert answer == 'headless-ok'; "
                "assert loaded.name == 'Headless SDK Agent'; "
                "assert MetaHarness is not None; "
                "assert capabilities()['features']['meta_harness']['available']; "
                "assert 'memorizz.ui.app' not in sys.modules; "
                "print('headless-ok')"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert sdk.returncode == 0, sdk.stderr
    assert sdk.stdout.strip() == "headless-ok"
