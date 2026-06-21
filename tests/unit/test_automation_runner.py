# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The automation runner suppresses automation-management tools.

An agent executing a scheduled job should do the task, not create/list/run other
automations — leaving those tools on distracts smaller models (off-task / empty
runs). The runner loads the agent with automations_enabled=False.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from memorizz.automation import runner
from memorizz.automation.models import AutomationJob


def _job():
    return AutomationJob(
        job_id="j1",
        agent_id="a1",
        name="n",
        schedule_type="interval",
        interval_seconds=60,
        timezone="UTC",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        action_type="agent_query",
        action_config={"query_template": "What is 2+2?"},
    )


@pytest.mark.unit
def test_runner_loads_agent_with_management_tools_suppressed():
    job = _job()
    fake = MagicMock()
    fake.run.return_value = "4"
    fake._current_memory_id = "m1"
    with patch(
        "memorizz.automation.runner.MemAgent.load", return_value=fake
    ) as load_mock:
        result = runner.execute_job_action(
            job, scheduled_for_utc=job.next_run_at, memory_provider=object()
        )
    assert load_mock.call_args.kwargs.get("automations_enabled") is False
    assert result["response"] == "4"


@pytest.mark.unit
def test_execute_job_action_strips_management_tools(tmp_path):
    from memorizz.llms.llm_factory import create_llm_provider
    from memorizz.memagent.builders import MemAgentBuilder
    from memorizz.memory_provider.filesystem import FileSystemConfig, FileSystemProvider

    prov = FileSystemProvider(FileSystemConfig(root_path=str(tmp_path)))
    model = create_llm_provider({"provider": "ollama", "model": "qwen2.5:7b"})
    agent = (
        MemAgentBuilder()
        .with_model(model)
        .with_memory_provider(prov)
        .with_automations_enabled(True)
        .build()
    )
    # a default agent carries automation-management tools
    assert "automation_create_job" in agent.tool_manager.tools

    job = AutomationJob(
        job_id="j1",
        agent_id=agent.agent_id,
        name="n",
        schedule_type="interval",
        interval_seconds=60,
        timezone="UTC",
        next_run_at=datetime(2030, 1, 1, tzinfo=timezone.utc),
        action_type="agent_query",
        action_config={"query_template": "hi"},
    )
    with patch(
        "memorizz.automation.runner.MemAgent.load", return_value=agent
    ), patch.object(agent, "run", return_value="done"):
        runner.execute_job_action(
            job, scheduled_for_utc=job.next_run_at, memory_provider=prov
        )
    # the runner stripped the management tools for the scheduled run
    assert "automation_create_job" not in agent.tool_manager.tools
