"""Passive evaluation through non-streaming and streaming MemAgent runs."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from memorizz.enums import MemoryType
from memorizz.long_term.procedural.skillbox import Skill, SkillStatus
from memorizz.long_term.procedural.workflow.canonicalization import (
    canonical_hash,
    canonical_signature,
)
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider


class OneVectorEmbedding:
    def get_embedding(self, _text, **_kwargs):
        return [1.0]

    def get_provider_info(self):
        return {"provider": "test", "dimensions": 1}


class ToolCallingModel:
    provider = "test"
    model_name = "test-shadow-runtime"

    def __init__(self):
        self.generate_calls = 0
        self.stream_calls = 0
        self.messages = []

    @staticmethod
    def _tool_response(call_id):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id=call_id,
                                function=SimpleNamespace(
                                    name="observe_order",
                                    arguments=json.dumps({"order_id": "R-1"}),
                                ),
                            )
                        ],
                    )
                )
            ]
        )

    def generate(self, messages, tools=None, **_kwargs):
        self.generate_calls += 1
        self.messages.append(list(messages))
        if self.generate_calls == 1:
            return self._tool_response("nonstream-call")
        return "Observed order R-1."

    def generate_stream(self, messages, tools=None, **_kwargs):
        self.stream_calls += 1
        self.messages.append(list(messages))
        yield {
            "type": "tool_calls",
            "response": self._tool_response("stream-call"),
        }
        yield {"type": "content", "content": "Observed order R-1."}
        yield {"type": "done", "content": "Observed order R-1."}

    def get_last_usage(self):
        return {}


@pytest.mark.integration
def test_shadow_evaluation_uses_shared_post_store_funnel_without_reexecution(
    tmp_path,
):
    provider = FileSystemProvider(
        FileSystemConfig(
            root_path=Path(tmp_path) / "shadow-runtime",
            embedding_provider=OneVectorEmbedding(),
            lazy_vector_indexes=True,
        )
    )
    model = ToolCallingModel()
    tool_calls = []

    def observe_order(order_id: str):
        """Read one order without changing it."""
        tool_calls.append(order_id)
        return {"order_id": order_id, "status": "completed"}

    agent = MemAgent(
        model=model,
        memory_provider=provider,
        tools=[observe_order],
        agent_id="shadow-runtime-agent",
        continual_learning=True,
        continual_learning_config={
            "require_shadow": True,
            "shadow_evaluation_enabled": True,
            "shadow_evaluation_min_similarity": 0.0,
            "promotion_every_n_runs": 0,
        },
    )
    expected_hash = canonical_hash(
        canonical_signature(
            {
                "Step 1: observe_order": {
                    "arguments": {"order_id": "R-1"},
                    "error": None,
                }
            }
        )
    )
    shadow = Skill(
        name="Observe an order safely",
        description="Use when the user asks to inspect an order.",
        content="Call observe_order once and report the current state.",
        queries=["inspect an order"],
        tools_used=["observe_order"],
        agent_id=agent.agent_id,
        user_id="user-a",
        source_canonical_hash=expected_hash,
        source_workflow_ids=["training-workflow"],
        status=SkillStatus.SHADOW,
        created_at=datetime.now() - timedelta(seconds=1),
        embedding=[1.0],
    )
    agent.continual_learning_manager.skillbox.add_skill(shadow)

    response = agent.run(
        "Inspect order R-1.",
        memory_id="nonstream-memory",
        user_id="user-a",
    )
    assert response == "Observed order R-1."
    list(
        agent.run_stream(
            "Inspect order R-1 again.",
            memory_id="stream-memory",
            user_id="user-a",
        )
    )
    calls_before_drain = (model.generate_calls, model.stream_calls, len(tool_calls))
    assert agent.continual_learning_manager.drain_shadow_evaluations(timeout=2)

    # The evaluator does not call the model or execute the tool a second time.
    assert (model.generate_calls, model.stream_calls, len(tool_calls)) == (
        *calls_before_drain[:2],
        2,
    )
    prompt_text = json.dumps(model.messages, default=str)
    assert shadow.name not in prompt_text
    assert shadow.content not in prompt_text

    workflows = provider.list_all(memory_store_type=MemoryType.WORKFLOW_MEMORY)
    assert len(workflows) == 2
    assert all(doc.get("skills_activated") == [] for doc in workflows)
    assert all(len(doc.get("shadow_evaluations") or []) == 1 for doc in workflows)
    assert all(
        doc["shadow_evaluations"][0]["skill_id"] == shadow.skill_id for doc in workflows
    )
    readiness = agent.continual_learning_manager.get_shadow_readiness(shadow.skill_id)
    assert readiness["observations"] == 2
    # Advisory metrics never activate the skill.
    assert (
        agent.continual_learning_manager.skillbox.get_skill_by_id(
            shadow.skill_id
        ).status
        == SkillStatus.SHADOW
    )
