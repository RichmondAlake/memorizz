# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Deep Research specific orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...coordination.shared_memory import SharedMemoryMessageType
from ...enums import ApplicationMode
from ...internet_access import get_default_internet_access_provider
from ...memagent.builders import create_deep_research_agent
from ...multi_agent_orchestrator import MultiAgentOrchestrator
from ...task_decomposition import SubTask

logger = logging.getLogger(__name__)


class DeepResearchOrchestrator(MultiAgentOrchestrator):
    """Extends MultiAgentOrchestrator with Deep Research conventions."""

    def __init__(self, root_agent, delegates, synthesis_agent):
        super().__init__(root_agent, delegates)
        self.synthesis_agent = synthesis_agent
        self._command_to_task: Dict[str, SubTask] = {}
        self._delegate_lookup = {agent.agent_id: agent for agent in delegates}
        self._validate_agent_modes()

    def _validate_agent_modes(self):
        """Ensure all agents run with Deep Research mode."""
        for agent in [self.root_agent, self.synthesis_agent, *self.delegates]:
            if (
                getattr(agent, "application_mode", None)
                != ApplicationMode.DEEP_RESEARCH
            ):
                logger.warning(
                    "Agent %s is not configured with ApplicationMode.DEEP_RESEARCH",
                    getattr(agent, "agent_id", "unknown"),
                )

    def execute(self, user_query: str, memory_id: str = None, thread_id: str = None):
        """Execute Deep Research workflow."""
        return self.execute_multi_agent_workflow(user_query, memory_id, thread_id)

    # Hook overrides -----------------------------------------------------------
    def _after_task_decomposition(
        self, sub_tasks: List[SubTask], user_query: str
    ) -> None:
        if not self.shared_memory_id:
            return

        for task in sub_tasks:
            command_id = task.task_id or f"cmd_{task.assigned_agent_id}"
            self._command_to_task[command_id] = task
            metadata = {
                "original_query": user_query,
                "dependencies": task.dependencies,
            }
            self.shared_memory.post_command(
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                command_id=command_id,
                target_agent_id=task.assigned_agent_id,
                instructions=task.description,
                priority=task.priority,
                dependencies=task.dependencies,
                metadata=metadata,
            )
            self.shared_memory.post_status(
                memory_id=self.shared_memory_id,
                agent_id=task.assigned_agent_id,
                command_id=command_id,
                status="queued",
                progress=0,
            )

    def _after_task_completion(self, task: SubTask, result) -> None:
        if not self.shared_memory_id:
            return

        command_id = task.task_id or f"cmd_{task.assigned_agent_id}"
        agent = self._delegate_lookup.get(task.assigned_agent_id)
        summary_refs = []
        if agent and hasattr(agent, "list_context_summaries"):
            try:
                summary_refs = [
                    item.get("summary_id")
                    for item in agent.list_context_summaries()
                    if item.get("summary_id")
                ]
            except Exception as exc:
                logger.warning(
                    "Failed to collect summary references for %s: %s",
                    task.assigned_agent_id,
                    exc,
                )

        self.shared_memory.post_status(
            memory_id=self.shared_memory_id,
            agent_id=task.assigned_agent_id,
            command_id=command_id,
            status="completed",
            progress=100,
            summary_ids=summary_refs,
        )
        self.shared_memory.post_report(
            memory_id=self.shared_memory_id,
            agent_id=task.assigned_agent_id,
            command_id=command_id,
            findings=result or "",
            summary_ids=summary_refs,
        )

    def _consolidate_results(
        self, original_query: str, sub_task_results: List[Dict[str, Any]]
    ) -> str:
        """Route consolidation through the synthesis agent with shared-memory context."""
        synthesis_input = f"Original query: {original_query}\n\nSub-task findings:\n"
        for result in sub_task_results:
            synthesis_input += (
                f"- Task {result.get('task_id')}: {result.get('result')}\n"
            )

        if self.shared_memory_id:
            self.shared_memory.add_blackboard_entry(
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={
                    "message_type": SharedMemoryMessageType.REPORT.value,
                    "payload": {
                        "command_id": "synthesis",
                        "agent_id": self.synthesis_agent.agent_id,
                        "notes": "Synthesis stage initiated",
                    },
                },
                entry_type=SharedMemoryMessageType.REPORT.value,
            )

        return self.synthesis_agent.run(
            synthesis_input, memory_id=self.shared_memory_id
        )


class DeepResearchWorkflow:
    """Thin wrapper that builds a full deep-research stack from simple instructions."""

    DEFAULT_DELEGATE_INSTRUCTIONS = [
        "Research specialist: gather relevant facts, figures, and citations.",
        "Analyst: interpret the findings and highlight implications or trends.",
    ]
    DEFAULT_ROOT_INSTRUCTION = "Root coordinator: decompose tasks, direct delegates, and enforce shared-memory protocol."
    DEFAULT_SYNTHESIS_INSTRUCTION = (
        "Synthesis expert: merge delegate reports into a coherent, cited response."
    )

    def __init__(self, orchestrator: DeepResearchOrchestrator):
        self._orchestrator = orchestrator

    @classmethod
    def from_config(
        cls,
        memory_provider,
        delegate_instructions: Optional[List[str]] = None,
        root_instruction: Optional[str] = None,
        synthesis_instruction: Optional[str] = None,
        internet_provider=None,
    ) -> "DeepResearchWorkflow":
        """Instantiate a coordinated Deep Research workflow with sane defaults."""

        delegate_instructions = (
            delegate_instructions or cls.DEFAULT_DELEGATE_INSTRUCTIONS
        )
        ip = internet_provider or get_default_internet_access_provider()

        def build_agent(instruction: str):
            return (
                create_deep_research_agent(instruction, internet_provider=ip)
                .with_memory_provider(memory_provider)
                .build()
            )

        root_agent = build_agent(root_instruction or cls.DEFAULT_ROOT_INSTRUCTION)
        delegates = [build_agent(text) for text in delegate_instructions]
        synthesis_agent = build_agent(
            synthesis_instruction or cls.DEFAULT_SYNTHESIS_INSTRUCTION
        )

        orchestrator = DeepResearchOrchestrator(
            root_agent=root_agent,
            delegates=delegates,
            synthesis_agent=synthesis_agent,
        )
        return cls(orchestrator)

    def run(self, query: str, **kwargs) -> str:
        """Execute the deep research workflow."""
        return self._orchestrator.execute(query, **kwargs)
