# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import concurrent.futures
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .coordination.shared_memory.shared_memory import SharedMemory
from .task_decomposition import SubTask, TaskDecomposer

if TYPE_CHECKING:
    from .memagent import MemAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent workflows with hierarchical coordination support.

    This orchestrator supports both flat and hierarchical multi-agent scenarios:
    - Flat: Root agent coordinates directly with delegate agents
    - Hierarchical: Delegate agents can have their own sub-agents, all coordinated
      within a single shared memory session for complete visibility and control
    """

    def __init__(
        self,
        root_agent: "MemAgent",
        delegates: List["MemAgent"],
        *,
        delegation_plan: Any = None,
        persist_participants: bool = False,
        workflow_id: Optional[str] = None,
    ):
        self.root_agent = root_agent
        self.delegates = delegates
        self.shared_memory = SharedMemory(root_agent.memory_provider)
        self.task_decomposer = TaskDecomposer(root_agent)
        self.shared_memory_id = None
        self.delegation_plan = delegation_plan
        self.persist_participants = bool(persist_participants)
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.last_workflow_report: Optional[Dict[str, Any]] = None
        self._request_user_id: Optional[str] = None
        self._request_context: Optional[Dict[str, Any]] = None
        self._tool_context: Optional[Dict[str, Any]] = None
        self._trace_id: Optional[str] = None
        self.is_nested_orchestrator = (
            False  # Flag to track if this is a sub-level orchestrator
        )

    def execute_multi_agent_workflow(
        self,
        user_query: str,
        memory_id: str = None,
        thread_id: str = None,
        *,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tool_context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        delegation_plan: Any = None,
        return_report: bool = False,
    ) -> Any:
        """
        Execute a multi-agent workflow with hierarchical coordination support.

        The workflow intelligently handles nested agent scenarios:
        1. First checks if the root agent is already part of an active shared session
        2. If yes, joins that session as a sub-agent rather than creating a new one
        3. If no, creates a new root-level shared session
        4. Ensures all agent activities are tracked in a unified shared memory

        Parameters:
            user_query (str): The task to be executed
            memory_id (str): Memory context for the execution
            thread_id (str): Thread context

        Returns:
            str: The consolidated response from all agents
        """

        logger.info(f"Starting multi-agent workflow for query: {user_query}")
        logger.info(f"Root agent ID: {self.root_agent.agent_id}")
        logger.info(f"Number of delegates: {len(self.delegates)}")
        logger.info(f"Delegate IDs: {[agent.agent_id for agent in self.delegates]}")
        self._request_user_id = user_id
        self._request_context = dict(context or {})
        self._tool_context = dict(tool_context or {})
        self._trace_id = trace_id or str(uuid.uuid4())
        if self.persist_participants:
            self._prepare_participants()

        try:
            # HIERARCHICAL COORDINATION: Check if we should join an existing session
            existing_session = self._find_or_create_shared_session()

            if existing_session:
                self.shared_memory_id = str(existing_session.get("_id"))
                logger.info(f"Joining existing shared session: {self.shared_memory_id}")
                self.is_nested_orchestrator = True

                # Register our delegates as sub-agents in the existing session
                delegate_ids = [agent.agent_id for agent in self.delegates]
                if delegate_ids:
                    self.shared_memory.register_sub_agents(
                        memory_id=self.shared_memory_id,
                        parent_agent_id=self.root_agent.agent_id,
                        sub_agent_ids=delegate_ids,
                    )

            # **FIX: Add shared memory ID to root agent's memory_ids array**
            # This ensures consistency between single-agent and multi-agent memory management
            if (
                self.shared_memory_id
                and self.root_agent.agent_id
                and self.shared_memory_id not in (self.root_agent.memory_ids or [])
            ):
                # Initialize memory_ids if it's None
                if self.root_agent.memory_ids is None:
                    self.root_agent.memory_ids = []

                # Add shared memory ID to agent's memory_ids array
                self.root_agent.memory_ids.append(self.shared_memory_id)
                logger.info(
                    f"Added shared memory ID {self.shared_memory_id} to root agent's memory_ids array"
                )

                # Persist the updated memory_ids to the memory provider
                if hasattr(
                    self.root_agent.memory_provider, "update_memagent_memory_ids"
                ) and self._participant_is_persisted(self.root_agent):
                    update_success = (
                        self.root_agent.memory_provider.update_memagent_memory_ids(
                            self.root_agent.agent_id, self.root_agent.memory_ids
                        )
                    )
                    logger.info(
                        f"Persisted memory_ids to storage: {'success' if update_success else 'failed'}"
                    )
                else:
                    logger.debug(
                        "Root participant is not persisted; skipping memory-id update"
                    )

            # **FIX: Also add shared memory ID to delegate agents' memory_ids arrays**
            # This ensures all participating agents have access to the shared memory
            for delegate in self.delegates:
                if (
                    self.shared_memory_id
                    and delegate.agent_id
                    and self.shared_memory_id not in (delegate.memory_ids or [])
                ):
                    # Initialize memory_ids if it's None
                    if delegate.memory_ids is None:
                        delegate.memory_ids = []

                    # Add shared memory ID to delegate's memory_ids array
                    delegate.memory_ids.append(self.shared_memory_id)
                    logger.info(
                        f"Added shared memory ID {self.shared_memory_id} to delegate agent {delegate.agent_id}'s memory_ids array"
                    )

                    # Persist the updated memory_ids to the memory provider
                    if hasattr(
                        delegate.memory_provider, "update_memagent_memory_ids"
                    ) and self._participant_is_persisted(delegate):
                        update_success = (
                            delegate.memory_provider.update_memagent_memory_ids(
                                delegate.agent_id, delegate.memory_ids
                            )
                        )
                        logger.info(
                            f"Persisted delegate {delegate.agent_id} memory_ids to storage: {'success' if update_success else 'failed'}"
                        )
                    else:
                        logger.debug(
                            "Delegate %s is not persisted; skipping memory-id update",
                            delegate.agent_id,
                        )

            # Log the start of multi-agent execution with hierarchy context
            self.shared_memory.add_blackboard_entry(
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={
                    "original_query": user_query,
                    "started_at": datetime.now().isoformat(),
                    "orchestrator_type": (
                        "nested" if self.is_nested_orchestrator else "root"
                    ),
                    "delegate_count": len(self.delegates),
                },
                entry_type="workflow_start",
            )

            # 2. Decompose task into sub-tasks
            logger.info("Starting task decomposition...")
            sub_tasks = self._enhance_task_decomposition_with_hierarchy(
                user_query,
                plan=(
                    delegation_plan
                    if delegation_plan is not None
                    else self.delegation_plan
                ),
            )
            logger.info(f"Task decomposition resulted in {len(sub_tasks)} sub-tasks")

            if not sub_tasks:
                # Fallback to single agent execution
                logger.warning("Task decomposition failed, falling back to root agent")
                logger.info("Executing fallback with root agent...")
                fallback_tool_context = dict(tool_context or {})
                fallback_tool_context["_memorizz_skip_delegation"] = True
                result = self.root_agent.run(
                    user_query,
                    memory_id,
                    thread_id,
                    user_id=user_id,
                    context=context,
                    tool_context=fallback_tool_context,
                )
                logger.info(
                    f"Root agent returned: {result[:100]}..."
                    if result
                    else "No result from root agent"
                )
                self.last_workflow_report = {
                    "ok": True,
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "fallback": True,
                    "response": result,
                    "tasks": [],
                }
                return self.last_workflow_report if return_report else result

            # Log task decomposition
            self.shared_memory.add_blackboard_entry(
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={"sub_tasks": [task.to_dict() for task in sub_tasks]},
                entry_type="task_decomposition",
            )
            self._after_task_decomposition(sub_tasks, user_query)

            # 3. Execute sub-tasks in parallel
            logger.info(f"Executing {len(sub_tasks)} sub-tasks in parallel...")
            sub_task_results = self._execute_sub_tasks_parallel(
                sub_tasks, memory_id, thread_id
            )
            logger.info(
                f"Sub-task execution completed with {len(sub_task_results)} results"
            )

            # 4. Consolidate results
            logger.info("Starting result consolidation...")
            consolidated_response = self._consolidate_results(
                user_query, sub_task_results
            )
            logger.info(
                f"Consolidation completed: {consolidated_response[:100]}..."
                if consolidated_response
                else "No consolidated response"
            )

            # 5. Update shared memory with final result
            self.shared_memory.add_blackboard_entry(
                memory_id=self.shared_memory_id,
                agent_id=self.root_agent.agent_id,
                content={
                    "consolidated_response": consolidated_response,
                    "completed_at": datetime.now().isoformat(),
                },
                entry_type="workflow_complete",
            )

            # Mark session as completed
            self.shared_memory.update_session_status(self.shared_memory_id, "completed")

            logger.info("Multi-agent workflow completed successfully")
            failed = [
                item
                for item in sub_task_results
                if item.get("status") not in {"completed"}
            ]
            self.last_workflow_report = {
                "ok": not failed,
                "partial": bool(failed),
                "workflow_id": self.workflow_id,
                "shared_memory_id": self.shared_memory_id,
                "trace_id": self._trace_id,
                "user_id": user_id,
                "response": consolidated_response,
                "tasks": sub_task_results,
                "failures": failed,
            }
            return self.last_workflow_report if return_report else consolidated_response

        except Exception as e:
            logger.error(f"Error in multi-agent workflow: {e}", exc_info=True)
            # Mark session as failed
            if self.shared_memory_id:
                self.shared_memory.update_session_status(
                    self.shared_memory_id, "failed"
                )

            # Fallback to single agent execution
            logger.info("Attempting fallback to root agent due to error...")
            try:
                fallback_tool_context = dict(tool_context or {})
                fallback_tool_context["_memorizz_skip_delegation"] = True
                result = self.root_agent.run(
                    user_query,
                    memory_id,
                    thread_id,
                    user_id=user_id,
                    context=context,
                    tool_context=fallback_tool_context,
                )
                logger.info(
                    f"Fallback completed: {result[:100]}..."
                    if result
                    else "No result from fallback"
                )
                self.last_workflow_report = {
                    "ok": False,
                    "fallback": True,
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "error": str(e),
                    "response": result,
                }
                return self.last_workflow_report if return_report else result
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}", exc_info=True)
                return f"Multi-agent workflow failed: {str(e)}. Fallback also failed: {str(fallback_error)}"

    def _participant_is_persisted(self, agent: "MemAgent") -> bool:
        provider = getattr(agent, "memory_provider", None)
        lookup = getattr(provider, "retrieve_memagent", None)
        if not callable(lookup):
            return False
        try:
            return lookup(agent.agent_id) is not None
        except Exception:
            return False

    def _prepare_participants(self) -> None:
        """Persist previously unsaved participants when explicitly requested."""
        for agent in [self.root_agent, *self.delegates]:
            if self._participant_is_persisted(agent):
                continue
            if not getattr(agent, "memory_provider", None):
                raise ValueError(
                    f"Participant {agent.agent_id} has no memory provider to persist"
                )
            agent.save()

    def _execute_sub_tasks_parallel(
        self, sub_tasks: List[SubTask], memory_id: str, thread_id: str
    ) -> List[Dict[str, Any]]:
        """Execute a dependency graph and expose partial/blocked states."""
        agent_map = {agent.agent_id: agent for agent in self.delegates}
        sorted_tasks = sorted(sub_tasks, key=lambda item: item.priority)
        results: List[Dict[str, Any]] = []
        completed_tasks: set[str] = set()
        failed_tasks: set[str] = set()

        task_ids = [task.task_id for task in sorted_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Delegation plans must use unique task IDs")
        known_ids = set(task_ids)

        def mark_unrunnable(task: SubTask, status: str, reason: str) -> None:
            task.status = status
            task.result = {
                "error": reason,
                "dependencies": list(task.dependencies),
            }
            failed_tasks.add(task.task_id)
            results.append(task.to_dict())

        for task in sorted_tasks:
            missing = [dep for dep in task.dependencies if dep not in known_ids]
            if missing:
                mark_unrunnable(
                    task, "blocked", f"Unknown dependencies: {', '.join(missing)}"
                )
            elif task.assigned_agent_id not in agent_map:
                mark_unrunnable(
                    task,
                    "failed",
                    f"Unknown delegate: {task.assigned_agent_id}",
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.delegates))
        ) as executor:
            future_to_task: Dict[Any, SubTask] = {}
            while True:
                for task in sorted_tasks:
                    if task.status != "pending":
                        continue
                    if any(dep in failed_tasks for dep in task.dependencies):
                        mark_unrunnable(
                            task, "blocked", "One or more dependencies failed"
                        )
                        continue
                    if all(dep in completed_tasks for dep in task.dependencies):
                        task.status = "in_progress"
                        future = executor.submit(
                            self._execute_single_task,
                            task,
                            agent_map[task.assigned_agent_id],
                            memory_id,
                            thread_id,
                        )
                        future_to_task[future] = task

                if not future_to_task:
                    for task in sorted_tasks:
                        if task.status == "pending":
                            mark_unrunnable(
                                task,
                                "blocked",
                                "Dependency cycle or unsatisfied dependency",
                            )
                    break

                done, _ = concurrent.futures.wait(
                    future_to_task,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    task = future_to_task.pop(future)
                    try:
                        result = future.result()
                        task.result = result
                        task.status = "completed"
                        completed_tasks.add(task.task_id)
                        results.append(task.to_dict())
                        self.shared_memory.add_blackboard_entry(
                            memory_id=self.shared_memory_id,
                            agent_id=task.assigned_agent_id,
                            content={"task_id": task.task_id, "result": result},
                            entry_type="task_completion",
                        )
                        self._after_task_completion(task, result)
                    except Exception as exc:
                        logger.error("Error executing task %s: %s", task.task_id, exc)
                        task.status = "failed"
                        task.result = {"error": str(exc)}
                        failed_tasks.add(task.task_id)
                        results.append(task.to_dict())

        by_id = {result["task_id"]: result for result in results}
        return [by_id[task.task_id] for task in sorted_tasks if task.task_id in by_id]

    def _execute_single_task(
        self, task: SubTask, agent: "MemAgent", memory_id: str, thread_id: str
    ) -> str:
        """Execute a single sub-task with an agent."""

        try:
            # Log task start
            self.shared_memory.add_blackboard_entry(
                memory_id=self.shared_memory_id,
                agent_id=agent.agent_id,
                content={
                    "task_id": task.task_id,
                    "description": task.description,
                    "started_at": datetime.now().isoformat(),
                },
                entry_type="task_start",
            )

            request_context = dict(self._request_context or {})
            request_context["delegation"] = {
                "workflow_id": self.workflow_id,
                "trace_id": self._trace_id,
                "task_id": task.task_id,
                "dependencies": list(task.dependencies),
            }
            tool_context = dict(self._tool_context or {})
            tool_context.update(
                {
                    "workflow_id": self.workflow_id,
                    "trace_id": self._trace_id,
                    "delegated_task_id": task.task_id,
                    "delegated_by": self.root_agent.agent_id,
                }
            )
            result = agent.run(
                task.description,
                memory_id,
                thread_id,
                user_id=self._request_user_id,
                context=request_context,
                tool_context=tool_context,
            )

            return result

        except Exception as e:
            logger.error(
                f"Error executing task {task.task_id} with agent {agent.agent_id}: {e}"
            )
            raise

    def _consolidate_results(
        self, original_query: str, sub_task_results: List[Dict[str, Any]]
    ) -> str:
        """Consolidate results from all sub-tasks into a final response."""

        try:
            # Create consolidation prompt
            consolidation_prompt = self.task_decomposer.create_consolidation_prompt(
                original_query, sub_task_results
            )

            # Use root agent to consolidate
            messages = [
                {"role": "system", "content": consolidation_prompt},
                {
                    "role": "user",
                    "content": "Please provide a consolidated response based on the sub-task results.",
                },
            ]

            if not self.root_agent.model:
                raise ValueError("Root agent has no configured LLMProvider")
            response = self.root_agent.model.generate(messages, tools=None)
            return self.task_decomposer._response_text(response)

        except Exception as e:
            logger.error(f"Error consolidating results: {e}")

            # Fallback: simple concatenation
            consolidated = f"Results for: {original_query}\n\n"
            for result in sub_task_results:
                consolidated += f"Task: {result.get('description', 'Unknown')}\n"
                consolidated += f"Result: {result.get('result', 'No result')}\n\n"

            return consolidated

    def get_shared_memory_context(self) -> str:
        """Get shared memory context for inclusion in agent prompts."""

        if not self.shared_memory_id:
            return ""

        try:
            # Get blackboard entries
            entries = self.shared_memory.get_blackboard_entries(self.shared_memory_id)

            if not entries:
                return ""

            context = "\n\n---------SHARED MEMORY CONTEXT---------\n"
            context += "Multi-agent coordination information:\n\n"

            for entry in entries[-10:]:  # Last 10 entries
                context += f"Agent: {entry.get('agent_id', 'Unknown')}\n"
                context += f"Type: {entry.get('entry_type', 'Unknown')}\n"
                context += f"Content: {entry.get('content', 'No content')}\n"
                context += f"Time: {entry.get('created_at', 'Unknown')}\n"
                context += "---\n"

            return context

        except Exception as e:
            logger.error(f"Error getting shared memory context: {e}")
            return ""

    def _find_or_create_shared_session(self) -> Optional[Dict[str, Any]]:
        """
        Intelligent session management for hierarchical multi-agent coordination.

        This method implements the core logic for hierarchical coordination:
        1. Checks if the root agent is already participating in an active shared session
        2. If found, returns that session to enable joining (hierarchical mode)
        3. If not found, creates a new root-level session (flat mode)

        This ensures that sub-agents don't create isolated sessions but instead
        join the existing coordination context, enabling true hierarchical workflows.

        Returns:
            Optional[Dict[str, Any]]: Existing session to join, or None if new session created
        """
        try:
            # Check if our root agent is already part of an active shared session
            existing_session = self.shared_memory.find_active_session_for_agent(
                self.root_agent.agent_id,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
            )

            if existing_session:
                logger.info(
                    f"Found existing session for agent {self.root_agent.agent_id}"
                )
                logger.info(
                    f"Session hierarchy: {self.shared_memory.get_agent_hierarchy(str(existing_session.get('_id')))}"
                )
                return existing_session

            # No existing session found - create a new root-level session
            logger.info(
                "No existing session found, creating new root-level shared session"
            )
            delegate_ids = [agent.agent_id for agent in self.delegates]

            self.shared_memory_id = self.shared_memory.create_shared_session(
                root_agent_id=self.root_agent.agent_id,
                delegate_agent_ids=delegate_ids,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
                trace_id=self._trace_id,
            )

            logger.info(f"Created new shared memory session: {self.shared_memory_id}")
            logger.info(f"Initial delegates: {delegate_ids}")

            # Return None to indicate we created a new session (not joining existing)
            return None

        except Exception as e:
            logger.error(f"Error in session management: {e}", exc_info=True)
            # Fallback: create new session
            delegate_ids = [agent.agent_id for agent in self.delegates]
            self.shared_memory_id = self.shared_memory.create_shared_session(
                root_agent_id=self.root_agent.agent_id,
                delegate_agent_ids=delegate_ids,
                workflow_id=self.workflow_id,
                user_id=self._request_user_id,
                trace_id=self._trace_id,
            )
            return None

    def _enhance_task_decomposition_with_hierarchy(
        self, user_query: str, *, plan: Any = None
    ) -> List[SubTask]:
        """
        Enhanced task decomposition that considers the complete agent hierarchy.

        Traditional decomposition only considers immediate delegates. This enhanced
        version looks at the full shared memory session to understand the complete
        agent capabilities available across all hierarchy levels.

        Parameters:
            user_query (str): The task to decompose

        Returns:
            List[SubTask]: Decomposed tasks optimized for the full hierarchy
        """
        try:
            # Get the complete agent hierarchy from shared memory
            hierarchy = self.shared_memory.get_agent_hierarchy(self.shared_memory_id)

            logger.info(f"Task decomposition considering hierarchy: {hierarchy}")

            # Standard task decomposition with immediate delegates
            sub_tasks = self.task_decomposer.decompose_task(
                user_query, self.delegates, plan=plan
            )

            # TODO: Future enhancement - analyze sub_agent capabilities for optimal task assignment
            # This could involve:
            # 1. Collecting capabilities from all sub-agents in the hierarchy
            # 2. Re-optimizing task assignments based on the full capability set
            # 3. Creating more granular tasks that leverage specific sub-agent strengths

            logger.info(
                f"Decomposed into {len(sub_tasks)} tasks for {hierarchy.get('total_agents', 0)} total agents"
            )

            return sub_tasks

        except Exception as e:
            logger.error(f"Error in enhanced task decomposition: {e}")
            # Fallback to standard decomposition
            return self.task_decomposer.decompose_task(
                user_query, self.delegates, plan=plan
            )

    # Hook methods for specialized orchestrators ---------------------------------
    def _after_task_decomposition(
        self, sub_tasks: List[SubTask], user_query: str
    ) -> None:
        """Optional hook for subclasses after decomposition."""
        return

    def _after_task_completion(self, task: SubTask, result: Any) -> None:
        """Optional hook after individual task completion."""
        return
