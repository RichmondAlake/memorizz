# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .memagent import MemAgent

logger = logging.getLogger(__name__)


class SubTask:
    """Represents a decomposed sub-task for delegation."""

    def __init__(
        self,
        task_id: str,
        description: str,
        assigned_agent_id: str,
        priority: int = 1,
        dependencies: List[str] = None,
    ):
        self.task_id = task_id
        self.description = description
        self.assigned_agent_id = assigned_agent_id
        self.priority = priority
        self.dependencies = dependencies or []
        self.status = "pending"  # pending, in_progress, completed, failed, skipped
        self.result = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_agent_id": self.assigned_agent_id,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SubTask":
        """Restore a sub-task from its JSON-safe persisted representation."""
        if not isinstance(value, dict):
            raise TypeError("SubTask.from_dict() requires a dictionary")
        task_id = str(value.get("task_id") or "").strip()
        description = str(value.get("description") or "").strip()
        assigned_agent_id = str(value.get("assigned_agent_id") or "").strip()
        if not task_id:
            raise ValueError("A persisted SubTask requires task_id")
        if not description:
            raise ValueError("A persisted SubTask requires description")
        if not assigned_agent_id:
            raise ValueError("A persisted SubTask requires assigned_agent_id")
        dependencies = value.get("dependencies") or []
        if not isinstance(dependencies, list):
            raise TypeError("SubTask dependencies must be a list")
        task = cls(
            task_id=task_id,
            description=description,
            assigned_agent_id=assigned_agent_id,
            priority=int(value.get("priority", 1)),
            dependencies=[str(item) for item in dependencies],
        )
        status = str(value.get("status") or "pending").strip().lower()
        if status not in {
            "pending",
            "in_progress",
            "completed",
            "failed",
            "skipped",
            "blocked",
        }:
            raise ValueError(f"Unsupported SubTask status '{status}'")
        task.status = status
        task.result = value.get("result")
        return task


def normalize_delegation_plan(plan: Any) -> Any:
    """Normalize deterministic plans to JSON-safe dictionaries.

    Callable plans remain runtime-only. A list/tuple is copied and every
    :class:`SubTask` entry is serialized so agent persistence never receives a
    Python object that Oracle (or another JSON-backed provider) cannot encode.
    """
    if plan is None or callable(plan):
        return plan
    if not isinstance(plan, (list, tuple)):
        raise TypeError("A deterministic delegation plan must be a list")
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(plan):
        if isinstance(item, SubTask):
            task = item
        elif isinstance(item, dict):
            value = dict(item)
            value.setdefault("task_id", f"task_{index + 1}")
            task = SubTask.from_dict(value)
        else:
            raise TypeError("Delegation plan entries must be SubTask or dict")
        normalized.append(task.to_dict())
    return normalized


def normalize_delegation_config(
    config: Any, *, for_persistence: bool = False
) -> Dict[str, Any]:
    """Return a copied, JSON-safe delegation configuration."""
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError("Delegation configuration must be a dictionary")
    normalized = dict(config)
    if "plan" in normalized:
        plan = normalized["plan"]
        if callable(plan) and for_persistence:
            raise TypeError(
                "Callable delegation plans are runtime-only; provide a list of "
                "SubTask/dict entries before persisting the agent"
            )
        normalized["plan"] = normalize_delegation_plan(plan)
    strategy = str(normalized.get("consolidation_strategy") or "model").lower()
    if strategy not in {"model", "deterministic", "primary", "structured"}:
        raise ValueError(
            "consolidation_strategy must be model, deterministic, primary, or "
            "structured"
        )
    if "consolidation_strategy" in normalized:
        normalized["consolidation_strategy"] = strategy
    if strategy == "primary" and normalized.get("primary_task_id") is not None:
        primary_task_id = str(normalized["primary_task_id"]).strip()
        if not primary_task_id:
            raise ValueError("primary_task_id cannot be empty")
        normalized["primary_task_id"] = primary_task_id
    for key in (
        "max_dependency_context_chars",
        "max_consolidation_result_chars",
    ):
        if key in normalized:
            normalized[key] = max(1_000, min(int(normalized[key]), 100_000))
    if "required_finding_ids" in normalized:
        values = normalized.get("required_finding_ids") or []
        if not isinstance(values, (list, tuple, set)):
            raise TypeError("required_finding_ids must be a list")
        normalized["required_finding_ids"] = list(
            dict.fromkeys(str(item).strip() for item in values if str(item).strip())
        )
    if "adaptive_escalation" in normalized:
        value = normalized.get("adaptive_escalation")
        if value in (None, False):
            normalized["adaptive_escalation"] = {"enabled": False}
        elif value is True:
            normalized["adaptive_escalation"] = {"enabled": True}
        elif isinstance(value, dict):
            adaptive = dict(value)
            adaptive["enabled"] = bool(adaptive.get("enabled", True))
            task_ids = adaptive.get("escalation_task_ids") or []
            if not isinstance(task_ids, (list, tuple, set)):
                raise TypeError("adaptive escalation_task_ids must be a list")
            adaptive["escalation_task_ids"] = list(
                dict.fromkeys(
                    str(item).strip() for item in task_ids if str(item).strip()
                )
            )
            descriptions = adaptive.get("criterion_descriptions") or {}
            if not isinstance(descriptions, dict):
                raise TypeError("adaptive criterion_descriptions must be a dictionary")
            adaptive["criterion_descriptions"] = {
                str(key).strip(): str(description).strip()
                for key, description in descriptions.items()
                if str(key).strip() and str(description).strip()
            }
            normalized["adaptive_escalation"] = adaptive
        else:
            raise TypeError("adaptive_escalation must be bool, dict, or None")
    return normalized


class TaskDecomposer:
    """Handles task decomposition for multi-agent coordination."""

    def __init__(self, root_agent: "MemAgent"):
        self.root_agent = root_agent

    def analyze_delegate_capabilities(
        self, delegates: List["MemAgent"]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze the capabilities of delegate agents.

        Parameters:
            delegates (List[MemAgent]): List of delegate agents

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of agent_id to capabilities
        """
        capabilities = {}

        for agent in delegates:
            persona_context = None
            if hasattr(agent, "persona_manager") and agent.persona_manager:
                persona_obj = getattr(agent.persona_manager, "current_persona", None)
                if persona_obj and hasattr(persona_obj, "generate_system_prompt_input"):
                    persona_context = persona_obj.generate_system_prompt_input()
                elif persona_obj:
                    persona_context = str(persona_obj)
            else:
                persona_obj = getattr(agent, "persona", None)
                if persona_obj and hasattr(persona_obj, "generate_system_prompt_input"):
                    persona_context = persona_obj.generate_system_prompt_input()
                elif persona_obj:
                    persona_context = str(persona_obj)

            agent_capabilities = {
                "agent_id": agent.agent_id,
                "instruction": agent.instruction,
                "persona": persona_context,
                "tools": [],
                "application_mode": agent.application_mode.value,
            }

            # Extract tool capabilities
            manager = getattr(agent, "tool_manager", None)
            if manager is not None:
                for tool in manager.get_tool_metadata() or []:
                    if callable(tool):
                        agent_capabilities["tools"].append(
                            {
                                "name": getattr(tool, "__name__", "callable"),
                                "description": inspect.getdoc(tool) or "",
                            }
                        )
                        continue
                    if hasattr(tool, "model_dump"):
                        tool = tool.model_dump()
                    if not isinstance(tool, dict):
                        logger.debug(
                            "Ignoring unsupported tool metadata for agent %s: %s",
                            agent.agent_id,
                            type(tool).__name__,
                        )
                        continue
                    function = tool.get("function")
                    agent_capabilities["tools"].append(
                        {
                            "name": (
                                function.get("name")
                                if isinstance(function, dict)
                                else tool.get("name")
                            ),
                            "description": (
                                function.get("description")
                                if isinstance(function, dict)
                                else tool.get("description")
                            ),
                        }
                    )
            elif isinstance(agent.tools, list):
                for tool in agent.tools:
                    if callable(tool):
                        agent_capabilities["tools"].append(
                            {
                                "name": getattr(tool, "__name__", "callable"),
                                "description": inspect.getdoc(tool) or "",
                            }
                        )
                    elif isinstance(tool, dict):
                        function = tool.get("function")
                        agent_capabilities["tools"].append(
                            {
                                "name": (
                                    function.get("name")
                                    if isinstance(function, dict)
                                    else tool.get("name")
                                ),
                                "description": (
                                    function.get("description")
                                    if isinstance(function, dict)
                                    else tool.get("description")
                                ),
                            }
                        )

            capabilities[agent.agent_id] = agent_capabilities

        return capabilities

    def decompose_task(
        self,
        user_query: str,
        delegates: List["MemAgent"],
        plan: Any = None,
    ) -> List[SubTask]:
        """
        Decompose a complex task into sub-tasks aligned with delegate capabilities.

        Parameters:
            user_query (str): The original user query/task
            delegates (List[MemAgent]): List of available delegate agents

        Returns:
            List[SubTask]: List of decomposed sub-tasks
        """
        try:
            if callable(plan):
                plan = plan(user_query, delegates)
            if plan is not None:
                capabilities = self.analyze_delegate_capabilities(delegates)
                return self._coerce_plan(plan, capabilities)
            logger.info(f"Starting task decomposition for query: {user_query}")
            logger.info(f"Number of delegates: {len(delegates)}")

            # Analyze delegate capabilities
            logger.info("Analyzing delegate capabilities...")
            capabilities = self.analyze_delegate_capabilities(delegates)
            logger.info(f"Analyzed capabilities for {len(capabilities)} agents")

            # Create decomposition prompt
            logger.info("Creating decomposition prompt...")
            decomposition_prompt = self._create_decomposition_prompt(
                user_query, capabilities
            )

            # Use the configured LLMProvider/model — never a vendor client or
            # hard-coded deployment name.
            messages = [
                {"role": "system", "content": decomposition_prompt},
                {
                    "role": "user",
                    "content": f"Please decompose this task: {user_query}",
                },
            ]

            logger.info("Calling LLM for task decomposition...")
            if not self.root_agent.model:
                raise ValueError("Root agent has no configured LLMProvider")
            response = self.root_agent.model.generate(messages, tools=None)
            response_text = self._response_text(response)

            logger.info(
                f"LLM response received: {response_text[:200]}..."
                if response_text
                else "No response text"
            )

            # Parse the response to extract sub-tasks
            logger.info("Parsing decomposition response...")
            sub_tasks = self._parse_decomposition_response(response_text, capabilities)
            logger.info(f"Successfully parsed {len(sub_tasks)} sub-tasks")

            return sub_tasks

        except Exception as e:
            logger.error(f"Error decomposing task: {e}", exc_info=True)
            return []

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            return str(getattr(message, "content", "") or "")
        return str(response or "")

    def _coerce_plan(
        self, plan: Any, capabilities: Dict[str, Dict[str, Any]]
    ) -> List[SubTask]:
        if not isinstance(plan, list):
            raise TypeError("A deterministic delegation plan must be a list")
        values: List[SubTask] = []
        for index, item in enumerate(plan):
            if isinstance(item, SubTask):
                task = item
            elif isinstance(item, dict):
                item_value = dict(item)
                item_value.setdefault("task_id", f"task_{index + 1}")
                task = SubTask.from_dict(item_value)
            else:
                raise TypeError("Delegation plan entries must be SubTask or dict")
            if task.assigned_agent_id not in capabilities:
                raise ValueError(
                    f"Unknown delegate '{task.assigned_agent_id}' in delegation plan"
                )
            if not task.description.strip():
                raise ValueError("Delegation task descriptions cannot be empty")
            values.append(task)
        return values

    def _create_decomposition_prompt(
        self, user_query: str, capabilities: Dict[str, Dict[str, Any]]
    ) -> str:
        """Create the system prompt for task decomposition."""

        prompt = """You are a task decomposition specialist for a multi-agent system. Your job is to analyze a complex user query and break it down into specific sub-tasks that can be assigned to different specialized agents.

AVAILABLE AGENTS AND THEIR CAPABILITIES:
"""

        for agent_id, caps in capabilities.items():
            prompt += f"\nAgent ID: {agent_id}\n"
            prompt += (
                f"Specialization: {caps.get('instruction', 'General assistant')}\n"
            )
            if caps.get("persona"):
                prompt += f"Persona: {caps['persona'][:200]}...\n"

            if caps.get("tools"):
                prompt += "Available Tools:\n"
                for tool in caps["tools"]:
                    prompt += f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}\n"
            prompt += "\n"

        prompt += """
DECOMPOSITION RULES:
1. Break down the user query into specific, actionable sub-tasks
2. Assign each sub-task to the most appropriate agent based on their capabilities
3. Consider task dependencies and execution order
4. Each sub-task should be independent or have clear dependencies
5. Prioritize tasks (1=highest, 5=lowest)

RESPONSE FORMAT:
Respond with a JSON array of sub-tasks in this exact format:
[
    {
        "task_id": "task_1",
        "description": "Specific task description",
        "assigned_agent_id": "agent_id",
        "priority": 1,
        "dependencies": []
    }
]

Only respond with the JSON array, no additional text.
"""

        return prompt

    def _parse_decomposition_response(
        self, response_text: str, capabilities: Dict[str, Dict[str, Any]]
    ) -> List[SubTask]:
        """Parse the decomposition response and create SubTask objects."""
        try:
            # Extract JSON from response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            task_data = json.loads(response_text)
            if isinstance(task_data, dict):
                task_data = task_data.get("sub_tasks") or task_data.get("tasks")
            if not isinstance(task_data, list):
                raise ValueError("Decomposition response must be a JSON task array")

            sub_tasks = []
            for task in task_data:
                if not isinstance(task, dict):
                    logger.warning("Ignoring non-object delegation task: %r", task)
                    continue
                # Validate that assigned agent exists
                if task.get("assigned_agent_id") in capabilities:
                    sub_task = SubTask(
                        task_id=task.get("task_id"),
                        description=task.get("description"),
                        assigned_agent_id=task.get("assigned_agent_id"),
                        priority=task.get("priority", 1),
                        dependencies=task.get("dependencies", []),
                    )
                    sub_tasks.append(sub_task)
                else:
                    logger.warning(f"Invalid agent assignment in task: {task}")

            return sub_tasks

        except Exception as e:
            logger.error(f"Error parsing decomposition response: {e}")
            return []

    def create_consolidation_prompt(
        self,
        original_query: str,
        sub_task_results: List[Dict[str, Any]],
        *,
        memory_context: str = "",
        max_result_chars: int = 12_000,
    ) -> str:
        """Create a bounded, evidence-preserving consolidation prompt."""

        result_limit = max(1_000, min(int(max_result_chars), 100_000))

        prompt = f"""ORIGINAL USER QUERY:
{original_query}

AUTHORITATIVE RETRIEVED MEMORY:
{memory_context or "No retrieved memory was available to the coordinator."}

SUB-TASK RESULTS:
"""

        for result in sub_task_results:
            result_text = str(result.get("result", "No result"))
            if len(result_text) > result_limit:
                result_text = result_text[:result_limit] + "\u2026"
            prompt += f"\nTask: {result.get('description', 'Unknown task')}\n"
            prompt += f"Agent: {result.get('assigned_agent_id', 'Unknown agent')}\n"
            prompt += f"Status: {result.get('status', 'Unknown status')}\n"
            prompt += f"Result: {result_text}\n"
            prompt += "---\n"

        prompt += """
CONSOLIDATION INSTRUCTIONS:
1. Return one concise, standalone answer to the original query.
2. Preserve every distinct, material finding that is supported by concrete evidence.
3. Accept a factual claim only when it is supported by a sub-task result and is
   consistent with the authoritative retrieved memory. Do not invent new claims.
4. Preserve source identifiers, file/line evidence, verification facts, and severity.
5. Deduplicate overlapping findings without weakening their evidence.
6. If results conflict, state the conflict instead of guessing.
7. Omit tangents that are outside the original query or retrieved requirements.
8. If the evidence is incomplete, state the limitation; do not fill the gap from
   general knowledge.
"""

        return prompt
