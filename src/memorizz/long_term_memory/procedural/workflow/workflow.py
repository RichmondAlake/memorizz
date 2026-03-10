# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider


# Workflow outcome enum
class WorkflowOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class Workflow:
    def __init__(
        self,
        name: str,
        description: str = "",
        steps: Dict[str, Any] = None,
        workflow_id: str = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        memory_id: str = None,
        agent_id: str = None,
        outcome: WorkflowOutcome = None,
        user_query: str = None,
    ):
        """
        Initialize a new Workflow instance.

        Parameters:
        -----------
        name : str
            The name of the workflow.
        description : str
            A description of what the workflow does.
        steps : Dict[str, Any]
            The steps that make up the workflow.
        workflow_id : str
            The unique identifier for the workflow.
        created_at : datetime
            When the workflow was created.
        updated_at : datetime
            When the workflow was last updated.
        memory_id : str
            The memory ID associated with this workflow.
        agent_id : str
            The agent ID that created/owns this workflow.
        outcome : WorkflowOutcome
            The outcome of the workflow (SUCCESS/FAILURE).
        user_query : str
            The original user query that triggered this workflow.
        """
        self.name = name
        self.description = description
        self.steps = steps or {}
        # Use MongoDB ObjectId for better performance
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.memory_id = memory_id or str(uuid.uuid4())
        self.agent_id = agent_id
        self.outcome = outcome or WorkflowOutcome.SUCCESS
        self.user_query = user_query

        # Generate the embedding based on the workflow's attributes
        self.embedding = self._generate_embedding()

    def _generate_embedding(self):
        """
        Generate an embedding vector for the workflow based on its attributes.

        Returns:
        --------
        list or numpy.array: The embedding vector representing the workflow.
        """
        # Convert steps to string representation
        steps_str = str(self.steps)

        embedding_input = f"{self.name} {self.description} {steps_str} {self.outcome.value} {self.user_query or ''}"
        return get_embedding(embedding_input)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the workflow to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            The workflow as a dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "workflow_id": self.workflow_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "outcome": self.outcome.value,
            "embedding": self.embedding,
            "user_query": self.user_query,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workflow":
        """
        Create a Workflow instance from a dictionary.

        Parameters:
        -----------
        data : Dict[str, Any]
            The dictionary containing workflow data.

        Returns:
        --------
        Workflow
            A new Workflow instance.
        """
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            steps=data.get("steps", {}),
            workflow_id=data.get("workflow_id"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
            memory_id=data.get("memory_id"),
            agent_id=data.get("agent_id"),
            outcome=WorkflowOutcome(data.get("outcome", WorkflowOutcome.SUCCESS.value)),
            user_query=data.get("user_query"),
        )

    def store_workflow(self, provider: MemoryProvider) -> str:
        """
        Store the workflow in the memory provider.

        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for storage.

        Returns:
        --------
        str
            The ID of the stored workflow.
        """
        workflow_data = self.to_dict()
        return provider.store(
            workflow_data, memory_store_type=MemoryType.WORKFLOW_MEMORY
        )

    @staticmethod
    def retrieve_workflow_by_id(
        workflow_id: str, provider: MemoryProvider
    ) -> Optional["Workflow"]:
        """
        Retrieve a workflow from the memory provider by workflow_id.

        Parameters:
        -----------
        workflow_id : str
            The unique identifier of the workflow to retrieve.
        provider : MemoryProvider
            The memory provider to use for retrieval.

        Returns:
        --------
        Optional[Workflow]
            The retrieved workflow, or None if not found.
        """
        workflow_data = provider.retrieve_by_id(
            workflow_id, memory_store_type=MemoryType.WORKFLOW_MEMORY
        )
        if workflow_data:
            return Workflow.from_dict(workflow_data)
        return None

    @staticmethod
    def retrieve_workflows_by_query(
        query: str, provider: MemoryProvider, limit: int = 5
    ) -> List["Workflow"]:
        """
        Retrieve workflows from the memory provider by query.

        Parameters:
        -----------
        query : str
            The query string to search for workflows.
        provider : MemoryProvider
            The memory provider to use for retrieval.
        limit : int
            Maximum number of workflows to retrieve.

        Returns:
        --------
        List[Workflow]
            A list of retrieved workflows, empty list if none found.
        """
        workflow_data = provider.retrieve_by_query(
            query, memory_store_type=MemoryType.WORKFLOW_MEMORY, limit=limit
        )
        if workflow_data is None:
            return []
        return [Workflow.from_dict(workflow) for workflow in workflow_data]

    @staticmethod
    def delete_workflow(workflow_id: str, provider: MemoryProvider) -> bool:
        """
        Delete a workflow from the memory provider using its workflow_id.

        Parameters:
        -----------
        workflow_id : str
            The unique identifier of the workflow to delete.
        provider : MemoryProvider
            The memory provider to use for deletion.

        Returns:
        --------
        bool
            True if deletion was successful, False otherwise.
        """
        return provider.delete_by_id(
            workflow_id, memory_store_type=MemoryType.WORKFLOW_MEMORY
        )

    def update_workflow(self, provider: MemoryProvider) -> bool:
        """
        Update the workflow in the memory provider.

        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for update.

        Returns:
        --------
        bool
            True if update was successful, False otherwise.
        """
        self.updated_at = datetime.now()
        workflow_data = self.to_dict()
        return provider.update_by_id(
            self.workflow_id,
            workflow_data,
            memory_store_type=MemoryType.WORKFLOW_MEMORY,
        )

    def add_step(self, step_name: str, step_data: Dict[str, Any]) -> None:
        """
        Add a new step to the workflow.

        Parameters:
        -----------
        step_name : str
            The name of the step to add.
        step_data : Dict[str, Any]
            The data for the step.
        """
        self.steps[step_name] = step_data
        self.updated_at = datetime.now()

    def remove_step(self, step_name: str) -> None:
        """
        Remove a step from the workflow.

        Parameters:
        -----------
        step_name : str
            The name of the step to remove.
        """
        if step_name in self.steps:
            del self.steps[step_name]
            self.updated_at = datetime.now()

    def get_step(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific step from the workflow.

        Parameters:
        -----------
        step_name : str
            The name of the step to get.

        Returns:
        --------
        Optional[Dict[str, Any]]
            The step data if found, None otherwise.
        """
        return self.steps.get(step_name)
