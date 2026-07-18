# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from ....embeddings import get_embedding
from ....enums.memory_type import MemoryType
from ....memory_provider import MemoryProvider
from .canonicalization import canonical_hash as _canonical_hash
from .canonicalization import canonical_signature as _canonical_signature


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
        user_id: str = None,
        canonical_hash: str = None,
        canonical_signature: List[Dict[str, Any]] = None,
        step_count: int = None,
        promoted_skill_id: str = None,
        skills_activated: List[str] = None,
        embedding: List[float] = None,
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
        user_id : str, optional
            Multi-tenant scope for this run.
        canonical_hash : str, optional
            Trajectory identity — sha256 over the canonical signature.
            Computed automatically at store time when unset.
        canonical_signature : List[Dict[str, Any]], optional
            The canonical unit list behind ``canonical_hash`` (kept on the
            document for debuggability).
        step_count : int, optional
            ``len(canonical_signature)`` after retry collapse.
        promoted_skill_id : str, optional
            Stamped when this trajectory class is covered by an ACTIVE
            learned skill; drives retrieval suppression. Cleared on skill
            demotion.
        skills_activated : List[str], optional
            Skill IDs that were injected into THIS run's context, whether
            or not the run followed them — the attribution field the
            continual-learning monitor reads.
        embedding : List[float], optional
            Pre-computed embedding. When provided (e.g. loading a stored
            document via ``from_dict``) the embedding API is not called.
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
        self.user_id = user_id
        self.canonical_hash = canonical_hash
        self.canonical_signature = canonical_signature
        self.step_count = step_count
        self.promoted_skill_id = promoted_skill_id
        self.skills_activated = list(skills_activated) if skills_activated else []

        # Reuse a stored embedding when one is supplied (round-tripping via
        # from_dict) so loading never re-bills the embedding API.
        self.embedding = (
            embedding if embedding is not None else self._generate_embedding()
        )

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
        data = {
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
            "canonical_hash": self.canonical_hash,
            "canonical_signature": self.canonical_signature,
            "step_count": self.step_count,
            "promoted_skill_id": self.promoted_skill_id,
            "skills_activated": self.skills_activated,
        }
        if self.user_id is not None:
            data["user_id"] = self.user_id
        return data

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
            user_id=data.get("user_id"),
            canonical_hash=data.get("canonical_hash"),
            canonical_signature=data.get("canonical_signature"),
            step_count=data.get("step_count"),
            promoted_skill_id=data.get("promoted_skill_id"),
            skills_activated=data.get("skills_activated"),
            embedding=data.get("embedding"),
        )

    def refresh_canonical_fields(self, force: bool = False) -> None:
        """Compute canonical_signature / canonical_hash / step_count.

        No-op when steps are empty, and when the hash is already set unless
        ``force`` is passed (steps may have been appended since load).
        """
        if not self.steps:
            return
        if self.canonical_hash and not force:
            return
        self.canonical_signature = _canonical_signature(self.steps)
        self.canonical_hash = _canonical_hash(self.canonical_signature)
        self.step_count = len(self.canonical_signature)

    def store_workflow(self, provider: MemoryProvider) -> str:
        """
        Store the workflow in the memory provider.

        Canonical trajectory-identity fields are computed here — never at
        the capture call sites — so the streaming and non-streaming paths
        in MemAgent can't drift.

        Parameters:
        -----------
        provider : MemoryProvider
            The memory provider to use for storage.

        Returns:
        --------
        str
            The ID of the stored workflow.
        """
        self.refresh_canonical_fields()
        workflow_data = self.to_dict()
        return provider.store(
            workflow_data, memory_store_type=MemoryType.WORKFLOW_MEMORY
        )

    @staticmethod
    def retrieve_workflows_by_query(
        query: str,
        provider: MemoryProvider,
        limit: int = 5,
        exclude_promoted: bool = True,
    ) -> List["Workflow"]:
        """
        Retrieve workflows from the memory provider by query.

        Trajectories covered by an ACTIVE learned skill (truthy
        ``promoted_skill_id``) are excluded by default — the skill is their
        compiled form; retrieving both would duplicate context. Demotion
        clears the stamp, which makes the raw trajectories retrievable
        again. The overfetch + post-filter here is the provider-agnostic
        correctness backstop; a provider-side pre-filter is an optional
        optimization on top.

        Parameters:
        -----------
        query : str
            The query string to search for workflows.
        provider : MemoryProvider
            The memory provider to use for retrieval.
        limit : int
            Maximum number of workflows to retrieve.
        exclude_promoted : bool
            When True (default), drop rows stamped with a
            ``promoted_skill_id``.

        Returns:
        --------
        List[Workflow]
            A list of retrieved workflows, empty list if none found.
        """
        fetch_limit = limit * 2 if exclude_promoted else limit
        workflow_data = provider.retrieve_by_query(
            query, memory_store_type=MemoryType.WORKFLOW_MEMORY, limit=fetch_limit
        )
        if workflow_data is None:
            return []
        if exclude_promoted:
            workflow_data = [
                doc for doc in workflow_data if not doc.get("promoted_skill_id")
            ]
        return [Workflow.from_dict(doc) for doc in workflow_data[:limit]]

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
