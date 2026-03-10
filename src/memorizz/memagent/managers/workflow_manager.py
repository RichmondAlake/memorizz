# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Workflow management functionality for MemAgent."""

import logging
from typing import Any, Dict, List, Optional

from ...long_term_memory.procedural.workflow.workflow import Workflow, WorkflowOutcome

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manages workflow execution and orchestration for MemAgent.

    This class encapsulates workflow-related functionality that was
    previously embedded in the main MemAgent class.
    """

    def __init__(self):
        """Initialize the workflow manager."""
        self.active_workflows = {}
        self.workflow_history = []
        self._workflow_cache = {}

    def execute_workflow(
        self, workflow: Workflow, context: Dict[str, Any]
    ) -> WorkflowOutcome:
        """
        Execute a workflow.

        Args:
            workflow: The Workflow instance to execute.
            context: Context dictionary for the workflow.

        Returns:
            WorkflowOutcome containing the result.
        """
        try:
            logger.info(
                f"Executing workflow: {workflow.name if hasattr(workflow, 'name') else 'unnamed'}"
            )

            # Track active workflow
            workflow_id = self._generate_workflow_id()
            self.active_workflows[workflow_id] = {
                "workflow": workflow,
                "context": context,
                "status": "running",
            }

            # Execute the workflow
            outcome = workflow.execute(context)

            # Update tracking
            self.active_workflows[workflow_id]["status"] = "completed"
            self.active_workflows[workflow_id]["outcome"] = outcome

            # Add to history
            self._add_to_history(workflow_id, workflow, context, outcome)

            # Clean up active workflow
            del self.active_workflows[workflow_id]

            logger.info(
                f"Workflow completed with status: {outcome.status if hasattr(outcome, 'status') else 'unknown'}"
            )
            return outcome

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowOutcome(result=f"Error: {str(e)}", status="failed")

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """
        Get currently active workflows.

        Returns:
            Dictionary of active workflows.
        """
        return self.active_workflows.copy()

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel an active workflow.

        Args:
            workflow_id: ID of the workflow to cancel.

        Returns:
            True if cancelled, False otherwise.
        """
        try:
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "cancelled"
                del self.active_workflows[workflow_id]
                logger.info(f"Cancelled workflow: {workflow_id}")
                return True
            else:
                logger.warning(f"Workflow not found for cancellation: {workflow_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False

    def get_workflow_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get workflow execution history.

        Args:
            limit: Maximum number of history entries to return.

        Returns:
            List of workflow history entries.
        """
        if limit:
            return self.workflow_history[-limit:]
        return self.workflow_history.copy()

    def clear_history(self):
        """Clear the workflow execution history."""
        self.workflow_history.clear()
        logger.debug("Cleared workflow history")

    def _generate_workflow_id(self) -> str:
        """Generate a unique workflow ID."""
        import uuid

        return f"workflow_{uuid.uuid4().hex[:8]}"

    def _add_to_history(
        self,
        workflow_id: str,
        workflow: Workflow,
        context: Dict[str, Any],
        outcome: WorkflowOutcome,
    ):
        """Add a workflow execution to history."""
        from datetime import datetime

        history_entry = {
            "id": workflow_id,
            "workflow_name": getattr(workflow, "name", "unnamed"),
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "outcome": {
                "result": outcome.result if hasattr(outcome, "result") else None,
                "status": outcome.status if hasattr(outcome, "status") else "unknown",
            },
        }

        self.workflow_history.append(history_entry)

        # Limit history size
        max_history = 100
        if len(self.workflow_history) > max_history:
            self.workflow_history = self.workflow_history[-max_history:]
