"""Business-outcome evaluation for captured workflows."""

from memorizz.long_term.procedural.workflow import WorkflowOutcome
from memorizz.memagent.builders import MemAgentBuilder
from memorizz.memagent.core import MemAgent


class _CapturedWorkflow:
    def __init__(self, outcome=WorkflowOutcome.SUCCESS):
        self.steps = {"Step 1: do_work": {"result": {"ok": True}}}
        self.outcome = outcome
        self.stored_outcome = None

    def store_workflow(self, _provider):
        self.stored_outcome = self.outcome
        return "workflow-record-1"


class _LearningRecorder:
    def __init__(self):
        self.recorded_outcome = None
        self.cycles = 0

    def record_run_outcome(self, workflow, record_id=None):
        assert record_id == "workflow-record-1"
        self.recorded_outcome = workflow.outcome

    def maybe_run_scheduled_cycle(self):
        self.cycles += 1


def _agent_with_persistence(**kwargs):
    """Build an agent whose captured-workflow double accepts any provider."""
    agent = MemAgent(**kwargs)
    agent.memory_provider = object()
    return agent


def test_business_failure_is_stored_and_used_for_learning():
    agent = _agent_with_persistence(workflow_outcome_evaluator=lambda workflow: False)
    recorder = _LearningRecorder()
    agent.continual_learning_manager = recorder
    workflow = _CapturedWorkflow()

    assert agent._persist_workflow_run(workflow) is True
    assert workflow.stored_outcome == WorkflowOutcome.FAILURE
    assert recorder.recorded_outcome == WorkflowOutcome.FAILURE
    assert recorder.cycles == 1


def test_business_evaluator_cannot_upgrade_execution_failure():
    agent = _agent_with_persistence(workflow_outcome_evaluator=lambda workflow: True)
    workflow = _CapturedWorkflow(outcome=WorkflowOutcome.FAILURE)

    assert agent._persist_workflow_run(workflow) is True
    assert workflow.stored_outcome == WorkflowOutcome.FAILURE


def test_evaluator_exception_fails_closed_without_dropping_workflow():
    def broken_evaluator(_workflow):
        raise RuntimeError("rubric service unavailable")

    agent = _agent_with_persistence(workflow_outcome_evaluator=broken_evaluator)
    workflow = _CapturedWorkflow()

    assert agent._persist_workflow_run(workflow) is True
    assert workflow.stored_outcome == WorkflowOutcome.FAILURE


def test_evaluator_accepts_outcome_string_and_none():
    string_agent = _agent_with_persistence(
        workflow_outcome_evaluator=lambda workflow: "failure"
    )
    string_workflow = _CapturedWorkflow()
    assert string_agent._persist_workflow_run(string_workflow) is True
    assert string_workflow.stored_outcome == WorkflowOutcome.FAILURE

    none_agent = _agent_with_persistence(
        workflow_outcome_evaluator=lambda workflow: None
    )
    none_workflow = _CapturedWorkflow()
    assert none_agent._persist_workflow_run(none_workflow) is True
    assert none_workflow.stored_outcome == WorkflowOutcome.SUCCESS


def test_invalid_evaluator_result_fails_closed():
    agent = _agent_with_persistence(
        workflow_outcome_evaluator=lambda workflow: {"passed": True}
    )
    workflow = _CapturedWorkflow()

    assert agent._persist_workflow_run(workflow) is True
    assert workflow.stored_outcome == WorkflowOutcome.FAILURE


def test_builder_and_clone_preserve_runtime_evaluator():
    evaluator = lambda workflow: True
    builder = MemAgentBuilder().with_workflow_outcome_evaluator(evaluator)

    assert builder.build().workflow_outcome_evaluator is evaluator
    assert builder.clone().build().workflow_outcome_evaluator is evaluator
