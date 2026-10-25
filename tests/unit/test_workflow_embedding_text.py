"""What a workflow is embedded from: its request and tools, not tool output."""

import json

from memorizz.enums import MemoryType
from memorizz.finetune.swap import embedding_text
from memorizz.long_term.procedural.workflow import Workflow, WorkflowOutcome

PAGE = "THE ALGORITHM " + "lorem ipsum " * 20000


def _workflow(**overrides):
    fields = dict(
        name="Tool Execution for Query",
        description="Workflow tracking tool usage for: can you open the algorithm page",
        user_query="can you open the algorithm page",
        steps={
            "Step 1: mcp_list_tools": {"arguments": {}, "result": "x" * 50000},
            "Step 2: notion__notion-search": {
                "arguments": {"query": "algorithm"},
                "result": PAGE,
            },
            "Step 3: notion__notion-fetch": {
                "arguments": {"id": "page-1"},
                "result": PAGE,
            },
        },
        generate_embedding=False,
    )
    fields.update(overrides)
    return Workflow(**fields)


def test_the_embedding_names_the_request_and_tools_but_not_their_output():
    text = _workflow().embedding_input()
    assert "can you open the algorithm page" in text
    assert "mcp_list_tools notion__notion-search notion__notion-fetch" in text
    assert "success" in text
    assert "lorem" not in text and "xxxx" not in text
    assert len(text) < 1000


def test_a_pasted_document_is_cut_so_the_tools_still_count():
    text = _workflow(user_query="summarise this: " + PAGE).embedding_input()
    assert "notion__notion-fetch" in text
    assert len(text) < 5000


def test_a_failed_run_says_so():
    workflow = _workflow(outcome=WorkflowOutcome.FAILURE)
    assert "failure" in workflow.embedding_input()


def test_storing_embeds_the_short_text(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "memorizz.long_term.procedural.workflow.workflow.get_embedding",
        lambda text: seen.append(text) or [0.1, 0.2],
    )

    class _Store:
        def store(self, data, memory_store_type):
            return data["workflow_id"]

    workflow = _workflow()
    workflow.store_workflow(_Store())
    assert seen == [workflow.embedding_input()]
    assert workflow.embedding == [0.1, 0.2]


def test_re_embedding_a_stored_row_uses_the_same_text():
    workflow = _workflow()
    row = workflow.to_dict()
    assert embedding_text(MemoryType.WORKFLOW_MEMORY, row) == (
        workflow.embedding_input()
    )
    # Stores with JSON columns hand steps and outcome back as text.
    row["steps"] = json.dumps(row["steps"])
    assert embedding_text(MemoryType.WORKFLOW_MEMORY, row) == (
        workflow.embedding_input()
    )
