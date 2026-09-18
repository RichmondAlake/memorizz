"""The persistent walkthrough uses the public SDK without paid model calls."""

import json

from examples.notion.live_demo import create_overview, exercise
from memorizz.enums import MemoryType
from memorizz.memory_provider.notion import NotionClient, provision_notion_workspace
from tests.fixtures.notion_api import NotionAPI


def test_persistent_demo_has_all_memory_types_real_agent_traces_and_clear_labels(
    tmp_path,
):
    api = NotionAPI()
    client = NotionClient(
        "demo-secret-token", session=api, sleep=lambda _: None, clock=lambda: 0
    )
    workspace = provision_notion_workspace(api.parent_id, client=client)
    result = exercise(
        workspace,
        tmp_path,
        client=client,
        token="demo-secret-token",
        progress=lambda _: None,
    )
    assert set(result["memory_counts"]) == {kind.value for kind in MemoryType}
    assert result["model"]["provider"] == "deterministic-demo"
    assert result["synthetic_total_tokens"] == 60
    assert result["trace_bundles"] == 2
    assert result["index_pending"] == 0
    overview = create_overview(workspace, result, client)
    assert overview["parent"]["page_id"] == workspace["root_page_id"]
    body = api.calls[-1][2]
    assert "synthetic responses and token counts" in json.dumps(body)
    assert "not a billing statement" in json.dumps(body)
    assert "demo-secret-token" not in json.dumps(api.pages)
