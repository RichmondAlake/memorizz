"""The playground's proactive routes: the level control and the suggestion card."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from memorizz.enums import MemoryType  # noqa: E402
from memorizz.memagent import MemAgent  # noqa: E402
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider  # noqa: E402
from memorizz.ui import state  # noqa: E402
from memorizz.ui.app import create_app  # noqa: E402
from memorizz.ui.routers import proactive as routes  # noqa: E402
from tests.unit.test_proactive import QUIET, Scripted, calls, offer  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture()
def page(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    provider = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", embedding_provider=None)
    )
    model = Scripted()
    agent = MemAgent(
        name="Ada",
        agent_id=f"ada-{tmp_path.name}",
        model=model,
        memory_provider=provider,
        memory_ids=["trip"],
        auto_register=False,
    )
    agent.save()
    for second, (role, text) in enumerate(
        (
            ("user", "I fly to Lisbon on 14 March for a conference at the LX Factory."),
            (
                "assistant",
                "Noted. You still need a hotel and a way in from the airport.",
            ),
        )
    ):
        provider.store(
            {
                "memory_id": "trip",
                "thread_id": "t1",
                "role": role,
                "content": text,
                "agent_id": agent.agent_id,
                "timestamp": f"2026-11-28T10:00:0{second}+00:00",
            },
            MemoryType.CONVERSATION_MEMORY,
        )

    def live(agent_id, record):
        # The page's agent, on the test's model, with the level last saved.
        agent.with_proactive(routes._policy_of(record))
        return agent

    routes._agents.clear()
    monkeypatch.setattr(routes, "_live_agent", live)
    with patch.dict(state._state, {"provider": provider}):
        client = TestClient(create_app(), follow_redirects=False)
        yield client, f"/api/agents/{agent.agent_id}/proactive", model, agent


def test_an_agent_starts_off_and_the_level_is_saved_on_it(page):
    client, base, _, agent = page

    assert client.get(base).json() == {
        "enabled": False,
        "level": "off",
        "label": "off",
        "predict": False,
        "scan": False,
        "allow_external_actions": False,
    }
    set_level = client.post(base, json={"level": "medium", "scan": True}).json()

    assert (set_level["level"], set_level["scan"]) == ("medium", True)
    assert set_level["evaluations_allowed"] == 12 and set_level["recent"] == []
    saved = agent.memory_provider.retrieve_memagent(agent.agent_id).proactive
    assert saved["level"] == "medium" and saved["scan_tools"] == ["*"]
    assert client.post(base, json={"level": "ultra"}).json()["label"] == "ultra"
    assert client.post(base, json={"level": "off"}).json()["enabled"] is False
    assert agent.memory_provider.retrieve_memagent(agent.agent_id).proactive is None
    assert client.post(base, json={"level": "extreme"}).status_code == 400
    assert client.get("/api/agents/nobody/proactive").status_code == 404


def test_the_page_asks_and_gets_a_suggestion_it_can_answer(page):
    client, base, model, _ = page
    assert client.post(base + "/suggestions", json={"memory_id": "trip"}).json() == {
        "status": "disabled",
        "suggestions": [],
    }
    client.post(base, json={"level": "low"})
    model.then(*offer())

    asked = client.post(base + "/suggestions", json={"memory_id": "trip"}).json()

    (suggestion,) = asked["suggestions"]
    assert asked["status"] == "ready"
    assert (suggestion["title"], suggestion["status"], suggestion["decision"]) == (
        "Find a hotel",
        "offered",
        "ask",
    )
    assert client.get(base).json()["recent"][0]["title"] == "Find a hotel"
    # The page sends an accepted step through the chat, so this only records it.
    answer = base + f"/suggestions/{suggestion['id']}"
    accepted = client.post(answer + "/accept", json={"run": False}).json()
    assert accepted["suggestion"]["status"] == "done"
    assert accepted["suggestion"]["by"] == "user"
    assert client.post(answer + "/accept", json={}).status_code == 409
    assert (
        client.post(base + "/suggestions/unknown/dismiss", json={}).status_code == 404
    )
    assert client.post(answer + "/explode", json={}).status_code == 404


def test_not_now_keeps_the_page_quiet(page):
    client, base, model, _ = page
    client.post(base, json={"level": "low"})
    model.then(*offer())
    suggestion = client.post(base + "/suggestions", json={"memory_id": "trip"}).json()[
        "suggestions"
    ][0]

    client.post(base + f"/suggestions/{suggestion['id']}/snooze", json={"seconds": 600})

    assert client.get(base).json()["snoozed_until"] is not None
    again = client.post(
        base + "/suggestions", json={"memory_id": "trip", "force": True}
    ).json()
    assert again == {"status": "snoozed", "suggestions": []}
    assert (
        client.post(base + "/snooze", json={"seconds": 0}).json()["snoozed_until"]
        is None
    )


def test_at_medium_the_page_is_told_what_the_agent_did(page):
    client, base, model, _ = page
    client.post(base, json={"level": "medium"})
    model.then(*offer(), "Here are the open items.")

    acted = client.post(base + "/suggestions", json={"memory_id": "trip"}).json()

    (suggestion,) = acted["suggestions"]
    assert acted["status"] == "acted"
    assert (suggestion["status"], suggestion["by"]) == ("done", "agent")
    assert suggestion["result"] == "Here are the open items."
    assert client.get(base).json()["actions_today"] == 1


def test_a_quiet_agent_gives_the_page_nothing(page):
    client, base, model, _ = page
    client.post(base, json={"level": "low"})
    model.then(*QUIET)

    assert client.post(base + "/suggestions", json={"memory_id": "trip"}).json() == {
        "status": "quiet",
        "suggestions": [],
    }
    assert calls  # the scripted-call helper is shared with the engine's tests


def test_changing_level_wakes_snooze_and_invalid_snooze_is_rejected(page):
    client, base, model, _ = page
    client.post(base, json={"level": "low"})
    model.then(*offer())
    suggestion = client.post(base + "/suggestions", json={"memory_id": "trip"}).json()[
        "suggestions"
    ][0]

    # Snooze via suggestion, then change level -> wakes immediately.
    assert (
        client.post(
            base + f"/suggestions/{suggestion['id']}/snooze", json={"seconds": 600}
        ).status_code
        == 200
    )
    assert client.get(base).json()["snoozed_until"] is not None
    woken = client.post(base, json={"level": "medium"}).json()
    assert woken["snoozed_until"] is None

    # Unknown suggestion_id returns 404, invalid seconds returns 400 on both routes.
    assert (
        client.post(base + "/suggestions/nonexistent/snooze", json={}).status_code
        == 404
    )
    assert (
        client.post(
            base + f"/suggestions/{suggestion['id']}/snooze", json={"seconds": "abc"}
        ).status_code
        == 400
    )
    assert client.post(base + "/snooze", json={"seconds": "abc"}).status_code == 400
    assert client.post(base + "/snooze", json={"seconds": -5}).status_code == 400


def test_get_proactive_reconciles_pending_approvals(page):
    from tests.unit.test_proactive import hold_room

    client, base, model, agent = page
    agent.tool_manager.add_tool(hold_room)
    client.post(base, json={"level": "medium"})
    model.then(*offer(), calls(("hold_room", {"hotel": "Loft"})), "unused")

    acted = client.post(base + "/suggestions", json={"memory_id": "trip"}).json()
    suggestion = acted["suggestions"][0]
    assert suggestion["status"] == "awaiting_approval"
    assert suggestion["proposal_id"]

    # GET exposes proposal_id for inline approval UI.
    state_before = client.get(base).json()
    assert state_before["recent"][0]["proposal_id"] == suggestion["proposal_id"]

    # Rejecting the proposal reconciles on GET without needing agent.suggestions().
    agent.reject(suggestion["proposal_id"], approver_id="tester", reason="No")
    state_after = client.get(base).json()
    assert state_after["recent"][0]["status"] == "dismissed"


def test_a_proactive_agent_wears_a_badge_in_the_list_and_the_playground(page):
    """Shown the way the Self-aware badge is: there while it applies."""
    client, base, _, agent = page
    playground = f"/agents/{agent.agent_id}/playground"

    # Off: no badge in the list; the playground's chip is there but hidden,
    # so the level control can show it without a reload.
    assert "fleet-proactive" not in client.get("/agents").text
    html = client.get(playground).text
    chip = html.split('id="pg-proactive-chip"')[1].split("</button>")[0]
    assert " hidden" in chip and "Proactive ·" not in chip
    assert 'id="pg-proactive-fact"' in html

    assert client.post(base, json={"level": "medium"}).status_code == 200
    listing = client.get("/agents").text
    pill = listing.split("fleet-proactive")[1].split("</a>")[0]
    assert pill.endswith("Proactive · Medium")
    assert "waits for you" in pill and playground in pill
    chip = client.get(playground).text.split('id="pg-proactive-chip"')[1]
    chip = chip.split("</button>")[0]
    assert " hidden" not in chip and chip.endswith("Proactive · Medium")
    assert 'data-level="medium"' in chip

    # High with prediction is the experimental level, and is named as such.
    client.post(base, json={"level": "ultra"})
    assert "Proactive · Ultra" in client.get("/agents").text
    assert "Proactive · Ultra" in client.get(playground).text

    client.post(base, json={"level": "off"})
    assert "fleet-proactive" not in client.get("/agents").text


def test_the_badge_reads_any_saved_setting_and_survives_a_bad_one():
    from memorizz.ui.helpers import proactive_badge

    assert proactive_badge({"proactive": None})["enabled"] is False
    assert proactive_badge({})["label"] == ""
    assert proactive_badge({"proactive": True})["label"] == "Proactive · Low"
    high = proactive_badge({"proactive": {"enabled": True, "level": "high"}})
    assert (high["level"], high["label"]) == ("high", "Proactive · High")

    class Saved:
        proactive = {"enabled": True, "level": "high", "predict": True}

    assert proactive_badge(Saved())["label"] == "Proactive · Ultra"
    # A level this install does not know is shown as not proactive, not a 500.
    assert proactive_badge({"proactive": {"level": "reckless"}})["enabled"] is False
