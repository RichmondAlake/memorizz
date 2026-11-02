# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Countdown timers, and the two automation bugs a pomodoro request ran into."""

from __future__ import annotations

import io
import json
import os
import stat
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rich.console import Console

from memorizz import timers
from memorizz.automation import schedule
from memorizz.cli import timer_commands, ui
from memorizz.memagent import MemAgent
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider

pytestmark = pytest.mark.unit

LLM = {"provider": "ollama", "model": "qwen2.5:7b"}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MEMORIZZ_DEFAULT_TIMEZONE", raising=False)
    return tmp_path / "home"


@pytest.fixture()
def provider(tmp_path):
    store = FileSystemProvider(
        FileSystemConfig(root_path=tmp_path / "memory", lazy_vector_indexes=True)
    )
    yield store
    store.close()


def _tool(agent, name):
    return agent.tool_manager.get_tool_callable(name)


# -- the two bugs ------------------------------------------------------------


def test_a_one_shot_waits_for_its_delay():
    now = datetime(2026, 11, 2, 1, 36, tzinfo=timezone.utc)
    next_run = schedule.compute_next_run_at(
        schedule_type="one_shot",
        cron_expr=None,
        interval_seconds=1500,
        after_utc=now,
        tz_name="UTC",
    )
    assert next_run == now + timedelta(minutes=25)


def test_a_one_shot_with_no_delay_still_runs_straight_away():
    now = datetime(2026, 11, 2, 1, 36, tzinfo=timezone.utc)
    assert (
        schedule.compute_next_run_at(
            schedule_type="one_shot",
            cron_expr=None,
            interval_seconds=None,
            after_utc=now,
            tz_name="UTC",
        )
        == now
    )


def test_a_negative_delay_is_refused():
    with pytest.raises(ValueError, match="negative"):
        schedule.compute_next_run_at(
            schedule_type="one_shot",
            cron_expr=None,
            interval_seconds=-1,
            after_utc=datetime(2026, 11, 2, tzinfo=timezone.utc),
            tz_name="UTC",
        )


def test_an_automation_needs_no_timezone_named(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    with patch.object(schedule, "local_timezone_name", lambda: "Europe/London"):
        result = _tool(agent, "automation_create_job")(
            name="Reminder",
            schedule_type="one_shot",
            interval_seconds=60,
            query_template="Say hello.",
        )
    assert result["ok"], result


def test_a_cron_time_on_a_server_has_to_name_its_timezone(provider):
    shared = MemAgent(llm_config=LLM, memory_provider=provider, multi_tenant=True)
    assert "depends on a timezone: ask for one" in shared._build_system_prompt()
    with pytest.raises(ValueError, match="timezone is required"):
        _tool(shared, "automation_create_job")(
            name="Digest",
            schedule_type="cron",
            cron_expr="0 8 * * *",
            query_template="Say hello.",
        )


def test_a_relative_automation_on_a_server_needs_no_timezone(provider):
    shared = MemAgent(llm_config=LLM, memory_provider=provider, multi_tenant=True)
    result = _tool(shared, "automation_create_job")(
        name="Reminder",
        schedule_type="interval",
        interval_seconds=3600,
        query_template="Say hello.",
    )
    assert result["ok"], result


def test_a_named_timezone_still_wins(provider, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_DEFAULT_TIMEZONE", "Asia/Tokyo")
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    result = _tool(agent, "automation_create_job")(
        name="Reminder",
        schedule_type="one_shot",
        interval_seconds=60,
        query_template="Say hello.",
    )
    assert agent.automation_manager.store.get_job(result["job_id"]).timezone == (
        "Asia/Tokyo"
    )


@pytest.mark.parametrize(
    "configured,link,expected",
    [
        ("Asia/Tokyo", "/var/db/timezone/zoneinfo/Europe/London", "Asia/Tokyo"),
        (":Asia/Tokyo", "/etc/localtime", "Asia/Tokyo"),
        ("", "/var/db/timezone/zoneinfo/Europe/London", "Europe/London"),
        ("Not/AZone", "/usr/share/zoneinfo/Europe/Paris", "Europe/Paris"),
        ("", "/etc/localtime", None),
    ],
)
def test_the_machines_timezone_is_found(monkeypatch, configured, link, expected):
    monkeypatch.setenv("TZ", configured)
    monkeypatch.setattr(schedule.os.path, "realpath", lambda path: link)
    monkeypatch.setattr(schedule, "_ETC_TIMEZONE", "/no/such/file")
    assert schedule.local_timezone_name() == expected


def test_a_container_can_say_through_etc_timezone(tmp_path, monkeypatch):
    named = tmp_path / "timezone"
    named.write_text("Europe/Paris\n", encoding="utf-8")
    monkeypatch.setenv("TZ", "")
    monkeypatch.setattr(schedule.os.path, "realpath", lambda path: "/etc/localtime")
    monkeypatch.setattr(schedule, "_ETC_TIMEZONE", str(named))
    assert schedule.local_timezone_name() == "Europe/Paris"


def test_a_cron_time_isnt_guessed_when_the_machine_cant_say(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    with patch(
        "memorizz.memagent.managers.automation_manager.local_timezone_name",
        lambda: None,
    ), pytest.raises(ValueError, match="ask the user"):
        _tool(agent, "automation_create_job")(
            name="Digest",
            schedule_type="cron",
            cron_expr="0 8 * * *",
            query_template="Say hello.",
        )


def test_a_relative_job_doesnt_mind_when_the_machine_cant_say(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    with patch(
        "memorizz.memagent.managers.automation_manager.local_timezone_name",
        lambda: None,
    ):
        result = _tool(agent, "automation_create_job")(
            name="Reminder",
            schedule_type="one_shot",
            interval_seconds=60,
            query_template="Say hello.",
        )
    assert result["ok"], result
    assert agent.automation_manager.store.get_job(result["job_id"]).timezone == "UTC"


# -- the store ---------------------------------------------------------------


def test_a_timer_counts_down_and_then_finishes():
    started = timers.start(1500, "Focus", now=1000.0)
    assert started["state"] == "running"
    assert started["ends_at"] == 2500.0

    [running] = timers.listed(now=1100.0)
    assert running["remaining"] == 1400.0

    [finished] = timers.listed(now=2500.0)
    assert finished["state"] == "finished"
    assert finished["remaining"] == 0
    assert timers.running(now=2500.0) == []


@pytest.mark.parametrize("seconds", [0, -5, None])
def test_a_timer_needs_a_length(seconds):
    with pytest.raises(ValueError, match="at least one second"):
        timers.start(seconds)


def test_a_timer_has_a_limit():
    with pytest.raises(ValueError, match="24 hours"):
        timers.start(timers.MAX_SECONDS + 1)


def test_labels_are_kept_short_and_have_a_default():
    assert timers.start(60)["label"] == "Timer"
    assert timers.start(60, "  a\n b ")["label"] == "a b"
    assert len(timers.start(60, "x" * 500)["label"]) == 60


def test_running_timers_come_first_soonest_first():
    timers.start(100, "Done", now=0.0)
    timers.start(900, "Later", now=0.0)
    timers.start(300, "Sooner", now=0.0)
    assert [item["label"] for item in timers.listed(now=200.0)] == [
        "Sooner",
        "Later",
        "Done",
    ]


def test_cancel_takes_the_only_timer_or_asks_which():
    first = timers.start(60, "One", now=0.0)
    assert timers.cancel(now=10.0)["timer_id"] == first["timer_id"]
    timers.start(60, "One", now=0.0)
    second = timers.start(60, "Two", now=0.0)
    assert timers.cancel(now=10.0) is None
    assert timers.cancel(second["timer_id"], now=10.0)["label"] == "Two"
    assert timers.cancel("nope") is None


def test_a_dismissed_timer_doesnt_ring_at_the_next_launch():
    tea = timers.start(10, "Tea", now=0.0)
    timers.dismiss(tea["timer_id"])
    assert timers.take_finished(now=20.0) == []


def test_a_timer_that_ended_long_ago_doesnt_ring():
    timers.start(10, "Old", now=0.0)
    assert timers.take_finished(now=timers.KEEP_FINISHED_SECONDS + 11) == []


@pytest.mark.parametrize(
    "row",
    [
        {"timer_id": "a"},
        {"timer_id": "b", "label": "x", "seconds": "60", "started_at": 0, "ends_at": 1},
        {"timer_id": 3, "label": "x", "seconds": 60, "started_at": 0, "ends_at": 1},
        {"label": "x", "seconds": 60, "started_at": 0, "ends_at": 1},
        "not a row",
    ],
)
def test_a_row_that_isnt_a_timer_breaks_nothing(home, row):
    home.mkdir(parents=True, exist_ok=True)
    (home / "timers.json").write_text(json.dumps([row]), encoding="utf-8")
    assert timers.listed() == []
    assert timers.running() == []
    assert timers.take_finished() == []
    assert ui.timer_fragment() == ""


def test_a_home_it_cant_write_to_doesnt_stop_the_cli(monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(timers, "take_finished", refuse)
    monkeypatch.setattr(timers, "running", refuse)
    timer_commands.announce_finished(Console(file=io.StringIO()))
    assert ui.timer_fragment() == ""


def test_timer_says_when_it_cant_save(monkeypatch):
    def refuse(*args, **kwargs):
        raise PermissionError("read-only home")

    monkeypatch.setattr(timers, "start", refuse)
    session = _session()
    timer_commands.cmd_timer(session, "5")
    assert "read-only home" in _printed(session)


def test_finished_timers_are_announced_once():
    timers.start(10, "Tea", now=0.0)
    assert timers.take_finished(now=5.0) == []
    assert [item["label"] for item in timers.take_finished(now=11.0)] == ["Tea"]
    assert timers.take_finished(now=12.0) == []
    # The UI still shows it until someone puts it away.
    assert timers.listed(now=12.0)[0]["state"] == "finished"


def test_a_dismissed_timer_is_gone_and_old_ones_clear_themselves():
    tea = timers.start(10, "Tea", now=0.0)
    assert timers.dismiss(tea["timer_id"]) is True
    assert timers.listed(now=20.0) == []
    assert timers.dismiss("nope") is False

    timers.start(10, "Old", now=0.0)
    assert timers.listed(now=timers.KEEP_FINISHED_SECONDS + 11) == []


def test_the_file_is_private_and_a_damaged_one_is_no_crash(home):
    timers.start(60, "Focus")
    path = home / "timers.json"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    path.write_text("{ not json", encoding="utf-8")
    assert timers.listed() == []
    assert timers.start(60, "Again")["label"] == "Again"


@pytest.mark.parametrize(
    "seconds,shown",
    [(0, "0:00"), (0.2, "0:01"), (59, "0:59"), (1500, "25:00"), (3600, "1:00:00")],
)
def test_the_clock_reads_like_a_clock(seconds, shown):
    assert timers.format_time(seconds) == shown


# -- the tools ---------------------------------------------------------------


def test_starting_a_timer_needs_no_approval_or_timezone(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    assert (
        agent.tool_manager.get_tool_policy("start_timer")["requires_approval"] is False
    )
    result = _tool(agent, "start_timer")(minutes=25, label="Focus")
    assert result["ok"] and result["length"] == "25:00"
    assert timers.running()[0]["agent_id"] == agent.agent_id


def test_the_model_always_has_the_timer(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    assert "start_timer" in agent.semantic_tool_router.always_visible


def test_the_prompt_says_to_start_a_timer_without_questions(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    assert "call 'start_timer' straight away" in agent._build_system_prompt()

    shared = MemAgent(llm_config=LLM, memory_provider=provider, multi_tenant=True)
    assert "start_timer" not in shared._build_system_prompt()


def test_a_bad_length_comes_back_as_an_answer_the_model_can_use(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    assert "at least one second" in _tool(agent, "start_timer")()["error"]


def test_list_and_cancel_tools(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider)
    _tool(agent, "start_timer")(minutes=25, label="Focus")
    [item] = _tool(agent, "list_timers")()["timers"]
    assert item["label"] == "Focus"
    assert _tool(agent, "cancel_timer")()["label"] == "Focus"
    assert "list_timers" in _tool(agent, "cancel_timer")()["error"]


def test_an_agent_serving_several_users_has_no_timer(provider):
    agent = MemAgent(llm_config=LLM, memory_provider=provider, multi_tenant=True)
    assert "start_timer" not in agent.tool_manager.tools


def test_turning_automations_off_takes_the_timer_too(provider):
    agent = MemAgent(
        llm_config=LLM, memory_provider=provider, automations_enabled=False
    )
    assert "start_timer" not in agent.tool_manager.tools


# -- the CLI -----------------------------------------------------------------


def _session():
    return SimpleNamespace(
        agent=SimpleNamespace(agent_id="agent-1"),
        console=Console(file=io.StringIO(), width=120),
    )


def _printed(session):
    return session.console.file.getvalue()


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("25", 1500),
        ("90s", 90),
        ("25m", 1500),
        ("1h", 3600),
        ("1.5", 90),
        ("2hrs", 7200),
    ],
)
def test_lengths(text, seconds):
    assert timer_commands.parse_length(text) == seconds


def test_timer_starts_lists_and_cancels():
    session = _session()
    timer_commands.cmd_timer(session, "25 deep work")
    assert "deep work: 25:00" in _printed(session)

    timer_commands.cmd_timer(session, "")
    assert "deep work" in _printed(session)

    timer_commands.cmd_timer(session, "cancel")
    assert "Cancelled deep work" in _printed(session)
    assert timers.running() == []


def test_timer_explains_itself():
    session = _session()
    timer_commands.cmd_timer(session, "soon")
    assert "/timer cancel [id]" in _printed(session)
    timer_commands.cmd_timer(session, "")
    assert "No timers running" in _printed(session)
    timer_commands.cmd_timer(session, "cancel")
    assert "Say which timer" in _printed(session)
    timer_commands.cmd_timer(session, "9999h")
    assert "24 hours" in _printed(session)


def test_a_label_is_shown_as_typed():
    session = _session()
    timer_commands.cmd_timer(session, "5 [tea]")
    assert "[tea]" in _printed(session)


def test_the_status_bar_counts_down():
    assert ui.timer_fragment() == ""
    timers.start(1500, "Focus")
    timers.start(3000, "Long")
    fragment = ui.timer_fragment()
    assert "Focus 25:00" in fragment or "Focus 24:59" in fragment
    assert "+1" in fragment


def test_a_label_cant_break_the_status_bar():
    timers.start(60, "<b>&")
    assert "&lt;b&gt;&amp;" in ui.timer_fragment()


def test_the_end_of_a_timer_rings_once(monkeypatch):
    timers.start(10, "Tea", now=time.time() - 20)
    console = Console(file=io.StringIO(), width=80, force_terminal=True)
    timer_commands.announce_finished(console)
    timer_commands.announce_finished(console)
    output = console.file.getvalue()
    assert output.count("Tea finished (0:10)") == 1
    assert output.count("\a") == 1


# -- the local UI ------------------------------------------------------------


@pytest.fixture()
def client(provider):
    from fastapi.testclient import TestClient

    from memorizz.ui import state
    from memorizz.ui.app import create_app

    with patch.dict(
        state._state,
        {"provider": provider, "provider_type": "filesystem", "connection_info": {}},
    ):
        yield TestClient(create_app(), follow_redirects=False)


def test_the_ui_sees_a_timer_started_in_the_cli(client):
    started = timers.start(1500, "Focus")
    [item] = client.get("/api/timers").json()["timers"]
    assert item["timer_id"] == started["timer_id"]
    assert item["state"] == "running"


def test_cancel_and_dismiss_from_the_ui(client):
    started = timers.start(1500, "Focus")
    assert client.post(f"/api/timers/{started['timer_id']}/cancel").status_code == 200
    assert client.get("/api/timers").json()["timers"] == []
    assert client.post(f"/api/timers/{started['timer_id']}/cancel").status_code == 404

    tea = timers.start(1, "Tea", now=0.0)
    assert client.post(f"/api/timers/{tea['timer_id']}/dismiss").status_code == 200
    assert client.get("/api/timers").json()["timers"] == []


def test_a_limited_account_gets_no_timers(client):
    from memorizz.ui.trace_access import current_principal

    token = current_principal.set(SimpleNamespace(restricted=True))
    try:
        from memorizz.ui.routers import timers as route

        with pytest.raises(Exception) as caught:
            route._unrestricted()
    finally:
        current_principal.reset(token)
    assert caught.value.status_code == 403


def test_the_playground_has_a_place_for_timers():
    from pathlib import Path

    import memorizz.ui

    root = Path(memorizz.ui.__file__).parent
    page = (root / "templates" / "playground.html").read_text(encoding="utf-8")
    assert 'id="pg-timers"' in page
    assert "playground-timers.js" in page
    assert (root / "static" / "js" / "playground-timers.js").exists()
