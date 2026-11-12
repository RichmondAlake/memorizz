# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""The shell every page shares: grouped navigation, the top bar's search,
the theme set before first paint, and the Overview's change from the period
before."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
from memorizz.ui import state
from memorizz.ui.app import create_app
from memorizz.ui.dashboard import period_deltas

THEME_SCRIPT = "document.documentElement.classList.toggle('theme-light', light)"


class _Events(list):
    coverage = {"source_rows": 4}


def _run(turn, *, ago: timedelta, status="success", tokens=1000):
    started = datetime.now(timezone.utc) - ago
    identity = {"turn_id": turn, "agent_id": "a1"}
    return [
        {**identity, "trace_kind": "turn_start", "timestamp": started.isoformat()},
        {
            **identity,
            "trace_kind": "model_result",
            "status": "success",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "input_tokens": tokens,
            "output_tokens": 0,
            "cost_usd": "0.01",
            "timestamp": started.isoformat(),
        },
        {
            **identity,
            "trace_kind": "turn_result",
            "status": status,
            "timestamp": (started + timedelta(seconds=2)).isoformat(),
        },
    ]


@pytest.fixture()
def client(tmp_path):
    provider = FileSystemProvider(
        FileSystemConfig(root_path=Path(tmp_path) / "ui", lazy_vector_indexes=True)
    )
    with patch.dict(
        state._state,
        {
            "provider": provider,
            "provider_type": "filesystem",
            "connection_info": {"root_path": str(provider.root_path)},
        },
    ):
        yield TestClient(create_app(), follow_redirects=False)
    provider.close()


def _totals(**overrides):
    totals = {
        "runs": 10,
        "success_rate": 90.0,
        "p95_ms": 1000.0,
        "model_calls": 10,
        "input_tokens": 900,
        "output_tokens": 100,
        "cost_usd": Decimal("1.00"),
    }
    return {**totals, **overrides}


@pytest.mark.unit
def test_period_deltas_set_a_tone_only_where_a_direction_is_good_or_bad():
    deltas = period_deltas(
        _totals(runs=15, success_rate=80.0, p95_ms=500.0, cost_usd=Decimal("2.00")),
        _totals(),
    )
    # More runs or more spend is neither good nor bad on its own.
    assert deltas["runs"] == {"arrow": "▲", "text": "50%", "tone": ""}
    assert deltas["cost_usd"] == {"arrow": "▲", "text": "100%", "tone": ""}
    # Success rate compares in points; a faster p95 is good.
    assert deltas["success_rate"] == {
        "arrow": "▼",
        "text": "10.0 pts",
        "tone": "is-bad",
    }
    assert deltas["p95_ms"] == {"arrow": "▼", "text": "50%", "tone": "is-good"}
    assert deltas["model_calls"] == {"arrow": "", "text": "No change", "tone": ""}


@pytest.mark.unit
def test_period_deltas_need_something_to_compare_with():
    assert period_deltas(_totals(), None) == {}
    assert period_deltas(None, _totals()) == {}
    # A figure missing on either side is left out, the rest are kept.
    deltas = period_deltas(_totals(), _totals(success_rate=None, p95_ms=None))
    assert "success_rate" not in deltas and "p95_ms" not in deltas
    assert "runs" in deltas
    # Nothing was spent before, so there is no percentage to show.
    assert "cost_usd" not in period_deltas(_totals(), _totals(cost_usd=Decimal(0)))


@pytest.mark.unit
def test_overview_tiles_show_the_change_from_the_period_before(client):
    events = _Events(
        [
            *_run("now-1", ago=timedelta(hours=1)),
            *_run("now-2", ago=timedelta(hours=2)),
            *_run("before-1", ago=timedelta(hours=30)),
        ]
    )
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, events)
    ):
        html = client.get("/dashboard?window=24h").text
    assert 'title="Change from the period before"' in html
    assert '<span class="visually-hidden">Up</span> 100%</span>' in html


@pytest.mark.unit
def test_overview_shows_no_change_when_the_earlier_period_was_not_read(client):
    events = _Events(
        [
            *_run("now-1", ago=timedelta(hours=1)),
            *_run("before-1", ago=timedelta(hours=30)),
        ]
    )
    # The read stopped 30 hours back: the 24 hours before are only part read.
    events.coverage = {"source_rows": 2, "truncated": True}
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, events)
    ):
        html = client.get("/dashboard?window=24h").text
    assert "kpi-delta" not in html
    # Nothing ran in the period before: there is nothing to compare with.
    alone = _Events(_run("now-1", ago=timedelta(hours=1)))
    with patch(
        "memorizz.ui.routers.traces._selection_window", return_value=(None, alone)
    ):
        assert "kpi-delta" not in client.get("/dashboard?window=24h").text


@pytest.mark.unit
def test_every_page_has_grouped_navigation_search_and_one_theme_switch(client):
    html = client.get("/memory/knowledge-base").text
    for section in ("Build", "Memory", "Connect", "Operate", "Admin"):
        assert (
            f'<div class="nav-section" role="presentation"><span>{section}</span>'
            in html
        )
    assert "<span>Overview</span>" in html
    # Search opens from the top bar and, on a phone, from the top strip.
    assert html.count("data-cmdk-open") == 2
    assert html.count('id="theme-toggle-btn"') == 1
    assert 'id="disconnect-btn"' in html
    assert '<a class="skip-link" href="#main-content">' in html
    assert '<main class="main-content" id="main-content"' in html
    # The in-page dialogs are loaded once, for every page.
    assert html.count("/static/js/page-dialogs.js") == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, group, area, href",
    [
        ("/dashboard", None, "Overview", "/dashboard"),
        ("/agents/new", "Build", "Agents", "/agents"),
        ("/memory/entity", "Memory", "Memory types", "/memory/knowledge-base"),
        ("/fine-tuning", "Memory", "Continual learning", "/memory/workflows"),
        ("/traces/usage", "Operate", "Observability", "/traces"),
        ("/settings", "Admin", "Settings", "/settings"),
    ],
)
def test_the_top_bar_says_where_the_page_sits_in_the_sidebar(
    client, path, group, area, href
):
    html = client.get(path).text
    crumbs = html.split('<nav class="app-crumbs" aria-label="Breadcrumb">')[1]
    crumbs = crumbs.split("</nav>")[0]
    assert f'<li><a href="{href}">{area}</a></li>' in crumbs
    assert crumbs.count("<li>") == (2 if group else 1)
    if group:
        # The group is the sidebar's own heading, so the two never disagree.
        assert f"<li>{group}</li>" in crumbs
        assert f'role="presentation"><span>{group}</span>' in html
    # The header no longer repeats it above the title.
    assert "page-eyebrow" not in html


@pytest.mark.unit
def test_shortcuts_open_from_the_top_bar_and_cover_the_sidebar(client):
    import re

    html = client.get("/dashboard").text
    assert html.count("data-shortcuts-open") == 1
    script = (
        Path(__file__).resolve().parents[2] / "src/memorizz/ui/static/js/app.js"
    ).read_text()
    table = script.split("const GO_TO = [")[1].split("];")[0]
    rows = re.findall(r'\["(\w)", "[^"]+", "([^"]+)"\]', table)
    keys = [key for key, _ in rows]
    assert len(keys) == len(set(keys)) >= 12
    for _, target in rows:
        # Every go-to target is a page the sidebar links to.
        assert f'href="{target}' in html, target


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/dashboard", "/connect"])
def test_the_theme_is_set_before_the_page_paints(client, path):
    html = client.get(path).text
    head = html.split("</head>")[0]
    assert THEME_SCRIPT in head
    assert head.index(THEME_SCRIPT) < head.index('rel="stylesheet"')
    assert "/static/css/shell.css" in head


@pytest.mark.unit
def test_sign_in_page_uses_the_shared_theme(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORIZZ_UI_AUTH_TOKEN", "shell-test-token")
    monkeypatch.setenv("MEMORIZZ_UI_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    html = TestClient(create_app(), follow_redirects=False).get("/login").text
    assert THEME_SCRIPT in html
    assert "/static/css/shell.css" in html
    assert '<section class="auth-card" aria-labelledby="login-title">' in html


@pytest.mark.unit
def test_connect_page_carries_no_forced_colours(client):
    html = client.get("/connect").text
    assert "!important" not in html
    assert "#00d9ff" not in html
    assert "/static/css/pages/connect.css" in html


@pytest.mark.unit
def test_no_page_opens_a_browser_pop_up():
    """confirm(), prompt() and alert() block the page; MemorizzDialogs replaces
    them everywhere."""
    import re

    ui = Path(__file__).resolve().parents[2] / "src" / "memorizz" / "ui"
    call = re.compile(r"(?<![.\w])(?:window\.)?(?:confirm|prompt|alert)\(")
    offenders = []
    for path in [*ui.glob("templates/*.html"), *ui.glob("static/js/*.js")]:
        if path.name == "page-dialogs.js":
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if call.search(line):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


@pytest.mark.unit
def test_the_phone_drawer_is_not_left_under_its_scrim():
    """`body > *` puts the layout on its own stacking level, below the scrim;
    the open drawer has to lift out of it or its links cannot be clicked."""
    css = Path(__file__).resolve().parents[2] / "src/memorizz/ui/static/css"
    assert (
        "body > * {\n    position: relative;\n    z-index: 1;"
        in (css / "styles.css").read_text()
    )
    shell = (css / "shell.css").read_text()
    assert "body.mobile-nav-open .layout {\n        z-index: auto;" in shell
