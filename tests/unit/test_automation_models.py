from datetime import datetime, timezone

import pytest


def test_validate_timezone_name_accepts_utc():
    from memorizz.automation.schedule import ZoneInfo, validate_timezone_name

    if ZoneInfo is None:
        pytest.skip("zoneinfo unavailable in this environment")

    tz = validate_timezone_name("UTC")
    assert tz is not None


def test_compute_next_run_at_interval():
    from memorizz.automation.schedule import compute_next_run_at

    after = datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)
    nxt = compute_next_run_at(
        schedule_type="interval",
        cron_expr=None,
        interval_seconds=60,
        tz_name="UTC",
        after_utc=after,
    )
    assert nxt == datetime(2026, 2, 13, 9, 1, tzinfo=timezone.utc)


def test_compute_next_run_at_cron_daily_9am_utc():
    from memorizz.automation.schedule import compute_next_run_at

    after = datetime(2026, 2, 13, 8, 0, tzinfo=timezone.utc)
    nxt = compute_next_run_at(
        schedule_type="cron",
        cron_expr="0 9 * * *",
        interval_seconds=None,
        tz_name="UTC",
        after_utc=after,
    )
    assert nxt == datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)


def test_render_query_template_variables():
    from memorizz.automation.schedule import render_query_template

    scheduled = datetime(2026, 2, 13, 9, 0, tzinfo=timezone.utc)
    text = render_query_template(
        "Today is {today_iso} at {scheduled_for_iso} ({timezone})",
        scheduled_for_utc=scheduled,
        tz_name="UTC",
    )
    assert "2026-02-13" in text
    assert "(UTC)" in text


def test_parse_cron_expr_requires_5_fields():
    from memorizz.automation.schedule import parse_cron_expr

    with pytest.raises(ValueError):
        parse_cron_expr("0 9 * * * *")
