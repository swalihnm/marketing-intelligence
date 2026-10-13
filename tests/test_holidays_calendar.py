import pandas as pd

from ingest.holidays_calendar import _black_friday_cyber_monday, build_calendar, to_weekly_indicator


def test_black_friday_is_fourth_friday_of_november():
    events = _black_friday_cyber_monday([2026])
    black_fridays = [d for d, name in events.items() if name == "Black Friday"]
    assert len(black_fridays) == 1
    bf = black_fridays[0]
    assert bf.month == 11
    assert bf.weekday() == 4  # Friday


def test_cyber_monday_is_three_days_after_black_friday():
    events = _black_friday_cyber_monday([2026])
    black_friday = next(d for d, name in events.items() if name == "Black Friday")
    cyber_monday = next(d for d, name in events.items() if name == "Cyber Monday")
    assert (cyber_monday - black_friday).days == 3
    assert cyber_monday.weekday() == 0  # Monday


def test_build_calendar_includes_known_us_holidays():
    daily = build_calendar(country="US", start="2026-01-01", end="2026-12-31")
    names = set(daily["holiday_name"])
    assert "Christmas Day" in names
    assert "Independence Day" in names
    assert "Black Friday" in names


def test_weekly_indicator_counts_holidays_per_week():
    daily = build_calendar(country="US", start="2026-01-01", end="2026-12-31")
    weekly = to_weekly_indicator(daily)
    assert (weekly["n_holidays"] >= 1).all()
    assert weekly["is_holiday_week"].all()
