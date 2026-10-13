"""Holiday calendar via the `holidays` package — real data, no network
call or API key required (it's a bundled rules database, not a live
lookup). Per the spec: "plus retail events (Black Friday, Ramadan and Eid
if you want GCC relevance)."

Produces a weekly indicator panel: for each ISO week, which holidays fall
in it, plus hand-added retail events the `holidays` package doesn't cover
(Black Friday, Cyber Monday — US-style "day after Thanksgiving" and the
following Monday).

Run: python -m ingest.holidays_calendar --country US --start 2021-01-01 --end 2026-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import holidays as holidays_pkg
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "holidays"


def _black_friday_cyber_monday(years: list[int]) -> dict:
    """US-style: Black Friday = day after 4th Thursday of November;
    Cyber Monday = the following Monday. Added manually since the
    `holidays` package treats these as informal, not public, holidays."""
    events = {}
    for year in years:
        nov1 = pd.Timestamp(year=year, month=11, day=1)
        thursdays = pd.date_range(nov1, nov1 + pd.Timedelta(days=29), freq="W-THU")
        fourth_thursday = thursdays[3]
        black_friday = fourth_thursday + pd.Timedelta(days=1)
        cyber_monday = black_friday + pd.Timedelta(days=3)
        events[black_friday.date()] = "Black Friday"
        events[cyber_monday.date()] = "Cyber Monday"
    return events


def build_calendar(country: str = "US", start: str = "2021-01-01", end: str = "2026-12-31",
                    extra_subdivisions: list[str] | None = None) -> pd.DataFrame:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    years = list(range(start_ts.year, end_ts.year + 1))

    country_holidays = holidays_pkg.country_holidays(country, years=years)
    events = dict(country_holidays)
    events.update(_black_friday_cyber_monday(years))

    rows = [
        {"date": pd.Timestamp(d), "holiday_name": name}
        for d, name in events.items()
        if start_ts <= pd.Timestamp(d) <= end_ts
    ]
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return daily


def to_weekly_indicator(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["week"] = df["date"].dt.to_period("W-SUN").dt.start_time
    weekly = df.groupby("week").agg(
        n_holidays=("holiday_name", "count"),
        holiday_names=("holiday_name", lambda x: "; ".join(x)),
    ).reset_index()
    weekly["is_holiday_week"] = True
    return weekly


def run(country: str = "US", start: str = "2021-01-01", end: str = "2026-12-31"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = build_calendar(country, start, end)
    weekly = to_weekly_indicator(daily)
    daily.to_csv(OUT_DIR / f"holidays_{country}_daily.csv", index=False)
    weekly.to_csv(OUT_DIR / f"holidays_{country}_weekly.csv", index=False)
    print(f"built {len(daily)} holiday events for {country}, {start} to {end}")
    print(f"wrote {OUT_DIR / f'holidays_{country}_weekly.csv'} ({len(weekly)} holiday-weeks)")
    return daily, weekly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="US")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-12-31")
    args = parser.parse_args()
    daily, weekly = run(args.country, args.start, args.end)
    print(daily.tail(10))
