"""FRED (Federal Reserve Economic Data) macro covariate ingestion.

Real data — this is the one Tier 3 source the spec names explicitly:
"Macro conditions confound marketing effectiveness and controlling for
them is proper practice." Pulls the four series the spec lists:

  - UMCSENT   University of Michigan Consumer Sentiment
  - CPIAUCSL  CPI, all urban consumers
  - RSXFS     Retail sales, excluding food services
  - PCE       Personal consumption expenditures

All four are monthly series; resampled to weekly (forward-filled — macro
releases are monthly, so a given week's "value" is the most recent release
available by that week, which is the honest real-time-available reading)
to align with the MMM's weekly grain.

Requires FRED_API_KEY (free, from
https://fredaccount.stlouisfed.org/apikeys) set in a local .env file — not
committed; see .env.example. This module is the one piece of the MMM media
spine in this project that runs on real data; everything else in
simulate/dgp.py remains simulated (see docs/01-data-sources-and-honesty.md).

Run: python -m ingest.fred_client
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "fred"

SERIES = {
    "UMCSENT": "consumer_sentiment",
    "CPIAUCSL": "cpi",
    "RSXFS": "retail_sales",
    "PCE": "personal_consumption_expenditures",
}


def _get_client() -> Fred:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Copy .env.example to .env and fill in a "
            "free key from https://fredaccount.stlouisfed.org/apikeys"
        )
    return Fred(api_key=api_key)


def fetch_series(fred: Fred, series_id: str, start: str = "2015-01-01") -> pd.Series:
    s = fred.get_series(series_id, observation_start=start)
    s.index.name = "date"
    s.name = series_id
    return s


def fetch_all(start: str = "2015-01-01") -> pd.DataFrame:
    fred = _get_client()
    frames = {}
    for series_id, friendly_name in SERIES.items():
        s = fetch_series(fred, series_id, start=start)
        frames[friendly_name] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df


def to_weekly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Resample monthly macro releases to a weekly grain via forward-fill —
    a given week carries the most recently *available* monthly reading,
    which is the realistic real-time view (not a look-ahead interpolation)."""
    weekly = monthly_df.resample("W-MON").ffill()
    return weekly


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly = fetch_all()
    monthly.to_csv(OUT_DIR / "fred_macro_monthly.csv")
    weekly = to_weekly(monthly)
    weekly.to_csv(OUT_DIR / "fred_macro_weekly.csv")
    print(f"fetched {len(monthly)} monthly observations, {monthly.index.min()} to {monthly.index.max()}")
    print(f"wrote {OUT_DIR / 'fred_macro_monthly.csv'}")
    print(f"wrote {OUT_DIR / 'fred_macro_weekly.csv'} ({len(weekly)} weekly rows)")
    return monthly, weekly


if __name__ == "__main__":
    monthly, weekly = run()
    print(monthly.tail())
