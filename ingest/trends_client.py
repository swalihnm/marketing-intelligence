"""Google Trends ingestion via pytrends — real data, no API key required.

Brand search interest as an organic-demand proxy / mediator variable, per
the spec: "Brand search is a genuinely useful MMM covariate." Unlike FRED,
pytrends scrapes Google's public Trends UI rather than calling a stable
documented API, so it is more prone to rate-limiting/blocking — handled
here with a retry/backoff and an honest failure mode (raises rather than
silently returning empty data).

Run: python -m ingest.trends_client --keyword "<brand or category>"
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "trends"


def fetch_trends(keyword: str, timeframe: str = "today 5-y", retries: int = 3, backoff: float = 5.0) -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=0)
    last_error = None
    for attempt in range(retries):
        try:
            pytrends.build_payload([keyword], timeframe=timeframe)
            df = pytrends.interest_over_time()
            if df.empty:
                raise RuntimeError(f"Google Trends returned no data for '{keyword}'")
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])
            df.index.name = "date"
            return df
        except Exception as e:  # pytrends raises various transient errors (429s, etc.)
            last_error = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Google Trends fetch failed for '{keyword}' after {retries} attempts: {last_error}")


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.resample("W-MON").mean()


def run(keyword: str, timeframe: str = "today 5-y"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_trends(keyword, timeframe=timeframe)
    weekly = to_weekly(raw)
    safe_name = keyword.lower().replace(" ", "_")
    raw.to_csv(OUT_DIR / f"trends_{safe_name}_raw.csv")
    weekly.to_csv(OUT_DIR / f"trends_{safe_name}_weekly.csv")
    print(f"fetched {len(raw)} rows for '{keyword}', {raw.index.min()} to {raw.index.max()}")
    print(f"wrote {OUT_DIR / f'trends_{safe_name}_weekly.csv'} ({len(weekly)} weekly rows)")
    return raw, weekly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", default="online shopping", help="brand/category search term")
    parser.add_argument("--timeframe", default="today 5-y")
    args = parser.parse_args()
    raw, weekly = run(args.keyword, args.timeframe)
    print(weekly.tail())
