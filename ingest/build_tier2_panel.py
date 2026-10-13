"""Joins the Tier 2 reference media spine (Robyn's demo dataset) with the
real Tier 3 external covariates (FRED, Google Trends, weather), all
re-fetched for Robyn's exact historical window (2015-11-23 to
2019-11-11), into one aligned weekly panel.

This is the join that `docs/01-data-sources-and-honesty.md` and
`progress.md` previously said wasn't attempted, because no real-or-
reference media spine existed to join the real covariates to. Robyn's
dataset is that spine — simulated by its publisher (Meta), not this
project, and not real spend either, but a different and more-independently-
built kind of simulated data than `simulate/dgp.py`'s, which is why the
spec treats it as a usable "reference spine" category of its own.

**What's real here**: FRED macro series, Google Trends search interest,
and weather, for genuinely the correct historical weeks. **What's not**:
the revenue and channel spend themselves (Robyn's simulated demo data),
and the `price` column (Robyn's dataset has no price series — filled with
a small-noise placeholder around a constant, same technique used in
`causal/mmm_vs_did.py`, documented not hidden).

Run: python -m ingest.build_tier2_panel
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ingest.robyn_reference import channels as robyn_channels
from ingest.robyn_reference import load as load_robyn

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"

FRED_WEEKLY_PATH = RAW_DIR / "fred" / "fred_macro_weekly.csv"
TRENDS_WEEKLY_PATH = RAW_DIR / "trends" / "trends_online_shopping_weekly.csv"
WEATHER_WEEKLY_PATH = RAW_DIR / "weather" / "weather_london_robyn_weekly.csv"


def build_panel(seed: int = 7) -> pd.DataFrame:
    robyn = load_robyn()
    start, end = robyn["date"].min(), robyn["date"].max()

    fred = pd.read_csv(FRED_WEEKLY_PATH, index_col=0, parse_dates=True)
    fred = fred.reindex(robyn["date"], method="ffill")
    fred.index.name = "date"

    trends = pd.read_csv(TRENDS_WEEKLY_PATH, index_col=0, parse_dates=True)
    trends = trends.reindex(robyn["date"], method="nearest")
    trends.columns = ["brand_search_interest"]

    weather = pd.read_csv(WEATHER_WEEKLY_PATH, index_col=0, parse_dates=True)
    weather = weather[(weather.index >= start) & (weather.index <= end)]
    weather = weather.reindex(robyn["date"], method="nearest")

    panel = robyn.set_index("date").copy()
    panel = panel.join(fred, how="left")
    panel = panel.join(trends, how="left")
    panel = panel.join(weather, how="left")

    rng = np.random.default_rng(seed)
    panel["price"] = 100.0 + rng.normal(0, 2.0, len(panel))

    panel = panel.reset_index()
    return panel


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    out_path = OUT_DIR / "tier2_full_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"built panel: {len(panel)} weeks, {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"columns: {panel.columns.tolist()}")
    n_missing = panel.isna().sum()
    if n_missing.any():
        print("missing values per column:\n", n_missing[n_missing > 0])
    print(f"wrote {out_path}")
    return panel


if __name__ == "__main__":
    panel = run()
    print(panel.head())
