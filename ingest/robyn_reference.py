"""Meta Robyn demo dataset — the Tier 2 reference media spine named in the
spec ("Meta Robyn demo dataset (dt_simulated_weekly...) — weekly revenue
with media spend across channels. Use these as the media spine. Be
explicit in your docs about which parts are real and which are simulated.")

**This is Meta's own simulated dataset**, bundled with the open-source
Robyn package for demonstration purposes — not real spend from a real
business. It is a *different kind* of simulated data from this project's
own `simulate/dgp.py`: Robyn's publishers built it to resemble real
marketing data at a level of detail/realism this project has no way to
verify independently, whereas `simulate/dgp.py`'s ground truth is fully
known because this project wrote the generating process itself. Treated
here as a reference spine to attach real Tier 3 covariates to — not as
"real spend data," which it is not.

Source: https://github.com/facebookexperimental/Robyn (R/data/dt_simulated_weekly.RData).
Distributed as an R `.RData` binary; parsed here via `pyreadr` without
needing R installed. 208 weeks, 2015-11-23 to 2019-11-11. Channels: tv,
ooh (out-of-home), print, facebook, search (spend columns `tv_S`, `ooh_S`,
`print_S`, `facebook_S`, `search_S`).

Run: python -m ingest.robyn_reference
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "robyn"
OUT_DIR = REPO_ROOT / "data" / "processed"

RDATA_URL = "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/R/data/dt_simulated_weekly.RData"

CHANNEL_SPEND_COLUMNS = {
    "tv_S": "tv",
    "ooh_S": "ooh",
    "print_S": "print",
    "facebook_S": "facebook",
    "search_S": "search",
}


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "dt_simulated_weekly.RData"
    if not dest.exists():
        resp = requests.get(RDATA_URL, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def load() -> pd.DataFrame:
    path = download()
    result = pyreadr.read_r(str(path))
    df = result["dt_simulated_weekly"]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.rename(columns={"DATE": "date", **CHANNEL_SPEND_COLUMNS})
    df = df.sort_values("date").reset_index(drop=True)
    df["t"] = range(len(df))
    return df


def channels() -> list[str]:
    return list(CHANNEL_SPEND_COLUMNS.values())


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    out_path = OUT_DIR / "robyn_reference_spine.csv"
    df.to_csv(out_path, index=False)
    print(f"loaded {len(df)} weeks, {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"channels: {channels()}")
    print(f"wrote {out_path}")
    return df


if __name__ == "__main__":
    df = run()
    print(df.head())
