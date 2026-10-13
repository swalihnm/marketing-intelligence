"""Weather ingestion via Open-Meteo — real data, no API key required.

Per the spec: relevant "if your category is weather-sensitive." Pulls
daily mean temperature and precipitation for a given location and
aggregates to the MMM's weekly grain. Open-Meteo's free archive API needs
no authentication.

Run: python -m ingest.weather_client --lat 51.5072 --lon -0.1276 --name london
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "raw" / "weather"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum",
        "timezone": "auto",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temperature_mean_c": daily["temperature_2m_mean"],
        "precipitation_sum_mm": daily["precipitation_sum"],
    }).set_index("date")
    return df


def to_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    weekly = daily_df.resample("W-MON").agg({
        "temperature_mean_c": "mean",
        "precipitation_sum_mm": "sum",
    })
    return weekly


def run(lat: float, lon: float, name: str, start: str = "2021-01-01", end: str | None = None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    daily = fetch_weather(lat, lon, start, end)
    weekly = to_weekly(daily)
    daily.to_csv(OUT_DIR / f"weather_{name}_daily.csv")
    weekly.to_csv(OUT_DIR / f"weather_{name}_weekly.csv")
    print(f"fetched {len(daily)} daily rows for {name} ({lat},{lon}), {daily.index.min()} to {daily.index.max()}")
    print(f"wrote {OUT_DIR / f'weather_{name}_weekly.csv'} ({len(weekly)} weekly rows)")
    return daily, weekly


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=51.5072)
    parser.add_argument("--lon", type=float, default=-0.1276)
    parser.add_argument("--name", default="london")
    parser.add_argument("--start", default="2021-01-01")
    args = parser.parse_args()
    daily, weekly = run(args.lat, args.lon, args.name, start=args.start)
    print(weekly.tail())
