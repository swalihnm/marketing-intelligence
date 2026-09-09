"""Simulates weekly revenue across many geo markets (DMA-style sub-regions
within one country) for the geo-lift experiment design.

Why simulated rather than the real UCI Online Retail II data used in the
customer economics layer: that dataset is 97%+ UK revenue by value, with
every other "country" both tiny and not a comparable retail market (a
handful of customers each) — not a usable set of matched geo markets. Real
geo-lift experiments for single-country/single-website businesses split by
sub-national region (US DMAs are the standard example) precisely because
there's no natural country-level split either; this simulates that
situation directly, with a known ground-truth injected effect, so the
experiment design can be validated the same way the MMM was (see
simulate/recovery_study.py): design it, then check whether it actually
recovers a known effect, rather than trusting it on faith.

Run: python -m experiment.geo_simulate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_MARKETS = 40
PRE_WEEKS = 26
POST_WEEKS = 8
TREATMENT_MARKET = "market_00"
TRUE_LIFT_PCT = 0.12  # true injected treatment effect, +12% of revenue


@dataclass
class GeoSimResult:
    df: pd.DataFrame
    treatment_market: str
    true_lift_pct: float
    pre_weeks: int
    post_weeks: int


def simulate_geo_markets(
    n_markets: int = N_MARKETS,
    pre_weeks: int = PRE_WEEKS,
    post_weeks: int = POST_WEEKS,
    treatment_market_idx: int = 0,
    true_lift_pct: float = TRUE_LIFT_PCT,
    seed: int = 11,
) -> GeoSimResult:
    rng = np.random.default_rng(seed)
    total_weeks = pre_weeks + post_weeks
    t = np.arange(total_weeks)

    # shared macro seasonality/trend, same shape for every market (common factor)
    common = 0.08 * np.sin(2 * np.pi * t / 52) + 0.0015 * t

    market_names = [f"market_{i:02d}" for i in range(n_markets)]
    # each market has its own size (revenue level) and its own sensitivity
    # to the shared seasonal factor, plus idiosyncratic AR(1) noise —
    # mirrors real DMA heterogeneity (bigger markets, different seasonal mix).
    base_levels = rng.uniform(8_000, 60_000, size=n_markets)
    season_sensitivity = rng.uniform(0.5, 1.5, size=n_markets)
    noise_sd_frac = rng.uniform(0.03, 0.08, size=n_markets)

    rows = []
    for i, name in enumerate(market_names):
        ar_noise = np.zeros(total_weeks)
        ar_noise[0] = rng.normal(0, noise_sd_frac[i] * base_levels[i])
        phi = 0.4
        for w in range(1, total_weeks):
            ar_noise[w] = (
                phi * ar_noise[w - 1]
                + np.sqrt(1 - phi ** 2) * rng.normal(0, noise_sd_frac[i] * base_levels[i])
            )
        revenue = base_levels[i] * (1 + season_sensitivity[i] * common) + ar_noise
        rows.append(pd.DataFrame({
            "market": name,
            "week": t,
            "period": np.where(t < pre_weeks, "pre", "post"),
            "revenue": revenue,
        }))

    df = pd.concat(rows, ignore_index=True)

    treatment_market = market_names[treatment_market_idx]
    is_treatment_post = (df["market"] == treatment_market) & (df["period"] == "post")
    df.loc[is_treatment_post, "revenue"] *= (1 + true_lift_pct)

    return GeoSimResult(
        df=df, treatment_market=treatment_market, true_lift_pct=true_lift_pct,
        pre_weeks=pre_weeks, post_weeks=post_weeks,
    )


if __name__ == "__main__":
    result = simulate_geo_markets()
    print(f"markets: {result.df['market'].nunique()}, weeks: {result.df['week'].nunique()}, "
          f"treatment: {result.treatment_market}, true lift: {result.true_lift_pct:.1%}")
    print(result.df.groupby(["market", "period"])["revenue"].mean().unstack().head())
