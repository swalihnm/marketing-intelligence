"""Ground-truth data generating process for marketing mix modelling.

This module simulates weekly revenue data where every structural parameter
(channel elasticities, adstock decay, saturation curves, seasonality, trend,
price effect, and cross-channel spend correlation) is known in advance. It
exists so `mmm/model.py` can be validated by asking a direct question: given
data generated under known parameters, does the Bayesian model's posterior
recover them?

Nothing here is fit to real data. See docs/01-data-sources-and-honesty.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

DEFAULT_CHANNELS = ["tv", "search", "social", "display"]

# Illustrative "channel archetype" priors used both to generate ground truth
# and to inform the model's priors in mmm/priors.py. TV/OOH-style channels
# carry effects for longer (slow decay); lower-funnel digital channels decay
# fast. This mirrors real MMM practice of setting informative decay priors
# by channel type rather than leaving alpha unconstrained.
CHANNEL_ARCHETYPES = {
    "tv": dict(decay="slow", mean_weekly_spend=60_000, spend_sigma=0.35),
    "search": dict(decay="fast", mean_weekly_spend=35_000, spend_sigma=0.30),
    "social": dict(decay="fast", mean_weekly_spend=25_000, spend_sigma=0.40),
    "display": dict(decay="medium", mean_weekly_spend=20_000, spend_sigma=0.45),
}

DECAY_RANGES = {
    "slow": (0.55, 0.80),
    "medium": (0.35, 0.55),
    "fast": (0.10, 0.35),
}


@dataclass
class ChannelParams:
    name: str
    beta: float       # true incremental revenue scale
    alpha: float       # geometric adstock decay rate, in [0, 1)
    s: float           # hill shape parameter
    k: float           # hill half-saturation point (in adstocked-spend units)
    mean_weekly_spend: float
    spend_sigma: float


@dataclass
class TrueParams:
    channels: dict[str, ChannelParams]
    intercept: float
    trend_coef: float
    season_coefs: dict[str, float]  # a1, b1, a2, b2 (annual + semi-annual Fourier)
    price_coef: float
    price_baseline: float
    noise_sd: float
    spend_correlation: float
    n_weeks: int
    max_lag: int = 8


def _draw_channel_params(rng: np.random.Generator, name: str) -> ChannelParams:
    arche = CHANNEL_ARCHETYPES[name]
    lo, hi = DECAY_RANGES[arche["decay"]]
    alpha = rng.uniform(lo, hi)
    beta = rng.uniform(0.8, 2.5) * 1e5
    s = rng.uniform(1.2, 2.5)
    mean_spend = arche["mean_weekly_spend"]
    # place the half-saturation point near a plausible multiple of the
    # channel's typical *adstocked* spend, so the simulated curve actually
    # bends within the observed operating range instead of staying linear.
    approx_adstock_scale = mean_spend / (1 - alpha)
    k = rng.uniform(0.6, 1.3) * approx_adstock_scale
    return ChannelParams(
        name=name,
        beta=beta,
        alpha=alpha,
        s=s,
        k=k,
        mean_weekly_spend=mean_spend,
        spend_sigma=arche["spend_sigma"],
    )


def sample_true_params(
    n_weeks: int = 156,
    channels: list[str] | None = None,
    spend_correlation: float = 0.5,
    seed: int = 42,
) -> TrueParams:
    channels = channels or DEFAULT_CHANNELS
    rng = np.random.default_rng(seed)
    channel_params = {c: _draw_channel_params(rng, c) for c in channels}
    return TrueParams(
        channels=channel_params,
        intercept=float(rng.uniform(1.5e5, 2.5e5)),
        trend_coef=float(rng.uniform(50, 300)),
        season_coefs=dict(
            a1=float(rng.uniform(2e4, 5e4)),
            b1=float(rng.uniform(1e4, 3e4)),
            a2=float(rng.uniform(0.5e4, 1.5e4)),
            b2=float(rng.uniform(0.5e4, 1.5e4)),
        ),
        price_coef=float(rng.uniform(-4e4, -1e4)),
        price_baseline=100.0,
        noise_sd=float(rng.uniform(1.5e4, 3e4)),
        spend_correlation=spend_correlation,
        n_weeks=n_weeks,
    )


def _simulate_correlated_spend(
    rng: np.random.Generator,
    true: TrueParams,
) -> dict[str, np.ndarray]:
    """Generate weekly spend per channel with a shared 'budget cycle' factor.

    Each channel's log-spend is a mix of a common AR(1) latent factor (shared
    marketing budget cycle) and an idiosyncratic AR(1) component. The mixing
    weight is sqrt(spend_correlation), so higher `spend_correlation` produces
    more collinear spend across channels — the condition that makes MMM hard
    and is the point of the correlation-sensitivity study.
    """
    T = true.n_weeks
    rho = true.spend_correlation
    phi = 0.6  # AR(1) persistence for both the shared and idiosyncratic factors

    def ar1(n, phi, rng):
        x = np.zeros(n)
        x[0] = rng.normal()
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal()
        return x

    latent = ar1(T, phi, rng)
    spend = {}
    for name, cp in true.channels.items():
        idio = ar1(T, phi, rng)
        shared_component = np.sqrt(rho) * latent + np.sqrt(1 - rho) * idio
        log_spend = (
            np.log(cp.mean_weekly_spend)
            + cp.spend_sigma * shared_component
        )
        spend[name] = np.exp(log_spend)
    return spend


def simulate(true: TrueParams, seed: int = 123) -> pd.DataFrame:
    """Generate the panel implied by `true`, including per-channel true
    contributions so downstream code can check decomposition, not just the
    final revenue series.
    """
    rng = np.random.default_rng(seed)
    T = true.n_weeks
    t = np.arange(T)

    spend = _simulate_correlated_spend(rng, true)

    trend = true.trend_coef * t
    season = (
        true.season_coefs["a1"] * np.sin(2 * np.pi * t / 52)
        + true.season_coefs["b1"] * np.cos(2 * np.pi * t / 52)
        + true.season_coefs["a2"] * np.sin(4 * np.pi * t / 52)
        + true.season_coefs["b2"] * np.cos(4 * np.pi * t / 52)
    )
    price = true.price_baseline + rng.normal(0, 3.0, size=T).cumsum() * 0.05
    price = np.clip(price, 70, 140)
    price_effect = true.price_coef * (price - true.price_baseline)

    contributions = {}
    adstocked = {}
    for name, cp in true.channels.items():
        ad = geometric_adstock_numpy(spend[name], cp.alpha, max_lag=true.max_lag)
        adstocked[name] = ad
        contributions[name] = cp.beta * hill(ad, cp.s, cp.k)

    noise = rng.normal(0, true.noise_sd, size=T)

    revenue = (
        true.intercept
        + trend
        + season
        + price_effect
        + sum(contributions.values())
        + noise
    )

    df = pd.DataFrame({"t": t, "revenue": revenue, "price": price, "trend": trend, "season": season})
    for name in true.channels:
        df[f"spend_{name}"] = spend[name]
        df[f"adstock_{name}"] = adstocked[name]
        df[f"contribution_{name}"] = contributions[name]

    start = pd.Timestamp("2021-01-04")  # a Monday
    df["date"] = start + pd.to_timedelta(df["t"] * 7, unit="D")
    return df


def true_params_to_frame(true: TrueParams) -> pd.DataFrame:
    rows = []
    for name, cp in true.channels.items():
        rows.append(dict(channel=name, param="beta", true_value=cp.beta))
        rows.append(dict(channel=name, param="alpha", true_value=cp.alpha))
        rows.append(dict(channel=name, param="s", true_value=cp.s))
        rows.append(dict(channel=name, param="k", true_value=cp.k))
    rows.append(dict(channel=None, param="intercept", true_value=true.intercept))
    rows.append(dict(channel=None, param="trend_coef", true_value=true.trend_coef))
    for k, v in true.season_coefs.items():
        rows.append(dict(channel=None, param=f"season_{k}", true_value=v))
    rows.append(dict(channel=None, param="price_coef", true_value=true.price_coef))
    rows.append(dict(channel=None, param="noise_sd", true_value=true.noise_sd))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    true = sample_true_params()
    df = simulate(true)
    print(df.head())
    print(true_params_to_frame(true))
