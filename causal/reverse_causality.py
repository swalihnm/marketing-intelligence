"""Reverse causality: budget set as a percentage of forecast revenue.

The spec names this the single biggest identification threat to MMM
coefficients, and until this module, it was explicitly documented (see
docs/03-causal-identification.md) as unaddressed anywhere in this codebase.
This closes that gap with a known-ground-truth simulation, following the
same validate-before-trusting pattern as the rest of this project.

Deliberately uses a LINEAR spend-revenue relationship (no adstock, no Hill
saturation) rather than the full mmm/model.py specification — the point
here is the endogeneity/IV mechanics, and two-stage least squares (2SLS) is
a linear method that doesn't directly apply to a nonlinear model without a
control-function or nonlinear-IV extension, which is out of scope. This is
a stated simplification, not a re-derivation of the full MMM.

Mechanism:
  - `latent_shock_t`: an unobserved (to the analyst) AR(1) market-condition
    process — the confounder. It affects revenue directly.
  - `budget_t`: set partly as a function of a noisy *forecast* of
    `latent_shock_t` (planners have some real foresight into demand
    conditions when setting budget) — this is the reverse-causality
    mechanism: budget responds to the same thing that also drives revenue,
    not just to spend's own causal effect.
  - `cpm_t`: an exogenous auction-price shock, independent of
    `latent_shock_t` — spend = budget / cpm, so cpm shifts realized spend
    for a reason unrelated to demand conditions. This is the spec's own
    suggested instrument ("a platform CPM shock").
  - `revenue_t = intercept + true_beta * spend_t + latent_shock_t + noise_t`

Naive OLS (regressing revenue on spend, as if spend were exogenous) is
biased because spend correlates with the unobserved `latent_shock_t`. 2SLS
using `cpm_t` as an instrument isolates the variation in spend driven by
the exogenous cost shock, and should recover something closer to the true
beta.

Run: python -m causal.reverse_causality
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

TRUE_BETA = 3.5  # revenue per unit spend, linear


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, sigma)
    return x


def simulate_endogenous_budget(
    n_weeks: int = 260,
    true_beta: float = TRUE_BETA,
    budget_sensitivity: float = 400.0,
    seed: int = 55,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    latent_shock = _ar1(n_weeks, phi=0.7, sigma=8000.0, rng=rng)
    forecast_noise = rng.normal(0, 3000.0, n_weeks)
    forecast_of_latent_shock = latent_shock + forecast_noise  # imperfect foresight

    base_budget = 40_000.0
    budget = base_budget + budget_sensitivity * forecast_of_latent_shock / 1000.0
    budget = np.clip(budget, 5_000.0, None)

    cpm_shock = rng.normal(0, 0.15, n_weeks)  # exogenous auction-price noise
    cpm = 1.0 * (1 + cpm_shock)
    cpm = np.clip(cpm, 0.5, 1.5)
    spend = budget / cpm  # cpm shifts realized spend, independent of latent_shock

    noise = rng.normal(0, 5_000.0, n_weeks)
    revenue = 200_000.0 + true_beta * spend + latent_shock + noise

    return pd.DataFrame({
        "week": np.arange(n_weeks), "spend": spend, "revenue": revenue,
        "cpm": cpm, "budget": budget, "latent_shock": latent_shock,
    })


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """X must already include an intercept column. Returns coefficient vector."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def naive_ols_estimate(df: pd.DataFrame) -> float:
    X = np.column_stack([np.ones(len(df)), df["spend"].values])
    coefs = _ols(df["revenue"].values, X)
    return float(coefs[1])


def iv_2sls_estimate(df: pd.DataFrame) -> float:
    """Two-stage least squares using cpm as the instrument for spend."""
    # stage 1: spend ~ intercept + cpm
    X1 = np.column_stack([np.ones(len(df)), df["cpm"].values])
    stage1_coefs = _ols(df["spend"].values, X1)
    spend_hat = X1 @ stage1_coefs

    # stage 2: revenue ~ intercept + spend_hat
    X2 = np.column_stack([np.ones(len(df)), spend_hat])
    stage2_coefs = _ols(df["revenue"].values, X2)
    return float(stage2_coefs[1])


def run_study(true_beta: float = TRUE_BETA) -> dict:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = simulate_endogenous_budget(true_beta=true_beta)

    naive = naive_ols_estimate(df)
    iv = iv_2sls_estimate(df)

    # instrument relevance check: cpm must actually predict spend
    corr_cpm_spend = float(np.corrcoef(df["cpm"], df["spend"])[0, 1])
    corr_cpm_latent = float(np.corrcoef(df["cpm"], df["latent_shock"])[0, 1])

    result = dict(
        true_beta=true_beta,
        naive_ols_estimate=naive,
        naive_pct_error=abs(naive - true_beta) / true_beta * 100,
        iv_2sls_estimate=iv,
        iv_pct_error=abs(iv - true_beta) / true_beta * 100,
        instrument_relevance_corr_cpm_spend=corr_cpm_spend,
        instrument_exclusion_check_corr_cpm_latent=corr_cpm_latent,
    )
    pd.DataFrame([result]).to_csv(REPORT_DIR / "reverse_causality_iv.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["True beta", "Naive OLS\n(spend treated as exogenous)", "IV / 2SLS\n(cpm as instrument)"]
    values = [true_beta, naive, iv]
    colors = ["#264653", "#e76f51", "#2a9d8f"]
    ax.bar(methods, values, color=colors)
    ax.axhline(true_beta, color="#264653", linestyle="--", linewidth=1)
    ax.set_ylabel("estimated beta (revenue per unit spend)")
    ax.set_title(
        f"Reverse causality: naive OLS {result['naive_pct_error']:.0f}% off, "
        f"IV {result['iv_pct_error']:.0f}% off"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reverse_causality_iv.png", dpi=150)
    plt.close(fig)

    return result


if __name__ == "__main__":
    result = run_study()
    for k, v in result.items():
        print(f"{k}: {v}")
