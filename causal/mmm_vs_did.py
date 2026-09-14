"""MMM estimate vs. quasi-experimental (DiD) estimate, for the same
simulated channel-launch effect.

The spec calls this a senior-level move: "here is my MMM estimate, here is
a quasi-experimental estimate of the same effect, here is why they differ."
`causal/did.py` and `simulate/dgp.py` were, until this module, independently
parameterized simulations — no single dataset fed both an MMM fit and a
DiD estimate for literally the same effect. This closes that gap.

Design: reuse `causal.did.simulate_two_region_launch`'s two-region panel
(region A gets the channel launch at a known week, region B is a pure
control that never gets it, both share a regional trend/seasonality
confound). Fit a single-channel MMM (`mmm/model.py`, same model class used
for the main 4-channel recovery study) on region A's data **alone** — the
way an MMM is actually run in practice: one time series, no control
region. Compare its estimated average post-launch contribution against
the DiD estimate (which *does* use region B as control) and the known true
effect.

Why they should differ: the MMM only has trend + 2-harmonic seasonality to
control for the regional confound. If those terms don't perfectly match
the true confound's shape (they don't here — the confound is a linear
trend plus a single sine term, which the model's terms can capture almost
exactly, so this is a relatively favorable case for the MMM), any leakage
biases the estimated beta. DiD sidesteps the shape-matching problem
entirely by using the actual observed control region instead of a
parametric proxy for "what would have happened anyway."

Run: python -m causal.mmm_vs_did
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal.did import estimate_did, simulate_two_region_launch
from mmm.model import build_mmm, fit_mmm
from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"


def fit_mmm_on_region_a(df: pd.DataFrame, launch_week: int, channel: str = "launch",
                         draws: int = 600, tune: int = 600, chains: int = 4, cores: int = 4):
    region_a = df[df["region"] == "A_launch"].reset_index(drop=True).copy()
    region_a = region_a.rename(columns={"spend": f"spend_{channel}"})

    model = build_mmm(region_a, channels=[channel], max_lag=8)
    idata = fit_mmm(model, draws=draws, tune=tune, chains=chains, cores=cores, target_accept=0.95)

    beta = float(idata.posterior["beta"].sel(channel=channel).values.mean())
    alpha = float(idata.posterior["alpha"].sel(channel=channel).values.mean())
    s = float(idata.posterior["hill_s"].sel(channel=channel).values.mean())
    k = float(idata.posterior["hill_k"].sel(channel=channel).values.mean())
    r_hat_max = float(
        __import__("arviz").summary(idata, ci_prob=0.9, ci_kind="hdi")["r_hat"].max()
    )
    n_divergences = int(idata.sample_stats["diverging"].sum())

    spend = region_a[f"spend_{channel}"].values
    adstocked = geometric_adstock_numpy(spend, alpha, max_lag=8)
    mmm_contribution = beta * hill(adstocked, s, k)

    week = region_a["week"].values
    mmm_avg_post_contribution = float(mmm_contribution[week >= launch_week].mean())

    return dict(
        beta=beta, alpha=alpha, s=s, k=k,
        r_hat_max=r_hat_max, n_divergences=n_divergences,
        mmm_avg_post_contribution=mmm_avg_post_contribution,
    )


def run_reconciliation(launch_week: int = 26):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, true_effect = simulate_two_region_launch(launch_week=launch_week)
    did_result = estimate_did(df)
    mmm_result = fit_mmm_on_region_a(df, launch_week=launch_week)

    result = dict(
        true_effect=true_effect,
        did_estimate=did_result["did_estimate"],
        did_pct_error=abs(did_result["did_estimate"] - true_effect) / true_effect * 100,
        mmm_estimate=mmm_result["mmm_avg_post_contribution"],
        mmm_pct_error=abs(mmm_result["mmm_avg_post_contribution"] - true_effect) / true_effect * 100,
        mmm_r_hat_max=mmm_result["r_hat_max"],
        mmm_n_divergences=mmm_result["n_divergences"],
    )
    pd.DataFrame([result]).to_csv(REPORT_DIR / "mmm_vs_did_reconciliation.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["True effect", "MMM\n(region A alone,\ntrend+season control)", "DiD\n(region B as control)"]
    values = [true_effect, result["mmm_estimate"], result["did_estimate"]]
    colors = ["#264653", "#e9c46a", "#2a9d8f"]
    ax.bar(methods, values, color=colors)
    ax.axhline(true_effect, color="#264653", linestyle="--", linewidth=1)
    ax.set_ylabel("average post-launch weekly contribution")
    ax.set_title(
        f"MMM vs. DiD, same effect: MMM {result['mmm_pct_error']:.0f}% off, "
        f"DiD {result['did_pct_error']:.0f}% off"
    )
    ax.text(
        0.5, -0.32,
        f"WARNING: the MMM bar looks close to true by coincidence — its posterior is NOT trustworthy here\n"
        f"(r-hat={result['mmm_r_hat_max']:.2f}, {result['mmm_n_divergences']} divergences out of 2,400 draws). "
        f"See docs/03-causal-identification.md.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#e63946", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_mmm_vs_did.png", dpi=150)
    plt.close(fig)

    return result


if __name__ == "__main__":
    result = run_reconciliation()
    for k, v in result.items():
        print(f"{k}: {v}")
