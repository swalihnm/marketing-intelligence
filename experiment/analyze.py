"""Runs the actual geo-lift analysis on the simulated post-period data and
checks whether the design detects the known injected effect
(`experiment/geo_simulate.py::TRUE_LIFT_PCT`) — the same validate-before-
trusting move as `simulate/recovery_study.py` for the MMM: design an
experiment, then prove the design would actually work before recommending
anyone run it for real.

Estimator: difference-in-differences on the matched pair — the post-period
average (treatment - scaled control) gap, minus the pre-period average gap
(to net out any baseline offset the matching didn't fully close).

Run: python -m experiment.analyze
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from experiment.geo_matching import run_matching
from experiment.power import minimum_detectable_effect, mde_curve

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
FIG_DIR = REPO_ROOT / "outputs" / "figures"


def run_analysis(alpha: float = 0.05, power: float = 0.8):
    sim, match = run_matching()
    best = match["best"]
    control, scale = best["control"], best["scale"]

    pivot = sim.df.pivot(index="week", columns="market", values="revenue")
    gap = pivot[sim.treatment_market] - pivot[control] * scale

    pre_gap = gap.iloc[:sim.pre_weeks]
    post_gap = gap.iloc[sim.pre_weeks:]

    did_estimate = float(post_gap.mean() - pre_gap.mean())
    tracking_error_std = float(pre_gap.std())
    se = tracking_error_std / np.sqrt(sim.post_weeks)
    z_stat = did_estimate / se
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    significant = p_value < alpha

    baseline_mean = float(pivot[sim.treatment_market].iloc[:sim.pre_weeks].mean())
    estimated_lift_pct = did_estimate / baseline_mean
    true_lift_abs = sim.true_lift_pct * baseline_mean

    mde_at_planned_duration = minimum_detectable_effect(tracking_error_std, sim.post_weeks, alpha, power)
    mde_pct_at_planned_duration = mde_at_planned_duration / baseline_mean

    from experiment.power import required_duration_weeks
    duration_needed_for_true_effect = required_duration_weeks(true_lift_abs, tracking_error_std, alpha, power)

    results = dict(
        treatment_market=sim.treatment_market,
        control_market=control,
        pre_period_correlation=best["correlation"],
        tracking_error_std=tracking_error_std,
        true_lift_pct=sim.true_lift_pct,
        true_lift_abs=true_lift_abs,
        did_estimate_abs=did_estimate,
        estimated_lift_pct=estimated_lift_pct,
        z_stat=z_stat,
        p_value=p_value,
        significant_at_alpha=significant,
        planned_post_weeks=sim.post_weeks,
        mde_abs_at_planned_duration=mde_at_planned_duration,
        mde_pct_at_planned_duration=mde_pct_at_planned_duration,
        duration_weeks_needed_for_true_effect=duration_needed_for_true_effect,
    )

    pd.DataFrame([results]).to_csv(REPORT_DIR / "geo_experiment_analysis.csv", index=False)

    # MDE-vs-duration curve — the chart a stakeholder actually needs to see
    weeks_range = np.arange(2, 40)
    mde_pcts = mde_curve(tracking_error_std, weeks_range, alpha, power) / baseline_mean

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(weeks_range, mde_pcts * 100, color="#457b9d")
    ax.axhline(sim.true_lift_pct * 100, color="red", linestyle="--", label=f"true effect ({sim.true_lift_pct:.0%})")
    ax.axvline(sim.post_weeks, color="grey", linestyle=":", label=f"planned duration ({sim.post_weeks} weeks)")
    ax.set_xlabel("post-period duration (weeks)")
    ax.set_ylabel("minimum detectable effect (%)")
    ax.set_title(f"MDE vs. test duration (alpha={alpha}, power={power})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_geo_mde_curve.png", dpi=150)
    plt.close(fig)

    return results


if __name__ == "__main__":
    results = run_analysis()
    for k, v in results.items():
        print(f"{k}: {v}")
