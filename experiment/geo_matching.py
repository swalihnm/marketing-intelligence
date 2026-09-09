"""Matched-market selection for a geo-lift design: for the treatment market,
find the control market whose pre-period revenue trend correlates most
closely with it, following the spec's instruction to "match on pre-period
revenue trend." A single best-match pair is used (rather than a weighted
synthetic-control blend of many donors) because that is what the spec
explicitly describes; `docs/03-causal-identification.md` already lists
synthetic control as a separate, not-yet-built method.

The pre-period tracking error between the matched pair (the standard
deviation of their weekly revenue gap, pre-period only) becomes the noise
term `experiment/power.py` uses for the minimum-detectable-effect
calculation, and the same matched pair is used in `experiment/analyze.py`
to test whether the design detects the known injected effect from
`experiment/geo_simulate.py`.

Run: python -m experiment.geo_matching
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.geo_simulate import simulate_geo_markets

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"


def find_best_control(df: pd.DataFrame, treatment_market: str, pre_weeks: int) -> dict:
    pre = df[df["period"] == "pre"]
    pivot = pre.pivot(index="week", columns="market", values="revenue")

    treatment_series = pivot[treatment_market]
    candidates = [c for c in pivot.columns if c != treatment_market]

    scores = []
    for c in candidates:
        control_series = pivot[c]
        corr = np.corrcoef(treatment_series, control_series)[0, 1]
        # scale the control to the treatment's level before computing tracking
        # error, since markets differ in size — what matters for matching is
        # trend shape, not absolute revenue.
        scale = treatment_series.mean() / control_series.mean()
        scaled_control = control_series * scale
        tracking_error_std = float((treatment_series - scaled_control).std())
        scores.append(dict(control=c, correlation=corr, scale=scale, tracking_error_std=tracking_error_std))

    scores_df = pd.DataFrame(scores).sort_values("correlation", ascending=False)
    best = scores_df.iloc[0].to_dict()
    return dict(best=best, all_scores=scores_df)


def _plot_matched_pair(df: pd.DataFrame, treatment_market: str, best: dict, pre_weeks: int, out_path: Path) -> None:
    pivot = df.pivot(index="week", columns="market", values="revenue")
    control = best["control"]
    scale = best["scale"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pivot.index, pivot[treatment_market], color="#e63946", label=f"treatment ({treatment_market})")
    ax.plot(pivot.index, pivot[control] * scale, color="#457b9d", label=f"matched control ({control}, scaled)")
    ax.axvline(pre_weeks - 0.5, color="grey", linestyle="--", linewidth=1, label="treatment starts")
    ax.set_xlabel("week")
    ax.set_ylabel("revenue")
    ax.set_title(
        f"Matched market pair — pre-period correlation={best['correlation']:.3f}, "
        f"tracking error (pre-period std of gap)={best['tracking_error_std']:.0f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_matching():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sim = simulate_geo_markets()
    match = find_best_control(sim.df, sim.treatment_market, sim.pre_weeks)
    match["all_scores"].to_csv(REPORT_DIR / "geo_matching_scores.csv", index=False)

    _plot_matched_pair(sim.df, sim.treatment_market, match["best"], sim.pre_weeks, FIG_DIR / "fig_geo_matched_pair.png")

    return sim, match


if __name__ == "__main__":
    sim, match = run_matching()
    print("treatment market:", sim.treatment_market, "true lift:", sim.true_lift_pct)
    print("best match:", match["best"])
    print("\ntop 5 candidates:\n", match["all_scores"].head())
