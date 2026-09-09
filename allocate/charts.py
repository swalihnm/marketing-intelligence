"""Charts for the budget allocator: per-channel response curves with
posterior uncertainty bands (current + optimal spend marked), and the
budget sweep curve.

Run: python -m allocate.charts
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from allocate.optimizer import (
    IDATA_PATH, budget_sweep, load_posterior_draws, marginal_roi,
    optimize_allocation, response_per_draw,
)
from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"


def _response_curve_single_channel(spend_range: np.ndarray, channel_idx: int, draws: dict) -> np.ndarray:
    """Contribution of ONE channel across a range of its own spend levels,
    per posterior draw (other channels don't matter — contributions are
    additive across channels in this model)."""
    alpha = draws["alpha"][:, channel_idx]
    s = draws["s"][:, channel_idx]
    k = draws["k"][:, channel_idx]
    beta = draws["beta"][:, channel_idx]
    adstock_ss = spend_range[None, :] / (1 - alpha[:, None])
    saturated = hill(adstock_ss, s[:, None], k[:, None])
    return beta[:, None] * saturated  # (n_draws, len(spend_range))


def plot_response_curves(draws: dict, current_spend: dict, optimal_spend: dict, out_path: Path) -> None:
    channels = draws["channels"]
    fig, axes = plt.subplots(1, len(channels), figsize=(4.2 * len(channels), 4), sharey=False)
    if len(channels) == 1:
        axes = [axes]

    for i, (ax, c) in enumerate(zip(axes, channels)):
        max_x = max(current_spend[c], optimal_spend[c]) * 2.5
        spend_range = np.linspace(1.0, max_x, 100)
        curves = _response_curve_single_channel(spend_range, i, draws)
        mean_curve = curves.mean(axis=0)
        lo = np.percentile(curves, 5, axis=0)
        hi = np.percentile(curves, 95, axis=0)

        ax.fill_between(spend_range, lo, hi, color="#457b9d", alpha=0.25, label="90% interval")
        ax.plot(spend_range, mean_curve, color="#457b9d", label="posterior mean")
        ax.axvline(current_spend[c], color="grey", linestyle="--", linewidth=1, label="current spend")
        ax.axvline(optimal_spend[c], color="#e63946", linestyle="--", linewidth=1, label="optimal spend")
        ax.set_title(c)
        ax.set_xlabel("weekly spend")
        if i == 0:
            ax.set_ylabel("expected weekly channel contribution")
            ax.legend(fontsize=8)

    fig.suptitle("Channel response curves (steady-state adstock approximation) — current vs. optimal spend")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_budget_sweep(sweep_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_df["total_budget"], sweep_df["expected_response"], marker="o", color="#2a9d8f")
    for _, row in sweep_df.iterrows():
        ax.annotate(f"{row['multiplier']}x", (row["total_budget"], row["expected_response"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("total weekly budget")
    ax.set_ylabel("expected weekly incremental revenue (optimal allocation)")
    ax.set_title("Budget sweep: diminishing returns to total spend")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_marginal_roi(roi_df: pd.DataFrame, out_path: Path) -> None:
    roi_df = roi_df.sort_values("marginal_roi_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    yerr = np.array([
        roi_df["marginal_roi_mean"] - roi_df["hdi_lower"],
        roi_df["hdi_upper"] - roi_df["marginal_roi_mean"],
    ])
    ax.errorbar(roi_df["marginal_roi_mean"], roi_df["channel"], xerr=yerr, fmt="o", color="#e76f51", capsize=4)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1, label="breakeven (£1 revenue per £1 spend)")
    ax.set_xlabel("marginal ROI at current spend (revenue per additional £1)")
    ax.set_title("Marginal ROI by channel, at current spend")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_allocation_study():
    from simulate.dgp import sample_true_params, simulate as dgp_simulate

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    true = sample_true_params(n_weeks=156, spend_correlation=0.5, seed=42)
    df = dgp_simulate(true, seed=42 + 1000)
    channels = list(true.channels.keys())
    current_spend = {c: float(df[f"spend_{c}"].mean()) for c in channels}

    draws = load_posterior_draws(IDATA_PATH, channels, n_draws=300)
    total_budget = sum(current_spend.values())

    result = optimize_allocation(draws, current_spend, total_budget)
    roi = marginal_roi(draws, current_spend)
    sweep = budget_sweep(draws, current_spend)

    plot_response_curves(draws, current_spend, result["optimal_spend"], FIG_DIR / "fig_response_curves.png")
    plot_budget_sweep(sweep, FIG_DIR / "fig_budget_sweep.png")
    plot_marginal_roi(roi, FIG_DIR / "fig_marginal_roi.png")

    roi.to_csv(REPORT_DIR / "marginal_roi.csv", index=False)
    sweep.to_csv(REPORT_DIR / "budget_sweep.csv", index=False)
    pd.DataFrame([{
        "total_budget": total_budget,
        **{f"current_{c}": v for c, v in current_spend.items()},
        **{f"optimal_{c}": v for c, v in result["optimal_spend"].items()},
        "expected_gain_mean": result["expected_gain_mean"],
        "expected_gain_hdi90_lower": result["expected_gain_hdi90"][0],
        "expected_gain_hdi90_upper": result["expected_gain_hdi90"][1],
        "expected_response_current": result["expected_response_current"],
        "expected_response_optimal": result["expected_response_optimal"],
    }]).to_csv(REPORT_DIR / "allocation_recommendation.csv", index=False)

    return dict(result=result, roi=roi, sweep=sweep)


if __name__ == "__main__":
    out = run_allocation_study()
    print(out["result"])
