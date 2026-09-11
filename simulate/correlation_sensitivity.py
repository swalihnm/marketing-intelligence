"""Correlation-sensitivity study: how does parameter recovery degrade as
cross-channel spend correlation rises? The spec calls this "an excellent
interview artefact" and it's a natural follow-on to
`simulate/recovery_study.py` — same validation logic, swept across the
`spend_correlation` parameter that controls how collinear channel spend is
(see `simulate/dgp.py::_simulate_correlated_spend`).

Deliberately lighter-weight than the full recovery study per run (no prior
predictive check, no trace/PPC plots, no written report) — this needs 4+
full model fits, and the headline recovery metrics (HDI coverage, mean %
error on beta/alpha) are all that's needed to see the degradation trend.

Run: python -m simulate.correlation_sensitivity
"""
from __future__ import annotations

import time
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mmm.model import build_mmm, fit_mmm
from simulate.dgp import sample_true_params, simulate, true_params_to_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

HDI_PROB = 0.90
CORRELATIONS = [0.3, 0.5, 0.7, 0.9]


def fit_and_score(
    spend_correlation: float,
    n_weeks: int = 156,
    seed: int = 42,
    draws: int = 800,
    tune: int = 800,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.95,
) -> dict:
    true = sample_true_params(n_weeks=n_weeks, spend_correlation=spend_correlation, seed=seed)
    df = simulate(true, seed=seed + 1000)
    channels = list(true.channels.keys())
    true_df = true_params_to_frame(true)

    model = build_mmm(df, channels, max_lag=true.max_lag)

    t0 = time.time()
    idata = fit_mmm(model, draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept)
    elapsed = time.time() - t0

    summary = az.summary(idata, ci_prob=HDI_PROB, ci_kind="hdi")
    r_hat_max = float(summary["r_hat"].max())
    n_divergences = int(idata.sample_stats["diverging"].sum())

    rows = []
    for param in ["beta", "alpha", "hill_s", "hill_k"]:
        post = idata.posterior[param]
        true_param = "s" if param == "hill_s" else ("k" if param == "hill_k" else param)
        for c in channels:
            vals = post.sel(channel=c).values.flatten()
            mean = vals.mean()
            lo, hi = az.hdi(vals, prob=HDI_PROB)
            true_val = true_df.loc[(true_df.channel == c) & (true_df.param == true_param), "true_value"].iloc[0]
            rows.append(dict(
                param=param, channel=c, true_value=true_val, posterior_mean=mean,
                covered=bool(lo <= true_val <= hi),
                pct_error=abs(mean - true_val) / abs(true_val) * 100,
            ))
    per_param = pd.DataFrame(rows)

    headline = per_param[per_param.param.isin(["beta", "alpha", "hill_s", "hill_k"])]
    return dict(
        spend_correlation=spend_correlation,
        elapsed_seconds=round(elapsed, 1),
        r_hat_max=r_hat_max,
        n_divergences=n_divergences,
        headline_coverage_pct=float(headline["covered"].mean() * 100),
        mean_pct_error_beta=float(per_param.loc[per_param.param == "beta", "pct_error"].mean()),
        mean_pct_error_alpha=float(per_param.loc[per_param.param == "alpha", "pct_error"].mean()),
        mean_pct_error_hill_s=float(per_param.loc[per_param.param == "hill_s", "pct_error"].mean()),
        mean_pct_error_hill_k=float(per_param.loc[per_param.param == "hill_k", "pct_error"].mean()),
    )


def run_sweep(correlations=CORRELATIONS, **fit_kwargs) -> pd.DataFrame:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for rho in correlations:
        print(f"--- fitting at spend_correlation={rho} ---", flush=True)
        result = fit_and_score(rho, **fit_kwargs)
        print(result, flush=True)
        rows.append(result)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(REPORT_DIR / "correlation_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    ax.plot(results_df["spend_correlation"], results_df["headline_coverage_pct"], marker="o", color="#2a9d8f")
    ax.axhline(90, color="grey", linestyle="--", linewidth=1, label="nominal 90% HDI target")
    ax.set_xlabel("cross-channel spend correlation")
    ax.set_ylabel("headline parameter HDI coverage (%)")
    ax.set_title("Coverage vs. spend correlation")
    ax.set_ylim(0, 105)
    ax.legend()

    ax2 = axes[1]
    for col, label, color in [
        ("mean_pct_error_beta", "beta", "#e63946"),
        ("mean_pct_error_alpha", "alpha", "#457b9d"),
        ("mean_pct_error_hill_s", "hill_s", "#f4a261"),
        ("mean_pct_error_hill_k", "hill_k", "#2a9d8f"),
    ]:
        ax2.plot(results_df["spend_correlation"], results_df[col], marker="o", label=label, color=color)
    ax2.set_xlabel("cross-channel spend correlation")
    ax2.set_ylabel("mean absolute % error")
    ax2.set_title("Parameter recovery error vs. spend correlation")
    ax2.legend()

    fig.suptitle("Identification degrades as cross-channel spend correlation rises")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_correlation_sensitivity.png", dpi=150)
    plt.close(fig)

    report_lines = [
        "# Correlation-Sensitivity Study",
        "",
        "How parameter recovery degrades as cross-channel spend correlation rises "
        "from 0.3 to 0.9 — the multicollinearity problem that makes MMM hard, "
        "and the reason `mmm/priors.py` uses informative priors rather than flat ones.",
        "",
        "| spend correlation | headline coverage | mean % error (beta) | mean % error (alpha) | r_hat max | divergences |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in results_df.iterrows():
        report_lines.append(
            f"| {r['spend_correlation']} | {r['headline_coverage_pct']:.1f}% | "
            f"{r['mean_pct_error_beta']:.1f}% | {r['mean_pct_error_alpha']:.1f}% | "
            f"{r['r_hat_max']:.4f} | {int(r['n_divergences'])} |"
        )
    report_lines += [
        "",
        "Full numbers: `outputs/reports/correlation_sensitivity.csv`. "
        "Figure: `outputs/figures/fig_correlation_sensitivity.png`.",
    ]
    (REPORT_DIR / "correlation_sensitivity_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return results_df


if __name__ == "__main__":
    df = run_sweep()
    print(df)
