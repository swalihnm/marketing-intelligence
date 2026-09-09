"""Parameter recovery study: the headline validation for this project.

Real multi-channel marketing spend data is proprietary (see
docs/01-data-sources-and-honesty.md), so there is no ground truth to check
the MMM against when it runs on real data. This script substitutes the next
best thing: generate data under KNOWN parameters (simulate/dgp.py), fit the
model (mmm/model.py) to it exactly as it would be fit to real data, and
check whether the posterior recovers the parameters that generated the data.

Outputs (outputs/figures/, outputs/reports/):
  - recovery_report.md            human-readable summary
  - recovery_table.csv            true vs posterior mean/HDI/coverage, per parameter
  - diagnostics_summary.csv       r_hat / ess / divergences
  - fig_recovery_beta.png / _alpha.png / _hill_s.png / _hill_k.png
  - fig_posterior_predictive.png
  - fig_prior_predictive.png
  - fig_trace.png
  - idata_recovery.nc             full InferenceData, for reuse

Run: python -m simulate.recovery_study
"""
from __future__ import annotations

import time
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate.dgp import sample_true_params, simulate, true_params_to_frame
from mmm.model import build_mmm, fit_mmm, sample_prior_predictive

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

HDI_PROB = 0.90


def _forest_recovery_plot(idata, true_df, param: str, channels: list[str], out_path: Path):
    post = idata.posterior[param]  # dims: chain, draw, channel
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(channels) + 1.5))
    for i, c in enumerate(channels):
        vals = post.sel(channel=c).values.flatten()
        mean = vals.mean()
        lo, hi = az.hdi(vals, prob=HDI_PROB)
        true_param = "s" if param == "hill_s" else ("k" if param == "hill_k" else param)
        true_val = true_df.loc[(true_df.channel == c) & (true_df.param == true_param), "true_value"].iloc[0]
        covered = lo <= true_val <= hi
        color = "#2a9d8f" if covered else "#e76f51"
        ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, capsize=4)
        ax.scatter([true_val], [i], marker="x", color="black", s=80, zorder=5)
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_title(f"Parameter recovery: {param}  (dot+bar = posterior mean & {int(HDI_PROB*100)}% HDI, x = true)")
    ax.set_xlabel(param)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_recovery_study(
    n_weeks: int = 156,
    spend_correlation: float = 0.5,
    seed: int = 42,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.95,
):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    true = sample_true_params(n_weeks=n_weeks, spend_correlation=spend_correlation, seed=seed)
    df = simulate(true, seed=seed + 1000)
    channels = list(true.channels.keys())
    true_df = true_params_to_frame(true)

    model = build_mmm(df, channels, max_lag=true.max_lag)

    # --- prior predictive check: do the priors alone imply plausible revenue? ---
    prior_idata = sample_prior_predictive(model, samples=500)
    prior_rev = prior_idata.prior_predictive["revenue_obs"].values.reshape(-1, len(df))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(min(80, prior_rev.shape[0])):
        ax.plot(df["date"], prior_rev[i], color="grey", alpha=0.08)
    ax.plot(df["date"], df["revenue"], color="#264653", linewidth=1.5, label="observed (simulated) revenue")
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.set_title("Prior predictive check: priors generate plausible revenue before seeing data")
    ax.set_ylabel("revenue")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_prior_predictive.png", dpi=150)
    plt.close(fig)

    # --- fit ---
    t0 = time.time()
    idata = fit_mmm(model, draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept)
    elapsed = time.time() - t0

    try:
        idata.to_netcdf(str(REPORT_DIR / "idata_recovery.nc"))
    except Exception as e:  # optional artifact; don't block the rest of the report
        print(f"warning: could not save idata netcdf ({e})")

    # --- diagnostics ---
    summary = az.summary(idata, ci_prob=HDI_PROB, ci_kind="hdi")
    summary.to_csv(REPORT_DIR / "posterior_summary.csv")

    r_hat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    n_divergences = int(idata.sample_stats["diverging"].sum())

    diag = pd.DataFrame([{
        "n_weeks": n_weeks,
        "spend_correlation": spend_correlation,
        "chains": chains,
        "draws_per_chain": draws,
        "tune": tune,
        "elapsed_seconds": round(elapsed, 1),
        "r_hat_max": r_hat_max,
        "ess_bulk_min": ess_bulk_min,
        "n_divergences": n_divergences,
    }])
    diag.to_csv(REPORT_DIR / "diagnostics_summary.csv", index=False)

    # --- recovery table (per-channel + global params) ---
    rows = []
    for param in ["beta", "alpha", "hill_s", "hill_k"]:
        post = idata.posterior[param]
        for c in channels:
            vals = post.sel(channel=c).values.flatten()
            mean = vals.mean()
            lo, hi = az.hdi(vals, prob=HDI_PROB)
            true_param = "s" if param == "hill_s" else ("k" if param == "hill_k" else param)
            true_val = true_df.loc[
                (true_df.channel == c) & (true_df.param == true_param), "true_value"
            ].iloc[0]
            rows.append(dict(
                channel=c, param=param, true_value=true_val,
                posterior_mean=mean, hdi_lower=lo, hdi_upper=hi,
                covered=bool(lo <= true_val <= hi),
                pct_error=abs(mean - true_val) / abs(true_val) * 100,
            ))
    for param, true_param in [
        ("intercept", "intercept"), ("trend_coef", "trend_coef"),
        ("season_a1", "season_a1"), ("season_b1", "season_b1"),
        ("season_a2", "season_a2"), ("season_b2", "season_b2"),
        ("price_coef", "price_coef"), ("sigma", "noise_sd"),
    ]:
        vals = idata.posterior[param].values.flatten()
        mean = vals.mean()
        lo, hi = az.hdi(vals, prob=HDI_PROB)
        true_val = true_df.loc[true_df.param == true_param, "true_value"].iloc[0]
        rows.append(dict(
            channel=None, param=param, true_value=true_val,
            posterior_mean=mean, hdi_lower=lo, hdi_upper=hi,
            covered=bool(lo <= true_val <= hi),
            pct_error=abs(mean - true_val) / abs(true_val) * 100,
        ))
    recovery_table = pd.DataFrame(rows)
    recovery_table.to_csv(REPORT_DIR / "recovery_table.csv", index=False)

    # --- recovery plots per parameter family ---
    for param in ["beta", "alpha", "hill_s", "hill_k"]:
        _forest_recovery_plot(idata, true_df, param, channels, FIG_DIR / f"fig_recovery_{param}.png")

    # --- posterior predictive check ---
    ppc = idata.posterior_predictive["revenue_obs"].values.reshape(-1, len(df))
    ppc_mean = ppc.mean(axis=0)
    ppc_lo = np.percentile(ppc, 5, axis=0)
    ppc_hi = np.percentile(ppc, 95, axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(df["date"], ppc_lo, ppc_hi, color="#457b9d", alpha=0.25, label="90% posterior predictive interval")
    ax.plot(df["date"], ppc_mean, color="#457b9d", label="posterior predictive mean")
    ax.plot(df["date"], df["revenue"], color="#e63946", linewidth=1.2, label="observed (simulated) revenue")
    ax.legend()
    ax.set_title("Posterior predictive check")
    ax.set_ylabel("revenue")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_posterior_predictive.png", dpi=150)
    plt.close(fig)

    # --- trace plot (subset of params to keep it legible) ---
    az.plot_trace(idata, var_names=["beta", "alpha", "hill_s", "hill_k"])
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_trace.png", dpi=120)
    plt.close(fig)

    # --- report ---
    coverage_rate = recovery_table["covered"].mean() * 100
    mean_pct_error_beta = recovery_table.loc[recovery_table.param == "beta", "pct_error"].mean()
    mean_pct_error_alpha = recovery_table.loc[recovery_table.param == "alpha", "pct_error"].mean()

    report_lines = [
        "# Parameter Recovery Study",
        "",
        f"Simulated {n_weeks} weeks, {len(channels)} channels "
        f"({', '.join(channels)}), spend correlation = {spend_correlation}, seed = {seed}.",
        "",
        "## Sampling diagnostics",
        "",
        f"- Elapsed: {elapsed:.1f}s ({chains} chains x {cores} cores, {tune} tune / {draws} draws)",
        f"- Max r_hat across all parameters: **{r_hat_max:.4f}** (target: < 1.01)",
        f"- Min bulk ESS across all parameters: **{ess_bulk_min:.0f}**",
        f"- Divergences: **{n_divergences}**",
        "",
        "## Recovery result",
        "",
        f"- {coverage_rate:.1f}% of parameters had their true value inside the "
        f"{int(HDI_PROB*100)}% posterior HDI ({recovery_table['covered'].sum()}/{len(recovery_table)}).",
        f"- Mean absolute % error, beta (channel elasticity scale): {mean_pct_error_beta:.1f}%",
        f"- Mean absolute % error, alpha (adstock decay): {mean_pct_error_alpha:.1f}%",
        "",
        "Full per-parameter numbers: `outputs/reports/recovery_table.csv`.",
        "",
        "## Figures",
        "",
        "- `fig_prior_predictive.png` — priors alone imply a plausible revenue range before fitting",
        "- `fig_recovery_beta.png`, `fig_recovery_alpha.png`, `fig_recovery_hill_s.png`, `fig_recovery_hill_k.png`"
        " — posterior mean + HDI (dot/bar) vs true value (x) per channel; red = HDI missed the true value",
        "- `fig_posterior_predictive.png` — fitted model against the (simulated) observed series",
        "- `fig_trace.png` — MCMC traces for the channel-level parameters",
        "",
        "## Interpretation",
        "",
        "Coverage at or above the nominal HDI probability, with r_hat below 1.01 and no "
        "divergences, is evidence the sampler is well-behaved and the model's functional form "
        "(geometric adstock + Hill saturation) is estimable from data of this size and "
        "collinearity. It does **not** prove the model works on real spend data — only that the "
        "estimator is unbiased under the assumptions encoded in the DGP. Any channel with 'red' "
        "coverage above should be treated as a case study in identification failure, not ignored.",
    ]
    (REPORT_DIR / "recovery_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return dict(
        elapsed=elapsed, r_hat_max=r_hat_max, ess_bulk_min=ess_bulk_min,
        n_divergences=n_divergences, coverage_rate=coverage_rate,
        recovery_table=recovery_table,
    )


if __name__ == "__main__":
    result = run_recovery_study()
    print("elapsed", result["elapsed"])
    print("r_hat_max", result["r_hat_max"])
    print("ess_bulk_min", result["ess_bulk_min"])
    print("n_divergences", result["n_divergences"])
    print("coverage_rate", result["coverage_rate"])
    print(result["recovery_table"])
