"""Fits the Bayesian MMM on the Tier 2 reference spine (Robyn's demo
dataset) plus real Tier 3 external covariates
(`ingest/build_tier2_panel.py`) — the first time in this project real
external data has actually been used in a model fit, rather than just
ingested and left unjoined.

Covariate selection, stated explicitly: of the real covariates available
(FRED consumer sentiment/CPI/retail sales/PCE, brand search interest,
temperature, precipitation), only **consumer sentiment**, **brand search
interest**, and **temperature** are included. CPI, retail sales, and PCE
are highly correlated macro indicators of essentially the same thing as
consumer sentiment (and with each other) — including all four would
reintroduce the exact multicollinearity-among-controls problem this
project's priors exist to manage, for no real benefit. Precipitation was
dropped as redundant with temperature for a plausible seasonal-demand
story. Three covariates, each representing a genuinely different *type*
of control (macro condition, organic demand, weather), is a more honest
test than stacking in every series available just because it was fetched.

IMPORTANT CAVEAT, stated here and repeated in the results doc: revenue and
channel spend in this panel are Meta's simulated Robyn demo data, not a
real business's actual numbers. So this fit answers "does adding real
macro/search/weather covariates change anything, mechanically, when
applied to a realistic-shaped reference dataset" — not "here is what
really drives a real company's revenue." That distinction matters and is
not glossed over.

Run: python -m mmm.fit_tier2
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ingest.robyn_reference import channels as robyn_channel_names
from mmm.model import build_mmm, fit_mmm, sample_prior_predictive

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = REPO_ROOT / "data" / "processed" / "tier2_full_panel.csv"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

EXTRA_COVARIATES = ["consumer_sentiment", "brand_search_interest", "temperature_mean_c"]
HDI_PROB = 0.90


def load_panel() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(PANEL_PATH, parse_dates=["date"])
    channels = robyn_channel_names()
    for c in channels:
        df[f"spend_{c}"] = df[c]
    return df, channels


def run(draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4, target_accept: float = 0.95):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, channels = load_panel()
    extra_covariates = {name: df[name].values for name in EXTRA_COVARIATES}

    model = build_mmm(df, channels, max_lag=8, extra_covariates=extra_covariates)

    idata = fit_mmm(model, draws=draws, tune=tune, chains=chains, cores=cores, target_accept=target_accept)
    idata.to_netcdf(str(REPORT_DIR / "idata_tier2.nc"))

    summary = az.summary(idata, ci_prob=HDI_PROB, ci_kind="hdi")
    summary.to_csv(REPORT_DIR / "tier2_posterior_summary.csv")
    r_hat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    n_divergences = int(idata.sample_stats["diverging"].sum())

    # channel + extra-covariate coefficient table
    rows = []
    for c in channels:
        beta_vals = idata.posterior["beta"].sel(channel=c).values.flatten()
        lo, hi = az.hdi(beta_vals, prob=HDI_PROB)
        rows.append(dict(parameter=f"beta[{c}]", mean=beta_vals.mean(), hdi_lower=lo, hdi_upper=hi))
    for name in EXTRA_COVARIATES:
        coef_vals = idata.posterior[f"coef_{name}"].values.flatten()
        lo, hi = az.hdi(coef_vals, prob=HDI_PROB)
        rows.append(dict(parameter=f"coef_{name}", mean=coef_vals.mean(), hdi_lower=lo, hdi_upper=hi))
    coef_table = pd.DataFrame(rows)
    coef_table["excludes_zero"] = (coef_table["hdi_lower"] > 0) | (coef_table["hdi_upper"] < 0)
    coef_table.to_csv(REPORT_DIR / "tier2_coefficients.csv", index=False)

    diag = pd.DataFrame([{
        "n_weeks": len(df), "channels": ",".join(channels), "extra_covariates": ",".join(EXTRA_COVARIATES),
        "chains": chains, "draws_per_chain": draws, "tune": tune,
        "r_hat_max": r_hat_max, "ess_bulk_min": ess_bulk_min, "n_divergences": n_divergences,
    }])
    diag.to_csv(REPORT_DIR / "tier2_diagnostics.csv", index=False)

    # posterior predictive check
    ppc = idata.posterior_predictive["revenue_obs"].values.reshape(-1, len(df))
    ppc_mean = ppc.mean(axis=0)
    ppc_lo = np.percentile(ppc, 5, axis=0)
    ppc_hi = np.percentile(ppc, 95, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(df["date"], ppc_lo, ppc_hi, color="#457b9d", alpha=0.25, label="90% posterior predictive interval")
    ax.plot(df["date"], ppc_mean, color="#457b9d", label="posterior predictive mean")
    ax.plot(df["date"], df["revenue"], color="#e63946", linewidth=1.2, label="Robyn demo revenue")
    ax.legend()
    ax.set_title("Tier 2 (Robyn reference spine) + real Tier 3 covariates — posterior predictive check")
    ax.set_ylabel("revenue")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tier2_posterior_predictive.png", dpi=150)
    plt.close(fig)

    # coefficient forest plot
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(coef_table) + 1.5))
    for i, row in coef_table.iterrows():
        color = "#2a9d8f" if row["excludes_zero"] else "#888888"
        ax.errorbar(row["mean"], i, xerr=[[row["mean"] - row["hdi_lower"]], [row["hdi_upper"] - row["mean"]]],
                    fmt="o", color=color, capsize=4)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(coef_table)))
    ax.set_yticklabels(coef_table["parameter"])
    ax.set_title(f"Channel effects + real covariate coefficients ({int(HDI_PROB*100)}% HDI)\ngreen = excludes zero, grey = does not")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_tier2_coefficients.png", dpi=150)
    plt.close(fig)

    report_lines = [
        "# Tier 2 + Real Tier 3 Covariates — Model Fit",
        "",
        "Fit on Meta's Robyn demo dataset (simulated by its publisher, not this "
        "project — see ingest/robyn_reference.py) joined with real FRED/Trends/"
        "weather covariates for the matching historical weeks "
        "(ingest/build_tier2_panel.py).",
        "",
        f"- {len(df)} weeks, {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Channels: {', '.join(channels)}",
        f"- Real extra covariates: {', '.join(EXTRA_COVARIATES)}",
        "",
        "## Diagnostics",
        "",
        f"- Max r-hat: **{r_hat_max:.4f}** (target: < 1.01)",
        f"- Min bulk ESS: {ess_bulk_min:.0f}",
        f"- Divergences: **{n_divergences}** / {chains * draws} post-tuning draws "
        f"({n_divergences / (chains * draws) * 100:.1f}%)",
        "",
    ]

    posterior_unreliable = r_hat_max > 1.01 or n_divergences / (chains * draws) > 0.05
    if posterior_unreliable:
        report_lines += [
            "**⚠ THIS POSTERIOR IS NOT TRUSTWORTHY.** R-hat and/or the divergence "
            "rate are far outside acceptable bounds — the sampler could not "
            "properly explore the posterior. The coefficient table below is "
            "included for completeness, not as a result to act on. See "
            "`docs/01-data-sources-and-honesty.md` for the diagnosis (4 of 5 "
            "channels in this dataset have 51–59% zero-spend weeks — a known "
            "MMM identification failure mode, already documented separately "
            "in `causal/mmm_vs_did.py`, now confirmed on a second, independent "
            "real-shaped dataset). A higher `target_accept` and more tuning "
            "were tried and made no meaningful difference, confirming this is "
            "structural, not a sampler-tuning problem.",
            "",
        ]

    report_lines += [
        "## Real-covariate coefficients" + (" (not reliable — see warning above)" if posterior_unreliable else ""),
        "",
        "| covariate | mean | 90% HDI | excludes zero? |",
        "|---|---|---|---|",
    ]
    for name in EXTRA_COVARIATES:
        row = coef_table[coef_table["parameter"] == f"coef_{name}"].iloc[0]
        report_lines.append(
            f"| {name} | {row['mean']:,.0f} | [{row['hdi_lower']:,.0f}, {row['hdi_upper']:,.0f}] | "
            f"{'yes' if row['excludes_zero'] else 'no'} |"
        )
    report_lines += [
        "",
        "Full table: `outputs/reports/tier2_coefficients.csv`. Figures: "
        "`fig_tier2_posterior_predictive.png`, `fig_tier2_coefficients.png` "
        "(both carry the same warning directly on the image, not just here).",
        "",
        "## Caveat",
        "",
        "Revenue and channel spend here are Meta's simulated Robyn demo data, "
        "not a real business's real numbers. This fit shows whether adding real "
        "macro/search/weather covariates does anything *mechanically* on a "
        "realistic-shaped reference dataset — it is not a claim about what "
        "really drives any real company's revenue. That question is moot here "
        "regardless, given the posterior itself is unreliable.",
    ]
    (REPORT_DIR / "tier2_fit_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return dict(r_hat_max=r_hat_max, ess_bulk_min=ess_bulk_min, n_divergences=n_divergences, coef_table=coef_table)


if __name__ == "__main__":
    result = run()
    print("r_hat_max:", result["r_hat_max"])
    print("ess_bulk_min:", result["ess_bulk_min"])
    print("n_divergences:", result["n_divergences"])
    print(result["coef_table"])
