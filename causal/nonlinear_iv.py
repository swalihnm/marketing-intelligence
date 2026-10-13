"""Extends the reverse-causality/IV correction (`causal/reverse_causality.py`)
to the full nonlinear adstock+Hill-saturation model — the gap stated
explicitly in `docs/06-assumptions-and-limitations.md`: "2SLS is a linear
method and doesn't directly apply to a nonlinear model without a
control-function or nonlinear-IV extension, which is out of scope here."
This is that extension.

**Control function approach** (the standard way to handle endogenous
regressors in a nonlinear model, when 2SLS itself doesn't apply): instead
of replacing the endogenous variable with first-stage fitted values (which
is what 2SLS does, and which doesn't compose cleanly with the nonlinear
adstock/Hill transform), estimate the first-stage residual — the part of
spend *not* explained by the exogenous instrument — and include it as an
extra covariate in the structural (nonlinear) model. If spend is
endogenous, that residual is correlated with the unobserved confounder;
controlling for it directly purges the bias from the channel's own
coefficient, the same logic as a Durbin-Wu-Hausman test, extended here to
do correction rather than just detection.

Validated the same way everything else in this project is: simulate data
with a KNOWN true beta/alpha/s/k and a known endogeneity mechanism, then
check whether the naive nonlinear MMM is biased and whether the
control-function correction fixes it — building on `mmm/model.py`'s
`extra_covariates` parameter (added for this), not a separate model.

Unlike `causal/mmm_vs_did.py`'s finding (a step-like spend pattern leaves
adstock/saturation unidentified), spend here varies across a wide
multiplicative range by construction — genuinely enough variation to
identify the nonlinear curve, confirmed in the result: both fits sampled
cleanly (r-hat ≈ 1.00, zero divergences), unlike Tier 2's failure.

**Actual result, reported as found rather than as hoped**: the naive fit
was already close to true (well within normal sampling noise), and the
control-function correction made the estimate *worse*, not better — see
`docs/03-causal-identification.md` for the full diagnosis. Root cause,
confirmed by direct measurement, not speculation: the instrument (`cpm`)
is weak relative to the endogeneity mechanism (first-stage R² ≈ 0.33), so
the first-stage residual ends up highly correlated with spend itself
(r ≈ 0.82) rather than cleanly isolating the confound-driven component.
Adding a near-copy of the channel's own spend as a second covariate
destabilizes the nonlinear channel-effect parameters rather than
correcting them. This is a genuine, diagnosed limitation of the naive
"just add the residual" control-function approach on a nonlinear model —
not evidence that nonlinear IV correction is impossible, but evidence that
it needs a stronger instrument and/or a more principled joint estimation
approach than this module implements. The "extending IV to the nonlinear
MMM" gap is explored and diagnosed here, not fully closed.

Run: python -m causal.nonlinear_iv
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from mmm.model import build_mmm, fit_mmm
from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

TRUE_BETA = 250_000.0
TRUE_ALPHA = 0.4
TRUE_S = 2.0
TRUE_K = 60_000.0
HDI_PROB = 0.90


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, sigma)
    return x


def simulate(
    n_weeks: int = 156,
    true_beta: float = TRUE_BETA,
    true_alpha: float = TRUE_ALPHA,
    true_s: float = TRUE_S,
    true_k: float = TRUE_K,
    budget_sensitivity: float = 0.6,
    seed: int = 71,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)

    latent_shock = _ar1(n_weeks, phi=0.7, sigma=0.35, rng=rng)  # unobserved confound, on a log-ish scale
    forecast_noise = rng.normal(0, 0.15, n_weeks)
    forecast_of_latent_shock = latent_shock + forecast_noise  # imperfect foresight

    base_budget = 50_000.0
    # multiplicative response to the forecast — genuinely wide spend
    # variation (unlike a flat step function), so adstock/saturation stay
    # identifiable; this is the endogeneity mechanism (budget responds to
    # the same thing that also directly drives revenue).
    budget = base_budget * np.exp(budget_sensitivity * forecast_of_latent_shock)

    cpm_shock = rng.normal(0, 0.12, n_weeks)  # exogenous auction-price noise
    cpm = np.clip(1.0 * (1 + cpm_shock), 0.5, 1.5)
    spend = budget / cpm  # instrument-driven variation in realized spend

    adstocked = geometric_adstock_numpy(spend, true_alpha, max_lag=8)
    true_contribution = true_beta * hill(adstocked, true_s, true_k)

    noise = rng.normal(0, 15_000.0, n_weeks)
    revenue = 500_000.0 + true_contribution + 80_000.0 * latent_shock + noise

    price = 100.0 + rng.normal(0, 2.0, n_weeks)

    df = pd.DataFrame({
        "t": t, "revenue": revenue, "spend_x": spend, "cpm": cpm,
        "price": price, "latent_shock": latent_shock,
    })
    return df, float(true_contribution.mean())


def first_stage_residual(df: pd.DataFrame) -> np.ndarray:
    """Spend ~ intercept + cpm (OLS). Residual carries the variation in
    spend NOT explained by the exogenous instrument — the part plausibly
    correlated with the unobserved confound."""
    X = np.column_stack([np.ones(len(df)), df["cpm"].values])
    coefs = np.linalg.lstsq(X, df["spend_x"].values, rcond=None)[0]
    fitted = X @ coefs
    return df["spend_x"].values - fitted


def first_stage_diagnostics(df: pd.DataFrame, v_hat: np.ndarray) -> dict:
    """Instrument-strength diagnostics, computed whether or not the
    correction below ends up working — see run()'s docstring on why these
    numbers turned out to matter more than the correction itself."""
    X = np.column_stack([np.ones(len(df)), df["cpm"].values])
    coefs = np.linalg.lstsq(X, df["spend_x"].values, rcond=None)[0]
    fitted = X @ coefs
    ss_res = ((df["spend_x"].values - fitted) ** 2).sum()
    ss_tot = ((df["spend_x"].values - df["spend_x"].mean()) ** 2).sum()
    first_stage_r2 = 1 - ss_res / ss_tot
    residual_spend_corr = float(np.corrcoef(df["spend_x"], v_hat)[0, 1])
    return dict(first_stage_r2=float(first_stage_r2), residual_spend_corr=residual_spend_corr)


def _fit_and_extract(df: pd.DataFrame, extra_covariates: dict | None, draws=1000, tune=1000, chains=4, cores=4):
    model = build_mmm(df, channels=["x"], max_lag=8, extra_covariates=extra_covariates)
    idata = fit_mmm(model, draws=draws, tune=tune, chains=chains, cores=cores, target_accept=0.95)

    beta = float(idata.posterior["beta"].sel(channel="x").values.mean())
    alpha = float(idata.posterior["alpha"].sel(channel="x").values.mean())
    s = float(idata.posterior["hill_s"].sel(channel="x").values.mean())
    k = float(idata.posterior["hill_k"].sel(channel="x").values.mean())

    adstocked = geometric_adstock_numpy(df["spend_x"].values, alpha, max_lag=8)
    avg_contribution = float((beta * hill(adstocked, s, k)).mean())

    summary = az.summary(idata, ci_prob=HDI_PROB, ci_kind="hdi")
    r_hat_max = float(summary["r_hat"].max())
    n_divergences = int(idata.sample_stats["diverging"].sum())

    residual_coef = None
    if extra_covariates:
        name = list(extra_covariates.keys())[0]
        residual_coef = dict(
            mean=float(idata.posterior[f"coef_{name}"].values.mean()),
        )

    return dict(
        beta=beta, alpha=alpha, s=s, k=k, avg_contribution=avg_contribution,
        r_hat_max=r_hat_max, n_divergences=n_divergences, residual_coef=residual_coef,
    )


def run(draws: int = 1000, tune: int = 1000, chains: int = 4, cores: int = 4):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, true_avg_contribution = simulate()

    naive = _fit_and_extract(df, extra_covariates=None, draws=draws, tune=tune, chains=chains, cores=cores)

    v_hat = first_stage_residual(df)
    instrument_diag = first_stage_diagnostics(df, v_hat)
    corrected = _fit_and_extract(
        df, extra_covariates={"spend_residual": v_hat}, draws=draws, tune=tune, chains=chains, cores=cores
    )

    correction_helped = (
        abs(corrected["avg_contribution"] - true_avg_contribution)
        < abs(naive["avg_contribution"] - true_avg_contribution)
    )

    result = dict(
        true_avg_contribution=true_avg_contribution,
        naive_avg_contribution=naive["avg_contribution"],
        naive_pct_error=abs(naive["avg_contribution"] - true_avg_contribution) / true_avg_contribution * 100,
        naive_r_hat_max=naive["r_hat_max"], naive_n_divergences=naive["n_divergences"],
        corrected_avg_contribution=corrected["avg_contribution"],
        corrected_pct_error=abs(corrected["avg_contribution"] - true_avg_contribution) / true_avg_contribution * 100,
        corrected_r_hat_max=corrected["r_hat_max"], corrected_n_divergences=corrected["n_divergences"],
        control_function_coef_mean=corrected["residual_coef"]["mean"],
        first_stage_r2=instrument_diag["first_stage_r2"],
        residual_spend_correlation=instrument_diag["residual_spend_corr"],
        correction_helped=correction_helped,
    )
    pd.DataFrame([result]).to_csv(REPORT_DIR / "nonlinear_iv_correction.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = ["True\naverage contribution", "Naive nonlinear MMM\n(spend treated as exogenous)",
               "Control-function\n(first-stage residual included)"]
    values = [true_avg_contribution, naive["avg_contribution"], corrected["avg_contribution"]]
    # grey, not green, for "corrected" — it did not actually improve on naive here;
    # color should not imply success that the numbers don't support.
    colors = ["#264653", "#e76f51", "#6c757d"]
    ax.bar(methods, values, color=colors)
    ax.axhline(true_avg_contribution, color="#264653", linestyle="--", linewidth=1)
    ax.set_ylabel("average weekly contribution")
    verdict = "control function improved on naive" if correction_helped else "control function made it WORSE"
    ax.set_title(
        f"Nonlinear IV attempt: naive {result['naive_pct_error']:.1f}% off, "
        f"corrected {result['corrected_pct_error']:.1f}% off\n{verdict}",
        fontsize=11,
    )
    fig.text(
        0.5, 0.01,
        f"Instrument diagnostics: first-stage R²={instrument_diag['first_stage_r2']:.2f} (weak), "
        f"corr(residual, spend)={instrument_diag['residual_spend_corr']:.2f} (high) — the residual is "
        f"nearly collinear with spend itself, not a clean isolation of the confound. See docs/03.",
        ha="center", va="bottom", fontsize=8, color="#e63946", wrap=True,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig_nonlinear_iv_correction.png", dpi=150)
    plt.close(fig)

    return result


if __name__ == "__main__":
    result = run()
    for k, v in result.items():
        print(f"{k}: {v}")
