"""Constrained budget allocator: optimizes expected incremental revenue over
the full MMM posterior (`outputs/reports/idata_recovery.nc`, from
`simulate/recovery_study.py`) rather than over point estimates — the point
of doing this at all, per the spec: "A marketer who is told 'reallocate and
gain 8%' will not act; one told 'gain 4% to 12%, 90% credible' will."

Steady-state approximation: the response function per channel needs a
single spend level to evaluate, but the fitted model's adstock is a
time-series transform. This allocator assumes constant weekly spend at the
candidate allocation, under which geometric adstock converges to a steady
state: `adstock_ss = spend / (1 - alpha)` (the closed-form limit of
`sum_l alpha^l * spend` for constant spend as the lag horizon grows). This
is standard practice for MMM budget optimization and is stated explicitly
here because it's an approximation, not a re-derivation of the fitted
time-series model — it does not capture ramp-up/ramp-down dynamics from
changing spend levels.

Optimizer: scipy `trust-constr`, not CVXPY. The spec lists CVXPY, but CVXPY
requires a convex problem, and the Hill saturation function is S-shaped
(non-concave) for shape parameter s > 1 — which the fitted posterior
regularly draws (see docs/04) — so the allocation objective is not convex
in general. `trust-constr` handles the smooth non-convex, constrained
problem directly; it isn't guaranteed global-optimal, but for a small
number of channels with a sensible starting point that's an acceptable,
documented tradeoff. SLSQP (scipy's other common choice here) was tried
first and rejected: starting from current spend, it regularly failed to
converge and returned an infeasible point (budget constraint violated by
1-2%, `result.success=False`) without that failure surfacing anywhere
downstream — a real bug caught by `tests/test_api.py`, not a hypothetical
one. `trust-constr` converges reliably on the same inputs; SLSQP from a
different starting point is kept as a fallback attempt, and if both fail
`optimize_allocation` raises rather than returning a silently-infeasible
allocation.

Run: python -m allocate.optimizer
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint, minimize

from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "outputs" / "reports"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
IDATA_PATH = REPORT_DIR / "idata_recovery.nc"


def load_posterior_draws(idata_path: Path, channels: list[str], n_draws: int = 300, seed: int = 0) -> dict:
    idata = az.from_netcdf(str(idata_path))
    post = idata.posterior

    def flat(name):
        return post[name].values.reshape(-1, len(channels))

    beta, alpha, s, k = flat("beta"), flat("alpha"), flat("hill_s"), flat("hill_k")
    total = beta.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=min(n_draws, total), replace=False)
    return dict(beta=beta[idx], alpha=alpha[idx], s=s[idx], k=k[idx], channels=channels)


def response_per_draw(spend_vector: np.ndarray, draws: dict) -> np.ndarray:
    """Expected weekly incremental revenue, one value per posterior draw,
    for a candidate constant-spend allocation (steady-state adstock)."""
    adstock_ss = spend_vector[None, :] / (1 - draws["alpha"])
    saturated = hill(adstock_ss, draws["s"], draws["k"])
    contribution = draws["beta"] * saturated
    return contribution.sum(axis=1)


def expected_response(spend_vector: np.ndarray, draws: dict) -> float:
    return float(response_per_draw(spend_vector, draws).mean())


def optimize_allocation(
    draws: dict,
    current_spend: dict[str, float],
    total_budget: float,
    channel_caps: dict[str, tuple[float, float]] | None = None,
    max_shift_frac: float = 0.5,
) -> dict:
    channels = draws["channels"]
    n = len(channels)
    current = np.array([current_spend[c] for c in channels])

    shift_lb = np.maximum(0.0, current * (1 - max_shift_frac))
    shift_ub = current * (1 + max_shift_frac)
    if channel_caps is not None:
        cap_lb = np.array([channel_caps[c][0] for c in channels])
        cap_ub = np.array([channel_caps[c][1] for c in channels])
        lb = np.maximum(shift_lb, cap_lb)
        ub = np.minimum(shift_ub, cap_ub)
    else:
        lb, ub = shift_lb, shift_ub
    bounds = list(zip(lb, ub))

    def neg_objective(x):
        return -expected_response(x, draws)

    x0 = np.clip(current, lb, ub)
    if x0.sum() > total_budget:
        x0 = x0 * (total_budget / x0.sum())

    # trust-constr, not SLSQP (despite SLSQP being the more common default
    # choice for this kind of problem): empirically, SLSQP starting from
    # `current` spend regularly fails to converge on this objective and
    # returns an infeasible point (result.success=False, budget constraint
    # violated by 1-2%) — see progress.md for the reproduction. trust-constr
    # converges reliably to a feasible point on the same problem, with a
    # comparable or better objective value. SLSQP is kept as a fallback
    # attempt (from a different, interior starting point) in case
    # trust-constr fails on some future input; if BOTH fail, this raises
    # rather than silently returning an infeasible allocation.
    budget_constraint = NonlinearConstraint(lambda x: x.sum(), -np.inf, total_budget)
    result = minimize(
        neg_objective, x0, method="trust-constr", bounds=bounds, constraints=[budget_constraint],
        options={"maxiter": 1000},
    )

    if not result.success or result.x.sum() > total_budget * (1 + 1e-4):
        x0_fallback = lb + (ub - lb) * 0.3
        if x0_fallback.sum() > total_budget:
            x0_fallback = x0_fallback * (total_budget / x0_fallback.sum())
        fallback = minimize(
            neg_objective, x0_fallback, method="SLSQP", bounds=bounds,
            constraints=[{"type": "ineq", "fun": lambda x: total_budget - x.sum()}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if fallback.success and fallback.x.sum() <= total_budget * (1 + 1e-4):
            result = fallback

    if not result.success or result.x.sum() > total_budget * (1 + 1e-4):
        raise RuntimeError(
            f"budget allocation optimization failed to converge to a feasible point "
            f"(status: {result.message}, spend sum {result.x.sum():.2f} vs budget {total_budget:.2f})"
        )

    optimal_spend = {c: float(result.x[i]) for i, c in enumerate(channels)}
    gain_draws = response_per_draw(result.x, draws) - response_per_draw(current, draws)

    return dict(
        success=True,
        optimal_spend=optimal_spend,
        current_spend=dict(current_spend),
        total_budget=total_budget,
        expected_gain_mean=float(gain_draws.mean()),
        expected_gain_hdi90=tuple(float(v) for v in az.hdi(gain_draws, prob=0.90)),
        expected_response_current=float(response_per_draw(current, draws).mean()),
        expected_response_optimal=float(response_per_draw(result.x, draws).mean()),
    )


def marginal_roi(draws: dict, current_spend: dict[str, float], delta_frac: float = 0.01) -> pd.DataFrame:
    """d(revenue)/d(spend) per channel at current spend, per posterior draw
    (finite difference), holding other channels fixed."""
    channels = draws["channels"]
    current = np.array([current_spend[c] for c in channels])
    rows = []
    for i, c in enumerate(channels):
        delta = max(current[i] * delta_frac, 1.0)
        up, down = current.copy(), current.copy()
        up[i] += delta
        down[i] -= delta
        d_response = (response_per_draw(up, draws) - response_per_draw(down, draws)) / (2 * delta)
        lo, hi = az.hdi(d_response, prob=0.90)
        rows.append(dict(channel=c, marginal_roi_mean=float(d_response.mean()),
                          hdi_lower=float(lo), hdi_upper=float(hi)))
    return pd.DataFrame(rows)


def budget_sweep(draws: dict, current_spend: dict[str, float], multipliers=(0.8, 1.0, 1.2, 1.5),
                  max_shift_frac: float = 0.5) -> pd.DataFrame:
    base_budget = sum(current_spend.values())
    rows = []
    for m in multipliers:
        res = optimize_allocation(draws, current_spend, base_budget * m, max_shift_frac=max_shift_frac)
        row = dict(multiplier=m, total_budget=base_budget * m,
                   expected_response=res["expected_response_optimal"], success=res["success"])
        row.update({f"spend_{c}": v for c, v in res["optimal_spend"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from simulate.dgp import sample_true_params, simulate as dgp_simulate

    true = sample_true_params(n_weeks=156, spend_correlation=0.5, seed=42)
    df = dgp_simulate(true, seed=42 + 1000)
    channels = list(true.channels.keys())
    current_spend = {c: float(df[f"spend_{c}"].mean()) for c in channels}

    draws = load_posterior_draws(IDATA_PATH, channels, n_draws=300)
    total_budget = sum(current_spend.values())

    result = optimize_allocation(draws, current_spend, total_budget)
    print("current spend:", current_spend)
    print("optimal spend:", result["optimal_spend"])
    print("expected gain (mean):", result["expected_gain_mean"])
    print("expected gain (90% HDI):", result["expected_gain_hdi90"])

    roi = marginal_roi(draws, current_spend)
    print("\nmarginal ROI:\n", roi)

    sweep = budget_sweep(draws, current_spend)
    print("\nbudget sweep:\n", sweep)
