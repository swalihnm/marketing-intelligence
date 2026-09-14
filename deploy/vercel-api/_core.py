"""Dependency-light core logic for the trimmed Vercel API: numpy + scipy
only, no PyMC/PyTensor/numba/arviz at runtime. Serves a pre-fitted
posterior (exported once, offline, via ../export_for_vercel.py) rather
than fitting live. See README.md in this directory for why.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from scipy.stats import norm

DATA_DIR = Path(__file__).resolve().parent / "data"


def hill(x, s, k):
    x_s = x ** s
    return x_s / (x_s + k ** s)


def geometric_adstock(spend: np.ndarray, alpha: float, max_lag: int = 8) -> np.ndarray:
    spend = np.asarray(spend, dtype=float)
    out = np.zeros_like(spend)
    for lag in range(max_lag + 1):
        weight = alpha ** lag
        if lag == 0:
            out += weight * spend
        else:
            out[lag:] += weight * spend[:-lag]
    return out


def load_posterior():
    payload = json.loads((DATA_DIR / "posterior_draws.json").read_text(encoding="utf-8"))
    return dict(
        channels=payload["channels"],
        beta=np.array(payload["beta"]),
        alpha=np.array(payload["alpha"]),
        s=np.array(payload["s"]),
        k=np.array(payload["k"]),
        current_spend=payload["current_spend"],
    )


def load_history():
    return json.loads((DATA_DIR / "history.json").read_text(encoding="utf-8"))


def response_per_draw(spend_vector: np.ndarray, draws: dict) -> np.ndarray:
    adstock_ss = spend_vector[None, :] / (1 - draws["alpha"])
    saturated = hill(adstock_ss, draws["s"], draws["k"])
    return (draws["beta"] * saturated).sum(axis=1)


def expected_response(spend_vector: np.ndarray, draws: dict) -> float:
    return float(response_per_draw(spend_vector, draws).mean())


def hdi_90(samples: np.ndarray) -> tuple[float, float]:
    """Simple equal-tailed 90% interval (avoids needing arviz's HDI impl)."""
    lo = float(np.percentile(samples, 5))
    hi = float(np.percentile(samples, 95))
    return lo, hi


def optimize_allocation(draws: dict, total_budget: float, max_shift_frac: float = 0.5,
                         channel_caps: dict | None = None) -> dict:
    channels = draws["channels"]
    current = np.array([draws["current_spend"][c] for c in channels])

    shift_lb = np.maximum(0.0, current * (1 - max_shift_frac))
    shift_ub = current * (1 + max_shift_frac)
    if channel_caps:
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

    budget_constraint = NonlinearConstraint(lambda x: x.sum(), -np.inf, total_budget)
    result = minimize(neg_objective, x0, method="trust-constr", bounds=bounds,
                       constraints=[budget_constraint], options={"maxiter": 1000})

    if not result.success or result.x.sum() > total_budget * (1 + 1e-4):
        x0b = lb + (ub - lb) * 0.3
        if x0b.sum() > total_budget:
            x0b = x0b * (total_budget / x0b.sum())
        fallback = minimize(neg_objective, x0b, method="SLSQP", bounds=bounds,
                             constraints=[{"type": "ineq", "fun": lambda x: total_budget - x.sum()}],
                             options={"maxiter": 300, "ftol": 1e-10})
        if fallback.success and fallback.x.sum() <= total_budget * (1 + 1e-4):
            result = fallback

    if not result.success or result.x.sum() > total_budget * (1 + 1e-4):
        raise RuntimeError("allocation optimization failed to converge to a feasible point")

    optimal_spend = {c: float(result.x[i]) for i, c in enumerate(channels)}
    gain_draws = response_per_draw(result.x, draws) - response_per_draw(current, draws)
    lo, hi = hdi_90(gain_draws)

    return dict(
        optimal_spend=optimal_spend,
        current_spend=dict(draws["current_spend"]),
        total_budget=total_budget,
        expected_gain_mean=float(gain_draws.mean()),
        expected_gain_hdi90_lower=lo,
        expected_gain_hdi90_upper=hi,
    )


def minimum_detectable_effect(tracking_error_std: float, n_weeks: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    se = tracking_error_std / np.sqrt(n_weeks)
    return (z_alpha + z_power) * se


def required_duration_weeks(target_mde_abs: float, tracking_error_std: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return ((z_alpha + z_power) * tracking_error_std / target_mde_abs) ** 2
