"""Power analysis for a matched-market geo-lift test.

Standard geo-lift power calculation: the noise term is the pre-period
tracking error between the matched treatment/control pair (
`experiment/geo_matching.py`) — i.e. how much the two markets' revenue
diverges even with no treatment effect at all. Averaging over more
post-period weeks shrinks the standard error of the estimated gap by
sqrt(n), the same way any repeated-measurement average does.

MDE = (z_(alpha/2) + z_power) * tracking_error_std / sqrt(n_weeks)

This is a two-sided test on the average weekly (treatment - control) gap
during the post-period, against the null that the average gap equals the
pre-period average gap (i.e. the standard geo-lift diff-in-diff setup).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def minimum_detectable_effect(
    tracking_error_std: float,
    n_weeks: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """MDE in absolute revenue units, for the average weekly gap over `n_weeks`."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    se = tracking_error_std / np.sqrt(n_weeks)
    return (z_alpha + z_power) * se


def required_duration_weeks(
    target_mde_abs: float,
    tracking_error_std: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Weeks of post-period needed to detect `target_mde_abs` at the given alpha/power."""
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    return ((z_alpha + z_power) * tracking_error_std / target_mde_abs) ** 2


def mde_curve(tracking_error_std: float, weeks_range, alpha: float = 0.05, power: float = 0.8) -> np.ndarray:
    return np.array([minimum_detectable_effect(tracking_error_std, w, alpha, power) for w in weeks_range])
