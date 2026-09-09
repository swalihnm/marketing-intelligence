"""Adstock / carryover transforms.

Geometric adstock is implemented twice on purpose:
  - `geometric_adstock_numpy` runs on plain numpy arrays and is used by the
    simulator (simulate/dgp.py) and any offline analytics.
  - `geometric_adstock_pt` builds the same computation out of PyTensor ops so
    it can sit inside a PyMC model, where `alpha` is a random variable rather
    than a fixed float.

Both use a truncated-lag formulation instead of a recursive scan:

    adstock_t = sum_{l=0}^{L} alpha^l * spend_{t-l}

`L` (max_lag) is a fixed integer chosen so alpha**L is negligible for the
range of decay rates we put priors on; it is NOT estimated. Truncating this
way avoids `pytensor.scan`, which is markedly slower to build/sample under
PyMC's default (non-compiled) backend.

Weibull (delayed-peak) adstock is deferred — see progress.md cut order.
"""
from __future__ import annotations

import numpy as np


def geometric_adstock_numpy(spend: np.ndarray, alpha: float, max_lag: int = 8) -> np.ndarray:
    """Apply geometric adstock to a 1D spend series.

    spend: shape (T,)
    alpha: decay rate in [0, 1)
    """
    spend = np.asarray(spend, dtype=float)
    T = spend.shape[0]
    out = np.zeros(T, dtype=float)
    for lag in range(max_lag + 1):
        weight = alpha ** lag
        if lag == 0:
            out += weight * spend
        else:
            out[lag:] += weight * spend[:-lag]
    return out


def geometric_adstock_pt(spend, alpha, max_lag: int = 8):
    """PyTensor version of geometric adstock for use inside a PyMC model.

    spend: PyTensor vector (or pm.Data), shape (T,)
    alpha: PyTensor scalar random variable in (0, 1)
    """
    import pytensor.tensor as pt

    T = spend.shape[0]
    out = spend * (alpha ** 0)
    for lag in range(1, max_lag + 1):
        shifted = pt.concatenate([pt.zeros(lag), spend[:-lag]])
        out = out + (alpha ** lag) * shifted
    return out
