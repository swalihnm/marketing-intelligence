"""Saturation / diminishing-returns transform.

Hill function: f(x) = x^s / (x^s + k^s)
  - s: shape parameter (steepness of the curve)
  - k: half-saturation point, in the same units as x (adstocked spend) — the
    spend level at which the channel is delivering half its ceiling effect.

The same expression works unchanged for numpy arrays and PyTensor tensors
(both overload **, +, /), so there is a single implementation rather than
numpy/pytensor variants like adstock.py.
"""
from __future__ import annotations


def hill(x, s, k):
    x_s = x ** s
    return x_s / (x_s + k ** s)
