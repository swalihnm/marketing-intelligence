"""Synthetic acquisition-channel assignment for the customer economics layer.

The UCI Online Retail II dataset has no marketing-channel or acquisition-cost
field — no public transactional dataset does; that data lives in ad
platforms, not order databases, which is exactly the kind of proprietary
data gap this whole project is built around (see
docs/01-data-sources-and-honesty.md). The spec's CAC/LTV-by-channel analysis
needs a channel dimension to exist at all, so this module assigns one:
deterministically (seeded by customer_id, reproducible), with a fixed,
clearly-labeled illustrative CAC per channel.

This is NOT real acquisition data. Every output built on it — the
cac_ltv_by_channel numbers, the payback-period chart — is a demonstration of
the analysis pattern on a plausible cost structure, not a real finding about
real channel economics. That distinction is repeated in every place these
numbers surface (docs, chart titles, report text) rather than stated once
and forgotten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# channel -> (assignment probability, illustrative CAC in GBP)
# CAC ordering (paid > email > organic > direct) mirrors typical real-world
# cost structures; the specific numbers are assumptions, not estimates.
CHANNEL_CONFIG = {
    "paid_search": dict(prob=0.28, cac=42.0),
    "paid_social": dict(prob=0.22, cac=35.0),
    "email": dict(prob=0.15, cac=8.0),
    "organic": dict(prob=0.20, cac=4.0),
    "direct": dict(prob=0.15, cac=1.5),
}


def assign_synthetic_channels(customer_ids: list[int], seed: int = 7) -> pd.DataFrame:
    channels = list(CHANNEL_CONFIG.keys())
    probs = [CHANNEL_CONFIG[c]["prob"] for c in channels]
    probs = np.array(probs) / np.sum(probs)

    rng = np.random.default_rng(seed)
    assigned = rng.choice(channels, size=len(customer_ids), p=probs)

    df = pd.DataFrame({"customer_id": customer_ids, "channel": assigned})
    df["cac"] = df["channel"].map(lambda c: CHANNEL_CONFIG[c]["cac"])
    return df
