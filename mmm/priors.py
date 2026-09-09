"""Prior specifications for the Bayesian MMM.

Priors are not a formality here — see docs/02-mmm-methodology.md for the
full argument. In short: channel spend is heavily correlated (see
simulate/dgp.py's spend_correlation), so the likelihood alone weakly
identifies each channel's effect. Priors are what let the posterior
separate correlated channels instead of collapsing into an arbitrary split
between them.

`channel_type` below must agree with simulate/dgp.CHANNEL_ARCHETYPES for the
recovery study to be a fair test: the model is given the same kind of
qualitative decay-speed knowledge a real analyst would bring ("TV effects
carry for longer than search"), not the exact true alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

# channel -> qualitative decay speed, used to set the Beta(a, b) prior on
# adstock alpha. This mirrors simulate/dgp.DECAY_RANGES but is deliberately
# a *belief*, supplied independently, not read off the true parameters.
CHANNEL_DECAY_BELIEF = {
    "tv": "slow",
    "search": "fast",
    "social": "fast",
    "display": "medium",
}

# Beta(a, b) parameters chosen so the prior mean/spread roughly matches the
# qualitative belief, while remaining wide enough to be overturned by data.
DECAY_BETA_PRIOR = {
    "slow": (6.0, 3.0),      # mean ~0.67, weight of ~9 "pseudo-observations"
    "medium": (4.0, 4.0),    # mean ~0.50
    "fast": (2.5, 6.0),      # mean ~0.29
}


@dataclass
class ChannelPriorConfig:
    decay_belief: str          # "slow" | "medium" | "fast"
    beta_sigma: float          # HalfNormal sigma for the channel's beta (revenue scale)
    hill_s_alpha: float = 2.0  # Gamma(alpha, beta) for hill shape s
    hill_s_beta: float = 1.0
    hill_k_sigma: float = 1.0  # scale multiplier on HalfNormal(k), relative to k_reference


def default_channel_prior(channel: str, mean_weekly_spend: float) -> ChannelPriorConfig:
    belief = CHANNEL_DECAY_BELIEF.get(channel, "medium")
    # beta prior scale: weakly informative, wide enough to cover plausible
    # incremental revenue but centered low so the model must be convinced by
    # data (and the positivity constraint) rather than defaulting to large effects.
    return ChannelPriorConfig(
        decay_belief=belief,
        beta_sigma=3e5,
        hill_k_sigma=max(mean_weekly_spend * 4, 1e4),
    )
