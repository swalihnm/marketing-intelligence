# MMM methodology

## Functional form

```
revenue_t = intercept + trend_t + seasonality_t + price_effect_t
          + sum_c beta_c * hill(adstock(spend_c,t, alpha_c), s_c, k_c)
          + epsilon_t
```

`transforms/adstock.py` implements geometric adstock (`adstock_t = sum_l
alpha^l * spend_{t-l}`, truncated at `max_lag=8`) twice: a numpy version for
the simulator, a PyTensor version for the PyMC model, because `alpha` is a
fixed float in the former and a random variable in the latter. Weibull
(delayed-peak) adstock is deferred — see the cut order in `progress.md`;
geometric is kept because the spec explicitly says never to drop it in favor
of the parameter recovery study.

`transforms/saturation.py` implements the Hill function `f(x) = x^s / (x^s +
k^s)`. `s` (shape) and `k` (half-saturation point, in adstocked-spend units)
are estimated per channel, not hardcoded — `k` is the number that's directly
interpretable to a marketer ("this channel is half-saturated around $X/week
of adstocked spend") and is surfaced in the recovery report and, later, the
dashboard's response-curve view.

## Why priors are not optional here

Channel spend is generated with a shared latent "budget cycle" factor
(`simulate/dgp._simulate_correlated_spend`), controlled by a
`spend_correlation` parameter — this mirrors the real-world pattern where
marketing budgets across channels move together (overall budget cycles,
seasonal pushes), which is exactly what makes MMM's identification problem
hard: the likelihood alone cannot cleanly separate two channels whose spend
moves in lockstep.

`mmm/priors.py` sets **informative decay priors by channel archetype**
(`CHANNEL_DECAY_BELIEF`: e.g. TV → slow decay, `Beta(6,3)`, mean ≈0.67;
search/social → fast decay, `Beta(2.5,6)`, mean ≈0.29) — the kind of
qualitative knowledge ("TV effects linger longer than search clicks") a
real analyst brings to the problem, deliberately supplied independently of
`simulate/dgp.py`'s true draws so the recovery study isn't circular (the
model is told the *category* of decay speed, not the *value*).

`beta_c` is constrained positive (`HalfNormal`) since spend cannot reduce
revenue. `s` and `k` get weakly informative priors (`Gamma(2,1)`,
`HalfNormal` scaled to each channel's observed mean spend) — informative
enough to keep the sampler out of implausible regions, wide enough to be
overturned by data.

## Diagnostics

Required and checked in every recovery run (`simulate/recovery_study.py`):
r-hat (< 1.01), bulk ESS, divergence count, trace plots, posterior
predictive check, prior predictive check. Numbers for the latest run:
`outputs/reports/diagnostics_summary.csv` and `recovery_report.md`.

## Sampling backend note

No C++ compiler is available in this environment (`g++`/`cl.exe` not on
PATH). PyMC 6 / PyTensor fall back automatically to a Python/numba-based
linker rather than compiling C — sampling works, just somewhat slower than
a compiled backend. `numba` is a required dependency for this reason (see
`requirements.txt`). Chains are sampled with `cores=4` (multiprocessing);
scripts that call `pm.sample` with `cores>1` guard the entry point with
`if __name__ == "__main__":` because Windows uses the `spawn` multiprocessing
start method.

## Causal identification

Not yet written — planned as `docs/03-causal-identification.md`, alongside
`causal/did.py` / `causal/iv.py`. See `progress.md`.
