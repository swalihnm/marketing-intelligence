# Parameter recovery validation

This is the headline validation for the whole project (see
`docs/01-data-sources-and-honesty.md` for why: real multi-channel spend data
is proprietary, so there is no ground truth to check the MMM against on real
data — this substitutes for that). Full methodology in
`simulate/recovery_study.py`; this doc records the result of the latest run
and what it does/doesn't prove. Regenerate with `python -m
simulate.recovery_study` — that command rewrites
`outputs/reports/recovery_report.md` and everything below should match it.

## Setup

156 simulated weeks, 4 channels (tv, search, social, display) with distinct
true adstock decay rates, Hill saturation curves, and elasticities, spend
correlation = 0.5 (channels' spend moves together via a shared latent
"budget cycle" factor — see `simulate/dgp.py`), seed = 42. Fit with PyMC:
4 chains x 4 cores, 1000 tuning + 1000 draws, `target_accept=0.95`.

## Result

- **Sampling was clean**: max r-hat 1.0033 (target < 1.01), min bulk ESS
  1213, **zero divergences**. No sign of a poorly-behaved posterior.
- **100% of the 16 headline channel parameters** (beta, alpha, hill_s,
  hill_k, across all 4 channels) had their true value inside the 90%
  posterior HDI.
- 95.8% coverage overall (23/24), including nuisance trend/seasonality/price
  terms. The one miss was a secondary seasonal Fourier term
  (`season_b2`, a minor harmonic contributing a small amount of the
  seasonality signal) — not a channel-level parameter, and not something a
  budget decision depends on.
- Mean absolute % error: beta 31.7%, alpha 21.7%, hill_s 7.2%, hill_k 39.9%.
  These are not small — HDI widths for beta and hill_k in particular often
  span 2-4x. That's expected and worth stating plainly: 156 weeks of
  moderately correlated (0.5) channel spend does not pin down elasticities
  tightly even when the model is exactly correctly specified. It's also
  precisely why the budget allocator (planned) is designed to optimize over
  posterior draws rather than point estimates — the uncertainty shown here
  is the uncertainty a real recommendation has to carry forward.

Full numbers: `outputs/reports/recovery_table.csv` and
`diagnostics_summary.csv`. Figures: `outputs/figures/fig_recovery_*.png`
(forest plots, true value vs. posterior mean+HDI per channel),
`fig_posterior_predictive.png` (tight fit to the simulated series, as
expected for a well-specified model on its own generated data),
`fig_prior_predictive.png`, `fig_trace.png`.

## What this does and doesn't prove

**Does prove:** the estimator (PyMC model + priors in `mmm/model.py` /
`mmm/priors.py`) is unbiased under the assumptions encoded in the DGP, and
the sampler can separate 4 channels' effects even with correlated spend at
this dataset size, without divergences or convergence problems.

**Doesn't prove:** that this works on real spend data, where channels are
often more correlated than 0.5, where the true functional form isn't
exactly geometric-adstock-plus-Hill, and where confounders exist that
aren't in the DGP at all (see `docs/03-causal-identification.md`). The
correlation-sensitivity sweep (`docs/09-correlation-sensitivity.md`) is
the follow-on: repeating this study at spend correlation 0.3 → 0.9. Result:
coverage held at 100% throughout, but point-estimate accuracy degraded
monotonically (beta's mean error rose from 28% at correlation 0.3 to 39%
at 0.9) — a closer proxy for how fragile real-world identification might
be, and a reassuring sign that the model gets honestly less precise under
collinearity rather than overconfidently wrong.

## Reproducing

```bash
python -m simulate.recovery_study
```

Takes ~4 minutes on this machine (no C++ compiler present — PyMC/PyTensor
fall back to a numba-backed pure-Python linker; see
`docs/02-mmm-methodology.md`). Deterministic given the seed.
