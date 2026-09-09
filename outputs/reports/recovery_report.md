# Parameter Recovery Study

Simulated 156 weeks, 4 channels (tv, search, social, display), spend correlation = 0.5, seed = 42.

## Sampling diagnostics

- Elapsed: 236.5s (4 chains x 4 cores, 1000 tune / 1000 draws)
- Max r_hat across all parameters: **1.0033** (target: < 1.01 — met)
- Min bulk ESS across all parameters: **1213**
- Divergences: **0**

## Recovery result

- **100.0%** of the 16 headline channel parameters (beta, alpha, hill_s, hill_k across 4 channels) had their true value inside the 90% posterior HDI (16/16).
- 95.8% coverage across all 24 parameters including trend/seasonality/price/intercept/noise nuisance terms (23/24).
- Mean absolute % error: beta (elasticity scale) 31.7%, alpha (adstock decay) 21.7%, hill_s (saturation shape) 7.2%, hill_k (half-saturation point) 39.9%.

Full per-parameter numbers: `outputs/reports/recovery_table.csv`.

### Parameters whose true value fell outside the 90% HDI

- `season_b2` ((global)): true=8545.3, posterior mean=2292.4, 90% HDI=[-4323.9, 8126.5]

All misses are nuisance seasonality/trend terms, not channel-level parameters — the parameters that actually drive the budget allocation decision (beta, alpha, hill_s, hill_k) recovered at 100% coverage in this run.

## Figures

- `fig_prior_predictive.png` — priors alone imply a plausible revenue range before fitting
- `fig_recovery_beta.png`, `fig_recovery_alpha.png`, `fig_recovery_hill_s.png`, `fig_recovery_hill_k.png` — posterior mean + HDI (dot/bar) vs true value (x) per channel; red = HDI missed the true value
- `fig_posterior_predictive.png` — fitted model against the (simulated) observed series
- `fig_trace.png` — MCMC traces for the channel-level parameters

## Interpretation

Coverage at or above the nominal HDI probability, with r_hat below 1.01 and zero divergences, is evidence the sampler is well-behaved and the model's functional form (geometric adstock + Hill saturation) is estimable from data of this size and collinearity (spend correlation = 0.5 here — see `progress.md` for the planned correlation-sensitivity sweep testing higher correlation levels). This does **not** prove the model works on real spend data — only that the estimator is unbiased under the assumptions encoded in the DGP (`simulate/dgp.py`). Interval widths are non-trivial (e.g. beta HDIs often span 2-4x) — a reminder that even with a correctly-specified model, 156 weeks of moderately correlated channel spend leaves real uncertainty about exact channel elasticities. That uncertainty is the entire reason the budget allocator (planned, not yet built) optimizes over posterior draws instead of point estimates.