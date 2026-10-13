# Tier 2 + Real Tier 3 Covariates — Model Fit

Fit on Meta's Robyn demo dataset (simulated by its publisher, not this project — see ingest/robyn_reference.py) joined with real FRED/Trends/weather covariates for the matching historical weeks (ingest/build_tier2_panel.py).

- 208 weeks, 2015-11-23 to 2019-11-11
- Channels: tv, ooh, print, facebook, search
- Real extra covariates: consumer_sentiment, brand_search_interest, temperature_mean_c

## Diagnostics

- Max r-hat: **3.4334** (target: < 1.01)
- Min bulk ESS: 5
- Divergences: **3465** / 4000 post-tuning draws (86.6%)

**WARNING: THIS POSTERIOR IS NOT TRUSTWORTHY.** R-hat and/or the divergence rate are far outside acceptable bounds — the sampler could not properly explore the posterior. The coefficient table below is included for completeness, not as a result to act on. See docs/01-data-sources-and-honesty.md for the diagnosis (4 of 5 channels in this dataset have 51-59% zero-spend weeks — a known MMM identification failure mode, already documented separately in causal/mmm_vs_did.py, now confirmed on a second, independent real-shaped dataset). A higher target_accept and more tuning were tried and made no meaningful difference, confirming this is structural, not a sampler-tuning problem.

## Real-covariate coefficients (not reliable — see warning above)

| covariate | mean | 90% HDI | excludes zero? |
|---|---|---|---|
| consumer_sentiment | -14,816 | [-64,851, 14,467] | no |
| brand_search_interest | 46,668 | [-1, 120,913] | no |
| temperature_mean_c | -32,836 | [-109,771, 1,276] | no |

Full table: `outputs/reports/tier2_coefficients.csv`. Figures: `fig_tier2_posterior_predictive.png`, `fig_tier2_coefficients.png` (both carry the same warning directly on the image, not just here).

## Caveat

Revenue and channel spend here are Meta's simulated Robyn demo data, not a real business's real numbers. This fit shows whether adding real macro/search/weather covariates does anything *mechanically* on a realistic-shaped reference dataset — it is not a claim about what really drives any real company's revenue. That question is moot here regardless, given the posterior itself is unreliable.