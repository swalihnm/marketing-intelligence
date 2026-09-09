# Geo-lift experiment: pre-registration

Written before looking at post-period results — decided in advance, exactly
as the design requires (see `docs/05-experimentation-notes.md` on why
peeking and after-the-fact analysis choices invalidate this). For this
build, "before looking at results" means before running
`experiment/analyze.py`; the commit history is the record of that order.

## Hypothesis

The treatment market's marketing intervention causes a positive lift in
weekly revenue, measurable against its matched control market.

- **H0**: average post-period (treatment − matched control) revenue gap =
  average pre-period gap (no incremental effect).
- **H1**: average post-period gap > average pre-period gap (positive
  incremental effect). Two-sided test used in practice (see below) so an
  unexpected negative effect is also caught, not just ruled out by
  assumption.

## Primary metric

Weekly revenue, treatment market vs. matched control market (matched on
pre-period revenue-trend correlation — see `experiment/geo_matching.py`).
Estimator: difference-in-differences — `(post-period average gap) −
(pre-period average gap)`.

## Design

- **Treatment market**: one designated market (`experiment/geo_simulate.py`
  — `market_00` in the current simulated run).
- **Control**: single best-matched market by pre-period revenue-trend
  correlation (not a multi-donor synthetic control blend — see
  `experiment/geo_matching.py`'s docstring for why).
- **Pre-period**: 26 weeks, used only for matching and for estimating the
  tracking-error noise term — never for the effect estimate itself.
- **Post-period / test duration**: 8 weeks, chosen against the power
  analysis below.
- **Significance level**: alpha = 0.05, two-sided.
- **Target power**: 0.80.

## Power analysis (decided before running the test)

Noise term: the pre-period tracking error (std of the weekly gap between
treatment and matched control) — `experiment/power.py`. At the matched
pair's observed tracking error, an 8-week post-period gives a minimum
detectable effect of **~5.1%** of baseline weekly revenue (see
`outputs/figures/fig_geo_mde_curve.png` and
`outputs/reports/geo_experiment_analysis.csv`). The effect this design is
meant to catch (the business's minimum effect worth acting on, stated
in advance) is assumed to be **10%** — 8 weeks was chosen because it clears
that bar with headroom, not tuned after seeing the outcome.

## Stopping rule

**Fixed sample size.** The test runs for the full 8 weeks and is analyzed
once, at the end. No interim looks, no early stopping on significance — see
`docs/05-experimentation-notes.md` for why.

## Analysis method

Difference-in-differences on the matched pair, two-sided z-test using the
pre-period tracking error as the standard error of the average post-period
gap (`experiment/analyze.py`). Reported: point estimate (absolute and %),
z-statistic, p-value, and a plain significant/not-significant call at
alpha=0.05 — decided as the readout in advance, not chosen after seeing
which framing looks better.

## What would make this not worth acting on

Even if significant, a detected effect below the business's stated minimum
practical effect (10%, matching the power analysis target) would not
justify a budget reallocation on its own — see
`docs/05-experimentation-notes.md`'s note on statistical vs. practical
significance.
