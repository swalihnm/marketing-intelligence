# Results

Consolidated results, updated as each layer lands. Currently covers: MMM
parameter recovery (see `docs/04-parameter-recovery-validation.md` for the
full writeup) and the customer economics layer (this file). Not yet
covered: response curves on real-shaped media data, causal identification,
budget allocation, experiment design — those layers aren't built (see
`progress.md`).

## Customer economics (UCI Online Retail II — see docs/01)

Real transactional data: 805,549 cleaned line items, 5,878 customers, Dec
2009–Dec 2011. Reproduce with `python -m ingest.uci_retail_loader` then
`python -m customer.cohorts` and `python -m customer.clv`.

### Cohorts and retention

25 monthly acquisition cohorts, ranging from 28 to 955 customers (the small
early-2009-12 and late-2011-12 cohorts are partial-month data-boundary
artifacts, not real seasonality). Average first-order value sits mostly in
the £300–£450 range across cohorts, with a handful of outlier months (e.g.
2010-09 at £600, 2011-12 at £707) — plausible for a small-N wholesale-mixed
customer base rather than a sign of a data problem, but worth a closer look
before quoting cohort-by-cohort figures narrowly.

`outputs/figures/fig_cohort_retention_heatmap.png` shows the standard
decay pattern: roughly 20–40% of a cohort returns in month 1, decaying from
there, with no cohort showing materially different retention shape from
its neighbors (no visible before/after break — i.e., nothing in this
window looks like a retention-changing intervention). Full table:
`outputs/reports/retention_curves.csv`.

### RFM segmentation

`outputs/figures/fig_rfm_segments.png` / `outputs/reports/rfm_segmentation.csv`.
6 named segments from quintile R/F scores. Two roughly-equal-sized poles
dominate — **Hibernating** (1,636 customers) and **Champions** (1,552) —
with **Loyal Customers** (1,258) and **Needs Attention** (835) in between,
and small **New Customers** (306) / **At Risk** (291) segments. The
bimodal split (large Hibernating + large Champions, comparatively few
customers in the "at risk of leaving soon" middle) is consistent with the
retention heatmap: customers who don't return early tend to not return at
all, rather than trickling off gradually.

### CAC / LTV by (synthetic) channel

**Read this section only after reading `customer/channel_synth.py`'s
docstring and `docs/01-data-sources-and-honesty.md`: the channel assignment
is synthetic (no real acquisition-channel data exists in this dataset), and
so is the per-channel CAC. These numbers demonstrate the analysis pattern
on a plausible cost structure — they are not a real finding about which
channel is actually most efficient.**

| channel | customers | avg LTV | (synthetic) CAC | LTV:CAC |
|---|---|---|---|---|
| direct | 894 | £3,631 | £1.50 | 2421x |
| organic | 1,165 | £2,971 | £4.00 | 743x |
| email | 827 | £3,089 | £8.00 | 386x |
| paid_social | 1,313 | £3,084 | £35.00 | 88x |
| paid_search | 1,679 | £2,640 | £42.00 | 63x |

Full table + payback period: `outputs/reports/cac_ltv_by_channel.csv`,
`payback_by_channel.csv`, `outputs/figures/fig_ltv_cac_by_channel.png`. The
ratios are extreme (63x–2421x) because the illustrative CAC values are
small relative to this dataset's real average customer LTV (£2,600–£3,600)
— a reminder that the ratio's magnitude here is an artifact of the assumed
CAC scale, not a signal to read literally. What is real: the relative
ordering by avg LTV is fairly flat across channels (£2,640–£3,631, a
~1.4x spread) — in this dataset, which customers you compare barely
matters for average spend; only the assumed CAC drives the ranking. That's
exactly why real channel-attribution data (not available here) matters for
this specific analysis.

### Customer lifetime value: BG/NBD + Gamma-Gamma vs. naive

Calibration/holdout split: calibration ends 2011-06-09, holdout is the
final 6 months of data (through 2011-12-09). Fit `BetaGeoFitter` +
`GammaGammaFitter` (`lifetimes`) on calibration-period behavior, predict
6-month-forward CLV, compare against actual holdout revenue and against a
naive baseline (flat extrapolation of each customer's calibration-period
average monthly spend).

**Headline finding: BG/NBD + Gamma-Gamma did not clearly outperform the
naive baseline on this dataset**, and that's reported as-is rather than
massaged:

| | BG/NBD + Gamma-Gamma | Naive (flat extrapolation) |
|---|---|---|
| MAE vs. actual holdout revenue | £732 | £659 |
| Correlation with actual holdout revenue | 0.073 | 0.068 |

Both methods correlate weakly with actual holdout spend (~0.07) — 6 months
of calibration history is a weak predictor of the next 6 months for this
customer base regardless of method, which is itself informative. The
scatter plot (`outputs/figures/fig_clv_validation.png`) shows both models,
BG/NBD+Gamma-Gamma more so, over-predicting for a large mass of customers
who had near-zero actual holdout revenue — i.e., customers who looked
likely to return based on calibration behavior but churned anyway.

**Model-fit caveats, surfaced rather than hidden:**

- The fitted Gamma-Gamma shape parameter **q = 0.33** (< 1). The model's
  textbook mean is only well-defined for q > 1; the fit here is consistent
  with — and evidence for — a mild violation of the model's
  frequency/monetary-independence assumption (measured correlation on
  calibration data: **0.17**, not zero) and a monetary-value distribution
  that's heavy-tailed and often single-observation for low-frequency
  customers. Predicted CLV is clipped at 0 as a pragmatic floor (lifetime
  value can't be negative); this does not fix the underlying assumption
  violation, it just prevents nonsensical negative outputs from it.
- **50 customers** (top 1% by calibration purchase frequency — wholesale/
  bulk-buying accounts up to 75 repeat orders, not typical retail behavior)
  were excluded from fitting both models (standard practice for this kind
  of outlier), though still included in prediction and evaluation.
- **534 of 4,967 customers (10.8%)** got a NaN prediction from the
  underlying `lifetimes` hypergeometric calculation — all customers
  acquired in the last few days of the calibration window (near-zero
  observed history, a known numerical edge case for this class of model).
  Imputed to 0 rather than dropped, so every customer still has a row;
  reported explicitly rather than silently absorbed into the averages.

Full numbers: `outputs/reports/clv_validation_metrics.csv`,
`clv_predictions.csv`. Reproduce with `python -m customer.clv`.

**Interpretation:** this is a legitimate, useful result, not a failed
experiment — it demonstrates the discipline the whole project is built
around (validate before trusting, report what breaks). The honest read is
that off-the-shelf BG/NBD + Gamma-Gamma isn't obviously the right tool for
this particular customer base (short observation window, mixed retail/
wholesale customers, high dropout after one purchase) without further work
— e.g. separating wholesale accounts entirely, a longer calibration window,
or a model less sensitive to the independence assumption. That's a
different, more specific claim than "CLV modeling doesn't work here," and
it's the claim the data actually supports.
