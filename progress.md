# Progress

Tracking actual status against `PROJECT-03-marketing-mix-modelling.md`'s
4-week plan (Section 10). Updated as work lands — this file, not memory of
the spec, is the source of truth on what's actually built.

Last updated: 2026-09-09.

## Status at a glance

| Layer | Status |
|---|---|
| Simulator + parameter recovery (Tier 1 data) | **Done** |
| Bayesian MMM (adstock + Hill saturation, PyMC) | **Done** |
| Diagnostics (r-hat, ESS, divergences, PPC, prior PC) | **Done** |
| Correlation-sensitivity study (identification degradation) | Not started |
| Data ingestion — MMM media spine (Robyn/Meridian, Trends, FRED, weather) | Not started |
| Data ingestion — customer data (Tier 4) | **Done, substituted dataset** (Olist → UCI Online Retail II; see docs/01) |
| SQL analytics — cohorts, retention, RFM, CAC/LTV | **Done** |
| SQL analytics — category contribution, delivery-impact-on-repeat | **Not feasible on substitute dataset** (needs Olist; see docs/01) |
| Causal identification section + DiD/IV | Not started |
| Customer layer — BG/NBD, Gamma-Gamma CLV | **Done** (validated on holdout; did not beat naive baseline — see docs/07) |
| Experimentation design (geo-lift, power analysis) | **Done** (design validated against a known injected effect — see docs/05) |
| Budget allocator (posterior-based, constrained) | Not started |
| Dashboard / API / deployment | Not started |

## Week 1 target (spec): "Simulator + parameter recovery + data ingestion"

- [x] `simulate/dgp.py` — ground-truth DGP: per-channel beta/alpha/s/k,
      trend, 2-harmonic seasonality, price effect, correlated cross-channel
      spend via a shared latent factor (`spend_correlation` param).
- [x] `transforms/adstock.py` — geometric adstock, numpy + PyTensor versions.
      Weibull deferred (spec's own cut order — kept geometric, dropped Weibull).
- [x] `transforms/saturation.py` — Hill saturation, single implementation
      (numpy/PyTensor both support the same operator overloads).
- [x] `mmm/priors.py` — channel-archetype decay priors, set independently of
      the DGP's true draws so the recovery test isn't circular.
- [x] `mmm/model.py` — full Bayesian MMM in PyMC matching the spec's
      functional form (Section 6.2).
- [x] `simulate/recovery_study.py` — fits the model to simulated data, reports
      HDI coverage of true parameters, r-hat/ESS/divergences, prior and
      posterior predictive checks, forest-style recovery plots per parameter.
- [x] `tests/test_transforms.py` — unit tests on adstock/saturation math
      (5 passing).
- [ ] Correlation-sensitivity sweep (spend correlation 0.3 → 0.9 vs recovery
      degradation) — spec calls this "an excellent interview artefact";
      deferred to next pass, not yet run.
- [ ] Data ingestion (Tier 2 Robyn/Meridian spine, Tier 3 Trends/FRED/weather,
      Tier 4 Olist) — not started.

**Latest recovery run results** (156 weeks, 4 channels, spend correlation
0.5, seed 42): max r-hat **1.0033**, min bulk ESS **1213**, **0
divergences**. **100% of the 16 headline channel parameters** (beta, alpha,
hill_s, hill_k) recovered inside their 90% posterior HDI; 95.8% overall
(23/24 — the one miss was a minor seasonal harmonic, not a channel
parameter). Full writeup: `docs/04-parameter-recovery-validation.md`. Raw
outputs: `outputs/reports/recovery_report.md`, `recovery_table.csv`,
`diagnostics_summary.csv` (regenerated on every `python -m
simulate.recovery_study` run — check timestamps if re-verifying).

## Phase 2: Customer economics layer

Real transactional data (UCI Online Retail II, substituted for Olist — see
docs/01-data-sources-and-honesty.md for why), not simulated.

- [x] `ingest/uci_retail_loader.py` — downloads/cleans the dataset, builds
      `data/processed/warehouse.duckdb` (805,549 rows, 5,878 customers,
      Dec 2009–Dec 2011).
- [x] `customer/channel_synth.py` — synthetic acquisition-channel + CAC
      assignment (documented as illustrative, not real — no public
      transactional dataset has real channel attribution).
- [x] `sql/analytics/acquisition_cohorts.sql`, `retention_curves.sql`,
      `rfm_segmentation.sql`, `cac_ltv_by_channel.sql` — all built and run
      via DuckDB.
- [x] `customer/cohorts.py` — runs the above, produces
      `fig_cohort_retention_heatmap.png`, `fig_rfm_segments.png`,
      `fig_ltv_cac_by_channel.png`.
- [x] `customer/clv.py` — BG/NBD + Gamma-Gamma CLV (`lifetimes`), fit on a
      calibration period and validated against actual holdout-period
      revenue, compared against a naive flat-extrapolation baseline.
      **Finding, reported as-is: BG/NBD+Gamma-Gamma did not clearly beat
      naive on this dataset** (MAE 732 vs 659, corr 0.073 vs 0.068) — with
      diagnosed reasons (Gamma-Gamma q=0.33 <1, mild frequency/monetary
      correlation violating the independence assumption, wholesale-account
      outliers). Full writeup: `docs/07-results.md`.
- [ ] `sql/analytics/category_contribution.sql`,
      `delivery_impact_on_repeat.sql` — not feasible on this substitute
      dataset (no category taxonomy, no delivery timestamps); would need
      real Olist access.
- [x] `tests/test_channel_synth.py` — unit tests (3 passing, 8/8 total
      across the repo).

## Phase 3: Experimentation design (geo-lift + power analysis)

Spec explicitly flags this as one of two things to never drop (the other,
parameter recovery, was already done) — prioritized ahead of the causal
identification section and the budget allocator for that reason. Built on
simulated multi-market data (real UCI data is 97%+ single-country by
revenue — not usable for geo matching; see `experiment/geo_simulate.py`
docstring), with a known injected effect so the design itself could be
validated, same discipline as the MMM recovery study.

- [x] `experiment/geo_simulate.py` — 40 simulated DMA-style markets, 26-week
      pre-period + 8-week post-period, known +12% injected lift in one
      treatment market.
- [x] `experiment/geo_matching.py` — matched control selection by
      pre-period trend correlation; produces the matched-pair chart.
- [x] `experiment/power.py` — MDE and required-duration calculations from
      pre-period tracking error.
- [x] `experiment/analyze.py` — the actual DiD test, run on the (known-
      ground-truth) post-period data.
- [x] `docs/08-geo-lift-preregistration.md` — hypothesis, primary metric,
      design, stopping rule, decided in advance.
- [x] `docs/05-experimentation-notes.md` — peeking, fixed sample size,
      multiple comparisons, statistical vs. practical significance, plus
      the validation result.
- [x] `tests/test_power.py` — unit tests (4 passing, 12/12 total across the
      repo).

**Validation result**: the design detected the known +12% injected effect
(estimated 17.1%, z=9.4, p<0.001) — the overshoot is expected single-draw
estimation noise, not a design flaw. The planned 8-week duration turned out
to have more power than strictly needed (MDE ~5.1% vs. the 12% true
effect) — itself a useful finding, not just a "it worked" result. Full
numbers: `outputs/reports/geo_experiment_analysis.csv`.

## Environment notes (matters for reproducing runs)

- No C++ compiler (`g++`/`cl.exe`) is present in this environment. PyMC 6 /
  PyTensor sample fine anyway via a Python/numba fallback — confirmed
  working, just slower than a compiled backend. `numba` is pinned in
  `requirements.txt` for this reason.
- `.venv` created with the system Python 3.12. Packages installed so far:
  numpy, pandas, scipy, matplotlib, pymc, arviz, numba, pytest. Everything
  else in `requirements.txt` is commented out until the phase that needs it
  starts (avoids installing a large stack — duckdb/cvxpy/fastapi/etc — before
  there's code that uses it).
- On Windows, `pm.sample(..., cores>1)` uses multiprocessing with the
  `spawn` start method — scripts calling it guard the entry point with
  `if __name__ == "__main__":`. Confirmed `cores=4` works and roughly
  4x's throughput vs `cores=1`.
- **arviz 1.3.0 API note** (this pip-installed version, not the older/more
  commonly-documented arviz API): `az.summary(idata, ci_prob=..., ci_kind=
  "hdi")` not `hdi_prob=...`; `az.hdi(values, prob=...)` not `hdi_prob=...`;
  `az.plot_trace(...)` no longer accepts a `compact` kwarg. Saving
  InferenceData to netcdf (`idata.to_netcdf(...)`) requires `h5py` (or
  `netCDF4`) installed separately — not pulled in automatically by
  pymc/arviz. All fixed in the current code; noting here so a future
  session doesn't re-diagnose the same three errors.

## Weeks 2–4 (spec plan, not started)

- Week 2: full diagnostics suite is actually already done as part of Week 1's
  recovery study (diagnostics are shared machinery between recovery and any
  future real-data fit) — response curves per channel are not yet built as a
  standalone deliverable (currently only exist implicitly via the recovery
  forest plots).
- Week 3: customer layer (cohorts, CLV) — done (Phase 2). Experiment design —
  done (Phase 3). Causal identification (DiD/IV) — not started.
- Week 4: allocator, dashboard, deploy, docs — not started.

## Cut order if time runs short (per spec, restated here as a reminder)

Drop, in this order, before dropping anything else: IV analysis (keep DiD),
Weibull adstock (keep geometric — already the current state), budget sweep.
**Never drop the parameter recovery study or the experiment design** — both
are now done (Phases 1 and 3). Remaining work (causal identification,
allocator, dashboard) can be prioritized/cut freely against this list
without touching either protected piece.
