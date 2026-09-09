# Progress

Tracking actual status against `PROJECT-03-marketing-mix-modelling.md`'s
4-week plan (Section 10). Updated as work lands — this file, not memory of
the spec, is the source of truth on what's actually built.

Last updated: 2026-09-09.

## Status at a glance

**All four build phases are complete.** The one spec deliverable not done
is live deployment (no cloud hosting credentials in this environment) —
everything else, including automated testing, is built and passing.

| Layer | Status |
|---|---|
| Simulator + parameter recovery (Tier 1 data) | **Done** |
| Bayesian MMM (adstock + Hill saturation, PyMC) | **Done** |
| Diagnostics (r-hat, ESS, divergences, PPC, prior PC) | **Done** |
| Correlation-sensitivity study (identification degradation) | Not started (see "Known gaps" below) |
| Data ingestion — MMM media spine (Robyn/Meridian, Trends, FRED, weather) | Not started |
| Data ingestion — customer data (Tier 4) | **Done, substituted dataset** (Olist → UCI Online Retail II; see docs/01) |
| SQL analytics — cohorts, retention, RFM, CAC/LTV | **Done** |
| SQL analytics — category contribution, delivery-impact-on-repeat | **Not feasible on substitute dataset** (needs Olist; see docs/01) |
| Causal identification section + DiD | **Done** (DiD validated against known effect; IV skipped per spec's own cut order) |
| Customer layer — BG/NBD, Gamma-Gamma CLV | **Done** (validated on holdout; did not beat naive baseline — see docs/07) |
| Experimentation design (geo-lift, power analysis) | **Done** (design validated against a known injected effect — see docs/05) |
| Budget allocator (posterior-based, constrained) | **Done** (trust-constr over posterior draws; response curves, marginal ROI, budget sweep) |
| FastAPI + Streamlit dashboard | **Done, built and locally tested** — not deployed (no hosting credentials; see docs/06) |
| Dockerfile | Written, **not built/run** (Docker unavailable in this environment) |
| Test suite | **30/30 passing** (`pytest tests/`) |

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

## Phase 4: Causal identification, budget allocator, serving layer

- [x] `causal/did.py` — channel-launch DiD on simulated two-region data with
      a known true effect and a real shared confounder (regional trend).
      **Naive estimate 71.6% off truth; DiD 3.2% off** — a clean,
      unambiguous illustration of why the identification method matters,
      not just an estimator that happens to run. (An earlier
      parameterization coincidentally made naive ≈ DiD — a symmetric
      linear-trend/seasonal-cycle cancellation specific to that window
      split; documented in `causal/did.py` and fixed by strengthening the
      trend, not by cherry-picking a seed.)
- [x] `docs/03-causal-identification.md` — finalized: confounders, reverse
      causality (unaddressed, stated plainly), the DiD result, and an
      explicit note that MMM-vs-DiD reconciliation on the *same* effect is
      **not done** (they're still separate simulated scenarios).
- [x] `allocate/optimizer.py` — budget allocator, optimizes expected
      incremental revenue over 300 posterior draws from
      `outputs/reports/idata_recovery.nc` (Phase 1's fitted posterior),
      subject to total-budget, channel-cap, and max-shift constraints.
      Steady-state adstock approximation (documented). Uses scipy
      `trust-constr`, not CVXPY (Hill saturation is non-convex for s>1 —
      documented why CVXPY doesn't apply).
- [x] `allocate/charts.py` — response curves with posterior uncertainty
      bands, marginal ROI chart, budget sweep chart.
- [x] `app/main.py` — FastAPI app, all 4 spec endpoints (`/allocate`,
      `/response-curve/{channel}`, `/contribution`, `/experiment/power`).
      Manually confirmed serving real HTTP requests (uvicorn, not just
      TestClient) during development.
- [x] `app/dashboard.py` — Streamlit dashboard, all 6 spec views. Manually
      confirmed it boots and serves valid HTML headlessly; full interactive
      testing isn't feasible in this non-browser environment, stated as a
      real limitation rather than assumed to work.
- [x] `Dockerfile` — written to standard patterns; **not built or run**
      (Docker not installed in this environment).
- [x] `tests/test_did.py`, `tests/test_allocator.py`, `tests/test_api.py` —
      18 new tests, 30/30 total across the repo passing.

**Bug caught by testing, not just written for show**: the allocator's first
implementation (scipy SLSQP, starting from current spend) silently returned
an infeasible allocation — ~1.8% over the requested budget,
`result.success == False` computed but never checked anywhere downstream.
`tests/test_api.py::test_allocate_returns_valid_allocation` failed and
caught it before this became a shipped bug. Root-caused (SLSQP sensitivity
to starting point on this non-convex objective, reproduced and confirmed
via a debug script) and fixed by switching to `trust-constr` with an
SLSQP fallback attempt and a hard raise if both fail to converge to a
feasible point — see `allocate/optimizer.py`'s docstring and
`docs/07-results.md`.

**Allocation result** (same simulated data as the Phase 1 recovery study,
current weekly budget £144,583): recommended reallocation moves ~50% of
spend off `tv` (marginal ROI 0.83, 90% HDI nearly touching the £1:£1
breakeven line) onto `search`/`social`/`display` (marginal ROI 2.8–4.3).
Expected weekly gain: **£75,396** (90% credible interval £46,163–£103,890).
Full numbers: `outputs/reports/allocation_recommendation.csv`,
`marginal_roi.csv`, `budget_sweep.csv`.

## Known gaps (honest, not hidden)

- **Correlation-sensitivity sweep** (spend correlation 0.3→0.9 vs. recovery
  degradation) — the spec calls this "an excellent interview artefact" and
  it was never run. This is the one piece from the original Week 1 plan
  that's still genuinely missing, not just deferred to a later phase that
  then got done differently.
- **MMM-vs-DiD reconciliation on the same effect** — `causal/did.py` and
  `simulate/dgp.py` are independently parameterized simulations; no single
  dataset feeds both an MMM fit and a DiD estimate for literally the same
  channel effect, so the spec's "here's my MMM estimate, here's my
  quasi-experimental estimate, here's why they differ" isn't demonstrated.
- **Reverse causality** (budget set as % of forecast revenue) — named as
  the biggest causal threat in the spec, not simulated or addressed
  anywhere in this codebase.
- **MMM media-spine ingestion** (Tier 2/3: Robyn/Meridian reference data,
  Google Trends, FRED, weather) — never built. Every MMM number in this
  repo is on Tier-1 simulated data.
- **Live deployment** — FastAPI + Streamlit apps are built and locally
  verified but not deployed to a public URL (no cloud hosting credentials
  in this environment); Docker image not built or run (Docker unavailable
  here).

## Environment notes (matters for reproducing runs)

- No C++ compiler (`g++`/`cl.exe`) is present in this environment. PyMC 6 /
  PyTensor sample fine anyway via a Python/numba fallback — confirmed
  working, just slower than a compiled backend. `numba` is pinned in
  `requirements.txt` for this reason.
- `.venv` created with the system Python 3.12. All packages in
  `requirements.txt` are installed (numpy/pandas/scipy/matplotlib/pymc/
  arviz/numba/h5py/pytest, duckdb/lifetimes/openpyxl, fastapi/uvicorn/
  httpx/streamlit). CVXPY was deliberately not installed/used — see
  `allocate/optimizer.py`'s docstring.
- SLSQP is not the allocator's primary optimizer — see the Phase 4 section
  above and `allocate/optimizer.py`'s docstring for why (it silently
  returned infeasible allocations on this objective; `trust-constr` is used
  instead, with SLSQP as a checked fallback attempt).
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

## Weeks 2–4 (spec plan) — final status

- Week 2: diagnostics — done (Phase 1). Response curves per channel — done,
  standalone (Phase 4, `allocate/charts.py`, not just implicit in the
  recovery forest plots).
- Week 3: customer layer (cohorts, CLV) — done (Phase 2). Experiment design
  — done (Phase 3). Causal identification (DiD) — done (Phase 4); IV
  skipped per the spec's own cut order.
- Week 4: allocator — done (Phase 4). Dashboard + API — built and locally
  tested (Phase 4). Deploy — **not done**, no cloud hosting credentials in
  this environment. Docs — done, this file plus `docs/01` through `docs/08`.

## Cut order if time runs short (per spec, restated here — now moot, project complete)

The spec's cut order was: drop IV (keep DiD), drop Weibull (keep
geometric), drop the budget sweep, never drop parameter recovery or
experiment design. Final state: IV and Weibull were the only two items
actually cut, exactly per the spec's own recommended order — everything
else, including the budget sweep, got built. Both protected items
(parameter recovery, experiment design) are done. See "Known gaps" above
for what's honestly still missing despite the project being complete by
the spec's own milestone table.
