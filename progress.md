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
| Data ingestion (Robyn/Meridian spine, Trends, FRED, weather, Olist) | Not started |
| SQL analytics (cohorts, retention, RFM, CAC/LTV, category, delivery) | Not started |
| Causal identification section + DiD/IV | Not started |
| Customer layer (BG/NBD, Gamma-Gamma CLV) | Not started |
| Experimentation design (geo-lift, power analysis) | Not started |
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
- Week 3: customer layer (cohorts, CLV) + causal identification + experiment
  design — not started.
- Week 4: allocator, dashboard, deploy, docs — not started.

## Cut order if time runs short (per spec, restated here as a reminder)

Drop, in this order, before dropping anything else: IV analysis (keep DiD),
Weibull adstock (keep geometric — already the current state), budget sweep.
**Never drop the parameter recovery study or the experiment design** — of
those two, parameter recovery is done; experiment design is not started yet
and should be prioritized over layers like the dashboard if time is tight.
