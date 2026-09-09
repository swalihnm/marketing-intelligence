# Assumptions and limitations

- The media spine is, at this stage, entirely simulated (`simulate/dgp.py`).
  Real multi-channel spend data is proprietary; nothing in this repo is
  presented as real spend until Tier 2/3 data is wired in (see
  `docs/01-data-sources-and-honesty.md`). The parameter recovery study is
  what validates the estimator in the absence of real ground truth — it
  proves the estimator is unbiased under known conditions, not that it will
  behave identically on real, messier data.
- Geometric adstock only, for now. Weibull (delayed-peak) adstock is in the
  spec's cut list and has not been implemented.
- The customer economics layer (Olist) has not been built yet, so there is
  no cross-check yet between MMM channel efficiency and long-run customer
  value (spec section 7.3).
- No causal identification section or quasi-experimental estimate (DiD/IV)
  exists yet. Until it does, MMM coefficients in this repo should not be
  read causally at all — reverse causality from budget-setting-as-%-of-
  forecast-revenue is a known, unaddressed threat.
- No live or designed experiment exists yet (geo-lift design is planned,
  not built).
- Sampling runs on a Python/numba fallback backend (no C++ compiler in this
  environment) — functionally correct, but slower than a compiled backend;
  runtimes noted in `docs/02-mmm-methodology.md` reflect that.

This file will grow as later layers land; see `progress.md` for what's
current.
