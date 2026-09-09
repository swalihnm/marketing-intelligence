# Assumptions and limitations

Updated for the completed build (all four phases — see `progress.md`).
Grouped by what's actually still a gap, not a changelog.

## Data

- The MMM media spine is, and remains, entirely simulated
  (`simulate/dgp.py`). Real multi-channel spend data is proprietary; Tier
  2/3 ingestion (Robyn/Meridian reference spine, Google Trends, FRED,
  weather) was never built — see `docs/01-data-sources-and-honesty.md`.
  Every MMM result in this repo (recovery, response curves, the allocator,
  the DiD comparison) is on simulated data with known ground truth, not
  real spend.
- The customer economics layer runs on real data (UCI Online Retail II),
  substituted for the spec's Olist (Kaggle auth unavailable here) — see
  `docs/01`. It has no acquisition-channel field; CAC/LTV-by-channel uses a
  documented synthetic channel assignment, not real attribution.
- The MMM and customer-economics layers use genuinely different datasets
  (one simulated, one real) and are never merged. The spec's suggested
  cross-check ("MMM says a channel is efficient but its customers churn
  fast") isn't possible here for that reason — there's no single dataset
  where both a channel's MMM contribution and its customers' retention are
  observed together.

## Modelling

- Geometric adstock only. Weibull (delayed-peak) adstock is on the spec's
  own cut list and was not implemented.
- MMM coefficients (`beta_c`) are descriptive, not causal, except where
  cross-checked — and only `causal/did.py`'s specific simulated scenario
  has been cross-checked (against its own known ground truth, not against
  the MMM's `beta_c` for the same effect — those two are still separate
  simulations; see `docs/03-causal-identification.md`'s "not yet done"
  section). Reverse causality from budget set as a percentage of forecast
  revenue — the single biggest threat the spec names — is not addressed or
  simulated anywhere in this codebase.
- IV analysis was not implemented, per the spec's own cut order ("drop IV,
  keep DiD").
- The budget allocator's response function uses a steady-state adstock
  approximation (`spend / (1 - alpha)`), valid for constant weekly spend at
  the candidate allocation — it does not model ramp-up/ramp-down dynamics
  from a spend change, only the eventual steady state.
- The allocator uses scipy `trust-constr`, not CVXPY as the spec lists —
  documented in `allocate/optimizer.py`: the Hill saturation function is
  non-convex for shape parameter s>1 (which the fitted posterior regularly
  draws), so the allocation problem isn't convex and CVXPY doesn't apply
  directly. `trust-constr` isn't guaranteed globally optimal, but is
  reliable for this problem size, and is checked, not assumed — SLSQP was
  tried first and found to silently return infeasible allocations from some
  starting points, which `tests/test_api.py` caught.

## Experimentation

- The geo-lift design and power analysis are validated against a **known
  injected effect** on simulated multi-market data
  (`experiment/geo_simulate.py`) — real UCI retail data is 97%+
  single-country by revenue and unusable for genuine geo matching (see
  `docs/01`). No live or real experiment has been run; this is a validated
  design and a pre-registration document, not a result.

## Serving

- The FastAPI app (`app/main.py`) and Streamlit dashboard
  (`app/dashboard.py`) are built, and both were manually confirmed to serve
  real HTTP requests correctly (not just via test client) during
  development. **Neither is deployed to a public/live URL.** That needs
  cloud hosting credentials (a Vercel/Fly/Render/similar account) that
  aren't available in this non-interactive build environment — stated
  plainly rather than glossed over or faked. A `Dockerfile` is included but
  **not built or run** in this environment either (Docker isn't installed
  here); it follows standard patterns but is unverified beyond a read-
  through.

## Environment

- Sampling runs on a Python/numba fallback backend (no C++ compiler in this
  environment) — functionally correct, but slower than a compiled backend;
  runtimes noted in `docs/02-mmm-methodology.md` reflect that.

See `progress.md` for the phase-by-phase build log this file summarizes.
