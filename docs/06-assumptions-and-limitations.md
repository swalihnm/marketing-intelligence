# Assumptions and limitations

Updated through the real-data ingestion and Tier 2 fit pass — see
`progress.md`. Grouped by what's actually still a gap, not a changelog.

## Data

- The project's headline MMM results (parameter recovery, correlation
  sensitivity, the allocator) are all on Tier-1 **simulated** data with
  known ground truth (`simulate/dgp.py`) — that remains true and is the
  basis for those results' validation claims.
- Real Tier 2/3 data has since been pulled in and actually used in a fit:
  Meta's own Robyn reference demo spine (`ingest/robyn_reference.py`) and
  real FRED/Google Trends/weather covariates
  (`ingest/fred_client.py`, `trends_client.py`, `weather_client.py`),
  joined (`ingest/build_tier2_panel.py`) and fit (`mmm/fit_tier2.py`). That
  fit's posterior is **not usable** (see Modelling, below) — real data
  being available doesn't mean every fit on it succeeds, and this one
  didn't, for a diagnosed and documented reason.
- The customer economics layer runs on real data (UCI Online Retail II),
  substituted for the spec's Olist (Kaggle auth unavailable here) — see
  `docs/01`. It has no acquisition-channel field; CAC/LTV-by-channel uses a
  documented synthetic channel assignment, not real attribution.
- The MMM and customer-economics layers use genuinely different datasets
  and are never merged. The spec's suggested cross-check ("MMM says a
  channel is efficient but its customers churn fast") isn't possible here
  for that reason — there's no single dataset where both a channel's MMM
  contribution and its customers' retention are observed together.

## Modelling

- Geometric adstock only. Weibull (delayed-peak) adstock is on the spec's
  own cut list and was not implemented.
- MMM coefficients (`beta_c`) are descriptive, not causal, except where
  cross-checked — `causal/did.py` (DiD vs. naive), `causal/reverse_causality.py`
  (reverse-causality bias vs. IV correction), and `causal/mmm_vs_did.py`
  (MMM vs. DiD on the literal same simulated effect) all cross-check
  against known ground truth. See `docs/03-causal-identification.md`.
- `causal/mmm_vs_did.py` surfaced a real limitation, not a hypothetical
  one: a single-channel MMM fit on a newly-launched channel's short, low-
  variation spend history produces a severely pathological posterior
  (r-hat 3.43, divergences in 64% of draws) even though its point estimate
  happens to land close to the true effect — the point estimate alone
  would have looked fine without checking diagnostics. New channels with
  narrow spend-range histories are a genuine MMM blind spot that DiD
  doesn't share (DiD only needs a comparable control unit, not spend-level
  variation to trace a saturation curve from). **Confirmed independently**
  on a second, real-shaped dataset: `mmm/fit_tier2.py`'s fit on Meta's real
  Robyn reference spine (4 of 5 channels with 51–59% zero-spend weeks)
  produced the identical signature — r-hat 3.43, 86.6% divergent draws,
  unchanged by a higher `target_accept` and more tuning. This is now a
  twice-confirmed, not a one-off, limitation.
- IV was implemented (`causal/reverse_causality.py`), despite the spec's
  own cut order allowing it to be dropped, because it directly closes the
  "reverse causality" gap the spec calls the single biggest threat — but
  only on a simplified **linear** spend→revenue specification. Extending
  it to the full nonlinear adstock+Hill model was then attempted
  (`causal/nonlinear_iv.py`, a control-function approach) and **did not
  work**: the naive nonlinear fit was already close to true (0.7% off),
  and the control-function correction made it *worse* (14.7% off) because
  the instrument was too weak relative to the endogeneity mechanism
  (first-stage R² = 0.33), leaving the "control" residual nearly collinear
  with the channel's own spend (r = 0.82) rather than cleanly isolating
  the confound. A stronger instrument or a more principled joint-estimation
  approach would be needed — not attempted here. See `docs/03`.
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

- The full FastAPI app (`app/main.py`) and Streamlit dashboard
  (`app/dashboard.py`) are built and were manually confirmed to serve real
  HTTP requests correctly during development, but are not deployed (too
  heavy for serverless — PyMC/PyTensor/numba/arviz).
- A trimmed, numpy+scipy+FastAPI-only version (`deploy/vercel-api/`) **is
  deployed and live**, by the user, via Vercel's dashboard "Drop to
  Deploy" (the connected Vercel MCP integration itself hit a permission
  error creating new projects — not an engineering gap, see `progress.md`).
  Build success was confirmed via the user's own screenshot. The Vercel
  connector has since disconnected from this session entirely, so
  real-HTTP endpoint re-verification is pending the URL again.
- A `Dockerfile` is included but **not built or run** in this environment
  (Docker isn't installed here); it follows standard patterns but is
  unverified beyond a read-through.

## Environment

- Sampling runs on a Python/numba fallback backend (no C++ compiler in this
  environment) — functionally correct, but slower than a compiled backend;
  runtimes noted in `docs/02-mmm-methodology.md` reflect that.

See `progress.md` for the phase-by-phase build log this file summarizes.
