# Data sources and honesty

Genuine multi-channel marketing spend data is proprietary. No public dataset
gives you real spend-by-channel alongside real revenue for the same
business. This project handles that constraint by being explicit about which
layer runs on what kind of data, rather than blending simulated and real
data and letting a reader assume it's all one thing.

## Tier 1 — Simulation for validation (built, status: done)

`simulate/dgp.py` is a data generating process with fully known ground
truth: channel elasticities (`beta`), geometric adstock decay (`alpha`),
Hill saturation shape/half-saturation (`s`, `k`), trend, two-harmonic
seasonality, a price effect, and cross-channel spend correlation via a
shared latent "budget cycle" factor. `simulate/recovery_study.py` fits the
real model (`mmm/model.py`) to this data and checks whether the posterior
recovers the known parameters. Results: `outputs/reports/recovery_report.md`.

This is the most rigorous thing this project can do about the "no real
ground truth" problem, and it is why it was built first, before any modeling
choice was locked in.

## Tier 2 — Reference datasets for the modelling spine (status: built, joined, fit attempted)

**`ingest/robyn_reference.py`** pulls Meta Robyn's own demo dataset
(`dt_simulated_weekly`) directly from Robyn's public GitHub repo —
distributed as an R `.RData` binary, parsed in Python via `pyreadr` with
no R installation needed. 208 weeks (2015-11-23 to 2019-11-11), 5 real
channel spend series (TV, out-of-home, print, Facebook, search) and real
revenue, **as simulated by Meta for Robyn's own documentation** — not
this project's simulation, and not a real business's real numbers either.
Stated plainly in that module's docstring.

**`ingest/build_tier2_panel.py`** joins this spine with the Tier 3 real
covariates below, re-fetched for Robyn's exact historical window so the
join is genuinely aligned in time (not just date-stamped the same by
coincidence) — the first real external data this project has actually fed
into a model fit, closing the gap previously stated here.

**Result, reported as-is**: **`mmm/fit_tier2.py`** fit the same Bayesian
MMM used throughout this project (extended with an optional
`extra_covariates` parameter, backward-compatible — see `mmm/model.py`) on
this joined panel. The posterior is **not usable**: max r-hat 3.43, 86.6%
of post-tuning draws diverged. A higher `target_accept` (0.99) and more
tuning were tried and made essentially no difference (r-hat 3.43 → 3.43,
divergence rate 89% → 87%) — ruling out "needs more tuning" and confirming
this is structural. The cause, checked directly: 4 of the 5 channels in
this dataset have **51–59% zero-spend weeks** (realistic flighted/pulsed
media planning — campaigns pause and resume), which doesn't give the
adstock/Hill-saturation parameters enough spend-level variation to
identify from. This is the *same* failure mode `causal/mmm_vs_did.py`
found on a single simulated channel with a step-function spend pattern —
now confirmed on a second, independent, real-shaped dataset with five
channels at once. Full report: `outputs/reports/tier2_fit_report.md`.
Figures (`fig_tier2_posterior_predictive.png`, `fig_tier2_coefficients.png`)
carry the warning directly on the image, since both look deceptively
reasonable at a glance — the posterior predictive check tracks revenue's
shape well, and the coefficient forest plot shows plausible-looking
intervals, neither of which means the parameters are actually identified.

**Why this is reported rather than hidden or "fixed" by removing the
problem channels**: a real MMM practitioner runs into exactly this —
pulsed campaigns are completely normal media planning, not a data error —
and the honest, useful finding is that the model's identification
requirements (the same ones validated favorably in `docs/04` on
continuously-varying simulated spend) are a real constraint on which real
campaign histories this approach can actually be fit to, not a detail to
paper over.

## Tier 3 — Real external covariates (status: built, verified, joined to Tier 2 — see above)

All four planned sources are built and confirmed pulling real, live data:

- **`ingest/fred_client.py`** — FRED macro series (`UMCSENT`, `CPIAUCSL`,
  `RSXFS`, `PCE`) via the official FRED API. Requires a free API key
  (`.env`, gitignored — see `.env.example`). Verified: 140 real monthly
  observations, 2015-01 through 2026-08.
- **`ingest/trends_client.py`** — Google Trends brand/category search
  interest via `pytrends`, no key needed. Verified: 262 real weekly
  observations.
- **`ingest/weather_client.py`** — Open-Meteo daily temperature and
  precipitation, no key needed. Verified: 2,112 real daily observations
  (303 weekly, aggregated).
- **`ingest/holidays_calendar.py`** — US holiday calendar via the
  `holidays` package, plus hand-added Black Friday/Cyber Monday (computed,
  not looked up — the `holidays` package treats these as informal, not
  public, holidays). No network call. Verified: 87 real holiday events,
  2021–2026.

These are joined to the Tier 2 Robyn spine above (not to `simulate/dgp.py`'s
own simulated spend/revenue, which has no actual causal relationship to
real-world consumer sentiment, search interest, or weather — joining real
covariates to *that* series would still produce meaningless coefficients
dressed up as if they meant something, regardless of Tier 2 existing).
Tested where their own logic is deterministic
(`tests/test_holidays_calendar.py`; FRED/Trends/weather need live network
calls, not suitable for a committed test suite).

## Tier 4 — Real customer-level data (status: done, substituted dataset)

The spec's first choice, Olist, is hosted on Kaggle, which requires an
authenticated API token to download programmatically (`kaggle datasets
download` needs `~/.kaggle/kaggle.json`) — not available in this
non-interactive build environment (confirmed: an unauthenticated request to
Kaggle's download endpoint returns 404). Rather than block on that, this
project uses the spec's own named alternative instead (Section 3, Tier 4:
"Alternative if you prefer one unified dataset: UCI Online Retail II"):

**UCI Online Retail II** — https://archive.ics.uci.edu/dataset/502/online+retail+ii,
directly downloadable with no authentication. 1,067,371 raw transaction line
items, a real UK-based online retailer, Dec 2009–Dec 2011. After removing
cancelled invoices and line items with no customer ID (see
`ingest/uci_retail_loader.py::clean`): 805,549 rows, 5,878 real customers.
This is real transactional data — not simulated — loaded into
`data/processed/warehouse.duckdb` and used for the cohort/retention/RFM/CLV
layer (`customer/`, `sql/analytics/`).

**What this substitution costs**, stated plainly: Online Retail II has no
product-category taxonomy and no delivery/shipping timestamps, so two of
the spec's six planned analyses aren't feasible on this dataset —
`category_contribution.sql` and `delivery_impact_on_repeat.sql` are not
built (Olist, if it becomes accessible, would support both). It also has
**no acquisition-channel field** — see the synthetic-channel note below,
which is a UCI limitation but would equally have been an Olist limitation;
no public order-level dataset ships real marketing-channel attribution.

**Synthetic acquisition channel** (`customer/channel_synth.py`): the
CAC/LTV-by-channel analysis the spec calls for needs a channel dimension
that doesn't exist in real order data. Each customer is assigned a channel
deterministically (seeded, reproducible) with a fixed illustrative CAC per
channel. This is exactly the same "Tier 1 simulation" logic applied to a
different variable — and it is labeled everywhere it surfaces (module
docstring, SQL comment, chart titles, report text) as illustrative, not a
real finding about real channel economics. Full results and the honesty
caveat are in `docs/07-results.md`.

This is a **separate dataset** from the MMM media spine (Tiers 1–3) and is
never silently merged with it — the cohort layer and the MMM layer
demonstrate different capabilities on different data. Any place they're
compared (Section 7.3 of the spec: "MMM says a channel is efficient but the
cohort says its customers churn fast") will say so explicitly — not yet
possible since the MMM spine is still Tier 1 simulation only.

## Current state of this file

As of this build pass, all four tiers are implemented: Tier 1 (simulator),
Tier 2 (Robyn reference spine, joined with real Tier 3 covariates and fit
— result: an unusable posterior, reported honestly, not a success story),
Tier 3 (real external covariates), and Tier 4 (real customer data).
`progress.md` tracks which tiers/layers are live and the full build
history.
