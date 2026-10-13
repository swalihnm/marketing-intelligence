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

## Tier 2 — Reference datasets for the modelling spine (status: not started)

Planned: Meta Robyn's demo dataset (`dt_simulated_weekly`,
`dt_prophet_holidays`) and/or Google Meridian's sample data, as the media
spine once the model has been validated on Tier 1. These are themselves
partly simulated by their publishers — that will be stated wherever they're
used, not implied to be real spend.

## Tier 3 — Real external covariates (status: ingestion built and verified; not joined to a spine)

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

**What's deliberately *not* done**: none of these four real covariate
series have been joined to the MMM and used in a model fit. That would
require a real (or Tier-2 reference) media spine to join them to —
`simulate/dgp.py`'s simulated spend/revenue series has no actual causal
relationship to real-world consumer sentiment, search interest, or
weather, so regressing simulated revenue on real macro data would produce
meaningless coefficients dressed up as if they meant something. Rather
than do that for the sake of "using" the real data, these four ingestion
modules are built, tested where their logic is deterministic
(`tests/test_holidays_calendar.py`), and ready to join to a real media
spine if Tier 2 is ever built — not wired into a fit that would be
scientifically empty.

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

As of this build pass: Tier 1 (simulator), Tier 3 (real external
covariates — ingestion only, not joined to a spine), and Tier 4 (real
customer data) are implemented. Tier 2 (a real/reference media spine) is
the one piece that would make Tier 3's data usable in an actual MMM fit,
and remains not started. `progress.md` tracks which tiers/layers are live.
