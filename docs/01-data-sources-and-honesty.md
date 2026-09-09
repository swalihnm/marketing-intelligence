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

## Tier 3 — Real external covariates (status: not started)

Planned: Google Trends (via `pytrends`, brand search as an organic demand /
mediator proxy), FRED macro series (`UMCSENT`, `CPIAUCSL`, `RSXFS`, `PCE`),
Open-Meteo weather, and a holiday calendar. These are real data, used as
control covariates alongside the Tier 2 media spine.

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

As of this build pass, only Tier 1 is implemented. `progress.md` tracks
which tiers/layers are live.
