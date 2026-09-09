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

## Tier 4 — Real customer-level data (status: not started)

Planned: Olist Brazilian E-Commerce (~100K real orders, 2016–2018) for the
cohort/retention/CAC/LTV layer (`customer/`, `sql/analytics/`). This is real
transactional data, but it is a **separate dataset** from the MMM media
spine and will never be silently merged with it — the cohort layer and the
MMM layer demonstrate different capabilities on different data, and any
place they're compared (Section 7.3 of the spec: "MMM says a channel is
efficient but the cohort says its customers churn fast") will say so
explicitly.

## Current state of this file

As of this build pass, only Tier 1 is implemented. `progress.md` tracks
which tiers/layers are live.
