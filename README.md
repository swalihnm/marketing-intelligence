# Marketing Mix Modelling, Budget Allocation & Customer Economics

A Bayesian marketing mix model (MMM) that estimates the incremental revenue
contribution of each marketing channel, a constrained budget allocator built
on top of the posterior, a customer economics layer (cohorts, retention,
CAC/LTV, CLV), a validated geo-lift experiment design, a difference-in-
differences causal check, and a FastAPI + Streamlit serving layer. Built
from the spec in `PROJECT-03-marketing-mix-modelling.md`.

The deliverable is not an R-squared. It's a **budget reallocation
recommendation with uncertainty bands** (£63,958/week expected gain at the
same total budget, 90% credible interval £35,817–£92,071 — see
`docs/07-results.md`), backed by a
validated experiment design and a causal-identification cross-check, not
just a fitted model taken on faith.

**Status: all four build phases complete, plus the correlation-sensitivity
sweep, reverse-causality/IV, and MMM-vs-DiD reconciliation — 37/37 tests
passing.** A trimmed budget-allocator API is built, locally tested, and
ready to deploy to Vercel — currently blocked on a permission on the
connected Vercel account (see `progress.md`, Phase 5). See `progress.md`
for the full build log and `docs/06-assumptions-and-limitations.md` for
every honestly-stated gap.

## Why this project has a data section before a modelling section

Genuine multi-channel marketing spend data is proprietary. This project does
not pretend otherwise. See `docs/01-data-sources-and-honesty.md` for exactly
which parts of the pipeline run on simulated data with known ground truth,
which run on public reference datasets, and which run on real transactional
data (UCI Online Retail II, substituted for the spec's original choice,
Olist — Kaggle needs an auth token this environment doesn't have; the doc
explains the substitution and what it costs).

## Headline result: parameter recovery

Before trusting any MMM output on real-shaped data, this project asks a
sharper question first: **if you know the true channel elasticities, adstock
decay rates, and saturation curves that generated a revenue series, does the
Bayesian model's posterior recover them?**

`simulate/dgp.py` generates weekly revenue data under known parameters
(including correlated cross-channel spend, the actual enemy of MMM
identification). `mmm/model.py` fits the same model a real analysis would
use. `simulate/recovery_study.py` fits the model to the simulated data and
reports coverage of the true parameters inside the posterior HDI, r-hat,
divergences, and forest-plot recovery figures.

**Result**: 100% of the 16 headline channel parameters recovered inside
their 90% HDI, max r-hat 1.0033, zero divergences.

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python -m simulate.recovery_study
```

Outputs land in `outputs/figures/` and `outputs/reports/` — see
`outputs/reports/recovery_report.md` after running.

**Correlation-sensitivity sweep** (`simulate/correlation_sensitivity.py`,
`docs/09-correlation-sensitivity.md`): the same validation repeated at
spend correlation 0.3 → 0.9. Coverage held at 100% throughout; point-
estimate accuracy degraded monotonically (beta's mean error: 28% → 39%) —
exactly the calibrated-but-less-precise behavior you want from a Bayesian
estimator under rising multicollinearity.

## Budget allocator

`allocate/optimizer.py` optimizes expected incremental revenue over the
full posterior (300 draws from the fitted MMM), not point estimates —
subject to total-budget, per-channel-cap, and max-organizational-shift
constraints. Uses scipy `trust-constr`, not CVXPY (documented why: Hill
saturation is non-convex for the shape parameters the posterior draws).

```bash
python -m allocate.charts   # runs the allocation, produces response curves, marginal ROI, budget sweep
```

## Customer economics layer

Real transactional data (UCI Online Retail II — 805,549 cleaned line items,
5,878 customers, Dec 2009–Dec 2011): cohorts, retention, RFM segmentation,
CAC/LTV by (synthetic — see docs) channel, and BG/NBD + Gamma-Gamma CLV
validated on a holdout period against a naive baseline. Headline finding
reported as-is, not massaged: **the CLV model didn't clearly beat naive
extrapolation on this dataset**, with diagnosed reasons. Full results:
`docs/07-results.md`.

```bash
python -m ingest.uci_retail_loader   # downloads + cleans + builds the warehouse
python -m customer.cohorts           # cohorts, retention, RFM, CAC/LTV
python -m customer.clv               # BG/NBD + Gamma-Gamma, holdout-validated
```

## Experimentation design: geo-lift + power analysis

The spec flags this as one of two things to never cut (parameter recovery
is the other, and is done). Built on 40 simulated DMA-style markets with a
**known injected effect**, so the design itself could be validated the same
way the MMM was: matched-market selection by pre-period trend correlation,
power analysis (minimum detectable effect vs. test duration), and a
difference-in-differences test that — checked against the known
ground-truth effect — actually detects it (+12% true, +17.1% estimated,
p<0.001). Pre-registration doc: `docs/08-geo-lift-preregistration.md`. Full
writeup, including why the point estimate overshoots: `docs/05-experimentation-notes.md`.

```bash
python -m experiment.analyze   # runs the simulation, matching, and DiD test end to end
```

## Causal identification

Three cross-checks against known simulated effects, all in
`docs/03-causal-identification.md`:

- `causal/did.py`: a channel launches in one region and not another, with
  a real shared confounder (a regional trend). Naive before/after estimate
  was **70.9% off** true; DiD (non-launch region as control) was **3.2%
  off**.
- `causal/reverse_causality.py`: budget set partly as a function of
  forecast demand — the spec's named biggest causal threat, addressed with
  an exogenous cost-shifter instrument ("platform CPM shock," the spec's
  own suggestion). Naive OLS was **14.8% off** true beta; IV/2SLS was
  **0.14% off**.
- `causal/mmm_vs_did.py`: fits a single-channel MMM and DiD on the
  *literal same* simulated effect. The MMM's point estimate looked fine
  (2.5% off) but its posterior was badly broken (r-hat 3.43, divergences
  in 64% of draws) — a newly-launched channel's narrow spend history
  doesn't give MMM enough variation to identify, a limitation DiD doesn't
  share. The chart carries this warning directly on the image so the
  apparent agreement can't be misread without checking diagnostics.

```bash
python -m causal.did
python -m causal.reverse_causality
python -m causal.mmm_vs_did
```

## Serving layer: API + dashboard

`app/main.py` (FastAPI, all 4 spec endpoints) and `app/dashboard.py`
(Streamlit, all 6 spec views). Both manually confirmed to serve real HTTP
requests correctly during development — **not deployed to a public URL**
(no cloud hosting credentials in this environment; stated directly rather
than glossed over).

```bash
uvicorn app.main:app --reload        # API at http://127.0.0.1:8000/docs
streamlit run app/dashboard.py       # dashboard at http://127.0.0.1:8501
```

## Testing

```bash
python -m pytest tests/ -v
```

37 tests across transforms, the allocator (including a real bug the tests
caught — see `progress.md`), causal identification (DiD, reverse-causality/
IV, MMM-vs-DiD), power analysis, the synthetic channel assignment, and the
FastAPI app (via `TestClient`, no server needed).

## Repository layout

```
simulate/       ground-truth DGP + parameter recovery harness           [built]
transforms/     adstock (geometric) + saturation (Hill)                  [built]
mmm/            Bayesian MMM in PyMC — priors, model, sampling           [built]
ingest/         UCI Online Retail II loader [built]; Robyn/Meridian, Trends, FRED, weather [not built]
sql/analytics/  cohorts, retention, RFM, CAC/LTV [built]; category, delivery [not feasible — see docs/01]
causal/         DiD, reverse-causality/IV, MMM-vs-DiD reconciliation — all validated [built]
customer/       cohorts, channel_synth, BG/NBD + Gamma-Gamma CLV          [built]
allocate/       posterior-based constrained budget optimizer + charts    [built]
experiment/     geo-lift design + power analysis, validated              [built]
app/            FastAPI + Streamlit dashboard, built and locally tested; not deployed
docs/           methodology, data honesty, results, limitations           [built]
```

## Stack

Python 3.12, PyMC 5/6 + ArviZ (no C++ compiler required — falls back to a
pure-Python/numba backend automatically), NumPy/Pandas/SciPy/Matplotlib,
DuckDB, `lifetimes`, FastAPI, Streamlit — all installed and in use. CVXPY
deliberately not used (see `allocate/optimizer.py`). Polars, `statsmodels`,
`pytrends`, `fredapi` not installed — MMM media-spine ingestion beyond the
simulator was never built (see `progress.md`).

## Limitations

See `docs/06-assumptions-and-limitations.md` for the full list. Highlights:
the MMM media spine is entirely simulated (real spend data is proprietary
and Tier 2/3 ingestion was never built — Trends/weather/holidays need no
key and could be added; FRED needs a free key only the user can get); the
reverse-causality IV correction is linear-only, not extended to the full
nonlinear adstock+Hill model; and nothing in this repo is deployed to a
live URL yet (a trimmed API is ready, blocked on a Vercel permission).
