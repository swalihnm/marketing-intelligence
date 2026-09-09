# Marketing Mix Modelling, Budget Allocation & Customer Economics

A Bayesian marketing mix model (MMM) that estimates the incremental revenue
contribution of each marketing channel, a constrained budget allocator built
on top of the posterior, and a customer economics layer (cohorts, retention,
CAC/LTV). Built from the spec in `PROJECT-03-marketing-mix-modelling.md`.

The deliverable is not an R-squared. It's a **budget reallocation
recommendation with uncertainty bands**, plus a stated plan (a designed
geo-lift experiment) to prove it.

See `progress.md` for what's built vs. planned — this is a multi-week build
and that file is the source of truth on current status, updated as work
lands.

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

Run it yourself:

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
python -m simulate.recovery_study
```

Outputs land in `outputs/figures/` and `outputs/reports/` — see
`outputs/reports/recovery_report.md` after running.

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

## Repository layout

```
simulate/       ground-truth DGP + parameter recovery harness           [built]
transforms/     adstock (geometric) + saturation (Hill)                  [built]
mmm/            Bayesian MMM in PyMC — priors, model, sampling           [built]
ingest/         UCI Online Retail II loader [built]; Robyn/Meridian, Trends, FRED, weather [planned]
sql/analytics/  cohorts, retention, RFM, CAC/LTV [built]; category, delivery [not feasible — see docs/01]
causal/         DiD / IV supporting identification analysis               [planned]
customer/       cohorts, channel_synth, BG/NBD + Gamma-Gamma CLV          [built]
allocate/       posterior-based constrained budget optimizer              [planned]
experiment/     geo-lift design + power analysis                          [built]
app/            FastAPI + dashboard, deployed                             [planned]
docs/           methodology, data honesty, results, limitations           [in progress]
```

## Stack

Python 3.12, PyMC 5/6 + ArviZ (no C++ compiler required — falls back to a
pure-Python/numba backend automatically), NumPy/Pandas/SciPy/Matplotlib,
DuckDB, `lifetimes` — all installed and in use. Polars, CVXPY,
`statsmodels`, `pytrends`, `fredapi`, FastAPI, Streamlit planned for later
phases (see `requirements.txt`).

## Limitations

See `docs/06-assumptions-and-limitations.md`. In short: the media spine will
be partly simulated because real spend data is proprietary; the parameter
recovery study is what stands in for that missing ground truth; MMM
coefficients are causal only under stated, documented assumptions.
