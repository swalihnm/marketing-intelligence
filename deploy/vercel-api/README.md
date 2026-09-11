# Trimmed API — Vercel deployment

A reduced-dependency deployment of 4 of the project's API endpoints
(`allocate`, `response_curve`, `power`, `contribution`), for live hosting on
Vercel. **Not the full `app/main.py` FastAPI app** — that depends on PyMC,
PyTensor, numba, and arviz, which are too heavy/slow-cold-starting for a
serverless function (see `progress.md` and `docs/06-assumptions-and-limitations.md`
for the reasoning). This directory is a self-contained, numpy+scipy-only
reimplementation of the same logic, serving a **pre-fitted posterior**
(exported once from `outputs/reports/idata_recovery.nc` via
`deploy/export_for_vercel.py`) rather than fitting live.

## What's NOT in this deployment

- Live MMM model fitting (PyMC) — the posterior is exported once, offline,
  and bundled as static JSON data (`data/posterior_draws.json`,
  `data/history.json`).
- The customer economics layer, causal identification (DiD), and geo-lift
  experiment modules — those aren't served by an API at all in this
  project; they're scripts/notebooks-style modules run locally, producing
  the charts and CSVs in `outputs/`.
- CVXPY (never used — see `allocate/optimizer.py`).

## Files

```
index.html              self-documenting landing page
requirements.txt        numpy + scipy only
api/
  _shared.py             core logic: hill/adstock math, allocator (scipy
                          trust-constr, same approach as allocate/optimizer.py),
                          power analysis. Not a route itself — files
                          prefixed with `_` are excluded from Vercel's
                          routing but importable by sibling functions.
  health.py               GET  /api/health
  allocate.py             POST /api/allocate
  response_curve.py       GET  /api/response_curve?channel=tv
  power.py                POST /api/power
  contribution.py         GET  /api/contribution?start=&end=
data/
  posterior_draws.json    300 posterior draws x {beta,alpha,s,k} x 4 channels
  history.json            156-week simulated revenue/spend series
```

## Regenerating the data files

If the MMM posterior changes (a fresh `python -m simulate.recovery_study`
run), regenerate the exported data before redeploying:

```bash
python -m deploy.export_for_vercel
```

## Deployment

Deployed via the Vercel MCP connector directly from this session — not
through a git-triggered pipeline. To redeploy after a change, the file tree
under this directory is pushed again the same way. See `progress.md` for
the deployment record (URL, date, what was verified).
