# Trimmed API — Vercel deployment

A reduced-dependency deployment of 4 of the project's API endpoints
(`allocate`, `response-curve`, `power`, `contribution`), for live hosting on
Vercel. **Not the full `app/main.py` FastAPI app** — that depends on PyMC,
PyTensor, numba, and arviz, which are too heavy/slow-cold-starting for a
serverless function (see `progress.md` and `docs/06-assumptions-and-limitations.md`
for the reasoning). This directory is a self-contained, numpy+scipy-only
reimplementation of the same logic (plus FastAPI itself, for the web layer),
serving a **pre-fitted posterior** (exported once from
`outputs/reports/idata_recovery.nc` via `deploy/export_for_vercel.py`)
rather than fitting live.

## Structure: single-entrypoint FastAPI app, not multi-file `/api` handlers

The first attempt at this deployment used Vercel's older
`BaseHTTPRequestHandler`-per-file convention (one `.py` per route under
`api/`). That convention turned out to only be reliably auto-detected for
*existing* Vercel projects, not brand-new ones deployed via drag-and-drop —
a fresh deploy failed with "No python entrypoint found in default
locations." Vercel's currently-documented, zero-config path for new Python
projects is a single `app.py` exposing a FastAPI `app` object
(https://vercel.com/docs/frameworks/backend/fastapi), which is what this
directory now uses.

## What's NOT in this deployment

- Live MMM model fitting (PyMC) — the posterior is exported once, offline,
  and bundled as static JSON data (`data/posterior_draws.json`,
  `data/history.json`).
- The customer economics layer, causal identification (DiD/IV/MMM-vs-DiD),
  and geo-lift experiment modules — those aren't served by an API at all in
  this project; they're scripts run locally, producing the charts and CSVs
  in `outputs/`.
- CVXPY (never used — see `allocate/optimizer.py`).

## Files

```
app.py                  FastAPI app — all routes (GET /, /health, POST /allocate,
                         GET /response-curve/{channel}, GET /contribution,
                         POST /experiment/power)
_core.py                 dependency-light core logic: hill/adstock math, the
                          allocator (scipy trust-constr, same approach as
                          allocate/optimizer.py), power analysis
requirements.txt         fastapi + numpy + scipy
data/
  posterior_draws.json    80 posterior draws x {beta,alpha,s,k} x 4 channels
  history.json            156-week simulated revenue/spend series
```

## Regenerating the data files

If the MMM posterior changes (a fresh `python -m simulate.recovery_study`
run), regenerate the exported data before redeploying:

```bash
python -m deploy.export_for_vercel
```

## Running locally

```bash
cd deploy/vercel-api
pip install -r requirements.txt
uvicorn app:app --reload
```

## Deployment

First attempted via the Vercel MCP connector directly from this session,
which failed with a 403 "you don't have permission to create a project" —
an account/role restriction, not an engineering issue. Deployed instead via
Vercel's dashboard "Drop to Deploy" (drag-and-drop this folder), which
doesn't hit the same permission check as programmatic project creation. See
`progress.md` for the deployment record (URL, date, what was verified).
