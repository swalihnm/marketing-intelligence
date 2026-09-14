"""Trimmed budget-allocator API — single-entrypoint FastAPI app, Vercel's
documented zero-config Python deployment pattern
(https://vercel.com/docs/frameworks/backend/fastapi). Depends on numpy,
scipy, and fastapi only — no PyMC/PyTensor/numba/arviz at runtime — serving
a pre-fitted MMM posterior exported via ../export_for_vercel.py. See
README.md in this directory.

Run locally: uvicorn app:app --reload
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from _core import (
    geometric_adstock, hill, load_history, load_posterior,
    minimum_detectable_effect, optimize_allocation, required_duration_weeks,
)

app = FastAPI(
    title="Marketing Mix Modelling — trimmed API",
    description="Budget allocation, response curves, revenue decomposition, and experiment power — serving a pre-fitted Bayesian MMM posterior (numpy+scipy only, no PyMC at runtime).",
)


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html><head><title>MMM Trimmed API</title>
    <style>body{font-family:-apple-system,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;color:#222}
    code,pre{background:#f0f0f0;padding:2px 6px;border-radius:3px}pre{padding:12px}</style></head>
    <body>
    <h1>Marketing Mix Modelling — trimmed API</h1>
    <p>A reduced-dependency (numpy + scipy + FastAPI only) deployment of the budget
    allocator, response curves, and experiment power calculator from a larger
    marketing-mix-modelling project — serving a pre-fitted Bayesian MMM posterior
    (80 draws, 4 simulated channels) rather than fitting live. Full PyMC model fitting,
    the customer economics layer, and the causal identification analysis are not in
    this deployment — they run locally in the full project.</p>
    <p>Interactive API docs: <a href="/docs">/docs</a></p>
    <h2>GET /health</h2>
    <h2>POST /allocate</h2><pre>{"budget": 150000, "max_shift": 0.5}</pre>
    <h2>GET /response-curve/{channel}</h2>
    <h2>POST /experiment/power</h2><pre>{"tracking_error_std": 800, "baseline_weekly_revenue": 15000, "n_weeks": 8}</pre>
    <h2>GET /contribution?start=&end=</h2>
    </body></html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


class AllocateRequest(BaseModel):
    budget: float = Field(..., gt=0)
    channel_caps: dict[str, tuple[float, float]] | None = None
    max_shift: float = Field(0.5, ge=0)


@app.post("/allocate")
def allocate(req: AllocateRequest):
    draws = load_posterior()
    try:
        return optimize_allocation(draws, req.budget, max_shift_frac=req.max_shift, channel_caps=req.channel_caps)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/response-curve/{channel}")
def response_curve(channel: str, n_points: int = 50, max_multiplier: float = 3.0):
    draws = load_posterior()
    if channel not in draws["channels"]:
        raise HTTPException(status_code=404, detail=f"unknown channel '{channel}'. Known: {draws['channels']}")
    idx = draws["channels"].index(channel)
    max_x = draws["current_spend"][channel] * max_multiplier
    spend_range = np.linspace(1.0, max_x, n_points)

    alpha, s, k, beta = draws["alpha"][:, idx], draws["s"][:, idx], draws["k"][:, idx], draws["beta"][:, idx]
    adstock_ss = spend_range[None, :] / (1 - alpha[:, None])
    curves = beta[:, None] * hill(adstock_ss, s[:, None], k[:, None])
    mean = curves.mean(axis=0)
    lo = np.percentile(curves, 5, axis=0)
    hi = np.percentile(curves, 95, axis=0)

    return [
        {"spend": float(sp), "mean": float(m), "hdi_lower": float(l), "hdi_upper": float(h)}
        for sp, m, l, h in zip(spend_range, mean, lo, hi)
    ]


@app.get("/contribution")
def contribution(start: str | None = None, end: str | None = None):
    history = load_history()
    dates = history["date"]

    mask = [True] * len(dates)
    if start:
        mask = [m and d >= start for m, d in zip(mask, dates)]
    if end:
        mask = [m and d <= end for m, d in zip(mask, dates)]
    idx = [i for i, m in enumerate(mask) if m]
    if not idx:
        raise HTTPException(status_code=400, detail="no data in the requested date range")

    draws = load_posterior()
    result = {"date": [dates[i] for i in idx], "revenue": [history["revenue"][i] for i in idx]}
    for i, c in enumerate(draws["channels"]):
        alpha_mean = float(draws["alpha"][:, i].mean())
        s_mean = float(draws["s"][:, i].mean())
        k_mean = float(draws["k"][:, i].mean())
        beta_mean = float(draws["beta"][:, i].mean())
        spend = np.array([history[f"spend_{c}"][j] for j in idx])
        adstocked = geometric_adstock(spend, alpha_mean, max_lag=8)
        result[f"contribution_{c}"] = (beta_mean * hill(adstocked, s_mean, k_mean)).tolist()
    return result


class PowerRequest(BaseModel):
    tracking_error_std: float = Field(..., gt=0)
    baseline_weekly_revenue: float = Field(..., gt=0)
    n_weeks: float | None = None
    target_mde_pct: float | None = None
    alpha: float = 0.05
    power: float = 0.8


@app.post("/experiment/power")
def experiment_power(req: PowerRequest):
    if req.n_weeks is None and req.target_mde_pct is None:
        raise HTTPException(status_code=400, detail="provide n_weeks and/or target_mde_pct")

    result = {}
    if req.n_weeks is not None:
        mde_abs = minimum_detectable_effect(req.tracking_error_std, req.n_weeks, req.alpha, req.power)
        result["mde_abs"] = mde_abs
        result["mde_pct"] = mde_abs / req.baseline_weekly_revenue
    if req.target_mde_pct is not None:
        target_abs = req.target_mde_pct * req.baseline_weekly_revenue
        result["required_duration_weeks"] = required_duration_weeks(target_abs, req.tracking_error_std, req.alpha, req.power)
    return result
