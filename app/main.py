"""FastAPI serving layer for the MMM allocator, response curves, revenue
decomposition, and experiment power calculator — the API surface described
in the spec's Section 8.

Deployment note (stated here rather than glossed over): this app is built
and tested locally (`tests/test_api.py`, using FastAPI's TestClient — no
network binding required) and can be run locally with `uvicorn
app.main:app --reload`. It is **not deployed to a public/live URL** — that
needs cloud hosting credentials (e.g. a Vercel/Fly/Render account) that
aren't available in this build environment. See
docs/06-assumptions-and-limitations.md.

Run locally: uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs (FastAPI's auto-generated OpenAPI UI)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from allocate.charts import _response_curve_single_channel
from allocate.optimizer import IDATA_PATH, load_posterior_draws, optimize_allocation
from experiment.power import minimum_detectable_effect, required_duration_weeks
from simulate.dgp import sample_true_params, simulate as dgp_simulate
from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Marketing Mix Modelling API",
    description="Budget allocation, response curves, revenue decomposition, and experiment power — backed by the Bayesian MMM posterior.",
)

_state: dict = {}


def _load_state() -> dict:
    if _state:
        return _state
    true = sample_true_params(n_weeks=156, spend_correlation=0.5, seed=42)
    df = dgp_simulate(true, seed=42 + 1000)
    channels = list(true.channels.keys())
    current_spend = {c: float(df[f"spend_{c}"].mean()) for c in channels}
    draws = load_posterior_draws(IDATA_PATH, channels, n_draws=300)
    _state.update(dict(true=true, df=df, channels=channels, current_spend=current_spend, draws=draws))
    return _state


class AllocateRequest(BaseModel):
    budget: float = Field(..., gt=0, description="Total weekly budget to allocate")
    channel_caps: dict[str, tuple[float, float]] | None = Field(
        None, description="Optional per-channel (min, max) absolute spend caps"
    )
    max_shift: float = Field(0.5, ge=0, description="Max fractional change vs. current spend per channel")


class AllocateResponse(BaseModel):
    optimal_spend: dict[str, float]
    current_spend: dict[str, float]
    total_budget: float
    expected_gain_mean: float
    expected_gain_hdi90_lower: float
    expected_gain_hdi90_upper: float


@app.post("/allocate", response_model=AllocateResponse)
def allocate(req: AllocateRequest):
    state = _load_state()
    result = optimize_allocation(
        state["draws"], state["current_spend"], req.budget,
        channel_caps=req.channel_caps, max_shift_frac=req.max_shift,
    )
    return AllocateResponse(
        optimal_spend=result["optimal_spend"],
        current_spend=result["current_spend"],
        total_budget=result["total_budget"],
        expected_gain_mean=result["expected_gain_mean"],
        expected_gain_hdi90_lower=result["expected_gain_hdi90"][0],
        expected_gain_hdi90_upper=result["expected_gain_hdi90"][1],
    )


class ResponseCurvePoint(BaseModel):
    spend: float
    mean: float
    hdi_lower: float
    hdi_upper: float


@app.get("/response-curve/{channel}", response_model=list[ResponseCurvePoint])
def response_curve(channel: str, n_points: int = 50, max_multiplier: float = 3.0):
    state = _load_state()
    if channel not in state["channels"]:
        raise HTTPException(status_code=404, detail=f"unknown channel '{channel}'. Known: {state['channels']}")
    idx = state["channels"].index(channel)
    max_x = state["current_spend"][channel] * max_multiplier
    spend_range = np.linspace(1.0, max_x, n_points)
    curves = _response_curve_single_channel(spend_range, idx, state["draws"])
    mean = curves.mean(axis=0)
    lo = np.percentile(curves, 5, axis=0)
    hi = np.percentile(curves, 95, axis=0)
    return [
        ResponseCurvePoint(spend=float(s), mean=float(m), hdi_lower=float(l), hdi_upper=float(h))
        for s, m, l, h in zip(spend_range, mean, lo, hi)
    ]


class ContributionPoint(BaseModel):
    date: str
    revenue: float
    base: float  # intercept + trend + seasonality + price, at posterior mean


@app.get("/contribution")
def contribution(start: str | None = None, end: str | None = None):
    """Revenue decomposition (posterior-mean parameters) over a date range."""
    state = _load_state()
    df = state["df"]
    sub = df.copy()
    if start:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end:
        sub = sub[sub["date"] <= pd.Timestamp(end)]
    if sub.empty:
        raise HTTPException(status_code=400, detail="no data in the requested date range")

    draws = state["draws"]
    channels = state["channels"]
    result = {"date": sub["date"].dt.strftime("%Y-%m-%d").tolist(), "revenue": sub["revenue"].tolist()}
    for i, c in enumerate(channels):
        alpha_mean = float(draws["alpha"][:, i].mean())
        s_mean = float(draws["s"][:, i].mean())
        k_mean = float(draws["k"][:, i].mean())
        beta_mean = float(draws["beta"][:, i].mean())
        adstocked = geometric_adstock_numpy(sub[f"spend_{c}"].values, alpha_mean, max_lag=8)
        result[f"contribution_{c}"] = (beta_mean * hill(adstocked, s_mean, k_mean)).tolist()
    return result


class PowerRequest(BaseModel):
    tracking_error_std: float = Field(..., gt=0, description="Pre-period tracking error (noise std) of the matched-pair gap")
    baseline_weekly_revenue: float = Field(..., gt=0)
    n_weeks: float | None = Field(None, description="If given, compute the MDE at this test duration")
    target_mde_pct: float | None = Field(None, description="If given, compute the duration needed to detect this effect size (fraction of baseline)")
    alpha: float = 0.05
    power: float = 0.8


class PowerResponse(BaseModel):
    mde_abs: float | None = None
    mde_pct: float | None = None
    required_duration_weeks: float | None = None


@app.post("/experiment/power", response_model=PowerResponse)
def experiment_power(req: PowerRequest):
    if req.n_weeks is None and req.target_mde_pct is None:
        raise HTTPException(status_code=400, detail="provide n_weeks and/or target_mde_pct")

    resp = PowerResponse()
    if req.n_weeks is not None:
        mde_abs = minimum_detectable_effect(req.tracking_error_std, req.n_weeks, req.alpha, req.power)
        resp.mde_abs = mde_abs
        resp.mde_pct = mde_abs / req.baseline_weekly_revenue
    if req.target_mde_pct is not None:
        target_abs = req.target_mde_pct * req.baseline_weekly_revenue
        resp.required_duration_weeks = required_duration_weeks(target_abs, req.tracking_error_std, req.alpha, req.power)
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}
