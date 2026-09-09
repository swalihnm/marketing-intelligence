"""Streamlit dashboard — the 6 views from the spec's Section 8.

Run locally: streamlit run app/dashboard.py
Not deployed to a public URL — see app/main.py's docstring and
docs/06-assumptions-and-limitations.md for why.

Calls the allocator/response-curve/contribution/power functions directly
(not over HTTP) for the interactive views — simpler and avoids a hard
dependency on the FastAPI server also being up; app/main.py remains the
separate, testable API surface for programmatic access.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from allocate.charts import _response_curve_single_channel
from allocate.optimizer import IDATA_PATH, load_posterior_draws, marginal_roi, optimize_allocation
from experiment.power import minimum_detectable_effect, required_duration_weeks
from simulate.dgp import sample_true_params, simulate as dgp_simulate
from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

st.set_page_config(page_title="Marketing Mix Modelling", layout="wide")


@st.cache_resource
def load_state():
    true = sample_true_params(n_weeks=156, spend_correlation=0.5, seed=42)
    df = dgp_simulate(true, seed=42 + 1000)
    channels = list(true.channels.keys())
    current_spend = {c: float(df[f"spend_{c}"].mean()) for c in channels}
    draws = load_posterior_draws(IDATA_PATH, channels, n_draws=300)
    return dict(true=true, df=df, channels=channels, current_spend=current_spend, draws=draws)


state = load_state()
channels = state["channels"]
current_spend = state["current_spend"]
draws = state["draws"]
df = state["df"]

st.title("Marketing Mix Modelling & Budget Allocation")
st.caption(
    "Media spine is Tier-1 simulated data with known ground truth (see docs/01-data-sources-and-honesty.md). "
    "Not a live business dashboard — a working demo of the full pipeline end to end."
)

tabs = st.tabs([
    "1. Budget recommendation", "2. Response curves", "3. Contribution decomposition",
    "4. Customer economics", "5. Model diagnostics", "6. Experiment designer",
])

# --- 1. Budget recommendation ---
with tabs[0]:
    st.subheader("Scenario controls")
    col1, col2 = st.columns(2)
    with col1:
        total_budget = st.slider(
            "Total weekly budget (£)", min_value=int(sum(current_spend.values()) * 0.5),
            max_value=int(sum(current_spend.values()) * 2.0), value=int(sum(current_spend.values())), step=1000,
        )
    with col2:
        max_shift = st.slider("Max change per channel vs. current (fraction)", 0.1, 1.0, 0.5, 0.05)

    if st.button("Recommend allocation"):
        with st.spinner("Optimizing over the posterior..."):
            result = optimize_allocation(draws, current_spend, float(total_budget), max_shift_frac=max_shift)
        alloc_df = pd.DataFrame({
            "channel": channels,
            "current": [current_spend[c] for c in channels],
            "recommended": [result["optimal_spend"][c] for c in channels],
        }).set_index("channel")
        st.bar_chart(alloc_df)
        lo, hi = result["expected_gain_hdi90"]
        st.metric(
            "Expected weekly revenue gain",
            f"£{result['expected_gain_mean']:,.0f}",
            f"90% credible interval: £{lo:,.0f} to £{hi:,.0f}",
        )

# --- 2. Response curves ---
with tabs[1]:
    st.subheader("Channel response curves (steady-state adstock approximation)")
    st.image(str(FIG_DIR / "fig_response_curves.png"))
    st.subheader("Marginal ROI at current spend")
    st.image(str(FIG_DIR / "fig_marginal_roi.png"))
    st.dataframe(marginal_roi(draws, current_spend))
    st.subheader("Budget sweep")
    st.image(str(FIG_DIR / "fig_budget_sweep.png"))

# --- 3. Contribution decomposition ---
with tabs[2]:
    st.subheader("Revenue decomposition (posterior-mean parameters)")
    date_range = st.slider(
        "Date range", min_value=df["date"].min().to_pydatetime(), max_value=df["date"].max().to_pydatetime(),
        value=(df["date"].min().to_pydatetime(), df["date"].max().to_pydatetime()),
    )
    sub = df[(df["date"] >= date_range[0]) & (df["date"] <= date_range[1])]
    decomp = pd.DataFrame({"date": sub["date"]})
    for i, c in enumerate(channels):
        alpha_mean = float(draws["alpha"][:, i].mean())
        s_mean = float(draws["s"][:, i].mean())
        k_mean = float(draws["k"][:, i].mean())
        beta_mean = float(draws["beta"][:, i].mean())
        adstocked = geometric_adstock_numpy(sub[f"spend_{c}"].values, alpha_mean, max_lag=8)
        decomp[c] = beta_mean * hill(adstocked, s_mean, k_mean)
    decomp["base (trend+season+price+intercept)"] = sub["revenue"].values - decomp[channels].sum(axis=1).values
    st.area_chart(decomp.set_index("date"))

# --- 4. Customer economics ---
with tabs[3]:
    st.subheader("Cohort retention")
    st.image(str(FIG_DIR / "fig_cohort_retention_heatmap.png"))
    st.subheader("RFM segments")
    st.image(str(FIG_DIR / "fig_rfm_segments.png"))
    st.subheader("LTV : CAC by (synthetic) channel")
    st.caption("Synthetic channel assignment — illustrative, not real acquisition data. See docs/01.")
    st.image(str(FIG_DIR / "fig_ltv_cac_by_channel.png"))
    st.subheader("CLV validation: BG/NBD + Gamma-Gamma vs. naive")
    st.image(str(FIG_DIR / "fig_clv_validation.png"))

# --- 5. Model diagnostics ---
with tabs[4]:
    st.subheader("Parameter recovery")
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(FIG_DIR / "fig_recovery_beta.png"))
        st.image(str(FIG_DIR / "fig_recovery_hill_s.png"))
    with col2:
        st.image(str(FIG_DIR / "fig_recovery_alpha.png"))
        st.image(str(FIG_DIR / "fig_recovery_hill_k.png"))
    st.subheader("Posterior predictive check")
    st.image(str(FIG_DIR / "fig_posterior_predictive.png"))
    st.subheader("Diagnostics summary")
    st.dataframe(pd.read_csv(REPORT_DIR / "diagnostics_summary.csv"))
    st.subheader("Causal identification: DiD vs. naive")
    st.image(str(FIG_DIR / "fig_did_channel_launch.png"))

# --- 6. Experiment designer ---
with tabs[5]:
    st.subheader("Geo-lift design (built and validated — see docs/08, docs/05)")
    st.image(str(FIG_DIR / "fig_geo_matched_pair.png"))
    st.image(str(FIG_DIR / "fig_geo_mde_curve.png"))

    st.subheader("MDE calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        tracking_error = st.number_input("Pre-period tracking error (std)", value=800.0, min_value=1.0)
    with c2:
        baseline_rev = st.number_input("Baseline weekly revenue", value=15000.0, min_value=1.0)
    with c3:
        n_weeks_input = st.number_input("Planned test duration (weeks)", value=8.0, min_value=1.0)

    mde_abs = minimum_detectable_effect(tracking_error, n_weeks_input)
    st.metric("Minimum detectable effect", f"{mde_abs / baseline_rev:.1%}", f"£{mde_abs:,.0f} absolute")

    target_pct = st.slider("Target effect size to design for (%)", 1, 50, 10) / 100
    required_weeks = required_duration_weeks(target_pct * baseline_rev, tracking_error)
    st.metric(f"Weeks needed to detect a {target_pct:.0%} effect", f"{required_weeks:.1f} weeks")
