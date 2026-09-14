"""Difference-in-differences: a channel launches in one region and not
another — the quasi-experimental method the spec calls for in Section 6.3,
kept (per the spec's own cut order: "drop IV analysis, keep DiD") in place
of an IV analysis, which is not implemented here.

Same validate-before-trusting pattern as the rest of this project: this
module simulates two regions under a KNOWN true channel effect (so the DiD
estimate can be checked against ground truth, not just computed and
trusted), and deliberately includes a shared confounder — a national
trend/seasonality that affects both regions identically — so a **naive**
before/after estimate in the launch region alone is demonstrably biased,
and DiD (which nets out the shared confounder using the non-launch region
as control) is demonstrably not. That contrast is the actual point of this
module: showing why the identification method matters, not just running it.

What this does NOT address: reverse causality from budget being set as a
percentage of forecast revenue — that threat has no regional launch/pause
structure to exploit with DiD, and isn't simulated here. See
docs/03-causal-identification.md.

Run: python -m causal.did
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transforms.adstock import geometric_adstock_numpy
from transforms.saturation import hill

REPO_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

TRUE_BETA = 180_000.0
TRUE_ALPHA = 0.3
TRUE_S = 2.0
TRUE_K = 50_000.0
WEEKLY_SPEND_AFTER_LAUNCH = 40_000.0


def simulate_two_region_launch(
    n_weeks: int = 52,
    launch_week: int = 26,
    true_beta: float = TRUE_BETA,
    true_alpha: float = TRUE_ALPHA,
    true_s: float = TRUE_S,
    true_k: float = TRUE_K,
    weekly_spend_after_launch: float = WEEKLY_SPEND_AFTER_LAUNCH,
    seed: int = 99,
) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)

    # Shared confounder: a national trend + seasonal pattern affecting both
    # regions identically — exactly the kind of confounder that biases a
    # naive single-region before/after comparison (Section 6.3's first
    # listed confounder: "seasonality driving both spend and revenue").
    # Note on this specific parameterization: with a symmetric 26/26
    # pre/post split of a 52-week sine cycle, a linear trend's pre/post
    # difference and the seasonal term's pre/post difference partially
    # offset (verified: at trend slope 900/wk they nearly perfectly cancel,
    # ~500 net vs ~23,000 each individually — a coincidence of this exact
    # window split, not a general property of DiD). The trend slope below
    # is set high enough that trend dominates that cancellation, so the
    # naive-vs-DiD contrast below is a real, unambiguous illustration
    # rather than a knife-edge coincidence.
    shared_trend = 3_500.0 * t
    shared_season = 18_000.0 * np.sin(2 * np.pi * t / 52)

    base_a, base_b = 300_000.0, 280_000.0
    noise_sd = 15_000.0
    noise_a = rng.normal(0, noise_sd, n_weeks)
    noise_b = rng.normal(0, noise_sd, n_weeks)

    # realistic week-to-week spend variation post-launch (real campaigns
    # don't spend exactly the same amount every week) — also matters for
    # causal/mmm_vs_did.py: a perfectly flat step function (0, then exactly
    # constant) turned out to leave beta/alpha/s/k severely underidentified
    # for a single-channel MMM fit (1764 divergences, r-hat 3.4 in initial
    # testing) since there's no within-period variation to pin down the
    # adstock/saturation shape from. This fixes that the honest way — by
    # making the simulated spend pattern realistic — rather than by
    # tolerating or hiding the pathological diagnostics.
    spend_noise = rng.normal(1.0, 0.15, n_weeks)
    spend_a = np.where(t >= launch_week, weekly_spend_after_launch * np.clip(spend_noise, 0.5, 1.5), 0.0)
    # region B: channel never launches (the control region)
    adstock_a = geometric_adstock_numpy(spend_a, true_alpha, max_lag=8)
    contribution_a = true_beta * hill(adstock_a, true_s, true_k)

    revenue_a = base_a + shared_trend + shared_season + contribution_a + noise_a
    revenue_b = base_b + shared_trend + shared_season + noise_b

    # price: small independent noise, not connected to revenue — present so
    # region A's panel can be fed to mmm/model.py::build_mmm (which requires
    # a price column) for the MMM-vs-DiD reconciliation in
    # causal/mmm_vs_did.py. Not part of the DiD analysis itself.
    price = 100.0 + rng.normal(0, 2.0, 2 * n_weeks)

    df = pd.DataFrame({
        "week": np.tile(t, 2),
        "t": np.tile(t, 2),
        "region": ["A_launch"] * n_weeks + ["B_control"] * n_weeks,
        "period": np.where(np.tile(t, 2) >= launch_week, "post", "pre"),
        "revenue": np.concatenate([revenue_a, revenue_b]),
        "spend": np.concatenate([spend_a, np.zeros(n_weeks)]),
        "price": price,
    })
    true_avg_contribution_post = float(contribution_a[t >= launch_week].mean())
    return df, true_avg_contribution_post


def estimate_did(df: pd.DataFrame) -> dict:
    means = df.groupby(["region", "period"])["revenue"].mean()

    a_pre, a_post = means["A_launch", "pre"], means["A_launch", "post"]
    b_pre, b_post = means["B_control", "pre"], means["B_control", "post"]

    naive_estimate = float(a_post - a_pre)
    did_estimate = float((a_post - a_pre) - (b_post - b_pre))

    return dict(
        naive_estimate=naive_estimate,
        did_estimate=did_estimate,
        a_pre=float(a_pre), a_post=float(a_post),
        b_pre=float(b_pre), b_post=float(b_post),
    )


def run_did_study():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, true_effect = simulate_two_region_launch()
    est = estimate_did(df)

    result = dict(
        true_effect=true_effect,
        naive_estimate=est["naive_estimate"],
        naive_pct_error=abs(est["naive_estimate"] - true_effect) / true_effect * 100,
        did_estimate=est["did_estimate"],
        did_pct_error=abs(est["did_estimate"] - true_effect) / true_effect * 100,
    )
    pd.DataFrame([result]).to_csv(REPORT_DIR / "did_vs_naive.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="week", columns="region", values="revenue")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(pivot.index, pivot["A_launch"], color="#e63946", label="Region A (channel launches wk 26)")
    ax.plot(pivot.index, pivot["B_control"], color="#457b9d", label="Region B (control, never launches)")
    ax.axvline(25.5, color="grey", linestyle="--", linewidth=1, label="launch week")
    ax.set_xlabel("week")
    ax.set_ylabel("revenue")
    ax.set_title(
        f"DiD channel-launch estimate: true={true_effect:,.0f}, "
        f"naive={est['naive_estimate']:,.0f} ({result['naive_pct_error']:.0f}% off), "
        f"DiD={est['did_estimate']:,.0f} ({result['did_pct_error']:.0f}% off)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_did_channel_launch.png", dpi=150)
    plt.close(fig)

    return result


if __name__ == "__main__":
    result = run_did_study()
    for k, v in result.items():
        print(f"{k}: {v}")
