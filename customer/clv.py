"""Customer lifetime value: BG/NBD (purchase frequency + probability alive)
+ Gamma-Gamma (monetary value), fit on real transaction data (UCI Online
Retail II, via data/processed/warehouse.duckdb) and validated on a holdout
period — not just fit-and-trust.

Validation design: split the observation window into a calibration period
and a holdout period. Fit both models on calibration-period behavior only,
predict each customer's expected revenue over the holdout period length,
then compare that prediction against what customers actually spent in the
holdout period. Compared against a naive baseline (flat extrapolation of
each customer's calibration-period average monthly spend) to show whether
the BG/NBD + Gamma-Gamma approach actually earns its complexity.

Run: python -m customer.clv
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

REPO_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_PATH = REPO_ROOT / "data" / "processed" / "warehouse.duckdb"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"

FREQ = "W"  # weekly grain for recency/frequency/T, consistent with the MMM layer


def _load_transactions() -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE_PATH), read_only=True)
    df = con.execute(
        "SELECT customer_id, invoice_date, revenue FROM transactions"
    ).fetchdf()
    con.close()
    return df


def run_clv_study(holdout_months: int = 6):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    txns = _load_transactions()
    obs_end = txns["invoice_date"].max()
    cal_end = obs_end - pd.DateOffset(months=holdout_months)

    cal_holdout = calibration_and_holdout_data(
        txns,
        customer_id_col="customer_id",
        datetime_col="invoice_date",
        calibration_period_end=cal_end,
        observation_period_end=obs_end,
        freq=FREQ,
        monetary_value_col="revenue",
    )

    # --- BG/NBD: purchase frequency + probability-alive, calibration data only ---
    # A handful of customers in this dataset are wholesale/bulk-buying B2B
    # accounts (frequency in the dozens-to-75 range) rather than typical
    # retail customers, and destabilize both models' MLE fit (and cause
    # numerically unstable predictions for some other customers). Standard
    # practice: fit on the bulk of the population (below the 99th
    # percentile of calibration frequency), still predict/evaluate for
    # everyone. Count excluded is reported in metrics, not hidden.
    freq_cap = cal_holdout["frequency_cal"].quantile(0.99)
    fit_mask = cal_holdout["frequency_cal"] <= freq_cap
    n_excluded_from_fit = int((~fit_mask).sum())

    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        cal_holdout.loc[fit_mask, "frequency_cal"],
        cal_holdout.loc[fit_mask, "recency_cal"],
        cal_holdout.loc[fit_mask, "T_cal"],
    )

    # --- Gamma-Gamma: monetary value, repeat customers only (frequency_cal > 0) ---
    repeat = cal_holdout[cal_holdout["frequency_cal"] > 0]
    corr = repeat[["frequency_cal", "monetary_value_cal"]].corr().iloc[0, 1]

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(repeat.loc[fit_mask, "frequency_cal"], repeat.loc[fit_mask, "monetary_value_cal"])
    # NOTE: the fitted q parameter for this dataset comes out below 1 (see
    # metrics/report) — the Gamma-Gamma model's implied mean is only
    # well-defined for q>1, and its independence assumption (frequency vs.
    # monetary value) is mildly violated here (correlation ~0.17, reported
    # below). Both are consistent with this dataset's monetary_value
    # distribution being heavy-tailed and largely single-observation for
    # low-frequency customers. Documented as a genuine model-fit limitation
    # rather than silently patched: predictions are clipped at 0 below
    # (lifetime value cannot be negative), which is a pragmatic floor, not
    # a fix for the underlying assumption violation.

    # --- predicted CLV over the holdout window, for every customer ---
    monetary_for_pred = cal_holdout["monetary_value_cal"].replace(0, np.nan)
    monetary_for_pred = monetary_for_pred.fillna(repeat["monetary_value_cal"].mean())
    predicted_clv = ggf.customer_lifetime_value(
        bgf,
        cal_holdout["frequency_cal"],
        cal_holdout["recency_cal"],
        cal_holdout["T_cal"],
        monetary_for_pred,
        time=holdout_months,
        freq=FREQ,
        discount_rate=0.0,
    )
    n_nan_predictions = int(predicted_clv.isna().sum())
    # remaining NaNs are customers with ~0 calibration history (acquired in
    # the last few days of the calibration window) — a known numerical edge
    # case in the underlying hypergeometric prediction for T close to 0.
    # No history means no basis to predict repeat behavior from; imputed to
    # 0 rather than dropped, so every customer still gets a row.
    predicted_clv = predicted_clv.fillna(0.0).clip(lower=0.0)

    # --- naive baseline: flat extrapolation of calibration-period average monthly spend ---
    cal_span_months = (cal_end - txns["invoice_date"].min()).days / 30.44
    cal_span_months = max(cal_span_months, 1.0)
    naive_clv = (cal_holdout["monetary_value_cal"] * cal_holdout["frequency_cal"]) / cal_span_months * holdout_months
    # customers with a single calibration purchase (frequency_cal == 0, monetary_value_cal == 0)
    # get 0 under this naive rule too — same blind spot as the naive approach in practice.

    results = pd.DataFrame({
        "customer_id": cal_holdout.index,
        "frequency_cal": cal_holdout["frequency_cal"].values,
        "predicted_clv_bgnbd_gg": predicted_clv.values,
        "naive_clv": naive_clv.values,
        "actual_holdout_revenue": cal_holdout["monetary_value_holdout"].values * cal_holdout["frequency_holdout"].values,
    })

    def mae(pred, actual):
        return float(np.mean(np.abs(pred - actual)))

    def corr_with_actual(pred, actual):
        if pred.std() == 0 or actual.std() == 0:
            return float("nan")
        return float(np.corrcoef(pred, actual)[0, 1])

    metrics = dict(
        n_customers=len(results),
        n_repeat_customers_calibration=int((cal_holdout["frequency_cal"] > 0).sum()),
        n_excluded_from_model_fit_bulk_outliers=n_excluded_from_fit,
        n_predictions_imputed_zero_history_edge_case=n_nan_predictions,
        gamma_gamma_q_parameter=float(ggf.summary.loc["q", "coef"]),
        freq_monetary_correlation_calibration=float(corr),
        mae_bgnbd_gg=mae(results["predicted_clv_bgnbd_gg"], results["actual_holdout_revenue"]),
        mae_naive=mae(results["naive_clv"], results["actual_holdout_revenue"]),
        corr_bgnbd_gg=corr_with_actual(results["predicted_clv_bgnbd_gg"].values, results["actual_holdout_revenue"].values),
        corr_naive=corr_with_actual(results["naive_clv"].values, results["actual_holdout_revenue"].values),
        holdout_months=holdout_months,
        calibration_period_end=str(cal_end.date()),
        observation_period_end=str(obs_end.date()),
    )

    results.to_csv(REPORT_DIR / "clv_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / "clv_validation_metrics.csv", index=False)

    # --- scatter: predicted vs actual, both methods ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    cap = np.percentile(results["actual_holdout_revenue"], 99)
    for ax, col, title, mae_val, corr_val in [
        (axes[0], "predicted_clv_bgnbd_gg", "BG/NBD + Gamma-Gamma", metrics["mae_bgnbd_gg"], metrics["corr_bgnbd_gg"]),
        (axes[1], "naive_clv", "Naive (flat extrapolation)", metrics["mae_naive"], metrics["corr_naive"]),
    ]:
        ax.scatter(results["actual_holdout_revenue"], results[col], alpha=0.25, s=12, color="#457b9d")
        ax.plot([0, cap], [0, cap], color="red", linestyle="--", linewidth=1, label="y = x")
        ax.set_xlim(0, cap)
        ax.set_ylim(0, cap)
        ax.set_xlabel("actual holdout revenue")
        ax.set_ylabel("predicted CLV")
        ax.set_title(f"{title}\nMAE={mae_val:.0f}, corr={corr_val:.2f}")
        ax.legend()
    fig.suptitle(f"Predicted vs actual customer revenue over {holdout_months}-month holdout")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_clv_validation.png", dpi=150)
    plt.close(fig)

    return metrics, results


if __name__ == "__main__":
    metrics, results = run_clv_study()
    for k, v in metrics.items():
        print(f"{k}: {v}")
