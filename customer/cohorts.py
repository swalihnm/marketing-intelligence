"""Cohort, retention, RFM, and CAC/LTV analytics on the UCI Online Retail II
warehouse (data/processed/warehouse.duckdb, built by
ingest/uci_retail_loader.py).

Runs the SQL in sql/analytics/, adds the synthetic channel assignment
(customer/channel_synth.py), and produces the "required charts" from the
spec that this substitute dataset supports: cohort retention heatmap,
RFM segment distribution, LTV-to-CAC by channel with payback period.

Not produced here (dataset doesn't support them — see
docs/01-data-sources-and-honesty.md):
  - category_contribution — no product category taxonomy in this dataset
  - delivery_impact_on_repeat — no delivery/shipping timestamps
  - spend vs revenue by channel, channel spend correlation matrix — those
    are MMM charts (outputs/figures/fig_recovery_*), not customer-layer ones

Run: python -m customer.cohorts
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from customer.channel_synth import assign_synthetic_channels

REPO_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_PATH = REPO_ROOT / "data" / "processed" / "warehouse.duckdb"
SQL_DIR = REPO_ROOT / "sql" / "analytics"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
REPORT_DIR = REPO_ROOT / "outputs" / "reports"


def _run_sql_file(con, filename: str) -> pd.DataFrame:
    sql = (SQL_DIR / filename).read_text(encoding="utf-8")
    return con.execute(sql).fetchdf()


def _ensure_synthetic_channels(con) -> None:
    customer_ids = con.execute(
        "SELECT DISTINCT customer_id FROM transactions"
    ).fetchdf()["customer_id"].tolist()
    channel_df = assign_synthetic_channels(customer_ids)
    con.execute("CREATE OR REPLACE TABLE customer_channel_synthetic AS SELECT * FROM channel_df")


def _plot_retention_heatmap(retention: pd.DataFrame, out_path: Path, max_months: int = 12) -> None:
    retention = retention[retention["months_since_acquisition"].between(0, max_months)]
    pivot = retention.pivot(index="cohort_month", columns="months_since_acquisition", values="retention_rate")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([d.strftime("%Y-%m") for d in pivot.index])
    ax.set_xlabel("months since acquisition")
    ax.set_ylabel("acquisition cohort")
    ax.set_title("Cohort retention heatmap (share of cohort active in that month)")
    fig.colorbar(im, ax=ax, label="retention rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_rfm_segments(rfm: pd.DataFrame, out_path: Path) -> None:
    counts = rfm["segment"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(counts.index, counts.values, color="#457b9d")
    ax.set_xlabel("customers")
    ax.set_title("RFM segment distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_ltv_cac(cac_ltv: pd.DataFrame, payback: pd.DataFrame, out_path: Path) -> None:
    merged = cac_ltv.merge(payback, on="channel")
    merged = merged.sort_values("ltv_to_cac_ratio", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.barh(merged["channel"], merged["ltv_to_cac_ratio"], color="#2a9d8f")
    ax1.axvline(1.0, color="red", linestyle="--", linewidth=1, label="breakeven (LTV=CAC)")
    ax1.set_xlabel("LTV : CAC ratio")
    ax1.set_title("LTV : CAC by (synthetic) channel")
    ax1.legend()

    ax2.barh(merged["channel"], merged["payback_months"], color="#e76f51")
    ax2.set_xlabel("months to payback")
    ax2.set_title("Payback period by (synthetic) channel")

    fig.suptitle("Synthetic channel assignment — illustrative, not real acquisition data (see docs)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_customer_analytics():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(WAREHOUSE_PATH))
    _ensure_synthetic_channels(con)

    cohorts = _run_sql_file(con, "acquisition_cohorts.sql")
    retention = _run_sql_file(con, "retention_curves.sql")
    rfm = _run_sql_file(con, "rfm_segmentation.sql")
    cac_ltv = _run_sql_file(con, "cac_ltv_by_channel.sql")

    # payback period: months for a customer's average monthly revenue to
    # cover the channel's CAC. Computed here (not a standalone .sql file)
    # since it's a derived metric specific to this chart.
    customer_rate = con.execute(
        """
        SELECT
            customer_id,
            SUM(revenue) AS total_revenue,
            GREATEST(DATE_DIFF('month', MIN(invoice_date), MAX(invoice_date)), 0) + 1 AS tenure_months
        FROM transactions
        GROUP BY customer_id
        """
    ).fetchdf()
    customer_rate["avg_monthly_revenue"] = customer_rate["total_revenue"] / customer_rate["tenure_months"]
    channel_map = con.execute("SELECT customer_id, channel, cac FROM customer_channel_synthetic").fetchdf()
    rate_by_channel = customer_rate.merge(channel_map, on="customer_id")
    payback = rate_by_channel.groupby(["channel", "cac"], as_index=False)["avg_monthly_revenue"].mean()
    payback["payback_months"] = payback["cac"] / payback["avg_monthly_revenue"]
    payback = payback[["channel", "payback_months"]]

    con.close()

    cohorts.to_csv(REPORT_DIR / "acquisition_cohorts.csv", index=False)
    retention.to_csv(REPORT_DIR / "retention_curves.csv", index=False)
    rfm.to_csv(REPORT_DIR / "rfm_segmentation.csv", index=False)
    cac_ltv.to_csv(REPORT_DIR / "cac_ltv_by_channel.csv", index=False)
    payback.to_csv(REPORT_DIR / "payback_by_channel.csv", index=False)

    _plot_retention_heatmap(retention, FIG_DIR / "fig_cohort_retention_heatmap.png")
    _plot_rfm_segments(rfm, FIG_DIR / "fig_rfm_segments.png")
    _plot_ltv_cac(cac_ltv, payback, FIG_DIR / "fig_ltv_cac_by_channel.png")

    return dict(cohorts=cohorts, retention=retention, rfm=rfm, cac_ltv=cac_ltv, payback=payback)


if __name__ == "__main__":
    result = run_customer_analytics()
    print("cohorts:\n", result["cohorts"])
    print("\ncac_ltv:\n", result["cac_ltv"])
    print("\npayback:\n", result["payback"])
    print("\nrfm segment counts:\n", result["rfm"]["segment"].value_counts())
