"""Loader for the UCI Online Retail II dataset (real transactional data).

This is the Tier 4 "real customer-level data" source for the customer
economics layer (cohorts, retention, RFM, CLV) — see
docs/01-data-sources-and-honesty.md for why UCI Online Retail II was used
instead of the spec's first choice, Olist (Kaggle requires an API token
that isn't configured in this environment; UCI's dataset is a directly
downloadable public file, no auth required, and the spec explicitly names
it as an acceptable alternative for this layer).

Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
541,909 transaction line items, UK-based online retailer, Dec 2009 - Dec
2011, real (not simulated) data.

Columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price,
Customer ID, Country. There is no channel/category/delivery field in this
dataset — see customer/channel_synth.py for how the channel dimension is
handled, and docs/01-data-sources-and-honesty.md for what that means for
which spec-listed analyses are and aren't feasible on this substitute data.

Run: python -m ingest.uci_retail_loader
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX = REPO_ROOT / "data" / "raw" / "online_retail_ii" / "online_retail_II.xlsx"
WAREHOUSE_PATH = REPO_ROOT / "data" / "processed" / "warehouse.duckdb"

SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def load_raw() -> pd.DataFrame:
    frames = [pd.read_excel(RAW_XLSX, sheet_name=s) for s in SHEETS]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "Invoice": "invoice",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "price",
        "Customer ID": "customer_id",
        "Country": "country",
    })
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Real-world e-commerce data cleaning, kept minimal and documented:

    - drop line items with no customer id (can't attribute to a customer,
      so useless for cohort/CLV analysis, which is this layer's purpose)
    - drop cancelled invoices (Invoice code starting with 'C') and any
      non-positive quantity/price rows (returns, adjustments, data errors)
    - compute revenue = quantity * price
    """
    df = df.dropna(subset=["customer_id"]).copy()
    df["customer_id"] = df["customer_id"].astype(int)
    df["invoice"] = df["invoice"].astype(str)

    is_cancellation = df["invoice"].str.startswith("C")
    df = df[~is_cancellation]
    df = df[(df["quantity"] > 0) & (df["price"] > 0)]

    df["revenue"] = df["quantity"] * df["price"]
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df.reset_index(drop=True)


def build_warehouse(df: pd.DataFrame) -> None:
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WAREHOUSE_PATH))
    con.execute("CREATE OR REPLACE TABLE transactions AS SELECT * FROM df")
    con.execute(
        """
        CREATE OR REPLACE TABLE customer_first_order AS
        SELECT customer_id, MIN(invoice_date) AS first_order_date
        FROM transactions
        GROUP BY customer_id
        """
    )
    con.close()


if __name__ == "__main__":
    raw = load_raw()
    print(f"raw rows: {len(raw):,}")
    df = clean(raw)
    print(f"cleaned rows: {len(df):,}, customers: {df['customer_id'].nunique():,}, "
          f"date range: {df['invoice_date'].min()} to {df['invoice_date'].max()}")
    build_warehouse(df)
    print(f"warehouse written to {WAREHOUSE_PATH}")
