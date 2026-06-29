"""
src/data_processor.py -- data production + batch ingestion.

Two jobs, in order:
  1. generate_orderbook()  -> creates the synthetic order-book data
  2. ingest_parquet()      -> uploads it to Supabase

Run directly to do both (generate if missing, then ingest a test batch):
    python -m src.data_processor

Generation needs no database, so you can always get a parquet file even if
Supabase isn't wired up yet.
"""

import os
import numpy as np
import pandas as pd
from src.database import insert_rows

PARQUET_PATH = "data/orderbook.parquet"


# ----------------------------------------------------------------------
# 1. GENERATE  (no database needed)
# ----------------------------------------------------------------------
def generate_orderbook(n_snapshots=200_000, seed=42, tick=0.05, start_price=500.0):
    """Synthetic NSE-style order-book snapshots with a built-in imbalance signal."""
    rng = np.random.default_rng(seed)

    steps = rng.choice([-1, 0, 1], size=n_snapshots, p=[0.25, 0.50, 0.25])
    mid = start_price + np.cumsum(steps) * tick
    bid_price = np.round(mid - tick / 2, 2)
    ask_price = np.round(mid + tick / 2, 2)

    base_bid = rng.integers(50, 500, size=n_snapshots).astype(float)
    base_ask = rng.integers(50, 500, size=n_snapshots).astype(float)
    lean = np.r_[steps[1:], 0]
    bid_qty = base_bid + np.where(lean > 0, rng.integers(100, 400, n_snapshots), 0)
    ask_qty = base_ask + np.where(lean < 0, rng.integers(100, 400, n_snapshots), 0)

    t0 = pd.Timestamp("2024-01-02 09:15:00")
    gaps = rng.integers(50, 150, size=n_snapshots)
    ts = t0 + pd.to_timedelta(np.cumsum(gaps), unit="ms")

    return pd.DataFrame({
        "timestamp": ts,
        "bid_price": bid_price, "bid_qty": bid_qty,
        "ask_price": ask_price, "ask_qty": ask_qty,
    })

    # >>> REAL DATA HOOK <<<
    # Later, replace this function with a loader for real NSE data that returns
    # a DataFrame with the same five columns. Nothing else has to change.


def generate_and_save(path=PARQUET_PATH):
    """Generate the data and write it to parquet. Returns the path."""
    df = generate_orderbook()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Generated {len(df):,} rows -> {path}")
    return path


# ----------------------------------------------------------------------
# 2. INGEST  (needs Supabase set up)
# ----------------------------------------------------------------------
def ingest_parquet(path=PARQUET_PATH, limit=None):
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)

    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    records = df[["timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"]] \
                .to_dict(orient="records")

    print(f"Ingesting {len(records):,} rows into Supabase orderbook table...")
    insert_rows("orderbook", records)
    print("Ingestion complete.")


if __name__ == "__main__":
    # Step 1: make sure the parquet exists (generate if not).
    if not os.path.exists(PARQUET_PATH):
        print("No parquet found -- generating data first...")
        generate_and_save()
    else:
        print(f"Found existing {PARQUET_PATH}")

    # Step 2: ingest a small test batch. Remove limit=2000 for the full load.
    ingest_parquet(limit=2000)
