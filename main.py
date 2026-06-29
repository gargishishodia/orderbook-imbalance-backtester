"""
main.py -- the orchestrator (Compute + Reporting layers).

Pipeline:
  1. fetch order-book rows from Supabase  (Persistence -> Compute)
  2. compute signals + run backtest        (Compute)
  3. save the run's metrics to Supabase    (queryable experiment log)
  4. plot the equity curves                (Reporting)

Run:  python main.py
      python main.py --local   (use the parquet file instead of Supabase)
"""

import sys
import pandas as pd
from src import backtester

THRESHOLD = 0.3
COST_BPS = 1.0


def get_data(use_local):
    if use_local:
        print("Loading from local parquet (--local mode)...")
        return pd.read_parquet("data/orderbook.parquet")
    from src import database
    print("Fetching order-book data from Supabase...")
    df = database.fetch_orderbook()
    print(f"  fetched {len(df):,} rows")
    return df


def main():
    use_local = "--local" in sys.argv
    df = get_data(use_local)
    if df.empty:
        raise SystemExit("No data. Ingest first (python -m src.data_processor) or use --local.")

    print("Computing signals and running backtest...")
    df = backtester.compute_signals(df, threshold=THRESHOLD)
    df = backtester.run_backtest(df, cost_per_trade_bps=COST_BPS)
    m = backtester.metrics(df)

    print("\n--- Results ---")
    for k, v in m.items():
        print(f"  {k:20} {v}")

    if not use_local:
        from src import database
        database.save_results(m, THRESHOLD, COST_BPS)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["equity_gross"].values, label="gross (before costs)")
        ax.plot(df["equity_net"].values, label="net (after costs)")
        ax.set_title("Order-book imbalance strategy: equity curve")
        ax.set_xlabel("snapshot"); ax.set_ylabel("cumulative return")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("equity_curve.png", dpi=120)
        print("\nSaved chart -> equity_curve.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart)")


if __name__ == "__main__":
    main()