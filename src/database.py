"""
src/database.py -- data-access layer (read/write helpers only).
Connection comes from supabase_client.py, so there's ONE client, not two.
"""

import pandas as pd
from src.supabase_client import get_client


def insert_rows(table, records, batch_size=1000):
    sb = get_client()
    for i in range(0, len(records), batch_size):
        sb.table(table).insert(records[i:i + batch_size]).execute()
        print(f"  {table}: inserted {min(i + batch_size, len(records)):,}/{len(records):,}")


def save_results(metrics_dict, threshold, cost_bps):
    sb = get_client()
    row = {
        "threshold": threshold, "cost_bps": cost_bps,
        "n_rows": metrics_dict["n_rows"], "n_trades": metrics_dict["n_trades"],
        "gross_sharpe": metrics_dict["gross_sharpe"], "net_sharpe": metrics_dict["net_sharpe"],
        "gross_return": metrics_dict["gross_return"], "net_return": metrics_dict["net_return"],
        "max_drawdown_net": metrics_dict["max_drawdown_net"], "hit_rate": metrics_dict["hit_rate"],
    }
    sb.table("backtest_results").insert(row).execute()
    print("  backtest_results: saved 1 run")


def fetch_orderbook(limit=None, page_size=1000):
    sb = get_client()
    rows, start = [], 0
    while True:
        q = (sb.table("orderbook").select("*").order("timestamp")
               .range(start, start + page_size - 1).execute())
        if not q.data:
            break
        rows.extend(q.data)
        start += page_size
        if limit and len(rows) >= limit:
            rows = rows[:limit]
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
