"""
Core backtest logic for the order-book imbalance strategy.

Kept as PURE FUNCTIONS that take a DataFrame and return a DataFrame, so they can
be tested against a local Parquet file OR data pulled from Supabase -- the logic
doesn't care where the rows came from. That separation is the whole point of the
"decoupled storage vs compute" architecture.

Strategy in plain words:
  - imbalance > +threshold  -> buy pressure  -> go long  (position = +1)
  - imbalance < -threshold  -> sell pressure -> go short (position = -1)
  - otherwise               -> flat          (position =  0)
We earn the next-step return times our position. Then we subtract trading costs
every time the position CHANGES (that's a trade). The before-vs-after-costs gap
is the headline result.
"""

import numpy as np
import pandas as pd


def compute_signals(df, threshold=0.3):
    """Add mid, imbalance, and target position columns."""
    df = df.copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["imbalance"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"])

    pos = np.zeros(len(df))
    pos[df["imbalance"] > threshold] = 1.0
    pos[df["imbalance"] < -threshold] = -1.0
    df["position"] = pos
    return df


def run_backtest(df, cost_per_trade_bps=1.0):
    """
    Walk the data, apply positions, accrue PnL, subtract costs on each trade.

    cost_per_trade_bps: round-trip-ish cost charged whenever position changes,
    in basis points of mid price (1 bp = 0.01%). Covers fee + half-spread +
    slippage lumped into one honest number.

    Returns df with: ret, gross_pnl, trade, cost, net_pnl, equity_gross, equity_net
    """
    df = df.copy()

    # Return of the asset from this step to the next.
    df["ret"] = df["mid"].shift(-1) / df["mid"] - 1

    # We hold this step's position into the next step's return.
    df["gross_pnl"] = df["position"] * df["ret"]

    # A "trade" happens when the position changes from the previous row.
    df["trade"] = df["position"].diff().abs().fillna(0) > 0

    # Cost: bps of price, charged on each trade.
    df["cost"] = df["trade"] * (cost_per_trade_bps / 1e4)

    df["net_pnl"] = df["gross_pnl"] - df["cost"]

    # Equity curves: cumulative sum of per-step returns (simple, additive model).
    df["equity_gross"] = df["gross_pnl"].fillna(0).cumsum()
    df["equity_net"] = df["net_pnl"].fillna(0).cumsum()
    return df


def metrics(df, periods_per_day=None):
    """Compute the headline performance numbers from a backtested df."""
    net = df["net_pnl"].dropna()
    gross = df["gross_pnl"].dropna()

    def sharpe(series):
        s = series.std()
        return float(series.mean() / s * np.sqrt(len(series))) if s > 0 else 0.0

    def max_drawdown(equity):
        peak = equity.cummax()
        dd = equity - peak
        return float(dd.min())

    n_trades = int(df["trade"].sum())
    return {
        "n_rows": int(len(df)),
        "n_trades": n_trades,
        "gross_sharpe": round(sharpe(gross), 3),
        "net_sharpe": round(sharpe(net), 3),
        "gross_return": round(float(gross.sum()), 5),
        "net_return": round(float(net.sum()), 5),
        "max_drawdown_net": round(max_drawdown(df["equity_net"].fillna(0)), 5),
        "hit_rate": round(float((net > 0).mean()), 3),
    }


if __name__ == "__main__":
    # Standalone test against the local Parquet (no Supabase needed).
    df = pd.read_parquet("orderbook.parquet")
    df = compute_signals(df, threshold=0.3)
    df = run_backtest(df, cost_per_trade_bps=1.0)
    m = metrics(df)
    print("Backtest on local parquet:")
    for k, v in m.items():
        print(f"  {k:20} {v}")