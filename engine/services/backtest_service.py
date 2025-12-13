# engine/services/backtest_service.py

import pandas as pd
import numpy as np

# -------------------------------------------------
# OPTION A — STRUCTURAL REALISM
# -------------------------------------------------
BACKTEST_START_DATE = pd.Timestamp("2005-01-01")


def run_backtest_service(df: pd.DataFrame, cost_bps: float = 0.5) -> pd.DataFrame:
    """
    Multi-asset backtest for weighted portfolios with transaction costs.

    Expects columns:
      - date
      - ticker
      - price
      - weight

    Returns a DataFrame with:
      - date
      - daily_ret
      - cumret
    """
    print("[BacktestService] Running backtest engine...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # -------------------------------------------------
    # 0) STRUCTURAL REALISM FILTER (OPTION A)
    # -------------------------------------------------
    df = df[df["date"] >= BACKTEST_START_DATE].copy()

    if df.empty:
        raise ValueError(
            f"[BacktestService] No data after BACKTEST_START_DATE={BACKTEST_START_DATE}"
        )

    # -------------------------------------------------
    # 1) Per-asset daily returns
    # -------------------------------------------------
    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["price"].pct_change().fillna(0.0)

    # -------------------------------------------------
    # 2) Turnover + transaction costs
    # -------------------------------------------------
    df["prev_weight"] = df.groupby("ticker")["weight"].shift(1).fillna(0.0)
    df["turnover"] = (df["weight"] - df["prev_weight"]).abs()

    cost_rate = cost_bps / 10000.0
    df["cost"] = cost_rate * df["turnover"]

    # -------------------------------------------------
    # 3) Net PnL
    # -------------------------------------------------
    df["pnl_gross"] = df["ret"] * df["weight"]
    df["pnl_net"] = df["pnl_gross"] - df["cost"]

    # -------------------------------------------------
    # 4) Aggregate to portfolio level
    # -------------------------------------------------
    daily = (
        df.groupby("date", as_index=False)["pnl_net"]
          .sum()
          .rename(columns={"pnl_net": "daily_ret"})
          .sort_values("date")
    )

    # -------------------------------------------------
    # 5) Cumulative return
    # -------------------------------------------------
    daily["cumret"] = (1.0 + daily["daily_ret"]).cumprod()

    # -------------------------------------------------
    # 6) Save results
    # -------------------------------------------------
    daily.to_csv("results/backtest_results.csv", index=False)

    print(
        f"[BacktestService] Results saved to results/backtest_results.csv "
        f"(start={BACKTEST_START_DATE.date()})"
    )

    return daily
