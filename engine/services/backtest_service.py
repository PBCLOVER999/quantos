# engine/services/backtest_service.py

import pandas as pd
import numpy as np

# =================================================
# CONFIG — STRUCTURAL REALISM
# =================================================

BACKTEST_START_DATE = pd.Timestamp("2005-01-01")

# Transaction cost model
SLIPPAGE_BPS = 2.0        # slippage per unit turnover
COMMISSION_BPS = 0.0     # optional, keep zero for now
TOTAL_COST_BPS = SLIPPAGE_BPS + COMMISSION_BPS


def run_backtest_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-asset backtest with turnover-based transaction costs.

    REQUIRED INPUT COLUMNS:
      - date
      - ticker
      - price
      - weight   (EXECUTED weights, already lagged)

    OUTPUT:
      - date
      - daily_ret
      - cumret

    ALSO SAVES:
      - turnover
      - cost
      - gross_ret
      - net_ret
    """

    print("[BacktestService] Running backtest engine...")

    # -------------------------------------------------
    # 0) Sanity + structural realism
    # -------------------------------------------------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= BACKTEST_START_DATE].copy()

    if df.empty:
        raise ValueError(
            f"[BacktestService] No data after {BACKTEST_START_DATE.date()}"
        )

    # -------------------------------------------------
    # 1) Asset returns
    # -------------------------------------------------
    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker")["price"].pct_change().fillna(0.0)

    # -------------------------------------------------
    # 2) Turnover calculation
    # -------------------------------------------------
    df["prev_weight"] = (
        df.groupby("ticker")["weight"]
          .shift(1)
          .fillna(0.0)
    )

    df["turnover"] = (df["weight"] - df["prev_weight"]).abs()

    # -------------------------------------------------
    # 3) Transaction costs
    # -------------------------------------------------
    cost_rate = TOTAL_COST_BPS / 10000.0
    df["cost"] = df["turnover"] * cost_rate

    # -------------------------------------------------
    # 4) PnL
    # -------------------------------------------------
    df["pnl_gross"] = df["weight"] * df["ret"]
    df["pnl_net"] = df["pnl_gross"] - df["cost"]

    # -------------------------------------------------
    # 5) Aggregate to portfolio level
    # -------------------------------------------------
    daily = (
        df.groupby("date", as_index=False)
          .agg(
              gross_ret=("pnl_gross", "sum"),
              net_ret=("pnl_net", "sum"),
              turnover=("turnover", "sum"),
              cost=("cost", "sum"),
          )
          .sort_values("date")
    )

    daily["daily_ret"] = daily["net_ret"]
    daily["cumret"] = (1.0 + daily["daily_ret"]).cumprod()

    # -------------------------------------------------
    # 6) Save results
    # -------------------------------------------------
    daily.to_csv("results/backtest_results.csv", index=False)

    print(
        f"[BacktestService] Results saved to results/backtest_results.csv "
        f"(start={BACKTEST_START_DATE.date()}, cost_bps={TOTAL_COST_BPS})"
    )

    return daily
