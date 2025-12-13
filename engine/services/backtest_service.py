# engine/services/backtest_service.py

from __future__ import annotations

import pandas as pd
import numpy as np

# =================================================
# CONFIG — STRUCTURAL REALISM + RISK CONTROL
# =================================================

BACKTEST_START_DATE = pd.Timestamp("2005-01-01")

# Transaction cost model
SLIPPAGE_BPS = 10.0
COMMISSION_BPS = 0.0
TOTAL_COST_BPS = SLIPPAGE_BPS + COMMISSION_BPS

# Volatility targeting
TARGET_ANNUAL_VOL = 0.12      # 12% target vol
VOL_LOOKBACK = 63             # ~3 months
MAX_LEVERAGE = 2.0            # hard cap

# -------------------------------------------------
# Drawdown governor (PORTFOLIO LEVEL)
# -------------------------------------------------
DD_GOV_ENABLED = True

# Ordered from smallest → deepest drawdown
DD_LEVELS = [
    (-0.10, 0.75),   # >10% DD → 75% exposure
    (-0.20, 0.50),   # >20% DD → 50% exposure
    (-0.30, 0.25),   # >30% DD → 25% exposure
]

DD_DEFAULT_MULT = 1.00


# =================================================
# CORE BACKTEST
# =================================================

def run_backtest_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-asset backtest with:
      - turnover-based transaction costs
      - portfolio-level volatility targeting (NO lookahead)
      - portfolio-level drawdown governor

    REQUIRED INPUT:
      - date
      - ticker
      - price
      - weight   (EXECUTED weights, already lagged)

    OUTPUT:
      - date
      - daily_ret
      - cumret

    ALSO SAVES DIAGNOSTICS:
      - gross_ret
      - net_ret
      - turnover
      - cost
      - leverage
      - rolling_vol
      - dd_mult
      - drawdown
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
    # 2) Turnover
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
    # 4) Asset-level PnL
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
          .reset_index(drop=True)
    )

    daily["raw_daily_ret"] = daily["net_ret"]

    # -------------------------------------------------
    # 6) Volatility targeting (NO LOOKAHEAD)
    # -------------------------------------------------
    daily["rolling_vol"] = (
        daily["raw_daily_ret"]
        .rolling(VOL_LOOKBACK)
        .std()
        * np.sqrt(252)
    )

    daily["leverage"] = TARGET_ANNUAL_VOL / daily["rolling_vol"]
    daily["leverage"] = daily["leverage"].clip(upper=MAX_LEVERAGE)
    daily["leverage"] = daily["leverage"].fillna(0.0)

    daily["vt_ret"] = daily["raw_daily_ret"] * daily["leverage"]

    # -------------------------------------------------
    # 7) Drawdown governor (AFTER vol targeting)
    # -------------------------------------------------
    equity = 1.0
    peak = 1.0

    dd_mults = []
    drawdowns = []
    final_rets = []

    for r in daily["vt_ret"].values:
        drawdown = equity / peak - 1.0

        mult = DD_DEFAULT_MULT
        if DD_GOV_ENABLED:
            for dd_thr, m in DD_LEVELS:
                if drawdown <= dd_thr:
                    mult = m

        gov_r = r * mult

        equity *= (1.0 + gov_r)
        peak = max(peak, equity)

        dd_mults.append(mult)
        drawdowns.append(drawdown)
        final_rets.append(gov_r)

    daily["dd_mult"] = dd_mults
    daily["drawdown"] = drawdowns
    daily["daily_ret"] = final_rets

    # -------------------------------------------------
    # 8) Equity curve
    # -------------------------------------------------
    daily["cumret"] = (1.0 + daily["daily_ret"]).cumprod()

    # -------------------------------------------------
    # 9) Save results
    # -------------------------------------------------
    daily.to_csv("results/backtest_results.csv", index=False)

    print(
        f"[BacktestService] Results saved to results/backtest_results.csv "
        f"(start={BACKTEST_START_DATE.date()}, "
        f"target_vol={TARGET_ANNUAL_VOL:.0%}, "
        f"max_lev={MAX_LEVERAGE}, "
        f"dd_gov={DD_GOV_ENABLED}, "
        f"cost_bps={TOTAL_COST_BPS})"
    )

    return daily
