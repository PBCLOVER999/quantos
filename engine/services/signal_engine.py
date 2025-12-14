# engine/services/signal_engine.py

from __future__ import annotations

import pandas as pd
import numpy as np

from signals.alpha.basic_signals import compute_cross_sectional_momentum

# ====================================================
# REGIME CONFIG (MARKET TREND FILTER)
# ====================================================

REGIME_TICKER = "SPY"
REGIME_EMA_COL = "ema_200"

DEFAULT_REGIME = 0.0   # safe: risk-off until proven otherwise


# ====================================================
# MAIN SIGNAL ENGINE
# ====================================================

def run_signal_engine(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Signal Engine pipeline:

    1) Compute pure cross-sectional momentum alpha
    2) Compute market trend regime (SPY > EMA-200)
    3) Gate raw_signal by regime (NO lookahead)
    4) Output clean signal table
    """

    print("[QuantOS][SignalEngine] Computing cross-sectional momentum signals...")

    df = df_factors.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ------------------------------------------------
    # 1) PURE ALPHA (NO REGIME INSIDE)
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df)

    # ------------------------------------------------
    # 2) HARD GUARANTEE: regime column exists
    # ------------------------------------------------
    df["regime"] = DEFAULT_REGIME

    # ------------------------------------------------
    # 3) MARKET REGIME (SPY EMA-200)
    # ------------------------------------------------
    if REGIME_TICKER not in df["ticker"].unique():
        print("[SignalEngine] WARNING: SPY not found — regime forced OFF.")
    else:
        spy = (
            df[df["ticker"] == REGIME_TICKER]
            .loc[:, ["date", "price", REGIME_EMA_COL]]
            .sort_values("date")
            .copy()
        )

        spy = spy.dropna(subset=[REGIME_EMA_COL])

        if spy.empty:
            print("[SignalEngine] WARNING: SPY EMA missing — regime forced OFF.")
        else:
            spy["regime_spy"] = (spy["price"] > spy[REGIME_EMA_COL]).astype(float)

            df = df.merge(
                spy[["date", "regime_spy"]],
                on="date",
                how="left"
            )

            df["regime"] = df["regime_spy"].fillna(DEFAULT_REGIME)
            df.drop(columns=["regime_spy"], inplace=True)

    # ------------------------------------------------
    # 4) GATE ALPHA BY REGIME (SAFE)
    # ------------------------------------------------
    df["raw_signal"] = df["raw_signal"] * df["regime"]

    # ------------------------------------------------
    # 5) FINAL OUTPUT (EXPLICIT)
    # ------------------------------------------------
    out_cols = [
        "date",
        "ticker",
        "price",
        "mom_252",
        "alpha_score",
        "cs_rank",
        "raw_signal",
        "regime",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    df_out = df[out_cols].copy()

    print("[QuantOS][SignalEngine] Signal snapshot:")
    print(df_out.head())

    return df_out
