# engine/services/signal_engine.py

import numpy as np
import pandas as pd

from signals.alpha.basic_signals import compute_cross_sectional_momentum

# ====================================================
# REGIME CONFIG
# ====================================================

REGIME_TICKER = "SPY"
REGIME_EMA_COL = "ema_200"

# 🔒 Regime persistence (ANTI-WHIPSAW)
REGIME_CONFIRM_DAYS = 10   # <<< THIS IS THE KEY CHANGE


# ====================================================
# MAIN SIGNAL ENGINE
# ====================================================

def run_signal_engine(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Signal Engine
    - Alpha: Cross-sectional momentum (pure)
    - Regime: SPY EMA-200 with persistence filter
    """

    print("[QuantOS][SignalEngine] Computing cross-sectional momentum signals...")

    # ------------------------------------------------
    # 1) Alpha (PURE)
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df_factors)

    # ------------------------------------------------
    # 2) Default regime = OFF (safe)
    # ------------------------------------------------
    df["regime"] = 0.0

    # ------------------------------------------------
    # 3) SPY EMA-200 regime detection
    # ------------------------------------------------
    if REGIME_TICKER not in df["ticker"].unique():
        print("[SignalEngine] WARNING: SPY not found — forcing risk-on.")
        df["regime"] = 1.0
    else:
        spy = (
            df[df["ticker"] == REGIME_TICKER]
            .loc[:, ["date", "price", REGIME_EMA_COL]]
            .dropna()
            .rename(columns={
                "price": "spy_price",
                REGIME_EMA_COL: "spy_ema"
            })
            .sort_values("date")
        )

        if spy.empty:
            print("[SignalEngine] WARNING: SPY EMA unavailable — forcing risk-on.")
            df["regime"] = 1.0
        else:
            # Raw regime
            spy["raw_regime"] = (spy["spy_price"] > spy["spy_ema"]).astype(int)

            # 🔒 Persistence filter
            spy["regime"] = (
                spy["raw_regime"]
                .rolling(REGIME_CONFIRM_DAYS)
                .mean()
                .ge(1.0)
                .astype(int)
            )

            df = df.merge(
                spy[["date", "regime"]],
                on="date",
                how="left"
            )

            df["regime"] = df["regime"].fillna(0.0)

    # ------------------------------------------------
    # 4) Gate alpha by regime
    # ------------------------------------------------
    df["raw_signal"] = df["raw_signal"] * df["regime"]

    # ------------------------------------------------
    # 5) Final output
    # ------------------------------------------------
    cols = [
        "date",
        "ticker",
        "price",
        "mom_252",
        "cs_rank",
        "raw_signal",
        "regime",
    ]
    cols = [c for c in cols if c in df.columns]

    df_out = df[cols].copy()

    print("[QuantOS][SignalEngine] Signal snapshot:")
    print(df_out.head())

    return df_out
