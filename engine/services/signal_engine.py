# engine/services/signal_engine.py

import numpy as np
import pandas as pd

from signals.alpha.basic_signals import compute_cross_sectional_momentum

# ====================================================
# REGIME CONFIG
# ====================================================

REGIME_TICKER = "SPY"
REGIME_EMA_COL = "ema_200"


# ====================================================
# MAIN SIGNAL ENGINE
# ====================================================

def run_signal_engine(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Signal Engine
    - Alpha: Cross-sectional momentum (pure)
    - Regime: SPY EMA-200 filter
    """

    print("[QuantOS][SignalEngine] Computing cross-sectional momentum signals...")

    # ------------------------------------------------
    # 1) Alpha (PURE — no regime inside)
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df_factors)

    # ------------------------------------------------
    # 🔒 HARD GUARANTEE: regime ALWAYS exists
    # ------------------------------------------------
    df["regime"] = 1.0

    # ------------------------------------------------
    # 2) Market regime detection (SPY EMA-200)
    # ------------------------------------------------
    if REGIME_TICKER not in df["ticker"].unique():
        print("[SignalEngine] WARNING: SPY not found — forcing risk-on.")
    else:
        spy = (
            df[df["ticker"] == REGIME_TICKER]
            .loc[:, ["date", "price", REGIME_EMA_COL]]
            .dropna()
            .rename(columns={
                "price": "spy_price",
                REGIME_EMA_COL: "spy_ema_200"
            })
        )

        if spy.empty:
            print("[SignalEngine] WARNING: SPY has no EMA — forcing risk-on.")
        else:
            spy["regime_spy"] = (spy["spy_price"] > spy["spy_ema_200"]).astype(float)

            df = df.merge(
                spy[["date", "regime_spy"]],
                on="date",
                how="left"
            )

            # Prefer SPY regime where available
            df["regime"] = df["regime_spy"].combine_first(df["regime"])
            df.drop(columns=["regime_spy"], inplace=True)

    # ------------------------------------------------
    # 3) Gate signals by regime (SAFE)
    # ------------------------------------------------
    df["raw_signal"] = df["raw_signal"] * df["regime"]

    # ------------------------------------------------
    # 4) Final output
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
