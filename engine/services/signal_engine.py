# engine/services/signal_engine.py

import numpy as np
import pandas as pd

from signals.alpha.basic_signals import compute_cross_sectional_momentum

# ====================================================
# REGIME CONFIG
# ====================================================

REGIME_TICKER = "SPY"
REGIME_EMA_COL = "ema_200"
REGIME_CONFIRM_DAYS = 10   # persistence filter


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
    # 1) Alpha
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df_factors)

    # ------------------------------------------------
    # 2) SAFE default regime (ALWAYS EXISTS)
    # ------------------------------------------------
    df["regime"] = 0.0

    # ------------------------------------------------
    # 3) SPY trend regime
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
            spy["raw_regime"] = (spy["spy_price"] > spy["spy_ema"]).astype(int)

            # Persistence filter
            spy["spy_regime"] = (
                spy["raw_regime"]
                .rolling(REGIME_CONFIRM_DAYS)
                .mean()
                .ge(1.0)
                .astype(int)
            )

            # Merge TEMP column
            df = df.merge(
                spy[["date", "spy_regime"]],
                on="date",
                how="left"
            )

            # 🔒 HARD ASSIGN (NO KeyError POSSIBLE)
            df["regime"] = df["spy_regime"].fillna(0.0)
            df.drop(columns=["spy_regime"], inplace=True)

    # ------------------------------------------------
    # 4) Gate alpha
    # ------------------------------------------------
    df["raw_signal"] = df["raw_signal"] * df["regime"]

    # ------------------------------------------------
    # 5) Output
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
