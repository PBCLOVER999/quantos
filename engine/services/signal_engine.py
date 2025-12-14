# engine/services/signal_engine.py

from __future__ import annotations

import pandas as pd
import numpy as np

from signals.alpha.basic_signals import compute_cross_sectional_momentum

# ====================================================
# REGIME CONFIG
# ====================================================

REGIME_TICKER = "SPY"
REGIME_EMA_COL = "ema_200"

# ====================================================
# SIGNAL SMOOTHING CONFIG  <<< OPTION C >>>
# ====================================================

SIGNAL_EMA_HALFLIFE = 5        # persistence
SIGNAL_DEADZONE = 0.05         # avoid micro churn


# ====================================================
# MAIN SIGNAL ENGINE
# ====================================================

def run_signal_engine(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Signal Engine
    - Alpha: Cross-sectional momentum
    - Regime: SPY EMA-200
    - Signal smoothing for persistence
    """

    print("[QuantOS][SignalEngine] Computing cross-sectional momentum signals...")

    # ------------------------------------------------
    # 1) Alpha
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df_factors)

    # ------------------------------------------------
    # 2) Regime (default risk-on)
    # ------------------------------------------------
    df["regime"] = 1.0

    if REGIME_TICKER in df["ticker"].unique():
        spy = (
            df[df["ticker"] == REGIME_TICKER]
            .loc[:, ["date", "price", REGIME_EMA_COL]]
            .dropna()
            .rename(columns={
                "price": "spy_price",
                REGIME_EMA_COL: "spy_ema_200"
            })
        )

        if not spy.empty:
            spy["regime_spy"] = (spy["spy_price"] > spy["spy_ema_200"]).astype(float)
            df = df.merge(
                spy[["date", "regime_spy"]],
                on="date",
                how="left"
            )
            df["regime"] = df["regime_spy"].fillna(df["regime"])
            df.drop(columns=["regime_spy"], inplace=True)

    # ------------------------------------------------
    # 3) SIGNAL SMOOTHING (CRITICAL FIX)
    # ------------------------------------------------
    df = df.sort_values(["ticker", "date"])

    df["raw_signal"] = (
        df.groupby("ticker")["raw_signal"]
          .apply(
              lambda s: s.ewm(
                  halflife=SIGNAL_EMA_HALFLIFE,
                  adjust=False
              ).mean()
          )
          .reset_index(level=0, drop=True)
    )

    # Deadzone
    df.loc[df["raw_signal"].abs() < SIGNAL_DEADZONE, "raw_signal"] = 0.0

    # ------------------------------------------------
    # 4) Regime gating (AFTER smoothing)
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

    out = df[cols].copy()

    print("[QuantOS][SignalEngine] Signal snapshot:")
    print(out.head())

    return out
