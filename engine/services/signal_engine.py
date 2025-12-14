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
# SIGNAL SMOOTHING  (OPTION C)
# ====================================================

SIGNAL_EMA_HALFLIFE = 3      # persistence
SIGNAL_DEADZONE = 0.05       # suppress micro churn

# ====================================================
# UNIVERSE CONDITIONING  (OPTION D)
# ====================================================

MIN_VOL_20 = 0.01            # avoids dead assets
MIN_PRICE = 5.0              # avoids garbage


# ====================================================
# MAIN SIGNAL ENGINE
# ====================================================

def run_signal_engine(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    Signal Engine (STABLE VERSION)

    Pipeline order (CRITICAL):
    1. Alpha (cross-sectional momentum)
    2. Universe conditioning
    3. Regime construction (SPY EMA-200)
    4. Signal smoothing (EMA)
    5. Regime gating
    """

    print("[QuantOS][SignalEngine] Computing cross-sectional momentum signals...")

    # ------------------------------------------------
    # 0) Defensive copy + sort
    # ------------------------------------------------
    df = df_factors.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ------------------------------------------------
    # 1) Alpha (creates raw_signal, cs_rank, mom_252)
    # ------------------------------------------------
    df = compute_cross_sectional_momentum(df)

    # HARD GUARANTEE — raw_signal always exists
    if "raw_signal" not in df.columns:
        raise RuntimeError("[SignalEngine] raw_signal missing after alpha computation")

    # ------------------------------------------------
    # 2) Universe conditioning (SAFE, NO LOOKAHEAD)
    # ------------------------------------------------
    universe_mask = pd.Series(True, index=df.index)

    if "price" in df.columns:
        universe_mask &= df["price"] >= MIN_PRICE

    if "vol_20" in df.columns:
        universe_mask &= df["vol_20"] >= MIN_VOL_20

    # Zero signal for assets outside universe
    df.loc[~universe_mask, "raw_signal"] = 0.0

    # ------------------------------------------------
    # 3) Regime construction (DEFAULT RISK-ON)
    # ------------------------------------------------
    df["regime"] = 1.0

    if REGIME_TICKER in df["ticker"].unique() and REGIME_EMA_COL in df.columns:
        spy = (
            df[df["ticker"] == REGIME_TICKER]
            .loc[:, ["date", "price", REGIME_EMA_COL]]
            .dropna()
            .rename(columns={
                "price": "spy_price",
                REGIME_EMA_COL: "spy_ema"
            })
        )

        if not spy.empty:
            spy["regime_spy"] = (spy["spy_price"] > spy["spy_ema"]).astype(float)

            df = df.merge(
                spy[["date", "regime_spy"]],
                on="date",
                how="left"
            )

            df["regime"] = df["regime_spy"].fillna(df["regime"])
            df.drop(columns=["regime_spy"], inplace=True)

    # ------------------------------------------------
    # 4) Signal smoothing (EMA per ticker)
    # ------------------------------------------------
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

    # Deadzone (post-smoothing)
    df.loc[df["raw_signal"].abs() < SIGNAL_DEADZONE, "raw_signal"] = 0.0

    # ------------------------------------------------
    # 5) Regime gating (FINAL)
    # ------------------------------------------------
    df["raw_signal"] = df["raw_signal"] * df["regime"]

    # ------------------------------------------------
    # 6) Output
    # ------------------------------------------------
    out_cols = [
        "date",
        "ticker",
        "price",
        "mom_252",
        "cs_rank",
        "raw_signal",
        "regime",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    out = df[out_cols].copy()

    print("[QuantOS][SignalEngine] Signal snapshot:")
    print(out.head())

    return out
