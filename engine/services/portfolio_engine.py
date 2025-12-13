# engine/services/portfolio_engine.py

import pandas as pd
import numpy as np

# ====================================================
# CONFIG
# ====================================================

MAX_WEIGHT_PER_ASSET = 0.25

REGIME_GROSS_ON  = 1.00   # full risk-on exposure
REGIME_GROSS_OFF = 0.25   # throttled risk-off exposure

MAX_GROSS = 1.00          # absolute portfolio gross cap
TURNOVER_CAP_PER_DAY = 0.05


# ====================================================
# CORE LOGIC
# ====================================================

def _risk_model_for_day(day: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-aware, volatility-scaled cross-sectional momentum portfolio.
    """

    day = day.copy()
    day["weight_target"] = 0.0

    # ------------------------------------------------
    # 1) Regime throttle
    # ------------------------------------------------
    reg = float(day["regime"].iloc[0]) if "regime" in day.columns else 1.0
    gross_target = REGIME_GROSS_ON if reg > 0 else REGIME_GROSS_OFF

    # ------------------------------------------------
    # 2) Signals
    # ------------------------------------------------
    sig = day["raw_signal"].astype(float)
    longs  = sig > 0
    shorts = sig < 0

    if longs.sum() == 0 and shorts.sum() == 0:
        day["weight"] = 0.0
        return day

    # ------------------------------------------------
    # 3) Volatility scaling
    # ------------------------------------------------
    # Use vol_20, fallback if missing
    vol = day["vol_20"].replace(0, np.nan)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.fillna(0.0)

    # Apply direction
    raw_weights = inv_vol * sig.abs()

    # Normalize separately for long / short legs
    long_sum  = raw_weights[longs].sum()
    short_sum = raw_weights[shorts].sum()

    if long_sum > 0:
        day.loc[longs, "weight_target"] = (
            raw_weights[longs] / long_sum
        ) * (gross_target / 2)

    if short_sum > 0:
        day.loc[shorts, "weight_target"] = (
            -raw_weights[shorts] / short_sum
        ) * (gross_target / 2)

    # ------------------------------------------------
    # 4) Per-asset cap
    # ------------------------------------------------
    day["weight_target"] = day["weight_target"].clip(
        lower=-MAX_WEIGHT_PER_ASSET,
        upper= MAX_WEIGHT_PER_ASSET
    )

    # ------------------------------------------------
    # 5) Final gross normalization + hard cap
    # ------------------------------------------------
    gross = day["weight_target"].abs().sum()
    if gross > 0:
        day["weight_target"] *= (gross_target / gross)

    if day["weight_target"].abs().sum() > MAX_GROSS:
        day["weight_target"] *= MAX_GROSS / day["weight_target"].abs().sum()

    day["weight"] = day["weight_target"]
    return day

def _apply_turnover_cap(df: pd.DataFrame, max_turnover: float) -> pd.DataFrame:
    """
    Limit per-asset daily turnover.
    ALSO enforces final gross cap post-adjustment.
    """

    df = df.sort_values(["date", "ticker"]).copy()
    prev_w = {}
    out = []

    for date, day in df.groupby("date"):
        day = day.copy()
        weights = []

        for _, row in day.iterrows():
            t = row["ticker"]
            target = row["weight"]
            prev = prev_w.get(t, 0.0)

            delta = target - prev
            if abs(delta) > max_turnover:
                adj = prev + np.sign(delta) * max_turnover
            else:
                adj = target

            prev_w[t] = adj
            weights.append(adj)

        # ---------- FINAL GROSS CLIP (CRITICAL) ----------
        gross = sum(abs(w) for w in weights)
        if gross > MAX_GROSS and gross > 0:
            weights = [w / gross * MAX_GROSS for w in weights]

        day["weight"] = weights
        out.append(day)

    return pd.concat(out, ignore_index=True)


def build_risk_managed_mom_portfolio(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Full portfolio construction pipeline.
    """

    print("[PortfolioEngine] Building RISK-MANAGED cross-sectional momentum portfolio...")

    df = (
        df_signals
        .sort_values(["date", "ticker"])
        .groupby("date", group_keys=False)
        .apply(_risk_model_for_day)
        .reset_index(drop=True)
    )

    df = _apply_turnover_cap(df, TURNOVER_CAP_PER_DAY)

    port = df[["date", "ticker", "price", "raw_signal", "weight"]].copy()

    print("[QuantOS][PortfolioEngine] Portfolio snapshot:")
    print(port.head())

    df.to_csv("results/debug_portfolio.csv", index=False)
    return port
