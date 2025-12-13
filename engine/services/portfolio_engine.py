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
    Regime-aware cross-sectional momentum portfolio.

    - Risk-ON  → full momentum exposure
    - Risk-OFF → same alpha, reduced gross
    """

    day = day.copy()
    day["weight_target"] = 0.0

    # ------------------------------------------------
    # 1) Regime → gross target
    # ------------------------------------------------
    reg = float(day["regime"].iloc[0]) if "regime" in day.columns else 1.0
    gross_target = REGIME_GROSS_ON if reg > 0 else REGIME_GROSS_OFF

    # ------------------------------------------------
    # 2) Signals
    # ------------------------------------------------
    sig = day["raw_signal"].astype(float)
    longs  = sig > 0
    shorts = sig < 0

    n_long  = longs.sum()
    n_short = shorts.sum()

    if n_long == 0 and n_short == 0:
        day["weight"] = 0.0
        return day

    # ------------------------------------------------
    # 3) Equal-weight raw allocation (NO scaling yet)
    # ------------------------------------------------
    if n_long > 0:
        day.loc[longs, "weight_target"] = 1.0 / n_long

    if n_short > 0:
        day.loc[shorts, "weight_target"] = -1.0 / n_short

    # ------------------------------------------------
    # 4) Normalize to gross_target
    # ------------------------------------------------
    gross = day["weight_target"].abs().sum()
    if gross > 0:
        day["weight_target"] *= (gross_target / gross)

    # ------------------------------------------------
    # 5) Per-asset hard cap
    # ------------------------------------------------
    day["weight_target"] = day["weight_target"].clip(
        lower=-MAX_WEIGHT_PER_ASSET,
        upper=MAX_WEIGHT_PER_ASSET
    )

    # ------------------------------------------------
    # 6) FINAL gross clamp (safety)
    # ------------------------------------------------
    gross_now = day["weight_target"].abs().sum()
    if gross_now > MAX_GROSS and gross_now > 0:
        day["weight_target"] *= (MAX_GROSS / gross_now)

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
