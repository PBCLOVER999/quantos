# engine/services/portfolio_engine.py

from __future__ import annotations

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

MIN_ACTIVE_ASSETS = 4     # >>> CRITICAL OOS STABILITY <<<


# ====================================================
# CORE LOGIC
# ====================================================

def _risk_model_for_day(day: pd.DataFrame) -> pd.DataFrame:
    """
    Regime-aware, volatility-scaled cross-sectional momentum portfolio.

    SAFETY GUARANTEES:
    - No lookahead
    - Vol-scaling optional
    - Hard gross cap
    - Per-asset cap
    - Minimum cross-sectional breadth gate
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
    n_active = int(longs.sum() + shorts.sum())

    # >>> HARD GATE: avoid thin / unstable cross-sections
    if n_active < MIN_ACTIVE_ASSETS:
        day["weight"] = 0.0
        return day

    # ------------------------------------------------
    # 3) Volatility scaling (SAFE)
    # ------------------------------------------------
    if "vol_20" in day.columns:
        vol = day["vol_20"].replace(0.0, np.nan)
        inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scale = inv_vol
    else:
        scale = pd.Series(1.0, index=day.index)

    raw = scale * sig.abs()

    long_sum  = raw[longs].sum()
    short_sum = raw[shorts].sum()

    if long_sum > 0:
        day.loc[longs, "weight_target"] = (
            raw[longs] / long_sum
        ) * (gross_target / 2.0)

    if short_sum > 0:
        day.loc[shorts, "weight_target"] = (
            -raw[shorts] / short_sum
        ) * (gross_target / 2.0)

    # ------------------------------------------------
    # 4) Per-asset cap
    # ------------------------------------------------
    day["weight_target"] = day["weight_target"].clip(
        lower=-MAX_WEIGHT_PER_ASSET,
        upper= MAX_WEIGHT_PER_ASSET
    )

    # ------------------------------------------------
    # 5) Normalize to target gross
    # ------------------------------------------------
    gross = day["weight_target"].abs().sum()
    if gross > 0:
        day["weight_target"] *= gross_target / gross

    # ------------------------------------------------
    # 6) Hard gross cap (FINAL SAFETY)
    # ------------------------------------------------
    gross_now = day["weight_target"].abs().sum()
    if gross_now > MAX_GROSS and gross_now > 0:
        day["weight_target"] *= MAX_GROSS / gross_now

    day["weight"] = day["weight_target"]
    return day


def _apply_turnover_cap(df: pd.DataFrame, max_turnover: float) -> pd.DataFrame:
    """
    Enforces:
    - per-asset turnover cap
    - minimum holding period
    - final gross cap
    """

    df = df.sort_values(["date", "ticker"]).copy()

    prev_w = {}
    hold_days = {}

    out = []

    for date, day in df.groupby("date"):
        day = day.copy()
        weights = []

        for _, row in day.iterrows():
            t = row["ticker"]
            target = row["weight"]

            prev = prev_w.get(t, 0.0)
            hd = hold_days.get(t, 0)

            # -------------------------------
            # MIN HOLDING PERIOD ENFORCEMENT
            # -------------------------------
            if prev != 0.0 and hd < MIN_HOLD_DAYS:
                adj = prev
                hold_days[t] = hd + 1

            else:
                delta = target - prev
                if abs(delta) > max_turnover:
                    adj = prev + np.sign(delta) * max_turnover
                else:
                    adj = target

                # Reset hold counter if position opened/changed
                if prev == 0.0 and adj != 0.0:
                    hold_days[t] = 1
                elif adj == 0.0:
                    hold_days[t] = 0
                else:
                    hold_days[t] = hold_days.get(t, 0) + 1

            prev_w[t] = adj
            weights.append(adj)

        # -------------------------------
        # FINAL GROSS CAP
        # -------------------------------
        gross = sum(abs(w) for w in weights)
        if gross > MAX_GROSS and gross > 0:
            weights = [w / gross * MAX_GROSS for w in weights]

        day["weight"] = weights
        out.append(day)

    return pd.concat(out, ignore_index=True)


# ====================================================
# PUBLIC ENTRYPOINT
# ====================================================

def build_risk_managed_mom_portfolio(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic portfolio construction pipeline.
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
