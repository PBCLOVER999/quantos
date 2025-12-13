# engine/services/portfolio_engine.py

import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------

MAX_WEIGHT_PER_ASSET = 0.25
TARGET_GROSS_LEV = 1.0
TURNOVER_CAP_PER_DAY = 0.05


# ---------------- CORE LOGIC ----------------

def _risk_model_for_day(day: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw selection signals into volatility-scaled portfolio weights.
    Selection-based, cash-aware, risk-scaled.
    """

    day = day.copy()

    # ------------------------------------------------
    # 1) Extract raw signals
    # ------------------------------------------------
    sig = day["raw_signal"].astype(float)
    longs = sig > 0
    shorts = sig < 0

    n_long = longs.sum()
    n_short = shorts.sum()

    day["weight_target"] = 0.0

    if n_long == 0 and n_short == 0:
        return day

    # ------------------------------------------------
    # 2) Base equal weights (directional)
    # ------------------------------------------------
    if n_long > 0:
        w_long = min(TARGET_GROSS_LEV / max(n_long, 1), MAX_WEIGHT_PER_ASSET)
        day.loc[longs, "weight_target"] = w_long

    if n_short > 0:
        w_short = min(TARGET_GROSS_LEV / max(n_short, 1), MAX_WEIGHT_PER_ASSET)
        day.loc[shorts, "weight_target"] = -w_short

    # ------------------------------------------------
    # 3) Volatility scaling (inverse-vol)
    # ------------------------------------------------
    if "vol_20" in day.columns:
        vol = day["vol_20"].replace(0, np.nan)

        inv_vol = 1.0 / vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

        # Normalize cross-sectionally (mean = 1)
        inv_vol = inv_vol / inv_vol.mean()

        day["weight_target"] *= inv_vol.fillna(0.0)

    # ------------------------------------------------
    # 4) Gross exposure normalization
    # ------------------------------------------------
    gross = day["weight_target"].abs().sum()
    if gross > 0:
        day["weight_target"] = TARGET_GROSS_LEV * day["weight_target"] / gross

    # ------------------------------------------------
    # 5) Final per-asset safety cap
    # ------------------------------------------------
    day["weight_target"] = day["weight_target"].clip(
        lower=-MAX_WEIGHT_PER_ASSET,
        upper=MAX_WEIGHT_PER_ASSET
    )

    return day


def _apply_turnover_cap(df: pd.DataFrame, max_turnover: float) -> pd.DataFrame:
    """
    Limit per-asset daily turnover AFTER weights are finalized.
    """

    df = df.sort_values(["date", "ticker"]).copy()
    prev_w = {}

    out = []

    for date, day in df.groupby("date"):
        day = day.copy()
        weights = []

        for _, row in day.iterrows():
            t = row["ticker"]
            target = row["weight_target"]
            prev = prev_w.get(t, 0.0)

            delta = target - prev
            if abs(delta) > max_turnover:
                adj = prev + np.sign(delta) * max_turnover
            else:
                adj = target

            prev_w[t] = adj
            weights.append(adj)

        day["weight"] = weights
        out.append(day)

    return pd.concat(out, ignore_index=True)


def build_risk_managed_mom_portfolio(df_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Final portfolio construction.
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
