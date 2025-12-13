# signals/alpha/basic_signals.py

import numpy as np
import pandas as pd

# ====================================================
# STRATEGY CONFIG
# ====================================================

LOOKBACK_DAYS = 252        # 12-month momentum
LONG_PCT = 0.30            # top 30% long
SHORT_PCT = 0.30           # bottom 30% short
MIN_ASSETS = 4             # minimum assets required per day

# Vol-adjusted ranking (use existing factor vol_20)
VOL_COL = "vol_20"
MIN_VOL = 1e-8


# ====================================================
# 1) PER-TICKER MOMENTUM
# ====================================================

def _add_momentum_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute momentum per ticker:
        mom_252 = price / price.shift(LOOKBACK_DAYS) - 1
    """
    df = df.sort_values("date").copy()
    df["mom_252"] = df["price"] / df["price"].shift(LOOKBACK_DAYS) - 1.0
    return df


# ====================================================
# 2) CROSS-SECTIONAL RANKING (PER DAY)
# ====================================================

def _rank_cross_section_for_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a single date:
      - compute alpha_score = mom_252 / vol_20  (risk-adjusted momentum)
      - rank by alpha_score
      - assign +1 to top X%
      - assign -1 to bottom X%
    """
    df = df.copy()
    df["raw_signal"] = 0.0
    df["cs_rank"] = np.nan
    df["alpha_score"] = np.nan

    # Need mom_252 + vol column to trade
    needed = ["mom_252"]
    if VOL_COL in df.columns:
        needed.append(VOL_COL)

    tradable = df.dropna(subset=needed).copy()

    if len(tradable) < MIN_ASSETS:
        return df

    # Avoid divide-by-zero
    if VOL_COL in tradable.columns:
        vol = tradable[VOL_COL].clip(lower=MIN_VOL)
        tradable["alpha_score"] = tradable["mom_252"] / vol
    else:
        # Fallback: rank pure momentum if vol_20 isn't present
        tradable["alpha_score"] = tradable["mom_252"]

    # Sort strongest → weakest
    tradable = tradable.sort_values("alpha_score", ascending=False)

    n = len(tradable)
    n_long = int(np.floor(LONG_PCT * n))
    n_short = int(np.floor(SHORT_PCT * n))

    if n_long == 0 and n_short == 0:
        return df

    long_idx = tradable.iloc[:n_long].index
    short_idx = tradable.iloc[-n_short:].index

    # Rank 1..N
    tradable["cs_rank"] = np.arange(1, n + 1)

    # Write back
    df.loc[tradable.index, "alpha_score"] = tradable["alpha_score"]
    df.loc[tradable.index, "cs_rank"] = tradable["cs_rank"]
    df.loc[long_idx, "raw_signal"] = 1.0
    df.loc[short_idx, "raw_signal"] = -1.0

    return df


# ====================================================
# 3) MAIN ENTRYPOINT (PURE ALPHA — NO REGIME HERE)
# ====================================================

def compute_cross_sectional_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional momentum (vol-adjusted).

    Required:
      - date, ticker, price
    Uses if available:
      - vol_20 (preferred) for alpha_score

    Output:
      - mom_252, alpha_score, cs_rank, raw_signal
    """
    required = {"date", "ticker", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[CSMomentum] Missing required columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    # 1) Compute momentum per ticker
    out = (
        out.groupby("ticker", group_keys=False)
           .apply(_add_momentum_per_ticker)
           .reset_index(drop=True)
    )

    # 2) Rank cross-section per day using alpha_score
    out = (
        out.groupby("date", group_keys=False)
           .apply(_rank_cross_section_for_day)
           .reset_index(drop=True)
    )

    return out
