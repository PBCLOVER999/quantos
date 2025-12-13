# factors/library/basic_factors.py

import numpy as np
import pandas as pd


# ------------------------
# Helper: RSI
# ------------------------
def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ------------------------
# Per-ticker factor block
# ------------------------
def _compute_factors_for_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical factors for a single ticker.
    Expects columns:
      - date
      - price
    """
    df = df.sort_values("date").copy()
    price = df["price"].astype(float)

    # --- Returns ---
    df["ret_1d"] = price.pct_change(1)
    df["ret_5d"] = price.pct_change(5)
    df["ret_20d"] = price.pct_change(20)

    # --- EMAs ---
    df["ema_10"] = price.ewm(span=10, adjust=False).mean()
    df["ema_20"] = price.ewm(span=20, adjust=False).mean()
    df["ema_50"] = price.ewm(span=50, adjust=False).mean()
    df["ema_200"] = price.ewm(span=200, adjust=False).mean()

    # --- MACD (12/26 EMA) ---
    ema_12 = price.ewm(span=12, adjust=False).mean()
    ema_26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # --- Volatility & ATR-ish range ---
    daily_ret = price.pct_change()
    df["vol_10"] = daily_ret.rolling(10).std()
    df["vol_20"] = daily_ret.rolling(20).std()

    high_low = price.rolling(2).max() - price.rolling(2).min()
    df["atr_14"] = high_low.rolling(14).mean()

    # --- Oscillators ---
    df["rsi_14"] = _compute_rsi(price, 14)
    df["rsi_2"] = _compute_rsi(price, 2)

    # Stochastics
    low_14 = price.rolling(14).min()
    high_14 = price.rolling(14).max()
    df["stoch_k"] = 100.0 * (price - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Williams %R
    df["williams_r"] = -100.0 * (high_14 - price) / (high_14 - low_14)

    return df


# ------------------------
# Public API
# ------------------------
def compute_basic_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical factors per ticker on a multi-asset dataframe.

    Input df must have:
      - date
      - price
      - ticker

    Returns df with factor columns added. Drops warm-up NaNs.
    """
    required_cols = {"date", "price", "ticker"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[BasicFactors] Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.groupby("ticker", group_keys=False)
          .apply(_compute_factors_for_group)
    )

    # Drop warm-up rows where we don't yet have history
    df = df.dropna().reset_index(drop=True)

    return df
