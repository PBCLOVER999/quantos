# engine/services/risk_engine.py

import pandas as pd
import numpy as np


def apply_vol_targeting(
    df: pd.DataFrame,
    target_vol_annual: float = 0.10,
    vol_window: int = 20,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Applies volatility targeting to portfolio weights.

    Required columns:
      - date
      - weight
      - ret_1d   (portfolio-level return proxy)

    Returns df with:
      - realized_vol
      - vol_scaler
      - final_weight
    """

    required = {"date", "weight", "ret_1d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[RiskEngine] Missing columns: {missing}")

    data = df.copy().sort_values("date").reset_index(drop=True)

    # Estimate realized volatility (daily)
    data["realized_vol"] = (
        data["ret_1d"]
        .rolling(vol_window)
        .std()
        * np.sqrt(252)
    )

    # Volatility scaler
    data["vol_scaler"] = target_vol_annual / data["realized_vol"]

    # Clean + cap leverage
    data["vol_scaler"] = (
        data["vol_scaler"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, max_leverage)
    )

    # Final risk-adjusted weight
    data["final_weight"] = data["weight"] * data["vol_scaler"]

    return data
