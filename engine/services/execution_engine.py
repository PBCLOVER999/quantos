# engine/services/execution_engine.py

import pandas as pd


def apply_execution_lag(df: pd.DataFrame, lag_days: int = 2) -> pd.DataFrame:
    """
    Shifts the portfolio weights forward so trades happen lag_days later.
    This models realistic execution delay and eliminates lookahead bias.
    """
    df = df.copy().sort_values("date")

    required = {"date", "ticker", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[ExecutionEngine] Missing required columns: {missing}")

    df["exec_weight"] = df.groupby("ticker")["weight"].shift(lag_days).fillna(0.0)

    return df


def apply_holding_period(df: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    """
    Smooths weights by enforcing a holding period.
    Weight change is amortized over hold_days.
    """
    df = df.copy().sort_values("date")

    def _smooth(series):
        return series.diff().fillna(0).rolling(hold_days).mean().cumsum()

    df["exec_weight"] = (
        df.groupby("ticker")["weight"].apply(_smooth).fillna(0.0)
    )

    return df
