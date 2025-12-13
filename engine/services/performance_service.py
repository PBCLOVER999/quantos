# engine/services/performance_service.py

import os
import pandas as pd
import numpy as np


RESULTS_PATH = "results/perf_summary.csv"


def compute_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes basic performance statistics for the portfolio.
    Expects:
        daily_ret
        cumret
    """

    required = {"daily_ret", "cumret"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Performance] Missing required columns: {missing}")

    perf = {}

    # CAGR ---------------------------------------------------------
    total_years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    perf["CAGR"] = df["cumret"].iloc[-1] ** (1 / total_years) - 1

    # Volatility ----------------------------------------------------
    perf["Volatility"] = df["daily_ret"].std() * np.sqrt(252)

    # Sharpe --------------------------------------------------------
    perf["Sharpe"] = perf["CAGR"] / perf["Volatility"] if perf["Volatility"] > 0 else 0

    # Max Drawdown --------------------------------------------------
    rolling_max = df["cumret"].cummax()
    drawdown = df["cumret"] / rolling_max - 1
    perf["Max_Drawdown"] = drawdown.min()

    # Calmar Ratio --------------------------------------------------
    perf["Calmar"] = perf["CAGR"] / abs(perf["Max_Drawdown"]) if perf["Max_Drawdown"] < 0 else np.nan

    # Save results --------------------------------------------------
    os.makedirs("results", exist_ok=True)
    perf_df = pd.DataFrame([perf])
    perf_df.to_csv(RESULTS_PATH, index=False)

    print("[Performance] Performance summary saved to results/perf_summary.csv")

    return perf_df
