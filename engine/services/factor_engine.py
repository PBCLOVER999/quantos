# engine/services/factor_engine.py

import pandas as pd

from engine.services.data_ingest_multi import load_multi_asset_data
from factors.library.basic_factors import compute_basic_factors


def run_factor_engine() -> pd.DataFrame:
    """
    Factor Engine

    Responsibilities:
    -----------------
    1) Load multi-asset price data
    2) Compute technical factors per ticker (independently)
    3) Return a single combined DataFrame

    Output columns include (at minimum):
      - date
      - ticker
      - price
      - factor columns from basic_factors.py
    """

    print("[QuantOS][FactorEngine] Loading multi-asset data...")
    df_raw = load_multi_asset_data()

    required = {"date", "ticker", "price"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"[FactorEngine] Missing required columns: {missing}")

    # Safety: enforce types + sort
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    print("[QuantOS][FactorEngine] Computing factors per ticker...")

    all_frames = []

    for ticker, subdf in df_raw.groupby("ticker"):
        print(f"[FactorEngine] Processing {ticker} ({len(subdf)} rows)...")

        subdf = subdf.sort_values("date").reset_index(drop=True)

        # Compute factors (this already drops NaNs internally)
        factors = compute_basic_factors(subdf)

        # Ensure ticker column exists and is correct
        factors["ticker"] = ticker

        all_frames.append(factors)

    if not all_frames:
        raise RuntimeError("[FactorEngine] No factor data produced.")

    # Combine all tickers into one dataframe
    result = pd.concat(all_frames, ignore_index=True)

    # Final sanity sort
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    print("[QuantOS][FactorEngine] Factor snapshot:")
    print(result.head())

    return result
