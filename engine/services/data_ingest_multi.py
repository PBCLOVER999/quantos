# engine/services/data_ingest_multi.py

import os
from typing import List
import pandas as pd

# Folder with one CSV per asset (SPY.csv, AAPL.csv, TSLA.csv, etc.)
RAW_MULTI_DIR = os.path.join("quantos", "data", "raw_multi")


def _find_csv_files(folder: str) -> List[str]:
    """Return full paths of all .csv files in the folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"[MultiIngest] Folder not found: {folder}")

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".csv")
    ]

    if not files:
        raise ValueError(f"[MultiIngest] No CSV files found in {folder}/")

    return sorted(files)


def _load_single_csv(path: str) -> pd.DataFrame:
    """
    Load a single CSV and normalize to:
      - date
      - price
      - ticker

    Supports:
      * Raw yfinance-style CSVs with OHLCV
      * Already-normalized CSVs with columns ['date', 'price', 'ticker']
    """
    df = pd.read_csv(path)

    # Normalize column names → lowercase + strip
    col_map = {c.lower().strip(): c for c in df.columns}

    # ---- DATE COLUMN ----
    # Prefer explicit "date", otherwise assume first column is the date
    date_col = col_map.get("date", df.columns[0])

    # ---- PRICE COLUMN ----
    price_col = None

    # 1) Look for Adjusted Close variants
    for key in col_map:
        if "adj" in key and "close" in key:
            price_col = col_map[key]
            break

    # 2) Fallback: any "close" column
    if price_col is None:
        for key in col_map:
            if "close" in key:
                price_col = col_map[key]
                break

    # 3) Fallback: already-normalized 'price' column
    if price_col is None and "price" in col_map:
        price_col = col_map["price"]

    if price_col is None:
        raise ValueError(
            f"{path} is missing a usable Close or Adj Close column! "
            f"Columns found: {list(df.columns)}"
        )

    # Build normalized dataframe
    out = df[[date_col, price_col]].copy()
    out.columns = ["date", "price"]

    # Ensure datetime
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # 🔒 NEW: force price to numeric and drop junk rows
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    # Drop any rows where date or price couldn't be parsed
    out = out.dropna(subset=["date", "price"]).reset_index(drop=True)

    # Extract ticker from filename
    ticker = os.path.splitext(os.path.basename(path))[0].upper()
    out["ticker"] = ticker

    return out


def load_multi_asset_data() -> pd.DataFrame:
    """
    Load all CSVs in RAW_MULTI_DIR and return a single DataFrame containing:
      - date
      - price
      - ticker
    """
    files = _find_csv_files(RAW_MULTI_DIR)

    frames = []
    for path in files:
        frames.append(_load_single_csv(path))

    df = pd.concat(frames, ignore_index=True)

    print("[MultiIngest] Snapshot:")
    print(df.head())

    return df
