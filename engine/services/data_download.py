# engine/services/data_download.py

import yfinance as yf
import pandas as pd
import os

RAW_DIR = "quantos/data/raw_multi"

TICKERS = [
    "SPY",
    "QQQ",
    "AAPL",
    "TSLA",
    "GLD",
    "TLT",
    "BITO",   # Bitcoin ETF
    "IWM"     # Small Caps
]

def download_single(ticker: str):
    print(f"[DataDownload] Downloading {ticker} ...")

    df = yf.download(ticker, start="1990-01-01", progress=False)

    if df.empty:
        print(f"[DataDownload] ERROR: No data for {ticker}")
        return

    df = df.reset_index()

    # Standardize column names for QuantOS
    df = df.rename(columns={
        "Date": "date",
        "Close": "price"
    })

    df = df[["date", "price"]]  # keep only required columns
    df["ticker"] = ticker

    out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df.to_csv(out_path, index=False)

    print(f"[DataDownload] Saved -> {out_path}")


def download_all():
    print("[DataDownload] Starting full download...")
    os.makedirs(RAW_DIR, exist_ok=True)

    for t in TICKERS:
        download_single(t)

    print("[DataDownload] COMPLETE.")


if __name__ == "__main__":
    download_all()
