# engine/services/walkforward_service.py

from __future__ import annotations

import pandas as pd
from engine.services.performance_service import compute_performance


def _as_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def run_walkforward(
    df_results: pd.DataFrame,
    start_date: str = "2005-01-01",
    train_years: int = 5,
    test_years: int = 1,
    out_csv: str = "results/walkforward_summary.csv",
) -> pd.DataFrame:
    """
    Walk-forward evaluation on already-generated backtest results.

    Assumes df_results has at least:
      - date
      - strategy_ret

    We create rolling windows:
      Train: [t, t+train_years)
      Test : [t+train_years, t+train_years+test_years)
    and compute performance on each TEST window.
    """

    if "date" not in df_results.columns:
        raise ValueError("[WalkForward] df_results missing 'date' column")
    if "strategy_ret" not in df_results.columns:
        raise ValueError("[WalkForward] df_results missing 'strategy_ret' column")

    df = df_results.copy()
    df["date"] = _as_dt(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    start = pd.to_datetime(start_date)
    last = df["date"].max()

    rows = []
    t0 = start

    while True:
        train_end = t0 + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if train_end >= last:
            break

        test_slice = df[(df["date"] >= train_end) & (df["date"] < test_end)].copy()
        if len(test_slice) < 50:
            # too small to be meaningful
            t0 = t0 + pd.DateOffset(years=test_years)
            if t0 >= last:
                break
            continue

        perf = compute_performance(test_slice)

        # compute_performance might return dict or DataFrame depending on your implementation
        if isinstance(perf, dict):
            row = perf
        elif isinstance(perf, pd.DataFrame):
            # If it's a one-row DF, convert to dict
            if len(perf) == 1:
                row = perf.iloc[0].to_dict()
            else:
                row = {"note": "performance_service returned multi-row dataframe"}
        else:
            row = {"note": f"unexpected perf type: {type(perf)}"}

        row.update(
            {
                "train_start": t0.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": train_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "n_days_test": int(len(test_slice)),
            }
        )
        rows.append(row)

        # roll forward by test window
        t0 = t0 + pd.DateOffset(years=test_years)
        if t0 >= last:
            break

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"[WalkForward] Summary saved to {out_csv} (windows={len(out)})")
    return out
