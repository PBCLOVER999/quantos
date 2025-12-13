# engine/services/walkforward_service.py

import pandas as pd
import numpy as np


def run_walkforward(
    df_results: pd.DataFrame,
    start_date: str = "2005-01-01",
    train_years: int = 5,
    test_years: int = 1,
    out_csv: str = "results/walkforward_results.csv",
) -> pd.DataFrame:
    """
    Walk-forward evaluation using realized DAILY RETURNS.

    Expects df_results columns:
      - date
      - daily_ret

    Produces OOS metrics per window.
    """

    # -----------------------------
    # 0) Validation
    # -----------------------------
    required = {"date", "daily_ret"}
    missing = required - set(df_results.columns)
    if missing:
        raise ValueError(f"[WalkForward] Missing required columns: {missing}")

    df = df_results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    start = pd.to_datetime(start_date)
    end = df["date"].max()

    rows = []
    t0 = start

    # -----------------------------
    # 1) Rolling windows
    # -----------------------------
    while True:
        train_end = t0 + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        test_slice = df[
            (df["date"] >= train_end) &
            (df["date"] < test_end)
        ].copy()

        if len(test_slice) < 50:
            t0 += pd.DateOffset(years=1)
            continue

        # -----------------------------
        # 2) OOS metrics
        # -----------------------------
        mean_ret = test_slice["daily_ret"].mean()
        vol = test_slice["daily_ret"].std()

        sharpe = (
            mean_ret / vol * np.sqrt(252)
            if vol > 0 else 0.0
        )

        cumret = (1.0 + test_slice["daily_ret"]).prod()
        max_dd = (
            (1.0 + test_slice["daily_ret"])
            .cumprod()
            .div((1.0 + test_slice["daily_ret"]).cumprod().cummax())
            .sub(1.0)
            .min()
        )

        rows.append({
            "train_start": t0.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "test_start": train_end.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "n_test_days": int(len(test_slice)),
            "oos_cumret": float(cumret),
            "oos_sharpe": float(sharpe),
            "oos_max_drawdown": float(max_dd),
        })

        # roll forward one year
        t0 += pd.DateOffset(years=1)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    print(f"[WalkForward] Results saved to {out_csv} (windows={len(out)})")
    return out

