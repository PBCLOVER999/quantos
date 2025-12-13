# engine/backtest_engine.py

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    starting_equity: float = 100_000.0,
    trade_cost_bps: float = 1.0,
    slippage_bps: float = 0.5,
) -> pd.DataFrame:
    """
    Backtest engine with simple transaction costs + slippage.

    Parameters
    ----------
    df : DataFrame
        Must contain at least:
          - 'date'      : datetime-like
          - 'price'     : asset price
          - 'ret_1d'    : daily underlying return
          - 'position'  : position for the NEXT day (-1, 0, 1, etc.)

    starting_equity : float
        Initial equity in dollars.

    trade_cost_bps : float
        Commission cost in basis points per 1.0 unit of turnover.

    slippage_bps : float
        Extra cost in basis points per 1.0 unit of turnover to model slippage.

    Returns
    -------
    DataFrame with original columns plus:
      - position_prev
      - strategy_ret_gross
      - commission_cost
      - slippage_cost
      - trade_cost
      - strategy_ret
      - cumret
      - equity
    """

    required = {"date", "price", "ret_1d", "position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[Backtest] Missing required columns: {missing}")

    # Work on a clean, time-ordered copy
    data = df.copy().sort_values("date").reset_index(drop=True)

    # Ensure numeric types
    data["price"] = data["price"].astype(float)
    data["ret_1d"] = data["ret_1d"].astype(float)
    data["position"] = data["position"].astype(float)

    # Yesterday's position drives today's PnL
    data["position_prev"] = data["position"].shift(1).fillna(0.0)

    # Gross strategy return (before any costs)
    data["strategy_ret_gross"] = data["position_prev"] * data["ret_1d"]

    # Turnover in position units
    turnover_units = (data["position"] - data["position_prev"]).abs()

    # Per-unit costs in return space (bps -> fraction)
    commission_per_unit = trade_cost_bps / 10_000.0
    slippage_per_unit = slippage_bps / 10_000.0

    # Split costs so we can see them separately
    data["commission_cost"] = turnover_units * commission_per_unit
    data["slippage_cost"] = turnover_units * slippage_per_unit

    # Total cost
    data["trade_cost"] = data["commission_cost"] + data["slippage_cost"]

    # Net strategy return after all costs
    data["strategy_ret"] = data["strategy_ret_gross"] - data["trade_cost"]

    # Cumulative return (net of costs)
    data["cumret"] = (1.0 + data["strategy_ret"]).cumprod() - 1.0

    # Equity curve in dollars
    data["equity"] = starting_equity * (1.0 + data["cumret"])

    # Return full dataframe (all original cols + new ones)
    return data
