# main.py

from engine.services.factor_engine import run_factor_engine
from engine.services.signal_engine import run_signal_engine
from engine.services.portfolio_engine import build_risk_managed_mom_portfolio
from engine.services.execution_engine import apply_execution_lag
from engine.services.backtest_service import run_backtest_service
from engine.services.performance_service import compute_performance


def main():
    print("[QuantOS] Booting system...")

    # ============================================================
    # 1. FACTOR ENGINE
    # Input : raw multi-asset price data
    # Output: factors per ticker / date
    # ============================================================
    df_factors = run_factor_engine()

    # ============================================================
    # 2. SIGNAL ENGINE
    # Input : factor dataframe
    # Output: signals (raw_signal, regime, etc.)
    # ============================================================
    df_signals = run_signal_engine(df_factors)

    # ============================================================
    # 3. PORTFOLIO ENGINE
    # Input : signals dataframe
    # Output: risk-managed portfolio weights
    # ============================================================
    df_portfolio = build_risk_managed_mom_portfolio(df_signals)

    # ============================================================
    # 4. EXECUTION ENGINE  ✅ NEW (NO LOOKAHEAD)
    # Input : portfolio weights
    # Output: executed weights with lag
    # ============================================================
    df_exec = apply_execution_lag(df_portfolio, lag_days=2)

    # ============================================================
    # 5. BACKTEST ENGINE
    # Input : executed portfolio
    # Output: equity curve, returns, PnL
    # ============================================================
    df_results = run_backtest_service(df_exec)

    # ============================================================
    # 6. WALK-FORWARD (OOS) EVALUATION ✅
    # ============================================================
    run_walkforward(
        df_results,
        start_date="2005-01-01",
        train_years=5,
        test_years=1,
        out_csv="results/walkforward_summary.csv",
    )

    # ============================================================
    # 6. PERFORMANCE ENGINE
    # Input : backtest results
    # Output: performance metrics
    # ============================================================
    perf = compute_performance(df_results)

    print("[QuantOS] All services completed successfully.")
    print("[QuantOS] Results saved to results/backtest_results.csv")
    print("[QuantOS] Performance summary saved to results/perf_summary.csv")

    return perf


if __name__ == "__main__":
    main()
