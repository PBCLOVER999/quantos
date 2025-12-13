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
    # ============================================================
    df_factors = run_factor_engine()

    # ============================================================
    # 2. SIGNAL ENGINE
    # ============================================================
    df_signals = run_signal_engine(df_factors)

    # ============================================================
    # 3. PORTFOLIO ENGINE
    # ============================================================
    df_portfolio = build_risk_managed_mom_portfolio(df_signals)

    # ============================================================
    # 4. EXECUTION ENGINE (NO LOOKAHEAD)
    # ============================================================
    df_exec = apply_execution_lag(df_portfolio, lag_days=2)

    # ============================================================
    # 5. BACKTEST ENGINE
    # ============================================================
    df_results = run_backtest_service(df_exec)

    # ============================================================
    # 6. WALK-FORWARD (DISABLED FOR NOW)
    # ------------------------------------------------------------
    # Walk-forward is intentionally disabled until:
    #   - Core strategy validated
    #   - Hyperparameters stabilized
    #
    # Uncomment ONLY when ready.
    # ============================================================
    #
    # from engine.services.walkforward_service import run_walkforward
    #
    # run_walkforward(
    #     df_results,
    #     start_date="2005-01-01",
    #     train_years=5,
    #     test_years=1,
    #     out_csv="results/walkforward_summary.csv",
    # )

    # ============================================================
    # 7. PERFORMANCE ENGINE
    # ============================================================
    perf = compute_performance(df_results)

    print("[QuantOS] All services completed successfully.")
    print("[QuantOS] Results saved to results/backtest_results.csv")
    print("[QuantOS] Performance summary saved to results/perf_summary.csv")

    return perf


if __name__ == "__main__":
    main()
