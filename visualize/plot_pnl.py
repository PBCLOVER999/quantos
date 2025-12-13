import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_FILE = os.path.join("results", "backtest_results.csv")

def plot_performance() -> None:
    """
    Load backtest results CSV and show a simple cumulative P&L chart.
    """
    df = pd.read_csv(RESULTS_FILE, parse_dates=["date"])

    if "cumret" not in df.columns:
        raise ValueError("Expected 'cumret' column in backtest_results.csv")

    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["cumret"], label="Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("QuantOS Strategy Performance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_performance()
