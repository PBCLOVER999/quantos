import numpy as np

def sharpe(returns, risk_free=0.0):
    """
    Compute annualized Sharpe ratio.
    returns: pandas Series of strategy returns
    """
    excess = returns - risk_free
    return np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)


def max_drawdown(cumulative_returns):
    """
    Compute max drawdown from a cumulative return series.
    """
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def summary_stats(df):
    """
    Generate a simple performance summary.
    df must contain 'strategy_ret'
    """
    cum = (1 + df["strategy_ret"]).cumprod()
    
    return {
        "final_return": cum.iloc[-1] - 1,
        "sharpe": sharpe(df["strategy_ret"]),
        "max_drawdown": max_drawdown(cum)
    }
