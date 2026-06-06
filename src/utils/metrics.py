import numpy as np


def calculate_metrics(wealth_curve: np.ndarray) -> dict:
    """
    Calculate performance metrics from a wealth curve.

    Parameters:
    wealth_curve (np.ndarray): Array representing the wealth curve over time.

    Returns:
    dict: A dictionary containing calculated metrics.
    """

    annualized_return = (wealth_curve[-1] / wealth_curve[0]) ** (
        252 / len(wealth_curve)
    ) - 1

    daily_returns = np.diff(wealth_curve) / wealth_curve[:-1]
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)

    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(252)

    sortino_ratio = (
        annualized_return / downside_volatility if downside_volatility > 0 else np.nan
    )

    drawdows = 1 - wealth_curve / np.maximum.accumulate(wealth_curve)
    max_drawdown = np.max(drawdows)

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else np.nan

    var_95 = np.percentile(daily_returns, 5)
    cvar_95 = np.mean(daily_returns[daily_returns <= var_95])

    skewness = (
        np.mean((daily_returns - np.mean(daily_returns)) ** 3)
        / np.std(daily_returns) ** 3
    )

    kurtosis = (
        np.mean((daily_returns - np.mean(daily_returns)) ** 4)
        / np.std(daily_returns) ** 4
    )

    return {
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "VAR (95%)": -1 * var_95,
        "ES (95%)": -1 * cvar_95,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
    }
