import numpy as np
import pandas as pd

def max_drawdown(curve):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the 
    pnl_returns is a pandas Series.

    Parameters:
    pnl - A pandas Series representing period percentage returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve 
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    idx = curve.index
    drawdown = pd.Series(index = idx)
    duration = pd.Series(index = idx)

    # Loop over the index range
    for t in range(1, len(idx)):
        cur_hwm = max(hwm[t-1], curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - curve[t]
    return drawdown.max()

def sharpe_ratio(returns):
    """
    Create the Sharpe ratio for the strategy, based on a 
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    periods = len(returns)
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)

def std(returns):
    return np.std(returns)

def mean(returns):
    return np.mean(returns)