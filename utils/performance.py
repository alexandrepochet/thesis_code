import numpy as np
import pandas as pd

def max_drawdown(curve):
    hwm = [0]
    idx = curve.index
    drawdown = pd.Series(index = idx)
    duration = pd.Series(index = idx)

    for t in range(1, len(idx)):
        cur_hwm = max(hwm[t-1], curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= hwm[t] - curve[t]
    return drawdown.max()

def sharpe_ratio(returns):
    periods = len(returns)
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)

def std(returns):
    return np.std(returns)

def mean(returns):
    return np.mean(returns)