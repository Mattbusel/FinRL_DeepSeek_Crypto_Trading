import numpy as np
import pandas as pd
import empyrical as ep

def cumulative_returns(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return a pd.Series
    """
    return ep.cum_returns(returns_pct)

def sharpe_ratio(returns_pct, risk_free=0):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return float
    """
    returns = np.array(returns_pct)
    std = returns.std(ddof=1)
    if std == 0:
        sharpe_ratio = np.inf
    else:
        sharpe_ratio = (returns.mean()-risk_free) / std
    return sharpe_ratio

def max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    return ep.max_drawdown(returns_pct)

def return_over_max_drawdown(returns_pct):
    """
    returns_pct: percentage change, i.e. (r_(t+1) - r_t)/r_t

    return: float
    """
    mdd = abs(max_drawdown(returns_pct))
    returns = cumulative_returns(returns_pct).iloc[-1]
    if mdd == 0:
        return np.inf
    return returns/mdd


