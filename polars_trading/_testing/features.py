"""Implementations of feature functionality in Pandas from AFML."""

import numpy as np
import pandas as pd


def get_weights_ffd(d: float, thresh: float) -> np.ndarray:
    """Calculate weights for frac_diff_ffd.

    Args:
        d: The fractional difference.
        thresh: The threshold.

    Returns:
        np.array: The weights.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thresh:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


def frac_diff_ffd(series: pd.DataFrame, d: float, thresh: float = 1e-5) -> pd.DataFrame:
    """Calculate the fractional difference of a series.

    Args:
        series: The series.
        d: The fractional difference.
        thresh: The threshold.

    Returns:
        pd.DataFrame: The fractional difference.
    """
    w = get_weights_ffd(d, thresh)
    width = len(w) - 1
    df = {}
    for name in series.columns:
        series_f = series[[name]].ffill().dropna()
        df_ = pd.Series()
        for iloc1 in range(width, series_f.shape[0]):
            loc0 = series_f.index[iloc1 - width]
            loc1 = series_f.index[iloc1]
            df_[loc1] = np.dot(w.T, series_f.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    return pd.concat(df, axis=1)
