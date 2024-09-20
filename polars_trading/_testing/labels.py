import pandas as pd


def get_daily_vol(close: pd.Series, span0: int = 100) -> pd.Series:
    # This function calculates returns as close to 24 hours ago as possible and then
    # calculates the exponentially weighted moving standard deviation of those returns.
    close = close.sort_index().copy()
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(
        close.index[df0], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=span0).std().fillna(0.0)


def apply_pt_sl_on_t1(
    close: pd.Series, events: pd.DataFrame, pt_sl: tuple[float, float]
) -> None:
    """Apply stop loss and profit taking.

    AFML pg. 45
    """
