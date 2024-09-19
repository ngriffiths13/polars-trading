from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars_trading.typing import FrameType


def daily_vol(
    df: FrameType,
    timestamp_col: str,
    price_col: str,
    symbol_col: str | None = None,
    span: int = 100,
) -> FrameType:
    """This function calculates the daily volatility of a price series.

    It uses an the daily volatiity by looking back at the return from the oldest price
    in the last 24 hour period to the current price. It then calculates the exponential
    weighted standard deviation of the returns.

    Args:
    ----
        df (DataFrame): The DataFrame containing the price series.
        timestamp_col (str): The column name containing the timestamps.
        price_col (str): The column name containing the prices.
        symbol_col (str | None): The column name containing the symbols. If None, it is
            assumed that the prices are for a single symbol. Defaults to None.
        span (int): The span of the exponential weighted standard deviation. Defaults to
            100.

    Returns:
        FrameType: The DataFrame with the daily volatility.
    """
    returns = (
        df.sort(timestamp_col)
        .rolling(timestamp_col, period="24h", group_by=symbol_col)
        .agg(pl.last(price_col).truediv(pl.first(price_col)).sub(1).alias("return"))
    )
    returns = returns.filter(
        (pl.col(timestamp_col) - timedelta(hours=24))
        > pl.col(timestamp_col).min().over(symbol_col)
    )
    vol_expr = (
        pl.col("return")
        .ewm_std(span=span)
        .over(symbol_col)
        .alias("daily_return_volatility")
    )
    return_cols = (
        [
            timestamp_col,
            symbol_col,
            vol_expr,
        ]
        if symbol_col
        else [
            timestamp_col,
            vol_expr,
        ]
    )

    return returns.select(*return_cols)
