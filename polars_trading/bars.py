"""Module containing functions to generate different types of bars."""

import polars as pl

from polars_trading._utils import validate_columns
from polars_trading.typing import FrameType


def _ohlcv_expr(timestamp_col: str, price_col: str, size_col: str) -> list[pl.Expr]:
    return [
        pl.first(timestamp_col).name.suffix("_start"),
        pl.last(timestamp_col).name.suffix("_end"),
        pl.first(price_col).alias("open"),
        pl.max(price_col).alias("high"),
        pl.min(price_col).alias("low"),
        pl.last(price_col).alias("close"),
        ((pl.col(size_col) * pl.col(price_col)).sum() / pl.col(size_col).sum()).alias(
            "vwap"
        ),
        pl.sum(size_col).alias("volume"),
        pl.len().alias("n_trades"),
    ]


@validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
def time_bars(
    df: FrameType,
    *,
    timestamp_col: str,
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: str = "1m",
) -> FrameType:
    """Generate time bars for a given DataFrame.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate standard bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str, optional): The name of the price column in the DataFrame.
            Defaults to "price".
        size_col (str, optional): The name of the size column in the DataFrame.
            Defaults to "size".
        symbol_col (str, optional): The name of the symbol column in the DataFrame.
            Defaults to "symbol".
        bar_size (str, optional): The size of the bars to generate.
            Can use any number followed by a time symbol. For example:

            1s = 1 second
            2m = 2 minutes
            3h = 3 hours
            4d = 4 days
            5w = 5 weeks

            Defaults to "1m".

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with time bars.

    """
    return (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(pl.col(timestamp_col).dt.truncate(bar_size).alias("__time_group"))
        .group_by("__time_group", symbol_col)
        .agg(
            *_ohlcv_expr(timestamp_col, price_col, size_col),
        )
        .rename({"__time_group": timestamp_col})
        .sort(f"{timestamp_col}_end")
    )


@validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
def tick_bars(
    df: FrameType,
    *,
    timestamp_col: str,
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: int = 100,
) -> FrameType:
    """Generate tick bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column, a symbol column, and a bar size as input.
    The bar size is the number of ticks that will be aggregated into a single bar.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate tick bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int): The number of ticks to aggregate into a single bar.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with tick bars.

    """
    ohlcv = (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
    )
    ohlcv = ohlcv.with_columns(
        (
            ((pl.col(symbol_col).cum_count()).over(symbol_col, "__date") - 1)
            // bar_size
        ).alias(
            "__tick_group",
        )
    )

    return (
        ohlcv.group_by("__tick_group", symbol_col, "__date")
        .agg(*_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("__tick_group", "__date")
        .sort(f"{timestamp_col}_end")
    )
