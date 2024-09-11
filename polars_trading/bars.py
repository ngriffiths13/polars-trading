"""Module containing functions to generate different types of bars."""

import polars as pl
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB, validate_columns
from polars_trading.typing import FrameType, IntoExpr


def _bar_groups_expr(expr: IntoExpr, bar_size: float) -> pl.Expr:
    """Generate bar groups for a given expression.

    This expression will return a struct column with 2 fields: `id` and `amount`.
    These represent the group id field and the amount of the original value
    included in that group.

    This is intended to be used within a workflow and not by itself. The idea
    is that after generating the bar groups, you can make a duplicate row
    for each row where the amount does not equal the original input value and
    update the value to be the amount for the first one and the remainder for
    the duplicate. Then you can genrate your bars.

    Args:
    ----
        expr (IntoExpr): The expression to generate bar groups for.
        bar_size (float): The size of the bars to generate.

    Returns:
    -------
        pl.Expr: The expression with bar groups.

    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        kwargs={"bar_size": bar_size},
        is_elementwise=False,
        function_name="bar_groups",
    )


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
    split_by_date: bool = True,
) -> FrameType:
    """Generate tick bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column,
    a symbol column, and a bar size as input.
    The bar size is the number of ticks that will be aggregated into a single bar.
    This function will never overlap bars between different days.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate tick bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int): The number of ticks to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with tick bars.

    """
    over_cols = [symbol_col]
    ohlcv = df.drop_nulls(subset=price_col).sort(timestamp_col)
    if split_by_date:
        ohlcv = ohlcv.with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
        over_cols.append("__date")

    ohlcv = ohlcv.with_columns(
        (((pl.col(symbol_col).cum_count()).over(*over_cols) - 1) // bar_size).alias(
            "__tick_group",
        )
    )

    bars = (
        ohlcv.group_by("__tick_group", *over_cols)
        .agg(*_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("__tick_group")
        .sort(f"{timestamp_col}_end")
    )
    if split_by_date:
        bars = bars.drop("__date")
    return bars


@validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
def volume_bars(
    df: FrameType,
    *,
    timestamp_col: str,
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: int = 10_000,
    split_by_date: bool = True,
) -> FrameType:
    """Generate tick bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column,
    a symbol column, and a bar size as input.
    The bar size is the volume that will be aggregated into a single bar.
    This function will never overlap bars between different days.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate tick bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int): The volume to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with tick bars.

    """
    over_cols = [symbol_col]
    ohlcv = df.drop_nulls(subset=price_col).sort(timestamp_col)
    if split_by_date:
        ohlcv = ohlcv.with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
        over_cols.append("__date")

    ohlcv = ohlcv.with_columns(
        _bar_groups_expr(size_col, bar_size).over(*over_cols).alias("__bar_groups")
    ).unnest("__bar_groups")

    bars = (
        ohlcv.vstack(
            ohlcv.filter(pl.col(size_col) != pl.col("bar_group__amount")).with_columns(
                pl.col(size_col)
                .sub(pl.col("bar_group__amount"))
                .alias("bar_group__amount"),
                pl.col("bar_group__id") + 1,
            )
        )
        .select(
            pl.all().exclude(size_col, "bar_group__amount"),
            pl.col("bar_group__amount").alias(size_col),
        )
        .group_by(*over_cols, "bar_group__id")
        .agg(_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("bar_group__id")
        .sort(f"{timestamp_col}_end")
    )
    if split_by_date:
        bars = bars.drop("__date")
    return bars


@validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
def dollar_bars(
    df: FrameType,
    *,
    timestamp_col: str,
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: int = 1_000_000,
    split_by_date: bool = True,
) -> FrameType:
    """Generate tick bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column,
    a symbol column, and a bar size as input.
    The bar size is the dollar volume that will be aggregated into a single bar.
    This function will never overlap bars between different days.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate tick bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int): The dollar volume to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with dollar bars.

    """
    over_cols = [symbol_col]
    ohlcv = (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(
            pl.col(price_col).mul(pl.col(size_col)).alias("__dollar_volume"),
        )
    )
    if split_by_date:
        ohlcv = ohlcv.with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
        over_cols.append("__date")

    ohlcv = (
        ohlcv.with_columns(
            _bar_groups_expr("__dollar_volume", bar_size)
            .over(*over_cols)
            .alias("__bar_groups")
        )
        .unnest("__bar_groups")
        .with_columns(
            pl.col("bar_group__amount")
            .truediv(pl.col(price_col))
            .alias("bar_group__amount"),
        )
    )

    bars = (
        ohlcv.vstack(
            ohlcv.filter(pl.col(size_col) != pl.col("bar_group__amount")).with_columns(
                pl.col(size_col)
                .sub(pl.col("bar_group__amount"))
                .alias("bar_group__amount"),
                pl.col("bar_group__id") + 1,
            )
        )
        .select(
            pl.all().exclude(size_col, "bar_group__amount"),
            pl.col("bar_group__amount").alias(size_col),
        )
        .group_by(*over_cols, "bar_group__id")
        .agg(_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("bar_group__id")
        .sort(f"{timestamp_col}_end")
    )
    if split_by_date:
        bars = bars.drop("__date")
    return bars
