"""Module containing functions to generate different types of bars."""

import polars as pl
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB, parse_into_expr
from polars_trading.config import column_names
from polars_trading.typing import FrameType, IntoExpr, PolarsDataType


def _generate_output_schema(symbol_col: str) -> dict[str, PolarsDataType]:
    """Generate the output schema for the bar groups expression.

    Args:
    ----
        symbol_col (str): The name of the symbol column.

    Returns:
    -------
        dict[str, PolarsDataType]: The output schema for the bar groups expression.

    """
    return {
        symbol_col: pl.String,
        "ts_event_start": pl.Datetime("ns"),
        "ts_event_end": pl.Datetime("ns"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "vwap": pl.Float64,
        "n_trades": pl.UInt32,
    }


def _bar_groups_expr(
    expr: IntoExpr, bar_size: float, allow_splits: bool = True
) -> pl.Expr:
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
        allow_splits (bool): Whether to allow splitting a trade across multiple bars.

    Returns:
    -------
        pl.Expr: The expression with bar groups.

    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        kwargs={"bar_size": bar_size, "allow_splits": allow_splits},
        is_elementwise=False,
        function_name="bar_groups",
    )


def _ohlcv_expr(
    timestamp_col: IntoExpr, price_col: IntoExpr, size_col: IntoExpr
) -> list[pl.Expr]:
    ts_expr = parse_into_expr(timestamp_col)
    price_expr = parse_into_expr(price_col)
    size_expr = parse_into_expr(size_col)
    return [
        ts_expr.first().name.suffix("_start"),
        ts_expr.last().name.suffix("_end"),
        price_expr.first().alias("open"),
        price_expr.max().alias("high"),
        price_expr.min().alias("low"),
        price_expr.last().alias("close"),
        ((size_expr * price_expr).sum() / size_expr.sum()).alias("vwap"),
        size_expr.sum().alias("volume"),
        pl.len().alias("n_trades"),
    ]


def time_bars(
    df: FrameType,
    *,
    bar_size: str = "1m",
) -> FrameType:
    """Generate time bars for a given DataFrame.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate standard bars for.
        bar_size (str, optional): The size of the bars to generate.
            Can use any number followed by a time symbol. For example:

            1s = 1 second
            2m = 2 minutes
            3h = 3 hours
            4d = 4 days
            5w = 5 weeks

            Defaults to "1m".

    Raises:
    ------
        ValueError: If the DataFrame does not contain the required columns.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with time bars.

    """
    timestamp_col = column_names.timestamp
    price_col = column_names.price
    size_col = column_names.size
    symbol_col = column_names.symbol

    output_schema = _generate_output_schema(symbol_col)

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
        .cast(output_schema)
    )


def tick_bars(
    df: FrameType,
    *,
    bar_size: int = 100,
    split_by_date: bool = True,
) -> FrameType:
    """Generate tick bars for a given DataFrame.

    The function takes a DataFrame and generates tick bars by grouping a fixed number
    of ticks into each bar. This function will never overlap bars between different days.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate tick bars for.
        bar_size (int): The number of ticks to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with tick bars.

    """
    timestamp_col = column_names.timestamp
    price_col = column_names.price
    size_col = column_names.size
    symbol_col = column_names.symbol

    output_schema = _generate_output_schema(symbol_col)

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
    return bars.cast(output_schema)


def volume_bars(
    df: FrameType,
    *,
    bar_size: int = 10_000,
    split_by_date: bool = True,
) -> FrameType:
    """Generate volume bars for a given DataFrame.

    The function takes a DataFrame and generates volume bars by grouping a fixed volume
    of trades into each bar. This function will never overlap bars between different
    days when split_by_date is True.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate volume bars for.
        bar_size (int): The volume to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with volume bars.

    """
    timestamp_col = column_names.timestamp
    price_col = column_names.price
    size_col = column_names.size
    symbol_col = column_names.symbol

    output_schema = _generate_output_schema(symbol_col)

    over_cols = [symbol_col]
    bars = df.drop_nulls(subset=price_col).sort(timestamp_col)
    if split_by_date:
        bars = bars.with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
        over_cols.append("__date")

    bars = bars.with_columns(
        _bar_groups_expr(size_col, bar_size).over(*over_cols).alias("__bar_groups")
    )

    bars = (
        bars.explode("__bar_groups")
        .unnest("__bar_groups")
        .group_by(*over_cols, "bar_group__id")
        .agg(_ohlcv_expr(timestamp_col, price_col, "bar_group__amount"))
        .drop("bar_group__id")
        .sort(f"{timestamp_col}_end")
    )
    if split_by_date:
        bars = bars.drop("__date")
    return bars.cast(output_schema)


def dollar_bars(
    df: FrameType,
    *,
    bar_size: float = 1_000_000.0,
    split_by_date: bool = True,
) -> FrameType:
    """Generate dollar bars for a given DataFrame.

    The function takes a DataFrame and generates dollar bars by grouping a fixed dollar
    volume of trades into each bar. This function will never overlap bars between different days.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate dollar bars for.
        bar_size (float): The dollar volume to aggregate into a single bar.
        split_by_date (bool): Whether to split bars by date or not.

    Returns:
    -------
        FrameType: The DataFrame/LazyFrame with dollar bars.

    """
    timestamp_col = column_names.timestamp
    price_col = column_names.price
    size_col = column_names.size
    symbol_col = column_names.symbol

    output_schema: dict[str, PolarsDataType] = {
        symbol_col: pl.String,
        "ts_event_start": pl.Datetime("ns"),
        "ts_event_end": pl.Datetime("ns"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Int64,
        "vwap": pl.Float64,
        "n_trades": pl.UInt32,
    }

    over_cols = [symbol_col]
    bars = (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(pl.all().repeat_by(size_col))
        .explode(pl.all())
        .with_columns(pl.lit(1).alias(size_col))
    )
    if split_by_date:
        bars = bars.with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
        over_cols.append("__date")

    bars = bars.with_columns(
        _bar_groups_expr(price_col, bar_size, allow_splits=False)
        .over(*over_cols)
        .alias("__bar_groups")
    )

    bars = (
        bars.with_columns(pl.col("__bar_groups").list.first())
        .unnest("__bar_groups")
        .group_by(symbol_col, price_col, timestamp_col, "bar_group__id")
        .agg(pl.first("__date"), pl.col(size_col).sum(), pl.sum("bar_group__amount"))
        .group_by(*over_cols, "bar_group__id")
        .agg(_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("bar_group__id")
    )
    if split_by_date:
        bars = bars.drop("__date")
    return bars.cast(output_schema)
