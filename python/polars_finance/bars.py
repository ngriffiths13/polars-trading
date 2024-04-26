from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars.utils.udfs import _get_shared_lib_location
lib = _get_shared_lib_location(__file__)
print(lib)

from polars_finance.utils import parse_into_expr
from pathlib import Path

if TYPE_CHECKING:
    from polars.type_aliases import FrameType, IntoExpr



# def _dynamic_tick_bar_groups(thresholds: IntoExpr) -> pl.Expr:
#     expr = parse_into_expr(thresholds)
#     return expr.cast(pl.UInt16).register_plugin(
#         lib=lib,
#         symbol="dynamic_tick_bar_groups",
#         is_elementwise=False,
#         cast_to_supertypes=True,
#     )


def _ohlcv_expr(timestamp_col: str, price_col: str, size_col: str) -> list[pl.Expr]:
    return [
        pl.first(timestamp_col).name.prefix("begin_"),
        pl.last(timestamp_col).name.prefix("end_"),
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


def standard_bars(
    df: FrameType,
    timestamp_col: str = "ts_event",
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: str = "1m",
) -> FrameType:
    """
    Generate standard bars for a given DataFrame.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate standard bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str, optional): The name of the price column in the DataFrame. Defaults to "price".
        size_col (str, optional): The name of the size column in the DataFrame. Defaults to "size".
        symbol_col (str, optional): The name of the symbol column in the DataFrame. Defaults to "symbol".
        bar_size (str, optional): The size of the bars to generate.
            Can use any number followed by a time symbol. For example:

            1s = 1 second
            2m = 2 minutes
            3h = 3 hours
            4d = 4 days

            Defaults to "1m".

    """
    return (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(pl.col(timestamp_col).dt.truncate(bar_size))
        .group_by(timestamp_col, symbol_col)
        .agg(
            pl.first(price_col).alias("open"),
            pl.max(price_col).alias("high"),
            pl.min(price_col).alias("low"),
            pl.last(price_col).alias("close"),
            (
                (pl.col(size_col) * pl.col(price_col)).sum() / pl.col(size_col).sum()
            ).alias("vwap"),
            pl.sum(size_col).alias("volume"),
            pl.len().alias("n_trades"),
        )
        .sort(timestamp_col)
    )


def tick_bars(
    df: FrameType,
    timestamp_col: str = "ts_event",
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: int | pl.Expr = 100,
) -> FrameType:
    """
    Generate tick bars for a given DataFrame.

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

    """
    ohlcv = (
        df.drop_nulls(subset=price_col)
        .sort(timestamp_col)
        .with_columns(pl.col(timestamp_col).dt.date().alias("__date"))
    )
    if isinstance(bar_size, int):
        ohlcv = ohlcv.with_columns(
            (
                ((pl.col(symbol_col).cum_count()).over(symbol_col, "__date") - 1)
                // bar_size
            ).alias(
                "__tick_group",
            )
        )
    # elif isinstance(bar_size, pl.Expr):
    #     ohlcv = ohlcv.with_columns(
    #         _dynamic_tick_bar_groups(bar_size)
    #         .over(symbol_col, "__date")
    #         .alias("__tick_group")
        # )
    return (
        ohlcv.group_by("__tick_group", symbol_col, "__date")
        .agg(_ohlcv_expr(timestamp_col, price_col, size_col))
        .drop("__tick_group", "__date")
        .sort(f"end_{timestamp_col}")
    )


def volume_bars(
    df: FrameType,
    timestamp_col: str = "ts_event",
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: float | pl.Expr = 1_000_000,
) -> FrameType:
    """
    Generate volume bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column,
        a symbol column, and a bar size as input.
    The bar size is the total volume that will be aggregated into a single bar.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate volume bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int | float | dict[str, int | float] | pl.Expr): The total volume to
             aggregate into a single bar.

    Returns:
    -------
        FrameType: A DataFrame with volume bars.

    """
    df = df.sort(timestamp_col)
    if isinstance(bar_size, int | float):
        df = df.with_columns(pl.lit(bar_size).cast(pl.UInt32).alias("__PFIN_bar_size"))
    elif isinstance(bar_size, pl.Expr):
        df = df.with_columns(bar_size.cast(pl.UInt32).alias("__PFIN_bar_size"))
    else:
        msg = "bar_size must be an int, float, or pl.Expr"
        raise TypeError(msg)

    return (
        df.group_by(symbol_col, pl.col(timestamp_col).dt.date())
        .agg(
            register_plugin_function(
                plugin_path=Path(__file__).parent,
                function_name="volume_bars",
                is_elementwise=False,
                cast_to_supertype=False,
                args=[pl.col(timestamp_col), pl.col(price_col), pl.col(size_col), pl.col("__PFIN_bar_size")],
                changes_length=True,
            )
        )
        .explode("ohlcv")
        .unnest("ohlcv")
        .drop(timestamp_col)
    )


def dollar_bars(
    df: FrameType,
    timestamp_col: str = "ts_event",
    price_col: str = "price",
    size_col: str = "size",
    symbol_col: str = "symbol",
    bar_size: float | pl.Expr = 1_000_000,
) -> FrameType:
    """
    Generate dollar bars for a given DataFrame.

    The function takes a DataFrame, a timestamp column, a price column, a size column,
        a symbol column, and a bar size as input.
    The bar size is the total dollar amount that will be aggregated into a single bar.

    Args:
    ----
        df (FrameType): The DataFrame/LazyFrame to generate dollar bars for.
        timestamp_col (str): The name of the timestamp column in the DataFrame.
        price_col (str): The name of the price column in the DataFrame.
        size_col (str): The name of the size column in the DataFrame.
        symbol_col (str): The name of the symbol column in the DataFrame.
        bar_size (int | float | pl.Expr): The total dollar amount to aggregate into a single bar.

    Returns:
    -------
        FrameType: A DataFrame with dollar bars.

    """
    df = df.sort(timestamp_col)
    if isinstance(bar_size, int | float):
        df = df.with_columns(pl.lit(bar_size).cast(pl.Float64).alias("__PFIN_bar_size"))
    elif isinstance(bar_size, pl.Expr):
        df = df.with_columns(bar_size.cast(pl.Float64).alias("__PFIN_bar_size"))
    else:
        msg = "bar_size must be an int, float, or pl.Expr"
        raise TypeError(msg)

    return (
        df.group_by(symbol_col, pl.col(timestamp_col).dt.date())
        .agg(
            pl.col(timestamp_col).register_plugin(
                lib=lib,
                symbol="dollar_bars",
                is_elementwise=False,
                cast_to_supertypes=False,
                args=[pl.col(price_col), pl.col(size_col), pl.col("__PFIN_bar_size")],
                changes_length=True,
            )
        )
        .explode("ohlcv")
        .unnest("ohlcv")
    )


def tick_imbalance_bars():
    raise NotImplementedError("This function has not been implemented yet.")


def volume_imbalance_bars():
    raise NotImplementedError("This function has not been implemented yet.")


def dollar_imbalance_bars():
    raise NotImplementedError("This function has not been implemented yet.")


def tick_runs_bars():
    raise NotImplementedError("This function has not been implemented yet.")


def volume_runs_bars():
    raise NotImplementedError("This function has not been implemented yet.")


def dollar_runs_bars():
    raise NotImplementedError("This function has not been implemented yet.")


# ETF Trick?
