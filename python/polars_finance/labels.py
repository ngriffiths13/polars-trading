from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function
from polars.utils.udfs import _get_shared_lib_location

from polars_finance.utils import parse_into_expr

if TYPE_CHECKING:
    from polars.type_aliases import FrameType, IntoExpr

lib = _get_shared_lib_location(__file__)


def raw_forward_returns_expr(prices: IntoExpr, n_bars: int = 1) -> pl.Expr:
    price_expr = parse_into_expr(prices)
    return price_expr.shift(-n_bars) / price_expr - 1


def fixed_time_label_expr(
    price_series: IntoExpr,
    upper_threshold: float = 0.01,
    lower_threshold: float = -0.1,
    t: int = 1,
    symbol_col: str = "symbol",
) -> pl.Expr:
    return_expr = parse_into_expr(price_series)
    return_expr = return_expr.shift(-t).over(symbol_col) / return_expr - 1
    return (
        pl.when(return_expr > upper_threshold)
        .then(1)
        .when(return_expr < lower_threshold)
        .then(-1)
        .otherwise(0)
    )


def fixed_time_dynamic_threshold_label_expr(
    price_series: IntoExpr,
    span: int = 100,
    upper_multiplier: float = 1.0,
    lower_multiplier: float = 1.0,
    t: int = 1,
    symbol_col: str = "symbol",
) -> pl.Expr:
    price_expr = parse_into_expr(price_series)
    return_expr = price_expr.shift(-t).over(symbol_col) / price_expr - 1
    rolling_std = (
        (price_expr / price_expr.shift(t) - 1).ewm_std(span=span).over(symbol_col)
    )
    return (
        pl.when(return_expr > rolling_std * upper_multiplier)
        .then(1)
        .when(return_expr < rolling_std * -lower_multiplier)
        .then(-1)
        .otherwise(0)
    )


# TODO: Implement this function
def get_vertical_barrier(
    df: FrameType, date_col: str, barrier_size: str, symbol_col: str = "symbol"
) -> FrameType:
    raise NotImplementedError("This function is not yet implemented.")  # noqa: EM101


def triple_barrier_label(
    df: FrameType,
    price_series: IntoExpr,
    horizontal_width: IntoExpr,
    profit_taker: float | None = 1.0,
    stop_loss: float | None = 1.0,
    vertical_barrier: IntoExpr = 5,
    min_return: float = 0.0,
    *,
    use_vertical_barrier_sign: bool = True,
    seed_indicator: IntoExpr | None = None,
) -> FrameType:
    if seed_indicator is None:
        seed_indicator = pl.lit(value=True)
    else:
        seed_indicator = parse_into_expr(seed_indicator).cast(pl.Boolean)
    price_expr = parse_into_expr(price_series)
    horizontal_width_expr = parse_into_expr(horizontal_width)
    vertical_barrier_expr = parse_into_expr(vertical_barrier, dtype=pl.Int64)
    label_df = df.select(
        price_expr.alias("price"),
        horizontal_width_expr.alias("horizontal_width"),
        vertical_barrier_expr.alias("vertical_barrier"),
        seed_indicator.alias("seed_indicator"),
    )

    labels = label_df.with_columns(
        register_plugin_function(
            plugin_path=lib,
            function_name="triple_barrier_label",
            args=[
                pl.col("price"),
                pl.col("horizontal_width"),
                pl.col("vertical_barrier"),
                pl.col("seed_indicator"),
            ],
            cast_to_supertype=False,
            kwargs={
                "profit_taker": profit_taker,
                "stop_loss": stop_loss,
                "min_return": min_return,
                "use_vertical_barrier_sign": use_vertical_barrier_sign,
            },
        ).alias("triple_barrier_label")
    )
    return labels


def metalabel(): ...


def label_uniqueness(): ...


def avg_label_uniqueness(): ...


def sequential_bootstrap(): ...


# TODO: MC Experiment on 4.5.4


def return_attribution_weighting(): ...


def time_decay_weighting(): ...


def class_sample_weighting(): ...
