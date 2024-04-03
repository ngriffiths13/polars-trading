import polars as pl
from polars.type_aliases import IntoExpr, FrameType
from polars_finance.utils import parse_into_expr
from polars.utils.udfs import _get_shared_lib_location


lib = _get_shared_lib_location(__file__)


def raw_forward_returns(prices: IntoExpr, n_bars: int = 1):
    price_expr = parse_into_expr(prices)
    return price_expr.shift(-n_bars) / price_expr - 1


def fixed_time_label(
    price_series: IntoExpr,
    upper_threshold: float = 0.01,
    lower_threshold: float = -0.1,
    t: int = 1,
    symbol_col: str = "symbol",
):
    return_expr = parse_into_expr(price_series)
    return_expr = return_expr.shift(-t).over(symbol_col) / return_expr - 1
    return (
        pl.when(return_expr > upper_threshold)
        .then(1)
        .when(return_expr < lower_threshold)
        .then(-1)
        .otherwise(0)
    )


def fixed_time_dynamic_threshold_label(
    price_series: IntoExpr,
    span: int = 100,
    upper_multiplier: float = 1.0,
    lower_multiplier: float = 1.0,
    t: int = 1,
    symbol_col: str = "symbol",
):
    price_expr = parse_into_expr(price_series)
    return_expr = price_expr / price_expr.shift(-t).over(symbol_col) - 1
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
    raise NotImplementedError("This function is not yet implemented.")


# TODO: This needs to return a df
# TODO: Write rust funtion for variable shifts
def triple_barrier_label(
    df: FrameType,
    price_series: IntoExpr,
    horizontal_width: IntoExpr,
    pt: float,
    sl: float,
    vertical_barrier: IntoExpr = 5,
    min_return: float = 0.0,
    use_vertical_barrier_sign: bool = True,
    seed_indicator: IntoExpr | None = None,
):
    if seed_indicator is None:
        seed_indicator = pl.lit(True)
    else:
        seed_indicator = parse_into_expr(seed_indicator).cast(pl.Boolean)
    price_expr = parse_into_expr(price_series)
    horizontal_width_expr = parse_into_expr(horizontal_width)
    vertical_barrier_expr = parse_into_expr(vertical_barrier)
    labels = df.with_columns(
        price_expr.register_plugin(
            lib,
            "triple_barrier_label",
            kwargs={
                "pt": pt,
                "sl": sl,
                "min_return": min_return,
                "use_vertical_barrier_sign": use_vertical_barrier_sign,
            },
            args=[horizontal_width_expr, vertical_barrier_expr, seed_indicator],
            cast_to_supertypes=True,
        )
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
