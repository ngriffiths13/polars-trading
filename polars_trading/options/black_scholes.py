import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB


def black_scholes(s: IntoExprColumn, k: IntoExprColumn, t: IntoExprColumn, sigma: IntoExprColumn, r: IntoExprColumn, type_: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[s, k, t, sigma, r, type_],
        plugin_path=LIB,
        function_name="black_scholes",
        is_elementwise=True,
    )
