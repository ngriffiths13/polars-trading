import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB


def noop(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="noop",
        is_elementwise=True,
    )
