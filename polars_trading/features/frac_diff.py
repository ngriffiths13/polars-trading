"""Module containing functions to generate fractionally differentiated features."""

import polars as pl
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB
from polars_trading.typing import IntoExpr


def frac_diff(expr: IntoExpr, d: float, threshold: float) -> pl.Expr:
    """Generate expression to calculate the fractionally differentiated series.

    Args:
    ----
        expr: IntoExpr - The expression to calculate the fractionally differentiated
            series.
        d: float - The fractional difference.
        threshold: float - The threshold.

    Returns:
    -------
        pl.Expr: The expression to calculate the fractionally differentiated series.

    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr],
        kwargs={"d": d, "threshold": threshold},
        is_elementwise=False,
        function_name="frac_diff",
    )
