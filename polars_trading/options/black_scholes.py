import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from polars_trading._utils import LIB


def black_scholes(
    s: IntoExprColumn,
    k: IntoExprColumn,
    t: IntoExprColumn,
    sigma: IntoExprColumn,
    r: IntoExprColumn,
    type_: IntoExprColumn,
) -> pl.Expr:
    """Calculate the Black-Scholes option price for calls and puts.

    Args:
    ----
        s: IntoExprColumn
            The underlying asset price.
        k: IntoExprColumn
            The strike price of the option.
        t: IntoExprColumn
            The time to expiry in years.
        sigma: IntoExprColumn
            The implied volatility of the option (as a decimal).
        r: IntoExprColumn
            The risk-free interest rate (as a decimal).
        type_: IntoExprColumn
            The option type ("call" or "put").

    Returns:
    -------
        pl.Expr
            An expression representing the option price calculated using the
            Black-Scholes formula.
    """
    return register_plugin_function(
        args=[s, k, t, sigma, r, type_],
        plugin_path=LIB,
        function_name="black_scholes",
        is_elementwise=True,
    )
