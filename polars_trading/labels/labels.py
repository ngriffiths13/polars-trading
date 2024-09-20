"""This module contains functions to calculate labels for financial data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_trading._utils import parse_into_expr

if TYPE_CHECKING:
    from polars_trading.typing import IntoExpr


def _classify_by_threshold(values: pl.Expr, threshold: IntoExpr | None) -> pl.Expr:
    if threshold is None:
        return values.sign().cast(pl.Int32)
    threshold = parse_into_expr(threshold)
    return (
        pl.when(values > threshold.abs())
        .then(1)
        .when(values < -threshold.abs())
        .then(-1)
        .when(values.is_between(-threshold.abs(), threshold.abs()))
        .then(0)
        .otherwise(None)
    ).cast(pl.Int32)


def fixed_time_return_classification(
    prices: IntoExpr,
    window: int,
    threshold: IntoExpr | None = None,
    offset: int = 1,
    symbol: IntoExpr | None = None,
) -> pl.Expr:
    """Calculate the fixed time return as a classification label.

    This function calculates the fixed time return of a financial instrument. The
    return for time t is calculated as the percent change in the price at time
    t+{offset} to the price at time t+{offset + window}. The label is set to 1 if
    the return is greater than the threshold, -1 if the return is less than the
    negative threshold, and 0 otherwise.

    Args:
    ----
        prices: IntoExpr - The prices of the financial instrument.
        window: int - The number of periods to look forward.
        threshold: IntoExpr | None - The threshold to classify the return. If None,
            the return is classified as the sign of the return.
        offset: int - The offset of the starting period. Defaults to 1. This is so by
            default we attempt to avoid lookahead bias. At time t, you ideally make a
            decision to capture the return from t+1 to t+1+window. This way you do not
            include the current price in your calculation that you likely can't execute
            at.
        symbol: IntoExpr | None - The symbol of the financial instrument. This is used
            to calculate returns for multiple symbols at once. If None, it is assumed
            that the prices are for a single symbol.

    Returns:
    -------
        pl.Expr: The fixed time return classification label as an expression.
    """
    return_expr = fixed_time_return(prices, window, offset)
    if symbol is not None:
        return_expr = return_expr.over(parse_into_expr(symbol))
    return _classify_by_threshold(return_expr, threshold)


def fixed_time_return(
    prices: IntoExpr, window: int, offset: int = 1, symbol: IntoExpr | None = None
) -> pl.Expr:
    """Calculate the fixed time return.

    This function calculates the fixed time return of a financial instrument. The
    return for time t is calculated as the percent change in the price at time
    t+{offset} to the price at time t+{offset + window}.

    Args:
    ----
        prices: IntoExpr - The prices of the financial instrument.
        window: int - The number of periods to look forward.
        offset: int - The offset of the starting period. Defaults to 1. This is so by
            default we attempt to avoid lookahead bias. At time t, you ideally make a
            decision to capture the return from t+1 to t+1+window. This way you do not
            include the current price in your calculation that you likely can't execute
            at.
        symbol: IntoExpr | None - The symbol of the financial instrument. This is used
            to calculate returns for multiple symbols at once. If None, it is assumed
            that the prices are for a single symbol.

    Returns:
    -------
        pl.Expr: The fixed time return as an expression.
    """
    return_expr = (
        parse_into_expr(prices)
        .shift(-offset - window)
        .truediv(parse_into_expr(prices).shift(-offset))
        .sub(1)
    )
    if symbol is not None:
        return_expr = return_expr.over(parse_into_expr(symbol))
    return return_expr
