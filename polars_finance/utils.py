from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from polars.type_aliases import FrameType
from datetime import timedelta
from functools import reduce

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr, PolarsDataType


def parse_into_expr(
    expr: IntoExpr,
    *,
    str_as_lit: bool = False,
    list_as_lit: bool = True,
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """
    Parse a single input into an expression.

    Parameters
    ----------
    expr
        The input to be parsed as an expression.
    str_as_lit
        Interpret string input as a string literal. If set to `False` (default),
        strings are parsed as column names.
    list_as_lit
        Interpret list input as a lit literal, If set to `False`,
        lists are parsed as `Series` literals.
    dtype
        If the input is expected to resolve to a literal with a known dtype, pass
        this to the `lit` constructor.

    Returns
    -------
    polars.Expr
    """
    if isinstance(expr, pl.Expr):
        pass
    elif isinstance(expr, str) and not str_as_lit:
        expr = pl.col(expr)
    elif isinstance(expr, list) and not list_as_lit:
        expr = pl.lit(pl.Series(expr), dtype=dtype)
    else:
        expr = pl.lit(expr, dtype=dtype)

    return expr


def dynamic_shift(
    df: FrameType, value_col: str, shift_col: str, group_col: str | None = None
) -> FrameType:
    """
    Shift a column in a DataFrame by a dynamic amount.

    Parameters
    ----------
    df
        The DataFrame/LazyFrame to shift.
    shift_col
        The column to use for shifting.
    group_col
        The column to group by when shifting.

    Returns
    -------
    FrameType
    """
    df_ind = df.with_row_index()
    if isinstance(df, pl.DataFrame):
        shift_values = df[shift_col].unique().to_list()
    elif isinstance(df, pl.LazyFrame):
        shift_values = (
            df.select(pl.col(shift_col).unique()).collect()[shift_col].to_list()
        )
    else:
        raise ValueError("df must be a DataFrame or LazyFrame")

    shifted_dfs = []
    for shift_val in shift_values:
        if group_col is not None:
            shifted_dfs.append(
                df_ind.select(
                    "index",
                    pl.col(value_col)
                    .shift(shift_val)
                    .over(group_col)
                    .alias(f"{value_col}_shifted"),
                    pl.lit(shift_val).alias(shift_col).cast(pl.Int64),
                )
            )
        else:
            shifted_dfs.append(
                df_ind.select(
                    "index",
                    pl.col(value_col).shift(shift_val).alias(f"{value_col}_shifted"),
                    pl.lit(shift_val).alias(shift_col).cast(pl.Int64),
                )
            )
    return df_ind.join(
        pl.concat(shifted_dfs), on=["index", shift_col], how="left"
    ).drop("index")
