import polars as pl
from polars_finance.utils import parse_into_expr
from polars.type_aliases import IntoExpr


def balance_of_power(
    high: IntoExpr, low: IntoExpr, close: IntoExpr, open: IntoExpr
) -> pl.Expr:
    return (parse_into_expr(close) - parse_into_expr(open)) / (
        parse_into_expr(high) - parse_into_expr(low)
    )
