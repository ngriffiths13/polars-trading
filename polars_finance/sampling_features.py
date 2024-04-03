import polars as pl
from polars.utils.udfs import _get_shared_lib_location

from polars_finance.utils import parse_into_expr

from polars.type_aliases import IntoExpr

lib = _get_shared_lib_location(__file__)


def symmetric_cusum_filter(time_series: IntoExpr, threshold: float) -> pl.Expr:
    expr = parse_into_expr(time_series)
    return expr.register_plugin(
        lib=lib,
        symbol="symmetric_cusum_filter",
        kwargs={"threshold": threshold},
        is_elementwise=False,
        cast_to_supertypes=True,
    )
