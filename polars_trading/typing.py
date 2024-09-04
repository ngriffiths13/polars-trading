"""Type aliases for Polars DataFrame and Series objects."""

from typing import TYPE_CHECKING, TypeVar, Union

import polars as pl

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    from polars.datatypes import DataType, DataTypeClass

    IntoExpr: TypeAlias = Union[pl.Expr, str, pl.Series]
    PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]
    DataFrameTypes: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)
