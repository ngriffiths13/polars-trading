"""Type aliases for Polars DataFrame and Series objects."""

import sys
from typing import TypeVar, Union

import polars as pl

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias
from polars._typing import IntoExpr as IntoExpr
from polars.datatypes import DataType, DataTypeClass

PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]
DataFrameTypes: TypeAlias = Union[pl.DataFrame, pl.LazyFrame]

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)
