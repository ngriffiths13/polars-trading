"""Type aliases for Polars DataFrame and Series objects."""

import sys
from typing import TypeVar

import polars as pl

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing import TypeAlias
from polars._typing import IntoExpr, PolarsDataType
from polars.datatypes import DataType, DataTypeClass

DataFrameTypes: TypeAlias = pl.DataFrame | pl.LazyFrame

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

__all__ = [
    "DataFrameTypes",
    "DataType",
    "DataTypeClass",
    "FrameType",
    "IntoExpr",
    "PolarsDataType",
]
