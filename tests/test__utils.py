"""Utility functions for library."""
from polars_trading._utils import validate_columns
import pytest
import polars as pl


def test__validate_columns__valid():
    @validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
    def dummy_func(df, **kwargs):
        return df

    df = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "p": [1.0, 2.0, 3.0],
            "s": [1, 2, 3],
            "sy": ["A", "B", "C"],
        }
    )

    dummy_func(df, timestamp_col="ts", price_col="p", size_col="s", symbol_col="sy")


def test__validate_columns__invalid():
    @validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
    def dummy_func(df, **kwargs):
        return df

    df = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "p": [1.0, 2.0, 3.0],
            "s": [1, 2, 3],
            "sy": ["A", "B", "C"],
        }
    )

    with pytest.raises(ValueError):
        dummy_func(
            df, timestamp_col="ts", price_col="pr", size_col="s", symbol_col="sy"
        )


def test__validate_columns__df_not_passed():
    @validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
    def dummy_func(df, **kwargs):
        return df

    df = {
        "ts": [1, 2, 3],
        "p": [1.0, 2.0, 3.0],
        "s": [1, 2, 3],
        "sy": ["A", "B", "C"],
    }

    with pytest.raises(TypeError):
        dummy_func(df, timestamp_col="ts", price_col="p", size_col="s", symbol_col="sy")


def test__validate_columns__default_args():
    @validate_columns("timestamp_col", "price_col", "size_col", "symbol_col")
    def dummy_func(df, timestamp_col="ts", **kwargs):
        return df

    df = pl.DataFrame(
        {
            "ts": [1, 2, 3],
            "p": [1.0, 2.0, 3.0],
            "s": [1, 2, 3],
            "sy": ["A", "B", "C"],
        }
    )
    dummy_func(df, price_col="p", size_col="s", symbol_col="sy")
