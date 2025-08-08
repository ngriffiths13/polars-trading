"""Pandas benchmark tests for bars logic.

This module contains benchmark tests that compare Pandas implementations
of bar aggregation functions against the main Polars implementations.
"""

import pandas as pd
import pytest

from tests.testing_utils.pd_bars_helpers import (
    pandas_dollar_bars,
    pandas_tick_bars,
    pandas_time_bars,
    pandas_volume_bars,
)


@pytest.mark.pandas
@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__time_bars__pandas_benchmark(benchmark, trade_data):
    """Benchmarks the pandas implementation of time_bars function."""
    pd_df = trade_data.to_pandas()

    def pandas_time_bars_wrapper(df: pd.DataFrame) -> pd.DataFrame:
        return pandas_time_bars(df, "1d")

    benchmark(pandas_time_bars_wrapper, pd_df)


@pytest.mark.pandas
@pytest.mark.benchmark(group="tick_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__tick_bars__pandas_benchmark(benchmark, trade_data):
    """Benchmarks the pandas implementation of tick_bars function."""
    pd_df = trade_data.to_pandas()

    def pandas_tick_bars_wrapper(df: pd.DataFrame) -> pd.DataFrame:
        return pandas_tick_bars(df, 100)

    benchmark(pandas_tick_bars_wrapper, pd_df)


@pytest.mark.pandas
@pytest.mark.benchmark(group="volume_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__volume_bars__pandas_benchmark(benchmark, trade_data):
    """Benchmarks the pandas implementation of volume_bars function."""
    pd_df = trade_data.to_pandas()

    def pandas_volume_bars_wrapper(df: pd.DataFrame) -> pd.DataFrame:
        return pandas_volume_bars(df, 10_000)

    benchmark(pandas_volume_bars_wrapper, pd_df)


@pytest.mark.pandas
@pytest.mark.benchmark(group="dollar_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__dollar_bars__pandas_benchmark(benchmark, trade_data):
    """Benchmarks the pandas implementation of dollar_bars function."""
    pd_df = trade_data.to_pandas()

    def pandas_dollar_bars_wrapper(df: pd.DataFrame) -> pd.DataFrame:
        return pandas_dollar_bars(df, 1_000_000.0)

    benchmark(pandas_dollar_bars_wrapper, pd_df)
