from polars_trading.labels.dynamic_labels import (
    daily_vol,
    get_vertical_barrier_by_timedelta,
)
import polars as pl
from polars_trading._testing.labels import get_daily_vol
import pytest
from polars.testing import assert_frame_equal
from datetime import datetime, timedelta


@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 1},
    ],
    indirect=True,
)
def test__daily_vol__single_security(trade_data):
    pl_result = daily_vol(trade_data.lazy(), "ts_event", "price", None, 5).collect()
    pd_result = get_daily_vol(trade_data.to_pandas().set_index("ts_event")["price"], 5)
    pd_result = pl.from_pandas(pd_result.reset_index()).rename(
        {"price": "daily_return_volatility"}
    )
    assert_frame_equal(pl_result.drop_nulls(), pd_result)


@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__daily_vol__multi_security(trade_data):
    pl_result = (
        daily_vol(trade_data, "ts_event", "price", "symbol", 5).sort(
            "ts_event", "symbol"
        )
    ).drop_nulls()
    pd_result = (
        trade_data.to_pandas()
        .set_index("ts_event")[["symbol", "price"]]
        .groupby("symbol")["price"]
        .apply(get_daily_vol, 5)
    )
    pd_result = (
        pl.from_pandas(pd_result.reset_index())
        .rename({"price": "daily_return_volatility"})
        .sort("ts_event", "symbol")
    )
    assert_frame_equal(
        pl_result, pd_result, check_row_order=False, check_column_order=False
    )


@pytest.mark.benchmark(group="daily_vol")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 100_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__daily_vol__polars_benchmark(benchmark, trade_data):
    benchmark(daily_vol(trade_data.lazy(), "ts_event", "price", "symbol", 100).collect)


@pytest.mark.pandas
@pytest.mark.benchmark(group="daily_vol")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 100_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__daily_vol__pandas_benchmark(benchmark, trade_data):
    pd_df = trade_data.to_pandas().set_index("ts_event")[["symbol", "price"]]

    def get_daily_vol_pd(pd_df):
        return pd_df.groupby("symbol")["price"].apply(get_daily_vol).reset_index()

    benchmark(get_daily_vol_pd, pd_df)


def test__daily_vol__weekend_returns():
    df = pl.DataFrame(
        {
            "ts_event": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 3, 0, 0),
                datetime(2024, 1, 1, 6, 0, 0),
                datetime(2024, 1, 1, 9, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 1, 20, 0, 0),
                datetime(2024, 1, 2, 1, 0, 0),
                datetime(2024, 1, 3, 5, 0, 0),
                datetime(2024, 1, 3, 7, 0, 0),
                datetime(2024, 1, 3, 9, 0, 0),
            ],
            "price": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    expected = pl.from_repr("""
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────────┐
    │ ts_event            ┆ daily_return_volatility │
    │ ---                 ┆ ---                     │
    │ datetime[μs]        ┆ f64                     │
    ╞═════════════════════╪═════════════════════════╡
    │ 2024-01-01 00:00:00 ┆ null                    │
    │ 2024-01-01 03:00:00 ┆ null                    │
    │ 2024-01-01 06:00:00 ┆ null                    │
    │ 2024-01-01 09:00:00 ┆ null                    │
    │ 2024-01-01 15:00:00 ┆ null                    │
    │ 2024-01-01 20:00:00 ┆ null                    │
    │ 2024-01-02 01:00:00 ┆ 0.0                     │
    │ 2024-01-03 05:00:00 ┆ 4.141625                │
    │ 2024-01-03 07:00:00 ┆ 2.668519                │
    │ 2024-01-03 09:00:00 ┆ 1.792192                │
    └─────────────────────┴─────────────────────────┘
    """)

    result = daily_vol(df.lazy(), "ts_event", "price", None, 3).collect()
    assert_frame_equal(result, expected)


def test__get_vertical_barrier_by_timedelta__simple():
    df = pl.DataFrame(
        {
            "ts_event": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 3, 0, 0),
                datetime(2024, 1, 1, 6, 0, 0),
                datetime(2024, 1, 1, 9, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 1, 20, 0, 0),
                datetime(2024, 1, 2, 1, 0, 0),
                datetime(2024, 1, 3, 5, 0, 0),
                datetime(2024, 1, 3, 7, 0, 0),
                datetime(2024, 1, 3, 9, 0, 0),
            ],
        }
    )

    expected = pl.from_repr("""
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────┐
    │ ts_event            ┆ vertical_barrier    │
    │ ---                 ┆ ---                 │
    │ datetime[μs]        ┆ datetime[μs]        │
    ╞
    │ 2024-01-01 00:00:00 ┆ 2024-01-01 03:00:00 │
    │ 2024-01-01 03:00:00 ┆ 2024-01-01 06:00:00 │
    │ 2024-01-01 06:00:00 ┆ 2024-01-01 09:00:00 │
    │ 2024-01-01 09:00:00 ┆ 2024-01-01 15:00:00 │
    │ 2024-01-01 15:00:00 ┆ 2024-01-01 20:00:00 │
    │ 2024-01-01 20:00:00 ┆ 2024-01-02 01:00:00 │
    │ 2024-01-02 01:00:00 ┆ 2024-01-03 05:00:00 │
    │ 2024-01-03 05:00:00 ┆ 2024-01-03 07:00:00 │
    │ 2024-01-03 07:00:00 ┆ 2024-01-03 09:00:00 │
    │ 2024-01-03 09:00:00 ┆ null                │
    └─────────────────────┴─────────────────────┘
    """)

    result = get_vertical_barrier_by_timedelta(df.lazy(), "ts_event", "2h").collect()
    assert_frame_equal(result, expected)


def test__get_vertical_barrier_by_timedelta__skip_rows():
    df = pl.DataFrame(
        {
            "ts_event": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 3, 0, 0),
                datetime(2024, 1, 1, 6, 0, 0),
                datetime(2024, 1, 1, 9, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 1, 20, 0, 0),
                datetime(2024, 1, 2, 1, 0, 0),
                datetime(2024, 1, 3, 5, 0, 0),
                datetime(2024, 1, 3, 7, 0, 0),
                datetime(2024, 1, 3, 9, 0, 0),
            ],
        }
    )

    expected = pl.from_repr("""
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────┐
    │ ts_event            ┆ vertical_barrier    │
    │ ---                 ┆ ---                 │
    │ datetime[μs]        ┆ datetime[μs]        │
    ╞
    │ 2024-01-01 00:00:00 ┆ 2024-01-01 03:00:00 │
    │ 2024-01-01 03:00:00 ┆ 2024-01-01 06:00:00 │
    │ 2024-01-01 06:00:00 ┆ 2024-01-01 09:00:00 │
    │ 2024-01-01 09:00:00 ┆ 2024-01-01 15:00:00 │
    │ 2024-01-01 15:00:00 ┆ 2024-01-01 20:00:00 │
    │ 2024-01-01 20:00:00 ┆ 2024-01-02 01:00:00 │
    │ 2024-01-02 01:00:00 ┆ 2024-01-03 05:00:00 │
    │ 2024-01-03 05:00:00 ┆ 2024-01-03 09:00:00 │
    │ 2024-01-03 07:00:00 ┆ null                │
    │ 2024-01-03 09:00:00 ┆ null                │
    └─────────────────────┴─────────────────────┘
    """)

    result = get_vertical_barrier_by_timedelta(df.lazy(), "ts_event", "3h").collect()
    assert_frame_equal(result, expected)


def test__get_vertical_barrier_by_timedelta__timedelta():
    df = pl.DataFrame(
        {
            "ts_event": [
                datetime(2024, 1, 1, 0, 0, 0),
                datetime(2024, 1, 1, 3, 0, 0),
                datetime(2024, 1, 1, 6, 0, 0),
                datetime(2024, 1, 1, 9, 0, 0),
                datetime(2024, 1, 1, 15, 0, 0),
                datetime(2024, 1, 1, 20, 0, 0),
                datetime(2024, 1, 2, 1, 0, 0),
                datetime(2024, 1, 3, 5, 0, 0),
                datetime(2024, 1, 3, 7, 0, 0),
                datetime(2024, 1, 3, 9, 0, 0),
            ],
        }
    )

    expected = pl.from_repr("""
    shape: (10, 2)
    ┌─────────────────────┬─────────────────────┐
    │ ts_event            ┆ vertical_barrier    │
    │ ---                 ┆ ---                 │
    │ datetime[μs]        ┆ datetime[μs]        │
    ╞
    │ 2024-01-01 00:00:00 ┆ 2024-01-01 03:00:00 │
    │ 2024-01-01 03:00:00 ┆ 2024-01-01 06:00:00 │
    │ 2024-01-01 06:00:00 ┆ 2024-01-01 09:00:00 │
    │ 2024-01-01 09:00:00 ┆ 2024-01-01 15:00:00 │
    │ 2024-01-01 15:00:00 ┆ 2024-01-01 20:00:00 │
    │ 2024-01-01 20:00:00 ┆ 2024-01-02 01:00:00 │
    │ 2024-01-02 01:00:00 ┆ 2024-01-03 05:00:00 │
    │ 2024-01-03 05:00:00 ┆ 2024-01-03 07:00:00 │
    │ 2024-01-03 07:00:00 ┆ 2024-01-03 09:00:00 │
    │ 2024-01-03 09:00:00 ┆ null                │
    └─────────────────────┴─────────────────────┘
    """)

    result = get_vertical_barrier_by_timedelta(
        df.lazy(), "ts_event", timedelta(hours=2)
    ).collect()
    assert_frame_equal(result, expected)


@pytest.mark.benchmark(group="vertical_barrier")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 100_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__get_vertical_barrier_by_timedelta__benchmark(benchmark, trade_data):
    benchmark(get_vertical_barrier_by_timedelta, trade_data, "ts_event", "1h", "symbol")
