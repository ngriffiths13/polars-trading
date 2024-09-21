import pandas as pd
import polars as pl
import pytest
from datetime import datetime
from polars.testing import assert_frame_equal

from polars_trading.bars import time_bars, tick_bars, volume_bars, dollar_bars
from polars_trading._testing.data import generate_trade_data


def pandas_time_bars(df: pd.DataFrame, period: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df["ts_event"])
    df["pvt"] = df["price"] * df["size"]
    df = df.sort_index()
    resampled_df = (
        df.groupby([df.index.to_period(period), "symbol"])
        .agg(
            ts_event_start=pd.NamedAgg(column="ts_event", aggfunc="first"),
            ts_event_end=pd.NamedAgg(column="ts_event", aggfunc="last"),
            n_trades=pd.NamedAgg(column="ts_event", aggfunc="count"),
            open=pd.NamedAgg(column="price", aggfunc="first"),
            high=pd.NamedAgg(column="price", aggfunc="max"),
            low=pd.NamedAgg(column="price", aggfunc="min"),
            close=pd.NamedAgg(column="price", aggfunc="last"),
            volume=pd.NamedAgg(column="size", aggfunc="sum"),
            vwap=pd.NamedAgg(column="pvt", aggfunc="sum"),
        )
        .reset_index("symbol")
    )
    resampled_df.index.names = ["ts_event"]
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


def pandas_tick_bars(df: pd.DataFrame, n_ticks: int) -> pd.DataFrame:
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df = df.sort_values("ts_event")
    df["pvt"] = df["price"] * df["size"]
    df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    df["tick_group"] = (df.groupby(["symbol", "date"]).cumcount() // n_ticks).astype(
        int
    )
    resampled_df = (
        df.groupby(["date", "tick_group", "symbol"])
        .agg(
            ts_event_start=pd.NamedAgg(column="ts_event", aggfunc="first"),
            ts_event_end=pd.NamedAgg(column="ts_event", aggfunc="last"),
            n_trades=pd.NamedAgg(column="ts_event", aggfunc="count"),
            open=pd.NamedAgg(column="price", aggfunc="first"),
            high=pd.NamedAgg(column="price", aggfunc="max"),
            low=pd.NamedAgg(column="price", aggfunc="min"),
            close=pd.NamedAgg(column="price", aggfunc="last"),
            volume=pd.NamedAgg(column="size", aggfunc="sum"),
            vwap=pd.NamedAgg(column="pvt", aggfunc="sum"),
        )
        .reset_index()
        .drop(["date", "tick_group"], axis=1)
    )
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__time_bars__matches_pandas(trade_data):
    pd_df = pandas_time_bars(trade_data.to_pandas(), "1d")
    pd_df.index = pd_df.index.to_timestamp()
    pd_df = pd_df.reset_index()
    res = time_bars(trade_data, timestamp_col="ts_event", bar_size="1d")

    assert_frame_equal(
        res,
        pl.from_pandas(pd_df, schema_overrides=res.schema),
        check_row_order=False,
        check_column_order=False,
    )


@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__time_bars__polars_benchmark(benchmark, trade_data):
    benchmark(time_bars, trade_data, timestamp_col="ts_event", bar_size="1d")


@pytest.mark.pandas
@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__time_bars__pandas_benchmark(benchmark, trade_data):
    trade_data = trade_data.to_pandas()
    benchmark(pandas_time_bars, trade_data, "1d")


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__tick_bars__matches_pandas(trade_data):
    pd_df = pandas_tick_bars(trade_data.to_pandas(), 100)
    res = tick_bars(trade_data, timestamp_col="ts_event", bar_size=100)

    assert_frame_equal(
        res,
        pl.from_pandas(pd_df, schema_overrides=res.schema),
        check_row_order=False,
        check_column_order=False,
    )


@pytest.mark.benchmark(group="tick_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__tick_bars__polars_benchmark(benchmark, trade_data):
    benchmark(tick_bars, trade_data, timestamp_col="ts_event", bar_size=100)


@pytest.mark.pandas
@pytest.mark.benchmark(group="tick_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__tick_bars__pandas_benchmark(benchmark, trade_data):
    trade_data = trade_data.to_pandas()
    benchmark(pandas_tick_bars, trade_data, 100)


@pytest.mark.benchmark(group="volume_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__volume_bars__polars_benchmark(benchmark, trade_data):
    benchmark(volume_bars, trade_data, timestamp_col="ts_event", bar_size=10_000)


@pytest.mark.benchmark(group="dollar_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__dollar_bars__polars_benchmark(benchmark, trade_data):
    benchmark(dollar_bars, trade_data, timestamp_col="ts_event", bar_size=100_000)
