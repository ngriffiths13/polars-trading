import pandas as pd
import polars as pl
import pytest
from datetime import datetime
from polars.testing import assert_frame_equal

from polars_trading.bars import time_bars, tick_bars
from mimesis import Fieldset
from mimesis.locales import Locale
from functools import lru_cache


@lru_cache
def generate_trade_data(n_rows: int) -> pl.DataFrame:
    fs = Fieldset(locale=Locale.EN, i=n_rows)

    return pl.DataFrame(
        {
            "ts_event": fs("datetime"),
            "price": fs("finance.price", minimum=1, maximum=100),
            "size": fs("numeric.integer_number", start=10_000, end=100_000),
            "symbol": fs("choice.choice", items=["AAPL", "GOOGL", "MSFT"]),
        }
    )


@pytest.fixture
def trade_data(request):
    return generate_trade_data(request.param)


def pandas_time_bars(df: pd.DataFrame, period: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df["ts_event"])
    df["pvt"] = df["price"] * df["size"]
    resampled_df = (
        df.groupby([df.index.to_period(period), "symbol"])
        .agg({"price": "ohlc", "pvt": "sum", "size": "sum", "ts_event": "count"})
        .reset_index("symbol")
    )
    resampled_df.index.names = ["ts_event"]
    resampled_df.columns = [
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "n_trades",
    ]
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


def pandas_tick_bars(df: pd.DataFrame, n_ticks: int) -> pd.DataFrame:
    df["pvt"] = df["price"] * df["size"]
    df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    df["tick_group"] = (df.groupby(["symbol", "date"]).cumcount() // n_ticks).astype(int)
    resampled_df = (
        df.groupby(["date", "tick_group", "symbol"])
        .agg({"price": "ohlc", "pvt": "sum", "size": "sum", "ts_event": "count"})
        .reset_index(["symbol", "date"])
    )
    resampled_df.index.names = ["ts_event"]
    resampled_df.columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "vwap",
        "volume",
        "n_trades",
    ]
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


@pytest.mark.parametrize("trade_data", [1_000], indirect=True)
def test__time_bars__matches_pandas(trade_data):
    pd_df = pandas_time_bars(trade_data.to_pandas(), "1 min")
    pd_df.index = pd_df.index.to_timestamp()
    pd_df = pd_df.reset_index()
    res = time_bars(trade_data, timestamp_col="ts_event", bar_size="1m").drop(
        "ts_event_start", "ts_event_end"
    )

    assert_frame_equal(
        res, pl.from_pandas(pd_df, schema_overrides=res.schema), check_row_order=False
    )


@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize("trade_data", [100, 1_000], indirect=True)
def test__time_bars__polars_benchmark(benchmark, trade_data):
    benchmark(time_bars, trade_data, timestamp_col="ts_event", bar_size="1d")


@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize("trade_data", [100, 1_000], indirect=True)
def test__time_bars__pandas_benchmark(benchmark, trade_data):
    trade_data = trade_data.to_pandas()
    benchmark(pandas_time_bars, trade_data, "1d")


@pytest.mark.parametrize("trade_data", [1_000], indirect=True)
def test__tick_bars__matches_pandas(trade_data):
    pd_df = pandas_tick_bars(trade_data.to_pandas(), 100)
    # pd_df.index = pd_df.index.to_timestamp()
    pd_df = pd_df.reset_index()
    res = tick_bars(trade_data, timestamp_col="ts_event", bar_size=100).drop(
        "ts_event_start", "ts_event_end"
    )

    assert_frame_equal(
        res,
        pl.from_pandas(pd_df, schema_overrides=res.schema).drop("ts_event"),
        check_row_order=False,
    )
