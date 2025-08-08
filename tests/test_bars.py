import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_trading.bars import dollar_bars, tick_bars, time_bars, volume_bars
from polars_trading.config import Config
from tests.testing_utils.pd_bars_helpers import (
    pandas_tick_bars,
    pandas_time_bars,
    pandas_volume_bars,
)


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__time_bars__matches_pandas(trade_data):
    pd_df = pandas_time_bars(trade_data.to_pandas(), "1d")
    pd_df.index = pd_df.index.to_timestamp()
    pd_df = pd_df.reset_index()

    # Configure the column names to match test data
    with Config(timestamp_column="ts_event"):
        res = time_bars(trade_data, bar_size="1d")

    assert_frame_equal(
        res,
        pl.from_pandas(pd_df, schema_overrides=res.schema),
        check_row_order=False,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__tick_bars__matches_pandas(trade_data):
    pd_df = pandas_tick_bars(trade_data.to_pandas(), 100)

    # Configure the column names to match test data
    with Config(timestamp_column="ts_event"):
        res = tick_bars(trade_data, bar_size=100)

    assert_frame_equal(
        res,
        pl.from_pandas(pd_df, schema_overrides=res.schema),
        check_row_order=False,
        check_column_order=False,
    )


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 100, "n_companies": 3}], indirect=True
)
def test__volume_bars__matches_pandas(trade_data):
    """Test that polars volume_bars matches pandas implementation for small datasets."""
    pd_df = pandas_volume_bars(
        trade_data.to_pandas(), volume_threshold=50_000, split_by_date=True
    )

    # Configure the column names to match test data
    with Config(timestamp_column="ts_event"):
        res = volume_bars(trade_data, bar_size=50_000, split_by_date=True)

    # Convert pandas result to polars for comparison
    pd_polars = pl.from_pandas(pd_df, schema_overrides=res.schema)

    assert_frame_equal(
        res,
        pd_polars,
        check_row_order=False,
        check_column_order=False,
    )


def test__volume_bars__overflow_trade():
    """Test volume bars with overflow trade."""
    df = pl.DataFrame(
        [{"symbol": "A", "price": 3.0, "size": 8, "ts_event": dt.datetime(2021, 1, 1)}]
    )
    vol_bars = volume_bars(df, bar_size=5)
    expected = pl.DataFrame(
        [
            {
                "symbol": "A",
                "ts_event_start": dt.datetime(2021, 1, 1),
                "ts_event_end": dt.datetime(2021, 1, 1),
                "open": 3.0,
                "high": 3.0,
                "low": 3.0,
                "close": 3.0,
                "vwap": 3.0,
                "volume": 3,
                "n_trades": 1,
            },
            {
                "symbol": "A",
                "ts_event_start": dt.datetime(2021, 1, 1),
                "ts_event_end": dt.datetime(2021, 1, 1),
                "open": 3.0,
                "high": 3.0,
                "low": 3.0,
                "close": 3.0,
                "vwap": 3.0,
                "volume": 5,
                "n_trades": 1,
            },
        ],
        schema_overrides={"n_trades": pl.UInt32},
    )
    assert_frame_equal(vol_bars, expected, check_row_order=False, check_dtypes=False)


def test__dollar_volume_bars__overflow_trade():
    """Test volume bars with overflow trade."""
    df = pl.DataFrame(
        [{"symbol": "A", "price": 3.0, "size": 8, "ts_event": dt.datetime(2021, 1, 1)}]
    )
    # FIXME
    # I might need to explode this so each share gets its own row. Otherwise the rust side needs more info
    # I also need to handle the rust side by having a flag either allowing splits or not.
    bars = dollar_bars(df, bar_size=13.0)
    expected = pl.DataFrame(
        [
            {
                "symbol": "A",
                "ts_event_start": dt.datetime(2021, 1, 1),
                "ts_event_end": dt.datetime(2021, 1, 1),
                "open": 3.0,
                "high": 3.0,
                "low": 3.0,
                "close": 3.0,
                "vwap": 3.0,
                "volume": 3,
                "n_trades": 1,
            },
            {
                "symbol": "A",
                "ts_event_start": dt.datetime(2021, 1, 1),
                "ts_event_end": dt.datetime(2021, 1, 1),
                "open": 3.0,
                "high": 3.0,
                "low": 3.0,
                "close": 3.0,
                "vwap": 3.0,
                "volume": 5,
                "n_trades": 1,
            },
        ],
        schema_overrides={"n_trades": pl.UInt32},
    )
    assert_frame_equal(bars, expected, check_row_order=False, check_dtypes=False)
