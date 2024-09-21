from polars_trading.labels.dynamic_labels import daily_vol
import polars as pl
from polars_trading._testing.labels import get_daily_vol
import pytest
from polars.testing import assert_frame_equal


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
    assert_frame_equal(pl_result, pd_result)


@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 3},
    ],
    indirect=True,
)
def test__daily_vol__multi_security(trade_data):
    pl_result = (
        daily_vol(trade_data.lazy(), "ts_event", "price", "symbol", 5)
        .collect()
        .sort("ts_event", "symbol")
    )
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
        {"n_rows": 10_000, "n_companies": 3},
        {"n_rows": 100_000, "n_companies": 5},
        {"n_rows": 1_000_000, "n_companies": 10},
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
