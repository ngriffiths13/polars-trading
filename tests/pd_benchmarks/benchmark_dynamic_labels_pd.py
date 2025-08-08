import pytest

from polars_trading._testing.labels import get_daily_vol


@pytest.mark.pandas
@pytest.mark.benchmark(group="daily_vol")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__daily_vol__pandas_benchmark(benchmark, trade_data):
    pd_df = trade_data.to_pandas().set_index("ts_event")[["symbol", "price"]]

    def get_daily_vol_pd(pd_df):
        return pd_df.groupby("symbol")["price"].apply(get_daily_vol).reset_index()

    benchmark(get_daily_vol_pd, pd_df)
