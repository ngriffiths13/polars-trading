import pytest

from polars_trading.config import Config
from polars_trading.labels.dynamic_labels import (
    daily_vol,
)


@pytest.mark.benchmark(group="daily_vol")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__daily_vol__polars_benchmark(benchmark, trade_data):
    with Config(timestamp_column="ts_event"):
        benchmark(daily_vol(trade_data.lazy(), span=100).collect)


# @pytest.mark.benchmark(group="vertical_barrier")
# @pytest.mark.parametrize(
#     "trade_data",
#     [
#         {"n_rows": 10_000, "n_companies": 5},
#     ],
#     indirect=True,
# )
# def test__get_vertical_barrier_by_timedelta__benchmark(benchmark, trade_data):
#     with Config(timestamp_column="ts_event"):
#         benchmark(get_vertical_barrier_by_timedelta, trade_data, "1h")
