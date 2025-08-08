import pytest
from polars_trading.features.frac_diff import frac_diff


@pytest.mark.benchmark(group="frac_diff")
@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__frac_diff__benchmark_polars(benchmark, trade_data):
    trade_data = trade_data.lazy()
    benchmark(
        trade_data.select(
            "ts_event", "symbol", frac_diff("price", 0.5, 1e-3).alias("frac_diff")
        ).collect
    )
