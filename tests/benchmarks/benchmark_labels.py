import pytest

from polars_trading.labels.labels import fixed_time_return_classification


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test__fixed_time_return_classification__benchmark(trade_data, benchmark):
    """Benchmarks the `fixed_time_return_classification` function."""
    benchmark(
        trade_data.lazy()
        .select(fixed_time_return_classification("price", 50, 0.2, symbol="symbol"))
        .collect
    )
