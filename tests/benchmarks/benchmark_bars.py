import pytest

from polars_trading.bars import dollar_bars, tick_bars, time_bars, volume_bars
from polars_trading.config import Config


@pytest.mark.benchmark(group="time_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test_bench_time_bars(benchmark, trade_data):
    """Benchmarks the `time_bars` function."""
    with Config(timestamp_column="ts_event"):
        benchmark(time_bars, trade_data.lazy(), bar_size="1m")


@pytest.mark.benchmark(group="tick_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test_bench_tick_bars(benchmark, trade_data):
    """Benchmarks the `tick_bars` function."""
    with Config(timestamp_column="ts_event"):
        benchmark(tick_bars, trade_data.lazy(), bar_size=100)


@pytest.mark.benchmark(group="volume_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test_bench_volume_bars(benchmark, trade_data):
    """Benchmarks the `volume_bars` function."""
    with Config(timestamp_column="ts_event", size_column="size"):
        benchmark(volume_bars, trade_data.lazy(), bar_size=10_000)


@pytest.mark.benchmark(group="dollar_bars")
@pytest.mark.parametrize(
    "trade_data",
    [
        {"n_rows": 10_000, "n_companies": 5},
    ],
    indirect=True,
)
def test_bench_dollar_bars(benchmark, trade_data):
    """Benchmarks the `dollar_bars` function."""
    with Config(
        timestamp_column="ts_event",
        price_column="price",
        size_column="size",
    ):
        benchmark(dollar_bars, trade_data.lazy(), bar_size=1_000_000)
