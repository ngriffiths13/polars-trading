from datetime import date

import polars as pl
import pytest
from polars_finance.bars import volume_bars


@pytest.fixture(scope="module")
def all_data():
    return (
        pl.read_parquet(
            "/Users/nelsongriffiths/projects/polars_finance/data/itch.parquet"
        )
        .select(
            pl.col("symbol").cast(pl.Enum(["AAPL", "AMZN", "MSFT", "TQQQ"])),
            pl.col("ts_event").dt.convert_time_zone("US/Eastern"),
            "price",
            "size",
        )
        .sort("symbol", "ts_event")
    )


@pytest.fixture(scope="module")
def all_data_single_security():
    return (
        pl.scan_parquet(
            "/Users/nelsongriffiths/projects/polars_finance/data/itch.parquet"
        )
        .filter(pl.col("symbol") == "AAPL")
        .select(
            pl.col("symbol").cast(pl.Enum(["AAPL", "AMZN", "MSFT", "TQQQ"])),
            pl.col("ts_event").dt.convert_time_zone("US/Eastern"),
            "price",
            "size",
        )
        .sort("symbol", "ts_event")
        .collect()
    )


@pytest.fixture(scope="module")
def small_data():
    return (
        pl.scan_parquet(
            "/Users/nelsongriffiths/projects/polars_finance/data/itch.parquet"
        )
        .select(
            pl.col("symbol").cast(pl.Enum(["AAPL", "AMZN", "MSFT", "TQQQ"])),
            pl.col("ts_event").dt.convert_time_zone("US/Eastern"),
            "price",
            "size",
        )
        .sort("symbol", "ts_event")
        .filter(pl.col("ts_event").dt.date() < date(2024, 2, 20))
        .collect()
    )


@pytest.fixture(scope="module")
def small_data_single_security():
    return (
        pl.scan_parquet(
            "/Users/nelsongriffiths/projects/polars_finance/data/itch.parquet"
        )
        .select(
            pl.col("symbol").cast(pl.Enum(["AAPL", "AMZN", "MSFT", "TQQQ"])),
            pl.col("ts_event").dt.convert_time_zone("US/Eastern"),
            "price",
            "size",
        )
        .sort("symbol", "ts_event")
        .filter(
            pl.col("ts_event").dt.date() < date(2024, 2, 20), pl.col("symbol") == "AAPL"
        )
        .collect()
    )


# Python impl volume bars
def volume_bars_py(df: pl.DataFrame, bar_size) -> pl.DataFrame:
    def _to_ohlcv_df(bar_rows: list[tuple]):
        return pl.DataFrame(
            bar_rows, schema=["symbol", "ts_event", "price", "size"]
        ).select(
            pl.col("symbol").first(),
            pl.col("ts_event").first().alias("start_dt"),
            pl.col("ts_event").last().alias("end_dt"),
            pl.col("price").first().alias("open"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("price").last().alias("close"),
            ((pl.col("price") * pl.col("size")).sum() / pl.col("size").sum()).alias(
                "vwap"
            ),
            pl.col("size").sum().alias("volume"),
            pl.len().alias("n_transactions"),
        )

    ohlcv_rows = []
    current_bar = []
    for row in df.rows(named=True):
        remaining_size = bar_size - sum([r[-1] for r in current_bar])
        while row["size"] > remaining_size:
            current_bar.append(
                (row["symbol"], row["ts_event"], row["price"], remaining_size)
            )
            ohlcv_rows.append(_to_ohlcv_df(current_bar))
            current_bar = []
            row["size"] = row["size"] - remaining_size
            remaining_size = bar_size

        if row["size"] > 0:
            current_bar.append(
                (row["symbol"], row["ts_event"], row["price"], row["size"])
            )
    return pl.concat(ohlcv_rows)


@pytest.mark.benchmark(group="volume_bars_single_security")
def test__volume_bars_small_single_security__polars_plugin(
    small_data_single_security, benchmark
):
    benchmark(volume_bars, small_data_single_security, bar_size=100_000)


@pytest.mark.benchmark(group="volume_bars_single_security")
def test__volume_bars_small_single_security__python(
    small_data_single_security, benchmark
):
    benchmark(volume_bars_py, small_data_single_security, bar_size=100_000)
