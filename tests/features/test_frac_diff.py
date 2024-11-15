from polars_trading._testing.features import get_weights_ffd, frac_diff_ffd
from polars_trading._internal import get_weights_ffd_py
from polars_trading.features.frac_diff import frac_diff
import pytest
from polars.testing import assert_frame_equal
import pandas as pd
import polars as pl


def apply_pd_frac_diff(df: pd.DataFrame, d: float, threshold: float) -> pd.DataFrame:
    return (
        df.set_index("ts_event")
        .groupby("symbol")[["price"]]
        .apply(frac_diff_ffd, 0.5, 1e-3)
        .reset_index()
    )


def test__get_weights_ffd__matches_pandas():
    out = get_weights_ffd(0.5, 1e-3).flatten().tolist()
    out2 = get_weights_ffd_py(0.5, 1e-3)
    assert out == out2


@pytest.mark.benchmark(group="get_weights_ffd")
def test__get_weights_ffd__benchmark_rs(benchmark):
    benchmark(get_weights_ffd_py, 0.5, 1e-5)


@pytest.mark.benchmark(group="get_weights_ffd")
@pytest.mark.pandas
def test__get_weights_ffd__benchmark_pandas(benchmark):
    benchmark(get_weights_ffd, 0.5, 1e-5)


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__frac_diff__matches_pandas(trade_data):
    trade_data = trade_data.sort("ts_event")
    out = trade_data.select(
        "ts_event", "symbol", frac_diff("price", 0.5, 1e-3).alias("frac_diff")
    )
    out2 = pl.DataFrame(apply_pd_frac_diff(trade_data.to_pandas(), 0.5, 1e-3)).rename(
        {"level_1": "ts_event", "price": "frac_diff"}
    ).cast(out.schema)
    assert_frame_equal(
        out.drop_nulls().sort("ts_event", "symbol"),
        out2.sort("ts_event", "symbol"),
        check_column_order=False,
    )


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


@pytest.mark.benchmark(group="frac_diff")
@pytest.mark.pandas
@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__frac_diff__benchmark_pandas(benchmark, trade_data):
    benchmark(apply_pd_frac_diff, trade_data.to_pandas(), 0.5, 1e-5)
