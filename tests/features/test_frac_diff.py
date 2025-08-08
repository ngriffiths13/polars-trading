from polars_trading.features.frac_diff import frac_diff
import pytest
from polars.testing import assert_frame_equal
import polars as pl
from testing_utils.pd_features_helpers import apply_pd_frac_diff


@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__frac_diff__matches_pandas(trade_data):
    trade_data = trade_data.sort("ts_event")
    out = trade_data.select(
        "ts_event",
        "symbol",
        frac_diff("price", 0.5, 1e-3).over("symbol").alias("frac_diff"),
    )
    out2 = (
        pl.DataFrame(apply_pd_frac_diff(trade_data.to_pandas(), 0.5, 1e-3))
        .rename({"level_1": "ts_event", "price": "frac_diff"})
        .cast(out.schema)
    )
    assert_frame_equal(
        out.drop_nulls().sort("ts_event", "symbol"),
        out2.sort("ts_event", "symbol"),
        check_column_order=False,
    )
