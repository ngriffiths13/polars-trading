from testing_utils.pd_features_helpers import apply_pd_frac_diff
import pytest


@pytest.mark.benchmark(group="frac_diff")
@pytest.mark.pandas
@pytest.mark.parametrize(
    "trade_data", [{"n_rows": 10_000, "n_companies": 3}], indirect=True
)
def test__frac_diff__benchmark_pandas(benchmark, trade_data):
    benchmark(apply_pd_frac_diff, trade_data.to_pandas(), 0.5, 1e-5)
