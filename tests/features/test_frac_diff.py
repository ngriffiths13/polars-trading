from polars_trading._testing.features import get_weights_ffd
from polars_trading._internal import get_weights_ffd_py
import pytest

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
