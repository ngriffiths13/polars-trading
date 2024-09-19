import pytest
from polars_trading._testing.data import generate_trade_data


@pytest.fixture
def trade_data(request):
    return generate_trade_data(**request.param)
