import pytest

import polars_trading as plt
from polars_trading._testing.data import generate_trade_data


@pytest.fixture
def trade_data(request):
    return generate_trade_data(**request.param)


@pytest.fixture(scope="session", autouse=True)
def set_config():
    plt.Config.set_timestamp_column("ts_event")
