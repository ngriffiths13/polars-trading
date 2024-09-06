from functools import lru_cache

import polars as pl
from mimesis import Fieldset
from mimesis.locales import Locale


@lru_cache
def generate_trade_data(n_rows: int) -> pl.DataFrame:
    fs = Fieldset(locale=Locale.EN, i=n_rows)

    return pl.DataFrame(
        {
            "ts_event": fs(
                "datetime",
            ),
            "price": fs("finance.price", minimum=1, maximum=100),
            "size": fs("numeric.integer_number", start=10_000, end=100_000),
            "symbol": fs("choice.choice", items=["AAPL", "GOOGL", "MSFT"]),
        }
    )
