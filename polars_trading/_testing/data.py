from functools import lru_cache

import polars as pl
from mimesis import Fieldset, Finance
from mimesis.locales import Locale


@lru_cache
def generate_trade_data(n_rows: int, n_companies: int = 3) -> pl.DataFrame:
    fs = Fieldset(locale=Locale.EN, i=n_rows)

    return pl.DataFrame(
        {
            "ts_event": fs(
                "datetime",
            ),
            "price": fs("finance.price", minimum=1, maximum=100),
            "size": fs("numeric.integer_number", start=10_000, end=100_000),
            "symbol": fs(
                "choice.choice",
                items=[Finance().stock_ticker() for _ in range(n_companies)],
            ),
        }
    )
