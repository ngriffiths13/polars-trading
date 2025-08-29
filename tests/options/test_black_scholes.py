from polars_trading.options import black_scholes
import polars as pl


def test__black_scholes():
    df = pl.DataFrame(
        {
            "time_to_expiry": [0.25, 0.5, 0.75, 1.0],
            "option_type": ["put", "put", "call", "call"],
            "underlying": [101.0, 101.0, 101.0, 101.0],
            "strike": [100.0, 95.0, 105.0, 110.0],
            "implied_vol": [0.25, 0.28, 0.22, 0.24],
            "rate": [0.03, 0.03, 0.03, 0.03],
        }
    )

    results = df.with_columns(
        price=black_scholes(
            s=pl.col("underlying"),
            k=pl.col("strike"),
            t=pl.col("time_to_expiry"),
            sigma=pl.col("implied_vol"),
            r=pl.col("rate"),
            type_=pl.col("option_type"),
        )
    )

    answers = [4.164716, 4.524882, 6.924801, 7.308708]

    assert all(abs(a - b) < 1e-6 for a, b in zip(results["price"].to_list(), answers))
