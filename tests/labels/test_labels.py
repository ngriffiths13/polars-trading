import polars as pl
from polars.testing import assert_series_equal

from polars_trading.labels.labels import (
    fixed_time_return,
    fixed_time_return_classification,
)


def test__fixed_time_return__single_security():
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0],
            "s": [1, 2, 3, 4, 5],
            "sy": ["A", "A", "A", "A", "A"],
        }
    )

    result = df.with_columns(fixed_time_return("p", 1).alias("label"))
    assert_series_equal(
        result["label"],
        pl.Series(values=[0.5, 0.3333333333333333, 0.25, None, None]),
        check_names=False,
    )


def test__fixed_time_return__multi_security():
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "s": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "sy": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )

    result = df.with_columns(fixed_time_return("p", 1, symbol="sy").alias("label"))
    assert_series_equal(
        result["label"],
        pl.Series(
            values=[
                0.5,
                0.3333333333333333,
                0.25,
                None,
                None,
                0.5,
                0.3333333333333333,
                0.25,
                None,
                None,
            ]
        ),
        check_names=False,
    )


def test__fixed_time_return_classification__float_threshold():
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0],
            "s": [1, 2, 3, 4, 5],
            "sy": ["A", "A", "A", "A", "A"],
        }
    )

    result = df.with_columns(
        fixed_time_return_classification("p", 1, 0.3).alias("label")
    )
    assert_series_equal(
        result["label"],
        pl.Series(values=[1, 1, 0, None, None], dtype=pl.Int32),
        check_names=False,
    )


def test__fixed_time_return_classification__no_threshold():
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0],
            "s": [1, 2, 3, 4, 5],
            "sy": ["A", "A", "A", "A", "A"],
        }
    )

    result = df.with_columns(
        fixed_time_return_classification(
            "p",
            1,
        ).alias("label")
    )
    assert_series_equal(
        result["label"],
        pl.Series(values=[1, 1, 1, None, None], dtype=pl.Int32),
        check_names=False,
    )


def test__fixed_time_return_classification__expr_threshold():
    df = pl.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "p": [1.0, 2.0, 3.0, 4.0, 5.0],
            "t": [0.7, 0.3, 0.1, 0.3, 0.3],
            "s": [1, 2, 3, 4, 5],
            "sy": ["A", "A", "A", "A", "A"],
        }
    )

    result = df.with_columns(
        fixed_time_return_classification("p", 1, "t").alias("label")
    )
    assert_series_equal(
        result["label"],
        pl.Series(values=[0, 1, 1, None, None], dtype=pl.Int32),
        check_names=False,
    )
