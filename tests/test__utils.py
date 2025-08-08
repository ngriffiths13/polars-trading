import polars as pl

from polars_trading._utils import parse_into_expr


def test__parse_into_expr__expr_input():
    """Test parse_into_expr with pl.Expr input returns expression unchanged."""
    expr = pl.col("test")
    result = parse_into_expr(expr)

    assert result is expr  # Should return the same object


def test__parse_into_expr__string_as_column():
    """Test parse_into_expr with string input as column reference (default behavior)."""
    result = parse_into_expr("column_name")
    expected = pl.col("column_name")

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"column_name": [1, 2, 3]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__string_as_literal():
    """Test parse_into_expr with string input as literal when str_as_lit=True."""
    result = parse_into_expr("test_value", str_as_lit=True)
    expected = pl.lit("test_value")

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__list_as_literal():
    """Test parse_into_expr with list input as literal (default behavior)."""
    test_list = [1, 2, 3]
    result = parse_into_expr(test_list)
    expected = pl.lit(test_list)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__list_as_series():
    """Test parse_into_expr with list input as Series literal when list_as_lit=False."""
    test_list = [1, 2, 3]
    result = parse_into_expr(test_list, list_as_lit=False)
    expected = pl.lit(pl.Series(test_list))

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__list_as_series_with_dtype():
    """Test parse_into_expr with list input as Series with specific dtype."""
    test_list = [1, 2, 3]
    result = parse_into_expr(test_list, list_as_lit=False, dtype=pl.Int32)
    expected = pl.lit(pl.Series(test_list), dtype=pl.Int32)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__numeric_literal():
    """Test parse_into_expr with numeric input as literal."""
    result = parse_into_expr(42)
    expected = pl.lit(42)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__numeric_literal_with_dtype():
    """Test parse_into_expr with numeric input and specific dtype."""
    result = parse_into_expr(42, dtype=pl.Float64)
    expected = pl.lit(42, dtype=pl.Float64)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__boolean_literal():
    """Test parse_into_expr with boolean input as literal."""
    result = parse_into_expr(True)
    expected = pl.lit(True)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))


def test__parse_into_expr__none_literal():
    """Test parse_into_expr with None input as literal."""
    result = parse_into_expr(None)
    expected = pl.lit(None)

    # Test by evaluating in a dataframe context
    df = pl.DataFrame({"dummy": [1]})
    assert df.select(result).equals(df.select(expected))
