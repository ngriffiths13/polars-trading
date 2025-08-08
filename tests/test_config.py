"""Tests for polars_trading.config module."""

from __future__ import annotations

from polars_trading.config import DEFAULT_COLUMN_NAMES, Config, column_names


class TestConfig:
    """Test Config class core functionality."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        Config.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        Config.reset()

    def test_set_and_get(self) -> None:
        """Test setting and getting configuration values."""
        Config.set(open_column="o", close_column="c")
        assert Config.get("open_column") == "o"
        assert Config.get("close_column") == "c"

    def test_get_default_values(self) -> None:
        """Test getting default values when not configured."""
        assert Config.get("open_column") == "open"
        assert Config.get("close_column") == "close"

    def test_reset(self) -> None:
        """Test resetting configuration to defaults."""
        Config.set(open_column="custom")
        Config.reset()
        assert Config.get("open_column") == "open"

    def test_context_manager(self) -> None:
        """Test Config as context manager."""
        Config.set(open_column="global")

        with Config(open_column="context"):
            assert Config.get("open_column") == "context"

        assert Config.get("open_column") == "global"

    def test_specific_column_setters(self) -> None:
        """Test column-specific setter methods."""
        Config.set_open_column("o")
        Config.set_close_column("c")

        assert Config.get("open_column") == "o"
        assert Config.get("close_column") == "c"


class TestColumnNames:
    """Test ColumnNames functionality."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        Config.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        Config.reset()

    def test_default_column_access(self) -> None:
        """Test accessing default column names."""
        assert column_names.open == "open"
        assert column_names.close == "close"
        assert column_names.high == "high"
        assert column_names.low == "low"

    def test_configured_column_access(self) -> None:
        """Test accessing configured column names."""
        Config.set(open_column="o", close_column="c")

        assert column_names.open == "o"
        assert column_names.close == "c"

    def test_column_suffix_access(self) -> None:
        """Test accessing column names with _column suffix."""
        Config.set(open_column="o")

        assert column_names.open_column == "o"
        assert column_names.open == "o"


def test_default_column_names() -> None:
    """Test DEFAULT_COLUMN_NAMES constant."""
    expected_keys = {
        "open",
        "high",
        "low",
        "close",
        "size",
        "timestamp",
        "symbol",
        "price",
    }
    assert set(DEFAULT_COLUMN_NAMES.keys()) == expected_keys
