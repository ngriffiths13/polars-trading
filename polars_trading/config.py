from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    if sys.version_info >= (3, 11):
        from typing import Self, Unpack
    else:
        from typing import Self, Unpack

__all__ = [
    "Config",
    "column_names",
]

DEFAULT_COLUMN_NAMES = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "size": "size",
    "timestamp": "timestamp",
    "symbol": "symbol",
    "price": "price",
}


class ConfigParameters(TypedDict, total=False):
    """Parameters supported by the polars-trading Config."""

    open_column: str
    high_column: str
    low_column: str
    close_column: str
    size_column: str
    timestamp_column: str
    symbol_column: str
    price_column: str


_config_context: ContextVar[ConfigParameters | None] = ContextVar(
    "polars_trading_config", default=None
)


class Config(contextlib.ContextDecorator):
    """Configure polars-trading; offers options for default column names.

    Notes:
    -----
    Can also be used as a context manager OR a function decorator in order to
    temporarily scope the lifetime of specific options.

    Examples:
    --------
    >>> import polars_trading as plt
    >>>
    >>> plt.Config.set(open_column="o", close_column="c")
    >>> plt.column_names.open
        "o"
    """

    def __init__(self, **options: Unpack[ConfigParameters]) -> None:
        """Initialise a Config object instance for context manager usage."""
        self._options = options
        self._token = None

    def __enter__(self) -> Self:
        """Support setting Config options that are reset on scope exit."""
        current = _config_context.get({})
        updated = {**current, **self._options}
        self._token = _config_context.set(updated)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Reset any Config options that were set within the scope."""
        if self._token:
            _config_context.reset(self._token)

    @classmethod
    def set(cls, **options: Unpack[ConfigParameters]) -> type[Config]:
        """Set configuration values globally."""
        current = _config_context.get({})
        updated = {**current, **options}
        _config_context.set(updated)
        return cls

    @classmethod
    def get(cls, key: str) -> str:
        """Get a configuration value."""
        current = _config_context.get({})
        column_key = key.replace("_column", "") if key.endswith("_column") else key
        return current.get(key, DEFAULT_COLUMN_NAMES.get(column_key, ""))

    @classmethod
    def reset(cls) -> type[Config]:
        """Reset all polars-trading Config settings to their default state."""
        _config_context.set({})
        return cls

    @classmethod
    def set_open_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'open' column."""
        return cls.set(open_column=name)

    @classmethod
    def set_high_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'high' column."""
        return cls.set(high_column=name)

    @classmethod
    def set_low_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'low' column."""
        return cls.set(low_column=name)

    @classmethod
    def set_close_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'close' column."""
        return cls.set(close_column=name)

    @classmethod
    def set_size_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'size' column."""
        return cls.set(size_column=name)

    @classmethod
    def set_timestamp_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'timestamp' column."""
        return cls.set(timestamp_column=name)

    @classmethod
    def set_symbol_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'symbol' column."""
        return cls.set(symbol_column=name)

    @classmethod
    def set_price_column(cls, name: str) -> type[Config]:
        """Set the default name for the 'price' column."""
        return cls.set(price_column=name)


class ColumnNames:
    """Dynamic access to column names from configuration context."""

    def __getattr__(self, name: str) -> str:
        """Get column name from config context or default values.

        Supports both direct column names (e.g., 'open', 'close') and
        column key names (e.g., 'open_column', 'close_column').
        """
        current = _config_context.get({})

        if name.endswith("_column"):
            column_key = name
            default_key = name.replace("_column", "")
        else:
            column_key = f"{name}_column"
            default_key = name

        return current.get(column_key, DEFAULT_COLUMN_NAMES.get(default_key, ""))


column_names = ColumnNames()
