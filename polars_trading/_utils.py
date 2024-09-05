import inspect
from functools import wraps
from typing import Callable

import polars as pl

from polars_trading.typing import FrameType


def validate_columns(*column_args: str) -> Callable:
    """Validate that the specified columns exist in the dataframe.

    Args:
    ----
        *column_args: str - The column names to validate.

    Returns:
    -------
        Callable

    """

    def decorator(func: Callable) -> Callable:
        """Decorate validation of columns in a polars DataFrame.

        Args:
        ----
            func: Callable

        Returns:
        -------
            Callable

        """

        @wraps(func)
        def wrapper(*args: FrameType, **kwargs: str) -> Callable:
            """Validate that the specified columns exist in the dataframe.

            Args:
            ----
                *args: DataFrameTypes
                    The first argument must be a polars DataFrame.
                **kwargs: str
                    The column names to validate.

            Raises:
            ------
                TypeError: If the first argument is not a polars DataFrame.
                ValueError: If any of the specified columns do not exist.

            Returns:
            -------
                Callable

            """
            # Get the dataframe from args or kwargs
            df = args[0]

            if not isinstance(df, pl.DataFrame | pl.LazyFrame):
                raise TypeError("First argument must be a polars DataFrame")

            func_kwargs = {
                k: v.default
                for k, v in inspect.signature(func).parameters.items()
                if v.default is not inspect.Parameter.empty
            }
            func_kwargs.update(kwargs)
            required_columns = [func_kwargs[a] for a in column_args]
            # Check if all specified columns exist in the dataframe
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
