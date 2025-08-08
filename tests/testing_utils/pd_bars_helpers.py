"""Helper functions for benchmark testing against Polars implementations.

This module contains pandas and vanilla Python implementations of bar aggregation functions
that are used for benchmarking and validation against the main Polars implementations.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def pandas_time_bars(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Create time-based bars using pandas for benchmark comparison.

    Aggregates trade data into time-based bars using pandas operations.
    This function is used as a reference implementation for benchmarking
    against the Polars version.

    Args:
        df (pd.DataFrame): DataFrame containing trade data with columns:
            - ts_event: timestamp of the event
            - symbol: trading symbol
            - price: trade price
            - size: trade size/volume
        period (str): Time period for aggregation (e.g., '1d', '1h', '30min')

    Returns:
        pd.DataFrame: Aggregated bars with columns:
            - ts_event: timestamp index
            - symbol: trading symbol
            - ts_event_start: first timestamp in the bar
            - ts_event_end: last timestamp in the bar
            - n_trades: number of trades in the bar
            - open: opening price
            - high: highest price
            - low: lowest price
            - close: closing price
            - volume: total volume
            - vwap: volume weighted average price

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "ts_event": ["2023-01-01 09:00:00", "2023-01-01 10:00:00"],
        ...         "symbol": ["AAPL", "AAPL"],
        ...         "price": [150.0, 151.0],
        ...         "size": [100, 200],
        ...     }
        ... )
        >>> bars = pandas_time_bars(df, "1d")
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    df.index = pd.to_datetime(df["ts_event"])
    df["pvt"] = df["price"] * df["size"]
    df = df.sort_index()
    resampled_df = (
        df.groupby([df.index.to_period(period), "symbol"])
        .agg(
            ts_event_start=pd.NamedAgg(column="ts_event", aggfunc="first"),
            ts_event_end=pd.NamedAgg(column="ts_event", aggfunc="last"),
            n_trades=pd.NamedAgg(column="ts_event", aggfunc="count"),
            open=pd.NamedAgg(column="price", aggfunc="first"),
            high=pd.NamedAgg(column="price", aggfunc="max"),
            low=pd.NamedAgg(column="price", aggfunc="min"),
            close=pd.NamedAgg(column="price", aggfunc="last"),
            volume=pd.NamedAgg(column="size", aggfunc="sum"),
            vwap=pd.NamedAgg(column="pvt", aggfunc="sum"),
        )
        .reset_index("symbol")
    )
    resampled_df.index.names = ["ts_event"]
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


def pandas_tick_bars(df: pd.DataFrame, n_ticks: int) -> pd.DataFrame:
    """Create tick-based bars using pandas for benchmark comparison.

    Aggregates trade data into tick-based bars (fixed number of trades per bar)
    using pandas operations. This function is used as a reference implementation
    for benchmarking against the Polars version.

    Args:
        df (pd.DataFrame): DataFrame containing trade data with columns:
            - ts_event: timestamp of the event
            - symbol: trading symbol
            - price: trade price
            - size: trade size/volume
        n_ticks (int): Number of ticks (trades) per bar

    Returns:
        pd.DataFrame: Aggregated bars with columns:
            - symbol: trading symbol
            - ts_event_start: first timestamp in the bar
            - ts_event_end: last timestamp in the bar
            - n_trades: number of trades in the bar
            - open: opening price
            - high: highest price
            - low: lowest price
            - close: closing price
            - volume: total volume
            - vwap: volume weighted average price

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "ts_event": [
        ...             "2023-01-01 09:00:00",
        ...             "2023-01-01 09:01:00",
        ...             "2023-01-01 09:02:00",
        ...         ],
        ...         "symbol": ["AAPL", "AAPL", "AAPL"],
        ...         "price": [150.0, 151.0, 149.0],
        ...         "size": [100, 200, 150],
        ...     }
        ... )
        >>> bars = pandas_tick_bars(df, 2)
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    df["ts_event"] = pd.to_datetime(df["ts_event"])
    df = df.sort_values("ts_event")
    df["pvt"] = df["price"] * df["size"]
    df["date"] = pd.to_datetime(df["ts_event"]).dt.date
    df["tick_group"] = (df.groupby(["symbol", "date"]).cumcount() // n_ticks).astype(
        int
    )
    resampled_df = (
        df.groupby(["date", "tick_group", "symbol"])
        .agg(
            ts_event_start=pd.NamedAgg(column="ts_event", aggfunc="first"),
            ts_event_end=pd.NamedAgg(column="ts_event", aggfunc="last"),
            n_trades=pd.NamedAgg(column="ts_event", aggfunc="count"),
            open=pd.NamedAgg(column="price", aggfunc="first"),
            high=pd.NamedAgg(column="price", aggfunc="max"),
            low=pd.NamedAgg(column="price", aggfunc="min"),
            close=pd.NamedAgg(column="price", aggfunc="last"),
            volume=pd.NamedAgg(column="size", aggfunc="sum"),
            vwap=pd.NamedAgg(column="pvt", aggfunc="sum"),
        )
        .reset_index()
        .drop(["date", "tick_group"], axis=1)
    )
    resampled_df["vwap"] = resampled_df["vwap"] / resampled_df["volume"]
    return resampled_df


def pandas_volume_bars(
    df: pd.DataFrame, volume_threshold: int, split_by_date: bool = True
) -> pd.DataFrame:
    """Create volume-based bars using pandas for benchmark comparison.

    Aggregates trade data into volume-based bars (fixed volume per bar)
    using pandas operations. This function is used as a reference implementation
    for benchmarking against the Polars version. Trades may be split across
    bars if their volume would cause a bar to exceed the threshold.

    Args:
        df (pd.DataFrame): DataFrame containing trade data with columns:
            - ts_event: timestamp of the event (can be named 'timestamp')
            - symbol: trading symbol
            - price: trade price
            - size: trade size/volume
        volume_threshold (int): Volume threshold for each bar
        split_by_date (bool): Whether to prevent bars from spanning multiple dates

    Returns:
        pd.DataFrame: Aggregated bars with columns:
            - symbol: trading symbol
            - ts_event_start: first timestamp in the bar
            - ts_event_end: last timestamp in the bar
            - n_trades: number of trades in the bar
            - open: opening price
            - high: highest price
            - low: lowest price
            - close: closing price
            - volume: total volume
            - vwap: volume weighted average price

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "ts_event": [
        ...             "2023-01-01 09:00:00",
        ...             "2023-01-01 09:01:00",
        ...             "2023-01-01 09:02:00",
        ...         ],
        ...         "symbol": ["AAPL", "AAPL", "AAPL"],
        ...         "price": [150.0, 151.0, 149.0],
        ...         "size": [5000, 8000, 3000],
        ...     }
        ... )
        >>> bars = pandas_volume_bars(df, 10000)
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Handle different timestamp column names
    timestamp_col = "ts_event" if "ts_event" in df.columns else "timestamp"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df["pvt"] = df["price"] * df["size"]

    # Group by symbol and optionally by date
    group_cols = ["symbol"]
    if split_by_date:
        df["date"] = df[timestamp_col].dt.date
        group_cols.append("date")

    bars_list = []

    for group_keys, group_df in df.groupby(group_cols):
        group_df = group_df.copy().reset_index(drop=True)

        # Implement the same logic as Polars _bar_groups_expr
        # Calculate cumulative volume - when split_by_date=True, each group is already per-date
        group_df["cumulative_volume"] = group_df["size"].cumsum()

        # Step 1: Create initial rows like _bar_groups_expr does
        initial_rows = []
        for idx, row in group_df.iterrows():
            cumvol_start = row["cumulative_volume"] - row["size"]

            # Which bar does this trade start in?
            start_bar = int(cumvol_start // volume_threshold)

            # How much volume goes to the current bar?
            remaining_capacity = volume_threshold - (cumvol_start % volume_threshold)
            amount_for_current_bar = min(row["size"], remaining_capacity)

            new_row = row.copy()
            new_row["bar_group__id"] = start_bar
            new_row["bar_group__amount"] = amount_for_current_bar
            initial_rows.append(new_row)

        initial_df = pd.DataFrame(initial_rows)

        # Step 2: Create split trades for remainders (like Polars vstack operation)
        # Handle trades that span multiple bars by splitting them
        # To match Polars, we need to insert split rows immediately after the initial row
        all_rows = []
        for idx, row in initial_df.iterrows():
            # Add the initial row
            all_rows.append(row.to_dict())

            # If there's a remainder, create split rows
            if row["size"] != row["bar_group__amount"]:
                remaining_volume = row["size"] - row["bar_group__amount"]
                current_bar_id = row["bar_group__id"] + 1

                while remaining_volume > 0:
                    # Each subsequent bar gets either the full threshold or whatever is left
                    amount_for_this_bar = min(remaining_volume, volume_threshold)

                    remainder_row = row.to_dict()
                    remainder_row["bar_group__amount"] = amount_for_this_bar
                    remainder_row["bar_group__id"] = current_bar_id
                    all_rows.append(remainder_row)

                    remaining_volume -= amount_for_this_bar
                    current_bar_id += 1

        # Step 3: Replace size with bar_group__amount
        expanded_rows = []

        for row in all_rows:
            new_row = row.copy()
            new_row["size"] = row["bar_group__amount"]
            new_row["pvt"] = row["bar_group__amount"] * row["price"]
            new_row["bar_id"] = row["bar_group__id"]
            expanded_rows.append(new_row)

        # Convert to DataFrame and group by bar_id
        expanded_df = pd.DataFrame(expanded_rows)

        # Aggregate by bar_id
        bar_agg = expanded_df.groupby("bar_id").agg(
            {
                timestamp_col: ["first", "last"],
                "price": ["first", "max", "min", "last"],
                "size": "sum",
                "pvt": "sum",
                "symbol": "first",
            }
        )

        # Flatten column names
        bar_agg.columns = [
            "ts_event_start"
            if col[0] == timestamp_col and col[1] == "first"
            else "ts_event_end"
            if col[0] == timestamp_col and col[1] == "last"
            else "open"
            if col[0] == "price" and col[1] == "first"
            else "high"
            if col[0] == "price" and col[1] == "max"
            else "low"
            if col[0] == "price" and col[1] == "min"
            else "close"
            if col[0] == "price" and col[1] == "last"
            else "volume"
            if col[0] == "size"
            else "vwap_numerator"
            if col[0] == "pvt"
            else col[0]
            for col in bar_agg.columns
        ]

        # Calculate VWAP and add trade count
        bar_agg["vwap"] = bar_agg["vwap_numerator"] / bar_agg["volume"]
        bar_agg["n_trades"] = expanded_df.groupby("bar_id").size()

        # Clean up and add to results
        bar_agg = bar_agg.drop("vwap_numerator", axis=1)
        bar_agg = bar_agg.reset_index(drop=True)
        bars_list.append(bar_agg)

    # Combine all groups
    if bars_list:
        result = pd.concat(bars_list, ignore_index=True)
        result = result.sort_values("ts_event_end").reset_index(drop=True)
    else:
        # Return empty DataFrame with correct columns
        result = pd.DataFrame(
            columns=[
                "symbol",
                "ts_event_start",
                "ts_event_end",
                "n_trades",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
            ]
        )

    return result


def python_volume_bars(
    data: List[Dict[str, Any]], volume_threshold: int, split_by_date: bool = True
) -> List[Dict[str, Any]]:
    """Create volume-based bars using vanilla Python for benchmark comparison.

    Aggregates trade data into volume-based bars (fixed volume per bar)
    using pure Python operations. This function is used as a reference implementation
    for benchmarking against the Polars version. Trades may be split across
    bars if their volume would cause a bar to exceed the threshold.

    Args:
        data (List[Dict[str, Any]]): List of trade records with keys:
            - ts_event: timestamp of the event (datetime or string)
            - symbol: trading symbol (str)
            - price: trade price (float)
            - size: trade size/volume (int)
        volume_threshold (int): Volume threshold for each bar
        split_by_date (bool): Whether to prevent bars from spanning multiple dates

    Returns:
        List[Dict[str, Any]]: List of aggregated bar records with keys:
            - symbol: trading symbol
            - ts_event_start: first timestamp in the bar
            - ts_event_end: last timestamp in the bar
            - n_trades: number of trades in the bar
            - open: opening price
            - high: highest price
            - low: lowest price
            - close: closing price
            - volume: total volume
            - vwap: volume weighted average price

    Example:
        >>> data = [
        ...     {
        ...         "ts_event": "2023-01-01 09:00:00",
        ...         "symbol": "AAPL",
        ...         "price": 150.0,
        ...         "size": 5000,
        ...     },
        ...     {
        ...         "ts_event": "2023-01-01 09:01:00",
        ...         "symbol": "AAPL",
        ...         "price": 151.0,
        ...         "size": 8000,
        ...     },
        ...     {
        ...         "ts_event": "2023-01-01 09:02:00",
        ...         "symbol": "AAPL",
        ...         "price": 149.0,
        ...         "size": 3000,
        ...     },
        ... ]
        >>> bars = python_volume_bars(data, 10000)
    """
    # Convert data to list of dicts and sort by timestamp
    trades = []
    for trade in data:
        trade_copy = trade.copy()
        # Parse timestamp if it's a string
        if isinstance(trade_copy["ts_event"], str):
            trade_copy["ts_event"] = datetime.fromisoformat(
                trade_copy["ts_event"].replace("Z", "+00:00")
            )
        elif hasattr(trade_copy["ts_event"], "to_pydatetime"):
            # Handle pandas Timestamp
            trade_copy["ts_event"] = trade_copy["ts_event"].to_pydatetime()
        trades.append(trade_copy)

    # Sort by timestamp
    trades.sort(key=lambda x: x["ts_event"])

    # Group by symbol and optionally by date
    groups = defaultdict(list)
    for trade in trades:
        if split_by_date:
            key = (trade["symbol"], trade["ts_event"].date())
        else:
            key = trade["symbol"]
        groups[key].append(trade)

    all_bars = []

    for group_key, group_trades in groups.items():
        # Calculate cumulative volume
        cumulative_volume = 0
        for trade in group_trades:
            cumulative_volume += trade["size"]
            trade["cumulative_volume"] = cumulative_volume

        # Step 1: Create initial rows like _bar_groups_expr does
        initial_rows = []
        for trade in group_trades:
            cumvol_start = trade["cumulative_volume"] - trade["size"]

            # Which bar does this trade start in?
            start_bar = int(cumvol_start // volume_threshold)

            # How much volume goes to the current bar?
            remaining_capacity = volume_threshold - (cumvol_start % volume_threshold)
            amount_for_current_bar = min(trade["size"], remaining_capacity)

            initial_row = trade.copy()
            initial_row["bar_group__id"] = start_bar
            initial_row["bar_group__amount"] = amount_for_current_bar
            initial_rows.append(initial_row)

        # Step 2: Create split trades for remainders (like Polars vstack operation)
        # Handle trades that span multiple bars by recursively splitting them
        split_rows = []
        for row in initial_rows:
            if row["size"] != row["bar_group__amount"]:
                # Recursively split the remainder across multiple bars if needed
                remaining_volume = row["size"] - row["bar_group__amount"]
                current_bar_id = row["bar_group__id"] + 1

                while remaining_volume > 0:
                    # Each subsequent bar gets either the full threshold or whatever is left
                    amount_for_this_bar = min(remaining_volume, volume_threshold)

                    remainder_row = row.copy()
                    remainder_row["bar_group__amount"] = amount_for_this_bar
                    remainder_row["bar_group__id"] = current_bar_id
                    split_rows.append(remainder_row)

                    remaining_volume -= amount_for_this_bar
                    current_bar_id += 1

        # Step 3: Combine initial and split rows, then replace size with bar_group__amount
        all_rows = initial_rows + split_rows
        expanded_rows = []

        for row in all_rows:
            expanded_row = row.copy()
            expanded_row["size"] = row["bar_group__amount"]
            expanded_row["pvt"] = row["bar_group__amount"] * row["price"]
            expanded_row["bar_id"] = row["bar_group__id"]
            expanded_rows.append(expanded_row)

        # Group by bar_id and aggregate
        bar_groups = defaultdict(list)
        for row in expanded_rows:
            bar_groups[row["bar_id"]].append(row)

        # Aggregate each bar group
        for bar_id, bar_trades in bar_groups.items():
            # Sort by timestamp to ensure correct first/last ordering
            bar_trades.sort(key=lambda x: x["ts_event"])

            # Calculate aggregated values
            timestamps = [t["ts_event"] for t in bar_trades]
            prices = [t["price"] for t in bar_trades]
            sizes = [t["size"] for t in bar_trades]
            pvts = [t["pvt"] for t in bar_trades]

            total_volume = sum(sizes)
            total_pvt = sum(pvts)

            bar = {
                "symbol": bar_trades[0]["symbol"],
                "ts_event_start": min(timestamps),
                "ts_event_end": max(timestamps),
                "n_trades": len(bar_trades),
                "open": prices[0],  # First price chronologically
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],  # Last price chronologically
                "volume": total_volume,
                "vwap": total_pvt / total_volume if total_volume > 0 else 0.0,
            }
            all_bars.append(bar)

    # Sort bars by end timestamp
    all_bars.sort(key=lambda x: x["ts_event_end"])

    return all_bars


def pandas_dollar_bars(
    df: pd.DataFrame, dollar_threshold: float, split_by_date: bool = True
) -> pd.DataFrame:
    """Create dollar-based bars using pandas for benchmark comparison.

    Aggregates trade data into dollar-based bars (fixed dollar volume per bar)
    using pandas operations. This function is used as a reference implementation
    for benchmarking against the Polars version. Trades may be split across
    bars if their dollar volume would cause a bar to exceed the threshold.

    Args:
        df (pd.DataFrame): DataFrame containing trade data with columns:
            - ts_event: timestamp of the event (can be named 'timestamp')
            - symbol: trading symbol
            - price: trade price
            - size: trade size/volume
        dollar_threshold (float): Dollar volume threshold for each bar
        split_by_date (bool): Whether to prevent bars from spanning multiple dates

    Returns:
        pd.DataFrame: Aggregated bars with columns:
            - symbol: trading symbol
            - ts_event_start: first timestamp in the bar
            - ts_event_end: last timestamp in the bar
            - n_trades: number of trades in the bar
            - open: opening price
            - high: highest price
            - low: lowest price
            - close: closing price
            - volume: total volume
            - vwap: volume weighted average price

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {
        ...         "ts_event": [
        ...             "2023-01-01 09:00:00",
        ...             "2023-01-01 09:01:00",
        ...             "2023-01-01 09:02:00",
        ...         ],
        ...         "symbol": ["AAPL", "AAPL", "AAPL"],
        ...         "price": [150.0, 151.0, 149.0],
        ...         "size": [100, 200, 150],
        ...     }
        ... )
        >>> bars = pandas_dollar_bars(df, 30000.0)
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Handle different timestamp column names
    timestamp_col = "ts_event" if "ts_event" in df.columns else "timestamp"
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    df["dollar_volume"] = df["price"] * df["size"]

    # Group by symbol and optionally by date
    group_cols = ["symbol"]
    if split_by_date:
        df["date"] = df[timestamp_col].dt.date
        group_cols.append("date")

    bars_list = []

    for group_keys, group_df in df.groupby(group_cols):
        group_df = group_df.copy().reset_index(drop=True)

        # Calculate cumulative dollar volume
        group_df["cumulative_dollar_volume"] = group_df["dollar_volume"].cumsum()

        # Step 1: Create initial rows
        initial_rows = []
        for idx, row in group_df.iterrows():
            cumdollarvol_start = row["cumulative_dollar_volume"] - row["dollar_volume"]

            # Which bar does this trade start in?
            start_bar = int(cumdollarvol_start // dollar_threshold)

            # How much dollar volume goes to the current bar?
            remaining_capacity = dollar_threshold - (
                cumdollarvol_start % dollar_threshold
            )
            amount_for_current_bar = min(row["dollar_volume"], remaining_capacity)

            # Calculate the proportional size for this bar
            size_for_current_bar = row["size"] * (
                amount_for_current_bar / row["dollar_volume"]
            )

            new_row = row.copy()
            new_row["bar_group__id"] = start_bar
            new_row["bar_group__dollar_amount"] = amount_for_current_bar
            new_row["bar_group__size_amount"] = size_for_current_bar
            initial_rows.append(new_row)

        initial_df = pd.DataFrame(initial_rows)

        # Step 2: Create split trades for remainders
        all_rows = []
        for idx, row in initial_df.iterrows():
            # Add the initial row
            all_rows.append(row.to_dict())

            # If there's a remainder, create split rows
            if row["dollar_volume"] != row["bar_group__dollar_amount"]:
                remaining_dollar_volume = (
                    row["dollar_volume"] - row["bar_group__dollar_amount"]
                )
                remaining_size = row["size"] - row["bar_group__size_amount"]
                current_bar_id = row["bar_group__id"] + 1

                while remaining_dollar_volume > 0:
                    # Each subsequent bar gets either the full threshold or whatever is left
                    dollar_amount_for_this_bar = min(
                        remaining_dollar_volume, dollar_threshold
                    )
                    size_amount_for_this_bar = remaining_size * (
                        dollar_amount_for_this_bar / remaining_dollar_volume
                    )

                    remainder_row = row.to_dict()
                    remainder_row["bar_group__dollar_amount"] = (
                        dollar_amount_for_this_bar
                    )
                    remainder_row["bar_group__size_amount"] = size_amount_for_this_bar
                    remainder_row["bar_group__id"] = current_bar_id
                    all_rows.append(remainder_row)

                    remaining_dollar_volume -= dollar_amount_for_this_bar
                    remaining_size -= size_amount_for_this_bar
                    current_bar_id += 1

        # Step 3: Replace size and dollar_volume with proportional amounts
        expanded_rows = []

        for row in all_rows:
            new_row = row.copy()
            new_row["size"] = row["bar_group__size_amount"]
            new_row["dollar_volume"] = row["bar_group__dollar_amount"]
            new_row["pvt"] = row[
                "bar_group__dollar_amount"
            ]  # For dollar bars, pvt is dollar volume
            new_row["bar_id"] = row["bar_group__id"]
            expanded_rows.append(new_row)

        # Convert to DataFrame and group by bar_id
        expanded_df = pd.DataFrame(expanded_rows)

        # Aggregate by bar_id
        bar_agg = expanded_df.groupby("bar_id").agg(
            {
                timestamp_col: ["first", "last"],
                "price": ["first", "max", "min", "last"],
                "size": "sum",
                "dollar_volume": "sum",
                "symbol": "first",
            }
        )

        # Flatten column names
        bar_agg.columns = [
            "ts_event_start"
            if col[0] == timestamp_col and col[1] == "first"
            else "ts_event_end"
            if col[0] == timestamp_col and col[1] == "last"
            else "open"
            if col[0] == "price" and col[1] == "first"
            else "high"
            if col[0] == "price" and col[1] == "max"
            else "low"
            if col[0] == "price" and col[1] == "min"
            else "close"
            if col[0] == "price" and col[1] == "last"
            else "volume"
            if col[0] == "size"
            else "dollar_volume_total"
            if col[0] == "dollar_volume"
            else col[0]
            for col in bar_agg.columns
        ]

        # Calculate VWAP and add trade count
        bar_agg["vwap"] = bar_agg["dollar_volume_total"] / bar_agg["volume"]
        bar_agg["n_trades"] = expanded_df.groupby("bar_id").size()

        # Clean up and add to results
        bar_agg = bar_agg.drop("dollar_volume_total", axis=1)
        bar_agg = bar_agg.reset_index(drop=True)
        bars_list.append(bar_agg)

    # Combine all groups
    if bars_list:
        result = pd.concat(bars_list, ignore_index=True)
        result = result.sort_values("ts_event_end").reset_index(drop=True)
    else:
        # Return empty DataFrame with correct columns
        result = pd.DataFrame(
            columns=[
                "symbol",
                "ts_event_start",
                "ts_event_end",
                "n_trades",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
            ]
        )

    return result
