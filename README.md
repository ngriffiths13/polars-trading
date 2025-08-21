# Polars Trading
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/ngriffiths13/polars-trading)
[![codecov](https://codecov.io/github/ngriffiths13/polars-trading/graph/badge.svg?token=T0BPP3DAD3)](https://codecov.io/github/ngriffiths13/polars-trading)

## Overview
The `polars-trading` package is meant to provide some nice utilities for working with market data in Polars DataFrames. Much of the original inspiration has come from Marcos Lopez de Prado's book *Advances in Financial Machine Learning*. It is a work in progress with some basic functionality that will be added to over time.

## Installation
```bash
uv add polars-trading
```

## Basic Usage

### Creating Bars from Ticks

```python
import polars as pl
import polars_trading as plt

plt.Config.set(
    price_column="price",
    size_column="size",
    symbol_column="ticker",
    timestamp_column="ts_event"
)

# 15 minute time bars
plt.bars.time_bars(df, bar_size="15m")

# Bars w/ 100 ticks
plt.bars.tick_bars(df, bar_size=100)
```
