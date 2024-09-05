import marimo

__generated_with = "0.8.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    from polars_trading._testing.data import generate_trade_data
    from copy import deepcopy
    return deepcopy, generate_trade_data, pl


@app.cell
def __(generate_trade_data):
    df = generate_trade_data(100_000)
    return df,


@app.cell
def __(df, pl):
    bar_size = 1_000_000
    rows = list(df.sort("ts_event").filter(pl.col("symbol") == "AAPL").iter_rows(named=True))
    return bar_size, rows


@app.cell
def __(bar_size, deepcopy, rows):
    aggs = []
    row_group = []
    curr_vol = 0
    for row in rows:
        if curr_vol + row["size"] >= bar_size:
            tmp_row = deepcopy(row)
            tmp_row["size"] = bar_size - curr_vol
            row["size"] -= tmp_row["size"]
            row_group.append(tmp_row)
            aggs.append(row_group)
            row_group = []
            curr_vol = 0
        curr_vol += row["size"]
        row_group.append(row)
    return aggs, curr_vol, row, row_group, tmp_row


@app.cell
def __(aggs, pl):
    pl.DataFrame(aggs[0])
    return


if __name__ == "__main__":
    app.run()
