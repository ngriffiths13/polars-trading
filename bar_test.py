import marimo

__generated_with = "0.8.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import polars as pl
    return pl,


@app.cell
def __(pl):
    data = pl.read_parquet("./trade_data.parquet")
    pl_df = pl.read_parquet("./res.parquet")
    pd_df = pl.read_parquet("./pd_df.parquet")
    return data, pd_df, pl_df


@app.cell
def __(pl_df):
    pl_df
    return


@app.cell
def __(pd_df, pl, pl_df):
    pd_df.select("ts_event_start").hstack(
        pl_df.select(pl.col("ts_event_start").alias("pl"))
    ).with_columns((pl.col("ts_event_start") == pl.col("pl")).alias("eq")).filter(~pl.col("eq"))
    return


@app.cell
def __(data, pl):
    import datetime
    data.filter(pl.col("ts_event").dt.truncate("1m") == datetime.datetime(2024, 12, 16, 10, 50))
    return datetime,


if __name__ == "__main__":
    app.run()
