import pandas as pd
from polars_trading._testing.features import frac_diff_ffd


def apply_pd_frac_diff(df: pd.DataFrame, d: float, threshold: float) -> pd.DataFrame:
    return (
        df.set_index("ts_event")
        .groupby("symbol")[["price"]]
        .apply(frac_diff_ffd, d, threshold)
        .reset_index()
    )
