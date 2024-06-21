import pandas as pd
import polars as pl


def trim_timeseries(
    full_data: pl.DataFrame, target: str, uid: str, time: str
) -> pl.DataFrame:
    # Calculate the true start dates
    true_start = (
        full_data.filter(pl.col(target) > 0)
        .group_by(uid)
        .agg([pl.col(time).min().alias("min_dt")])
    )

    # Merge the true start dates back into the full dataset
    full_data = full_data.join(true_start, on=uid, how="left")

    # Filter the full dataset based on the true start dates
    full_data = full_data.filter(pl.col(time) >= pl.col("min_dt")).drop("min_dt")
    return full_data


def minimum_length_uid(
    full_data: pl.DataFrame, target: str, uid: str, time: str, min_length=364
) -> pl.DataFrame:
    # Calculate the true start dates
    quality_series = (
        full_data.groupby(uid)
        .agg(pl.col(time).n_unique().alias("ntimestep"))
        .filter(pl.col("ntimestep") >= min_length)
        .select(pl.col(uid).unique())
        .to_series()
        .to_list()
    )
    return quality_series
