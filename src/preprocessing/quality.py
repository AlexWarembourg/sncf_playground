import pandas as pd
import polars as pl

from typing import Dict, Union, List


def find_ts_outlier(
    data,
    ts_uid: Union[str, List[str]] = "ts_uid",
    y: str = "y",
    date_col: str = "date",
    deviation_strategy: str = "normal",
    n_sigma: int = 3,
    window_size: int = 90,
) -> pl.DataFrame:
    strategy = {
        "normal": 1.4826,
        "uniform": 1.16,
        "laplace": 2.04,
        "exponential": 2.08,
    }
    assert (
        deviation_strategy in strategy.keys()
    ), f"The choosen strategy must be one the of the following : {strategy.keys()}"

    outlier_d = (
        data.sort(by=[ts_uid, date_col])
        .groupby_rolling("date", period=f"{window_size}d", by=ts_uid)
        .agg(
            rolling_median=pl.col(y).median(),
            scaled_mad=strategy[deviation_strategy]
            * (pl.col(y) - pl.col(y).median()).abs().median(),
            y=pl.col(y).first(),
        )
        .with_columns(
            # ((pl.col(y) - pl.col("rolling_median")).abs()).alias("diff"),
            (pl.col("scaled_mad") * -n_sigma).alias("lower_bound"),
            (pl.col("scaled_mad") * n_sigma).alias("upper_bound"),
        )
        .with_columns(
            (pl.col(y).is_between(pl.col("lower_bound"), pl.col("upper_bound")))
            .cast(pl.Int32)
            .alias("not_outlier")
        )
    )
    data = data.join(
        outlier_d.select([ts_uid, date_col, "not_outlier"]),
        how="left",
        on=[ts_uid, date_col],
    )
    return data


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
    full_data: pl.DataFrame, uid: str, time: str, min_length: int = 364
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


def return_null_cols(data: pl.DataFrame) -> Dict[str, float]:
    return {
        k: v / round(data.shape[0], 2) * 100
        for k, v in data.select(pl.all().is_null().sum()).to_dicts()[0].items()
        if v > 0
    }
