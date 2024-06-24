import pandas as pd
import numpy as np
import polars as pl


def describe_timeseries(
    train_data, test_data, unique_id, time, dataset_name, time_granularity
):
    if isinstance(train_data, pl.DataFrame):
        train_data = train_data.to_pandas()
    if isinstance(test_data, pl.DataFrame):
        test_data = test_data.to_pandas()
    n_uid = train_data[unique_id].nunique()
    train_data["shiftedtime"] = (
        train_data.sort_values(by=[unique_id, time])
        .groupby(unique_id)[time]
        .transform("shift")
    )
    train_data["day_gap"] = (
        pd.to_datetime(train_data[time]) - pd.to_datetime(train_data["shiftedtime"])
    ).dt.days
    timeseries_desc = pd.DataFrame()
    timeseries_desc["Time Granularity"] = [time_granularity]
    timeseries_desc["Forecast Horizon Length"] = test_data[time].nunique()
    timeseries_desc["No. Time Series"] = n_uid
    timeseries_desc["Avg. Length of Time Series"] = (
        train_data.groupby(unique_id)[time].nunique().mean().round().astype(int)
    )
    timeseries_desc["Avg. Gap Between Days"] = train_data["day_gap"].mean()
    timeseries_desc["Train Time"] = (
        f"{train_data[time].min()} - {train_data[time].max()}"
    )
    timeseries_desc["Test Time"] = f"{test_data[time].min()} - {test_data[time].max()}"
    timeseries_desc = timeseries_desc.T.rename(columns={0: dataset_name})
    return timeseries_desc


def timeseries_length(pl_df: pl.DataFrame, ts_uid: str, date: str):
    return pl_df.groupby(ts_uid).agg(pl.col(date).n_unique())
