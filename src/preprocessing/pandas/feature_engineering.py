import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from sklearn.preprocessing import SplineTransformer


# B-spline functions
# ==============================================================================
def spline_transformer(period, degree=3, extrapolation="periodic"):
    """
    Returns a transformer that applies B-spline transformation.
    """
    return SplineTransformer(
        degree=degree,
        n_knots=period + 1,
        knots="uniform",
        extrapolation=extrapolation,
        include_bias=True,
    ).set_output(transform="pandas")


def from_day_to_time_fe(full_data: pd.DataFrame, time: str) -> pd.DataFrame:
    full_data[time] = pd.to_datetime(full_data[time])
    full_data["year"] = full_data[time].dt.isocalendar().year
    full_data["week"] = full_data[time].dt.isocalendar().week
    full_data["month"] = full_data[time].dt.month
    full_data["day"] = full_data[time].dt.day
    full_data["day_of_year"] = full_data[time].dt.dayofyear
    full_data["day_of_week"] = full_data[time].dt.dayofweek
    full_data["quarter"] = full_data[time].dt.quarter
    full_data["week_of_month"] = np.round(full_data["day"] / 7)
    full_data["is_weekend"] = (full_data["day_of_week"] >= 5).astype(int)
    return full_data


def pl_from_day_to_time_fe(
    full_data: pl.DataFrame, time: str, frequency: str
) -> pl.DataFrame:
    # Convert time column to datetime using str.to_date
    full_data = full_data.with_columns(pl.col(time).cast(pl.Date).alias(time))
    if frequency == "day":
        full_data = full_data.with_columns(
            [
                pl.col(time).dt.year().alias("year"),
                pl.col(time).dt.week().alias("week"),
                pl.col(time).dt.month().alias("month"),
                pl.col(time).dt.day().alias("day"),
                pl.col(time).dt.ordinal_day().alias("day_of_year"),
                pl.col(time).dt.weekday().alias("day_of_week"),
                pl.col(time).dt.quarter().alias("quarter"),
            ]
        ).with_columns(
            [
                ((pl.col("day") / 7).ceil().cast(pl.Int32)).alias("week_of_month"),
                (pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend"),
            ]
        )
    elif frequency == "week":
        full_data = full_data.with_columns(
            [
                pl.col(time).dt.year().alias("year"),
                pl.col(time).dt.week().alias("week"),
                pl.col(time).dt.month().alias("month"),
                pl.col(time).dt.quarter().alias("quarter"),
            ]
        )
    else:
        raise ValueError("Frequency must be either 'day' or 'week'.")
    return full_data


def create_spline_features_from_dayofyear(
    df: pd.DataFrame, d_col: str = "day_of_year"
) -> pd.DataFrame:

    spline_week = (
        spline_transformer(period=53).fit_transform(df[[d_col]]).astype(np.float16)
    )
    spline_week.columns = [f"spline_week{i}" for i in range(len(spline_week.columns))]

    spline_month = (
        spline_transformer(period=12).fit_transform(df[[d_col]]).astype(np.float16)
    )
    spline_month.columns = [
        f"spline_month{i}" for i in range(len(spline_month.columns))
    ]

    spline_quarter = (
        spline_transformer(period=4).fit_transform(df[[d_col]]).astype(np.float16)
    )
    spline_quarter.columns = [
        f"spline_quarter{i}" for i in range(len(spline_quarter.columns))
    ]
    df = pd.concat([df, spline_week, spline_month, spline_quarter], axis=1)
    return df
