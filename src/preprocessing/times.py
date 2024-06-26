import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import SplineTransformer
import holidays


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
    ).set_output(transform="polars")


def from_day_to_time_fe(
    full_data: pl.DataFrame, time: str, frequency: str
) -> pl.DataFrame:
    # Convert time column to datetime using str.to_date
    full_data = full_data.with_columns(pl.col(time).cast(pl.Date).alias(time))
    if frequency == "day":
        full_data = full_data.with_columns(
            [
                pl.col(time).dt.year().cast(pl.Int16).alias("year"),
                pl.col(time).dt.week().cast(pl.Int16).alias("week"),
                pl.col(time).dt.month().cast(pl.Int16).alias("month"),
                pl.col(time).dt.day().cast(pl.Int16).alias("day"),
                pl.col(time).dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
                pl.col(time).dt.weekday().cast(pl.Int16).alias("day_of_week"),
                pl.col(time).dt.quarter().cast(pl.Int16).alias("quarter"),
            ]
        ).with_columns(
            [
                ((pl.col("day") / 7).ceil().cast(pl.Int16)).alias("week_of_month"),
                (pl.col("day_of_week") >= 5).cast(pl.Int16).alias("is_weekend"),
            ]
        )
    elif frequency == "week":
        full_data = full_data.with_columns(
            [
                pl.col(time).dt.year().cast(pl.Int16).alias("year"),
                pl.col(time).dt.week().cast(pl.Int16).alias("week"),
                pl.col(time).dt.month().cast(pl.Int16).alias("month"),
                pl.col(time).dt.quarter().cast(pl.Int16).alias("quarter"),
            ]
        )
    else:
        raise ValueError("Frequency must be either 'day' or 'week'.")
    return full_data


def get_covid_table(start_year: int = 2015, end_year: int = 2024):
    data = {
        "Lockdown": ["First Lockdown", "Second Lockdown", "Third Lockdown (Partial)"],
        "StartDate": [
            pd.Timestamp("2020-03-17"),
            pd.Timestamp("2020-10-30"),
            pd.Timestamp("2021-03-20"),
        ],
        "EndDate": [
            pd.Timestamp("2020-05-11"),
            pd.Timestamp("2020-12-15"),
            pd.Timestamp("2021-05-03"),
        ],
    }

    # Create a DataFrame
    covid_confinement_df = pd.DataFrame(data)
    covid_confinement_df["flag"] = 1
    covid_confinement_df["date_range"] = covid_confinement_df.apply(
        lambda x: pd.date_range(x["StartDate"], x["EndDate"], freq="D"), axis=1
    )
    covid_confinement_df = covid_confinement_df.explode("date_range")
    covid_confinement_df["flag"] = (
        covid_confinement_df.sort_values(by=["Lockdown", "date_range"])
        .groupby("Lockdown")["flag"]
        .cumsum()
    )
    covid_confinement_df = (
        covid_confinement_df.set_index(["date_range", "Lockdown"])["flag"]
        .unstack(-1)
        .reset_index(names="date")
        .fillna(0)
    )
    df_dates = (
        pd.date_range(f"{str(start_year)}-01-01", f"{str(end_year)}-12-31", freq="D")
        .to_frame(name="date")
        .reset_index(drop=True)
    )
    df_dates = df_dates.merge(covid_confinement_df, how="left", on="date").fillna(0)
    return pl.from_pandas(df_dates).with_columns(
        pl.col("date").cast(pl.Date).alias("date")
    )


# Function to calculate s!ymmetrical days until or since the event
def symmetrical_days_until_event(date, event_dates):
    event_dates_extended = (
        event_dates
        + [d + timedelta(days=365) for d in event_dates]
        + [d - timedelta(days=365) for d in event_dates]
    )
    closest_event = min(event_dates_extended, key=lambda x: abs((x - date).days))
    delta = (closest_event - date).days
    if delta > 182:
        delta -= 365
    elif delta < -182:
        delta += 365
    return delta


def get_basic_holidays(min_year: int = 2015, max_year: int = 2024):
    df_dates = pd.DataFrame(
        {
            "date": pd.date_range(
                start=datetime(min_year, 1, 1),
                end=datetime(max_year, 12, 31),
            )
        }
    )

    country_holidays = holidays.France(years=np.arange(min_year, max_year))
    holiday_list = [
        {"date": date, "holiday": name}
        for date, name in sorted(country_holidays.items())
    ]
    holiday_df = pd.DataFrame(holiday_list).assign(
        year=lambda x: pd.to_datetime(x["date"]).dt.year
    )
    # Add columns for each event
    event_names = holiday_df["holiday"].unique()
    for event_name in event_names:
        event_dates = holiday_df[holiday_df["holiday"] == event_name]["date"].tolist()
        event_dates = [pd.to_datetime(x) for x in event_dates]
        df_dates[event_name] = df_dates["date"].apply(
            lambda x: symmetrical_days_until_event(x, event_dates)
        )

    df_dates = pl.from_pandas(df_dates).with_columns(
        pl.col("date").cast(pl.Date).alias("date")
    )
    return df_dates
