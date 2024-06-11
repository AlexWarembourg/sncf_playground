import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl
from datetime import timedelta
import datetime
import json
import toml

from src.preprocessing.polars.quality import trim_timeseries

from src.project_utils import forecast_to_submit, load_data
from src.preprocessing.polars.validation import freeze_validation_set
from src.project_utils import submit_fcst
from src.preprocessing.polars.times import from_day_to_time_fe, get_covid_table
from src.models.stats_wrapper import nixtla_fast_baseline

from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    AutoTheta,
    AutoARIMA,
    SeasonalNaive,
    AutoTBATS,
    AutoETS,
)

features = toml.load("data/features.toml")
macro_horizon = features["MACRO_HORIZON"]
p = Path(features["ABS_DATA_PATH"])
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]

in_dt = datetime.date(2017, 1, 1)

my_way_to_go_baseline = [
    HoltWinters(),
    AutoTheta(),
    AutoARIMA(),
    AutoETS(),
    AutoTBATS(season_length=[7]),
]

# Instantiate StatsForecast class as sf
sf = StatsForecast(
    models=my_way_to_go_baseline,
    fallback_model=SeasonalNaive(season_length=28),
    n_jobs=-1,
    freq="1d",
)

if __name__ == "__main__":
    train_data, test_data, submission = load_data(p)
    cov = get_covid_table(2015, 2024)
    cov = cov[np.any(cov.iloc[:, 1:] != 0, axis=1)]["date"].unique()
    train_data = (
        train_data.pipe(trim_timeseries, target="y", uid="station", time="date")
        # .with_columns(pl.col(y).log1p().cast(pl.Float32).alias(y))
        .pipe(from_day_to_time_fe, time="date", frequency="day").filter(
            (pl.col("date") >= in_dt)
            & (pl.col("date") != pl.datetime(2019, 12, 1))
            & (pl.col("date") != pl.datetime(2021, 11, 1))
            # & (pl.col("date").is_in(cov).not_())
        )
    )

    train_set, validation_set = freeze_validation_set(
        df=train_data, ts_uid=ts_uid, date=date_col, target=y, val_size=macro_horizon
    )

    # =============================
    # validation first
    # =============================

    fcst_df = nixtla_fast_baseline(
        df=train_set,
        sf_models=sf,
        date_col=date_col,
        uid=ts_uid,
        frequency="1d",
        y_col=y,
        horizon=181,
        conformal=False,
        exog=[],
        to_pd=False,
    )

    fcst_df.to_csv("out/submit/nixtla_validation.csv", sep="@", index=False)

    fcst_df = nixtla_fast_baseline(
        df=train_data,
        sf_models=sf,
        date_col=date_col,
        uid=ts_uid,
        frequency="1d",
        y_col=y,
        horizon=181,
        conformal=False,
        exog=[],
        to_pd=False,
    )

    fcst_df.to_csv("out/submit/nixtla_forecast.csv", sep="@", index=False)
