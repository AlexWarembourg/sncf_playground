from neuralforecast.core import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models import PatchTST, TSMixerx, iTransformer, NHITS, NBEATSx


import datetime
import json
import toml
import sys
import argparse
from copy import deepcopy


sys.path.insert(0, r"C:\Users\N000193384\Documents\sncf_project\sncf_playground")


from src.models.neural_wrapper import NeuralWrapper
from src.project_utils import load_data
from src.preprocessing.quality import trim_timeseries
from src.preprocessing.times import from_day_to_time_fe
from datetime import datetime, timedelta
import polars as pl
import toml
from pathlib import Path

features = toml.load("data/features.toml")
macro_horizon = features["MACRO_HORIZON"]
p = Path(features["ABS_DATA_PATH"])
ts_uid = features["ts_uid"]
date_col = features["date_col"]
y = features["y"]
times_cols = features["times_cols"]

if __name__ == "__main__":

    train_data, test_data, submission = load_data(p)
    train_data = (
        train_data.pipe(trim_timeseries, target=y, uid=ts_uid, time=date_col)
        .with_columns(pl.col(y).log1p().alias(y))
        .pipe(from_day_to_time_fe, time=date_col, frequency="day")
        .filter(
            (pl.col("date") >= datetime(2017, 1, 1))
            & (pl.col("date") != pl.datetime(2019, 12, 1))
            & (pl.col("date") != pl.datetime(2021, 11, 1))
        )
    )
    exog = ["job", "ferie", "vacances"] + times_cols
    n_series = train_data[ts_uid].n_unique()

    sf = NeuralForecast(
        [
            NHITS(
                h=macro_horizon,
                input_size=macro_horizon,
                dropout_prob_theta=0.1,
                n_freq_downsample=[30, 7, 1],
                interpolation_mode="cubic",
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
                stat_exog_list=[],  # static is rattached at unique id such as one line = one id with static feature set
            ),
            NBEATSx(
                h=macro_horizon,
                input_size=macro_horizon,
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
            ),
            PatchTST(
                h=macro_horizon,
                input_size=macro_horizon,
                max_steps=1000,
                early_stop_patience_steps=3,
            ),
            TSMixerx(
                h=macro_horizon,
                input_size=macro_horizon,
                n_series=n_series,
                max_steps=1000,
                early_stop_patience_steps=3,
                hist_exog_list=exog,  # time based with past known
                futr_exog_list=exog,  # time based with future known
            ),
            iTransformer(
                h=macro_horizon,
                input_size=macro_horizon,
                n_series=n_series,
                max_steps=1000,
                early_stop_patience_steps=3,
            ),
        ],
        freq="1d",
    )

    nf_base = NeuralWrapper(
        models=sf,
        ts_uid=ts_uid,
        date_col=date_col,
        target=y,
        forecast_horizon=macro_horizon,
        fill_strategy="forward",
        frequency="1d",
        levels=[95],
        conformalised=False,
        fitted=False,
        exog=exog,
    )

    train_data = train_data.pipe(nf_base.nixtla_reformat, date_col="date").drop_nulls()
    test_forecast = nf_base.forecast(
        train_data,
        val_size=macro_horizon,
        future_df=test_data.rename({ts_uid: "unique_id", date_col: "ds"}).pipe(
            from_day_to_time_fe, time="ds", frequency="day"
        ),
    )
    test_forecast.to_csv("out/submit/neural_test_set.csv", sep="@", index=False)
    valid_forecast, output_metrics = nf_base.evaluate_on_valid(train_data)
    valid_forecast.to_csv("out/neural_validation.csv", sep="@", index=False)
    print(output_metrics)
